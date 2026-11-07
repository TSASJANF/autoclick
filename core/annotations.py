"""YOLO 格式标注的读写与状态查询。

从 ui/annotate_tab.py 抽出的纯逻辑。历史缺陷：标注框的「目标归属」以
颜色字符串区分（两个目标颜色相同则标注互相污染），现统一以 target_id
为归属键，颜色只负责显示。
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass

from . import paths

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Box:
    """像素坐标标注框，owner_target_id 为归属键。"""
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    owner_target_id: str
    color: str = "#2ecc71"
    label: str = ""

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


def label_path_for(target_data_dir: str, image_path: str) -> str:
    """图片在指定目标数据目录下对应的标注文件路径。"""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(paths.resolve(target_data_dir), "labels", stem + ".txt")


def write_labels(label_path: str, boxes: Iterable[Box], img_w: int, img_h: int) -> None:
    """把像素坐标框写成 YOLO 归一化标注文件。"""
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    lines = []
    for b in boxes:
        cx = (b.x1 + b.width / 2) / img_w
        cy = (b.y1 + b.height / 2) / img_h
        w = b.width / img_w
        h = b.height / img_h
        lines.append(f"{b.cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    with open(label_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def read_labels(label_path: str, img_w: int, img_h: int,
                owner_target_id: str = "", color: str = "#2ecc71",
                label_prefix: str = "") -> list[Box]:
    """读取 YOLO 标注为像素坐标框；文件不存在返回空列表。"""
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            # round 而非 int 截断：0.3-0.25 类浮点误差会让 int() 掉 1 像素
            x1 = round((cx - w / 2) * img_w)
            y1 = round((cy - h / 2) * img_h)
            x2 = round((cx + w / 2) * img_w)
            y2 = round((cy + h / 2) * img_h)
            name = label_prefix or owner_target_id
            boxes.append(Box(x1, y1, x2, y2, cls_id, owner_target_id,
                             color, f"{name}:cls{cls_id}"))
    return boxes


def has_label(label_path: str) -> bool:
    """标注文件存在且非空。"""
    return os.path.exists(label_path) and os.path.getsize(label_path) > 0


def labeled_stems_map(target_data_dirs: dict[str, str]) -> dict[str, set[str]]:
    """扫描各目标 labels 目录，返回 target_id -> 非空标注的文件名主干集合。

    与整个图片列表无关，可被 UI 缓存（PERF-04）；仅在标注发生增删时
    需要失效重算。每个目标一次 scandir，取代旧版每图×每目标逐个 stat。
    """
    stems_map: dict[str, set[str]] = {}
    for tid, data_dir in target_data_dirs.items():
        labels_dir = os.path.join(paths.resolve(data_dir), "labels")
        stems: set[str] = set()
        if os.path.isdir(labels_dir):
            with os.scandir(labels_dir) as it:
                for entry in it:
                    if entry.name.endswith(".txt"):
                        try:
                            if entry.stat().st_size > 0:
                                stems.add(os.path.splitext(entry.name)[0])
                        except OSError:
                            continue
        stems_map[tid] = stems
    return stems_map


def status_from_stems(stems_map: dict[str, set[str]],
                      image_paths: Iterable[str],
                      current_target_id: str = "") -> dict[str, str]:
    """由缓存的 stems 集合计算每张图的状态（纯函数，无文件系统访问）。"""
    status: dict[str, str] = {}
    for image_path in image_paths:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        result = "none"
        for tid, stems in stems_map.items():
            if stem in stems:
                result = "selected" if tid == current_target_id else "other"
                break
        status[image_path] = result
    return status


def label_status_map(target_data_dirs: dict[str, str],
                     image_paths: Iterable[str],
                     current_target_id: str = "") -> dict[str, str]:
    """批量计算标注状态（labeled_stems_map + status_from_stems 的组合）。"""
    return status_from_stems(
        labeled_stems_map(target_data_dirs), image_paths, current_target_id)


def read_class_ids(labels_dir: str) -> set[int]:
    """扫描标注目录返回出现过的类别 ID 集合。"""
    ids: set[int] = set()
    if not os.path.isdir(labels_dir):
        return ids
    for fname in os.listdir(labels_dir):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(labels_dir, fname), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        ids.add(int(parts[0]))
                    except ValueError:
                        continue
    return ids
