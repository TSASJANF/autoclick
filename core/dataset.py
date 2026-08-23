"""训练数据集准备：定位已标注图片、组装 train_data 目录、生成 data.yaml。

从旧 core/trainer.py 的静态方法迁移，修复两个数据正确性缺陷：

1. 临时目录残留污染（严重）：旧版只在链接/拷贝前检查「文件不存在才创
   建」，但 train_data 从不清理。上一次训练随机抽的负样本空标签文件
   会残留到下一次训练，而负样本每次重新随机抽取 —— 结果旧负样本（可
   能已不适用）永远混入训练集。现改为每次重建前清空 train_data；

2. Windows 无特权时 os.symlink 必然失败，注释声称的符号链接方案名不
   副实，且残留判定掩盖了内容变化。统一改为复制语义，行为可预期。
"""

from __future__ import annotations

import logging
import os
import random
import shutil

from . import paths

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(directory: str) -> list[str]:
    """列出目录下所有图片文件名。"""
    if not os.path.isdir(directory):
        return []
    return [f for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS]


def find_labeled_images(dataset_dir: str, images_dir: str) -> list[str]:
    """找出在 dataset_dir/labels/ 有对应标注的图片文件名列表。"""
    labels_dir = os.path.join(paths.resolve(dataset_dir), "labels")
    if not os.path.isdir(labels_dir):
        return []
    label_stems = {os.path.splitext(f)[0] for f in os.listdir(labels_dir)
                   if f.endswith(".txt")}
    return [f for f in list_images(images_dir)
            if os.path.splitext(f)[0] in label_stems]


def get_dataset_stats(dataset_dir: str, images_dir: str) -> dict:
    """数据集统计：图片数、标注文件数、每类框数。"""
    labels_dir = os.path.join(paths.resolve(dataset_dir), "labels")
    images = list_images(images_dir)
    labels = ([f for f in os.listdir(labels_dir) if f.endswith(".txt")]
              if os.path.isdir(labels_dir) else [])

    class_counts: dict[int, int] = {}
    for fname in labels:
        with open(os.path.join(labels_dir, fname), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        cls_id = int(parts[0])
                    except ValueError:
                        continue
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

    return {"image_count": len(images), "label_count": len(labels),
            "class_counts": class_counts}


def count_negative_candidates(positive_count: int, unlabeled_count: int,
                              ratio_percent: int, min_positive: int) -> int:
    """按「负样本占训练集百分比」换算应抽取的负样本数（纯函数，可测）。"""
    if ratio_percent <= 0 or positive_count < min_positive or unlabeled_count <= 0:
        return 0
    if ratio_percent >= 100:
        return unlabeled_count
    target = round(positive_count * ratio_percent / (100 - ratio_percent))
    return min(target, unlabeled_count)


def prepare_training_dir(dataset_dir: str, class_names: list[str], images_dir: str,
                         negative_ratio: int = 0, min_positive: int = 5,
                         seed: int | None = None) -> dict:
    """组装训练目录（images/ + labels/ + data.yaml），返回统计信息。

    dataset_dir: 含 labels/ 的目标数据目录
    images_dir:  样本集图片目录
    negative_ratio: 负样本占训练集百分比（0 = 关闭）
    min_positive: 启用负样本所需最小正样本数
    seed: 负样本随机抽样种子（None = 随机）
    """
    abs_images_dir = paths.resolve(images_dir)
    abs_dataset_dir = paths.resolve(dataset_dir)

    labeled_files = find_labeled_images(dataset_dir, images_dir)
    if not labeled_files:
        raise ValueError("没有找到有标注的图片，请先标注图片")

    labeled_stems = {os.path.splitext(f)[0] for f in labeled_files}
    all_images = list_images(images_dir)
    unlabeled = [f for f in all_images
                 if os.path.splitext(f)[0] not in labeled_stems]

    positive_count = len(labeled_files)
    negative_count = count_negative_candidates(
        positive_count, len(unlabeled), negative_ratio, min_positive)
    rng = random.Random(seed)
    negative_files = rng.sample(unlabeled, negative_count) if negative_count else []

    # 每次重建前清空临时目录，杜绝上次训练的负样本残留（见模块 docstring）
    train_dir = os.path.join(abs_dataset_dir, "train_data")
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir, ignore_errors=True)
    train_images = os.path.join(train_dir, "images")
    train_labels = os.path.join(train_dir, "labels")
    os.makedirs(train_images)
    os.makedirs(train_labels)

    for img_file in labeled_files:
        stem = os.path.splitext(img_file)[0]
        shutil.copy2(os.path.join(abs_images_dir, img_file),
                     os.path.join(train_images, img_file))
        src_label = os.path.join(abs_dataset_dir, "labels", stem + ".txt")
        if os.path.exists(src_label):
            shutil.copy2(src_label, os.path.join(train_labels, stem + ".txt"))

    for img_file in negative_files:
        stem = os.path.splitext(img_file)[0]
        shutil.copy2(os.path.join(abs_images_dir, img_file),
                     os.path.join(train_images, img_file))
        # 负样本：空标注文件 = 图中无目标
        open(os.path.join(train_labels, stem + ".txt"), "w").close()

    yaml_content = (f"path: {train_dir}\n"
                    f"train: images\n"
                    f"val: images\n\n"
                    f"nc: {len(class_names)}\n"
                    f"names: {class_names}\n")
    yaml_path = os.path.join(train_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    logger.info("训练目录已就绪: %s (正 %d / 负 %d)",
                train_dir, positive_count, len(negative_files))
    return {
        "yaml_path": yaml_path,
        "positive_count": positive_count,
        "negative_count": len(negative_files),
        "total_count": positive_count + len(negative_files),
        "unlabeled_count": len(unlabeled),
    }


def cleanup_training_dir(data_yaml: str) -> None:
    """训练结束后清理 train_data 临时目录。"""
    try:
        train_dir = os.path.dirname(os.path.abspath(data_yaml))
        if os.path.basename(train_dir) == "train_data" and os.path.isdir(train_dir):
            shutil.rmtree(train_dir, ignore_errors=True)
    except OSError:
        logger.exception("清理训练临时目录失败: %s", data_yaml)


def suggest_hyperparams(positive_count: int, negative_count: int,
                        avg_width: float, avg_height: float,
                        min_side: int) -> tuple[int, int, int]:
    """按数据集规模与图片尺寸推荐 (img_size, batch, epochs)（纯函数）。

    与旧版 TrainTab._auto_adjust_params 的启发式一致：
    小数据集多轮次小批次，大图降批次防 OOM。
    """
    effective = positive_count + negative_count
    avg_side = (avg_width + avg_height) / 2

    if min_side <= 320:
        img_size = 320
    elif min_side <= 416:
        img_size = 416
    elif avg_side <= 600:
        img_size = 640
    elif avg_side <= 900:
        img_size = 800
    else:
        img_size = 1280

    if effective < 30:
        batch, epochs = 4, 150
    elif effective < 80:
        batch, epochs = 4, 120
    elif effective < 200:
        batch, epochs = 8, 80
    elif effective < 500:
        batch, epochs = 16, 60
    else:
        batch, epochs = 16, 40

    if img_size >= 1280 and batch > 8:
        batch = 8
    elif img_size >= 800 and batch > 16:
        batch = 16
    return img_size, batch, epochs
