"""目标仓库：targets.json 的加载、增删查与路径治理。

相对历史版本的修复：
1. 所有相对路径基于 APP_ROOT 解析（旧版依赖 CWD）；
2. 加载时对失效的绝对路径执行重定位修复（项目搬迁后模型断链问题，
   见 core.paths.repair）；
3. 保存时把能相对化的路径转为相对 POSIX 格式，保证可移植。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from . import paths
from . import json_store
from .models import Target, TARGET_COLORS

logger = logging.getLogger(__name__)


class TargetManager:
    def __init__(self):
        self.targets: list[Target] = []
        self._next_id = 1

    # ── 增删改 ────────────────────────────────────────────────────────

    def add(self, name: str, set_id: str = "", color: str = "",
            priority: int = 0, **kwargs) -> Target:
        if not color:
            used = {t.color for t in self.targets}
            color = next((c for c in TARGET_COLORS if c not in used),
                         TARGET_COLORS[len(self.targets) % len(TARGET_COLORS)])
        tid = f"target_{self._next_id}"
        self._next_id += 1
        t = Target(id=tid, name=name, color=color, priority=priority,
                   set_id=set_id, **kwargs)
        if not t.data_dir:
            if t.set_id:
                t.data_dir = os.path.join("data", "sets", t.set_id, "targets", tid)
            else:
                t.data_dir = os.path.join("data", "targets", tid)
        self.targets.append(t)
        return t

    def remove(self, target_id: str) -> None:
        self.targets = [t for t in self.targets if t.id != target_id]

    # ── 查询 ──────────────────────────────────────────────────────────

    def get(self, target_id: str) -> Optional[Target]:
        return next((t for t in self.targets if t.id == target_id), None)

    def get_enabled(self) -> list[Target]:
        return [t for t in self.targets if t.enabled]

    def get_by_priority(self) -> list[Target]:
        return sorted(self.get_enabled(), key=lambda t: t.priority)

    def get_by_set(self, set_id: str) -> list[Target]:
        return [t for t in self.targets if t.set_id == set_id]

    def get_opencv_targets(self) -> list[Target]:
        return [t for t in self.targets if t.detection_mode in ("opencv", "both")]

    def next_color(self) -> str:
        """返回一个未被占用的目标颜色。"""
        used = {t.color for t in self.targets}
        return next((c for c in TARGET_COLORS if c not in used),
                    TARGET_COLORS[len(self.targets) % len(TARGET_COLORS)])

    # ── 持久化 ────────────────────────────────────────────────────────

    def save(self) -> None:
        data = []
        for t in self.targets:
            d = t.to_dict()
            for key in ("model_path", "template_path", "source_image_path", "data_dir"):
                if d.get(key):
                    d[key] = paths.to_stored(paths.resolve(d[key]))
            data.append(d)
        json_store.write_json(paths.TARGETS_FILE, data)

    def load(self) -> None:
        data = json_store.read_json(paths.TARGETS_FILE, [])
        if not isinstance(data, list):
            logger.warning("targets.json 格式异常（应为列表），已忽略")
            data = []
        self.targets = []
        for d in data:
            if not isinstance(d, dict):
                continue
            t = Target.from_dict(d)
            for key in ("model_path", "template_path", "source_image_path", "data_dir"):
                # 磁盘存相对路径（可移植）；内存统一为绝对路径（可直接使用）
                setattr(t, key, str(paths.resolve(paths.repair(getattr(t, key)))))
            self.targets.append(t)
        max_id = 0
        for t in self.targets:
            try:
                max_id = max(max_id, int(t.id.split("_")[1]))
            except (IndexError, ValueError):
                pass
        self._next_id = max_id + 1
