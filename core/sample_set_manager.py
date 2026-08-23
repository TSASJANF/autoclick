"""样本集仓库：data/sets/ 目录与 sets_index.json 的管理。"""

from __future__ import annotations

import datetime
import logging
import os
import shutil
from typing import Optional

from . import paths
from . import json_store
from .models import SampleSet

logger = logging.getLogger(__name__)


def set_root(set_id: str) -> str:
    """样本集根目录（绝对路径）。"""
    return os.path.join(paths.SETS_DIR, set_id)


class SampleSetManager:
    def __init__(self):
        self.sets: list[SampleSet] = []
        self._next_id = 1

    # ── 增删改 ────────────────────────────────────────────────────────

    def add(self, name: str) -> SampleSet:
        sid = f"set_{self._next_id}"
        self._next_id += 1
        ss = SampleSet(id=sid, name=name,
                       created_at=datetime.datetime.now().isoformat(timespec="seconds"))
        os.makedirs(self.images_dir(ss.id), exist_ok=True)
        os.makedirs(self.targets_dir(ss.id), exist_ok=True)
        self.sets.append(ss)
        self.save()
        return ss

    def remove(self, set_id: str, delete_files: bool = True) -> None:
        ss = self.get(set_id)
        if ss is None:
            return
        if delete_files:
            root = set_root(set_id)
            if os.path.exists(root):
                shutil.rmtree(root, ignore_errors=True)
        self.sets = [s for s in self.sets if s.id != set_id]
        self.save()

    def rename(self, set_id: str, new_name: str) -> None:
        ss = self.get(set_id)
        if ss is None:
            return
        ss.name = new_name
        self.save()

    def update_settings(self, set_id: str, negative_sample_ratio: int | None = None,
                        negative_min_positive: int | None = None) -> None:
        ss = self.get(set_id)
        if ss is None:
            return
        if negative_sample_ratio is not None:
            ss.negative_sample_ratio = negative_sample_ratio
        if negative_min_positive is not None:
            ss.negative_min_positive = negative_min_positive
        self.save()

    # ── 查询 ──────────────────────────────────────────────────────────

    def get(self, set_id: str) -> Optional[SampleSet]:
        return next((s for s in self.sets if s.id == set_id), None)

    def get_all(self) -> list[SampleSet]:
        return list(self.sets)

    def ensure_default(self) -> SampleSet:
        if not self.sets:
            return self.add("默认样本集")
        return self.sets[0]

    # ── 目录定位（历史代码散落在 SampleSet 属性上，现集中为方法，
    #    供 UI 与训练数据准备共用）─────────────────────────────────────

    def images_dir(self, set_id: str) -> str:
        return os.path.join(set_root(set_id), "images")

    def targets_dir(self, set_id: str) -> str:
        return os.path.join(set_root(set_id), "targets")

    def session_file(self, set_id: str) -> str:
        return os.path.join(set_root(set_id), "session.json")

    def set_root(self, set_id: str) -> str:
        """样本集根目录（绝对路径）。"""
        return set_root(set_id)

    # ── 持久化 ────────────────────────────────────────────────────────

    def save(self) -> None:
        json_store.write_json(paths.SETS_INDEX_FILE,
                              [s.to_dict() for s in self.sets])

    def load(self) -> None:
        data = json_store.read_json(paths.SETS_INDEX_FILE, [])
        if not isinstance(data, list):
            logger.warning("sets_index.json 格式异常（应为列表），已忽略")
            data = []
        self.sets = [SampleSet.from_dict(d) for d in data if isinstance(d, dict)]
        max_id = 0
        for s in self.sets:
            try:
                max_id = max(max_id, int(s.id.split("_")[1]))
            except (IndexError, ValueError):
                pass
        self._next_id = max_id + 1
