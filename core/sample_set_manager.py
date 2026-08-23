import json
import os
import shutil
from dataclasses import dataclass, asdict
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SETS_ROOT = os.path.join(BASE_DIR, "data", "sets")
SETS_INDEX_FILE = os.path.join(SETS_ROOT, "sets_index.json")


@dataclass
class SampleSet:
    id: str  # e.g. "set_1"
    name: str  # e.g. "样本集1"
    created_at: str  # ISO timestamp
    negative_sample_ratio: int = 20  # 负样本占总训练集的百分比 (10-30)
    negative_min_positive: int = 5  # 启用负样本的最小正样本数 (1-100)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SampleSet":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def dir_path(self) -> str:
        """Return the root directory for this sample set."""
        return os.path.join(SETS_ROOT, self.id)

    @property
    def images_dir(self) -> str:
        """Return the images directory for this sample set."""
        return os.path.join(SETS_ROOT, self.id, "images")

    @property
    def targets_dir(self) -> str:
        """Return the targets directory for this sample set."""
        return os.path.join(SETS_ROOT, self.id, "targets")

    @property
    def session_file(self) -> str:
        """Return the session.json path for this sample set."""
        return os.path.join(SETS_ROOT, self.id, "session.json")

    @property
    def set_json_path(self) -> str:
        """Return the set.json path for this sample set."""
        return os.path.join(SETS_ROOT, self.id, "set.json")


class SampleSetManager:
    def __init__(self):
        self.sets: list[SampleSet] = []
        self._next_id = 1

    def add(self, name: str) -> SampleSet:
        """Create a new sample set with directory structure."""
        import datetime
        sid = f"set_{self._next_id}"
        self._next_id += 1
        now = datetime.datetime.now().isoformat(timespec="seconds")
        ss = SampleSet(id=sid, name=name, created_at=now)
        # Create directory structure
        os.makedirs(ss.images_dir, exist_ok=True)
        os.makedirs(ss.targets_dir, exist_ok=True)
        # Write set.json
        self._write_set_json(ss)
        self.sets.append(ss)
        self.save()
        return ss

    def remove(self, set_id: str, delete_files: bool = True):
        """Remove a sample set. Optionally delete its directory."""
        ss = self.get(set_id)
        if ss is None:
            return
        if delete_files and os.path.exists(ss.dir_path):
            shutil.rmtree(ss.dir_path)
        self.sets = [s for s in self.sets if s.id != set_id]
        self.save()

    def rename(self, set_id: str, new_name: str):
        """Rename a sample set."""
        ss = self.get(set_id)
        if ss is None:
            return
        ss.name = new_name
        self._write_set_json(ss)
        self.save()

    def update_settings(self, set_id: str, negative_sample_ratio: int | None = None,
                        negative_min_positive: int | None = None):
        """Update sample set settings."""
        ss = self.get(set_id)
        if ss is None:
            return
        if negative_sample_ratio is not None:
            ss.negative_sample_ratio = negative_sample_ratio
        if negative_min_positive is not None:
            ss.negative_min_positive = negative_min_positive
        self._write_set_json(ss)
        self.save()

    def get(self, set_id: str) -> Optional[SampleSet]:
        for s in self.sets:
            if s.id == set_id:
                return s
        return None

    def get_all(self) -> list[SampleSet]:
        return list(self.sets)

    def save(self):
        """Save sets index to sets_index.json."""
        os.makedirs(SETS_ROOT, exist_ok=True)
        data = [s.to_dict() for s in self.sets]
        tmp_path = SETS_INDEX_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, SETS_INDEX_FILE)

    def load(self):
        """Load sets index from sets_index.json."""
        if not os.path.exists(SETS_INDEX_FILE):
            return
        try:
            with open(SETS_INDEX_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return
        self.sets = [SampleSet.from_dict(d) for d in data]
        if self.sets:
            max_id = 0
            for s in self.sets:
                try:
                    num = int(s.id.split("_")[1])
                    if num > max_id:
                        max_id = num
                except (IndexError, ValueError):
                    pass
            self._next_id = max_id + 1

    def _write_set_json(self, ss: SampleSet):
        """Write individual set.json for a sample set."""
        os.makedirs(ss.dir_path, exist_ok=True)
        with open(ss.set_json_path, "w", encoding="utf-8") as f:
            json.dump(ss.to_dict(), f, ensure_ascii=False, indent=2)

    def ensure_default(self) -> SampleSet:
        """Ensure at least one sample set exists. Create '默认样本集' if empty."""
        if not self.sets:
            return self.add("默认样本集")
        return self.sets[0]
