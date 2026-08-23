import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TARGETS_FILE = os.path.join(BASE_DIR, "targets.json")
SHARED_IMAGES_DIR = os.path.join(BASE_DIR, "data", "shared_images")

TARGET_COLORS = [
    "#2ecc71", "#e74c3c", "#3498db", "#f1c40f", "#9b59b6",
    "#1abc9c", "#e67e22", "#ec407a", "#00bcd4", "#8bc34a",
]


@dataclass
class Target:
    id: str  # unique id
    name: str
    color: str  # hex color
    priority: int = 0  # lower = higher priority
    enabled: bool = True
    detection_mode: str = "ai"  # "ai" | "opencv" | "both"
    # AI model
    model_path: str = ""  # path to .pt model
    # OpenCV template
    template_path: str = ""  # path to template image
    match_threshold: float = 0.8
    source_image_path: str = ""  # path to source image (full screenshot)
    # Training data
    data_dir: str = ""  # path to labels directory
    # Sample set
    set_id: str = ""  #所属样本集 ID (e.g. "set_1")
    # Training base model (e.g. "yolov8n")
    model_base: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Target":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class TargetManager:
    def __init__(self):
        self.targets: list[Target] = []
        self._next_id = 1

    def add(self, name: str, set_id: str = "", color: str = "", priority: int = 0, **kwargs) -> Target:
        if not color:
            color = TARGET_COLORS[len(self.targets) % len(TARGET_COLORS)]
        tid = f"target_{self._next_id}"
        self._next_id += 1
        t = Target(id=tid, name=name, color=color, priority=priority, set_id=set_id, **kwargs)
        # Auto-set data_dir based on set_id
        if not t.data_dir:
            if t.set_id:
                t.data_dir = os.path.join("data", "sets", t.set_id, "targets", tid)
            else:
                t.data_dir = os.path.join("data", "targets", tid)
        self.targets.append(t)
        return t

    def remove(self, target_id: str):
        self.targets = [t for t in self.targets if t.id != target_id]

    def get(self, target_id: str) -> Optional[Target]:
        for t in self.targets:
            if t.id == target_id:
                return t
        return None

    def get_enabled(self) -> list[Target]:
        return [t for t in self.targets if t.enabled]

    def get_by_priority(self) -> list[Target]:
        """Get enabled targets sorted by priority (lower number = higher priority)."""
        return sorted(self.get_enabled(), key=lambda t: t.priority)

    def get_by_set(self, set_id: str) -> list[Target]:
        """Get all targets belonging to a specific sample set."""
        return [t for t in self.targets if t.set_id == set_id]

    def save(self):
        data = [t.to_dict() for t in self.targets]
        tmp_path = TARGETS_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, TARGETS_FILE)

    def load(self):
        if not os.path.exists(TARGETS_FILE):
            return
        try:
            with open(TARGETS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return
        self.targets = [Target.from_dict(d) for d in data]
        if self.targets:
            max_id = 0
            for t in self.targets:
                try:
                    num = int(t.id.split("_")[1])
                    if num > max_id:
                        max_id = num
                except (IndexError, ValueError):
                    pass
            self._next_id = max_id + 1
