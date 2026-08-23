"""应用配置（config.json）：用户可编辑的持久化设置。

清理历史遗留：旧 DEFAULT_CONFIG 中的 targets / opencv_templates 字段
从未被读取（真实数据在 targets.json），已删除。
"""

from __future__ import annotations

from . import json_store
from . import paths

DEFAULT_CONFIG = {
    "screenshot_default_enabled": False,
    "screenshot_default_path": "data/screenshots",
    "screenshot_hide_window": True,
    "window_capture_enabled": False,
    "window_capture_title": "",
    "hotkey_toggle_detection": "F9",       # 开始/停止检测全局热键
    "negative_sample_ratio": 20,           # 负样本占总训练集百分比 (10-30)
    "negative_min_positive": 5,            # 启用负样本的最小正样本数
}


def load_config() -> dict:
    data = json_store.read_json(paths.CONFIG_FILE, {})
    if not isinstance(data, dict):
        data = {}
    return json_store.merge_defaults(data, DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    json_store.write_json(paths.CONFIG_FILE, cfg)
