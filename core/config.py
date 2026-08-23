import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")

DEFAULT_CONFIG = {
    "screenshot_default_enabled": False,
    "screenshot_default_path": "data/screenshots",
    "screenshot_hide_window": True,
    "targets": [],  # list of target dicts
    "opencv_templates": [],  # list of template dicts
    "hotkey_toggle_detection": "F9",  #快捷键：开始/停止检测
    "negative_sample_ratio": 20,  # 负样本占总训练集的百分比 (10-30)
    "negative_min_positive": 5,  # 启用负样本的最小正样本数
}


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                # Merge with defaults for missing keys
                for k, v in DEFAULT_CONFIG.items():
                    if k not in cfg:
                        cfg[k] = v
                return cfg
        except (json.JSONDecodeError, IOError):
            pass
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict):
    tmp_path = CONFIG_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, CONFIG_PATH)
