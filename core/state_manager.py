"""应用运行状态（app_state.json）：窗口几何、各页签参数等会话记忆。

修复历史缺陷：旧 load_state 用 dict(DEFAULT_STATE) 浅拷贝，嵌套 dict
与模块级默认值共享引用，运行期修改会污染默认值。
"""

from __future__ import annotations

from . import json_store, paths

DEFAULT_STATE = {
    "schema_version": 1,
    "window": {
        "x": None,
        "y": None,
        "width": 1200,
        "height": 800,
        "maximized": False,
        "active_tab": 0,
    },
    "detect": {
        "confidence": 50,
        "fps": 10,
        "priority_enabled": False,
        "auto_click": False,
        "floating_auto_switch": False,
    },
    "annotate": {
        "selected_set_id": "",
        "last_target_id": "",
    },
    "train": {
        "epochs": 50,
        "batch_size": 16,
        "img_size": 640,
        "save_path": "",
        "last_target_id": "",
    },
    "opencv": {
        "threshold": 0.8,
    },
    "floating_window": {
        "x": None,
        "y": None,
    },
}


def load_state() -> dict:
    data = json_store.read_json(paths.STATE_FILE, {})
    if not isinstance(data, dict):
        data = {}
    return json_store.merge_defaults(data, DEFAULT_STATE)


def save_state(state: dict) -> None:
    json_store.write_json(paths.STATE_FILE, state)
