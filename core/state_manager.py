import os
import json

STATE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app_state.json")

DEFAULT_STATE = {
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
    },
    "annotate": {
        "selected_set_id": "",
    },
    "train": {
        "epochs": 50,
        "batch_size": 16,
        "img_size": 640,
        "model": "yolov8n.pt",
        "save_path": "",
    },
    "opencv": {
        "threshold": 0.8,
    },
}


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
                # Merge with defaults for missing keys
                for k, v in DEFAULT_STATE.items():
                    if k not in state:
                        state[k] = v
                    elif isinstance(v, dict):
                        for sk, sv in v.items():
                            if sk not in state[k]:
                                state[k][sk] = sv
                return state
        except (json.JSONDecodeError, IOError):
            pass
    return dict(DEFAULT_STATE)


def save_state(state: dict):
    tmp_path = STATE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, STATE_PATH)
