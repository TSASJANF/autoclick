"""配置/状态/目标仓库/样本集仓库单元测试。"""

import json
import os

from core import paths
from core.config import load_config, save_config
from core.sample_set_manager import SampleSetManager
from core.state_manager import DEFAULT_STATE, load_state, save_state
from core.target_manager import TargetManager


class TestConfig:
    def test_defaults_on_missing_file(self, tmp_project):
        cfg = load_config()
        assert cfg["hotkey_toggle_detection"] == "F9"
        assert "targets" not in cfg, "废弃字段不应再出现"

    def test_roundtrip_and_merge(self, tmp_project):
        save_config({"hotkey_toggle_detection": "Ctrl+F10"})
        cfg = load_config()
        assert cfg["hotkey_toggle_detection"] == "Ctrl+F10"
        assert cfg["negative_sample_ratio"] == 20  # 默认补齐


class TestState:
    def test_defaults_not_polluted(self, tmp_project):
        state = load_state()
        state["window"]["x"] = 12345
        assert DEFAULT_STATE["window"]["x"] is None, "默认值不可被运行期污染"

    def test_roundtrip(self, tmp_project):
        state = load_state()
        state["detect"]["fps"] = 30
        save_state(state)
        assert load_state()["detect"]["fps"] == 30


class TestTargetManager:
    def test_add_assigns_ids_and_data_dir(self, tmp_project):
        tm = TargetManager()
        t1 = tm.add("目标一", set_id="set_1")
        t2 = tm.add("目标二")
        assert t1.id == "target_1" and t2.id == "target_2"
        assert "set_1" in t1.data_dir

    def test_save_load_roundtrip_with_relative_paths(self, tmp_project):
        tm = TargetManager()
        tm.add("a", set_id="set_1", model_path=str(tmp_project / "runs" / "m.pt"))
        tm.save()
        tm2 = TargetManager()
        tm2.load()
        assert tm2.get("target_1").name == "a"
        assert tm2.get("target_1").model_path == str(tmp_project / "runs" / "m.pt")

    def test_load_repairs_relocated_absolute_path(self, tmp_project):
        """项目搬迁后，指向旧位置的绝对路径应被修复到 APP_ROOT。"""
        (tmp_project / "runs" / "detect" / "train" / "weights").mkdir(parents=True)
        best = tmp_project / "runs" / "detect" / "train" / "weights" / "best.pt"
        best.write_bytes(b"w")
        data = [{"id": "target_1", "name": "旧目标", "color": "#2ecc71",
                 "model_path": "D:\\旧机器\\runs\\detect\\train\\weights\\best.pt"}]
        paths.TARGETS_FILE.write_text(json.dumps(data, ensure_ascii=False),
                                      encoding="utf-8")

        tm = TargetManager()
        tm.load()
        assert tm.get("target_1").model_path == str(best)

    def test_invalid_detection_mode_normalized(self):
        from core.models import Target
        t = Target.from_dict({"id": "x", "name": "n", "color": "#fff",
                              "detection_mode": "magic"})
        assert t.detection_mode == "ai"


class TestSampleSetManager:
    def test_add_remove_roundtrip(self, tmp_project):
        m = SampleSetManager()
        ss = m.add("集合A")
        assert os.path.isdir(m.images_dir(ss.id))
        assert os.path.isdir(m.targets_dir(ss.id))
        m.save()

        m2 = SampleSetManager()
        m2.load()
        assert m2.get(ss.id).name == "集合A"
        assert m2.add("集合B").id == "set_2"  # 自增 ID 不复用

        m2.remove(ss.id, delete_files=True)
        assert not os.path.isdir(m.images_dir(ss.id))

    def test_ensure_default(self, tmp_project):
        m = SampleSetManager()
        ss = m.ensure_default()
        assert ss.name == "默认样本集"
