"""core.json_store 单元测试：原子写、损坏备份、默认值合并。"""

import json
import os

from core import json_store


class TestReadWrite:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "a.json"
        json_store.write_json(p, {"x": 1})
        assert json_store.read_json(p, {}) == {"x": 1}

    def test_missing_returns_default(self, tmp_path):
        assert json_store.read_json(tmp_path / "nope.json", {"d": 1}) == {"d": 1}

    def test_corrupt_file_backed_up_not_overwritten(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{ 这不是合法 JSON", encoding="utf-8")
        result = json_store.read_json(p, {"fallback": True})
        assert result == {"fallback": True}
        backups = list(tmp_path.glob("bad.json.corrupt-*"))
        assert len(backups) == 1, "损坏文件必须被备份而不是直接丢弃"
        assert "{ 这不是合法 JSON" in backups[0].read_text(encoding="utf-8")


class TestMergeDefaults:
    def test_missing_keys_filled(self):
        merged = json_store.merge_defaults({"a": 1}, {"a": 0, "b": {"x": 1}})
        assert merged == {"a": 1, "b": {"x": 1}}

    def test_nested_merge(self):
        loaded = {"window": {"x": 5}}
        defaults = {"window": {"x": None, "y": None, "w": 100}}
        merged = json_store.merge_defaults(loaded, defaults)
        assert merged["window"] == {"x": 5, "y": None, "w": 100}

    def test_defaults_not_polluted_by_mutation(self):
        defaults = {"window": {"x": 1}}
        merged = json_store.merge_defaults({}, defaults)
        merged["window"]["x"] = 999
        assert defaults["window"]["x"] == 1, "合并结果必须是深拷贝"
