"""pytest 共享夹具。"""

import os
import sys

import pytest

# 保证从项目根目录导入
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import paths  # noqa: E402


@pytest.fixture
def tmp_project(tmp_path, monkeypatch):
    """把 APP_ROOT 相关常量指到临时目录，测试持久化互不干扰。"""
    monkeypatch.setattr(paths, "APP_ROOT", tmp_path)
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(paths, "SETS_DIR", tmp_path / "data" / "sets")
    monkeypatch.setattr(paths, "TARGETS_FILE", tmp_path / "targets.json")
    monkeypatch.setattr(paths, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(paths, "STATE_FILE", tmp_path / "app_state.json")
    monkeypatch.setattr(paths, "SETS_INDEX_FILE", tmp_path / "data" / "sets" / "sets_index.json")
    return tmp_path
