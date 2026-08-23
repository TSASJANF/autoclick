"""core.paths 单元测试：路径解析、存储格式与失效路径修复。"""

from core import paths


class TestResolve:
    def test_relative_resolved_against_app_root(self, tmp_project):
        result = paths.resolve("data/sets/set_1")
        assert result == tmp_project / "data" / "sets" / "set_1"

    def test_backslash_relative(self, tmp_project):
        result = paths.resolve("data\\sets\\set_1")
        assert result == tmp_project / "data" / "sets" / "set_1"

    def test_absolute_passthrough(self, tmp_project):
        p = tmp_project / "somewhere" / "model.pt"
        assert paths.resolve(str(p)) == p

    def test_empty_returns_empty_path(self):
        assert str(paths.resolve("")) == "."


class TestToStored:
    def test_inside_root_becomes_relative_posix(self, tmp_project):
        p = tmp_project / "data" / "sets" / "set_1"
        assert paths.to_stored(p) == "data/sets/set_1"

    def test_outside_root_stays_absolute(self, tmp_project, tmp_path):
        outside = tmp_path / "elsewhere"
        stored = paths.to_stored(outside)
        assert stored.endswith("elsewhere")

    def test_empty(self):
        assert paths.to_stored("") == ""


class TestRepair:
    def test_existing_path_unchanged_and_relativized(self, tmp_project):
        p = tmp_project / "runs" / "detect" / "best.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        assert paths.repair(str(p)) == "runs/detect/best.pt"

    def test_dead_absolute_path_relocated_to_app_root(self, tmp_project):
        # 模拟项目搬迁：旧机器绝对路径失效，但 runs/ 结构在新位置存在
        (tmp_project / "runs" / "detect" / "train" / "weights").mkdir(parents=True)
        (tmp_project / "runs" / "detect" / "train" / "weights" / "best.pt").write_bytes(b"x")
        dead = "D:\\旧目录\\自动点击\\runs\\detect\\train\\weights\\best.pt"
        assert paths.repair(dead) == "runs/detect/train/weights/best.pt"

    def test_unrepairable_path_kept_as_is(self, tmp_project):
        dead = "D:\\不存在的盘\\model.pt"
        assert paths.repair(dead) == dead
