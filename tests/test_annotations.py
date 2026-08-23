"""core.annotations 单元测试：标注读写往返、归属键、状态批量查询。"""

import os

from core import annotations as anno


class TestLabelRoundtrip:
    def test_write_read_roundtrip(self, tmp_path):
        label = str(tmp_path / "labels" / "img1.txt")
        boxes = [anno.Box(10, 20, 110, 220, cls_id=0, owner_target_id="t1")]
        anno.write_labels(label, boxes, img_w=200, img_h=400)

        loaded = anno.read_labels(label, img_w=200, img_h=400)
        assert len(loaded) == 1
        b = loaded[0]
        assert (b.x1, b.y1, b.x2, b.y2) == (10, 20, 110, 220)
        assert b.cls_id == 0

    def test_read_missing_file_returns_empty(self, tmp_path):
        assert anno.read_labels(str(tmp_path / "nope.txt"), 100, 100) == []

    def test_malformed_lines_skipped(self, tmp_path):
        p = tmp_path / "bad.txt"
        p.write_text("not a label\n0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        boxes = anno.read_labels(str(p), 100, 100)
        assert len(boxes) == 1

    def test_empty_label_counts_as_no_annotation(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        assert anno.has_label(str(p)) is False


class TestLabelStatusMap:
    def _make_target_dir(self, root, tid, labeled_stems):
        """创建目标数据目录（含 labels/ 子目录），返回数据目录路径。"""
        d = root / tid / "labels"
        d.mkdir(parents=True)
        for stem in labeled_stems:
            (d / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        return str(root / tid)

    def test_status_classification(self, tmp_path):
        t1 = self._make_target_dir(tmp_path, "t1", ["img1"])
        t2 = self._make_target_dir(tmp_path, "t2", ["img2"])
        images = [str(tmp_path / "img1.png"), str(tmp_path / "img2.png"),
                  str(tmp_path / "img3.png")]

        status = anno.label_status_map(
            {"t1": t1, "t2": t2}, images, current_target_id="t1")

        assert status[images[0]] == "selected"    # 当前目标已标
        assert status[images[1]] == "other"       # 其他目标已标
        assert status[images[2]] == "none"        # 无人标注

    def test_empty_label_file_treated_as_unlabeled(self, tmp_path):
        d = tmp_path / "t" / "labels"
        d.mkdir(parents=True)
        (d / "img.txt").write_text("")   # 空文件不算已标注
        img = str(tmp_path / "img.png")
        status = anno.label_status_map({"t": str(d)}, [img], "t")
        assert status[img] == "none"


class TestReadClassIds:
    def test_collects_ids(self, tmp_path):
        (tmp_path / "a.txt").write_text("0 0.1 0.1 0.1 0.1\n1 0.1 0.1 0.1 0.1\n")
        (tmp_path / "b.txt").write_text("1 0.1 0.1 0.1 0.1\n")
        assert anno.read_class_ids(str(tmp_path)) == {0, 1}

    def test_missing_dir(self, tmp_path):
        assert anno.read_class_ids(str(tmp_path / "nope")) == set()
