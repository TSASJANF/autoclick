"""core.detection_engine / core.models 单元测试：点击策略、NMS、跨源去重。

这些是自动点击正确性的核心规则，历史上从未有测试覆盖。
"""

from core.detection_engine import select_click_targets
from core.models import Detection, nms


def det(target_id, conf, x1=0, y1=0, x2=10, y2=10, source="ai"):
    return Detection(x1, y1, x2, y2, conf, target_id, target_id, "#fff", source)


class TestSelectClickTargets:
    def test_no_priority_single_group_click_all(self):
        dets = [det("a", 0.9), det("a", 0.7), det("a", 0.5)]
        result = select_click_targets(dets, {}, use_priority=False, click_all=True)
        assert [d.conf for d in result] == [0.9, 0.7, 0.5]

    def test_no_priority_click_best_only(self):
        dets = [det("a", 0.6), det("b", 0.9), det("a", 0.8)]
        result = select_click_targets(dets, {}, use_priority=False, click_all=False)
        assert len(result) == 1 and result[0].conf == 0.9

    def test_priority_focuses_highest_priority_group(self):
        dets = [det("low", 0.99), det("high", 0.60)]
        result = select_click_targets(
            dets, {"high": 0, "low": 5}, use_priority=True, click_all=True)
        assert all(d.target_id == "high" for d in result)

    def test_priority_tie_broken_by_confidence(self):
        dets = [det("a", 0.70), det("b", 0.90)]
        result = select_click_targets(
            dets, {"a": 1, "b": 1}, use_priority=True, click_all=False)
        assert result[0].target_id == "b"

    def test_empty_input(self):
        assert select_click_targets([], {}, True, True) == []


class TestNms:
    def test_same_target_overlap_suppressed(self):
        a = det("a", 0.95, 0, 0, 100, 100)
        b = det("a", 0.80, 5, 5, 105, 105)   # 与 a 高度重叠
        c = det("a", 0.70, 500, 500, 600, 600)  # 无重叠
        kept = nms([b, c, a], iou_threshold=0.3, same_target_only=True)
        assert a in kept and c in kept and b not in kept

    def test_different_targets_kept_when_same_target_only(self):
        a = det("a", 0.95, 0, 0, 100, 100)
        b = det("b", 0.80, 5, 5, 105, 105)
        kept = nms([a, b], iou_threshold=0.3, same_target_only=True)
        assert len(kept) == 2

    def test_cross_source_dedupe(self):
        """both 模式：ai 与 opencv 命中同一物体时应合并为一个。"""
        from core.detection_engine import DetectionEngine
        a = det("t", 0.95, 0, 0, 100, 100, source="ai")
        b = det("t", 0.80, 2, 2, 102, 102, source="opencv")
        kept = DetectionEngine._dedupe_cross_source([a, b])
        assert len(kept) == 1 and kept[0].source == "ai"

    def test_cross_source_dedupe_preserves_different_targets(self):
        """不同目标的重叠框不应被跨源去重错误抑制。"""
        from core.detection_engine import DetectionEngine
        a = det("target_1", 0.95, 0, 0, 100, 100, source="ai")
        b = det("target_2", 0.80, 2, 2, 102, 102, source="ai")
        kept = DetectionEngine._dedupe_cross_source([a, b])
        assert len(kept) == 2


class TestDetectionGeometry:
    def test_center_and_iou(self):
        a = Detection(0, 0, 10, 10, 1.0, "a", "a", "#fff", "ai")
        b = Detection(5, 5, 15, 15, 1.0, "a", "a", "#fff", "ai")
        assert a.center == (5, 5)
        assert abs(a.iou(b) - 25 / 175) < 1e-9
        assert a.iou(a) == 1.0
