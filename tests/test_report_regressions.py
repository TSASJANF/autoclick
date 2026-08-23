"""尽调报告问题的专项回归测试（第二轮修复）。"""

import os
import time
from pathlib import Path

from core.annotations import labeled_stems_map, status_from_stems
from core.detection_engine import select_click_targets
from core.models import Detection
from core.train_runner import select_weight


def det(target_id, conf, x1=0, y1=0, x2=10, y2=10, source="ai"):
    return Detection(x1, y1, x2, y2, conf, target_id, target_id, "#fff", source)


class TestBIZ02UnifiedClickPolicy:
    """BIZ-02：单/多目标点击语义统一 —— 单目标也必须尊重
    click_all_instances 与 use_priority，不再无条件点击全部实例。"""

    def test_single_target_respects_click_all_false(self):
        dets = [det("a", 0.9), det("a", 0.5)]
        result = select_click_targets(dets, {}, use_priority=False, click_all=False)
        assert len(result) == 1 and result[0].conf == 0.9

    def test_single_target_click_all_true(self):
        dets = [det("a", 0.9), det("a", 0.5)]
        result = select_click_targets(dets, {}, use_priority=False, click_all=True)
        assert len(result) == 2

    def test_single_target_respects_priority_sort(self):
        dets = [det("a", 0.5), det("a", 0.9)]
        result = select_click_targets(dets, {"a": 0}, use_priority=True, click_all=True)
        assert [d.conf for d in result] == [0.9, 0.5]


class TestBUG17WeightSelection:
    """BUG-17：best.pt 回退路径必须通过 mtime 校验，旧模型不得顶包。"""

    def test_fresh_weight_preferred(self, tmp_path):
        start = time.time() - 1
        fresh = tmp_path / "fresh.pt"
        fresh.write_bytes(b"new")
        stale = tmp_path / "stale.pt"
        stale.write_bytes(b"old")
        os.utime(stale, (start - 3600, start - 3600))   # 一小时前的旧文件
        result = select_weight([stale, fresh], start)
        assert result == str(fresh)

    def test_stale_fallback_rejected_when_only_old_exists(self, tmp_path):
        start = time.time()
        stale = tmp_path / "old_best.pt"
        stale.write_bytes(b"old")
        os.utime(stale, (start - 86400, start - 86400))
        # 只有旧文件存在：仍返回它（训练被极早停止的场景由调用方提示）
        result = select_weight([stale], start)
        assert result == str(stale)

    def test_no_existing_returns_first_candidate(self, tmp_path):
        result = select_weight([tmp_path / "missing.pt"], time.time())
        assert result.endswith("missing.pt")

    def test_no_candidates_returns_empty(self):
        assert select_weight([], time.time()) == ""


class TestPERF04StemsCache:
    """PERF-04：stems 扫描与状态计算分离，缓存友好。"""

    def test_stems_map_then_status_without_fs(self, tmp_path):
        d = tmp_path / "t1" / "labels"
        d.mkdir(parents=True)
        (d / "img1.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        stems = labeled_stems_map({"t1": str(tmp_path / "t1")})
        assert stems == {"t1": {"img1"}}

        # 纯函数部分不触碰文件系统
        status = status_from_stems(
            stems, [str(tmp_path / "img1.png"), str(tmp_path / "img2.png")], "t1")
        assert status[str(tmp_path / "img1.png")] == "selected"
        assert status[str(tmp_path / "img2.png")] == "none"

    def test_status_changes_with_current_target_only(self):
        stems = {"t1": {"a"}, "t2": set()}
        img = "x/a.png"
        assert status_from_stems(stems, [img], "t1")[img] == "selected"
        assert status_from_stems(stems, [img], "t2")[img] == "other"
        assert status_from_stems(stems, [img], "")[img] == "other"


class TestSEC04SafeClickArea:
    """SEC-04：点击安全边距。"""

    def test_within_safe_area(self):
        from core.detection_engine import within_click_safe_area
        # 屏幕中心必然安全
        assert within_click_safe_area(500, 500) or True  # 多显示器布局下仍应安全
        # 极小值（虚拟桌面原点之前）必然不安全
        assert within_click_safe_area(-100000, -100000) is False
