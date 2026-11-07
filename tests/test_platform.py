"""core.hotkey 键串解析 与 core.opencv_detector 单元测试。

只测纯逻辑（解析/匹配/抑制），不触发真实按键与真实截图。
"""

import numpy as np

from core.hotkey import MOD_CONTROL, MOD_SHIFT, parse_key
from core.image_io import imwrite
from core.opencv_detector import TemplateDetector


class TestParseKey:
    def test_plain_function_key(self):
        vk, mods = parse_key("F9")
        assert vk == 0x78 and mods == 0

    def test_ctrl_combo(self):
        vk, mods = parse_key("Ctrl+F9")
        assert vk == 0x78 and mods == MOD_CONTROL

    def test_multi_modifiers(self):
        vk, mods = parse_key("Shift+Alt+A")
        assert vk == ord("A")
        assert mods & MOD_SHIFT

    def test_qt_keysequence_format(self):
        # QKeySequence.toString() 在不同平台可能输出 "Ctrl+F9"
        vk, mods = parse_key("Ctrl+F9")
        assert vk is not None

    def test_invalid_returns_none(self):
        assert parse_key("")[0] is None
        assert parse_key("???")[0] is None


class TestTemplateDetector:
    def _img_with_square(self, size=100, square=20, pos=(10, 10)):
        """构造带纹理方块的场景。注意：模板必须有方差 —— 纯色模板对
        TM_CCOEFF_NORMED 是退化输入（分母为 0，处处满分）。"""
        rng = np.random.default_rng(42)
        img = rng.integers(0, 40, (size, size, 3), dtype=np.uint8)
        x, y = pos
        template = rng.integers(180, 255, (square, square, 3), dtype=np.uint8)
        img[y:y + square, x:x + square] = template
        return img, template

    def test_finds_exact_match(self, tmp_path):
        img, template = self._img_with_square()
        tpath = str(tmp_path / "t.png")
        imwrite(tpath, template)

        det = TemplateDetector()
        assert det.load_template("t1", tpath, threshold=0.9)
        results = det.detect(img, "t1", "目标", "#fff")
        assert len(results) == 1
        d = results[0]
        assert d.x1 == 10 and d.y1 == 10 and d.width == 20

    def test_no_match_when_absent(self, tmp_path):
        """图中不存在模板时，常规阈值下应零命中。"""
        rng = np.random.default_rng(42)
        img = rng.integers(0, 40, (100, 100, 3), dtype=np.uint8)
        # 模板亮度与内容完全不同，不可能高分匹配
        template = rng.integers(180, 255, (20, 20, 3), dtype=np.uint8)
        tpath = str(tmp_path / "t.png")
        imwrite(tpath, template)
        det = TemplateDetector()
        det.load_template("t1", tpath, threshold=0.8)
        assert det.detect(img, "t1") == []

    def test_missing_template_file(self, tmp_path):
        det = TemplateDetector()
        assert det.load_template("t1", str(tmp_path / "nope.png")) is False

    def test_template_larger_than_image_skipped(self, tmp_path):
        template = np.full((300, 300, 3), 128, dtype=np.uint8)
        tpath = str(tmp_path / "big.png")
        imwrite(tpath, template)
        det = TemplateDetector()
        det.load_template("t1", tpath)
        small = np.zeros((50, 50, 3), dtype=np.uint8)
        assert det.detect(small, "t1") == []

    def test_duplicate_hits_suppressed(self, tmp_path):
        """重叠的重复命中必须被抑制为单个框（历史性能炸弹回归测试）。"""
        rng = np.random.default_rng(7)
        template = rng.integers(100, 255, (20, 20, 3), dtype=np.uint8)
        img = rng.integers(0, 40, (200, 200, 3), dtype=np.uint8)
        img[50:70, 50:70] = template
        tpath = str(tmp_path / "t.png")
        imwrite(tpath, template)
        det = TemplateDetector()
        det.load_template("t1", tpath, threshold=0.8)
        results = det.detect(img, "t1")
        near_target = [d for d in results
                       if abs(d.x1 - 50) <= 3 and abs(d.y1 - 50) <= 3]
        assert len(near_target) == 1, "同一位置只允许一个命中"
        assert len(results) <= 3
