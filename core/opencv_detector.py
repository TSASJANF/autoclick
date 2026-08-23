import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class OpenCVDetection:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, conf: float, target_id: str, target_name: str):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.conf = conf
        self.target_id = target_id
        self.target_name = target_name

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class OpenCVDetector:
    def __init__(self):
        self.templates: dict[str, np.ndarray] = {}  # target_id -> template image (BGR)
        self.thresholds: dict[str, float] = {}  # target_id -> threshold

    def load_template(self, target_id: str, template_path: str, threshold: float = 0.8) -> bool:
        """Load a template image. Returns True on success."""
        img = cv2.imread(template_path)
        if img is not None:
            self.templates[target_id] = img
            self.thresholds[target_id] = threshold
            logger.debug(f"模板已加载: {target_id}, 路径: {template_path}, 尺寸: {img.shape}, 阈值: {threshold}")
            return True
        else:
            logger.debug(f"模板加载失败: {target_id}, 路径: {template_path}")
            return False

    def remove_template(self, target_id: str):
        self.templates.pop(target_id, None)
        self.thresholds.pop(target_id, None)

    def detect(self, image: np.ndarray, target_ids: list[str] | None = None) -> list[OpenCVDetection]:
        detections = []
        ids = target_ids or list(self.templates.keys())
        logger.debug(f"开始检测, 已加载模板: {list(self.templates.keys())}, 输入图像尺寸: {image.shape}")

        for tid in ids:
            template = self.templates.get(tid)
            if template is None:
                logger.debug(f"模板 {tid} 未加载, 跳过")
                continue
            threshold = self.thresholds.get(tid, 0.8)

            # Multi-scale matching for robustness
            th, tw = template.shape[:2]
            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            max_val = np.max(result)
            min_val = np.min(result)
            mean_val = np.mean(result)
            logger.debug(f"模板 {tid}: 尺寸={tw}x{th}, 阈值={threshold}, 匹配度: 最大={max_val:.4f}, 最小={min_val:.4f}, 平均={mean_val:.4f}")
            locations = np.where(result >= threshold)
            logger.debug(f"模板 {tid}: 找到 {len(locations[0])} 个匹配位置")

            for pt in zip(*locations[::-1]):
                x1, y1 = int(pt[0]), int(pt[1])
                x2, y2 = x1 + tw, y1 + th
                conf = float(result[pt[1], pt[0]])
                detections.append(OpenCVDetection(x1, y1, x2, y2, conf, tid, tid))

        # Non-maximum suppression for overlapping detections
        return self._nms(detections)

    def _nms(self, detections: list[OpenCVDetection], iou_thresh: float = 0.3) -> list[OpenCVDetection]:
        if not detections:
            return []
        # Sort by confidence descending
        detections.sort(key=lambda d: d.conf, reverse=True)
        keep = []
        for det in detections:
            # Check overlap with already kept detections
            is_dup = False
            for kept in keep:
                if det.target_id == kept.target_id and self._iou(det, kept) > iou_thresh:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(det)
        return keep

    @staticmethod
    def _iou(a: OpenCVDetection, b: OpenCVDetection) -> float:
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0
