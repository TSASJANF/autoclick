import numpy as np
from ultralytics import YOLO


class Detection:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, conf: float, cls_id: int, name: str):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.conf = conf
        self.cls_id = cls_id
        self.name = name

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def __repr__(self):
        return f"Detection({self.name} {self.conf:.2f} [{self.x1},{self.y1},{self.x2},{self.y2}])"


class Detector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = None
        self.model_path = model_path
        self.names: dict[int, str] = {}

    def load(self, model_path: str | None = None):
        path = model_path or self.model_path
        self.model = YOLO(path)
        self.names = self.model.names
        self.model_path = path

    def detect(self, image: np.ndarray, conf: float = 0.5) -> list[Detection]:
        if self.model is None:
            return []
        results = self.model(image, conf=conf, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                c = float(box.conf[0])
                cls_id = int(box.cls[0])
                name = self.names.get(cls_id, str(cls_id))
                detections.append(Detection(x1, y1, x2, y2, c, cls_id, name))
        return detections
