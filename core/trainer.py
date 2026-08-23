import os
import random
import shutil
from pathlib import Path

from ultralytics import YOLO
from PySide6.QtCore import QThread, Signal

from core.target_manager import SHARED_IMAGES_DIR


def get_device_info() -> tuple[str, str]:
    """Return (device_str, info_message) for training."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return "0", f"GPU: {gpu_name} (CUDA)"
        else:
            return "cpu", "GPU 不可用, 使用 CPU 训练 (安装 CUDA 版 PyTorch 可加速)"
    except ImportError:
        return "cpu", "PyTorch 未安装, 使用 CPU"


class TrainWorker(QThread):
    progress = Signal(str)   # log message
    finished = Signal(str)   # model path
    error = Signal(str)

    def __init__(self, data_yaml: str, epochs: int = 50, batch_size: int = 16,
                 img_size: int = 640, model_base: str = "yolov8n.pt",
                 project_dir: str = ""):
        super().__init__()
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.model_base = model_base
        self.project_dir = project_dir
        self._stop = False

    def request_stop(self):
        """Request cooperative stop."""
        self._stop = True

    def run(self):
        try:
            device, device_info = get_device_info()
            self.progress.emit(f"设备: {device_info}")

            # Load model with corruption retry
            self.progress.emit(f"加载模型: {self.model_base}")
            model = None
            last_error = None
            for attempt in range(2):  # max 2 attempts: original + 1 retry after corruption
                try:
                    model = YOLO(self.model_base)
                    break
                except Exception as e:
                    err_msg = str(e)
                    if "PytorchStreamReader" in err_msg and "failed finding central directory" in err_msg:
                        last_error = err_msg
                        if attempt == 0 and os.path.isfile(self.model_base):
                            self.progress.emit(f"模型文件损坏: {self.model_base}")
                            os.remove(self.model_base)
                            self.progress.emit("已删除损坏文件，重新下载中...")
                        else:
                            self.progress.emit("模型文件损坏且无法删除，无法继续")
                            raise
                    else:
                        raise
            if model is None:
                raise RuntimeError(f"模型加载失败: {last_error}")

            self.progress.emit(f"开始训练: epochs={self.epochs}, batch={self.batch_size}, imgsz={self.img_size}")
            self.progress.emit(f"数据: {self.data_yaml}")
            if self.project_dir:
                self.progress.emit(f"模型保存位置: {self.project_dir}")

            train_kwargs = dict(
                data=self.data_yaml,
                epochs=self.epochs,
                batch=self.batch_size,
                imgsz=self.img_size,
                device=device,
                verbose=True,
                workers=0,  # Use 0 workers to avoid multiprocessing issues on Windows
            )
            if self.project_dir:
                train_kwargs["project"] = self.project_dir

            results = model.train(**train_kwargs)

            # Find the best model path
            save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else None
            if save_dir and (save_dir / "weights" / "best.pt").exists():
                best_path = str(save_dir / "weights" / "best.pt")
            elif self.project_dir:
                best_path = str(Path(self.project_dir) / "train" / "weights" / "best.pt")
            else:
                best_path = str(Path("runs") / "detect" / "train" / "weights" / "best.pt")

            self.progress.emit(f"训练完成! 模型保存到: {best_path}")
            self.finished.emit(best_path)

        except Exception as e:
            self.error.emit(str(e))

        finally:
            self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """Clean up temporary training directory (train_data)."""
        try:
            yaml_path = Path(self.data_yaml)
            if yaml_path.exists():
                # train_data is the parent of data.yaml
                train_dir = yaml_path.parent
                if train_dir.name == "train_data" and train_dir.exists():
                    shutil.rmtree(train_dir, ignore_errors=True)
                    self.progress.emit(f"清理临时目录: {train_dir}")
        except Exception:
            pass


class Trainer:
    @staticmethod
    def find_labeled_images(dataset_dir: str, images_dir: str = None) -> list[str]:
        """Find image filenames that have corresponding label files.

        Returns list of image filenames (not full paths).
        """
        if images_dir is None:
            images_dir = SHARED_IMAGES_DIR

        abs_images_dir = os.path.abspath(images_dir)
        abs_labels_dir = os.path.join(os.path.abspath(dataset_dir), "labels")

        IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        labeled = []

        if not os.path.exists(abs_labels_dir):
            return labeled

        # Build set of label name stems
        label_stems = set()
        for f in os.listdir(abs_labels_dir):
            if f.endswith('.txt'):
                label_stems.add(os.path.splitext(f)[0])

        # Find matching images
        for img_file in os.listdir(abs_images_dir):
            stem, ext = os.path.splitext(img_file)
            if stem in label_stems and ext.lower() in IMAGE_EXTS:
                labeled.append(img_file)

        return labeled

    @staticmethod
    def create_data_yaml(dataset_dir: str, class_names: list[str], images_dir: str = None,
                         negative_ratio: int = 0, min_positive: int = 5) -> dict:
        """Create YOLO data.yaml with labeled images and optional negative samples.

        Creates train_data/images/ and train_data/labels/ with symlinks
        to the images that have corresponding labels, plus optional
        negative samples (images without labels, with empty label files).

        Args:
            dataset_dir: Directory containing labels/
            class_names: List of class names
            images_dir: Directory containing images (defaults to shared_images)
            negative_ratio: Percentage of negative samples in total set (0 = disabled)
            min_positive: Minimum positive samples required to enable negatives

        Returns:
            dict with keys: yaml_path, positive_count, negative_count, total_count
        """
        if images_dir is None:
            images_dir = SHARED_IMAGES_DIR

        abs_images_dir = os.path.abspath(images_dir)
        abs_dataset_dir = os.path.abspath(dataset_dir)
        abs_labels_dir = os.path.join(abs_dataset_dir, "labels")

        labeled_files = Trainer.find_labeled_images(dataset_dir, images_dir)
        if not labeled_files:
            raise ValueError("没有找到有标注的图片，请先标注图片")

        # Find negative sample candidates (images without labels for this target)
        IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        labeled_set = set(os.path.splitext(f)[0] for f in labeled_files)
        all_images = [f for f in os.listdir(abs_images_dir)
                      if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
        unlabeled = [f for f in all_images
                     if os.path.splitext(f)[0] not in labeled_set]

        # Calculate negative samples to include
        negative_files = []
        positive_count = len(labeled_files)
        if negative_ratio > 0 and positive_count >= min_positive and unlabeled:
            # negative / total = ratio/100 → negative = positive * ratio / (100 - ratio)
            target_neg = round(positive_count * negative_ratio / (100 - negative_ratio))
            target_neg = min(target_neg, len(unlabeled))
            if target_neg > 0:
                negative_files = random.sample(unlabeled, target_neg)

        # Create train_data directory structure
        train_dir = os.path.join(abs_dataset_dir, "train_data")
        train_images = os.path.join(train_dir, "images")
        train_labels = os.path.join(train_dir, "labels")
        os.makedirs(train_images, exist_ok=True)
        os.makedirs(train_labels, exist_ok=True)

        # Link positive samples (images + labels)
        for img_file in labeled_files:
            name_no_ext = os.path.splitext(img_file)[0]

            img_link = os.path.join(train_images, img_file)
            if not os.path.exists(img_link):
                try:
                    os.symlink(os.path.join(abs_images_dir, img_file), img_link)
                except OSError:
                    shutil.copy2(os.path.join(abs_images_dir, img_file), img_link)

            label_file = name_no_ext + ".txt"
            label_link = os.path.join(train_labels, label_file)
            if not os.path.exists(label_link):
                src = os.path.join(abs_labels_dir, label_file)
                if os.path.exists(src):
                    try:
                        os.symlink(src, label_link)
                    except OSError:
                        shutil.copy2(src, label_link)

        # Link negative samples (images + empty labels)
        for img_file in negative_files:
            name_no_ext = os.path.splitext(img_file)[0]

            img_link = os.path.join(train_images, img_file)
            if not os.path.exists(img_link):
                try:
                    os.symlink(os.path.join(abs_images_dir, img_file), img_link)
                except OSError:
                    shutil.copy2(os.path.join(abs_images_dir, img_file), img_link)

            # Create empty label file for negative sample
            label_link = os.path.join(train_labels, name_no_ext + ".txt")
            if not os.path.exists(label_link):
                with open(label_link, "w") as f:
                    pass  # empty file = no objects

        # Create data.yaml
        yaml_content = f"""path: {train_dir}
train: images
val: images

nc: {len(class_names)}
names: {class_names}
"""
        yaml_path = os.path.join(train_dir, "data.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        return {
            "yaml_path": yaml_path,
            "positive_count": positive_count,
            "negative_count": len(negative_files),
            "total_count": positive_count + len(negative_files),
            "unlabeled_count": len(unlabeled),
        }

    @staticmethod
    def get_dataset_stats(dataset_dir: str, images_dir: str = None) -> dict:
        """Get dataset statistics.

        Args:
            dataset_dir: Directory containing labels/
            images_dir: Directory containing images (defaults to shared_images)
        """
        if images_dir is None:
            images_dir = SHARED_IMAGES_DIR

        labels_dir = os.path.join(dataset_dir, "labels")

        images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.bmp', '.jpeg'))] if os.path.exists(images_dir) else []
        labels = [f for f in os.listdir(labels_dir) if f.endswith('.txt')] if os.path.exists(labels_dir) else []

        class_counts = {}
        for label_file in labels:
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

        return {
            "image_count": len(images),
            "label_count": len(labels),
            "class_counts": class_counts,
        }
