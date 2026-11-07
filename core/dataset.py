"""训练数据集准备：定位已标注图片、组装 train_data 目录、生成 data.yaml。

从旧 core/trainer.py 的静态方法迁移，修复的正确性缺陷：

1. 临时目录残留污染：每次重建前清空 train_data（旧版增量复用，上次
   训练的负样本空标签会永久混入后续训练）；

2. train = val 同目录（BIZ-01）：验证集等于训练集，mAP 虚高、best.pt
   的早停失去意义。现按 80/20 划分 train/val（确定性随机，种子固定）；

3. Windows 无符号链接权限时 os.symlink 必然失败、退化为整库复制
   （PERF-03）：改用硬链接（同卷 os.link 无需特权），失败再复制。

负样本抽样与划分共用一个可配置种子（默认固定），训练可复现（BIZ-03）。
"""

from __future__ import annotations

import logging
import os
import random
import shutil

from . import paths

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 固定默认种子：保证同一份数据两次训练的负样本抽取与划分一致（可复现）
DEFAULT_SEED = 42


def list_images(directory: str) -> list[str]:
    """列出目录下所有图片文件名。"""
    if not os.path.isdir(directory):
        return []
    return [f for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS]


def find_labeled_images(dataset_dir: str, images_dir: str) -> list[str]:
    """找出在 dataset_dir/labels/ 有对应标注的图片文件名列表。"""
    labels_dir = os.path.join(paths.resolve(dataset_dir), "labels")
    if not os.path.isdir(labels_dir):
        return []
    label_stems = {os.path.splitext(f)[0] for f in os.listdir(labels_dir)
                   if f.endswith(".txt")}
    return [f for f in list_images(images_dir)
            if os.path.splitext(f)[0] in label_stems]


def get_dataset_stats(dataset_dir: str, images_dir: str) -> dict:
    """数据集统计：图片数、标注文件数、每类框数。"""
    labels_dir = os.path.join(paths.resolve(dataset_dir), "labels")
    images = list_images(images_dir)
    labels = ([f for f in os.listdir(labels_dir) if f.endswith(".txt")]
              if os.path.isdir(labels_dir) else [])

    class_counts: dict[int, int] = {}
    for fname in labels:
        with open(os.path.join(labels_dir, fname), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        cls_id = int(parts[0])
                    except ValueError:
                        continue
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

    return {"image_count": len(images), "label_count": len(labels),
            "class_counts": class_counts}


def count_negative_candidates(positive_count: int, unlabeled_count: int,
                              ratio_percent: int, min_positive: int) -> int:
    """按「负样本占训练集百分比」换算应抽取的负样本数（纯函数，可测）。"""
    if ratio_percent <= 0 or positive_count < min_positive or unlabeled_count <= 0:
        return 0
    if ratio_percent >= 100:
        return unlabeled_count
    target = round(positive_count * ratio_percent / (100 - ratio_percent))
    return min(target, unlabeled_count)


def _place_file(src: str, dst: str) -> None:
    """把文件放入训练目录：硬链接优先（同卷、零拷贝、无需特权），
    失败回退复制（跨卷等情况）。"""
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)


def _val_count(total: int, val_ratio: float) -> int:
    """验证集数量：>=5 时按比例（至少 1 张）；2-4 张时 1 张；
    单张时 0（该图同时进 train/val，行为与旧版一致且明确记录）。"""
    if total >= 5:
        return max(1, round(total * val_ratio))
    if total >= 2:
        return 1
    return 0


def prepare_training_dir(dataset_dir: str, class_names: list[str], images_dir: str,
                         negative_ratio: int = 0, min_positive: int = 5,
                         seed: int = DEFAULT_SEED,
                         val_ratio: float = 0.2) -> dict:
    """组装训练目录（train/val 划分 + data.yaml），返回统计信息。

    dataset_dir: 含 labels/ 的目标数据目录
    images_dir:  样本集图片目录
    negative_ratio: 负样本占训练集百分比（0 = 关闭）
    min_positive: 启用负样本所需最小正样本数
    seed: 负样本抽样与 train/val 划分共用的随机种子
    val_ratio: 验证集占比
    """
    abs_images_dir = paths.resolve(images_dir)
    abs_dataset_dir = paths.resolve(dataset_dir)

    labeled_files = find_labeled_images(dataset_dir, images_dir)
    if not labeled_files:
        raise ValueError("没有找到有标注的图片，请先标注图片")

    labeled_stems = {os.path.splitext(f)[0] for f in labeled_files}
    all_images = list_images(images_dir)
    unlabeled = [f for f in all_images
                 if os.path.splitext(f)[0] not in labeled_stems]

    positive_count = len(labeled_files)
    negative_count = count_negative_candidates(
        positive_count, len(unlabeled), negative_ratio, min_positive)
    rng = random.Random(seed)
    negative_files = rng.sample(unlabeled, negative_count) if negative_count else []

    # 每次重建前清空临时目录，杜绝上次训练的负样本残留
    train_dir = os.path.join(abs_dataset_dir, "train_data")
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir, ignore_errors=True)

    # (图片文件, 标签源路径或None) 的全集，再做确定性 train/val 划分
    samples: list[tuple[str, str | None]] = [
        (f, os.path.join(abs_dataset_dir, "labels",
                         os.path.splitext(f)[0] + ".txt"))
        for f in labeled_files
    ] + [(f, None) for f in negative_files]

    total = len(samples)
    n_val = _val_count(total, val_ratio)
    order = list(range(total))
    rng.shuffle(order)
    val_idx = set(order[:n_val])

    split_dirs = {}
    for split in ("train", "val"):
        split_dirs[split] = (
            os.path.join(train_dir, "images", split),
            os.path.join(train_dir, "labels", split),
        )
        os.makedirs(split_dirs[split][0])
        os.makedirs(split_dirs[split][1])

    single_sample = total == 1
    for i, (img_file, label_src) in enumerate(samples):
        stem = os.path.splitext(img_file)[0]
        # 单样本场景：同一文件进 train 和 val（链接两次，无拷贝开销）
        targets = ("train", "val") if single_sample else \
            (("val",) if i in val_idx else ("train",))
        for split in targets:
            images_dst, labels_dst = split_dirs[split]
            _place_file(os.path.join(abs_images_dir, img_file),
                        os.path.join(images_dst, img_file))
            label_dst = os.path.join(labels_dst, stem + ".txt")
            if label_src and os.path.exists(label_src):
                _place_file(label_src, label_dst)
            elif not os.path.exists(label_dst):
                open(label_dst, "w").close()   # 负样本：空标注

    train_count = total - (0 if single_sample else n_val)
    val_count = 1 if single_sample else n_val
    logger.info("训练目录已就绪: %s (正 %d / 负 %d / train %d / val %d)",
                train_dir, positive_count, len(negative_files),
                train_count, val_count)

    yaml_content = (f"path: {train_dir}\n"
                    f"train: images/train\n"
                    f"val: images/val\n\n"
                    f"nc: {len(class_names)}\n"
                    f"names: {class_names}\n")
    yaml_path = os.path.join(train_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    return {
        "yaml_path": yaml_path,
        "positive_count": positive_count,
        "negative_count": len(negative_files),
        "total_count": total,
        "unlabeled_count": len(unlabeled),
        "train_count": train_count,
        "val_count": val_count,
        "seed": seed,
    }


def cleanup_training_dir(data_yaml: str) -> None:
    """训练结束后清理 train_data 临时目录。"""
    try:
        train_dir = os.path.dirname(os.path.abspath(data_yaml))
        if os.path.basename(train_dir) == "train_data" and os.path.isdir(train_dir):
            shutil.rmtree(train_dir, ignore_errors=True)
    except OSError:
        logger.exception("清理训练临时目录失败: %s", data_yaml)


def suggest_hyperparams(positive_count: int, negative_count: int,
                        avg_width: float, avg_height: float,
                        min_side: int) -> tuple[int, int, int]:
    """按数据集规模与图片尺寸推荐 (img_size, batch, epochs)（纯函数）。

    与旧版 TrainTab._auto_adjust_params 的启发式一致：
    小数据集多轮次小批次，大图降批次防 OOM。
    """
    effective = positive_count + negative_count
    avg_side = (avg_width + avg_height) / 2

    if min_side <= 320:
        img_size = 320
    elif min_side <= 416:
        img_size = 416
    elif avg_side <= 600:
        img_size = 640
    elif avg_side <= 900:
        img_size = 800
    else:
        img_size = 1280

    if effective < 30:
        batch, epochs = 4, 150
    elif effective < 80:
        batch, epochs = 4, 120
    elif effective < 200:
        batch, epochs = 8, 80
    elif effective < 500:
        batch, epochs = 16, 60
    else:
        batch, epochs = 16, 40

    if img_size >= 1280 and batch > 8:
        batch = 8
    elif img_size >= 800 and batch > 16:
        batch = 16
    return img_size, batch, epochs
