"""core.dataset 单元测试：负样本计算、训练目录组装（含残留清理）、超参推荐。"""

import os

import pytest

from core import dataset as ds


def _make_set(root, images, labeled):
    """构造 images_dir + dataset_dir/labels。返回 (images_dir, dataset_dir)。"""
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    labels_dir = root / "target" / "labels"
    labels_dir.mkdir(parents=True)
    for name in images:
        (images_dir / name).write_bytes(b"fake")
    for stem in labeled:
        (labels_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    return str(images_dir), str(root / "target")


class TestNegativeCandidates:
    def test_ratio_math(self):
        # 10 正样本 + 25% 负样本占比 -> neg = 10*25/75 ≈ 3
        assert ds.count_negative_candidates(10, 100, 25, 5) == 3

    def test_capped_by_unlabeled(self):
        assert ds.count_negative_candidates(10, 2, 25, 5) == 2

    def test_disabled_when_ratio_zero(self):
        assert ds.count_negative_candidates(10, 100, 0, 5) == 0

    def test_disabled_below_min_positive(self):
        assert ds.count_negative_candidates(3, 100, 20, 5) == 0


class TestFindLabeledImages:
    def test_only_labeled_returned(self, tmp_path):
        images_dir, ds_dir = _make_set(
            tmp_path, ["a.png", "b.png", "c.jpg", "d.txt"],
            labeled=["a", "c"])
        assert sorted(ds.find_labeled_images(ds_dir, images_dir)) == ["a.png", "c.jpg"]

    def test_no_labels_dir(self, tmp_path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        assert ds.find_labeled_images(str(tmp_path), str(images_dir)) == []


class TestPrepareTrainingDir:
    def test_structure_and_counts(self, tmp_path):
        images_dir, ds_dir = _make_set(
            tmp_path, ["a.png", "b.png", "c.png"], labeled=["a", "b"])
        result = ds.prepare_training_dir(
            ds_dir, ["class_0"], images_dir,
            negative_ratio=25, min_positive=1, seed=42)

        train_root = os.path.join(ds_dir, "train_data")
        assert result["positive_count"] == 2
        assert result["total_count"] == 3
        assert result["train_count"] + result["val_count"] == 3
        assert result["val_count"] == 1   # 3 张图 >= 2 → 1 张验证
        assert os.path.exists(os.path.join(train_root, "images", "train"))
        assert os.path.exists(os.path.join(train_root, "images", "val"))
        assert os.path.exists(os.path.join(train_root, "labels", "train"))
        assert os.path.exists(os.path.join(train_root, "labels", "val"))
        assert os.path.exists(result["yaml_path"])
        # yaml 必须指向不同目录（train ≠ val，BIZ-01）
        with open(result["yaml_path"], encoding="utf-8") as f:
            yaml_text = f.read()
        assert "train: images/train" in yaml_text
        assert "val: images/val" in yaml_text
        # 负样本必须有空标注文件
        empty_labels = []
        for split in ("train", "val"):
            label_dir = os.path.join(ds_dir, "train_data", "labels", split)
            for f in os.listdir(label_dir):
                if os.path.getsize(os.path.join(label_dir, f)) == 0:
                    empty_labels.append(f)
        assert len(empty_labels) == result["negative_count"]

    def test_no_labeled_raises(self, tmp_path):
        images_dir, ds_dir = _make_set(tmp_path, ["a.png"], labeled=[])
        with pytest.raises(ValueError):
            ds.prepare_training_dir(ds_dir, ["c"], images_dir)

    def test_stale_train_data_wiped(self, tmp_path):
        """关键回归：上次训练的残留负样本不得混入本次训练。"""
        images_dir, ds_dir = _make_set(
            tmp_path, ["a.png", "b.png", "c.png"], labeled=["a", "b"])

        ds.prepare_training_dir(ds_dir, ["class_0"], images_dir, seed=1)
        # 模拟训练中断：train_data 残留

        ds.prepare_training_dir(
            ds_dir, ["class_0"], images_dir, negative_ratio=0, seed=2)
        for split in ("train", "val"):
            labels = os.listdir(os.path.join(ds_dir, "train_data", "labels", split))
            for f in labels:
                assert os.path.getsize(os.path.join(
                    ds_dir, "train_data", "labels", split, f)) > 0, \
                    "本次未启用负样本，不允许残留上次的空标签"

    def test_seed_reproducibility(self, tmp_path):
        """同一 seed 两次组装：负样本与划分完全一致（BIZ-03）。"""
        images_dir, ds_dir = _make_set(
            tmp_path, [f"i{i}.png" for i in range(20)], labeled=["i0", "i1"])

        def snapshot():
            r = ds.prepare_training_dir(ds_dir, ["c"], images_dir,
                                        25, 1, seed=7)
            files = {s: sorted(os.listdir(os.path.join(
                ds_dir, "train_data", "images", s))) for s in ("train", "val")}
            return r, files

        r1, f1 = snapshot()
        r2, f2 = snapshot()
        assert r1["negative_count"] == r2["negative_count"]
        assert f1 == f2, "固定种子下两次组装的结果必须一致"

    def test_single_sample_goes_to_both_splits(self, tmp_path):
        """单样本场景：同一图进 train 和 val（行为明确，与旧版一致）。"""
        images_dir, ds_dir = _make_set(tmp_path, ["only.png"], labeled=["only"])
        result = ds.prepare_training_dir(ds_dir, ["c"], images_dir)
        assert result["train_count"] == 1 and result["val_count"] == 1
        assert os.path.exists(os.path.join(ds_dir, "train_data", "images", "train", "only.png"))
        assert os.path.exists(os.path.join(ds_dir, "train_data", "images", "val", "only.png"))


class TestSuggestHyperparams:
    def test_small_dataset(self):
        # min_side=200 -> img_size=320；12 个样本 -> batch=4, epochs=150
        assert ds.suggest_hyperparams(10, 2, 1920, 1080, 200) == (320, 4, 150)

    def test_medium_size_images(self):
        img, batch, epochs = ds.suggest_hyperparams(10, 2, 700, 500, 640)
        assert img == 640 and batch == 4 and epochs == 150

    def test_large_images_reduce_batch(self):
        img, batch, _ = ds.suggest_hyperparams(100, 100, 2500, 1400, 2200)
        assert img == 1280 and batch == 8


class TestDatasetStats:
    def test_stats(self, tmp_path):
        images_dir, ds_dir = _make_set(
            tmp_path, ["a.png", "b.png"], labeled=["a"])
        stats = ds.get_dataset_stats(ds_dir, images_dir)
        assert stats["image_count"] == 2
        assert stats["label_count"] == 1
        assert stats["class_counts"] == {0: 1}
