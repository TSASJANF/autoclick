"""[已废弃] 本脚本为 data/sets 样本集结构的一次性迁移工具，迁移已完成。
仅作历史保留，新代码请勿调用。

原始说明：
数据迁移脚本：将旧的扁平结构迁移到样本集结构。

旧结构:
  data/shared_images/          — 所有图片
  data/session.json            — 标注会话
  data/targets/target_N/       — 每个目标的标注
  targets.json                 — 目标配置（无 set_id 字段）

新结构:
  data/sets/set_1/             — 默认样本集
    images/                    — 图片（从 shared_images 迁移）
    session.json               — 会话（从 data/session.json 迁移）
    targets/target_N/          — 目标标注（从 data/targets/ 迁移）
    set.json                   — 样本集元数据
  data/sets/sets_index.json    — 样本集索引
  targets.json                 — 目标配置（添加 set_id 字段）

运行方式: python migrate_to_sets.py
"""

import os
import json
import shutil
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SETS_ROOT = os.path.join(DATA_DIR, "sets")
OLD_SHARED_IMAGES = os.path.join(DATA_DIR, "shared_images")
OLD_SESSION_FILE = os.path.join(DATA_DIR, "session.json")
OLD_TARGETS_DIR = os.path.join(DATA_DIR, "targets")
TARGETS_FILE = os.path.join(BASE_DIR, "targets.json")
BACKUP_DIR = os.path.join(DATA_DIR, "_backup_before_migration")


def migrate():
    print("=" * 60)
    print("数据迁移：扁平结构 → 样本集结构")
    print("=" * 60)

    # Check if migration has already been done
    if os.path.exists(SETS_ROOT):
        sets_index = os.path.join(SETS_ROOT, "sets_index.json")
        if os.path.exists(sets_index):
            print("\n[跳过] data/sets/ 已存在且包含 sets_index.json，看起来已经迁移过了。")
            print("如需重新迁移，请先手动删除 data/sets/ 目录。")
            return

    # Check for old data
    has_shared_images = os.path.isdir(OLD_SHARED_IMAGES)
    has_targets = os.path.isdir(OLD_TARGETS_DIR)
    has_session = os.path.isfile(OLD_SESSION_FILE)
    has_targets_json = os.path.isfile(TARGETS_FILE)

    if not has_shared_images and not has_targets:
        print("\n[跳过] 未发现旧数据（data/shared_images/ 和 data/targets/ 均不存在）。")
        return

    print(f"\n发现旧数据:")
    if has_shared_images:
        image_count = len([f for f in os.listdir(OLD_SHARED_IMAGES)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))])
        print(f"  • data/shared_images/ — {image_count} 张图片")
    if has_targets:
        target_dirs = [d for d in os.listdir(OLD_TARGETS_DIR)
                      if os.path.isdir(os.path.join(OLD_TARGETS_DIR, d))]
        print(f"  • data/targets/ — {len(target_dirs)} 个目标")
    if has_session:
        print(f"  • data/session.json — 标注会话")
    if has_targets_json:
        print(f"  • targets.json — 目标配置")

    # Confirm
    print(f"\n将创建默认样本集 '默认样本集' 并迁移所有数据。")
    print(f"旧数据将备份到 data/_backup_before_migration/")
    resp = input("\n确认开始迁移？(y/n): ").strip().lower()
    if resp != 'y':
        print("已取消。")
        return

    # Step 1: Create backup
    print("\n[1/6] 备份旧数据...")
    os.makedirs(BACKUP_DIR, exist_ok=True)
    if has_shared_images:
        shutil.copytree(OLD_SHARED_IMAGES, os.path.join(BACKUP_DIR, "shared_images"))
        print(f"  备份 shared_images/")
    if has_targets:
        shutil.copytree(OLD_TARGETS_DIR, os.path.join(BACKUP_DIR, "targets"))
        print(f"  备份 targets/")
    if has_session:
        shutil.copy2(OLD_SESSION_FILE, os.path.join(BACKUP_DIR, "session.json"))
        print(f"  备份 session.json")
    if has_targets_json:
        shutil.copy2(TARGETS_FILE, os.path.join(BACKUP_DIR, "targets.json"))
        print(f"  备份 targets.json")

    # Step 2: Create sample set directory structure
    print("\n[2/6] 创建样本集目录...")
    set_id = "set_1"
    set_dir = os.path.join(SETS_ROOT, set_id)
    set_images_dir = os.path.join(set_dir, "images")
    set_targets_dir = os.path.join(set_dir, "targets")
    os.makedirs(set_images_dir, exist_ok=True)
    os.makedirs(set_targets_dir, exist_ok=True)
    print(f"  创建 data/sets/{set_id}/")

    # Step 3: Move shared images
    print("\n[3/6] 迁移图片...")
    if has_shared_images:
        moved_count = 0
        for fname in os.listdir(OLD_SHARED_IMAGES):
            src = os.path.join(OLD_SHARED_IMAGES, fname)
            if os.path.isfile(src):
                dst = os.path.join(set_images_dir, fname)
                shutil.move(src, dst)
                moved_count += 1
        print(f"  移动 {moved_count} 个文件到 data/sets/{set_id}/images/")
    else:
        print("  无图片需要迁移")

    # Step 4: Move targets
    print("\n[4/6] 迁移目标标注...")
    if has_targets:
        for target_name in os.listdir(OLD_TARGETS_DIR):
            src = os.path.join(OLD_TARGETS_DIR, target_name)
            if os.path.isdir(src):
                dst = os.path.join(set_targets_dir, target_name)
                shutil.move(src, dst)
                print(f"  移动 {target_name}/")
    else:
        print("  无目标需要迁移")

    # Step 5: Update targets.json
    print("\n[5/6] 更新 targets.json...")
    if has_targets_json:
        with open(TARGETS_FILE, "r", encoding="utf-8") as f:
            targets = json.load(f)
        for t in targets:
            # Add set_id field
            t["set_id"] = set_id

            def _update_relative_path(old_path: str) -> str:
                """Update a relative path from data/targets/... to data/sets/set_id/targets/..."""
                if not old_path:
                    return old_path
                normalized = old_path.replace("\\", "/")
                if normalized.startswith("data/targets/"):
                    parts = normalized.split("/")
                    # data/targets/target_N/... -> data/sets/set_id/targets/target_N/...
                    rest = "/".join(parts[2:])  # target_N/...
                    return os.path.join("data", "sets", set_id, "targets", rest).replace("/", os.sep)
                return old_path

            # Update data_dir
            t["data_dir"] = _update_relative_path(t.get("data_dir", ""))
            # Update template_path
            t["template_path"] = _update_relative_path(t.get("template_path", ""))
            # Update source_image_path (if relative)
            t["source_image_path"] = _update_relative_path(t.get("source_image_path", ""))
        with open(TARGETS_FILE, "w", encoding="utf-8") as f:
            json.dump(targets, f, ensure_ascii=False, indent=2)
        print(f"  更新 {len(targets)} 个目标的 set_id 和 data_dir")
    else:
        print("  无 targets.json 需要更新")

    # Step 6: Create session.json and set.json
    print("\n[6/6] 创建样本集元数据...")
    now = datetime.datetime.now().isoformat(timespec="seconds")

    # Move session.json
    set_session_file = os.path.join(set_dir, "session.json")
    if has_session:
        shutil.move(OLD_SESSION_FILE, set_session_file)
        print(f"  移动 session.json")
    else:
        # Create empty session
        with open(set_session_file, "w", encoding="utf-8") as f:
            json.dump({"image_list": [], "current_index": -1}, f)
        print(f"  创建空 session.json")

    # Create set.json
    set_data = {
        "id": set_id,
        "name": "默认样本集",
        "created_at": now,
        "negative_sample_ratio": 20,
        "negative_min_positive": 5,
    }
    set_json_path = os.path.join(set_dir, "set.json")
    with open(set_json_path, "w", encoding="utf-8") as f:
        json.dump(set_data, f, ensure_ascii=False, indent=2)
    print(f"  创建 set.json")

    # Create sets_index.json
    sets_index = [set_data]
    sets_index_path = os.path.join(SETS_ROOT, "sets_index.json")
    with open(sets_index_path, "w", encoding="utf-8") as f:
        json.dump(sets_index, f, ensure_ascii=False, indent=2)
    print(f"  创建 sets_index.json")

    # Summary
    print("\n" + "=" * 60)
    print("迁移完成!")
    print("=" * 60)
    print(f"\n新目录结构:")
    print(f"  data/sets/{set_id}/")
    print(f"    images/       — 图片")
    print(f"    targets/      — 目标标注")
    print(f"    session.json  — 标注会话")
    print(f"    set.json      — 样本集元数据")
    print(f"\n备份位置: data/_backup_before_migration/")
    print(f"\n旧目录（shared_images/、targets/）中的文件已被移走。")
    print(f"如确认迁移无误，可手动删除备份目录。")


if __name__ == "__main__":
    migrate()
