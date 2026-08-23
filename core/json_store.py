"""原子 JSON 读写工具。

历史问题：所有持久化点各自手写 open/json.dump，损坏时静默吞异常返回
默认值，下一次 save 直接覆盖 —— 用户数据无预警丢失。本模块统一：

- 写入：临时文件 + os.replace，崩溃不会留下半截文件；
- 读取：JSON 损坏时先把损坏文件改名备份（*.corrupt-<时间戳>），再返回
  默认值，保证用户有机会手工抢救数据。
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def read_json(path: str | os.PathLike, default: Any) -> Any:
    """读取 JSON 文件；不存在或损坏时返回 default（损坏文件会被备份）。"""
    path = os.fspath(path)
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        backup = f"{path}.corrupt-{datetime.datetime.now():%Y%m%d_%H%M%S}"
        try:
            os.replace(path, backup)
            logger.warning("JSON 损坏已备份: %s -> %s (%s)", path, backup, e)
        except OSError:
            logger.exception("JSON 损坏且备份失败: %s", path)
    except OSError as e:
        logger.error("JSON 读取失败: %s (%s)", path, e)
    return default


def write_json(path: str | os.PathLike, data: Any) -> None:
    """原子写入 JSON 文件（临时文件 + 替换）。"""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def merge_defaults(loaded: dict, defaults: dict) -> dict:
    """递归合并：以 defaults 为底，loaded 覆盖已知键，缺失键补默认值。

    修复历史缺陷：旧实现 ``dict(DEFAULT)`` 是浅拷贝，嵌套 dict 为共享
    引用，运行期修改会污染模块级默认值。
    """
    import copy

    result = copy.deepcopy(defaults)
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_defaults(value, result[key])
        else:
            result[key] = value
    return result
