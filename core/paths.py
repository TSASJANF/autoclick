"""路径治理模块 —— 所有持久化路径的唯一事实来源。

历史问题：项目曾把 data_dir、template_path 等以相对路径存入 targets.json，
解析时依赖进程当前工作目录（CWD）；模型路径以绝对路径存储，项目目录一移动
即全部失效。本模块统一规则：

- 存储层永远存「相对 APP_ROOT 的路径」（POSIX 风格），跨目录可移植；
- 运行时通过 resolve() 得到绝对路径，与 CWD 无关；
- 加载历史数据时对失效的绝对路径执行一次「重定位修复」（re-root）。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# 应用根目录 = core/ 的上一级，与启动位置无关
APP_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = APP_ROOT / "data"
SETS_DIR = DATA_DIR / "sets"
RUNS_DIR = APP_ROOT / "runs"

# 已知的持久化文件
TARGETS_FILE = APP_ROOT / "targets.json"
CONFIG_FILE = APP_ROOT / "config.json"
STATE_FILE = APP_ROOT / "app_state.json"
SETS_INDEX_FILE = SETS_DIR / "sets_index.json"

# 绝对路径重定位修复时，从这些目录名开始截取尾部再拼回 APP_ROOT
_RE_ROOT_MARKERS = ("runs", "data")


def resolve(path_str: str | os.PathLike | None) -> Path:
    """把存储层路径解析为基于 APP_ROOT 的绝对路径。

    绝对路径按原样返回（供修复逻辑进一步处理）；相对路径一律相对
    APP_ROOT，而不是 CWD。
    """
    if not path_str:
        return Path()
    p = Path(str(path_str))
    if p.is_absolute():
        return p
    return (APP_ROOT / p).resolve()


def to_stored(path: str | os.PathLike | None) -> str:
    """把运行时路径转为适合写入 JSON 的存储格式。

    能相对化的存相对 POSIX 路径；不能（例如用户选择的外部目录）则存绝对。
    """
    if not path:
        return ""
    p = Path(str(path))
    if not p.is_absolute():
        p = resolve(p)
    try:
        return p.relative_to(APP_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def repair(path_str: str | None) -> str:
    """修复因项目搬迁而失效的绝对路径，返回存储格式的路径。

    典型场景：targets.json 中 model_path 指向旧机器/旧目录的绝对路径
    （如 D:\\脚本\\...\\runs\\detect\\...）。若该路径不存在，则尝试从
    runs/ 或 data/ 标记处截取尾部，拼到当前 APP_ROOT 下；命中即采用。
    未命中时保持原样（调用方按文件缺失处理）。
    """
    if not path_str:
        return ""
    p = Path(str(path_str))
    if p.exists():
        return to_stored(p)
    if p.is_absolute():
        normalized = p.as_posix()
        for marker in _RE_ROOT_MARKERS:
            idx = normalized.find(f"/{marker}/")
            if idx != -1:
                candidate = APP_ROOT / normalized[idx + 1:]
                if candidate.exists():
                    logger.info("路径修复: %s -> %s", p, candidate)
                    return to_stored(candidate)
    return str(path_str)
