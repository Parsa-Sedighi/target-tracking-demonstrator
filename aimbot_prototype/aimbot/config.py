"""YAML configuration with ``extends:`` inheritance and ``--set a.b=c`` overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PACKAGE_ROOT / "config"


class ConfigNode(dict):
    """dict with attribute access: ``cfg.camera.backend``. Missing keys raise AttributeError."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: Any = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = ConfigNode()
            node = node[part]
        node[parts[-1]] = _wrap(value)

    def to_dict(self) -> dict:
        return _unwrap(self)


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict) and not isinstance(obj, ConfigNode):
        return ConfigNode({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, ConfigNode):
        return ConfigNode({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _unwrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _unwrap(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unwrap(v) for v in obj]
    return obj


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive merge. A dict containing ``_replace: true`` replaces the base dict instead."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and v.get("_replace"):
            out[k] = {kk: copy.deepcopy(vv) for kk, vv in v.items() if kk != "_replace"}
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_config_path(path: str | os.PathLike) -> Path:
    p = Path(path)
    if p.exists():
        return p.resolve()
    for candidate in (CONFIG_DIR / p, CONFIG_DIR / f"{p}.yaml", CONFIG_DIR / p.name):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Config file not found: {path}")


def _load_with_extends(path: Path, depth: int = 0) -> dict:
    if depth > 8:
        raise RuntimeError("config 'extends' chain too deep")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        if not parent_path.exists():
            parent_path = resolve_config_path(parent)
        data = deep_merge(_load_with_extends(parent_path, depth + 1), data)
    return data


def load_config(path: str | os.PathLike = "default.yaml", overrides: Iterable[str] = ()) -> ConfigNode:
    cfg_path = resolve_config_path(path)
    data = _load_with_extends(cfg_path)
    cfg = _wrap(data)
    for item in overrides or ():
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        cfg.set_path(key.strip(), yaml.safe_load(raw))
    cfg["_source"] = str(cfg_path)
    return cfg
