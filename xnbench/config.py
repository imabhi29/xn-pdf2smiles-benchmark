"""Configuration loading.

Settings live in a YAML file. Every relative path is resolved against the
repository root, so nothing depends on the directory the user runs from.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


class Config(dict):
    """A dict that also allows attribute access: cfg.pdf.dpi."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value


def resolve_path(path: str | Path) -> Path:
    """Absolute paths are kept; relative ones are anchored at the repo root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: str | Path) -> Config:
    with open(resolve_path(path), encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def pick_device(requested: str) -> str:
    """Return 'cuda' when requested or available (for 'auto'), else 'cpu'."""
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
