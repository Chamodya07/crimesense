from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Return repository root from a file under ``services/``."""
    return Path(__file__).resolve().parents[1]


def get_rag_source_dir() -> Path:
    return get_project_root() / "data" / "rag_source"


def get_rag_artifacts_dir() -> Path:
    return get_project_root() / "artifacts" / "rag"


__all__ = ["get_project_root", "get_rag_source_dir", "get_rag_artifacts_dir"]
