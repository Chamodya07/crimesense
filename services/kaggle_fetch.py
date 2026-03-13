from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from services.utils_paths import get_project_root, get_rag_source_dir


def find_kaggle_json() -> Tuple[Optional[Path], List[Path]]:
    root = get_project_root()
    preferred = root / "kaggle.json"
    searched: List[Path] = [preferred]
    if preferred.exists():
        return preferred, searched

    for path in root.rglob("kaggle.json"):
        if path.is_file():
            return path, searched
    return None, searched


def _run_kaggle_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=True)


def ensure_kaggle_config() -> Tuple[bool, str]:
    src, searched = find_kaggle_json()
    if src is None:
        return False, f"kaggle.json not found under project root. Searched: {searched}"

    dest_dir = Path.home() / ".kaggle"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "kaggle.json"
    try:
        shutil.copy2(src, dest)
        if os.name != "nt":
            dest.chmod(0o600)
    except Exception as exc:  # noqa: BLE001
        return False, f"Failed to configure Kaggle credentials: {exc}"

    try:
        proc = _run_kaggle_cmd(["kaggle", "--version"])
    except FileNotFoundError:
        return False, "Kaggle CLI not installed or not available on PATH."
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        return False, f"Kaggle CLI failed.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

    return True, proc.stdout.strip()


def validate_dataset_downloadable(slug: str) -> Tuple[bool, str]:
    try:
        _run_kaggle_cmd(["kaggle", "datasets", "metadata", "-d", slug, "-p", "."])
        return True, "Dataset is accessible."
    except FileNotFoundError:
        return False, "Kaggle CLI not installed or not available on PATH."
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        return False, f"Dataset validation failed.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"


def download_kaggle_dataset(slug: str, out_dir: str | Path | None = None) -> List[str]:
    sanitized = slug.replace("/", "_")
    base = Path(out_dir) if out_dir else get_rag_source_dir()
    if not base.is_absolute():
        base = get_project_root() / base
    dest = base / sanitized
    dest.mkdir(parents=True, exist_ok=True)

    cmd = ["kaggle", "datasets", "download", "-d", slug, "--unzip", "-p", str(dest)]
    try:
        _run_kaggle_cmd(cmd)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Kaggle CLI not installed or not available on PATH.\n"
            f"Command: {' '.join(cmd)}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(
            f"Kaggle download failed for slug '{slug}'.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{stderr}"
        ) from exc

    files = [str(p) for p in dest.iterdir() if p.is_file()]
    return files


__all__ = [
    "find_kaggle_json",
    "ensure_kaggle_config",
    "validate_dataset_downloadable",
    "download_kaggle_dataset",
]
