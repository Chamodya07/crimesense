#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.utils_paths import get_project_root, get_rag_artifacts_dir, get_rag_source_dir


TEMP_FILE_SUFFIXES = {".tmp", ".temp", ".crdownload", ".part", ".partial"}


def _iter_files(path: Path) -> list[Path]:
    return [item for item in path.rglob("*") if item.is_file()]


def _is_empty_dir(path: Path) -> bool:
    try:
        next(path.iterdir())
    except StopIteration:
        return True
    return False


def _is_temp_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    name = path.name.lower()
    return suffix in TEMP_FILE_SUFFIXES or name.endswith(".tmp") or name.endswith(".partial")


def _prune_nested_paths(paths: list[Path]) -> list[Path]:
    selected: list[Path] = []
    for path in sorted({item.resolve() for item in paths}, key=lambda item: (len(item.parts), str(item))):
        if any(parent in selected for parent in path.parents):
            continue
        selected.append(path)
    return selected


def _scan_rag_source(root: Path) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    if not root.exists():
        return candidates

    all_dirs = [item for item in root.rglob("*") if item.is_dir()]
    all_dirs.sort(key=lambda item: len(item.parts), reverse=True)

    for path in all_dirs:
        files = _iter_files(path)
        if not files and _is_empty_dir(path):
            candidates.append((path, "empty RAG source directory"))
            continue

        has_csv = any(file.suffix.lower() == ".csv" for file in files)
        has_zip = any(file.suffix.lower() == ".zip" for file in files)
        if has_csv or has_zip:
            continue
        if files and all(_is_temp_file(file) for file in files):
            candidates.append((path, "temp-only/partial RAG source directory"))

    pruned = _prune_nested_paths([path for path, _ in candidates])
    reasons = {path.resolve(): reason for path, reason in candidates}
    return [(path, reasons[path.resolve()]) for path in pruned]


def _scan_rag_artifacts(root: Path) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    if not root.exists():
        return candidates

    for path in sorted([item for item in root.iterdir() if item.is_dir()]):
        file_names = {item.name.lower() for item in path.iterdir() if item.is_file()}
        has_index = "faiss.index" in file_names
        has_cases = "cases.csv" in file_names
        if has_index and has_cases:
            continue
        if not has_index and not has_cases:
            reason = "broken RAG artifact directory (missing faiss.index and cases.csv)"
        else:
            reason = "broken RAG artifact directory (missing one of faiss.index/cases.csv)"
        candidates.append((path, reason))
    return candidates


def _scan_scripts_empty_dirs(root: Path) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    if not root.exists():
        return candidates

    all_dirs = [item for item in root.rglob("*") if item.is_dir()]
    all_dirs.sort(key=lambda item: len(item.parts), reverse=True)
    for path in all_dirs:
        if _is_empty_dir(path):
            candidates.append((path, "empty scripts temp directory"))
    pruned = _prune_nested_paths([path for path, _ in candidates])
    reasons = {path.resolve(): reason for path, reason in candidates}
    return [(path, reasons[path.resolve()]) for path in pruned]


def _delete_path(path: Path) -> None:
    if path.is_dir():
        try:
            path.rmdir()
        except OSError:
            shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean empty/broken RAG source and artifact folders.")
    parser.add_argument("--apply", action="store_true", help="Actually delete the reported candidates.")
    args = parser.parse_args()

    rag_source_dir = get_rag_source_dir()
    rag_artifacts_dir = get_rag_artifacts_dir()
    scripts_dir = get_project_root() / "scripts"

    categories = {
        "data/rag_source": _scan_rag_source(rag_source_dir),
        "artifacts/rag": _scan_rag_artifacts(rag_artifacts_dir),
        "scripts": _scan_scripts_empty_dirs(scripts_dir),
    }

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"=== RAG CLEANUP ({mode}) ===")
    print(f"project_root: {get_project_root()}")

    total_candidates = 0
    total_deleted = 0
    for category, items in categories.items():
        print(f"\n[{category}]")
        if not items:
            print("  No candidates.")
            continue
        for path, reason in items:
            total_candidates += 1
            print(f"  DELETE {path}")
            print(f"    reason: {reason}")
            if args.apply:
                _delete_path(path)
                total_deleted += 1

    print("\n=== SUMMARY ===")
    print(f"mode: {mode}")
    print(f"candidate_count: {total_candidates}")
    print(f"deleted_count: {total_deleted}")
    for category, items in categories.items():
        print(f"{category}: {len(items)}")


if __name__ == "__main__":
    main()
