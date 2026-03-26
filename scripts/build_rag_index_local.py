#!/usr/bin/env python3
from __future__ import annotations

import argparse
import traceback
import sys
from pathlib import Path

# Allow imports from project root when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.utils_paths import get_project_root


def _resolve_to_root(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return get_project_root() / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local RAG index from CSV/Parquet.")
    parser.add_argument("--input", required=True, help="Local dataset path (CSV or Parquet).")
    parser.add_argument("--out", default="artifacts/rag", help="Output directory for RAG artifacts.")
    parser.add_argument("--max_rows", type=int, default=50000, help="Maximum rows to index.")
    parser.add_argument(
        "--dataset",
        choices=["nypd", "la", "auto"],
        default="auto",
        help="Dataset schema mode.",
    )
    args = parser.parse_args()

    project_root = get_project_root()
    input_path = _resolve_to_root(args.input)
    out_dir = _resolve_to_root(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== RAG BUILD (LOCAL) ===")
    print(f"project_root: {project_root}")
    print(f"input_path: {input_path}")
    print(f"input_exists: {input_path.exists()}")
    print(f"output_dir: {out_dir}")
    print(f"dataset_mode: {args.dataset}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    try:
        from services.rag_index import build_rag_index_from_file
    except ModuleNotFoundError as exc:
        print("Dependency/import error while loading RAG builder.")
        print(exc)
        print("Install dependencies: pip install sentence-transformers faiss-cpu")
        sys.exit(1)

    try:
        result = build_rag_index_from_file(
            file_path=input_path,
            out_dir=out_dir,
            max_rows=args.max_rows,
            dataset=args.dataset,
        )
    except ModuleNotFoundError as exc:
        print("Missing dependency during RAG build.")
        print(exc)
        print("Install dependencies: pip install sentence-transformers faiss-cpu")
        traceback.print_exc()
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print("RAG build failed with exception:")
        print(exc)
        traceback.print_exc()
        sys.exit(1)

    index_path = out_dir / "faiss.index"
    cases_path = out_dir / "cases.csv"
    assert index_path.exists(), f"faiss.index missing at {index_path}"
    assert cases_path.exists(), f"cases.csv missing at {cases_path}"

    selected_columns = result.get("selected_columns", {})
    print(f"detected_dataset: {result.get('dataset')}")
    if selected_columns:
        print("selected_columns:")
        for key in sorted(selected_columns.keys()):
            print(f"  {key}={selected_columns.get(key)}")
    else:
        print("selected_columns: (none detected)")
    print(f"rows_loaded_after_max_rows: {result.get('rows_indexed')}")
    print(f"Source file: {result['source_file']}")
    print(f"faiss.index exists: {index_path.exists()}")
    print(f"cases.csv exists: {cases_path.exists()}")
    print(f"final_faiss_index: {index_path}")
    print(f"final_cases_csv: {cases_path}")
    print("SUCCESS")
    print(f"{index_path}")
    print(f"{cases_path}")


if __name__ == "__main__":
    main()
