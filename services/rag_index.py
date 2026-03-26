from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Literal

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from services.utils_paths import get_project_root, get_rag_artifacts_dir


EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DatasetType = Literal["nypd", "la", "auto"]


def _resolve_to_root(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return get_project_root() / path


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for cand in candidates:
        hit = lowered.get(cand.strip().lower())
        if hit is not None:
            return str(hit)
    return None


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    return text


def _normalize_field(value: Any) -> str:
    text = _to_text(value)
    return text if text else "UNKNOWN"


def build_case_text(case_type: Any, weapon: Any, place: Any, area: Any) -> str:
    return (
        f"TYPE: {_normalize_field(case_type)} | "
        f"WEAPON: {_normalize_field(weapon)} | "
        f"PLACE: {_normalize_field(place)} | "
        f"AREA: {_normalize_field(area)}"
    )


def _load_dataframe(file_path: Path, max_rows: Optional[int]) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(file_path)
        if max_rows is not None:
            df = df.head(max_rows)
        return df

    # default to csv-style loading for common text datasets
    if max_rows is not None:
        return pd.read_csv(file_path, low_memory=False, nrows=max_rows)
    return pd.read_csv(file_path, low_memory=False)


def _print_available_columns(df: pd.DataFrame) -> None:
    print("Available columns in dataset:")
    for col in df.columns:
        print(f"- {col}")


def _nypd_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    type_col = _find_col(df, ["OFNS_DESC", "PD_DESC"])
    weapon_col = _find_col(df, ["WEAPON_DESC"])
    place_col = _find_col(df, ["PREM_TYP_DESC"])
    area_col = _find_col(df, ["BORO_NM"])
    return {
        "type_col": type_col,
        "weapon_col": weapon_col,
        "place_col": place_col,
        "area_col": area_col,
        "city_col": _find_col(df, ["CITY", "City", "city"]),
        "hour_col": _find_col(df, ["CMPLNT_FR_TM", "CMPLNT_TO_TM", "HOUR", "hour"]),
        "victim_age_col": _find_col(df, ["VIC_AGE_GROUP", "VIC_AGE"]),
        "victim_sex_col": _find_col(df, ["VIC_SEX"]),
        "victim_race_col": _find_col(df, ["VIC_RACE", "VIC_ETHNICITY"]),
        "suspect_age_col": _find_col(df, ["SUSP_AGE_GROUP", "SUSP_AGE"]),
        "suspect_sex_col": _find_col(df, ["SUSP_SEX"]),
        "suspect_race_col": _find_col(df, ["SUSP_RACE", "SUSP_ETHNICITY"]),
        "group_indicator_col": _find_col(
            df,
            ["GROUP_INVOLVEMENT", "GROUP_INVOLVED", "GANG_RELATED"],
        ),
        "prior_history_col": _find_col(
            df,
            ["PRIOR_HISTORY_PROXY", "PRIOR_HISTORY", "OFFENDER_HISTORY"],
        ),
        "law_category_col": _find_col(df, ["LAW_CAT_CD"]),
        "pd_description_col": _find_col(df, ["PD_DESC"]),
        "attempt_status_col": _find_col(df, ["CRM_ATPT_CPTD_CD"]),
        "status_desc_col": _find_col(df, ["STATUS_DESC", "Status Desc", "Status"]),
        "arrest_col": _find_col(df, ["ARREST", "ARRESTED", "ARREST_FLAG"]),
        "domestic_col": _find_col(df, ["DOMESTIC", "DV_FLAG", "IS_DOMESTIC"]),
    }


def _la_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    type_col = _find_col(df, ["Crm Cd Desc"])
    weapon_col = _find_col(df, ["Weapon Desc"])
    place_col = _find_col(df, ["Premis Desc"])
    area_col = _find_col(df, ["AREA NAME"])
    return {
        "type_col": type_col,
        "weapon_col": weapon_col,
        "place_col": place_col,
        "area_col": area_col,
        "city_col": _find_col(df, ["CITY", "City", "city"]),
        "hour_col": _find_col(df, ["TIME OCC", "HOUR", "hour"]),
        "victim_age_col": _find_col(df, ["Vict Age"]),
        "victim_sex_col": _find_col(df, ["Vict Sex"]),
        "victim_race_col": _find_col(df, ["Vict Descent"]),
        "suspect_age_col": _find_col(df, ["Suspect Age", "SUSP_AGE_GROUP"]),
        "suspect_sex_col": _find_col(df, ["Suspect Sex", "SUSP_SEX"]),
        "suspect_race_col": _find_col(df, ["Suspect Race", "SUSP_RACE"]),
        "group_indicator_col": _find_col(
            df,
            ["Group Involvement", "GROUP_INVOLVEMENT", "Part 1-2"],
        ),
        "prior_history_col": _find_col(
            df,
            ["PRIOR_HISTORY_PROXY", "PRIOR_HISTORY", "OFFENDER_HISTORY"],
        ),
        "law_category_col": _find_col(df, ["LAW_CAT_CD"]),
        "pd_description_col": _find_col(df, ["PD_DESC"]),
        "attempt_status_col": _find_col(df, ["CRM_ATPT_CPTD_CD"]),
        "status_desc_col": _find_col(df, ["Status Desc", "Status"]),
        "arrest_col": _find_col(df, ["ARREST", "ARRESTED", "ARREST_FLAG"]),
        "domestic_col": _find_col(df, ["DOMESTIC", "DV_FLAG", "IS_DOMESTIC"]),
    }


def _is_valid_mapping(mapping: Dict[str, Optional[str]]) -> bool:
    # Weapon is optional for some local datasets (e.g. NYPD historic export).
    return bool(mapping.get("type_col") and mapping.get("place_col"))


def _resolve_dataset_mapping(df: pd.DataFrame, dataset: DatasetType) -> Dict[str, Optional[str]]:
    requested = dataset.lower()
    if requested not in {"nypd", "la", "auto"}:
        raise ValueError(f"Unsupported dataset mode: {dataset}. Use one of: nypd, la, auto.")

    nypd = _nypd_mapping(df)
    la = _la_mapping(df)

    if requested == "nypd":
        if _is_valid_mapping(nypd):
            return {"dataset": "nypd", **nypd}
        _print_available_columns(df)
        raise ValueError(
            "Missing required NYPD columns. Required: "
            "OFNS_DESC (or PD_DESC), PREM_TYP_DESC. Optional: WEAPON_DESC."
        )

    if requested == "la":
        if _is_valid_mapping(la):
            return {"dataset": "la", **la}
        _print_available_columns(df)
        raise ValueError(
            "Missing required LA columns. Required: "
            "Crm Cd Desc, Premis Desc. Optional: Weapon Desc."
        )

    # auto mode
    if _is_valid_mapping(nypd):
        return {"dataset": "nypd", **nypd}
    if _is_valid_mapping(la):
        return {"dataset": "la", **la}

    _print_available_columns(df)
    raise ValueError(
        "Auto dataset detection failed. Dataset does not match required NYPD or LA columns."
    )


def _extract_hour(value: Any, dataset_name: str) -> Optional[int]:
    text = _to_text(value)
    if not text:
        return None

    cleaned = text
    if ":" in text:
        cleaned = text.split(":", 1)[0]
    cleaned = cleaned.strip()

    try:
        numeric = int(float(cleaned))
    except (TypeError, ValueError):
        return None

    if 0 <= numeric <= 23:
        return numeric

    if dataset_name == "la" and 0 <= numeric <= 2359:
        hour = numeric // 100
        if 0 <= hour <= 23:
            return hour

    return None


def _series_from_column(df: pd.DataFrame, column_name: Optional[str]) -> pd.Series:
    if column_name and column_name in df.columns:
        return df[column_name]
    return pd.Series([""] * len(df), index=df.index)


def _set_if_present(
    out_df: pd.DataFrame,
    src_df: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
    mapping_key: str,
    out_key: str,
) -> None:
    column = mapping.get(mapping_key)
    if column and column in src_df.columns:
        out_df[out_key] = src_df[column]


def build_rag_index_from_file(
    file_path: str | Path,
    out_dir: str | Path | None = None,
    max_rows: Optional[int] = 100000,
    dataset: DatasetType = "auto",
) -> Dict[str, Any]:
    input_path = _resolve_to_root(file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"RAG source file not found: {input_path}")

    output_dir = get_rag_artifacts_dir() if out_dir is None else _resolve_to_root(out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dataframe(input_path, max_rows=max_rows)
    if df.empty:
        raise ValueError(f"No rows available in source file: {input_path}")

    mapping = _resolve_dataset_mapping(df, dataset=dataset)
    type_col = mapping["type_col"]
    weapon_col = mapping["weapon_col"]
    place_col = mapping["place_col"]
    area_col = mapping["area_col"]
    print(f"resolved_dataset: {mapping.get('dataset', dataset)}")
    print(
        "selected_columns: "
        f"type_col={type_col}, "
        f"weapon_col={weapon_col}, "
        f"place_col={place_col}, "
        f"area_col={area_col}"
    )
    print(f"rows_loaded_after_max_rows: {len(df)}")

    case_id_col = _find_col(df, ["case_id", "id", "CMPLNT_NUM", "DR_NO"])
    if case_id_col:
        case_id_series = df[case_id_col].copy()
        case_id_series = case_id_series.fillna("")
        case_id_series = case_id_series.astype(str)
        missing_mask = case_id_series.str.strip() == ""
        if missing_mask.any():
            case_id_series.loc[missing_mask] = df.index[missing_mask].astype(str)
    else:
        case_id_series = df.index.astype(str)

    type_series = _series_from_column(df, type_col)
    weapon_series = _series_from_column(df, weapon_col)
    place_series = _series_from_column(df, place_col)
    area_series = _series_from_column(df, area_col)

    cases_df = pd.DataFrame(
        {
            "case_id": case_id_series,
            "type": type_series,
            "weapon": weapon_series,
            "place": place_series,
            "area": area_series,
        }
    )

    # Optional fields used by profiling evidence; added only when available.
    _set_if_present(cases_df, df, mapping, "city_col", "city")
    _set_if_present(cases_df, df, mapping, "victim_age_col", "victim_age")
    _set_if_present(cases_df, df, mapping, "victim_sex_col", "victim_sex")
    _set_if_present(cases_df, df, mapping, "victim_race_col", "victim_race")
    _set_if_present(cases_df, df, mapping, "suspect_age_col", "suspect_age")
    _set_if_present(cases_df, df, mapping, "suspect_sex_col", "suspect_sex")
    _set_if_present(cases_df, df, mapping, "suspect_race_col", "suspect_race")
    _set_if_present(cases_df, df, mapping, "group_indicator_col", "group_indicator")
    _set_if_present(cases_df, df, mapping, "prior_history_col", "prior_history")
    _set_if_present(cases_df, df, mapping, "law_category_col", "law_category")
    _set_if_present(cases_df, df, mapping, "pd_description_col", "pd_description")
    _set_if_present(cases_df, df, mapping, "attempt_status_col", "attempt_status")
    _set_if_present(cases_df, df, mapping, "status_desc_col", "status_desc")
    _set_if_present(cases_df, df, mapping, "arrest_col", "arrest")
    _set_if_present(cases_df, df, mapping, "domestic_col", "domestic")

    hour_col = mapping.get("hour_col")
    if hour_col and hour_col in df.columns:
        dataset_name = str(mapping.get("dataset") or dataset)
        hour_values = df[hour_col].apply(lambda value: _extract_hour(value, dataset_name))
        cases_df["hour"] = hour_values.apply(lambda value: "" if value is None else int(value))
        cases_df["is_night"] = hour_values.apply(
            lambda value: "" if value is None else bool(value >= 18 or value <= 5)
        )

    cases_df = cases_df.fillna("")
    cases_df["case_text"] = cases_df.apply(
        lambda row: build_case_text(
            row.get("type"),
            row.get("weapon"),
            row.get("place"),
            row.get("area"),
        ),
        axis=1,
    )

    texts = cases_df["case_text"].astype(str).tolist()
    try:
        model = SentenceTransformer(EMBED_MODEL_NAME)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load embedding model '{EMBED_MODEL_NAME}'. "
            "Ensure internet access on first run so the model can be downloaded."
        ) from exc
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings)

    index_path = output_dir / "faiss.index"
    cases_path = output_dir / "cases.csv"
    faiss.write_index(index, str(index_path))
    cases_df.to_csv(cases_path, index=False)

    return {
        "source_file": str(input_path),
        "out_dir": str(output_dir),
        "dataset": mapping.get("dataset", dataset),
        "selected_columns": {k: v for k, v in mapping.items() if k.endswith("_col")},
        "rows_indexed": int(len(cases_df)),
        "faiss_index": str(index_path),
        "cases_csv": str(cases_path),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_rag_index(
    input_path: str | Path,
    out_dir: str | Path | None = None,
    max_rows: Optional[int] = 100000,
    dataset: DatasetType = "auto",
) -> Dict[str, Any]:
    # Backward-compatible alias.
    return build_rag_index_from_file(input_path, out_dir=out_dir, max_rows=max_rows, dataset=dataset)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build RAG index from a local CSV/Parquet file.")
    parser.add_argument("--input", required=True, help="Path to CSV/Parquet source file")
    parser.add_argument("--out", default="artifacts/rag", help="Output directory for artifacts")
    parser.add_argument("--max_rows", type=int, default=100000, help="Maximum rows to index")
    parser.add_argument("--dataset", choices=["nypd", "la", "auto"], default="auto", help="Dataset schema mode")
    cli_args = parser.parse_args()

    result = build_rag_index_from_file(
        file_path=cli_args.input,
        out_dir=cli_args.out,
        max_rows=cli_args.max_rows,
        dataset=cli_args.dataset,
    )
    print(f"Indexed {result['rows_indexed']} rows")
    print(f"Dataset mode: {result['dataset']}")
    print(f"Index: {result['faiss_index']}")
    print(f"Cases: {result['cases_csv']}")


__all__ = ["build_case_text", "build_rag_index", "build_rag_index_from_file", "EMBED_MODEL_NAME"]
