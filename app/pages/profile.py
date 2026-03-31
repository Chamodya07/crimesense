from __future__ import annotations

import json
import sys
import datetime as dt
import hashlib
from pathlib import Path

import streamlit as st
from config import MODELS

# Ensure project root is on path for backend imports
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.auth_service import render_auth_status

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
CSS_FILE = ASSETS_DIR / "styles.css"
EM_DASH = "\u2014"
PROFILE_TRAITS = ("risk_level", "crime_severity", "offender_experience")


def make_json_safe(obj):
    """Recursively convert common non-JSON types to JSON-safe values."""
    try:
        import numpy as np  # type: ignore
    except Exception:  # noqa: BLE001
        np = None  # type: ignore

    try:
        import pandas as pd  # type: ignore
    except Exception:  # noqa: BLE001
        pd = None  # type: ignore

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, (dt.date, dt.datetime, dt.time)):
        return obj.isoformat()

    if np is not None:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)

    if pd is not None:
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return make_json_safe(obj.to_dict())
        if isinstance(obj, pd.DataFrame):
            return make_json_safe(obj.to_dict(orient="records"))

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]

    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            item_value = obj.item()
            if item_value is not obj:
                return make_json_safe(item_value)
        except Exception:  # noqa: BLE001
            pass

    return str(obj)


def inject_styles() -> None:
    if CSS_FILE.exists():
        st.markdown(f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _read_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _artifact_mtime_text(path: Path) -> str | None:
    try:
        if not path.exists():
            return None
        return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc).replace(microsecond=0).isoformat()
    except Exception:
        return None


def _resolve_tabular_data_version() -> str | None:
    candidates = [
        _ROOT / "artifacts" / "tabular" / "portable" / "model_meta.json",
        _ROOT / "artifacts" / "tabular" / "crimesense_tabular_only" / "model_meta.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        meta = _read_json_file(path)
        version = (
            meta.get("data_version")
            or meta.get("dataset_version")
            or meta.get("last_updated")
            or _artifact_mtime_text(path)
        )
        if version not in (None, ""):
            return str(version)
    return None


def _resolve_rag_data_version() -> str | None:
    meta_path = _ROOT / "artifacts" / "rag" / "meta.json"
    if meta_path.exists():
        meta = _read_json_file(meta_path)
        version = meta.get("built_at") or _artifact_mtime_text(meta_path)
        if version not in (None, ""):
            return str(version)

    for fallback in (
        _ROOT / "artifacts" / "rag" / "faiss.index",
        _ROOT / "artifacts" / "rag" / "cases.csv",
    ):
        version = _artifact_mtime_text(fallback)
        if version not in (None, ""):
            return str(version)
    return None


def _audit_model_version(fused: dict | None) -> str | None:
    if isinstance(fused, dict):
        value = fused.get("model_version") or (fused.get("fusion_meta") or {}).get("version")
        if value not in (None, ""):
            return str(value)
    fallback = getattr(MODELS, "MODEL_VERSION", None) or getattr(MODELS, "fusion_version", None)
    return str(fallback) if fallback not in (None, "") else None


def _audit_data_version(fused: dict | None) -> str | None:
    tab_ver = _resolve_tabular_data_version()
    rag_ver = _resolve_rag_data_version()

    parts = []
    if tab_ver:
        parts.append(f"tab:{tab_ver}")
    if rag_ver:
        parts.append(f"rag:{rag_ver}")

    if parts:
        return " | ".join(parts)

    if isinstance(fused, dict):
        fallback = (
            fused.get("data_version")
            or fused.get("rag_index_version")
            or (fused.get("explanations") or {}).get("data_version")
        )
        if fallback not in (None, ""):
            return str(fallback)
    return None


def _safe_float(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _resolve_city_cases_csv(city: str) -> Path:
    normalized_city = str(city or "").strip().upper()
    if "NYPD" in normalized_city:
        return _ROOT / "artifacts" / "rag" / "nypd" / "cases.csv"
    if "LA" in normalized_city:
        return _ROOT / "artifacts" / "rag" / "la" / "cases.csv"
    return _ROOT / "artifacts" / "rag" / "cases.csv"


@st.cache_data(show_spinner=False)
def load_primary_types(cases_csv: str) -> list[str]:
    values = load_unique_values(cases_csv, "type")
    display_to_original: dict[str, str] = {}
    for raw_text in values:
        display_text = raw_text.title()
        display_to_original.setdefault(display_text, raw_text)
    return sorted(display_to_original.keys())


@st.cache_data(show_spinner=False)
def load_unique_values(cases_csv: str, col: str) -> list[str]:
    csv_path = Path(cases_csv)
    if not csv_path.exists():
        return []

    try:
        import pandas as pd  # type: ignore

        values_df = pd.read_csv(csv_path, usecols=[col])
    except Exception:
        return []

    values: set[str] = set()
    for raw_value in values_df[col].dropna().tolist():
        text = str(raw_value).strip()
        if text:
            values.add(text)
    return sorted(values)


def load_places(cases_csv: str) -> list[str]:
    return load_unique_values(cases_csv, "place") or ["Unknown"]


@st.cache_data(show_spinner=False)
def load_la_weapons() -> list[str]:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return ["Unknown"]

    la_cases_csv = _ROOT / "artifacts" / "rag" / "la" / "cases.csv"
    if not la_cases_csv.exists():
        return ["Unknown"]

    try:
        la_df = pd.read_csv(la_cases_csv, usecols=["weapon"])
    except Exception:
        return ["Unknown"]

    invalid_values = {"", "UNKNOWN", "NONE", "NAN", "NULL", "N/A", "NA"}
    weapon_values: set[str] = set()
    for raw_value in la_df["weapon"].dropna().tolist():
        text = str(raw_value).strip()
        if not text or text.upper() in invalid_values:
            continue
        weapon_values.add(text)

    ordered_values = sorted(weapon_values)
    return ["Unknown", *ordered_values]


def _is_missing_evidence_value(value) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    return text.lower() in {"nan", "none", "unknown", "null"}


def _sanitize_evidence_display_df(df):
    display_df = df.copy()
    for col in display_df.columns:
        if col == "score":
            display_df[col] = display_df[col].apply(
                lambda value: round(float(value), 3) if _safe_float(value) is not None else value
            )
            continue
        display_df[col] = display_df[col].apply(lambda value: "" if _is_missing_evidence_value(value) else value)
    return display_df


def _should_show_evidence_column(df, col: str) -> bool:
    if col not in df.columns or df.empty:
        return False
    missing_ratio = df[col].apply(_is_missing_evidence_value).mean()
    return float(missing_ratio) <= 0.8


def _most_common_evidence_value(df, col: str) -> str:
    if col not in df.columns or df.empty:
        return ""
    series = df[col].apply(lambda value: "" if _is_missing_evidence_value(value) else str(value).strip())
    series = series[series != ""]
    if series.empty:
        return ""
    mode_values = series.mode()
    if mode_values.empty:
        return ""
    return str(mode_values.iloc[0]).strip()


def _rag_evidence_strength_band(top_score) -> str:
    score = _safe_float(top_score)
    if score is None:
        return "Low"
    if score >= 0.75:
        return "Strong"
    if score >= 0.6:
        return "Medium"
    return "Low"


def _evidence_distribution_text(df, col: str, limit: int) -> str:
    if col not in df.columns or df.empty:
        return ""
    series = df[col].apply(lambda value: "" if _is_missing_evidence_value(value) else str(value).strip())
    series = series[series != ""]
    if series.empty:
        return ""
    counts = series.value_counts().head(limit)
    return ", ".join(f"{label} ({count})" for label, count in counts.items())


def majority_value(
    series,
    min_n: int = 5,
    min_ratio: float = 0.60,
    unknown_set: set[str] | None = None,
):
    normalized_unknowns = {value.upper() for value in (unknown_set or {"", "UNKNOWN", "N/A", "NONE", "D"})}
    filtered_values = []
    for raw_value in series:
        if raw_value is None:
            continue
        text = str(raw_value).strip()
        if not text or text.upper() in normalized_unknowns:
            continue
        filtered_values.append(text)

    sample_size = len(filtered_values)
    if sample_size < min_n:
        return None, 0.0

    counts: dict[str, int] = {}
    for value in filtered_values:
        counts[value] = counts.get(value, 0) + 1

    top_value, top_count = max(counts.items(), key=lambda item: item[1])
    ratio = top_count / sample_size
    if ratio < min_ratio:
        return None, 0.0
    return top_value, ratio


def _prepare_rag_results_df(df, dataset_id: str | None):
    required_cols = [
        "score",
        "case_id",
        "type",
        "place",
        "area",
        "victim_age",
        "victim_sex",
        "victim_race",
        "suspect_age",
        "suspect_sex",
        "suspect_race",
    ]
    if str(dataset_id or "").strip().lower() == "nypd":
        required_cols = [
            "score",
            "case_id",
            "type",
            "place",
            "area",
            "law_category",
            "attempt_status",
            "victim_age",
            "victim_sex",
            "victim_race",
            "suspect_age",
            "suspect_sex",
            "suspect_race",
        ]
    display_df = df.copy()
    for col in required_cols:
        if col not in display_df.columns:
            display_df[col] = ""
    return display_df[required_cols]


def _build_saved_display_columns(df):
    preferred_saved_cols = [
        "score",
        "saved_time",
        "record_id",
        "type",
        "weapon",
        "place",
        "city",
        "area",
        "hour",
        "is_night",
        "suspect_age",
        "suspect_sex",
        "suspect_race",
        "group_indicator",
        "prior_history",
        "arrest",
        "domestic",
        "risk_level",
        "motive",
        "motive_confidence",
        "motive_band",
        "helpfulness",
        "feedback_text",
        "victim_age",
        "victim_gender",
        "victim_race",
    ]
    saved_cols = []
    for col in preferred_saved_cols:
        if col in df.columns:
            non_empty = df[col].apply(lambda x: str(x).strip() not in {"", "nan", "None"}).any()
            if non_empty:
                saved_cols.append(col)
    for col in df.columns:
        if col == "_raw" or col in saved_cols:
            continue
        non_empty = df[col].apply(lambda x: str(x).strip() not in {"", "nan", "None"}).any()
        if non_empty:
            saved_cols.append(col)
    return saved_cols


def _build_rag_summary(df) -> dict[str, object]:
    return _build_rag_summary_for_dataset(df, dataset_id=None)


def _resolve_rag_dataset_id(rag_meta, rag_df) -> str:
    if isinstance(rag_meta, dict):
        for key in ("dataset_id", "resolved_dataset"):
            value = str(rag_meta.get(key) or "").strip().lower()
            if value:
                return value

    meta_path = _ROOT / "artifacts" / "rag" / "meta.json"
    if meta_path.exists():
        try:
            file_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            file_meta = {}
        if isinstance(file_meta, dict):
            for key in ("dataset_id", "resolved_dataset"):
                value = str(file_meta.get(key) or "").strip().lower()
                if value:
                    return value

    if "law_category" in rag_df.columns or "attempt_status" in rag_df.columns:
        return "nypd"
    return "la"


def _build_rag_summary_for_dataset(df, dataset_id: str | None) -> dict[str, object]:
    summary: dict[str, object] = {}
    top_score = None
    if "score" in df.columns and not df.empty:
        top_score = df["score"].apply(_safe_float).dropna().max()
    summary["evidence_strength"] = _rag_evidence_strength_band(top_score)

    def most_common_or_none(series):
        cleaned = (
            series.astype(str)
            .replace({"None": "", "nan": "", "NaN": "", "": ""})
            .str.strip()
        )
        cleaned = cleaned[cleaned != ""]
        return cleaned.value_counts().idxmax() if len(cleaned) else None

    if str(dataset_id or "").strip().lower() == "nypd":
        if "law_category" in df.columns:
            most_law = most_common_or_none(df["law_category"])
            if most_law is not None:
                summary["most_common_law_category"] = most_law
        if "attempt_status" in df.columns:
            most_attempt = most_common_or_none(df["attempt_status"])
            if most_attempt is not None:
                summary["most_common_attempt_status"] = most_attempt

    for key, col in (
        ("most_common_type", "type"),
        ("most_common_place", "place"),
        ("most_common_area", "area"),
        ("most_common_weapon", "weapon"),
        ("most_common_victim_age", "victim_age"),
        ("most_common_victim_sex", "victim_sex"),
        ("most_common_victim_race", "victim_race"),
        ("most_common_suspect_age", "suspect_age"),
        ("most_common_suspect_sex", "suspect_sex"),
        ("most_common_suspect_race", "suspect_race"),
    ):
        if col not in df.columns:
            continue
        value, _ = majority_value(df[col])
        if value:
            summary[key] = value

    return summary


def _rag_summary_lines(summary: dict[str, object]) -> list[str]:
    lines = []
    mapping = [
        ("evidence_strength", "Evidence strength"),
        ("most_common_type", "Most common type"),
        ("most_common_place", "Most common place"),
        ("most_common_area", "Most common area"),
        ("most_common_law_category", "Most common law category"),
        ("most_common_attempt_status", "Most common attempt status"),
        ("most_common_weapon", "Most common weapon"),
        ("most_common_victim_age", "Most common victim age"),
        ("most_common_victim_sex", "Most common victim sex"),
        ("most_common_victim_race", "Most common victim race"),
        ("most_common_suspect_age", "Most common suspect age"),
        ("most_common_suspect_sex", "Most common suspect sex"),
        ("most_common_suspect_race", "Most common suspect race"),
    ]
    for key, label in mapping:
        value = summary.get(key)
        if value in (None, ""):
            continue
        lines.append(f"{label}: {value}")
    return lines

def _band_from_confidence(confidence: float | None) -> str | None:
    if confidence is None:
        return None
    if confidence >= 0.75:
        return "High"
    if confidence >= 0.55:
        return "Medium"
    return "Low"


def _max_probability(probabilities):
    if isinstance(probabilities, dict):
        values = [_safe_float(value) for value in probabilities.values()]
    elif isinstance(probabilities, (list, tuple)):
        values = [_safe_float(value) for value in probabilities]
    else:
        return None

    normalized = [value for value in values if value is not None]
    return max(normalized) if normalized else None


def _trait_probability_from_source(source: dict | None, trait: str):
    if not isinstance(source, dict):
        return None

    candidate = source.get(trait)
    if isinstance(candidate, dict):
        return _max_probability(candidate)
    return None


def _extract_trait_confidence(source: dict | None, trait: str):
    if not isinstance(source, dict):
        return None, None

    direct = source.get(trait)
    if isinstance(direct, dict):
        conf = _safe_float(direct.get("conf") or direct.get("confidence"))
        band = str(direct.get("band") or "").strip() or None
        if conf is not None or band:
            return conf, band

    confidence_maps = [source.get("confidence"), source.get("confidences")]
    band_maps = [source.get("band"), source.get("bands")]
    for conf_map in confidence_maps:
        if isinstance(conf_map, dict) and trait in conf_map:
            conf = _safe_float(conf_map.get(trait))
            band = None
            for band_map in band_maps:
                if isinstance(band_map, dict) and trait in band_map:
                    band = str(band_map.get(trait) or "").strip() or None
                    break
            return conf, band

    for probs_key in ("tabular_probs", "proba", "pred_proba", "class_probs"):
        top_prob = _trait_probability_from_source(source.get(probs_key), trait)
        if top_prob is not None:
            return top_prob, _band_from_confidence(top_prob)

    return None, None


def _format_trait_confidence_display(fused: dict, trait: str) -> str:
    sources = []

    final = fused.get("final")
    if isinstance(final, dict):
        sources.append(final)

    explanations = fused.get("explanations")
    if isinstance(explanations, dict):
        sources.append(explanations)

    tabular_out = fused.get("_tabular_output")
    if isinstance(tabular_out, dict):
        sources.append(tabular_out)
        tabular_explanations = tabular_out.get("explanations")
        if isinstance(tabular_explanations, dict):
            sources.append(tabular_explanations)

    for source in sources:
        conf, band = _extract_trait_confidence(source, trait)
        if conf is None:
            continue
        band_text = band or _band_from_confidence(conf)
        return f"{conf:.2f} ({band_text})" if band_text else f"{conf:.2f}"

    return EM_DASH


def _log_profile_event(action: str, case_id: str | None, fused: dict | None, meta: dict | None = None) -> None:
    try:
        from services.audit_service import log_event

        log_event(
            action=action,
            case_id=case_id,
            model_ver=_audit_model_version(fused),
            data_ver=_audit_data_version(fused),
            page="profile",
            meta=meta or {},
        )
    except Exception:
        pass


def main() -> None:
    st.set_page_config(
        page_title="Crime Sense | Profile Intake",
        page_icon="ðŸ“",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()
    render_auth_status("profile")

    st.title("Case Profiling")
    st.markdown("Provide structured details about the incident below.")

    # load tabular feature list for later mapping
    # if the model artefacts are missing we want to inform the user
    # explicitly instead of silently falling back to an empty feature set.
    model_missing = False
    try:
        from services.tabular_service import load_tabular_model, ModelNotAvailableError

        _meta = load_tabular_model()
        _features: list = _meta.get("features", []) or []
    except ModelNotAvailableError as err:  # clear message about missing files
        st.error(str(err))
        _features = []
        model_missing = True
    except Exception:
        # generic failure â€“ treat as no feature list, prediction will still
        # display an error later when the pipeline is invoked.
        _features = []
        model_missing = True

    # form inputs are named exactly to match the model's feature keys. we
    # retain a small translation dict only for backwards compatibility, but
    # it should normally remain empty.
    UI_TO_MODEL: dict = {}
    CITY_PLACEHOLDER = "Select city"
    PRIMARY_TYPE_PLACEHOLDER = "Select primary type"
    LOCATION_PLACEHOLDER = "Select location description"
    WEAPON_PLACEHOLDER = "Select weapon description"

    def _with_placeholder(options: list[str], placeholder: str) -> list[str]:
        values = [str(option) for option in options]
        if placeholder in values:
            return values
        return [placeholder, *values]

    # form data collection
    form_data: dict = {}
    # Incident Details
    with st.expander("Incident Details", expanded=True):
        city_key = "profile_city"
        city_options = [CITY_PLACEHOLDER, "NYPD (New York)", "LA (Los Angeles)"]
        if st.session_state.get(city_key) not in city_options:
            st.session_state[city_key] = CITY_PLACEHOLDER
        form_data["city"] = st.selectbox(
            "City",
            options=city_options,
            key=city_key,
        )
        # primary_type selectbox with ability to specify custom
        fallback_crime_options = ["", "Theft", "Assault", "Robbery", "Burglary", "Homicide", "Other"]
        if form_data["city"] == CITY_PLACEHOLDER:
            crime_options = fallback_crime_options
            place_options = ["Unknown"]
        else:
            cases_csv_path = _resolve_city_cases_csv(form_data["city"])
            crime_options = load_primary_types(str(cases_csv_path))
            if not crime_options:
                crime_options = fallback_crime_options
            place_options = load_places(str(cases_csv_path))
        primary_type_key = "profile_primary_type"
        primary_type_options = _with_placeholder(crime_options, PRIMARY_TYPE_PLACEHOLDER)
        if st.session_state.get(primary_type_key) not in primary_type_options:
            st.session_state[primary_type_key] = PRIMARY_TYPE_PLACEHOLDER
        primary_choice = st.selectbox("Primary type", options=primary_type_options, key=primary_type_key)
        if primary_choice == PRIMARY_TYPE_PLACEHOLDER:
            form_data["primary_type"] = ""
        elif primary_choice == "Other":
            form_data["primary_type"] = st.text_input("Specify crime type")
        else:
            form_data["primary_type"] = primary_choice
        # location description
        location_key = "profile_location_desc"
        location_options = _with_placeholder(place_options, LOCATION_PLACEHOLDER)
        if st.session_state.get(location_key) not in location_options:
            st.session_state[location_key] = LOCATION_PLACEHOLDER
        form_data["location_desc"] = st.selectbox(
            "Location description",
            options=location_options,
            key=location_key,
        )
        if form_data["location_desc"] == LOCATION_PLACEHOLDER:
            form_data["location_desc"] = ""
        weapon_options = load_la_weapons()
        weapon_key = "profile_weapon_desc"
        weapon_select_options = _with_placeholder(weapon_options, WEAPON_PLACEHOLDER)
        if st.session_state.get(weapon_key) not in weapon_select_options:
            st.session_state[weapon_key] = WEAPON_PLACEHOLDER
        form_data["weapon_desc"] = st.selectbox(
            "Weapon description",
            options=weapon_select_options,
            key=weapon_key,
        )
        if form_data["weapon_desc"] == WEAPON_PLACEHOLDER:
            form_data["weapon_desc"] = ""
        form_data["arrest"] = st.checkbox("Arrest made", value=False)
        form_data["domestic"] = st.checkbox("Domestic incident", value=False)
    # Time & Date
    with st.expander("Time & Date", expanded=True):
        date_col, hour_col, minute_col = st.columns([2, 1, 1])
        with date_col:
            form_data["incident_date"] = st.date_input("Date", value=dt.date.today())
        with hour_col:
            form_data["incident_hour"] = st.selectbox(
                "Hour",
                options=list(range(24)),
                format_func=lambda value: f"{value:02d}",
            )
        with minute_col:
            form_data["incident_minute"] = st.selectbox(
                "Minute",
                options=list(range(60)),
                format_func=lambda value: f"{value:02d}",
            )

        form_data["incident_time"] = dt.time(
            hour=form_data["incident_hour"],
            minute=form_data["incident_minute"],
        )
        form_data["hour"] = form_data["incident_hour"]
        form_data["is_night"] = form_data["hour"] >= 18 or form_data["hour"] <= 5
    # Location coordinates
    with st.expander("Location", expanded=False):
        form_data["latitude"] = st.number_input("Latitude", value=0.0, format="%f")
        form_data["longitude"] = st.number_input("Longitude", value=0.0, format="%f")
    # keep extraneous expanders for layout consistency
    with st.expander("Victim / Context (optional)", expanded=False):
        st.write("Not required for current model.")
        victim_age = st.number_input(
            "Victim age",
            min_value=0,
            max_value=120,
            value=0,
            step=1,
        )
        victim_gender = st.selectbox(
            "Victim gender",
            options=["", "Male", "Female", "Other", "Unknown"],
        )
        victim_race = st.selectbox(
            "Victim race",
            options=[
                "",
                "Unknown",
                "White",
                "Black / African American",
                "Hispanic / Latino",
                "Asian",
                "American Indian / Alaska Native",
                "Native Hawaiian / Pacific Islander",
                "Middle Eastern / North African",
                "Other",
            ],
        )
    # narrative field below tabs
    narrative = st.text_area("Narrative / Description (optional)", height=160)

    # helper to determine if form has any user-provided info. treat
    # zero/empty strings as "no input" and count a checked checkbox as input.
    def _form_is_empty(d: dict) -> bool:
        for k, v in d.items():
            if isinstance(v, str) and v.strip():
                return False
            if isinstance(v, (int, float)) and v != 0:
                return False
            if isinstance(v, bool) and v:
                return False
        return True

    form_empty = _form_is_empty(form_data)

    def _clear_prediction_state() -> None:
        st.session_state["has_prediction"] = False
        for key in (
            "profile_last_fused",
            "profile_last_case_dict",
            "profile_last_narrative",
            "profile_last_case_id",
            "profile_last_rag_results",
            "profile_last_saved_similar",
        ):
            st.session_state.pop(key, None)

    def _missing_required_fields(candidate: dict) -> list[str]:
        def _normalize(value) -> str:
            if value is None:
                return ""
            return str(value).strip()

        def _is_missing(value, placeholder: str, *, allow_unknown: bool = False) -> bool:
            text = _normalize(value)
            if not text:
                return True
            invalid_values = {placeholder.lower(), "none", "nan"}
            if not allow_unknown:
                invalid_values.add("unknown")
            return text.lower() in invalid_values

        missing_fields: list[str] = []
        if _is_missing(candidate.get("city"), CITY_PLACEHOLDER):
            missing_fields.append("City")
        if _is_missing(
            candidate.get("primary_type") or candidate.get("crime_type") or candidate.get("type"),
            PRIMARY_TYPE_PLACEHOLDER,
        ):
            missing_fields.append("Primary type")
        if _is_missing(
            candidate.get("location_desc")
            or candidate.get("location")
            or candidate.get("premise")
            or candidate.get("place"),
            LOCATION_PLACEHOLDER,
        ):
            missing_fields.append("Location description")
        if _is_missing(
            candidate.get("weapon_desc") or candidate.get("weapon"),
            WEAPON_PLACEHOLDER,
            allow_unknown=True,
        ):
            missing_fields.append("Weapon description")
        return missing_fields

    if "has_prediction" not in st.session_state:
        st.session_state["has_prediction"] = False

    def _compute_case_id(inputs: dict, narrative_text: str) -> str:
        subset = {
            "city": inputs.get("city"),
            "primary_type": inputs.get("primary_type") or inputs.get("crime_type") or inputs.get("type"),
            "location_desc": inputs.get("location_desc") or inputs.get("location") or inputs.get("premise"),
            "weapon_desc": inputs.get("weapon_desc") or inputs.get("weapon"),
            "incident_date": inputs.get("incident_date"),
            "incident_time": inputs.get("incident_time"),
            "hour": inputs.get("hour"),
            "is_night": inputs.get("is_night"),
            "arrest": inputs.get("arrest"),
            "domestic": inputs.get("domestic"),
            "latitude": inputs.get("latitude") if inputs.get("latitude") is not None else inputs.get("lat"),
            "longitude": inputs.get("longitude") if inputs.get("longitude") is not None else inputs.get("lon"),
            "narrative_text": narrative_text or "",
        }
        safe_subset = make_json_safe(subset)
        encoded = json.dumps(safe_subset, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]

    # Predict button logic
    predict_clicked = st.button("Predict")
    if predict_clicked or st.session_state.get("has_prediction", False):
        # early exit if the tabular model could not be loaded at startup
        if predict_clicked and model_missing:
            _clear_prediction_state()
            st.error("Cannot predict: tabular model failed to load. Check errors above.")
            return
        # build interim dict from form entries. apply any UIâ†’model translation
        # mapping so that the final payload uses the exact keys expected by
        # the tabular model. the translation dict is normally empty but kept
        # here for backwards compatibility.
        if predict_clicked:
            raw = {}
            for k, v in form_data.items():
                if v in ("", None):
                    continue
                # convert date objects to strings for serialization
                if isinstance(v, (dt.date, dt.datetime, dt.time)):
                    v = v.isoformat()
                model_key = UI_TO_MODEL.get(k, k)
                raw[model_key] = v

            def _clean_optional_text(value):
                if value in (None, ""):
                    return None
                text = str(value).strip()
                return text or None

            # construct case_dict with every feature name from the model
            case_dict: dict = {feat: raw.get(feat, 0) for feat in _features}
            for k, v in raw.items():
                if k not in case_dict:
                    case_dict[k] = v
            victim_age_to_save = None if victim_age == 0 else int(victim_age)
            case_dict["victim"] = {
                "age": victim_age_to_save,
                "gender": _clean_optional_text(victim_gender),
                "race": _clean_optional_text(victim_race),
            }

            missing_fields = _missing_required_fields(case_dict)
            if missing_fields:
                _clear_prediction_state()
                st.warning(
                    "Please fill in all required incident details before running prediction. "
                    f"Missing: {', '.join(missing_fields)}."
                )
                return

            case_id = _compute_case_id(case_dict, narrative)
            st.session_state["profile_last_case_id"] = case_id

            # compute some diagnostics about the final inputs we will send
            from services.tabular_service import prepare_tabular_row

            try:
                row_for_model, row_summary = prepare_tabular_row(case_dict)
            except Exception:
                # protect against our helper failing
                row_for_model, row_summary = case_dict, {"filled": 0, "defaulted": 0, "missing": []}

            # proceed with existing prediction pipeline
            from services.pipeline_service import run_profile
            from services.tabular_service import ModelNotAvailableError as TabErr
            from services.nlp_service import ModelNotAvailableError as NlpErr

            with st.spinner("Running prediction pipeline..."):
                try:
                    fused = run_profile(case_dict, narrative)
                except (TabErr, NlpErr) as err:
                    _clear_prediction_state()
                    st.error(str(err))
                    return
                except Exception as err:  # catch-all to avoid crashing the UI
                    _clear_prediction_state()
                    st.error(f"Prediction failed: {err}")
                    return

            st.session_state["profile_last_fused"] = fused
            st.session_state["profile_last_case_dict"] = case_dict
            st.session_state["profile_last_narrative"] = narrative
            st.session_state["has_prediction"] = True
            _log_profile_event(
                "predict",
                case_id,
                fused,
                {"narrative_present": bool((narrative or "").strip())},
            )
        else:
            fused = st.session_state.get("profile_last_fused")
            case_dict = st.session_state.get("profile_last_case_dict", {})
            narrative = st.session_state.get("profile_last_narrative", narrative)
            case_id = st.session_state.get("profile_last_case_id")
            if not fused or not st.session_state.get("has_prediction", False):
                _clear_prediction_state()
                return

        if not st.session_state.get("profile_last_case_id"):
            st.session_state["profile_last_case_id"] = _compute_case_id(case_dict, narrative)
        case_id = st.session_state.get("profile_last_case_id")

        # helper to clean display strings (remove mock tags)
        def clean_str(val):
            if isinstance(val, str):
                return val.replace("(mock)", "").strip()
            return val

        final = fused.get("final", {})
        # summary metrics â€“ only display the true model outputs now.
        from components.explanations import render_ethical_notice
        risk = clean_str(final.get("risk_level") or final.get("risk_category", "Unknown"))
        severity = clean_str(final.get("crime_severity", EM_DASH))
        experience = clean_str(final.get("offender_experience", EM_DASH))
        mot_info = final.get("motive") if isinstance(final.get("motive"), dict) else {}
        motive_pred = clean_str(mot_info.get("pred", "")) if mot_info else ""
        motive_conf = mot_info.get("conf") or mot_info.get("confidence")
        motive_band = mot_info.get("band")

        # mimic the prior fused summary style but only show relevant columns
        st.caption(f"Case ID: {case_id}")
        cols = st.columns(4)
        cols[0].metric("Risk level", risk)
        cols[1].metric("Crime severity", severity)
        cols[2].metric("Offender experience", experience)
        if motive_pred:
            delta_text = None
            if motive_conf is not None or motive_band:
                conf_str = f"{motive_conf:.2f}" if motive_conf is not None else "â€”"
                delta_text = f"{conf_str}{' (' + motive_band + ')' if motive_band else ''}"
            cols[3].metric("Final motive", motive_pred, delta=delta_text)
        else:
            cols[3].metric("Final motive", "â€”")

        # profiling traits table â€“ only keep the three targets (motive shown separately)
        import pandas as pd

        traits = []
        tabular_pred = fused.get("_tabular_output", {}).get("pred", {})
        for trait_name in PROFILE_TRAITS:
            raw_value = final.get(trait_name)
            if raw_value is None and isinstance(tabular_pred, dict):
                raw_value = tabular_pred.get(trait_name)
            if isinstance(raw_value, dict):
                prediction = clean_str(raw_value.get("pred", ""))
            else:
                prediction = clean_str(raw_value)
            traits.append(
                {
                    "Trait": trait_name,
                    "Prediction": prediction or EM_DASH,
                    "Confidence/Band": _format_trait_confidence_display(fused, trait_name),
                }
            )
        if traits:
            st.markdown("### Profiling traits")
            st.table(pd.DataFrame(traits))

        # motive details section
        st.markdown("### Motive details")
        if motive_pred:
            conf_display = f"{motive_conf:.2f}" if motive_conf is not None else "â€”"
            st.markdown(
                f"**Motive:** {motive_pred} (Confidence: {conf_display}, Band: {motive_band or 'â€”'})"
            )
        else:
            st.write("**Motive:** Not provided")
            if not narrative.strip():
                st.warning("Narrative missing â€” motive not predicted.")
        topk = fused.get("explanations", {}).get("nlp_topk", [])
        if topk:
            st.markdown("#### NLP topâ€‘k probabilities")
            df_topk = pd.DataFrame(topk)
            if "prob" in df_topk.columns:
                df_topk["prob"] = df_topk["prob"].astype(float).round(3)
            df_topk = df_topk.rename(columns={"label": "Label", "prob": "Probability"})
            st.table(df_topk)

        # similar past cases via RAG retrieval + similar saved history
        rag_message: str | None = None
        sim_cases: list = []
        saved_sim_cases: list = []
        saved_message: str | None = None
        saved_message_level = "info"
        saved_debug: dict = {}
        rag_meta: dict[str, object] = {}
        rag_summary_to_save: dict[str, object] = {}
        rag_top_cases_to_save: list[dict] = []
        rag_similar_saved_cases_to_save: list[dict] = []

        def _pick_case_value(source: dict, *keys: str):
            for key in keys:
                if key in source and source.get(key) not in (None, ""):
                    return source.get(key)
            return None

        def _clean_text_value(value):
            if value is None:
                return ""
            text = str(value).strip()
            if text.lower() in {"", "unknown", "none", "nan"}:
                return ""
            return text

        def _normalize_saved_value(value):
            return str(value or "").strip().lower()

        def _saved_location_keyword(value):
            text = _normalize_saved_value(value)
            if not text:
                return ""
            for keyword in ["street", "residence", "apartment", "house", "business", "store", "park", "school"]:
                if keyword in text:
                    return keyword
            return text

        def _saved_bool(value):
            if isinstance(value, bool):
                return value
            text = _normalize_saved_value(value)
            if text in {"true", "1", "yes", "y"}:
                return True
            if text in {"false", "0", "no", "n"}:
                return False
            return None

        raw_primary = _pick_case_value(case_dict, "primary_type", "crime_type", "type")
        primary_type = _clean_text_value(raw_primary)
        if not primary_type:
            primary_type = _clean_text_value(fused.get("final", {}).get("crime_severity")) or "Unknown"

        weapon_desc = _clean_text_value(_pick_case_value(case_dict, "weapon_desc", "weapon"))

        raw_location = _pick_case_value(case_dict, "location_desc", "location", "premise", "place")
        location_desc = _clean_text_value(raw_location)
        if not location_desc:
            location_candidates = [
                raw_location,
                case_dict.get("location_desc"),
                case_dict.get("location"),
                case_dict.get("premise"),
                case_dict.get("place"),
            ]
            selected_street = any(
                isinstance(item, str) and "street" in item.strip().lower()
                for item in location_candidates
            )
            location_desc = "STREET" if selected_street else ""

        hour_value = case_dict.get("hour")
        if hour_value in ("", "Unknown"):
            hour_value = None

        is_night_value = case_dict.get("is_night")
        if isinstance(is_night_value, str):
            lowered = is_night_value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                is_night_value = True
            elif lowered in {"false", "0", "no", "n"}:
                is_night_value = False
            else:
                is_night_value = None
        elif not isinstance(is_night_value, bool):
            is_night_value = None

        rag_case_dict = {
            "city": case_dict.get("city", ""),
            "primary_type": primary_type or "Unknown",
            "weapon_desc": weapon_desc,
            "location_desc": location_desc,
            "hour": hour_value,
            "is_night": is_night_value,
        }

        rag_query_text = ""
        try:
            from services.rag_service import build_query_text

            rag_query_text = build_query_text(rag_case_dict)
        except Exception:
            rag_query_text = ""

        try:
            from services.pipeline_service import get_evidence_bundle
            from services.firebase_service import list_history_records

            evidence_bundle = get_evidence_bundle(rag_case_dict, narrative, fused_output=fused, top_k=10)
            sim_cases = evidence_bundle.get("rag", []) or evidence_bundle.get("rag_results", []) or []
            rag_message = evidence_bundle.get("rag_message")
            rag_meta = evidence_bundle.get("rag_meta") or {}
            warning_list = evidence_bundle.get("warnings", []) or []
            if not rag_message:
                rag_warns = [w for w in warning_list if "rag" in str(w).lower()]
                rag_message = "; ".join(rag_warns) if rag_warns else None

            cur_type_raw = case_dict.get("primary_type")
            cur_type = _normalize_saved_value(cur_type_raw)
            saved_records = list_history_records(limit=200)
            observed_types: list[str] = []
            same_type_records: list[dict] = []
            for rec in saved_records:
                saved_inputs = rec.get("inputs") or {}
                saved_type = _normalize_saved_value(saved_inputs.get("primary_type"))
                if len(observed_types) < 10:
                    observed_types.append(saved_type)
                if not saved_type:
                    continue
                if saved_type == cur_type:
                    same_type_records.append(rec)

            saved_debug = {
                "cur_type_raw": cur_type_raw,
                "cur_type": cur_type,
                "total_records": len(saved_records),
                "same_type_records": len(same_type_records),
                "saved_types_sample": observed_types,
            }

            if not cur_type or cur_type == "unknown":
                saved_sim_cases = []
                saved_message = "Select a Primary type to see similar saved cases."
                saved_message_level = "info"
            else:
                scored_saved_cases = []
                for rec in same_type_records:
                    saved_inputs = rec.get("inputs") or {}
                    outputs = rec.get("outputs") or rec.get("fused_output") or rec.get("fusion_output") or {}
                    final_saved = outputs.get("final", {}) if isinstance(outputs, dict) else {}
                    motive_obj = final_saved.get("motive", {})
                    motive = motive_obj.get("pred") if isinstance(motive_obj, dict) else motive_obj
                    victim = saved_inputs.get("victim", {})
                    if not isinstance(victim, dict):
                        victim = rec.get("victim", {})
                    if not isinstance(victim, dict):
                        victim = {}

                    score = 1.0
                    if _normalize_saved_value(saved_inputs.get("weapon_desc")) == _normalize_saved_value(case_dict.get("weapon_desc")):
                        if _normalize_saved_value(case_dict.get("weapon_desc")):
                            score += 1.0
                    if _saved_location_keyword(saved_inputs.get("location_desc")) == _saved_location_keyword(case_dict.get("location_desc")):
                        if _saved_location_keyword(case_dict.get("location_desc")):
                            score += 1.0
                    if _saved_bool(saved_inputs.get("is_night")) == _saved_bool(case_dict.get("is_night")):
                        if _saved_bool(case_dict.get("is_night")) is not None:
                            score += 0.5

                    scored_saved_cases.append(
                        {
                            "score": round(float(score), 3),
                            "saved_time": str(rec.get("created_at_local") or rec.get("created_at") or ""),
                            "record_id": rec.get("id", ""),
                            "type": saved_inputs.get("primary_type") or "",
                            "weapon": saved_inputs.get("weapon_desc") or saved_inputs.get("weapon") or "",
                            "place": saved_inputs.get("location_desc") or "",
                            "area": saved_inputs.get("area") or saved_inputs.get("city") or "",
                            "hour": saved_inputs.get("hour"),
                            "is_night": saved_inputs.get("is_night"),
                            "risk_level": final_saved.get("risk_level") or final_saved.get("risk_category") or outputs.get("final_risk_category") or "",
                            "motive": motive or "",
                            "victim_age": victim.get("age") if victim.get("age") not in (None, "") else EM_DASH,
                            "victim_gender": victim.get("gender") if victim.get("gender") not in (None, "") else EM_DASH,
                            "victim_race": victim.get("race") if victim.get("race") not in (None, "") else EM_DASH,
                            "_raw": rec,
                        }
                    )

                scored_saved_cases.sort(
                    key=lambda item: (item.get("score", 0), item.get("saved_time", "")),
                    reverse=True,
                )
                saved_sim_cases = scored_saved_cases[:5]
                if not saved_sim_cases:
                    saved_message = f"No matching saved cases for Primary type: {cur_type_raw}"
                    saved_message_level = "warning"
                else:
                    saved_message = None
                    saved_message_level = "info"
        except Exception as err:  # catch-all to avoid crashing UI
            sim_cases = []
            saved_sim_cases = []
            rag_message = f"Error retrieving similar cases: {err}"
            saved_message = f"Error retrieving similar saved cases: {err}"
            saved_message_level = "warning"
            saved_debug = {}

        st.markdown("### Similar Past Cases")
        st.markdown("#### Top Similar Cases")
        if sim_cases:
            df_sim = _sanitize_evidence_display_df(pd.DataFrame(sim_cases))
            rag_dataset_id = _resolve_rag_dataset_id(rag_meta, df_sim)
            if rag_dataset_id == "nypd" and "law_category" not in df_sim.columns:
                df_sim["law_category"] = ""
            if rag_dataset_id == "nypd" and "attempt_status" not in df_sim.columns:
                df_sim["attempt_status"] = ""
            df_sim_display = _prepare_rag_results_df(df_sim, rag_dataset_id).head(10)
            rag_summary_to_save = _build_rag_summary_for_dataset(df_sim, rag_dataset_id)
            summary_lines = _rag_summary_lines(rag_summary_to_save)
            rag_top_cases_to_save = df_sim_display.to_dict(orient="records")
            st.markdown("\n".join(f"- {line}" for line in summary_lines))
            st.dataframe(df_sim_display, use_container_width=True)
        else:
            st.info(rag_message or "No similar cases found / index missing.")

        st.markdown("#### Similar Saved Cases")
        if saved_sim_cases:
            df_saved = pd.DataFrame(saved_sim_cases)
            saved_cols = _build_saved_display_columns(df_saved)
            rag_similar_saved_cases_to_save = df_saved.reindex(columns=saved_cols).head(10).to_dict(orient="records")
            st.dataframe(df_saved.reindex(columns=saved_cols))
        else:
            if saved_message_level == "warning" and saved_message:
                st.warning(saved_message)
            else:
                st.info(saved_message or "No similar saved cases found.")

        # feedback + save/export
        st.markdown("### Feedback")
        feedback_text = st.text_area("Feedback (optional)", key="profile_feedback_text")
        helpfulness = st.selectbox(
            "Was this helpful?",
            options=["", "Yes", "No", "Not sure"],
            key="profile_helpfulness",
        )

        def _pick_value(source: dict, *keys: str):
            for key in keys:
                value = source.get(key)
                if value not in (None, ""):
                    return value
            return None

        def _to_int_flag(value) -> int:
            if isinstance(value, bool):
                return int(value)
            if value is None:
                return 0
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "y"}:
                return 1
            if text in {"0", "false", "no", "n"}:
                return 0
            return 0

        def _to_int_or_none(value):
            if value in (None, ""):
                return None
            try:
                return int(float(value))
            except Exception:  # noqa: BLE001
                return None

        def _to_float_or_none(value):
            if value in (None, ""):
                return None
            try:
                return float(value)
            except Exception:  # noqa: BLE001
                return None

        def _to_text_or_none(value):
            if value in (None, ""):
                return None
            text = str(value).strip()
            return text or None

        inputs_payload = make_json_safe(dict(case_dict))
        inputs_payload["primary_type"] = (
            _pick_value(inputs_payload, "primary_type", "crime_type", "type") or "Unknown"
        )
        inputs_payload["location_desc"] = (
            _pick_value(inputs_payload, "location_desc", "location", "premise", "place") or ""
        )
        inputs_payload["weapon_desc"] = _pick_value(inputs_payload, "weapon_desc", "weapon") or ""
        inputs_payload["hour"] = _to_int_or_none(inputs_payload.get("hour"))
        inputs_payload["is_night"] = _to_int_flag(inputs_payload.get("is_night"))
        inputs_payload["arrest"] = _to_int_flag(inputs_payload.get("arrest"))
        inputs_payload["domestic"] = _to_int_flag(inputs_payload.get("domestic"))
        inputs_payload["latitude"] = _to_float_or_none(inputs_payload.get("latitude"))
        inputs_payload["longitude"] = _to_float_or_none(inputs_payload.get("longitude"))
        inputs_payload["narrative_text"] = narrative or inputs_payload.get("narrative_text") or ""
        victim_payload = inputs_payload.get("victim") if isinstance(inputs_payload.get("victim"), dict) else {}
        victim_age_value = _to_int_or_none(victim_payload.get("age"))
        if victim_age_value is None:
            victim_age_value = _to_text_or_none(victim_payload.get("age"))
        inputs_payload["victim"] = {
            "age": victim_age_value,
            "gender": _to_text_or_none(victim_payload.get("gender")),
            "race": _to_text_or_none(victim_payload.get("race")),
        }

        for optional_key, source_keys in (
            ("victim_name", ("victim_name", "victim")),
            ("victim_age", ("victim_age", "age")),
            ("victim_gender", ("victim_gender", "victim_sex", "gender")),
        ):
            optional_value = _pick_value(inputs_payload, *source_keys)
            if optional_value not in (None, ""):
                inputs_payload[optional_key] = optional_value
        if inputs_payload["victim"].get("age") not in (None, ""):
            inputs_payload["victim_age"] = inputs_payload["victim"].get("age")
        if inputs_payload["victim"].get("gender") not in (None, ""):
            inputs_payload["victim_gender"] = inputs_payload["victim"].get("gender")
        if inputs_payload["victim"].get("race") not in (None, ""):
            inputs_payload["victim_race"] = inputs_payload["victim"].get("race")

        nested_outputs = fused.get("outputs") if isinstance(fused.get("outputs"), dict) else {}
        outputs_to_save = {
            "final": fused.get("final") or nested_outputs.get("final"),
            "model_version": fused.get("model_version") or (fused.get("meta") or {}).get("model_version") or "",
            "fusion_meta": fused.get("fusion_meta") or {},
        }
        rag_results_to_save = []
        if isinstance(sim_cases, list):
            for item in sim_cases[:10]:
                if isinstance(item, dict):
                    rag_results_to_save.append(make_json_safe(item))

        rag_payload = {
            "dataset": case_dict.get("city", ""),
            "summary": rag_summary_to_save,
            "top_cases": rag_top_cases_to_save,
            "similar_saved_cases": rag_similar_saved_cases_to_save,
        }

        record_timestamp = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
        record_payload = {
            "created_at_local": record_timestamp,
            "inputs": inputs_payload,
            "narrative_text": narrative or "",
            "outputs": make_json_safe(outputs_to_save),
            "rag_results": rag_results_to_save,
            "rag": rag_payload,
            "feedback": {
                "text": feedback_text or "",
                "helpful": helpfulness or None,
            },
        }
        record_payload_safe = make_json_safe(record_payload)

        save_col, export_col = st.columns(2)
        with save_col:
            if st.button("Save", key="profile_save_button"):
                firestore_collection = "history"
                firebase_ready = False
                try:
                    import services.audit_service as audit_service
                    from services.case_id_service import generate_case_id
                    from services.firebase_service import firestore_safe, init_firebase, save_case_by_id

                    init_firebase()
                    firebase_ready = True
                    utc_now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
                    current_user = audit_service.get_current_user()
                    record_payload = {
                        "created_at_local": utc_now.isoformat(),
                        "inputs": inputs_payload,
                        "narrative_text": narrative or "",
                        "outputs": outputs_to_save,
                        "rag_results": rag_results_to_save,
                        "rag": rag_payload,
                        "feedback": {
                            "text": feedback_text or "",
                            "helpful": helpfulness or None,
                        },
                    }
                    save_case_id = generate_case_id(record_payload["inputs"].get("city", ""), utc_now)
                    record_payload["case_id"] = save_case_id
                    record_payload = firestore_safe(record_payload)
                    save_case_by_id(save_case_id, record_payload)
                    st.caption(f"Firestore collection: {firestore_collection} | init_firebase: ok")
                    st.success(f"Saved: {save_case_id}")

                    try:
                        audit_logged = audit_service.log_event(
                            action="save",
                            user=current_user,
                            case_id=save_case_id,
                            model_ver=_audit_model_version(fused),
                            data_ver=_audit_data_version(fused),
                            page="profile",
                        )
                        if not audit_logged:
                            st.warning("Saved to history, but audit logging failed.")
                    except Exception as audit_err:  # noqa: BLE001
                        st.warning(f"Saved to history, but audit logging failed: {type(audit_err).__name__}: {audit_err}")
                except Exception as e:  # noqa: BLE001
                    init_status = "ok" if firebase_ready else "failed"
                    st.caption(f"Firestore collection: {firestore_collection} | init_firebase: {init_status}")
                    outputs_obj = record_payload.get("outputs") if isinstance(record_payload, dict) else None
                    outputs_type = type(outputs_obj).__name__
                    outputs_keys = list(outputs_obj.keys()) if isinstance(outputs_obj, dict) else []
                    st.error(f"Save failed: {e} | outputs_type={outputs_type} | outputs_keys={outputs_keys}")

        with export_col:
            from services.export_service import build_profile_doc

            export_case_id = record_payload_safe.get("case_id") or case_id or "case"
            export_record = {
                "case_id": export_case_id,
                "timestamp": record_payload_safe.get("created_at_local") or record_timestamp,
                "inputs": record_payload_safe.get("inputs") or {},
                "outputs": record_payload_safe.get("outputs") or {},
                "rag_summary": (record_payload_safe.get("rag") or {}).get("summary") or {},
                "rag_results": (record_payload_safe.get("rag") or {}).get("top_cases") or [],
                "similar_saved_cases": (record_payload_safe.get("rag") or {}).get("similar_saved_cases") or [],
                "feedback": record_payload_safe.get("feedback") or {},
                "rag": record_payload_safe.get("rag") or {},
                "narrative_text": record_payload_safe.get("narrative_text") or "",
            }
            export_name = f"{export_case_id}_report.docx"
            export_error = None
            export_bytes = None
            try:
                export_bytes = build_profile_doc(export_record)
            except Exception as exc:  # noqa: BLE001
                export_error = str(exc)

            if export_bytes is not None:
                export_clicked = st.download_button(
                    "Export",
                    data=export_bytes,
                    file_name=export_name,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    key="export_profile_docx",
                )
            else:
                export_clicked = False
                st.button(
                    "Export",
                    disabled=True,
                    use_container_width=True,
                    key="export_profile_docx_disabled",
                    help=export_error or "Export unavailable",
                )
            if export_clicked:
                _log_profile_event("export", case_id, fused)

        # warnings and ethical notice
        warn = fused.get("fusion_meta", {}).get("warnings", [])
        if warn:
            st.markdown("### Warnings")
            for w in warn:
                st.warning(w)
        render_ethical_notice()

        # explanation expanders
        with st.expander("Why this prediction?"):
            expl = fused.get("explanations", {}).get("tabular_text_expl")
            tab_feats = fused.get("explanations", {}).get("tabular_shap_top", [])
            st.caption(
                f"tabular_text_expl present: {bool(expl)} | explanation keys: {list(fused.get('explanations', {}).keys())}"
            )
            if expl:
                st.markdown(str(expl.get("summary") or "").strip())
                st.markdown("**Key factors increasing the prediction:**")
                st.markdown("\n".join([f"- {x}" for x in (expl.get("drivers_up") or [])]))
                st.markdown("**Key factors decreasing the prediction:**")
                st.markdown("\n".join([f"- {x}" for x in (expl.get("drivers_down") or [])]))
            else:
                st.write("No tabular feature explanations available.")
            if tab_feats:
                df_feats = pd.DataFrame(tab_feats)[:8]
                df_feats = df_feats.rename(columns={"feature": "Feature", "value": "Value"})
                st.table(df_feats)


if __name__ == "__main__":
    main()
