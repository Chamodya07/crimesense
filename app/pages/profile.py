from __future__ import annotations

import json
import sys
import datetime as dt
import hashlib
from pathlib import Path

import streamlit as st

# Ensure project root is on path for backend imports
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.auth_service import render_auth_status

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
CSS_FILE = ASSETS_DIR / "styles.css"


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


def _audit_model_version(fused: dict | None) -> str | None:
    if not isinstance(fused, dict):
        return None
    return (
        fused.get("model_version")
        or (fused.get("fusion_meta") or {}).get("version")
        or None
    )


def _audit_data_version(fused: dict | None) -> str | None:
    if not isinstance(fused, dict):
        return None
    return (
        fused.get("data_version")
        or fused.get("rag_index_version")
        or (fused.get("explanations") or {}).get("data_version")
        or None
    )


def _log_profile_event(action: str, case_id: str | None, fused: dict | None, meta: dict | None = None) -> None:
    try:
        from services.audit_service import get_current_user_label, log_event

        log_event(
            action=action,
            user=get_current_user_label(),
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
    st.markdown("Provide structured details about the incident below. You may also paste a full case JSON if you prefer.")

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


    # --- Input tabs -------------------------------------------------------
    tab1, tab2 = st.tabs(["Form Input", "Paste JSON"])

    # form data collection
    form_data: dict = {}
    with tab1:
        # Incident Details
        with st.expander("Incident Details", expanded=True):
            form_data["city"] = st.text_input("City", value="Chicago")
            # primary_type selectbox with ability to specify custom
            crime_options = ["", "Theft", "Assault", "Robbery", "Burglary", "Homicide", "Other"]
            primary_choice = st.selectbox("Primary type", options=crime_options)
            if primary_choice == "Other":
                form_data["primary_type"] = st.text_input("Specify crime type")
            else:
                form_data["primary_type"] = primary_choice
            # location description
            loc_options = ["", "Street", "Residence", "Business", "Park", "Other"]
            loc_choice = st.selectbox("Location description", options=loc_options)
            if loc_choice == "Other":
                form_data["location_desc"] = st.text_input("Specify location description")
            else:
                form_data["location_desc"] = loc_choice
            form_data["weapon_desc"] = st.selectbox(
                "Weapon description",
                options=["", "None", "Knife", "Gun", "Blunt", "Unknown", "Other"],
            )
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
        with st.expander("Offender Indicators (optional)", expanded=False):
            st.write("These values are predicted and will appear after you click **Predict**.")

    # JSON paste tab
    case_json = ""
    json_valid = False
    parsed_json: dict = {}
    json_error: str | None = None
    with tab2:
        case_json = st.text_area("Paste case JSON", value="", height=200)
        if case_json.strip():
            try:
                parsed_json = json.loads(case_json)
                if not isinstance(parsed_json, dict):
                    raise ValueError("JSON must be an object with key/value pairs")
                json_valid = True
                st.success(f"Parsed JSON with {len(parsed_json.keys())} top-level keys.")
            except Exception as e:  # noqa: BLE001
                json_error = str(e)
                st.error(f"Invalid JSON: {json_error}")

    # narrative field below tabs
    narrative = st.text_area("Narrative / Description (optional)", height=160)
    st.caption("If narrative is provided, NLP + late fusion will run; otherwise tabular-only.")

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

    def _valid_required_fields(candidate: dict) -> bool:
        def _clean(value):
            if value is None:
                return ""
            text = str(value).strip()
            if text.lower() in {"", "unknown", "none", "nan"}:
                return ""
            return text

        primary = _clean(candidate.get("primary_type") or candidate.get("crime_type") or candidate.get("type"))
        location = _clean(
            candidate.get("location_desc")
            or candidate.get("location")
            or candidate.get("premise")
            or candidate.get("place")
        )
        return bool(primary and location)

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
        # validation
        if predict_clicked and case_json.strip() and not json_valid:
            _clear_prediction_state()
            st.error("Cannot predict: JSON is invalid.")
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

            # merge JSON override (JSON wins when keys collide)
            if json_valid and parsed_json:
                raw.update(parsed_json)

            # construct case_dict with every feature name from the model
            case_dict: dict = {feat: raw.get(feat, 0) for feat in _features}
            for k, v in raw.items():
                if k not in case_dict:
                    case_dict[k] = v

            if not _valid_required_fields(case_dict):
                _clear_prediction_state()
                st.warning("Please fill required fields: Crime type and Location.")
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

            # debug expander showing the features going to the model
            with st.expander("Debug (model inputs)"):
                non_zero = row_summary.get("filled", 0)
                st.write(f"Non-default features: {non_zero} of {len(_features)}")
                # preview up to 20 non-default key/value pairs so users can sanity
                # check what will be sent. don't dump the entire dict here to keep
                # the expander concise.
                nonzero_items = [(k, v) for k, v in row_for_model.items() if v not in (0, "", None)]
                if nonzero_items:
                    st.write("Sample non-default inputs:", nonzero_items[:20])
                if non_zero < 3:
                    st.warning("Most features are defaulted; model may output constant results.")

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
        severity = clean_str(final.get("crime_severity", "â€”"))
        experience = clean_str(final.get("offender_experience", "â€”"))
        mot_info = final.get("motive") if isinstance(final.get("motive"), dict) else {}
        motive_pred = clean_str(mot_info.get("pred", "")) if mot_info else ""
        motive_conf = mot_info.get("conf") or mot_info.get("confidence")
        motive_band = mot_info.get("band")

        # mimic the prior fused summary style but only show relevant columns
        st.markdown("### Fused offender profile (combined models)")
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
        allowed = {"risk_level", "crime_severity", "offender_experience"}
        for k, v in final.items():
            if k not in allowed:
                continue
            if isinstance(v, dict):
                pred = clean_str(v.get("pred", ""))
                band = v.get("band") or v.get("conf") or v.get("confidence", "")
                traits.append({"Trait": k, "Prediction": pred, "Confidence/Band": band})
            else:
                traits.append({"Trait": k, "Prediction": clean_str(v), "Confidence/Band": ""})
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

        rag_dir = _ROOT / "artifacts" / "rag"
        rag_index = rag_dir / "faiss.index"
        rag_cases = rag_dir / "cases.csv"
        has_index = rag_index.exists()
        has_cases = rag_cases.exists()
        rag_ready = has_index and has_cases

        with st.expander("Debug (RAG paths)"):
            st.write(f"artifacts/rag (absolute): {rag_dir.resolve()}")
            st.write(f"faiss.index path: {rag_index.resolve()}")
            st.write(f"cases.csv path: {rag_cases.resolve()}")
            st.write(f"faiss.index exists: {has_index}")
            st.write(f"cases.csv exists: {has_cases}")
            st.write("rag_case_dict:")
            st.json(make_json_safe(rag_case_dict))
            st.write(f"rag_query_text: {rag_query_text}")

        try:
            from services.pipeline_service import get_evidence_bundle

            evidence_bundle = get_evidence_bundle(rag_case_dict, narrative, fused_output=fused, top_k=5)
            sim_cases = evidence_bundle.get("rag", []) or evidence_bundle.get("rag_results", []) or []
            rag_message = evidence_bundle.get("rag_message")
            saved_sim_cases = evidence_bundle.get("saved", []) or evidence_bundle.get("saved_similar", []) or []
            saved_message = evidence_bundle.get("saved_message")
            warning_list = evidence_bundle.get("warnings", []) or []
            if not rag_message:
                rag_warns = [w for w in warning_list if "rag" in str(w).lower()]
                rag_message = "; ".join(rag_warns) if rag_warns else None
            if not saved_message:
                saved_warns = [w for w in warning_list if "saved" in str(w).lower() or "firebase" in str(w).lower()]
                saved_message = "; ".join(saved_warns) if saved_warns else None
        except Exception as err:  # catch-all to avoid crashing UI
            sim_cases = []
            saved_sim_cases = []
            rag_message = f"Error retrieving similar cases: {err}"
            saved_message = f"Error retrieving similar saved cases: {err}"

        st.markdown("### Similar Past Cases (RAG Evidence)")
        st.markdown("#### Top Similar Cases (RAG Index)")
        if sim_cases:
            df_sim = pd.DataFrame(sim_cases)
            base_cols = [
                "score",
                "case_id",
                "type",
                "place",
                "area",
            ]
            extra_cols = [
                "victim_age",
                "victim_sex",
                "victim_race",
                "suspect_age",
                "suspect_sex",
                "suspect_race",
            ]

            display_cols = [col for col in base_cols if col in df_sim.columns]
            for col in extra_cols:
                if col not in df_sim.columns:
                    continue
                non_empty = df_sim[col].apply(lambda x: str(x).strip() not in {"", "nan", "None"}).any()
                if non_empty:
                    display_cols.append(col)
            st.dataframe(df_sim.reindex(columns=display_cols))
        else:
            st.info(rag_message or "No similar cases found / index missing.")

        st.markdown("#### Similar Saved Cases (History)")
        if saved_sim_cases:
            df_saved = pd.DataFrame(saved_sim_cases)
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
                "victim_age",
                "victim_sex",
                "victim_race",
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
            ]
            saved_cols = []
            for col in preferred_saved_cols:
                if col in df_saved.columns:
                    non_empty = df_saved[col].apply(lambda x: str(x).strip() not in {"", "nan", "None"}).any()
                    if non_empty:
                        saved_cols.append(col)
            for col in df_saved.columns:
                if col == "_raw" or col in saved_cols:
                    continue
                non_empty = df_saved[col].apply(lambda x: str(x).strip() not in {"", "nan", "None"}).any()
                if non_empty:
                    saved_cols.append(col)
            st.dataframe(df_saved.reindex(columns=saved_cols))

            for idx, row in enumerate(saved_sim_cases, start=1):
                score = float(row.get("score", 0.0))
                record_id = row.get("record_id", f"#{idx}")
                with st.expander(f"Saved Case {idx}: {record_id} (score: {score:.3f})"):
                    st.json(make_json_safe(row.get("_raw", row)))
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

        for optional_key, source_keys in (
            ("victim_name", ("victim_name", "victim")),
            ("victim_age", ("victim_age", "age")),
            ("victim_gender", ("victim_gender", "victim_sex", "gender")),
        ):
            optional_value = _pick_value(inputs_payload, *source_keys)
            if optional_value not in (None, ""):
                inputs_payload[optional_key] = optional_value

        record_timestamp = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
        record_payload = {
            "case_id": case_id,
            "created_at_local": record_timestamp,
            "inputs": inputs_payload,
            "narrative_text": narrative or "",
            "outputs": make_json_safe(fused),
            "rag_results": make_json_safe(sim_cases) if isinstance(sim_cases, list) else [],
            "feedback": {
                "text": feedback_text or "",
                "helpful": helpfulness or None,
            },
        }
        record_payload_safe = make_json_safe(record_payload)

        save_col, export_col = st.columns(2)
        with save_col:
            if st.button("Save", key="profile_save_button"):
                try:
                    from services.firebase_service import (
                        FirebaseConfigError,
                        save_case_by_id,
                    )

                    save_case_by_id(case_id, record_payload_safe)
                    st.success(f"Saved to Firebase (record: {case_id}).")
                    _log_profile_event("save", case_id, fused)
                except FirebaseConfigError:
                    st.error("Firebase not configured. Add service account JSON to Streamlit secrets.")
                except Exception as err:  # noqa: BLE001
                    st.error(f"Failed to save record: {err}")

        with export_col:
            export_name = f"profile_record_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            export_clicked = st.download_button(
                "Export",
                data=json.dumps(record_payload_safe, ensure_ascii=False, indent=2),
                file_name=export_name,
                mime="application/json",
                use_container_width=True,
                key="profile_export_button",
            )
            if export_clicked:
                _log_profile_event("export", case_id, fused)

        # offender indicators expander with predicted metrics
        with st.expander("Offender Indicators (optional)"):
            st.write("Not applicable â€“ model does not predict additional indicators.")

        # warnings and ethical notice
        warn = fused.get("fusion_meta", {}).get("warnings", [])
        if warn:
            st.markdown("### Warnings")
            for w in warn:
                st.warning(w)
        render_ethical_notice()

        # explanation expanders
        with st.expander("Why this prediction?"):
            tab_feats = fused.get("explanations", {}).get("tabular_top_features", [])
            if tab_feats:
                df_feats = pd.DataFrame(tab_feats)[:8]
                df_feats = df_feats.rename(columns={"feature": "Feature", "value": "Value"})
                st.table(df_feats)
            else:
                st.write("No tabular feature explanations available.")

        with st.expander("NLP evidence (topâ€‘k)"):
            if topk:
                st.table(df_topk)
            else:
                st.write("No NLP motive topâ€‘k data available.")

        # debug output for developers
        with st.expander("DEBUG: raw fused output"):
            st.json(fused)


if __name__ == "__main__":
    main()
