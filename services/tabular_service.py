"""Tabular model service: load, prepare, predict with optional SHAP top features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys

# optional imports for runtime version reporting; kept local to avoid
# requiring heavy dependencies when only metadata is needed.


def get_runtime_versions() -> Dict[str, str]:
    """Return a dict of key library versions for diagnostics.

    Used when model loading fails due to incompatibilities.  The values are
    simple strings suitable for printing or including in error messages.
    """
    try:
        import sklearn
    except ImportError:  # pragma: no cover - sklearn should exist later
        skl = "<missing>"
    else:
        skl = getattr(sklearn, "__version__", "?")

    try:
        import numpy as _np
    except ImportError:  # pragma: no cover
        npv = "<missing>"
    else:
        npv = getattr(_np, "__version__", "?")

    try:
        import pandas as _pd
    except ImportError:  # pragma: no cover
        pdv = "<missing>"
    else:
        pdv = getattr(_pd, "__version__", "?")

    return {
        "python": sys.version.split()[0],
        "sklearn": skl,
        "numpy": npv,
        "pandas": pdv,
    }

import streamlit as st

from config import MODELS, PATHS


class ModelNotAvailableError(RuntimeError):
    """Raised when a required model artefact is missing."""


def _friendly_error(artifacts: List[str], base_dir: Path) -> str:
    """Build a friendly error message listing missing artifacts."""
    lines = [
        "Required tabular model artifacts were not found.",
        "",
        "Expected files:",
    ]
    for a in artifacts:
        lines.append(f"  - {base_dir / a}")
    lines.extend([
        "",
        "To enable predictions, place your trained tabular model in the artifacts folder:",
        f"  - {base_dir / 'model.joblib'} (or model_tabular_multioutput.joblib)",
        f"  - {base_dir / 'model_meta.json'}",
        "You may also put these inside a subdirectory such as",
        f"  {base_dir / 'crimesense_tabular_only'}/" ,
    ])
    return "\n".join(lines)


@st.cache_resource(show_spinner=False)
def load_tabular_model() -> Dict[str, Any]:
    """Load tabular artefacts and metadata.

    **Portable format (preferred)**
    - ``artifacts/tabular/portable/`` should contain:
      ``feature_columns.json``, ``label_encoders.joblib`` and
      ``xgb_<target>.json`` for every target named in ``model_meta.json``.
      A small ``model_meta.json`` (features/targets lists) is also required.
    - Returns a dict with keys ``portable`` (True), ``models`` (mapping of
      target->XGBClassifier), ``label_encoders``, ``features``, ``targets``
      and ``feature_columns`` along with ``version``.

    **Legacy joblib support is intentionally removed.**
    Attempting to load the old ``.joblib`` file will surface a friendly error
    instructing users to export portable artefacts from Colab; the UI will
    display this message via ``ModelNotAvailableError``.
    """
    base_dir: Path = PATHS.tabular_model_dir
    portable_dir = base_dir / "portable"

    # portable path takes precedence
    if portable_dir.exists():
        # check for required files
        missing: list[str] = []
        meta_path = portable_dir / "model_meta.json"
        if not meta_path.exists():
            missing.append("model_meta.json")
        feat_col_path = portable_dir / "feature_columns.json"
        if not feat_col_path.exists():
            missing.append("feature_columns.json")
        enc_path = portable_dir / "label_encoders.joblib"
        if not enc_path.exists():
            missing.append("label_encoders.joblib")
        if missing:
            raise ModelNotAvailableError(
                "Portable tabular artefacts incomplete. "
                f"Missing: {missing}\n"
                f"Expected files under {portable_dir}" )

        # load metadata
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        features = meta.get("features", []) if isinstance(meta, dict) else []
        targets = meta.get("targets", []) if isinstance(meta, dict) else []

        # load encoders and feature columns
        import joblib
        label_encoders = joblib.load(enc_path)
        with feat_col_path.open("r", encoding="utf-8") as f:
            feature_columns = json.load(f)

        # load each xgboost model
        try:
            import xgboost as xgb
        except ImportError as e:  # pragma: no cover - handled later
            raise ModelNotAvailableError(
                "xgboost is not installed; cannot load portable models. "
                "Install xgboost and restart." ) from e

        models: dict[str, Any] = {}
        for tgt in targets:
            model_file = portable_dir / f"xgb_{tgt}.json"
            if not model_file.exists():
                raise ModelNotAvailableError(
                    f"Missing portable model file for target '{tgt}': {model_file}"
                )
            clf = xgb.XGBClassifier()
            clf.load_model(str(model_file))
            models[tgt] = clf

        return {
            "portable": True,
            "models": models,
            "label_encoders": label_encoders,
            "features": features,
            "targets": targets,
            "feature_columns": feature_columns,
            "version": MODELS.tabular_model_version,
        }

    # fall back to human-friendly error rather than trying to unpickle a joblib
    # model which will likely fail due to sklearn mismatches on the user's PC.
    raise ModelNotAvailableError(
        "Tabular joblib model cannot be loaded on this machine. "
        "Please export portable XGBoost artefacts from Colab and place them "
        f"under {portable_dir} (see README)." )


def prepare_tabular_row(case_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Convert the incoming dict into a single-row feature vector and analyse
    how many values were provided.

    The returned tuple is ``(row, summary)`` where ``row`` is an ordered
    mapping of every feature name to a numeric value (defaults to ``0``) and
    ``summary`` contains diagnostics useful for debugging or logging:

    ``summary`` keys:
      * ``filled`` – count of features whose value was not ``0``/empty
      * ``defaulted`` – count filled with the default ``0``
      * ``missing`` – list of up to 20 features that were defaulted.

    The column order respects the ``features`` list provided in
    ``model_meta.json`` so that downstream code can comfortably build a
    NumPy/Pandas row without worrying about ordering.
    """
    try:
        info = load_tabular_model()
        features = info.get("features", [])
    except ModelNotAvailableError:
        # fallback to using whatever keys are present (mock situations)
        features = list(case_dict.keys()) if case_dict else []

    row: Dict[str, Any] = {}
    filled = 0
    missing: List[str] = []
    for col in features:
        val = case_dict.get(col, 0) or 0
        row[col] = val
        if val in (0, "", None):
            missing.append(col)
        else:
            filled += 1
    summary = {
        "filled": filled,
        "defaulted": len(features) - filled,
        "missing": missing[:20],
    }
    return row, summary


def predict_tabular(case_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run tabular model inference for a single case dictionary.

    Returns a dictionary with the following structure:

    ```json
    {
      "pred": {"target1": "label", ...},
      "meta": {"targets": [...], "features": [...], "source": "tabular"},
      "model_version": "...",
      "raw_features": {...},
      "shap_top": [("feature", value), ...]  # optional
    }
    ```

    The `shap_top` list is only produced when the real model is available and
    a SHAP explainer can be computed; it is omitted for mocks or if an error
    occurs.
    """
    try:
        info = load_tabular_model()
        features = info.get("features", [])
        targets = info.get("targets", [])
        version = info.get("version")

        # prepare input row from case_dict (missing features default to 0)
        row, row_summary = prepare_tabular_row(case_dict)

        # portable model branch
        if info.get("portable"):
            import pandas as pd
            # construct one‑row DataFrame and one-hot encode
            df = pd.DataFrame([row])
            df = pd.get_dummies(df)
            cols = info.get("feature_columns", []) or []
            df = df.reindex(columns=cols, fill_value=0)

            # run each target through its classifier and decode labels
            pred_dict: Dict[str, Any] = {}
            models = info.get("models", {})
            encoders = info.get("label_encoders", {})
            X = df.values
            for tgt in targets:
                clf = models.get(tgt)
                if clf is None:
                    pred_dict[tgt] = "Unknown"
                    continue
                num = clf.predict(X)[0]
                le = encoders.get(tgt)
                if le is not None:
                    try:
                        label = le.inverse_transform([num])[0]
                    except Exception:
                        label = str(num)
                else:
                    label = str(num)
                pred_dict[tgt] = label

            output: Dict[str, Any] = {
                "pred": pred_dict,
                "meta": {"targets": targets, "features": features, "source": "tabular"},
                "model_version": version,
                "raw_features": case_dict,
                "input_summary": row_summary,
            }
            return output

        # non-portable path should never be reached because load_tabular_model
        # now raises if portability is not available, but keep sanity check.
        model = info.get("model")
        if model is None:
            raise ModelNotAvailableError("No tabular model loaded.")

        # legacy joblib inference (fallback) -- kept minimal just in case
        X_row = [row.get(c, 0) for c in features]
        try:
            import numpy as np
            X = np.array(X_row).reshape(1, -1)
        except Exception:
            X = [X_row]

        preds = None
        if hasattr(model, "predict"):
            preds = model.predict(X)
        elif hasattr(model, "predict_proba"):
            preds = model.predict_proba(X)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()

        pred_dict: Dict[str, Any] = {}
        if preds is not None:
            first = preds[0] if preds else []
            if isinstance(first, (list, tuple)):
                vals = list(first)
            else:
                vals = [first]
            for tgt, val in zip(targets, vals):
                pred_dict[tgt] = str(val)
        for tgt in targets:
            pred_dict.setdefault(tgt, "Unknown")

        output = {
            "pred": pred_dict,
            "meta": {"targets": targets, "features": features, "source": "tabular"},
            "model_version": version,
            "raw_features": case_dict,
            "input_summary": row_summary,
        }
        return output

    except ModelNotAvailableError as e:
        # do not silently fallback to a mock; propagate the error so callers can
        # display it clearly. this prevents constant, meaningless predictions
        # when the real model is absent.
        raise


__all__ = ["ModelNotAvailableError", "load_tabular_model", "prepare_tabular_row", "predict_tabular"]
