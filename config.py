from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"  # fallback if artifacts/ missing
INDICES_DIR = BASE_DIR / "indices"


@dataclass(frozen=True)
class Paths:
    data_dir: Path = DATA_DIR
    cases_file: Path = DATA_DIR / "cases.jsonl"
    evidence_index_file: Path = INDICES_DIR / "cases_tfidf.pkl"
    tabular_model_dir: Path = ARTIFACTS_DIR / "tabular"
    nlp_model_dir: Path = ARTIFACTS_DIR / "nlp" / "model"


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model artefacts and versioning.

    Replace the default values with your actual model files when ready.
    """

    tabular_model_version: str = "tabular-mock-v0"
    nlp_model_version: str = "nlp-mock-v0"
    fusion_version: str = "fusion-mock-v0"
    dataset_version: str = "dataset-demo-v0"


PATHS = Paths()
MODELS = ModelConfig()


def ensure_dirs() -> None:
    """Create local directories used for data and indices if missing."""
    PATHS.data_dir.mkdir(parents=True, exist_ok=True)
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)



# RAG index defaults ----------------------------------------------------------
# location where users are expected to place their historical cases dataset
RAG_DATA_PATH = DATA_DIR / "cases.csv"
# limit on rows to index during interactive builds (None means no limit)
RAG_MAX_ROWS = 100000

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "ARTIFACTS_DIR",
    "MODELS_DIR",
    "INDICES_DIR",
    "PATHS",
    "MODELS",
    "ensure_dirs",
    "RAG_DATA_PATH",
    "RAG_MAX_ROWS",
]

