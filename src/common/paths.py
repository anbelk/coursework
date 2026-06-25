from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
REDUCED = DATA / "reduced"
CLUSTERING = DATA / "clustering"
TOPICS = DATA / "topics"
AUTHORS = DATA / "authors"
PREDICTIONS = DATA / "predictions"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
CONFIGS = ROOT / "configs"

EMBEDDINGS_PATH = DATA / "embeddings.npy"
PAPER_IDS_PATH = DATA / "paper_ids.json"
PAPERS_PATH = DATA / "openalex_clean.jsonl"
LLM_CACHE_PATH = DATA / "llm_cache.sqlite"

RAW_DIR = DATA / "raw"
PROCESSED_DIR = DATA / "processed"
ARTIFACTS_DIR = DATA / "artifacts"
ARTIFACT_EMBEDDINGS = ARTIFACTS_DIR / "embeddings"
ARTIFACT_CLUSTERING = ARTIFACTS_DIR / "clustering"
ARTIFACT_AUTHORS = ARTIFACTS_DIR / "authors"
ARTIFACT_UI = ARTIFACTS_DIR / "ui"


def variant_dir(name: str) -> Path:
    return CLUSTERING / name


def topic_dir(name: str) -> Path:
    return TOPICS / name


def model_dir(name: str) -> Path:
    return MODELS / name


def paper_embeddings_path() -> Path:
    return EMBEDDINGS_PATH


def papers_clean_path() -> Path:
    return PAPERS_PATH


def paper_ids_path() -> Path:
    return PAPER_IDS_PATH


def clustering_dir(variant: str) -> Path:
    return variant_dir(variant)


def topics_dir(variant: str) -> Path:
    return topic_dir(variant)


def predictions_dir() -> Path:
    return PREDICTIONS


def layout_dir() -> Path:
    return DATA / "layout"


def meta_dir() -> Path:
    return DATA / "meta"


def authors_dir() -> Path:
    return AUTHORS


def results_dir() -> Path:
    return RESULTS


def llm_cache_path() -> Path:
    return LLM_CACHE_PATH


def new_paper_embeddings_path() -> Path:
    return ARTIFACT_EMBEDDINGS / "paper_embeddings.npy"


def new_meta_cluster_embeddings_path() -> Path:
    return ARTIFACT_EMBEDDINGS / "meta_cluster_embeddings.npy"


def new_papers_clustering_dir(variant: str) -> Path:
    return ARTIFACT_CLUSTERING / "papers" / variant


def new_meta_clustering_dir() -> Path:
    return ARTIFACT_CLUSTERING / "meta"


def new_authors_artifacts_dir() -> Path:
    return ARTIFACT_AUTHORS


def new_ui_artifacts_dir() -> Path:
    return ARTIFACT_UI
