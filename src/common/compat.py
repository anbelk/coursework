from __future__ import annotations

from .artifacts import ensure_dirs, save_variant_artifacts
from .clustering_variants import (
    ALL_CLUSTERING_EVAL_VARIANTS,
    ALL_VARIANTS,
    BASELINE_K,
    BASELINE_K_GRID,
    BASELINE_VARIANT_NAMES,
    BASELINE_VARIANTS,
    FCM_VARIANTS,
    GMM_VARIANTS,
    HDBSCAN_VARIANTS,
    META_ALL_VARIANTS,
    META_FCM_VARIANTS,
    META_GMM_VARIANTS,
    META_HDBSCAN_VARIANTS,
    RANDOM_STATE,
    Variant,
    is_meta_baseline,
)
from .io import iter_jsonl, load_json, load_paper_ids, load_papers_by_id, load_papers_in_embedding_order, save_json
from .llm_cache import LLMCache, LLM_MODEL, LLM_VERSION
from .math import (
    cluster_sizes,
    hard_labels_from_proba,
    hdbscan_labels_from_proba,
    l2_normalize,
    minmax_score,
    representative_indices,
    weighted_centroids,
)
from .paths import (
    AUTHORS,
    CLUSTERING,
    CONFIGS,
    DATA,
    EMBEDDINGS_PATH,
    LLM_CACHE_PATH,
    MODELS,
    OUTPUTS,
    PAPER_IDS_PATH,
    PAPERS_PATH,
    PREDICTIONS,
    REDUCED,
    RESULTS,
    ROOT,
    TOPICS,
    authors_dir,
    clustering_dir,
    layout_dir,
    llm_cache_path,
    meta_dir,
    model_dir,
    paper_embeddings_path,
    paper_ids_path,
    papers_clean_path,
    predictions_dir,
    results_dir,
    topic_dir,
    topics_dir,
    variant_dir,
)
from .text import cyr_to_lat, compact_abstract, normalize_text, query_variants


def load_embeddings():
    import numpy as np

    arr = np.load(EMBEDDINGS_PATH).astype(np.float32, copy=False)
    return l2_normalize(arr)
