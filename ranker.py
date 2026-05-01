"""
ranker.py - LambdaMART reranking using a pre-trained LightGBM model.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ltr_model.lgb")


def load_model(model_path: Optional[str] = None):
    """Load and return the LightGBM Booster, or None on failure."""
    path = model_path or MODEL_PATH
    if not os.path.exists(path):
        logger.error(f"Model file not found: {path}")
        return None
    try:
        import lightgbm as lgb
        model = lgb.Booster(model_file=path)
        logger.info(f"LambdaMART model loaded ({model.num_feature()} features).")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def rerank(query: str, docs: List[Dict], features: np.ndarray,
           top_k: int = 10, model=None) -> List[Dict]:
    """Rerank docs using the LambdaMART model. Falls back to BM25 if no model."""
    if not docs:
        return []

    if model is None:
        for doc in docs:
            doc["ltr_score"] = doc.get("bm25_score", 0.0)
        return sorted(docs, key=lambda d: d["ltr_score"], reverse=True)[:top_k]

    try:
        n_docs = len(docs)
        expected = model.num_feature()

        if features.shape[1] < expected:
            pad = np.zeros((n_docs, expected - features.shape[1]), dtype=np.float32)
            features = np.hstack([features, pad])
        elif features.shape[1] > expected:
            features = features[:, :expected]

        scores = model.predict(features)

        scored = []
        for i, doc in enumerate(docs):
            d = dict(doc)
            d["ltr_score"] = float(scores[i])
            scored.append(d)

        scored.sort(key=lambda d: d["ltr_score"], reverse=True)
        return scored[:top_k]

    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        for doc in docs:
            doc["ltr_score"] = doc.get("bm25_score", 0.0)
        return sorted(docs, key=lambda d: d["ltr_score"], reverse=True)[:top_k]
