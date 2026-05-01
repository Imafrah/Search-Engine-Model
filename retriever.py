"""
retriever.py - BM25 retrieval over the document corpus.
"""

import re
import logging
from typing import List, Dict
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "for",
    "of", "in", "on", "at", "to", "and", "or", "but",
    "it", "its", "this", "that",
})


def _stem(token: str) -> str:
    """Simple suffix-stripping stemmer — no external deps needed."""
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"    # batteries → battery
    if token.endswith("ers") and len(token) > 4:
        return token[:-1]          # computers → computer
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]          # ranking → rank
    if token.endswith("tion") and len(token) > 5:
        return token[:-4]          # information → informat
    if token.endswith("est") and len(token) > 4:
        return token[:-3]          # fastest → fast
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]          # ranked → rank
    if token.endswith("s") and len(token) > 3:
        return token[:-1]          # laptops → laptop, monitors → monitor
    return token


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.split(r"[^a-z0-9]+", text)
    tokens = [t for t in tokens if len(t) >= 2 and t not in STOPWORDS]
    return [_stem(t) for t in tokens]


class BM25Retriever:
    def __init__(self, corpus: List[Dict]):
        self.corpus = corpus
        self._build_index()

    def _build_index(self):
        self.tokenized_docs = []
        for doc in self.corpus:
            # Title weighted 3x — more important than body
            combined = f"{doc['title']} {doc['title']} {doc['title']} {doc['text']}"
            self.tokenized_docs.append(tokenize(combined))
        self.bm25 = BM25Okapi(self.tokenized_docs)
        logger.info(f"BM25 index built over {len(self.corpus)} documents.")

    def retrieve(self, query: str, top_k: int = 200) -> List[Dict]:
        try:
            query_tokens = tokenize(query)
            if not query_tokens:
                return []
            scores = self.bm25.get_scores(query_tokens)

            # Take top_k by score — do NOT filter out zero scores here.
            # Zero-score docs are still valid candidates for LTR reranking.
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]

            results = []
            for idx in top_indices:
                doc = dict(self.corpus[idx])
                doc["bm25_score"] = float(scores[idx])
                results.append(doc)
            return results

        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}")
            return []