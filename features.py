"""
features.py - 136-feature extraction for (query, document) pairs.
USE_SEMANTIC = False — no torch/sentence-transformers needed.
"""

import math
import logging
import numpy as np
from typing import List, Dict
from collections import Counter

logger = logging.getLogger(__name__)

USE_SEMANTIC = False


def load_embedding_model():
    """No-op when USE_SEMANTIC is False."""
    if not USE_SEMANTIC:
        logger.info("Semantic embeddings DISABLED.")
        return


def _tokenize_simple(text: str) -> List[str]:
    import re
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) >= 1]


def _compute_tf(tokens: List[str]) -> Dict[str, float]:
    counter = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {term: count / total for term, count in counter.items()}


def _compute_idf(term: str, all_docs_tokens: List[List[str]]) -> float:
    n = len(all_docs_tokens)
    if n == 0:
        return 0.0
    df = sum(1 for dt in all_docs_tokens if term in set(dt))
    return math.log((n + 1) / (df + 1)) + 1 if df > 0 else 0.0


def _char_overlap(a: str, b: str) -> float:
    a_c, b_c = set(a.lower()), set(b.lower())
    union = a_c | b_c
    return len(a_c & b_c) / len(union) if union else 0.0


def _jaccard(sa: set, sb: set) -> float:
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def _lcs_word(a: List[str], b: List[str]) -> int:
    a, b = a[:50], b[:50]
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            curr[j] = prev[j - 1] + 1 if a[i-1] == b[j-1] else max(prev[j], curr[j-1])
        prev, curr = curr, [0] * (n + 1)
    return max(prev)


def extract_features(query: str, docs: List[Dict]) -> np.ndarray:
    n_docs = len(docs)
    if n_docs == 0:
        return np.zeros((0, 136), dtype=np.float32)

    features = np.zeros((n_docs, 136), dtype=np.float32)
    query_tokens = _tokenize_simple(query)
    query_set = set(query_tokens)
    query_len = len(query_tokens)

    doc_title_tokens = [_tokenize_simple(d.get("title", "")) for d in docs]
    doc_body_tokens = [_tokenize_simple(d.get("text", "")) for d in docs]

    query_idf = {t: _compute_idf(t, doc_body_tokens) for t in query_set}

    for i, doc in enumerate(docs):
        try:
            feat = np.zeros(136, dtype=np.float32)
            tt = doc_title_tokens[i]
            bt = doc_body_tokens[i]
            ts, bs = set(tt), set(bt)
            ttf, btf = _compute_tf(tt), _compute_tf(bt)
            bm25 = doc.get("bm25_score", 0.0)

            # Group 1: BM25/TF-IDF (0-19)
            feat[0] = bm25
            feat[1] = sum(ttf.get(t, 0) for t in query_tokens)
            feat[2] = sum(btf.get(t, 0) for t in query_tokens)
            feat[3] = float(np.mean([query_idf.get(t, 0) for t in query_tokens])) if query_tokens else 0
            feat[4] = sum(ttf.get(t, 0) * query_idf.get(t, 0) for t in query_tokens)
            feat[5] = sum(btf.get(t, 0) * query_idf.get(t, 0) for t in query_tokens)
            feat[6] = len(query_set & ts) / len(query_set) if query_set else 0
            feat[7] = len(query_set & bs) / len(query_set) if query_set else 0
            feat[8] = 1.0 if query.lower() in doc.get("title", "").lower() else 0
            feat[9] = 1.0 if query.lower() in doc.get("text", "").lower() else 0
            feat[10] = float(query_len)
            feat[11] = float(len(bt))
            feat[12] = float(len(tt))
            qtfs = [btf.get(t, 0) for t in query_tokens] if query_tokens else [0]
            feat[13], feat[14], feat[15], feat[16] = min(qtfs), max(qtfs), float(np.mean(qtfs)), sum(qtfs)
            feat[17] = len(query_set & ts) / len(ts) if ts else 0
            feat[18] = len(query_set & bs) / len(bs) if bs else 0
            feat[19] = sum(ttf.get(t, 0) * query_idf.get(t, 0) * 2.0 for t in query_tokens)

            # Group 2: Fast text-overlap (20-49)
            dt, db = doc.get("title", ""), doc.get("text", "")
            feat[20] = _char_overlap(query, dt)
            feat[21] = _char_overlap(query, db[:500])
            feat[22] = _jaccard(query_set, ts)
            feat[23] = _jaccard(query_set, bs)
            feat[24] = float(len(query_set & ts))
            feat[25] = float(len(query_set & bs))
            feat[26] = float(_lcs_word(query_tokens, tt))
            feat[27] = float(_lcs_word(query_tokens, bt[:100]))
            feat[28] = feat[26] / (query_len + 1e-8)
            feat[29] = feat[27] / (query_len + 1e-8)
            if query_len >= 2:
                qb = set(zip(query_tokens[:-1], query_tokens[1:]))
                tb = set(zip(tt[:-1], tt[1:])) if len(tt) >= 2 else set()
                bb = set(zip(bt[:100], bt[1:101])) if len(bt) >= 2 else set()
                feat[30] = len(qb & tb) / len(qb) if qb else 0
                feat[31] = len(qb & bb) / len(qb) if qb else 0
            feat[32] = len(query_set & ts) / (len(ts) + 1e-8)
            feat[33] = len(query_set & bs) / (len(bs) + 1e-8)

            # Group 3: Doc quality (50-79)
            dl = len(bt)
            feat[50] = 0 if dl < 100 else (1 if dl < 300 else 2)
            feat[51] = float(len(dt))
            feat[52] = 1.0 if any(c.isdigit() for c in dt) else 0
            feat[53] = 1.0 if any(t in {"how","what","why","when","where","which","who"} for t in _tokenize_simple(dt)) else 0
            ac = [c for c in dt if c.isalpha()]
            feat[54] = sum(1 for c in ac if c.isupper()) / len(ac) if ac else 0
            feat[55] = len(set(bt)) / len(bt) if bt else 0
            bc = Counter(bt)
            feat[56] = float(np.mean(list(bc.values()))) if bc else 0
            feat[57] = 1.0 if bt and bt[0] in query_set else 0
            feat[58] = 1.0 if tt and tt[0] in query_set else 0
            feat[59] = float(db.count(".") + db.count("!") + db.count("?"))
            sents = [s.strip() for s in db.split(".") if s.strip()]
            feat[60] = float(np.mean([len(s.split()) for s in sents])) if sents else 0
            feat[61] = 1.0 if "http" in db.lower() else 0
            feat[62] = 1.0 if any(m in db for m in ["\n-", "\n*", "\n1."]) else 0
            feat[63] = float(db.count("\n\n") + 1)
            sw = {"the","a","an","is","was","are","were","be","been","being","have","has","had","do","does","did","will","would","could","should","may","might","can","shall","of","in","to","for","with","on","at","by","from","and","or","but","not","no","this","that","it"}
            feat[64] = sum(1 for t in bt if t in sw) / len(bt) if bt else 0
            feat[65] = len(ts & bs) / len(ts) if ts else 0
            if bc and bt:
                probs = np.array(list(bc.values()), dtype=np.float32) / len(bt)
                feat[66] = -float(np.sum(probs * np.log2(probs + 1e-10)))
            feat[67] = float(max(bc.values())) if bc else 0
            feat[68] = _jaccard(query_set, ts)
            feat[69] = _jaccard(query_set, bs)
            feat[70] = math.log1p(dl)
            feat[71] = math.log1p(len(tt))
            feat[72] = query_len / (dl + 1)
            feat[73] = float(len(query_set & ts))
            feat[74] = float(len(query_set & bs))
            feat[75] = 1.0 if query_set.issubset(ts) else 0
            feat[76] = 1.0 if query_set.issubset(bs) else 0
            feat[77] = float(len(db))
            feat[78] = float(np.mean([len(t) for t in bt])) if bt else 0
            feat[79] = sum(1 for c in db if c.isdigit()) / (len(db) + 1)

            # Group 4: Derived (80-135)
            feat[80] = bm25 ** 2
            feat[81] = math.log1p(bm25)
            feat[82] = feat[20] ** 2
            feat[83] = feat[21] ** 2
            feat[84] = bm25 * feat[20]
            feat[85] = bm25 * feat[21]
            feat[86] = bm25 * feat[6]
            feat[87] = bm25 * feat[7]
            feat[88] = feat[20] * feat[6]
            feat[89] = feat[21] * feat[7]
            feat[90] = math.log1p(feat[4])
            feat[91] = math.log1p(feat[5])
            feat[92] = math.sqrt(max(0, bm25))
            feat[93] = bm25 ** 3
            feat[94] = 1.0 / (dl + 1)
            feat[95] = bm25 / (dl + 1)
            feat[96] = feat[4] * feat[3]
            feat[97] = feat[5] * feat[3]
            feat[98] = feat[6] * feat[7]
            feat[99] = min(feat[6], feat[7])
            feat[100] = max(feat[6], feat[7])
            feat[101] = feat[8] + feat[9]
            feat[102] = bm25 * feat[8]
            feat[103] = bm25 * feat[9]
            feat[104] = feat[20] * feat[8]
            feat[105] = feat[21] * feat[9]
            feat[106] = feat[4] / (query_len + 1)
            feat[107] = feat[5] / (query_len + 1)
            feat[108] = feat[70] * bm25
            feat[109] = feat[66] * bm25
            feat[110] = feat[55] * bm25
            feat[111] = feat[64] * bm25
            feat[112] = feat[60] * feat[6]
            feat[113] = feat[63] / (dl + 1)
            feat[114] = bm25 * feat[68]
            feat[115] = bm25 * feat[69]
            feat[116] = feat[20] * feat[68]
            feat[117] = feat[21] * feat[69]
            feat[118] = feat[4] + feat[5]
            feat[119] = feat[1] * feat[2]
            feat[120] = float(i) / (n_docs + 1)
            feat[121] = 1.0 / (i + 1)
            feat[122] = math.log1p(i)
            feat[123] = len(tt) / (len(bt) + 1)
            feat[124] = query_len / (len(tt) + 1)
            ct, cb = feat[6], feat[7]
            feat[125] = 2 * ct * cb / (ct + cb) if (ct + cb) > 0 else 0
            feat[126] = math.sqrt(max(0, bm25 * feat[20]))
            feat[127] = 0.6 * bm25 + 0.4 * feat[20]
            feat[128] = bm25 * len(tt)
            feat[129] = feat[20] * feat[55]
            feat[130] = feat[6] * feat[66]
            feat[131] = feat[16] * feat[3]
            feat[132] = feat[14] * feat[6]
            feat[133] = bm25 * feat[78]
            feat[134] = feat[20] * feat[63]
            feat[135] = 0.3*feat[0] + 0.2*feat[20] + 0.2*feat[21] + 0.15*feat[6] + 0.15*feat[7]

            features[i] = feat
        except Exception as e:
            logger.warning(f"Feature extraction failed for doc {i}: {e}")

    return features
