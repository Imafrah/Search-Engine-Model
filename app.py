"""
app.py — Streamlit Search Engine (LambdaMART + BM25)
Multi-source corpus: Wikipedia · Reddit · DuckDuckGo · BEIR
"""

import os
import sys
import time
import streamlit as st
import pandas as pd

from corpus import build_corpus, rebuild_corpus, get_corpus_stats, CORPUS_FILE
from retriever import BM25Retriever, tokenize
from features import extract_features
from ranker import load_model, rerank

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="LTR Search Engine", page_icon="🔍", layout="centered")

# ── CLI rebuild support ──────────────────────────────────────────────
# Allows: streamlit run app.py -- --rebuild
if "--rebuild" in sys.argv:
    build_corpus(force_rebuild=True)

# ── Startup checks ───────────────────────────────────────────────────
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ltr_model.lgb")

if not os.path.exists(CORPUS_FILE):
    st.error("⚠️ `corpus.json` not found. Run `python corpus.py` locally first, then commit the file.")
    st.stop()

if not os.path.exists(MODEL_FILE):
    st.error("⚠️ `ltr_model.lgb` not found. Add the model file to the repo.")
    st.stop()


# ── Cached resources (load once) ─────────────────────────────────────
@st.cache_resource(show_spinner="Loading corpus…")
def load_corpus():
    return build_corpus()

@st.cache_resource(show_spinner="Building BM25 index…")
def build_bm25(_docs):
    return BM25Retriever(_docs)

@st.cache_resource(show_spinner="Loading LambdaMART model…")
def load_ltr_model():
    return load_model(MODEL_FILE)


corpus_data = load_corpus()
bm25_retriever = build_bm25(corpus_data)
ltr_model = load_ltr_model()

# ── Corpus stats ─────────────────────────────────────────────────────
corpus_stats = get_corpus_stats(corpus_data)

# ── Source badge config ──────────────────────────────────────────────
SOURCE_CONFIG = {
    "wikipedia": {"badge": "W", "color": "#3b82f6", "bg": "rgba(59,130,246,0.15)", "dot": "🔵", "label": "Wikipedia"},
    "reddit":    {"badge": "R", "color": "#f97316", "bg": "rgba(249,115,22,0.15)", "dot": "🟠", "label": "Reddit"},
    "ddg":       {"badge": "D", "color": "#22c55e", "bg": "rgba(34,197,94,0.15)",  "dot": "🟢", "label": "DuckDuckGo"},
    "beir":      {"badge": "B", "color": "#a855f7", "bg": "rgba(168,85,247,0.15)", "dot": "🟣", "label": "BEIR"},
}

# ── Query expansion ──────────────────────────────────────────────────
EXPANSIONS = {
    "laptop": ["notebook", "computer", "MacBook", "PC"],
    "phone": ["smartphone", "mobile", "iPhone", "Android"],
    "best": ["top", "recommended", "review"],
    "cheap": ["budget", "affordable"],
    "fast": ["performance", "speed"],
    "student": ["college", "university", "education", "school"],
    "gaming": ["game", "gamer", "GPU", "performance"],
    "monitor": ["display", "screen"],
    "keyboard": ["mechanical", "typing"],
    "headphone": ["headset", "earphone", "audio"],
}

def expand_query(query: str) -> str:
    tokens = query.lower().split()
    expanded = list(tokens)
    for t in tokens:
        if t in EXPANSIONS:
            expanded.extend(EXPANSIONS[t])
    return " ".join(set(expanded))


# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .result-card {
        background: #1A1D23;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 12px;
        transition: border-color 0.2s;
    }
    .result-card:hover {
        border-color: rgba(79,142,247,0.3);
    }
    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 8px;
        background: rgba(79,142,247,0.15);
        color: #4F8EF7;
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 12px;
        vertical-align: middle;
    }
    .source-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.68rem;
        font-weight: 700;
        margin-left: 8px;
        vertical-align: middle;
        letter-spacing: 0.5px;
    }
    .result-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #818cf8;
        text-decoration: none;
    }
    .result-title:hover {
        color: #a5b4fc;
        text-decoration: underline;
    }
    .result-url {
        font-size: 0.78rem;
        color: #34d399;
        margin-top: 2px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .result-snippet {
        font-size: 0.88rem;
        color: #9d9daa;
        line-height: 1.6;
        margin-top: 8px;
    }
    .score-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 999px;
        margin-top: 10px;
        letter-spacing: 0.5px;
    }
    .score-high { background: rgba(52,211,153,0.12); color: #34d399; }
    .score-mid  { background: rgba(251,191,36,0.12); color: #fbbf24; }
    .score-low  { background: rgba(248,113,113,0.12); color: #f87171; }
    .debug-row {
        font-size: 0.72rem;
        color: #6b6b78;
        font-family: 'Courier New', monospace;
        margin-top: 8px;
        padding: 8px 12px;
        background: rgba(79,142,247,0.05);
        border-radius: 6px;
    }
    .debug-row span { color: #4F8EF7; font-weight: 600; }
    .corpus-bar {
        height: 8px;
        border-radius: 4px;
        margin-bottom: 4px;
    }
    .corpus-stat-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
        font-size: 0.85rem;
    }
    .corpus-stat-count {
        font-weight: 600;
        color: #e2e8f0;
        margin-left: auto;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k_candidates = st.slider("BM25 candidates", 50, 300, 200)
    top_n_results = st.slider("Results to show", 5, 20, 10)
    show_debug = st.toggle("Show debug info", value=False)

    st.divider()

    # ── Corpus Sources breakdown ──
    st.subheader("📊 Corpus Sources")
    total = corpus_stats["total"]
    by_source = corpus_stats.get("by_source", {})

    # Bar chart visualization
    for src_key in ["wikipedia", "reddit", "ddg", "beir"]:
        cfg = SOURCE_CONFIG.get(src_key, {})
        count = by_source.get(src_key, 0)
        pct = (count / total * 100) if total > 0 else 0
        bar_width = max(pct, 2)  # minimum visible width

        st.markdown(
            f'<div class="corpus-stat-row">'
            f'<span>{cfg.get("dot", "⚪")}</span>'
            f'<span style="color:{cfg.get("color", "#fff")}">{cfg.get("label", src_key)}</span>'
            f'<span class="corpus-stat-count">{count}</span>'
            f'</div>'
            f'<div class="corpus-bar" style="width:{bar_width}%;background:{cfg.get("color", "#666")}"></div>',
            unsafe_allow_html=True,
        )

    st.markdown(f"**Total: {total} documents**")

    st.divider()

    st.markdown("**NDCG@10:** 0.8665")
    st.markdown("**Training:** MSLR-WEB10K (all 5 folds)")
    st.markdown("**Best iteration:** 272 trees")

    st.divider()

    # ── Rebuild button ──
    if st.button("🔄 Rebuild Corpus", help="Re-fetch all sources (takes 5-8 min)"):
        with st.spinner("Rebuilding corpus from all sources…"):
            rebuild_corpus()
        st.cache_resource.clear()
        st.rerun()

# ── Header ───────────────────────────────────────────────────────────
st.title("🔍 LTR Search Engine")
st.caption("LambdaMART-powered · Multi-source corpus · BM25 + LTR reranking")

# ── Session state for query ──────────────────────────────────────────
if "query" not in st.session_state:
    st.session_state.query = ""

def set_query(q):
    st.session_state.query = q

# ── Search input ─────────────────────────────────────────────────────
query = st.text_input(
    "Search",
    value=st.session_state.query,
    placeholder="Type a query and press Enter…",
    label_visibility="collapsed",
    key="search_input",
)

# Example chips
cols = st.columns(4)
chips = [
    "best laptop for students",
    "python machine learning",
    "how does deep learning work",
    "best gaming monitor",
]
for col, chip in zip(cols, chips):
    if col.button(chip, use_container_width=True):
        set_query(chip)
        st.rerun()

# ── Search execution ─────────────────────────────────────────────────
active_query = query.strip() or st.session_state.query.strip()

if active_query:
    expanded = expand_query(active_query)

    with st.spinner("Searching…"):
        # BM25 retrieval
        t0 = time.time()
        candidates = bm25_retriever.retrieve(expanded, top_k=top_k_candidates)
        retrieval_ms = (time.time() - t0) * 1000

        if not candidates:
            st.warning(f'No results found for "{active_query}"')
        else:
            # Feature extraction + reranking
            t1 = time.time()
            feats = extract_features(active_query, candidates)
            ranked = rerank(active_query, candidates, feats,
                            top_k=top_n_results, model=ltr_model)
            ranking_ms = (time.time() - t1) * 1000
            total_ms = retrieval_ms + ranking_ms

            # Timing bar
            st.success(
                f"**{len(ranked)}** results from {len(candidates)} candidates "
                f"— BM25 {retrieval_ms:.0f}ms · LTR {ranking_ms:.0f}ms · "
                f"Total {total_ms:.0f}ms"
            )

            # Query term set for debug
            qt_set = set(tokenize(active_query))

            # Render results
            for r_idx, doc in enumerate(ranked, 1):
                score = doc.get("ltr_score", 0.0)
                sc = "score-high" if score > 0.7 else ("score-mid" if score > 0.3 else "score-low")
                snippet = doc.get("text", "")[:250]
                if len(doc.get("text", "")) > 250:
                    snippet += "…"

                # Source badge
                src = doc.get("source", "unknown")
                cfg = SOURCE_CONFIG.get(src, {"badge": "?", "color": "#666", "bg": "rgba(100,100,100,0.15)"})
                source_badge_html = (
                    f'<span class="source-badge" style="'
                    f'background:{cfg["bg"]};color:{cfg["color"]}">'
                    f'{cfg["badge"]}</span>'
                )

                card_html = f"""
                <div class="result-card">
                    <span class="rank-badge">#{r_idx}</span>
                    <a href="{doc.get('url','')}" target="_blank" class="result-title">{doc.get('title','Untitled')}</a>
                    {source_badge_html}
                    <div class="result-url">{doc.get('url','')}</div>
                    <div class="result-snippet">{snippet}</div>
                    <span class="score-badge {sc}">LTR: {score:.4f}</span>
                """

                if show_debug:
                    t_terms = len(qt_set & set(tokenize(doc.get("title", ""))))
                    b_terms = len(qt_set & set(tokenize(doc.get("text", ""))))
                    card_html += f"""
                    <div class="debug-row">
                        <span>BM25:</span> {doc.get('bm25_score',0):.4f} ·
                        <span>LTR:</span> {score:.6f} ·
                        <span>Query→Title:</span> {t_terms} ·
                        <span>Query→Body:</span> {b_terms} ·
                        <span>Source:</span> {src}
                    </div>
                    """

                card_html += "</div>"
                st.markdown(card_html, unsafe_allow_html=True)

            # Debug expander
            with st.expander("🔧 Debug details"):
                debug_rows = []
                for r_idx, doc in enumerate(ranked, 1):
                    debug_rows.append({
                        "rank": r_idx,
                        "title": doc.get("title", ""),
                        "source": doc.get("source", "?"),
                        "bm25_score": round(doc.get("bm25_score", 0), 4),
                        "ltr_score": round(doc.get("ltr_score", 0), 6),
                        "query_in_title": len(qt_set & set(tokenize(doc.get("title", "")))),
                        "query_in_body": len(qt_set & set(tokenize(doc.get("text", "")))),
                    })
                st.dataframe(pd.DataFrame(debug_rows), use_container_width=True)
                st.json({
                    "corpus_size": len(corpus_data),
                    "by_source": corpus_stats.get("by_source", {}),
                    "candidates_retrieved": len(candidates),
                    "model_loaded": ltr_model is not None,
                    "semantic_features": False,
                    "expanded_query": expanded,
                })
