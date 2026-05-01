"""
corpus.py - Multi-source document corpus builder.

Fetches documents from 4 real sources:
  1. Wikipedia (category API → full articles)
  2. Reddit (public JSON API)
  3. DuckDuckGo (web search snippets)
  4. BEIR (MS MARCO tech passages only — SciFact removed)

Run standalone:  python corpus.py
Expected output: 800-1200 documents in ~5-8 minutes.

Changes from previous version:
  - REMOVED SciFact entirely (biomedical papers, useless for tech search)
  - MS MARCO now filtered to tech-relevant passages only
  - MS MARCO limit increased: scans 5000, keeps up to 1000 tech-relevant ones
"""

import json
import os
import hashlib
import logging
import time
import re
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

CORPUS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.json")


# ─────────────────────────────────────────────────────────────────────
# SOURCE 1: WIKIPEDIA  (target: 400 docs)
# ─────────────────────────────────────────────────────────────────────

SEED_CATEGORIES = [
    "Laptops", "Smartphones", "Tablet_computers",
    "Computer_hardware", "Audio_equipment", "Computer_monitors",
    "Programming_languages", "Machine_learning", "Artificial_intelligence",
    "Computer_peripherals", "Video_game_hardware", "Computer_storage",
]


def _discover_titles_from_categories() -> List[str]:
    """Use Wikipedia's category API to dynamically discover article titles."""
    titles = []
    seen = set()
    session = requests.Session()
    session.headers.update({"User-Agent": "LTR-SearchEngine/1.0"})

    for cat in SEED_CATEGORIES:
        try:
            url = (
                f"https://en.wikipedia.org/w/api.php?action=query"
                f"&list=categorymembers&cmtitle=Category:{cat}"
                f"&cmlimit=40&cmtype=page&format=json"
            )
            resp = session.get(url, timeout=10)
            resp.raise_for_status()
            members = resp.json().get("query", {}).get("categorymembers", [])
            for m in members:
                title = m.get("title", "")
                if title and title not in seen:
                    seen.add(title)
                    titles.append(title)
        except Exception as e:
            logger.warning(f"Category fetch failed for {cat}: {e}")
            continue

    logger.info(f"Discovered {len(titles)} unique titles from {len(SEED_CATEGORIES)} categories")
    return titles


def _fetch_wiki_article(title: str) -> Dict | None:
    """Fetch a single Wikipedia article by exact title."""
    try:
        import wikipedia
        wikipedia.set_lang("en")
        page = wikipedia.page(title, auto_suggest=False)
        text = page.content[:2000]
        if len(text) < 100:
            return None
        return {
            "doc_id": f"wiki_{hashlib.md5(page.title.encode()).hexdigest()[:10]}",
            "title": page.title,
            "text": text,
            "url": page.url,
            "source": "wikipedia",
        }
    except Exception:
        return None


def fetch_wikipedia() -> List[Dict]:
    """Discover topics from Wikipedia categories, then fetch articles in parallel."""
    titles = _discover_titles_from_categories()
    if not titles:
        logger.error("No Wikipedia titles discovered — skipping source.")
        return []

    docs = []
    seen_ids = set()

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch_wiki_article, t): t for t in titles}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Wikipedia", unit="article"):
            try:
                doc = future.result()
                if doc and doc["doc_id"] not in seen_ids:
                    seen_ids.add(doc["doc_id"])
                    docs.append(doc)
            except Exception:
                continue

    logger.info(f"Wikipedia: fetched {len(docs)} articles")
    return docs


# ─────────────────────────────────────────────────────────────────────
# SOURCE 2: REDDIT  (target: 200 docs)
# ─────────────────────────────────────────────────────────────────────

SUBREDDITS = [
    "laptops", "SuggestALaptop", "buildapc", "hardware",
    "Android", "iphone", "smartphones", "phones",
    "MachineLearning", "learnprogramming", "Python",
    "headphones", "audiophile", "monitors", "MechanicalKeyboards",
    "college", "GradSchool", "students",
]

REDDIT_HEADERS = {"User-Agent": "LTR-SearchEngine/1.0 (educational project)"}


def _fetch_reddit_comments(permalink: str) -> str:
    """Fetch the top comment for a Reddit post."""
    try:
        url = f"https://www.reddit.com{permalink}.json?limit=1"
        resp = requests.get(url, headers=REDDIT_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if len(data) > 1:
            children = data[1].get("data", {}).get("children", [])
            if children:
                body = children[0].get("data", {}).get("body", "")
                return body
    except Exception:
        pass
    return ""


def _process_reddit_post(post_data: dict) -> Dict | None:
    """Process a single Reddit post into a document."""
    try:
        post = post_data.get("data", {})
        post_id = post.get("id", "")
        title = post.get("title", "")
        selftext = post.get("selftext", "")
        permalink = post.get("permalink", "")

        if not post_id or not title:
            return None

        text = selftext.strip()
        if len(text) < 100:
            comment = _fetch_reddit_comments(permalink)
            text = f"{title} {text} {comment}".strip()

        if len(text) < 80:
            return None

        text = text[:2000]

        return {
            "doc_id": f"reddit_{post_id}",
            "title": title,
            "text": text,
            "url": f"https://reddit.com{permalink}",
            "source": "reddit",
        }
    except Exception:
        return None


def fetch_reddit() -> List[Dict]:
    """Fetch hot + top posts from multiple subreddits via Reddit's public JSON API."""
    docs = []
    seen_ids = set()

    for sub in tqdm(SUBREDDITS, desc="Reddit", unit="sub"):
        try:
            for endpoint in ["hot", "top"]:
                params = "?limit=25&raw_json=1"
                if endpoint == "top":
                    params += "&t=month"
                url = f"https://www.reddit.com/r/{sub}/{endpoint}.json{params}"

                resp = requests.get(url, headers=REDDIT_HEADERS, timeout=15)
                if resp.status_code == 429:
                    logger.warning(f"Reddit rate limited on r/{sub}/{endpoint}, waiting 10s...")
                    time.sleep(10)
                    resp = requests.get(url, headers=REDDIT_HEADERS, timeout=15)
                resp.raise_for_status()

                posts = resp.json().get("data", {}).get("children", [])
                for p in posts:
                    doc = _process_reddit_post(p)
                    if doc and doc["doc_id"] not in seen_ids:
                        seen_ids.add(doc["doc_id"])
                        docs.append(doc)

            time.sleep(2)

        except Exception as e:
            logger.warning(f"Reddit fetch failed for r/{sub}: {e}")
            time.sleep(2)
            continue

    logger.info(f"Reddit: fetched {len(docs)} posts")
    return docs


# ─────────────────────────────────────────────────────────────────────
# SOURCE 3: DUCKDUCKGO  (target: 200 docs)
# ─────────────────────────────────────────────────────────────────────

DDG_QUERIES = [
    "best laptop for students 2024",
    "best budget smartphone 2024",
    "best gaming laptop under 1000",
    "best noise cancelling headphones",
    "python machine learning tutorial",
    "best gaming monitor 144hz",
    "best mechanical keyboard 2024",
    "iPhone vs Samsung which is better",
    "best tablet for college students",
    "RTX 4070 vs RX 7800 XT",
    "best wireless mouse for programming",
    "MacBook Air vs Dell XPS",
    "best SSD for gaming PC",
    "deep learning tutorial beginners",
    "best chromebook for students",
    "how to build a gaming PC",
    "best ultrawide monitor 2024",
    "best budget earbuds 2024",
    "ThinkPad vs MacBook for developers",
    "best phone camera 2024",
]


def fetch_ddg() -> List[Dict]:
    """Fetch search results from DuckDuckGo for curated queries."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.error("duckduckgo-search not installed — skipping DDG source.")
        return []

    docs = []
    seen_urls = set()

    for query in tqdm(DDG_QUERIES, desc="DuckDuckGo", unit="query"):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=10))

            for r in results:
                href = r.get("href", "")
                title = r.get("title", "")
                body = r.get("body", "")

                if not href or not body or len(body) < 30:
                    continue
                if href in seen_urls:
                    continue
                seen_urls.add(href)

                doc_id = f"ddg_{hashlib.md5(href.encode()).hexdigest()[:10]}"
                docs.append({
                    "doc_id": doc_id,
                    "title": title,
                    "text": body[:2000],
                    "url": href,
                    "source": "ddg",
                })

            time.sleep(1)

        except Exception as e:
            if "429" in str(e) or "Ratelimit" in str(e):
                logger.warning(f"DDG rate limited on '{query}', waiting 10s and retrying...")
                time.sleep(10)
                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.text(query, max_results=10))
                    for r in results:
                        href = r.get("href", "")
                        title = r.get("title", "")
                        body = r.get("body", "")
                        if not href or not body or len(body) < 30:
                            continue
                        if href in seen_urls:
                            continue
                        seen_urls.add(href)
                        doc_id = f"ddg_{hashlib.md5(href.encode()).hexdigest()[:10]}"
                        docs.append({
                            "doc_id": doc_id,
                            "title": title,
                            "text": body[:2000],
                            "url": href,
                            "source": "ddg",
                        })
                except Exception:
                    logger.warning(f"DDG retry also failed for '{query}', skipping.")
            else:
                logger.warning(f"DDG fetch failed for '{query}': {e}")
            continue

    logger.info(f"DuckDuckGo: fetched {len(docs)} results")
    return docs


# ─────────────────────────────────────────────────────────────────────
# SOURCE 4: BEIR — MS MARCO TECH PASSAGES ONLY
# SciFact removed: biomedical papers have zero relevance for tech search.
# MS MARCO contains real Bing query-document pairs — perfect for LTR.
# We scan up to 5000 passages and keep up to 1000 tech-relevant ones.
# ─────────────────────────────────────────────────────────────────────

# Keywords that must appear in a passage for it to be kept.
# Covers laptops, phones, monitors, ML/AI, software, gaming, and CS topics.
TECH_KEYWORDS = {
    "laptop", "laptops", "notebook", "ultrabook", "chromebook",
    "macbook", "thinkpad", "dell", "hp", "asus", "lenovo", "acer",
    "computer", "desktop", "pc", "processor", "cpu", "gpu", "graphics card",
    "ram", "memory", "ssd", "hard drive", "storage", "motherboard",
    "monitor", "display", "screen", "resolution", "refresh rate", "hz",
    "keyboard", "mouse", "webcam", "headphone", "earbuds", "speaker",
    "phone", "smartphone", "iphone", "android", "samsung", "pixel",
    "tablet", "ipad", "surface",
    "python", "javascript", "programming", "coding", "software",
    "machine learning", "deep learning", "neural network", "ai",
    "artificial intelligence", "data science", "algorithm",
    "gaming", "game", "fps", "rtx", "rx", "nvidia", "amd", "intel",
    "cloud", "server", "database", "linux", "windows", "macos",
    "router", "wifi", "bluetooth", "wireless", "network",
    "battery", "charging", "performance", "benchmark", "review",
    "best", "budget", "affordable", "student", "developer",
}


def fetch_beir() -> List[Dict]:
    """
    Load MS MARCO passages from HuggingFace BEIR collection.
    Filters to tech-relevant passages only.
    SciFact (biomedical) is intentionally excluded.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets package not installed — skipping BEIR source.")
        return []

    docs = []
    seen_ids = set()

    logger.info("Loading BEIR MS MARCO corpus (scanning 5000, keeping up to 1000 tech passages)...")
    try:
        dataset = load_dataset(
            "BeIR/msmarco",
            "corpus",
            split="corpus",
            streaming=True,
        )

        scanned = 0
        kept = 0
        MAX_SCAN = 5000
        MAX_KEEP = 1000

        for record in tqdm(dataset, desc="BEIR MS MARCO (tech filter)", unit="doc", total=MAX_SCAN):
            if scanned >= MAX_SCAN or kept >= MAX_KEEP:
                break

            scanned += 1
            title = record.get("title", "")
            text  = record.get("text", "")
            combined = f"{title} {text}".lower()

            # Skip if no tech keyword found
            if not any(kw in combined for kw in TECH_KEYWORDS):
                continue

            # Skip if too short
            if len(text.strip()) < 50:
                continue

            rid    = str(record.get("_id", ""))
            doc_id = f"msmarco_{rid}"

            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                docs.append({
                    "doc_id": doc_id,
                    "title": title,
                    "text": text[:2000],
                    "url": f"https://msmarco.org/doc/{rid}",
                    "source": "beir",
                })
                kept += 1

        logger.info(f"BEIR MS MARCO: scanned {scanned}, kept {kept} tech-relevant passages")

    except Exception as e:
        logger.error(f"BEIR MS MARCO load failed: {e}")

    logger.info(f"BEIR total: {len(docs)} docs")
    return docs


# ─────────────────────────────────────────────────────────────────────
# CORPUS BUILDER — MAIN
# ─────────────────────────────────────────────────────────────────────

def build_corpus(force_rebuild: bool = False) -> List[Dict]:
    """Build unified corpus from all sources. Cache to corpus.json."""
    if os.path.exists(CORPUS_FILE) and not force_rebuild:
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            docs = json.load(f)
        print(f"Loaded corpus: {len(docs)} docs from cache")
        logger.info(f"Loaded {len(docs)} docs from {CORPUS_FILE}")
        return docs

    all_docs = []

    # --- Source 1: Wikipedia ---
    print("\n=== Fetching Wikipedia... ===")
    try:
        wiki_docs = fetch_wikipedia()
        all_docs.extend(wiki_docs)
        print(f"[OK] Wikipedia: {len(wiki_docs)} docs")
    except Exception as e:
        print(f"[FAIL] Wikipedia fetch failed: {e}")
        logger.error(f"Wikipedia source failed: {e}")

    # --- Source 2: Reddit ---
    print("\n=== Fetching Reddit... ===")
    try:
        reddit_docs = fetch_reddit()
        all_docs.extend(reddit_docs)
        print(f"[OK] Reddit: {len(reddit_docs)} docs")
    except Exception as e:
        print(f"[FAIL] Reddit fetch failed: {e}")
        logger.error(f"Reddit source failed: {e}")

    # --- Source 3: DuckDuckGo ---
    print("\n=== Fetching DuckDuckGo... ===")
    try:
        ddg_docs = fetch_ddg()
        all_docs.extend(ddg_docs)
        print(f"[OK] DuckDuckGo: {len(ddg_docs)} docs")
    except Exception as e:
        print(f"[FAIL] DuckDuckGo fetch failed: {e}")
        logger.error(f"DuckDuckGo source failed: {e}")

    # --- Source 4: BEIR MS MARCO (tech filtered) ---
    print("\n=== Loading BEIR MS MARCO (tech passages only)... ===")
    try:
        beir_docs = fetch_beir()
        all_docs.extend(beir_docs)
        print(f"[OK] BEIR MS MARCO: {len(beir_docs)} tech-relevant passages")
    except Exception as e:
        print(f"[FAIL] BEIR load failed: {e}")
        logger.error(f"BEIR source failed: {e}")

    # --- Deduplicate by doc_id ---
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc["doc_id"] not in seen:
            seen.add(doc["doc_id"])
            unique_docs.append(doc)

    # --- Save ---
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(unique_docs, f, indent=2, ensure_ascii=False)

    # --- Summary ---
    wiki_n   = sum(1 for d in unique_docs if d["source"] == "wikipedia")
    reddit_n = sum(1 for d in unique_docs if d["source"] == "reddit")
    ddg_n    = sum(1 for d in unique_docs if d["source"] == "ddg")
    beir_n   = sum(1 for d in unique_docs if d["source"] == "beir")

    print(f"\n{'='*50}")
    print(f"  CORPUS BUILT SUCCESSFULLY")
    print(f"{'='*50}")
    print(f"  Total:          {len(unique_docs)} documents")
    print(f"  Wikipedia:      {wiki_n}")
    print(f"  Reddit:         {reddit_n}")
    print(f"  DuckDuckGo:     {ddg_n}")
    print(f"  BEIR MS MARCO:  {beir_n}  (tech-filtered, SciFact removed)")
    print(f"{'='*50}")
    print(f"  Saved to: {CORPUS_FILE}")

    return unique_docs


def rebuild_corpus() -> List[Dict]:
    """Force-rebuild the corpus (delete cache + rebuild)."""
    if os.path.exists(CORPUS_FILE):
        os.remove(CORPUS_FILE)
    return build_corpus(force_rebuild=True)


def get_corpus_stats(docs: List[Dict]) -> Dict:
    """Return corpus statistics by source."""
    by_source = {}
    sample_titles = {}
    for doc in docs:
        src = doc.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
        if src not in sample_titles:
            sample_titles[src] = []
        if len(sample_titles[src]) < 5:
            sample_titles[src].append(doc.get("title", "Untitled"))

    return {
        "total": len(docs),
        "by_source": by_source,
        "sample_titles": sample_titles,
    }


if __name__ == "__main__":
    import sys
    import io

    # Force UTF-8 output on Windows
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    force = "--rebuild" in sys.argv or "--force" in sys.argv

    print("=" * 50)
    print("  Multi-Source Corpus Builder")
    print("  Sources: Wikipedia | Reddit | DDG | BEIR MS MARCO")
    print("  Note: SciFact removed (biomedical, not tech)")
    print("=" * 50)

    if force:
        print("\n[REBUILD] Force rebuild requested -- fetching all sources fresh.\n")
        corpus = rebuild_corpus()
    else:
        corpus = build_corpus()

    avg_len = sum(len(d.get("text", "")) for d in corpus) // max(len(corpus), 1)
    print(f"\nAverage text length: {avg_len} chars/doc")
    print(f"Done! {len(corpus)} documents ready.")