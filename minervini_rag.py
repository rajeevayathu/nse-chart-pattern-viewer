"""
minervini_rag.py
────────────────
RAG (Retrieval-Augmented Generation) for Minervini chart analysis.

Usage
-----
# One-time: build the index from the PDF
python minervini_rag.py --build /path/to/book.pdf

# From generate_charts.py:
from minervini_rag import get_minervini_context
ctx = get_minervini_context(pattern='VCP', stage=2, pivot_pct=2.1, rs=92)
"""

import os
import re
import sys
import json
import math
import pickle
import argparse
import urllib.request as _req

# ── CONFIG ────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH  = os.path.join(SCRIPT_DIR, 'minervini_rag_index.pkl')
EMBED_MODEL = 'nomic-embed-text'
OLLAMA_URL  = 'http://localhost:11434'
CHUNK_WORDS = 380
OVERLAP_WORDS = 70
TOP_K = 4   # passages retrieved per query

# ── SECTION HINTS (used for metadata tagging) ─────────────────────────────────

_SECTION_TAGS = {
    'entry': ['entry', 'buy point', 'pivot', 'breakout', 'trigger', 'purchase'],
    'exit':  ['exit', 'stop loss', 'stop-loss', 'sell', 'cut loss', 'loss limit', 'take profit'],
    'vcp':   ['volatility contraction', 'vcp', 'contraction', 'tight', 'base'],
    'setup': ['setup', 'template', 'criteria', 'checklist', 'qualify', 'sepa'],
    'stage': ['stage 1', 'stage 2', 'stage 3', 'stage 4', 'trending up', 'stage analysis'],
    'rs':    ['relative strength', 'rs rating', 'rs line', 'relative performance'],
    'volume':['volume', 'accumulation', 'distribution', 'dry up', 'vol'],
    'ma':    ['moving average', '200-day', '150-day', '50-day', 'ma200', 'ma50'],
    'risk':  ['risk', 'position size', 'risk management', 'capital', 'portfolio'],
    'psychology': ['mindset', 'discipline', 'emotion', 'psychology', 'patience'],
}

def _tag_chunk(text):
    low = text.lower()
    return [tag for tag, kws in _SECTION_TAGS.items() if any(k in low for k in kws)]


# ── PDF EXTRACTION ────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_path):
    try:
        import pdfplumber
    except ImportError:
        sys.exit("pip install pdfplumber")

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"  Extracting {total} pages…", flush=True)
        for i, page in enumerate(pdf.pages):
            t = page.extract_text()
            if t:
                pages.append(t)
            if (i+1) % 50 == 0:
                print(f"    {i+1}/{total}", flush=True)
    return '\n'.join(pages)


def clean_text(raw):
    # collapse hyphenation, normalise whitespace
    text = re.sub(r'-\n', '', raw)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    # drop page headers/footers (short lines of ALL CAPS or numbers)
    text = re.sub(r'\b\d{1,3}\b', ' ', text)   # lone page numbers
    return text.strip()


# ── CHUNKING ──────────────────────────────────────────────────────────────────

def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(' '.join(words[start:end]))
        if end == len(words):
            break
        start += chunk_words - overlap
    return chunks


# ── OLLAMA EMBEDDINGS ─────────────────────────────────────────────────────────

def _embed(text):
    payload = json.dumps({'model': EMBED_MODEL, 'prompt': text}).encode()
    req = _req.Request(
        f'{OLLAMA_URL}/api/embeddings',
        data=payload,
        headers={'Content-Type': 'application/json'}
    )
    resp = _req.urlopen(req, timeout=30)
    return json.loads(resp.read())['embedding']


def embed_batch(texts, batch=32):
    embeddings = []
    for i, t in enumerate(texts):
        embeddings.append(_embed(t))
        if (i+1) % batch == 0:
            print(f"    Embedded {i+1}/{len(texts)}", flush=True)
    return embeddings


# ── COSINE SIMILARITY ─────────────────────────────────────────────────────────

def _dot(a, b):
    return sum(x*y for x,y in zip(a,b))

def _norm(a):
    return math.sqrt(sum(x*x for x in a))

def cosine(a, b):
    n = _norm(a) * _norm(b)
    return _dot(a,b) / n if n else 0.0


# ── INDEX BUILD ───────────────────────────────────────────────────────────────

def build_index(pdf_path, out_path=INDEX_PATH):
    print(f"\n📖 Building Minervini RAG index from:\n   {pdf_path}\n")
    raw  = extract_pdf_text(pdf_path)
    text = clean_text(raw)
    print(f"  Cleaned text: {len(text):,} chars")

    chunks = chunk_text(text)
    print(f"  Chunks: {len(chunks)}  (~{CHUNK_WORDS} words each, {OVERLAP_WORDS} overlap)")

    # tag each chunk
    meta = [{'tags': _tag_chunk(c), 'preview': c[:120]} for c in chunks]

    print(f"\n  Embedding {len(chunks)} chunks via {EMBED_MODEL}…")
    embeddings = embed_batch(chunks)
    print(f"  Done — embedding dim: {len(embeddings[0])}")

    index = {'chunks': chunks, 'embeddings': embeddings, 'meta': meta}
    with open(out_path, 'wb') as f:
        pickle.dump(index, f)
    print(f"\n  ✓ Index saved → {out_path}")
    print(f"    Size: {os.path.getsize(out_path)/1024/1024:.1f} MB")
    return index


# ── INDEX LOAD ────────────────────────────────────────────────────────────────

_INDEX_CACHE = None

def load_index(path=INDEX_PATH):
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        _INDEX_CACHE = pickle.load(f)
    return _INDEX_CACHE


# ── RETRIEVAL ─────────────────────────────────────────────────────────────────

def retrieve(query, top_k=TOP_K, tag_filter=None):
    """
    Return top_k (chunk_text, score, tags) tuples most relevant to query.
    Optionally filter to chunks that contain at least one tag in tag_filter.
    """
    idx = load_index()
    if idx is None:
        return []

    q_emb = _embed(query)
    scores = []
    for i, emb in enumerate(idx['embeddings']):
        tags = idx['meta'][i]['tags']
        if tag_filter and not any(t in tags for t in tag_filter):
            continue
        scores.append((cosine(q_emb, emb), i))

    scores.sort(reverse=True)
    return [(idx['chunks'][i], sc, idx['meta'][i]['tags'])
            for sc, i in scores[:top_k]]


# ── CONTEXT BUILDER (called from generate_charts.py) ─────────────────────────

_PATTERN_TAGS = {
    'VCP':              ['vcp', 'entry', 'setup', 'volume'],
    'BREAKOUT':         ['entry', 'volume', 'setup'],
    'CUP_HANDLE':       ['vcp', 'setup', 'entry', 'volume'],
    'DOUBLE_BOTTOM':    ['setup', 'entry', 'stage'],
    'ASC_TRIANGLE':     ['setup', 'entry', 'stage'],
    'BULL_FLAG':        ['setup', 'entry', 'vcp'],
    'TIGHT_BASE':       ['vcp', 'setup', 'stage'],
    'STAGE2':           ['stage', 'setup', 'ma'],
    'STAGE4':           ['stage', 'exit', 'risk'],
    'default':          ['setup', 'entry', 'vcp'],
}


def get_minervini_context(pattern='', stage=None, pivot_pct=None, rs=None,
                           action_hint='', top_k=TOP_K):
    """
    Build a query from chart context, retrieve relevant Minervini passages,
    return a formatted string ready for injection into the LLM prompt.
    Returns empty string if the RAG index doesn't exist.
    """
    if load_index() is None:
        return ''

    # Build a natural-language query from available signals
    parts = ['Minervini SEPA setup criteria']
    if pattern:
        parts.append(f'{pattern.replace("_"," ").lower()} chart pattern')
    if stage:
        parts.append(f'stage {stage} stock')
    if pivot_pct is not None:
        if pivot_pct < 5:
            parts.append('near pivot buy point breakout')
        elif pivot_pct < 15:
            parts.append('base building consolidation')
    if rs and rs >= 85:
        parts.append('high relative strength RS leader')
    if action_hint in ('BUY', 'WATCH'):
        parts.append('entry timing volume criteria')
    elif action_hint == 'AVOID':
        parts.append('avoid flawed setup stop loss')

    query = '. '.join(parts)
    tag_filter = _PATTERN_TAGS.get(pattern, _PATTERN_TAGS['default'])
    results = retrieve(query, top_k=top_k, tag_filter=tag_filter)

    if not results:
        # fallback without tag filter
        results = retrieve(query, top_k=top_k)

    if not results:
        return ''

    lines = ['=== Minervini Book Principles (from "Trade Like a Stock Market Wizard") ===']
    for i, (chunk, score, tags) in enumerate(results, 1):
        lines.append(f'\n[Passage {i}] (relevance {score:.2f} | topics: {", ".join(tags) or "general"})')
        lines.append(chunk.strip())
    lines.append('\n=== End of Book Principles ===')
    return '\n'.join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Minervini RAG index builder')
    parser.add_argument('--build', metavar='PDF', help='Path to the book PDF')
    parser.add_argument('--query', metavar='Q',   help='Test a retrieval query')
    parser.add_argument('--top',   type=int, default=3, help='Top-K results')
    args = parser.parse_args()

    if args.build:
        build_index(args.build)

    elif args.query:
        idx = load_index()
        if idx is None:
            print("Index not found — run with --build first.")
        else:
            print(f"\nQuery: {args.query}\n")
            for chunk, score, tags in retrieve(args.query, top_k=args.top):
                print(f"── Score {score:.3f}  tags={tags} ──")
                print(chunk[:600])
                print()
    else:
        if not os.path.exists(INDEX_PATH):
            print("Index not built yet.")
            print(f"Run:  python minervini_rag.py --build '/path/to/Trade Like a Stock Market Wizard.pdf'")
        else:
            idx = load_index()
            print(f"Index ready: {len(idx['chunks'])} chunks, dim={len(idx['embeddings'][0])}")
            print(f"Path: {INDEX_PATH}")
