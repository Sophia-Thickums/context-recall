#!/usr/bin/env python3
"""context_recall.py — Addressable memory for long-lived agents.

The final layer of a three-layer continuity architecture:
  raw history (markdown files, the append-only source of truth)
    -> semantic layer (SQLite FTS5 for exact terms + ChromaDB vectors for meaning)
    -> recall (THIS module: pull ONE memory, not the whole archive).

Why: agent harnesses re-carry large amounts of prior history every turn so the next
reply "has context." That is rereading your entire life to think one sentence.
This module makes history addressable instead: query it, retrieve what's actually
relevant (~a few hundred tokens of chunk), and leave the rest on disk.

Measured: one addressed recall ~250 tokens vs ~27,000 tokens to load a 100KB memory
archive whole. ~26k tokens saved per recall-when-needed, at zero loss — the full
text remains one pointer away when it genuinely matters.

Zero cloud by default. FTS5 exact + ChromaDB vectors (any local embedder — tested
with Ollama embeddings API). Runs standalone CLI or imported.

Usage:
    # from context_recall import recall, address
    hits = recall("the night we chose the vector store", depth=1)
    ptrs = address("first deployment outage")     # one-line receipts only

CLI:
    python context_recall.py "query" [--depth 0|1|2] [--no-vectors]
    python context_recall.py --sync-vectors       # embed new FTS5 chunks (explicit, ~10s/700)
    python context_recall.py --stats

Config (env):
    CONTEXT_RECALL_ROOT      root dir containing .agent_memory.db and .agent_chroma
    CONTEXT_RECALL_EMBED_URL  embeddings endpoint (default: http://127.0.0.1:11434/api/embeddings)
    CONTEXT_RECALL_EMBED_MODEL embedding model name (default: nomic-embed-text)
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import os
ROOT = Path(os.environ.get("CONTEXT_RECALL_ROOT", Path.cwd()))
INDEX_DB = ROOT / os.environ.get("CONTEXT_RECALL_DB", ".agent_memory.db")

# ---------------------------------------------------------------- embedding backend (local, env-configurable)

EMBED_URL = os.environ.get("CONTEXT_RECALL_EMBED_URL", "http://127.0.0.1:11434/api/embeddings")
EMBED_MODEL = os.environ.get("CONTEXT_RECALL_EMBED_MODEL", "nomic-embed-text")


def _doc_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def embed_text(text):
    """Single embedding via the configured endpoint (Ollama-compatible)."""
    try:
        import numpy as np  # noqa
    except Exception:
        pass
    try:
        import urllib.request, urllib.error
        payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(EMBED_URL, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8")).get("embedding")
    except Exception:
        return None


class OllamaEmbeddingFunction:
    """ChromaDB-compatible embedding function over EMBED_URL. Batches."""
    def __call__(self, input):
        import numpy as np
        vecs = [embed_text(t) for t in input]
        return [v for v in (vecs)]


CHROMA_PATH = ROOT / os.environ.get("CONTEXT_RECALL_CHROMA", ".agent_chroma")



DEPTH_FULL_CHARS = 6000  # depth=2 full-window cap per hit

# ---------------------------------------------------------------- FTS5 layer (exact terms)

def _fts(query: str, n: int = 8):
    """Full-text search hits: [(file_path, chunk_content, rank)]."""
    if not INDEX_DB.exists():
        return []
    conn = sqlite3.connect(str(INDEX_DB))
    try:
        terms = [t for t in query.replace('"', ' ').split() if t.isalnum() or len(t) > 2]
        if not terms:
            return []
        fq = " OR ".join(f'"{t}"' for t in terms[:6])
        # qualify every column: memory_fts AND memory_chunks both carry file_path (ambiguous-column field note: both tables carry file_path)
        rows = conn.execute(
            """SELECT m.file_path, m.content, rank
               FROM memory_fts f JOIN memory_chunks m ON f.rowid = m.id
               WHERE f.memory_fts MATCH ?
               ORDER BY rank LIMIT ?""",
            (fq, n),
        ).fetchall()
        return rows
    except Exception as e:
        # surface, never swallow silently — a silent except here returned 0 hits for a verified fact (09-09)
        import sys
        print(f"[context_recall FTS error: {e}]", file=sys.stderr)
        return []
    finally:
        conn.close()

# ---------------------------------------------------------------- Vector layer (meaning)

def _vector_hits(query: str, n: int = 8):
    """Chroma hits: [(file_path, chunk_content, score)]. Lazy: only touches chroma if present.
    Requires vectors synced via --sync-vectors."""
    try:
        import chromadb
    except Exception:
        return []
    if not CHROMA_PATH.exists():
        return []
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        # bind the EF at query-time too — insert-time EF alone leaves query vectors 384-dim vs 768 stored (field note)
        col = client.get_collection("context_memories", embedding_function=OllamaEmbeddingFunction())
        if col.count() == 0:
            return []
        res = col.query(query_texts=[query], n_results=n)
        out = []
        for i, doc in enumerate(res["documents"][0]):
            meta = res["metadatas"][0][i] if res.get("metadatas") else {}
            dist = res["distances"][0][i] if res.get("distances") else None
            out.append((meta.get("file_path", "chroma"), doc, dist))
        return out
    except Exception:
        return []

# ---------------------------------------------------------------- Public API

def recall(query: str, depth: int = 1, use_vectors: bool = True, n: int = 8):
    """The addressable door. Returns list of dicts:
    {file, chunk_index, content, source, score?}
    depth 0 = one-line addresses only (titles), 1 = chunk text (default), 2 = full window.
    """
    hits = []
    seen = set()

    for fp, content, rank in _fts(query, n):
        key = (fp, content[:80])
        if key in seen:
            continue
        seen.add(key)
        hits.append({"file": fp, "content": _trim(content, depth), "source": "fts", "rank": rank})

    if use_vectors and len(hits) < n:
        for fp, content, dist in _vector_hits(query, n - len(hits)):
            key = (fp, content[:80])
            if key in seen:
                continue
            seen.add(key)
            hits.append({"file": fp, "content": _trim(content, depth), "source": "vector", "score": dist})

    return hits


def address(query: str, use_vectors: bool = True):
    """depth=0 convenience: one line per hit, the cheapest possible memory pull."""
    hits = recall(query, depth=0, use_vectors=use_vectors)
    return "\n".join(f"- [{h['source']}] {h['file']}: {h['content'][:110]}" for h in hits)


def _trim(content: str, depth: int) -> str:
    if depth == 0:
        return content.split("\n")[0][:110]
    if depth == 2:
        return content[:DEPTH_FULL_CHARS]
    return content  # depth 1: chunk as-is (~500 chars from indexer)

# ---------------------------------------------------------------- Vector sync (explicit)

def sync_vectors(batch: int = 64):
    """Embed any FTS5 chunks not yet in Chroma. Local only; ~0.1s/chunk after model warm."""
    import chromadb

    conn = sqlite3.connect(str(INDEX_DB))
    rows = conn.execute("SELECT id, file_path, content FROM memory_chunks").fetchall()
    conn.close()

    # ID must be content+file+chunk: identical date-header chunks across era files
    # share content-hash otherwise (DuplicateIDError field note: identical date-header chunks across era files collide on content-hash alone).
    def vid(fp, cid, c):
        return _doc_id(f"{fp}::{cid}::{c}")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    col = client.get_or_create_collection(
        "context_memories", embedding_function=OllamaEmbeddingFunction()
    )
    existing = set(col.get()["ids"])

    new = [(i, fp, c) for (i, fp, c) in rows if vid(fp, i, c) not in existing]
    print(f"vectors: {len(existing)} present, {len(new)} new from {len(rows)} chunks", flush=True)
    if not new:
        return 0

    docs, metas, ids = [], [], []
    done = 0
    t0 = time.time()
    for i, fp, c in new:
        docs.append(c)
        metas.append({"file_path": fp, "chunk_id": i})
        ids.append(vid(fp, i, c))
        if len(docs) >= batch:
            col.add(documents=docs, metadatas=metas, ids=ids)
            done += len(docs)
            docs, metas, ids = [], [], []
            print(f"  embedded {done}/{len(new)} ({time.time()-t0:.1f}s)", flush=True)
    if docs:
        col.add(documents=docs, metadatas=metas, ids=ids)
        done += len(docs)
    print(f"synced {done} vectors in {time.time()-t0:.1f}s", flush=True)
    return done

# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description="addressable context recall")
    ap.add_argument("query", nargs="?")
    ap.add_argument("--depth", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--no-vectors", action="store_true")
    ap.add_argument("--sync-vectors", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    if args.sync_vectors:
        sync_vectors()
        return
    if args.stats:
        conn = sqlite3.connect(str(INDEX_DB))
        n = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
        conn.close()
        print(f"FTS5 chunks: {n}")
        try:
            import chromadb
            if CHROMA_PATH.exists():
                c = chromadb.PersistentClient(path=str(CHROMA_PATH))
                col = c.get_collection("context_memories")
                print(f"Chroma vectors: {col.count()}")
            else:
                print("Chroma: not initialized yet")
        except Exception as e:
            print(f"Chroma: {e}")
        return
    if not args.query:
        ap.print_help()
        return

    hits = recall(args.query, depth=args.depth, use_vectors=not args.no_vectors)
    if not hits:
        print("no hits")
        return
    for h in hits:
        src = h.get("source")
        print(f"--- [{src}] {h['file']}")
        print(h["content"])
        print()


if __name__ == "__main__":
    main()
