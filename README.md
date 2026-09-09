# context-recall

Addressable memory for long-lived agents — pull **one** memory (~250 tokens) instead of re-carrying your archive (~27,000 tokens) every turn.

## The problem

Agent harnesses keep continuity the brute-force way: the more history you have, the more of it gets re-sent per turn so the next reply can "remember." Long-running agents end up paying ~27k tokens per turn just to exist — rereading their entire life to think one sentence.

## The fix

Three layers, only the last is new here:

```
raw history   markdown files (append-only source of truth)
semantic      SQLite FTS5 (exact terms) + ChromaDB vectors (meaning)
recall        `recall("query", depth=1)` -> the ~250-token chunk you actually need
```

Measured (real agent, ~100KB memory archive): one addressed recall ≈ 250 tokens. Loading the archive whole ≈ 26,700 tokens. **~26,000 tokens saved per recall-when-needed.**

## Use

```python
from context_recall import recall, address

hits = recall("the night we chose the vector store", depth=1)   # chunk text
ptrs = address("first deployment outage")                        # one-line receipts
```

```bash
python context_recall.py "query" --depth 1         # search
python context_recall.py --sync-vectors            # embed new chunks (explicit, ~10s/700)
python context_recall.py --stats
```

## Design laws (learned in production)

1. **Recall over re-carry.** History should be addressable; only the relevant chunk rides the turn.
2. **FTS5 + vectors, not either/or.** Exact terms ("the night X happened") and meaning ("times the system failed") need different engines. Merge, dedupe, FTS-first.
3. **Never swallow.** A silent `except` once returned 0 hits for a verified fact and the agent concluded wrong. Errors surface to stderr always.
4. **IDs are content+file+chunk-index.** Identical date-headers across era files collide on content-hash alone.
5. **Bind the embedder at query-time too.** Insert-time-only embedding left queries a different dimension than stored vectors on a mid-build model warm.
6. **Vector sync is explicit.** `--sync-vectors` (~10s/700 chunks), so imports are never a surprise network call.

## Config

| Env var | Default | Purpose |
|---|---|---|
| `CONTEXT_RECALL_ROOT` | cwd | root holding `.agent_memory.db` + `.agent_chroma` |
| `CONTEXT_RECALL_EMBED_URL` | `http://127.0.0.1:11434/api/embeddings` | embeddings endpoint |
| `CONTEXT_RECALL_EMBED_MODEL` | `nomic-embed-text` | embedding model |

Fully local by default. No cloud path.

MIT.
