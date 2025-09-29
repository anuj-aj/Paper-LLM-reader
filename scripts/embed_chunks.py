#!/usr/bin/env python
"""
Embed chunks from Postgres and index vectors into OpenSearch (chunks_v2 alias).

Usage (inside Docker):
  # Ollama (default), nomic-embed-text (768 dims)
  docker exec -it paper_api python /app/scripts/embed_chunks.py --limit 500

  # OpenAI (set EMBED_DIM=1536 or 3072 to match your model)
  docker exec -it paper_api python /app/scripts/embed_chunks.py --provider openai --model text-embedding-3-small --limit 500
"""
import os
import sys
import time
import math
import json
import argparse
from datetime import datetime, timezone

import psycopg2
import requests

# ---------- ENV ----------
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB   = os.getenv("POSTGRES_DB", "curator")
PG_USER = os.getenv("POSTGRES_USER", "curator_user")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "curator_pwd")

OS_HOST   = os.getenv("OPENSEARCH_HOST", "localhost")
OS_PORT   = int(os.getenv("OPENSEARCH_PORT", "9200"))
OS_ALIAS  = os.getenv("OPENSEARCH_CHUNKS_V2_INDEX", "chunks_v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))  # must match your vector index

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---------- Helpers ----------
def pg_conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS
    )

def ensure_embedded_at_column(conn):
    with conn, conn.cursor() as cur:
        cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='chunks' AND column_name='embedded_at'
            ) THEN
                ALTER TABLE chunks ADD COLUMN embedded_at TIMESTAMPTZ;
            END IF;
        END$$;
        """)

def fetch_chunk_batch(conn, batch_size=100, only_missing=True, doc_id_filter=None):
    """
    Returns list of (chunk_id, doc_id, chunk_index, content, page_from, page_to).
    """
    with conn.cursor() as cur:
        if only_missing:
            where = "embedded_at IS NULL"
        else:
            where = "TRUE"
        params = []
        if doc_id_filter:
            where += " AND doc_id = %s"
            params.append(doc_id_filter)

        sql = f"""
        SELECT chunk_id, doc_id, chunk_index, content, page_from, page_to
        FROM chunks
        WHERE {where}
        ORDER BY chunk_id
        LIMIT %s
        """
        params.append(batch_size)
        cur.execute(sql, params)
        rows = cur.fetchall()
        return rows

def mark_embedded(conn, chunk_ids):
    if not chunk_ids:
        return
    with conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE chunks SET embedded_at = NOW() WHERE chunk_id = ANY(%s)",
            (chunk_ids,)
        )

def get_embedding_client(provider, model):
    if provider == "ollama":
        endpoint = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings"
        def embed_ollama(texts):
            out = []
            for t in texts:
                r = requests.post(endpoint, json={"model": model, "prompt": t}, timeout=60)
                r.raise_for_status()
                vec = r.json().get("embedding", [])
                if len(vec) != EMBED_DIM:
                    raise ValueError(f"Ollama returned dim={len(vec)}, expected {EMBED_DIM}")
                out.append(vec)
            return out
        return embed_ollama

    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        endpoint = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        def embed_openai(texts):
            # OpenAI supports batching; keep batches small to avoid rate limits
            r = requests.post(endpoint, headers=headers, json={"model": model, "input": texts}, timeout=60)
            r.raise_for_status()
            datas = r.json()["data"]
            vecs = [d["embedding"] for d in datas]
            # Validate dimension
            if any(len(v) != EMBED_DIM for v in vecs):
                raise ValueError(f"OpenAI returned a vector with unexpected dim (expected {EMBED_DIM})")
            return vecs
        return embed_openai

    else:
        raise ValueError(f"Unknown provider: {provider}")

def bulk_upsert_vectors(docs):
    """
    Upsert to OpenSearch alias using bulk API.
    docs: list of dicts with keys:
      chunk_id, doc_id, chunk_index, content, page_from, page_to, embedding (list[float])
    """
    if not docs:
        return
    base = f"http://{OS_HOST}:{OS_PORT}"
    url = f"{base}/{OS_ALIAS}/_bulk"
    lines = []
    for d in docs:
        lines.append({"index": {"_id": d["chunk_id"]}})
        lines.append({
            "chunk_id": d["chunk_id"],
            "doc_id": d["doc_id"],
            "chunk_index": d["chunk_index"],
            "content": d["content"],
            "page_from": d.get("page_from"),
            "page_to": d.get("page_to"),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "embedding": d["embedding"],
        })
    payload = "\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n"
    r = requests.post(url, data=payload, headers={"Content-Type": "application/x-ndjson"}, timeout=90)
    r.raise_for_status()
    j = r.json()
    if j.get("errors"):
        # Surface first error for debugging
        first = next((it for it in j.get("items", []) if it.get("index", {}).get("error")), None)
        raise RuntimeError(f"OpenSearch bulk reported errors; first: {first}")

def chunk_iter(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def main():
    ap = argparse.ArgumentParser(description="Embed chunks and index vectors into OpenSearch.")
    ap.add_argument("--provider", choices=["ollama","openai"], default=os.getenv("EMBED_PROVIDER","ollama"))
    ap.add_argument("--model", default=os.getenv("EMBED_MODEL","nomic-embed-text"))
    ap.add_argument("--batch-size", type=int, default=int(os.getenv("EMBED_BATCH","64")), help="PG read batch")
    ap.add_argument("--embed-batch", type=int, default=int(os.getenv("EMBED_API_BATCH","32")), help="API embed batch size")
    ap.add_argument("--limit", type=int, default=None, help="Max chunks to process")
    ap.add_argument("--only-missing", action="store_true", default=True, help="Only chunks with embedded_at IS NULL")
    ap.add_argument("--include-embedded", action="store_true", help="Process even if already embedded (overwrites)")
    ap.add_argument("--doc-id", help="Only process a specific doc_id")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep between API calls (rate limiting)")
    args = ap.parse_args()

    only_missing = not args.include_embedded if args.only_missing else True

    print(f"[cfg] provider={args.provider} model={args.model} dim={EMBED_DIM} alias={OS_ALIAS}")
    print(f"[cfg] batch-size={args.batch_size} embed-batch={args.embed_batch} limit={args.limit or '∞'}")

    embed = get_embedding_client(args.provider, args.model)

    conn = pg_conn()
    ensure_embedded_at_column(conn)

    total_done = 0
    while True:
        # Stop if limit hit
        remain = None if args.limit is None else max(0, args.limit - total_done)
        this_batch = min(args.batch_size, remain) if remain is not None else args.batch_size
        if this_batch <= 0:
            break

        rows = fetch_chunk_batch(conn, batch_size=this_batch, only_missing=only_missing, doc_id_filter=args.doc_id)
        if not rows:
            print("[embed] No more chunks to process.")
            break

        # Prepare texts and IDs
        chunk_ids = [r[0] for r in rows]
        docs_for_os = []
        texts = [r[3] for r in rows]  # content

        # Embed in sub-batches to respect API limits
        embeddings = []
        for sub in chunk_iter(texts, args.embed_batch):
            vecs = embed(sub)
            embeddings.extend(vecs)
            time.sleep(args.sleep)

        # Sanity
        if len(embeddings) != len(rows):
            raise RuntimeError("Embedding count mismatch")

        # Build OS docs
        for (chunk_id, doc_id, chunk_index, content, page_from, page_to), vec in zip(rows, embeddings):
            if len(vec) != EMBED_DIM:
                raise ValueError(f"Vector dim {len(vec)} != EMBED_DIM {EMBED_DIM}")
            docs_for_os.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "content": content,
                "page_from": page_from,
                "page_to": page_to,
                "embedding": vec,
            })

        # Upsert to OpenSearch and mark in PG
        bulk_upsert_vectors(docs_for_os)
        if only_missing:
            mark_embedded(conn, chunk_ids)

        total_done += len(rows)
        print(f"[embed] processed batch: {len(rows)}   total={total_done}")

        if args.limit and total_done >= args.limit:
            break

    conn.close()
    print(f"[done] total embedded: {total_done}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[fatal] {e}", file=sys.stderr)
        sys.exit(1)
