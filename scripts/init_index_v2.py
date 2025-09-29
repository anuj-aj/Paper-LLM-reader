#!/usr/bin/env python
import os, sys, argparse, requests, psycopg2

def ensure_db():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "curator"),
        user=os.getenv("POSTGRES_USER", "curator_user"),
        password=os.getenv("POSTGRES_PASSWORD", "curator_pwd"),
    )
    ddl = """
    CREATE TABLE IF NOT EXISTS documents (
      doc_id TEXT PRIMARY KEY,
      title TEXT, authors TEXT, year INT, abstract TEXT,
      pdf_url TEXT, created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS chunks (
      chunk_id TEXT PRIMARY KEY,
      doc_id TEXT NOT NULL,
      chunk_index INT NOT NULL,
      content TEXT NOT NULL,
      page_from INT, page_to INT,
      created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
    """
    with conn, conn.cursor() as cur:
        cur.execute(ddl)
    print("[ok] Postgres tables ensured")

def ensure_os_index(alias: str, dim: int):
    base = f"http://{os.getenv('OPENSEARCH_HOST','localhost')}:{os.getenv('OPENSEARCH_PORT','9200')}"
    # If alias already exists, assume setup is done
    a = requests.get(f"{base}/_alias/{alias}")
    if a.status_code == 200:
        print(f"[ok] OpenSearch alias '{alias}' already exists — skipping create")
        return
    # Create a single index and attach alias as write index
    index = f"{alias}_000001"
    body = {
      "settings": {
        "index": {"number_of_shards": 1, "number_of_replicas": 0, "knn": True}
      },
      "mappings": {
        "properties": {
          "chunk_id":   {"type": "keyword"},
          "doc_id":     {"type": "keyword"},
          "chunk_index":{"type": "integer"},
          "content":    {"type": "text"},
          "page_from":  {"type": "integer"},
          "page_to":    {"type": "integer"},
          "ingested_at":{"type": "date"},
          "embedding":  {
            "type": "knn_vector",
            "dimension": dim,
            "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"}
          }
        }
      },
      "aliases": { alias: { "is_write_index": True } }
    }
    r = requests.put(f"{base}/{index}", json=body, timeout=20)
    r.raise_for_status()
    print(f"[ok] OpenSearch index '{index}' created and write alias '{alias}' attached")

def main():
    p = argparse.ArgumentParser(description="One-shot init: DB tables + OS vector index+alias")
    p.add_argument("--alias", default=os.getenv("OPENSEARCH_CHUNKS_V2_INDEX","chunks_v2"))
    p.add_argument("--embed-dim", type=int, default=int(os.getenv("EMBED_DIM","768")))
    args = p.parse_args()

    ensure_db()
    ensure_os_index(alias=args.alias, dim=args.embed_dim)
    print("[done] init complete")

if __name__ == "__main__":
    main()
