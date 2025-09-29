import os, re, uuid, json
import fitz  # PyMuPDF
import psycopg2
import requests
from datetime import datetime

# -------- ENV --------
OS_HOST   = os.getenv("OPENSEARCH_HOST", "localhost")
OS_PORT   = int(os.getenv("OPENSEARCH_PORT", "9200"))
OS_CHIDX  = os.getenv("OPENSEARCH_CHUNKS_INDEX", "chunks_v1")

PG_HOST   = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT   = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB     = os.getenv("POSTGRES_DB", "curator")
PG_USER   = os.getenv("POSTGRES_USER", "curator_user")
PG_PASS   = os.getenv("POSTGRES_PASSWORD", "curator_pwd")

# -------- Utilities --------
def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunkify(text: str, max_chars=1000, overlap=150):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        # try to break on a sentence boundary near the end
        window = text[start:end]
        cut = window.rfind(". ")
        if cut != -1 and cut > max_chars * 0.6:
            end = start + cut + 1
            window = text[start:end]
        chunks.append(window.strip())
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def ensure_pg():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS
    )

def ensure_os_index():
    mapping = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"properties": {
            "chunk_id":   {"type": "keyword"},
            "doc_id":     {"type": "keyword"},
            "chunk_index":{"type": "integer"},
            "content":    {"type": "text"},
            "page_from":  {"type": "integer"},
            "page_to":    {"type": "integer"},
            "ingested_at":{"type": "date"}
        }}
    }
    url = f"http://{OS_HOST}:{OS_PORT}/{OS_CHIDX}"
    r = requests.get(url, timeout=5)
    if r.status_code == 200:
        return
    r = requests.put(url, json=mapping, timeout=10)
    r.raise_for_status()

def bulk_index_chunks(chunks):
    if not chunks:
        return 0
    url = f"http://{OS_HOST}:{OS_PORT}/{OS_CHIDX}/_bulk"
    lines = []
    for ch in chunks:
        lines.append({"index": {"_id": ch["chunk_id"]}})
        lines.append(ch)
    payload = "\n".join(json.dumps(x) for x in lines) + "\n"
    r = requests.post(url, data=payload, headers={"Content-Type": "application/x-ndjson"}, timeout=30)
    r.raise_for_status()
    j = r.json()
    if j.get("errors"):
        raise RuntimeError("OpenSearch bulk indexing reported errors")
    return len(chunks)

def upsert_pg_chunks(conn, chunks):
    if not chunks:
        return 0
    with conn, conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id    TEXT PRIMARY KEY,
            doc_id      TEXT NOT NULL,
            chunk_index INT  NOT NULL,
            content     TEXT NOT NULL,
            page_from   INT,
            page_to     INT,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );
        """)
        args = [
            (c["chunk_id"], c["doc_id"], c["chunk_index"], c["content"], c.get("page_from"), c.get("page_to"))
            for c in chunks
        ]
        cur.executemany("""
        INSERT INTO chunks (chunk_id, doc_id, chunk_index, content, page_from, page_to)
        VALUES (%s,%s,%s,%s,%s,%s)
        ON CONFLICT (chunk_id) DO UPDATE SET
          content=EXCLUDED.content,
          page_from=EXCLUDED.page_from,
          page_to=EXCLUDED.page_to;
        """, args)
    return len(chunks)

def pdf_to_text_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    blocks = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        text = clean_text(text)
        if text:
            blocks.append((i+1, text))  # 1-based page index
    doc.close()
    return blocks

def chunk_pdf(pdf_path, doc_id: str | None = None, max_chars=1000, overlap=150):
    # derive doc_id from filename if not provided
    if doc_id is None:
        base = os.path.basename(pdf_path)
        doc_id = os.path.splitext(base)[0]
    blocks = pdf_to_text_blocks(pdf_path)
    chunks = []
    idx = 0
    for page_no, text in blocks:
        parts = chunkify(text, max_chars=max_chars, overlap=overlap)
        for p in parts:
            chunks.append({
                "chunk_id":   f"{doc_id}:{idx}",
                "doc_id":     doc_id,
                "chunk_index": idx,
                "content":     p,
                "page_from":   page_no,
                "page_to":     page_no,
                "ingested_at": datetime.utcnow().isoformat() + "Z"
            })
            idx += 1
    return chunks

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chunk PDF(s) and index to PG + OpenSearch")
    parser.add_argument("pdf", nargs="+", help="Path(s) to PDF files")
    parser.add_argument("--doc-id", help="Optional explicit doc_id to associate")
    parser.add_argument("--max-chars", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()

    ensure_os_index()
    conn = ensure_pg()

    total_chunks = 0
    for path in args.pdf:
        assert os.path.exists(path), f"Not found: {path}"
        # use filename as doc_id if not provided (or reuse provided one)
        doc_id = args.doc_id or os.path.splitext(os.path.basename(path))[0]
        print(f"[chunk] Processing {path} as doc_id={doc_id} …")
        chunks = chunk_pdf(path, doc_id=doc_id, max_chars=args.max_chars, overlap=args.overlap)
        print(f"[chunk] Produced {len(chunks)} chunks")
        upsert_pg_chunks(conn, chunks)
        bulk_index_chunks(chunks)
        total_chunks += len(chunks)

    print(f"[chunk] Done. Total chunks indexed: {total_chunks}")

if __name__ == "__main__":
    main()
