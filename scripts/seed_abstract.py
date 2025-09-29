import os, sys, time, textwrap
import requests
import xml.etree.ElementTree as ET
import psycopg2
from psycopg2.extras import execute_values
import json

# -------- Config (reads from .env via docker-compose, else defaults) --------
OS_HOST   = os.getenv("OPENSEARCH_HOST", "localhost")
OS_PORT   = int(os.getenv("OPENSEARCH_PORT", "9200"))
OS_INDEX  = os.getenv("OPENSEARCH_INDEX_NAME", "papers_v1")

PG_HOST   = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT   = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB     = os.getenv("POSTGRES_DB", "curator")
PG_USER   = os.getenv("POSTGRES_USER", "curator_user")
PG_PASS   = os.getenv("POSTGRES_PASSWORD", "curator_pwd")
docker compose exec postgres sh -lc \
'psql -U curator_user -d curator -c "
SELECT doc_id, COUNT(*) AS n, SUM(CASE WHEN embedded_at IS NULL THEN 1 ELSE 0 END) AS pending
FROM chunks GROUP BY doc_id ORDER BY n DESC LIMIT 10;"'

ARXIV_Q   = os.getenv("ARXIV_SEARCH_QUERY", "cat:cs.AI")   # keep simple
ARXIV_MAX = int(os.getenv("ARXIV_RESULT_LIMIT", "1"))      # seed 1

# -------- Helpers --------
def arxiv_search(query, max_results=1):
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.text

def parse_arxiv_atom(atom_xml):
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(atom_xml)
    entries = []
    for e in root.findall("a:entry", ns):
        arxiv_id = e.findtext("a:id", default="", namespaces=ns).split("/")[-1]
        title = (e.findtext("a:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
        abstract = (e.findtext("a:summary", default="", namespaces=ns) or "").strip().replace("\n", " ")
        authors = [a.findtext("a:name", default="", namespaces=ns) for a in e.findall("a:author", ns)]
        year = None
        published = e.findtext("a:published", default="", namespaces=ns)
        if published:
            try:
                year = int(published[:4])
            except:
                year = None
        pdf_url = ""
        for link in e.findall("a:link", ns):
            if link.get("type") == "application/pdf":
                pdf_url = link.get("href")
                break
        entries.append({
            "doc_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": ", ".join([a for a in authors if a]),
            "year": year,
            "pdf_url": pdf_url,
            "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
        })
    return entries

def pg_connect():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS
    )

def ensure_pg_schema(conn):
    with conn, conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id     TEXT PRIMARY KEY,
            title      TEXT NOT NULL,
            authors    TEXT,
            year       INT,
            abstract   TEXT,
            arxiv_url  TEXT,
            pdf_url    TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)
        # Simple trigger-ish update
        cur.execute("""
        CREATE OR REPLACE FUNCTION set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
          NEW.updated_at = NOW();
          RETURN NEW;
        END; $$ LANGUAGE plpgsql;
        """)
        cur.execute("""
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_trigger WHERE tgname = 'documents_set_updated_at'
          ) THEN
            CREATE TRIGGER documents_set_updated_at
            BEFORE UPDATE ON documents
            FOR EACH ROW EXECUTE PROCEDURE set_updated_at();
          END IF;
        END $$;
        """)

def upsert_pg(conn, docs):
    rows = [
        (d["doc_id"], d["title"], d["authors"], d["year"], d["abstract"], d["arxiv_url"], d["pdf_url"])
        for d in docs
    ]
    with conn, conn.cursor() as cur:
        execute_values(cur, """
        INSERT INTO documents (doc_id, title, authors, year, abstract, arxiv_url, pdf_url)
        VALUES %s
        ON CONFLICT (doc_id) DO UPDATE SET
          title=EXCLUDED.title,
          authors=EXCLUDED.authors,
          year=EXCLUDED.year,
          abstract=EXCLUDED.abstract,
          arxiv_url=EXCLUDED.arxiv_url,
          pdf_url=EXCLUDED.pdf_url;
        """, rows)

def index_opensearch(docs):
    url = f"http://{OS_HOST}:{OS_PORT}/{OS_INDEX}/_bulk"
    lines = []
    for d in docs:
        meta = {"index": {"_id": d["doc_id"]}}
        body = {
            "doc_id": d["doc_id"],
            "title": d["title"],
            "authors": d["authors"],
            "year": d["year"],
            "abstract": d["abstract"],
        }
        lines.append(meta)
        lines.append(body)

    # newline-delimited JSON (action line + source line)
    payload = "\n".join([json.dumps(x) for x in lines]) + "\n"
    r = requests.post(url, data=payload, headers={"Content-Type":"application/x-ndjson"}, timeout=30)
    r.raise_for_status()
    j = r.json()
    if j.get("errors"):
        # surface the first error for quick debugging
        first_err = next((it for it in j.get("items", []) if "error" in it.get("index", {})), None)
        raise RuntimeError(f"OpenSearch bulk had errors: {first_err}")
    return len(lines)//2

def main():
    print(f"[seed] Fetching {ARXIV_MAX} paper(s) for '{ARXIV_Q}' from arXiv …")
    atom = arxiv_search(ARXIV_Q, ARXIV_MAX)
    docs = parse_arxiv_atom(atom)
    if not docs:
        print("[seed] No results from arXiv. Try a broader query or increase ARXIV_RESULT_LIMIT.")
        sys.exit(0)

    # Keep abstracts manageable
    for d in docs:
        if len(d["abstract"]) > 5000:
            d["abstract"] = d["abstract"][:5000] + " …"

    # Upsert Postgres
    print("[seed] Connecting Postgres and ensuring schema …")
    conn = pg_connect()
    ensure_pg_schema(conn)
    upsert_pg(conn, docs)
    print(f"[seed] Upserted {len(docs)} doc(s) into Postgres")

    # Index OpenSearch
    print(f"[seed] Indexing into OpenSearch index '{OS_INDEX}' …")
    n = index_opensearch(docs)
    print(f"[seed] Indexed {n} doc(s) into OpenSearch")

    # Print a quick preview
    d0 = docs[0]
    preview = textwrap.shorten(d0['abstract'], width=140, placeholder=" …")
    print("\n[seed] Example document")
    print(f"- id: {d0['doc_id']}")
    print(f"- title: {d0['title']}")
    print(f"- authors: {d0['authors']}")
    print(f"- year: {d0['year']}")
    print(f"- url: {d0['arxiv_url']}")
    print(f"- abstract: {preview}")

if __name__ == "__main__":
    main()
