#!/usr/bin/env python
import os
import re
import time
import argparse
import requests
import psycopg2
from psycopg2 import sql

# ---------- Config from env ----------
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB   = os.getenv("POSTGRES_DB", "curator")
PG_USER = os.getenv("POSTGRES_USER", "curator_user")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "curator_pwd")

OUT_DIR_DEFAULT = os.getenv("PDF_OUTPUT_DIR", "data/pdfs")
SLEEP_S = float(os.getenv("ARXIV_FETCH_SLEEP_S", "1.0"))

ARXIV_ABS = "https://export.arxiv.org/abs/"
ARXIV_PDF = "https://export.arxiv.org/pdf/"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "paper-curator-db-fetcher/0.1 (+local use)"})


ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")  # e.g., 2509.15207 or 2509.15207v1


def is_valid_id(s: str) -> bool:
    return bool(ID_RE.match(s.strip()))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def local_pdf_path(out_dir: str, doc_id_or_ver: str) -> str:
    """Returns local path like data/pdfs/2509.15207v1.pdf (adds v1 if no version given)."""
    base = doc_id_or_ver
    if "v" not in base:
        base = base + "v1"
    return os.path.join(out_dir, f"{base}.pdf")


def resolve_latest_version(doc_id_no_ver: str) -> str:
    """Hit the arXiv abs page to resolve the latest vN if no version is present."""
    url = ARXIV_ABS + doc_id_no_ver
    r = SESSION.get(url, timeout=20)
    r.raise_for_status()
    m = re.search(r"arXiv:(\d{4}\.\d{4,5}v\d+)", r.text)
    if m:
        return m.group(1)
    return doc_id_no_ver + "v1"


def url_from_row(doc_id: str, pdf_url: str | None, force_version_resolve: bool) -> tuple[str, str]:
    """
    Returns (resolved_id_with_version, pdf_url_to_download).
    Priority:
      1) use pdf_url from DB if it looks like a PDF
      2) else construct via doc_id (and resolve vN if requested)
    """
    if pdf_url and pdf_url.lower().endswith(".pdf"):
        # try to extract id with version from the url (best-effort)
        m = re.search(r"/pdf/(\d{4}\.\d{4,5}v\d+)\.pdf", pdf_url)
        if m:
            return m.group(1), pdf_url

    # build from doc_id
    if "v" in doc_id:
        resolved = doc_id
    else:
        resolved = resolve_latest_version(doc_id) if force_version_resolve else (doc_id + "v1")
    return resolved, f"{ARXIV_PDF}{resolved}.pdf"


def already_ok(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def download_pdf(resolved_id: str, url: str, out_path: str) -> None:
    print(f"[fetch] GET {url}")
    r = SESSION.get(url, stream=True, timeout=60)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "")
    if "pdf" not in ctype.lower():
        body = r.content[:500].decode("utf-8", errors="ignore")
        raise RuntimeError(f"Unexpected Content-Type '{ctype}' for {url}\nBody preview:\n{body}")

    tmp = out_path + ".part"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 15):
            if chunk:
                f.write(chunk)
    os.replace(tmp, out_path)
    print(f"[fetch] saved: {out_path}")


def fetch_rows(limit: int | None, only_missing_files: bool, only_no_chunks: bool, out_dir: str):
    """
    Pull doc_id + pdf_url from Postgres.
    - only_missing_files: filter to those whose PDF file is not present locally
    - only_no_chunks:     filter to those that have zero rows in chunks
    """
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS
    )
    try:
        with conn, conn.cursor() as cur:
            # Base select
            q = """
            SELECT d.doc_id, COALESCE(NULLIF(d.pdf_url, ''), NULL) AS pdf_url
            FROM documents d
            """
            # Only-no-chunks: left join chunks and keep those with count 0
            if only_no_chunks:
                q = """
                SELECT d.doc_id, COALESCE(NULLIF(d.pdf_url, ''), NULL) AS pdf_url
                FROM documents d
                LEFT JOIN (
                  SELECT doc_id, COUNT(*) AS c
                  FROM chunks
                  GROUP BY doc_id
                ) ch ON ch.doc_id = d.doc_id
                WHERE COALESCE(ch.c, 0) = 0
                """

            if limit:
                q += " LIMIT %s"

            if limit:
                cur.execute(q, (limit,))
            else:
                cur.execute(q)

            rows = cur.fetchall()

        # If only_missing_files, filter by local file presence
        if only_missing_files:
            filtered = []
            for doc_id, pdf_url in rows:
                # choose a likely local path (assume v1 if no version known yet)
                guess_path = local_pdf_path(out_dir, doc_id)
                if not already_ok(guess_path):
                    filtered.append((doc_id, pdf_url))
            rows = filtered

        return rows
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description="Fetch arXiv PDFs for doc_ids stored in Postgres 'documents'.")
    ap.add_argument("--out", default=OUT_DIR_DEFAULT, help=f"Output dir (default: {OUT_DIR_DEFAULT})")
    ap.add_argument("--limit", type=int, default=None, help="Max number of docs to process")
    ap.add_argument("--force", action="store_true", help="Re-download even if local file exists")
    ap.add_argument("--only-missing-files", action="store_true", help="Only docs whose local PDFs are missing")
    ap.add_argument("--only-no-chunks", action="store_true", help="Only docs that have zero chunks in PG")
    ap.add_argument("--resolve-version", action="store_true", help="Resolve latest vN for doc_ids without version via /abs/ page")
    ap.add_argument("--sleep", type=float, default=SLEEP_S, help="Sleep seconds between downloads")
    args = ap.parse_args()

    ensure_dir(args.out)

    rows = fetch_rows(
        limit=args.limit,
        only_missing_files=args.only_missing_files,
        only_no_chunks=args.only_no_chunks,
        out_dir=args.out
    )

    if not rows:
        print("[fetch] nothing to do.")
        return

    ok = 0
    for (doc_id, pdf_url) in rows:
        if not is_valid_id(doc_id):
            print(f"[fetch] skip invalid doc_id format: {doc_id}")
            continue

        resolved_id, url = url_from_row(doc_id, pdf_url, args.resolve_version)
        out_path = local_pdf_path(args.out, resolved_id)

        if already_ok(out_path) and not args.force:
            print(f"[fetch] exists, skip: {out_path}")
            ok += 1
            continue

        try:
            download_pdf(resolved_id, url, out_path)
            ok += 1
        except Exception as e:
            print(f"[fetch] ERROR {doc_id}: {e}")

        time.sleep(args.sleep)

    print(f"[fetch] done. {ok}/{len(rows)} processed.")


if __name__ == "__main__":
    main()
