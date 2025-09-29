from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
import os, requests

router = APIRouter(prefix="/search", tags=["search"])

# -------- env --------
OS_HOST   = os.getenv("OPENSEARCH_HOST", "localhost")
OS_PORT   = os.getenv("OPENSEARCH_PORT", "9200")
IDX_BM25  = os.getenv("OPENSEARCH_CHUNKS_V1_INDEX", "chunks_v1")
IDX_VEC   = os.getenv("OPENSEARCH_CHUNKS_V2_INDEX", "chunks_v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# Embedding (same as vector route)
PROVIDER     = os.getenv("EMBED_PROVIDER", "ollama")   # "ollama" | "openai"
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "host.docker.internal")
OLLAMA_PORT  = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# -------- models --------
class HybridHit(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    content: str
    page_from: Optional[int] = None
    page_to: Optional[int] = None
    score: float
    bm25_rank: Optional[int] = None
    knn_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    knn_score: Optional[float] = None

class HybridResponse(BaseModel):
    query: str
    k: int
    provider: str
    model: str
    hits: List[HybridHit]

# -------- helpers --------
def _base_url() -> str:
    return f"http://{OS_HOST}:{OS_PORT}"

def _embed_with_ollama(text: str) -> List[float]:
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings"
    r = requests.post(url, json={"model": OLLAMA_MODEL, "prompt": text}, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Ollama error {r.status_code}: {r.text}")
    vec = r.json().get("embedding", [])
    if not isinstance(vec, list) or len(vec) != EMBED_DIM:
        raise HTTPException(status_code=500, detail=f"Ollama embedding dim {len(vec)} != {EMBED_DIM}")
    return vec

def _embed_with_openai(text: str, model: str) -> List[float]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    r = requests.post(url, headers=headers, json={"model": model, "input": text}, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenAI error {r.status_code}: {r.text}")
    vec = r.json()["data"][0]["embedding"]
    if not isinstance(vec, list) or len(vec) != EMBED_DIM:
        raise HTTPException(status_code=500, detail=f"OpenAI embedding dim {len(vec)} != {EMBED_DIM}")
    return vec

def embed_query(q: str, provider: str, model: str) -> List[float]:
    if provider == "ollama":
        return _embed_with_ollama(q)
    elif provider == "openai":
        return _embed_with_openai(q, model)
    raise HTTPException(status_code=400, detail=f"Unknown provider '{provider}'")

def bm25_search(q: str, k: int) -> List[dict]:
    payload = {
        "size": k,
        "query": { "match": { "content": { "query": q } } },
        "_source": ["chunk_id","doc_id","chunk_index","content","page_from","page_to"]
    }
    r = requests.post(f"{_base_url()}/{IDX_BM25}/_search", json=payload, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenSearch BM25 error {r.status_code}: {r.text}")
    return r.json().get("hits", {}).get("hits", [])

def knn_search(vec: List[float], k: int) -> List[dict]:
    payload = {
        "size": k,
        "query": { "knn": { "embedding": { "vector": vec, "k": k } } },
        "_source": ["chunk_id","doc_id","chunk_index","content","page_from","page_to"]
    }
    r = requests.post(f"{_base_url()}/{IDX_VEC}/_search", json=payload, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenSearch KNN error {r.status_code}: {r.text}")
    return r.json().get("hits", {}).get("hits", [])

def rrf_fuse(bm25_hits: List[dict], knn_hits: List[dict], k: int, k_rrf: int = 60) -> List[HybridHit]:
    """Reciprocal Rank Fusion: score = Σ 1 / (k_rrf + rank)."""
    def key(h): return h.get("_source", {}).get("chunk_id") or h.get("_id")
    pool: Dict[str, Dict] = {}

    # Index BM25
    for rank, h in enumerate(bm25_hits, start=1):
        cid = key(h)
        pool.setdefault(cid, {"bm25_rank": rank})
        pool[cid]["bm25_score"] = float(h.get("_score", 0.0))

    # Index KNN
    for rank, h in enumerate(knn_hits, start=1):
        cid = key(h)
        pool.setdefault(cid, {"knn_rank": rank})
        pool[cid]["knn_score"] = float(h.get("_score", 0.0))

    fused: List[Tuple[str, float]] = []
    for cid, meta in pool.items():
        r = 0.0
        if "bm25_rank" in meta:
            r += 1.0 / (k_rrf + meta["bm25_rank"])
        if "knn_rank" in meta:
            r += 1.0 / (k_rrf + meta["knn_rank"])
        fused.append((cid, r))

    # Sort by fused score desc
    fused.sort(key=lambda x: x[1], reverse=True)

    # Build HybridHit with sources
    src_lookup = {}
    for h in bm25_hits + knn_hits:
        cid = key(h)
        if cid not in src_lookup:
            src_lookup[cid] = h

    results: List[HybridHit] = []
    for cid, fused_score in fused[:k]:
        h = src_lookup[cid]
        s = h.get("_source", {})
        results.append(HybridHit(
            chunk_id   = s.get("chunk_id") or h.get("_id"),
            doc_id     = s.get("doc_id",""),
            chunk_index= s.get("chunk_index", 0),
            content    = s.get("content","")[:2000],
            page_from  = s.get("page_from"),
            page_to    = s.get("page_to"),
            score      = fused_score,
            bm25_rank  = pool[cid].get("bm25_rank"),
            knn_rank   = pool[cid].get("knn_rank"),
            bm25_score = pool[cid].get("bm25_score"),
            knn_score  = pool[cid].get("knn_score"),
        ))
    return results

# -------- route --------
@router.get("/hybrid", response_model=HybridResponse)
def hybrid_search(
    q: str = Query(..., min_length=2),
    k: int = Query(8, ge=1, le=50),
    provider: str = Query(PROVIDER, pattern="^(ollama|openai)$"),
    model: str = Query(os.getenv("EMBED_MODEL", "nomic-embed-text"),
                       description="Embedding model: ollama('nomic-embed-text') or openai('text-embedding-3-small')")
):
    vec = embed_query(q, provider, model)
    bm25_hits = bm25_search(q, k=k*2)   # pull a bit more, fuse later
    knn_hits  = knn_search(vec, k=k*2)
    fused = rrf_fuse(bm25_hits, knn_hits, k=k, k_rrf=60)
    return HybridResponse(query=q, k=k, provider=provider, model=model, hits=fused)
