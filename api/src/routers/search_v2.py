from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import os, requests

router = APIRouter(prefix="/search", tags=["search"])

# --------- Env / defaults ---------
OS_HOST   = os.getenv("OPENSEARCH_HOST", "localhost")
OS_PORT   = os.getenv("OPENSEARCH_PORT", "9200")
OS_ALIAS  = os.getenv("OPENSEARCH_CHUNKS_V2_INDEX", "chunks_v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "host.docker.internal")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --------- Schemas ---------
class VectorHit(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    score: float
    content: str
    page_from: Optional[int] = None
    page_to: Optional[int] = None

class VectorSearchResponse(BaseModel):
    query: str
    provider: str
    model: str
    k: int
    hits: List[VectorHit]

# --------- Embedding clients ---------
def embed_with_ollama(text: str, model: str) -> List[float]:
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings"
    r = requests.post(url, json={"model": model, "prompt": text}, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Ollama error {r.status_code}: {r.text}")
    vec = r.json().get("embedding", [])
    if not isinstance(vec, list) or len(vec) != EMBED_DIM:
        raise HTTPException(status_code=500, detail=f"Unexpected embedding dim {len(vec)} (expected {EMBED_DIM})")
    return vec

def embed_with_openai(text: str, model: str) -> List[float]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    r = requests.post(url, headers=headers, json={"model": model, "input": text}, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenAI error {r.status_code}: {r.text}")
    vec = r.json()["data"][0]["embedding"]
    if not isinstance(vec, list) or len(vec) != EMBED_DIM:
        raise HTTPException(status_code=500, detail=f"Unexpected embedding dim {len(vec)} (expected {EMBED_DIM})")
    return vec

def embed_query(q: str, provider: str, model: str) -> List[float]:
    if provider == "ollama":
        return embed_with_ollama(q, model)
    elif provider == "openai":
        return embed_with_openai(q, model)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{provider}'")

# --------- OpenSearch KNN ---------
def knn_search(vec: List[float], k: int):
    base = f"http://{OS_HOST}:{OS_PORT}"
    payload = {
        "size": k,
        "query": { "knn": { "embedding": { "vector": vec, "k": k } } }
    }
    r = requests.post(f"{base}/{OS_ALIAS}/_search", json=payload, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenSearch error {r.status_code}: {r.text}")
    return r.json().get("hits", {}).get("hits", [])

# --------- Route ---------
@router.get("/vector", response_model=VectorSearchResponse)
def vector_search(
    q: str = Query(..., min_length=2, description="Natural-language query"),
    k: int = Query(5, ge=1, le=50),
    provider: str = Query(os.getenv("EMBED_PROVIDER", "ollama"), pattern="^(ollama|openai)$"),
    model: str = Query(OLLAMA_MODEL, description="Embedding model (ollama: nomic-embed-text; openai: text-embedding-3-small/large)")
):
    vec = embed_query(q, provider, model)
    hits = knn_search(vec, k)

    out = []
    for h in hits:
        src = h.get("_source", {})
        out.append(VectorHit(
            chunk_id = src.get("chunk_id") or h.get("_id"),
            doc_id   = src.get("doc_id", ""),
            chunk_index = src.get("chunk_index", 0),
            score    = float(h.get("_score", 0.0)),
            content  = src.get("content", "")[:2000],
            page_from= src.get("page_from"),
            page_to  = src.get("page_to"),
        ))

    return VectorSearchResponse(query=q, provider=provider, model=model, k=k, hits=out)
