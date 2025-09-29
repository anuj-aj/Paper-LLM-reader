from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os, requests, textwrap

router = APIRouter(tags=["ask"])

# -------- ENV --------
OS_HOST   = os.getenv("OPENSEARCH_HOST", "localhost")
OS_PORT   = os.getenv("OPENSEARCH_PORT", "9200")
IDX_BM25  = os.getenv("OPENSEARCH_CHUNKS_V1_INDEX", "chunks_v1")
IDX_VEC   = os.getenv("OPENSEARCH_CHUNKS_V2_INDEX", "chunks_v2")

EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "ollama")  # "ollama" | "openai"
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

GEN_PROVIDER = os.getenv("GEN_PROVIDER", "ollama")      # "ollama" | "openai"
GEN_MODEL    = os.getenv("GEN_MODEL", "llama3.1")       # adjust to your model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# -------- Schemas --------
class AskRequest(BaseModel):
    q: str
    k: int = 8
    max_ctx_chars: int = 6000
    embed_provider: Optional[str] = None
    embed_model: Optional[str] = None
    gen_provider: Optional[str] = None
    gen_model: Optional[str] = None

class Source(BaseModel):
    chunk_id: str
    doc_id: str
    page_from: Optional[int] = None
    page_to: Optional[int] = None
    snippet: str

class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]

# -------- Embeddings --------
def embed_with_ollama(text: str, model: str) -> List[float]:
    url = f"http://{os.getenv('OLLAMA_HOST','host.docker.internal')}:{int(os.getenv('OLLAMA_PORT','11434'))}/api/embeddings"
    r = requests.post(url, json={"model": model, "prompt": text}, timeout=60)
    if r.status_code != 200:
        raise HTTPException(502, f"Ollama embeddings error {r.status_code}: {r.text}")
    vec = r.json().get("embedding", [])
    if not isinstance(vec, list) or len(vec) != EMBED_DIM:
        raise HTTPException(500, f"Unexpected embedding dim {len(vec)} (expected {EMBED_DIM})")
    return vec

def embed_with_openai(text: str, model: str) -> List[float]:
    if not OPENAI_API_KEY:
        raise HTTPException(400, "OPENAI_API_KEY not set")
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    r = requests.post(url, headers=headers, json={"model": model, "input": text}, timeout=60)
    if r.status_code != 200:
        raise HTTPException(502, f"OpenAI embeddings error {r.status_code}: {r.text}")
    vec = r.json()["data"][0]["embedding"]
    if not isinstance(vec, list) or len(vec) != EMBED_DIM:
        raise HTTPException(500, f"Unexpected embedding dim {len(vec)} (expected {EMBED_DIM})")
    return vec

def embed_query(q: str, provider: str, model: str) -> List[float]:
    if provider == "ollama":
        return embed_with_ollama(q, model)
    if provider == "openai":
        return embed_with_openai(q, model)
    raise HTTPException(400, f"Unknown embed provider '{provider}'")

# -------- Retrieval (BM25 + KNN + RRF) --------
def os_base() -> str:
    return f"http://{OS_HOST}:{OS_PORT}"

def bm25(q: str, size: int) -> List[dict]:
    payload = {
        "size": size,
        "query": {"match": {"content": {"query": q}}},
        "_source": ["chunk_id","doc_id","chunk_index","content","page_from","page_to"]
    }
    r = requests.post(f"{os_base()}/{IDX_BM25}/_search", json=payload, timeout=60)
    if r.status_code != 200:
        raise HTTPException(502, f"OpenSearch BM25 error {r.status_code}: {r.text}")
    return r.json()["hits"]["hits"]

def knn(vec: List[float], size: int) -> List[dict]:
    payload = {
        "size": size,
        "query": {"knn": {"embedding": {"vector": vec, "k": size}}},
        "_source": ["chunk_id","doc_id","chunk_index","content","page_from","page_to"]
    }
    r = requests.post(f"{os_base()}/{IDX_VEC}/_search", json=payload, timeout=60)
    if r.status_code != 200:
        raise HTTPException(502, f"OpenSearch KNN error {r.status_code}: {r.text}")
    return r.json()["hits"]["hits"]

def rrf(bm25_hits: List[dict], knn_hits: List[dict], topk: int, k_rrf: int = 60) -> List[dict]:
    def cid(h): return h.get("_source",{}).get("chunk_id") or h.get("_id")
    pool = {}
    for rank, h in enumerate(bm25_hits, 1):
        d = pool.setdefault(cid(h), {"hit": h})
        d["bm25_rank"] = rank
        d["bm25_score"] = float(h.get("_score", 0.0))
    for rank, h in enumerate(knn_hits, 1):
        d = pool.setdefault(cid(h), {"hit": h})
        d["knn_rank"] = rank
        d["knn_score"] = float(h.get("_score", 0.0))
    fused = []
    for k, d in pool.items():
        s = 0.0
        if "bm25_rank" in d: s += 1.0 / (k_rrf + d["bm25_rank"])
        if "knn_rank"  in d: s += 1.0 / (k_rrf + d["knn_rank"])
        fused.append((k, s, d["hit"]))
    fused.sort(key=lambda x: x[1], reverse=True)
    return [h for _,_,h in fused[:topk]]

# -------- Generation --------
def generate_with_ollama(prompt: str, model: str) -> str:
    url = f"http://{os.getenv('OLLAMA_HOST','host.docker.internal')}:{int(os.getenv('OLLAMA_PORT','11434'))}/api/generate"
    r = requests.post(url, json={"model": model, "prompt": prompt, "stream": False}, timeout=120)
    if r.status_code != 200:
        raise HTTPException(502, f"Ollama generate error {r.status_code}: {r.text}")
    return r.json().get("response", "")

def generate_with_openai(prompt: str, model: str) -> str:
    if not OPENAI_API_KEY:
        raise HTTPException(400, "OPENAI_API_KEY not set")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise assistant that answers ONLY using the provided context. If the answer is not in the context, say you don't know."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        raise HTTPException(502, f"OpenAI generate error {r.status_code}: {r.text}")
    return r.json()["choices"][0]["message"]["content"]

def build_prompt(question: str, contexts: List[Source]) -> str:
    # Simple, token-safe-ish prompt by char budget
    ctx_blocks = []
    for s in contexts:
        cite = f"[{s.doc_id} p.{s.page_from or '?'}]"
        snippet = s.snippet.replace("\n", " ").strip()
        ctx_blocks.append(f"{cite} {snippet}")
    context_text = "\n\n".join(ctx_blocks)
    instr = (
        "Answer the question using ONLY the context below. "
        "Cite sources inline like [doc_id p.X]. If the answer is not answerable from context, say \"I don't know based on the provided documents.\""
    )
    return textwrap.dedent(f"""\
    {instr}

    Question:
    {question}

    Context:
    {context_text}
    """)

# -------- Route --------
@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest):
    q = body.q.strip()
    if not q:
        raise HTTPException(400, "Empty question")

    # Providers / models (defaults come from env)
    emb_provider = body.embed_provider or EMBED_PROVIDER
    emb_model    = body.embed_model or EMBED_MODEL
    gen_provider = body.gen_provider or GEN_PROVIDER
    gen_model    = body.gen_model or GEN_MODEL

    # 1) Embed query
    vec = embed_query(q, emb_provider, emb_model)

    # 2) Retrieve (BM25 + KNN + RRF)
    bm25_hits = bm25(q, size=body.k * 2)
    knn_hits  = knn(vec, size=body.k * 2)
    fused     = rrf(bm25_hits, knn_hits, topk=body.k, k_rrf=60)

    # 3) Build source list w/ char budget
    sources: List[Source] = []
    total = 0
    for h in fused:
        s = h.get("_source", {})
        snippet = (s.get("content") or "")[:800]
        if not snippet:
            continue
        entry = Source(
            chunk_id   = s.get("chunk_id") or h.get("_id"),
            doc_id     = s.get("doc_id",""),
            page_from  = s.get("page_from"),
            page_to    = s.get("page_to"),
            snippet    = snippet
        )
        if total + len(entry.snippet) > body.max_ctx_chars:
            break
        sources.append(entry)
        total += len(entry.snippet)
    if not sources:
        return AskResponse(question=q, answer="I don't know based on the provided documents.", sources=[])

    # 4) Prompt & generate
    prompt = build_prompt(q, sources)
    if gen_provider == "ollama":
        answer = generate_with_ollama(prompt, gen_model)
    elif gen_provider == "openai":
        answer = generate_with_openai(prompt, gen_model)
    else:
        raise HTTPException(400, f"Unknown gen provider '{gen_provider}'")

    return AskResponse(question=q, answer=answer.strip(), sources=sources)
