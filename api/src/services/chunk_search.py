from typing import List, Optional
from ..repositories.opensearch_chunks import search_chunks_bm25
from ..schemas.chunks import ChunkSearchResponse, ChunkHit

def chunks_search_service(q: str, size: int = 5, offset: int = 0, doc_id: Optional[str] = None) -> ChunkSearchResponse:
    res = search_chunks_bm25(q, size=size, from_=offset, doc_id=doc_id)
    hits = res.get("hits", {})
    total = hits.get("total", {}).get("value", 0) if isinstance(hits.get("total"), dict) else hits.get("total", 0)

    out: List[ChunkHit] = []
    for h in hits.get("hits", []):
        src = h.get("_source", {}) or {}
        hl = h.get("highlight", {}).get("content", None)
        out.append(ChunkHit(
            chunk_id=src.get("chunk_id", h.get("_id", "")),
            doc_id=src.get("doc_id", ""),
            chunk_index=src.get("chunk_index", 0),
            page_from=src.get("page_from"),
            page_to=src.get("page_to"),
            content=src.get("content", ""),
            score=h.get("_score", 0.0),
            highlights=hl
        ))
    return ChunkSearchResponse(total=total, hits=out)
