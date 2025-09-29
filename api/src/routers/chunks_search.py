from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from ..services.chunk_search import chunks_search_service
from ..schemas.chunks import ChunkSearchResponse

router = APIRouter(prefix="/search/chunks", tags=["Search (Chunks)"])

@router.get("", response_model=ChunkSearchResponse)
def search_chunks(
    q: str = Query(..., min_length=1, description="Keyword query over chunk content"),
    size: int = Query(5, ge=1, le=50),
    offset: int = Query(0, ge=0),
    doc_id: Optional[str] = Query(None, description="Filter to a specific document ID")
):
    try:
        return chunks_search_service(q=q, size=size, offset=offset, doc_id=doc_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Chunks search backend error: {e}")
