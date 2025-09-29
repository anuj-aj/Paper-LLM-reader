from fastapi import APIRouter, HTTPException, Query
from ..services.search import search_service
from ..schemas.search import SearchResponse

router = APIRouter(prefix="/search", tags=["Search"])

@router.get("", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Free-text query"),
    size: int = Query(5, ge=1, le=50),
    offset: int = Query(0, ge=0)
):
    try:
        return search_service(q=q, size=size, offset=offset)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Search backend error: {e}")
