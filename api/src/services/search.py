from ..repositories.opensearch import search_bm25
from ..schemas.search import SearchResponse, SearchHit

def search_service(q: str, size: int = 5, offset: int = 0) -> SearchResponse:
    res = search_bm25(q, size=size, from_=offset)
    hits = res.get("hits", {})
    total = hits.get("total", {}).get("value", 0) if isinstance(hits.get("total"), dict) else hits.get("total", 0)
    out_hits = []
    for h in hits.get("hits", []):
        src = h.get("_source", {})
        out_hits.append(SearchHit(
            doc_id=src.get("doc_id", ""),
            title=src.get("title", ""),
            authors=src.get("authors"),
            year=src.get("year"),
            abstract=src.get("abstract"),
            score=h.get("_score", 0.0),
        ))
    return SearchResponse(total=total, hits=out_hits)
