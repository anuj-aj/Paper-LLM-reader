from pydantic import BaseModel
from typing import List, Optional

class SearchHit(BaseModel):
    doc_id: str
    title: str
    authors: Optional[str] = None
    year: Optional[int] = None
    abstract: Optional[str] = None
    score: float

class SearchResponse(BaseModel):
    total: int
    hits: List[SearchHit]
