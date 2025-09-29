from pydantic import BaseModel
from typing import List, Optional

class ChunkHit(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    page_from: Optional[int] = None
    page_to: Optional[int] = None
    content: str
    score: float
    highlights: Optional[List[str]] = None

class ChunkSearchResponse(BaseModel):
    total: int
    hits: List[ChunkHit]
