from fastapi import FastAPI
from src.routers import health,search,chunks_search,search_v2, search_hybrid, ask, ingest_pdf
app = FastAPI(
    title="Paper Curator API",
    version="0.1.0",
    description="A backend for search, RAG, and research ingestion."
)


app.include_router(health.router)
app.include_router(search.router)
app.include_router(chunks_search.router)
app.include_router(search_v2.router)
app.include_router(search_hybrid.router)
app.include_router(ask.router)
app.include_router(ingest_pdf.router)

