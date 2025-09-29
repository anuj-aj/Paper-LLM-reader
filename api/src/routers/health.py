# api/src/routers/health_router.py

from fastapi import APIRouter
from src.services.health import check_postgres, check_opensearch

router = APIRouter(tags=["System"])

@router.get("/health")
def health_check():
    postgres_ok = check_postgres()
    opensearch_ok = check_opensearch()
    status = "ok" if postgres_ok and opensearch_ok else "degraded"

    return {
        "status": status,
        "postgres": postgres_ok,
        "opensearch": opensearch_ok
    }
