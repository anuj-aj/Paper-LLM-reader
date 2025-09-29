import os
import requests

OS_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OS_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OS_INDEX = os.getenv("OPENSEARCH_INDEX_NAME", "papers_v1")


def search_bm25(query: str, size: int = 5, from_: int = 0) -> dict:
    """
    Uses requests to call OpenSearch _search endpoint with BM25 multi_match.
    """
    url = f"http://{OS_HOST}:{OS_PORT}/{OS_INDEX}/_search"

    body = {
        "from": from_,
        "size": size,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title^2", "abstract"]
            }
        }
    }

    try:
        response = requests.post(url, json=body, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise RuntimeError(f"OpenSearch error: {e}")
