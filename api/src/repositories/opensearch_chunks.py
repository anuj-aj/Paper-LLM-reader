import os
import requests

OS_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OS_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OS_CHUNK_INDEX = os.getenv("OPENSEARCH_CHUNKS_INDEX", "chunks_v1")

def search_chunks_bm25(query: str, size: int = 5, from_: int = 0, doc_id: str | None = None) -> dict:
    """
    BM25 over chunks_v1.content with optional doc_id filter and highlights.
    """
    url = f"http://{OS_HOST}:{OS_PORT}/{OS_CHUNK_INDEX}/_search"

    must = [{"match": {"content": query}}]
    if doc_id:
        must.append({"term": {"doc_id": doc_id}})

    body = {
        "from": from_,
        "size": size,
        "query": {
            "bool": {
                "must": must
            }
        },
        "highlight": {
            "fields": {
                "content": {
                    "fragment_size": 160,
                    "number_of_fragments": 2,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"]
                }
            }
        }
    }

    try:
        resp = requests.post(url, json=body, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"OpenSearch (chunks) error: {e}")
