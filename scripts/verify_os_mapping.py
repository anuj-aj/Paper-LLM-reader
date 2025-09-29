import os, requests, pprint

base = f"http://{os.getenv('OPENSEARCH_HOST', 'localhost')}:{os.getenv('OPENSEARCH_PORT', '9200')}"
alias = os.getenv('OPENSEARCH_CHUNKS_V2_INDEX', 'chunks_v2')

print("\n[Aliases]")
print(requests.get(f"{base}/_alias/{alias}", timeout=10).text)

print("\n[Mapping]")
pprint.pp(requests.get(f"{base}/{alias}/_mapping", timeout=10).json())
