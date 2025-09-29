# api/src/services/health_service.py

import psycopg2
import requests
import os

def check_postgres():
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "curator"),
            user=os.getenv("POSTGRES_USER", "curator_user"),
            password=os.getenv("POSTGRES_PASSWORD", "curator_pwd"),
            connect_timeout=3
        )
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                return "ok"
    except Exception as e:
        return f"error: {str(e)}"

def check_opensearch():
    try:
        host = os.getenv("OPENSEARCH_HOST", "localhost")
        port = int(os.getenv("OPENSEARCH_PORT", "9200"))
        url = f"http://{host}:{port}/_cluster/health"
        r = requests.get(url, timeout=3)
        r.raise_for_status()
        status = r.json().get("status", "unknown")
        return status
    except Exception as e:
        return f"error: {str(e)}"
