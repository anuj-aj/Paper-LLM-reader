import os, time, requests, streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TIMEOUT = 120

st.set_page_config(page_title="Paper Reader", layout="wide")
st.title("📚 Paper Reader")

# --- Small helper wrappers ---
def _get(path, params=None):
    r = requests.get(f"{BACKEND_URL}{path}", params=params, timeout=TIMEOUT); r.raise_for_status(); return r.json()
def _post(path, json=None, files=None):
    r = requests.post(f"{BACKEND_URL}{path}", json=json, files=files, timeout=TIMEOUT); r.raise_for_status(); return r.json()

def track_job(job_id: str):
    states = {
        "uploaded":  "✅ Uploaded",
        "parsing":   "🔍 Parsing",
        "chunking":  "🧩 Chunking",
        "embedding": "🧠 Embedding",
        "indexing":  "📇 Indexing",
        "done":      "🎉 Done",
        "error":     "❌ Error",
    }
    with st.status("Processing…", expanded=True) as status:
        pb = st.progress(0); last = None
        while True:
            data = _get(f"/ingest/status/{job_id}")
            s, msg, p = data.get("state",""), data.get("message",""), data.get("progress",0.0)
            if s != last:
                st.write(states.get(s, s), ("– "+msg) if msg else "")
                last = s
            pb.progress(max(0, min(int(p*100), 100)))
            if s in ("done","error"):
                status.update(label="Completed" if s=="done" else "Failed", state="complete" if s=="done" else "error")
                break
            time.sleep(1.0)

# --- Sidebar global settings ---
with st.sidebar:
    st.markdown("### Settings")
    top_k = st.slider("Top‑K", 1, 20, 6)
    st.markdown("**Embeddings**")
    embed_provider = st.selectbox("Provider", ["ollama", "openai"], index=0, help="Where embeddings are computed")
    embed_model = st.text_input("Model", "nomic-embed-text")
    st.caption("Tip: nomic‑embed‑text→768 · text‑embedding‑3‑small→1536")
    st.markdown("**Generation**")
    gen_provider = st.selectbox("Provider ", ["ollama", "openai"], index=0, help="Where answers are generated")
    gen_model = st.text_input("Model ", "llama3:instruct")

# --- Tabs: Upload | Search | Ask ---
tab_upload, tab_search, tab_ask = st.tabs(["📤 Upload", "🔎 Search", "🧠 Ask"])

with tab_upload:
    st.subheader("Add PDFs to your corpus")
    files = st.file_uploader("Drop PDF(s) here", type=["pdf"], accept_multiple_files=True,
                             help="Uploads file(s) and starts: parsing → chunking → embedding → indexing")
    col1, col2 = st.columns([1,1])
    if files and col1.button("Start Upload"):
        for f in files:
            st.write(f"**Uploading:** {f.name}")
            try:
                job = _post("/ingest/upload", files={"file": (f.name, f.getvalue(), "application/pdf")})
                job_id = job.get("job_id")
                if not job_id: st.error("No job_id from backend"); continue
                track_job(job_id)
            except requests.HTTPError as e:
                st.error(f"HTTP {e.response.status_code}\n\n{e.response.text}")
            except Exception as e:
                st.error(str(e))
    with col2: st.caption("Click **Start Upload** after selecting files.")

with tab_search:
    st.subheader("Search your corpus")
    # Mode with concise hover help + caption
    mode = st.radio(
        "Mode",
        ["Vector", "Hybrid"],
        horizontal=True,
        help="Vector: semantic KNN over chunk embeddings • Hybrid: BM25 + Vector fusion"
    )
    st.caption("Vector = meaning-based · Hybrid = balances keywords + semantics")

    q = st.text_input("Query", placeholder="e.g., clip vision transformer")
    if q:
        endpoint = "/search/vector" if mode == "Vector" else "/search/hybrid"
        params = {"q": q, "k": top_k, "provider": embed_provider, "model": embed_model}
        with st.spinner("Searching…"):
            try:
                data = _get(endpoint, params); hits = data.get("hits", [])
            except requests.HTTPError as e:
                st.error(f"HTTP {e.response.status_code}\n\n{e.response.text}"); st.stop()
            except Exception as e:
                st.error(str(e)); st.stop()
        st.success(f"Found {len(hits)} result(s).")
        for i, h in enumerate(hits, 1):
            with st.container(border=True):
                st.markdown(f"**{i}.** doc=`{h.get('doc_id','')}` · chunk={h.get('chunk_index',0)} · score={(h.get('score',0) or 0):.3f}")
                st.write(h.get("content",""))
                st.caption(f"pages {h.get('page_from')}–{h.get('page_to')} • chunk_id={h.get('chunk_id','')}")

with tab_ask:
    st.subheader("Ask questions (RAG)")
    q2 = st.text_input("Your question", placeholder="e.g., How does CLIP encode images?")
    st.caption("RAG flow: retrieve top‑K chunks → prompt LLM → answer with citations")
    if q2:
        payload = {
            "q": q2, "k": top_k,
            "embed_provider": embed_provider, "embed_model": embed_model,
            "gen_provider": gen_provider, "gen_model": gen_model,
        }
        with st.spinner("Thinking…"):
            try:
                out = _post("/ask", json=payload)
            except requests.HTTPError as e:
                st.error(f"HTTP {e.response.status_code}\n\n{e.response.text}"); st.stop()
            except Exception as e:
                st.error(str(e)); st.stop()
        st.markdown("### Answer")
        st.write(out.get("answer","(no answer)"))
        st.markdown("### Sources")
        for s in out.get("sources", []):
            with st.container(border=True):
                st.markdown(f"**{s.get('doc_id','')}** · p.{s.get('page_from')}–{s.get('page_to')}")
                st.write(s.get("snippet",""))
                st.caption(f"chunk_id={s.get('chunk_id','')}")
