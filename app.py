
import tempfile, threading, time, logging, itertools, json
import streamlit as st
import pandas as pd
from pandas.errors import ParserError
import httpx                                 
from config   import Config
from pathlib import Path   
from executor import run as run_pipeline
from utils.logger import get_stream_handler, remove_stream_handler, init_root
init_root()

# ──────────────────────────  QUICK LLM PING  ────────────────────────────────
def ping_llm(base_url: str, api_key: str, model: str) -> tuple[bool, str]:
    """Return (ok?, message) after one 1-token /chat/completions call."""
    try:
        url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
        r = httpx.post(url, json=payload, headers=headers, timeout=5)
        return (r.status_code == 200,
                f"{r.status_code} {r.reason_phrase or r.text[:60]}")
    except Exception as e:
        return False, str(e)

# ──────────────────────────────  PAGE LAYOUT  ────────────────────────────────
st.set_page_config(page_title="Data-Augmentation Toolkit", layout="wide")
st.title("📚 Data-Augmentation Toolkit")

# ----- SIDEBAR ---------------------------------------------------------------
st.sidebar.header("LLM connection")
model_name = st.sidebar.text_input(
    "Model name", "")
api_key  = st.sidebar.text_input("API key", type="password")
base_url = st.sidebar.text_input(
    "Base URL", "")

# NEW ② ── health-check button (no layout change)
if st.sidebar.button("🔌 Health-check"):
    with st.spinner("Pinging LLM…"):
        ok, msg = ping_llm(base_url, api_key, model_name)
    (st.sidebar.success if ok else st.sidebar.error)(msg)

st.sidebar.header("Generation mode")
mode = st.sidebar.radio(
    "Select task",
    ("single-sft", "multi-sft", "single-align", "multi-align"),
    format_func=lambda m: {
        "single-sft":   "🗨️  Single-turn SFT",
        "multi-sft":    "💬 Multi-turn SFT",
        "single-align": "🎯 Single-turn Alignment",
        "multi-align":  "🔄 Multi-turn Alignment",
    }[m],
)

# … (everything below is **identical** to your current file) …

def _is_valid_jsonl(buf):
    buf.seek(0)
    for i, line in enumerate(itertools.islice(buf, 50), 1):
        try: json.loads(line)
        except json.JSONDecodeError as e:
            return False, f"line {i}: {e.msg}"
    return True, None

def _validate_uploaded_file(uploaded_file, mode: str) -> bool:
    if uploaded_file is None:
        st.error("📂 Please select the file first."); return False
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".jsonl"):
            ok, reason = _is_valid_jsonl(uploaded_file)
            if not ok: st.error(f"❌ JSONL broken – {reason}"); return False
        elif name.endswith(".csv"):
            uploaded_file.seek(0)
            try: pd.read_csv(uploaded_file, nrows=5)
            except (ParserError, UnicodeDecodeError) as e:
                st.error(f"❌ CSV broken: {e}"); return False
        else:
            st.error("❌ .jsonl / .csv only"); return False
    finally: uploaded_file.seek(0)
    return True

workers = st.sidebar.slider("Threads (concurrent requests)", 1, 16, 8)

# ----- MAIN PANEL ------------------------------------------------------------
uploaded = st.file_uploader("Upload JSONL / JSON file", type=["jsonl", "json"])
run_btn  = st.button("🚀 Run")

log_box      = st.empty()
download_box = st.empty()

def launch_job(uploaded_file):
    import tempfile, logging, time, json, itertools  # local scope
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.getvalue()); input_path = tmp.name

    cfg = Config(model_name=model_name, api_key=api_key, base_url=base_url,
                 mode=mode, input_file=input_path, max_workers=workers)

    hdl, buf = get_stream_handler(); results = {}
    def _worker():
        try:
            _csv, json_path = run_pipeline(cfg); results["json"] = json_path
        finally:
            remove_stream_handler(hdl)

    threading.Thread(target=_worker, daemon=True).start()
    while hdl in logging.getLogger().handlers:
        time.sleep(0.5); log_box.code(buf.getvalue())
    log_box.code(buf.getvalue())

    if results.get("json"):
        download_box.download_button("⬇ Download JSONL",
            data=open(results["json"], "rb").read(),
            file_name=Path(results["json"]).name,
            mime="application/json")

if run_btn and uploaded and _validate_uploaded_file(uploaded, mode):
    launch_job(uploaded)
