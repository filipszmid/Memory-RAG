import streamlit as st
import httpx
import json
import time
import os

# --- Page Config ---
st.set_page_config(
    page_title="Memory Agent Dashboard",
    layout="wide",
)

# --- Constants ---
API_URL = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_USER_ID = "filip_user"

# --- Provider Options ---
PROVIDER_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    "gemini": ["gemini-1.5-pro", "gemini-1.5-flash"],
    "ollama": ["llama3", "mistral", "phi3"]
}

# --- API Helpers ---
def call_chat(user_id, message, config, provider, model, temperature=0.7, top_p=1.0):
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{API_URL}/chat",
                json={
                    "user_id": user_id,
                    "message": message,
                    "config": config,
                    "provider": provider,
                    "model": model,
                    "temperature": temperature,
                    "top_p": top_p
                }
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        st.error(f"Chat API Error: {e}")
        return None

def get_facts(user_id):
    try:
        with httpx.Client() as client:
            resp = client.get(f"{API_URL}/facts/{user_id}")
            resp.raise_for_status()
            return resp.json().get("facts", [])
    except Exception as e:
        st.error(f"Facts API Error: {e}")
        return []

def delete_fact(fact_id):
    try:
        with httpx.Client() as client:
            resp = client.delete(f"{API_URL}/facts/{fact_id}")
            resp.raise_for_status()
            return True
    except Exception as e:
        st.error(f"Delete API Error: {e}")
        return False

def trigger_extraction(user_id, messages, provider, model):
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{API_URL}/extract",
                json={
                    "user_id": user_id,
                    "messages": messages,
                    "provider": provider,
                    "model": model
                }
            )
            resp.raise_for_status()
            return resp.json().get("job_id")
    except Exception as e:
        st.error(f"Extraction Error: {e}")
        return None

def check_job_status(job_id):
    try:
        with httpx.Client() as client:
            resp = client.get(f"{API_URL}/extract/{job_id}")
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        st.error(f"Status Check Error: {e}")
        return None

# --- UI Layout ---
st.title("Memory Agent Control Center")
st.markdown("---")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("System Configuration")
    user_id = st.text_input("User ID", value=DEFAULT_USER_ID)
    
    ds_mode = st.toggle("Data Science Mode", value=False, help="Enable advanced RAG tuning and observability.")
    
    st.subheader("Model & Provider")
    sel_provider = st.selectbox("Provider", options=list(PROVIDER_MODELS.keys()), index=0)
    sel_model = st.selectbox("Model", options=PROVIDER_MODELS[sel_provider], index=0)
    
    if ds_mode:
        st.divider()
        st.subheader("LLM Hyperparameters")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.1)
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
    else:
        temperature = 0.7
        top_p = 1.0
        
    st.subheader("Algorithm Settings")
    strategy = st.selectbox(
        "Retrieval Strategy",
        options=["hybrid", "vector", "bm25"],
        index=0,
        help="How facts are retrieved from ElasticSearch."
    )
    
    if ds_mode and strategy == "hybrid":
        alpha = st.slider("Hybrid Alpha", min_value=0.0, max_value=1.0, value=0.5, step=0.05, 
                          help="1.0 = Pure Vector, 0.0 = Pure Keyword")
    else:
        alpha = 0.5
        
    top_k = st.slider("Top K Facts", min_value=1, max_value=20, value=5)
    threshold = st.number_input("RAG Threshold", min_value=1, max_value=100, value=20, help="Inject full profile if facts < this.")
    
    st.subheader("Advanced Features")
    rerank = st.toggle("Enable Reranking", value=False, help="Use LLM to rerank retrieved facts for premium relevance.")
    if rerank and ds_mode:
        rerank_model = st.selectbox("Rerank Model", options=["gpt-4o-mini", "gemini-1.5-flash", "llama3"])
    else:
        rerank_model = "gpt-4o-mini"
    
    st.divider()
    if st.button("Process Memory", use_container_width=True, help="Extract facts from the current conversation and save to ElasticSearch."):
        if "messages" in st.session_state and st.session_state.messages:
            with st.status("Extracting Facts...", expanded=True) as status:
                clean_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                job_id = trigger_extraction(user_id, clean_messages, sel_provider, sel_model)
                
                if job_id:
                    status.update(label=f"Job queued (ID: {job_id}). Waiting for worker...", state="running")
                    
                    # Polling
                    finished = False
                    while not finished:
                        time.sleep(2)
                        job_res = check_job_status(job_id)
                        if not job_res:
                            break
                            
                        job_status = job_res.get("status")
                        if job_status == "finished":
                            new_facts_count = len(job_res.get("new_facts", []))
                            stored_count = len(job_res.get("stored_ids", []))
                            st.success(f"Successfully processed! Extracted {new_facts_count} facts. (Stored/Updated {stored_count} in ES)")
                            st.session_state.facts = get_facts(user_id)
                            finished = True
                        elif job_status == "failed":
                            st.error("Background extraction job failed.")
                            finished = True
                        else:
                            status.update(label=f"Worker is processing memory... (Status: {job_status})")
                
                status.update(label="Memory Processing Complete!", state="complete", expanded=False)
        else:
            st.warning("No messages to process.")

    if st.button("Reset Session", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Tabs ---
tab1, tab2 = st.tabs(["Chat Interface", "Knowledge Base"])

with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "context" in message and message["context"]:
                with st.expander("Context used"):
                    for fact in message["context"]:
                        st.caption(f"- {fact}")
            
            if ds_mode and message.get("trace"):
                with st.expander("Retrieval Trace"):
                    st.table(message["trace"])

with tab2:
    st.header("Stored Facts")
    if st.button("Refresh Knowledge Base"):
        facts = get_facts(user_id)
        st.session_state.facts = facts
    
    if "facts" not in st.session_state:
        st.session_state.facts = get_facts(user_id)
    
    facts = st.session_state.facts
    
    if not facts:
        st.info("No facts stored for this user yet.")
    else:
        for f in facts:
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.markdown(f"**[{f['category'].upper()}]** {f['fact']}")
                st.caption(f"ID: {f['id']} | Confidence: {f['confidence']} | Created: {f['created_at']}")
            with col2:
                if st.button("Delete", key=f"del_{f['id']}"):
                    if delete_fact(f["id"]):
                        st.success("Deleted!")
                        st.session_state.facts = [x for x in st.session_state.facts if x["id"] != f["id"]]
                        st.rerun()
            st.divider()

# --- Chat Input (Pinned to bottom) ---
if prompt := st.chat_input("Enter your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    rag_config = {
        "strategy": strategy,
        "top_k": top_k,
        "threshold": threshold,
        "alpha": alpha,
        "rerank": rerank,
        "reranker_model": rerank_model
    }
    
    result = call_chat(user_id, prompt, rag_config, sel_provider, sel_model, temperature, top_p)
    if result:
        response_text = result["response"]
        context = result.get("context_used", [])
        telemetry = result.get("telemetry", {})
        retrieval_trace = telemetry.get("retrieval_trace", [])
        
        trace_table = []
        for t in retrieval_trace:
            details = t.get("details", {})
            trace_table.append({
                "Fact": t["fact"][:60] + "...",
                "RRF Score": t["score"],
                "Vec Score": details.get("vector_score", "-"),
                "KW Score": details.get("keyword_score", "-")
            })
            
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_text,
            "context": context,
            "trace": trace_table if ds_mode else None
        })
    st.rerun()

# --- Custom Styling ---
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 15px;
        padding: 5px;
        margin-bottom: 5px;
    }
    .stExpander {
        border: none !important;
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)
