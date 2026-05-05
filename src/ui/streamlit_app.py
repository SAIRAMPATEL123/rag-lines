import os
from datetime import datetime, timedelta, timezone
import requests
import streamlit as st

API_BASE = os.getenv("RAG_API_BASE", "http://localhost:8000/api/v1")

st.set_page_config(page_title="RAG-Lines MVP UI", layout="wide")
st.title("RAG-Lines MVP Web UI")
st.caption(f"Backend: {API_BASE}")

tab1, tab2, tab3 = st.tabs(["Single Query", "Batch Query", "Schedule Query"])

with tab1:
    st.subheader("Ask a question")
    question = st.text_area("Question", height=120)
    top_k = st.slider("Top K", 1, 50, 20)
    include_context = st.checkbox("Include context", value=True)
    if st.button("Run Query"):
        resp = requests.post(
            f"{API_BASE}/query",
            json={"question": question, "top_k": top_k, "include_context": include_context},
            timeout=120,
        )
        if resp.ok:
            data = resp.json()
            st.success("Answer generated")
            st.write(data.get("answer", ""))
            st.json({"document_count": data.get("document_count", 0)})
            if include_context:
                st.text_area("Context used", data.get("context_used", ""), height=220)
        else:
            st.error(resp.text)

with tab2:
    st.subheader("Batch questions")
    lines = st.text_area("One question per line", height=180)
    b_top_k = st.slider("Batch Top K", 1, 50, 10)
    if st.button("Run Batch"):
        questions = [l.strip() for l in lines.splitlines() if l.strip()]
        resp = requests.post(
            f"{API_BASE}/batch-query",
            json={"questions": questions, "top_k": b_top_k, "include_context": False},
            timeout=300,
        )
        if resp.ok:
            st.json(resp.json())
        else:
            st.error(resp.text)

with tab3:
    st.subheader("Schedule one-off query")
    s_question = st.text_area("Scheduled question", height=120)
    delay_min = st.number_input("Run after (minutes)", min_value=1, max_value=240, value=5)
    if st.button("Schedule"):
        run_at = datetime.now(timezone.utc) + timedelta(minutes=int(delay_min))
        resp = requests.post(
            f"{API_BASE}/schedule-query",
            json={
                "question": s_question,
                "run_at_iso": run_at.isoformat(),
                "top_k": 10,
                "include_context": False,
            },
            timeout=60,
        )
        if resp.ok:
            data = resp.json()
            st.success(f"Scheduled job: {data['job_id']}")
            st.json(data)
        else:
            st.error(resp.text)
