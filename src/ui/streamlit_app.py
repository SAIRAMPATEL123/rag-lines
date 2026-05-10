import os
import time
from datetime import datetime, timedelta, timezone
import requests
import streamlit as st

API_BASE = os.getenv("RAG_API_BASE", "http://localhost:8000/api/v1")

st.set_page_config(page_title="RAG-Lines MVP UI", layout="wide")
st.title("🤖 RAG-Lines MVP Web UI")
st.caption(f"Backend: {API_BASE}")

tab1, tab2, tab3 = st.tabs(["Single Query", "Batch Query", "Schedule Query"])

# ── Single Query ──────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Ask a question")
    question = st.text_area("Question", height=120)
    top_k = st.slider("Top K", 1, 50, 20)
    include_context = st.checkbox("Include context", value=True)
    if st.button("Run Query"):
        with st.spinner("Thinking... (may take 30-60s on CPU)"):
            try:
                resp = requests.post(
                    f"{API_BASE}/query",
                    json={"question": question, "top_k": top_k, "include_context": include_context},
                    timeout=300,
                )
                if resp.ok:
                    data = resp.json()
                    st.success("✅ Answer generated")
                    st.markdown(f"**Answer:** {data.get('answer', '')}")
                    st.json({"document_count": data.get("document_count", 0)})
                    if include_context and data.get("context_used"):
                        st.text_area("Context used", data.get("context_used", ""), height=220)
                else:
                    st.error(f"Error: {resp.text}")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out — model is still thinking. Try a shorter question or switch to tinyllama.")

# ── Batch Query ───────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Batch questions")
    st.info("Enter one question per line. Each question is answered separately.")
    lines = st.text_area("Questions (one per line)", height=180)
    b_top_k = st.slider("Batch Top K", 1, 50, 10)
    if st.button("Run Batch"):
        questions = [l.strip() for l in lines.splitlines() if l.strip()]
        if not questions:
            st.warning("Please enter at least one question.")
        else:
            with st.spinner(f"Processing {len(questions)} questions... (this may take a while on CPU)"):
                try:
                    resp = requests.post(
                        f"{API_BASE}/batch-query",
                        json={"questions": questions, "top_k": b_top_k, "include_context": False},
                        timeout=600,
                    )
                    if resp.ok:
                        data = resp.json()
                        st.success(f"✅ {data.get('count', 0)} answers generated")
                        for i, result in enumerate(data.get("results", [])):
                            with st.expander(f"Q{i+1}: {questions[i]}"):
                                st.markdown(f"**Answer:** {result.get('answer', '')}")
                                st.caption(f"Documents used: {result.get('document_count', 0)}")
                    else:
                        st.error(f"Error: {resp.text}")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Batch timed out — try fewer questions or switch to tinyllama model.")

# ── Schedule Query ────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Schedule one-off query")
    st.info("Schedule a question to run at a future time. Result can be checked using the job ID.")
    s_question = st.text_area("Scheduled question", height=120)
    delay_min = st.number_input("Run after (minutes)", min_value=1, max_value=240, value=5)

    if st.button("Schedule"):
        run_at = datetime.now(timezone.utc) + timedelta(minutes=int(delay_min))
        try:
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
                job_id = data["job_id"]
                st.success(f"✅ Job scheduled!")
                st.json(data)
                st.markdown(f"**Check result at:** `{API_BASE}/schedule-query/{job_id}`")

                # Auto-poll for result
                st.info(f"⏳ Auto-checking result in {delay_min} minute(s)...")
                poll_placeholder = st.empty()
                deadline = datetime.now(timezone.utc) + timedelta(minutes=int(delay_min) + 3)

                while datetime.now(timezone.utc) < deadline:
                    time.sleep(10)
                    poll_resp = requests.get(f"{API_BASE}/schedule-query/{job_id}", timeout=10)
                    if poll_resp.ok:
                        job = poll_resp.json()
                        status = job.get("status")
                        poll_placeholder.info(f"Job status: **{status}**")
                        if status == "completed":
                            poll_placeholder.success("✅ Job completed!")
                            result = job.get("result", {})
                            st.markdown(f"**Answer:** {result.get('answer', '')}")
                            st.caption(f"Documents used: {result.get('document_count', 0)}")
                            break
                        elif status == "failed":
                            poll_placeholder.error(f"❌ Job failed: {job.get('error', 'Unknown error')}")
                            break
            else:
                st.error(f"Error: {resp.text}")
        except requests.exceptions.Timeout:
            st.error("⏱️ Schedule request timed out.")