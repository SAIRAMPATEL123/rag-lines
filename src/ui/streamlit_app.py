import os
import time
from datetime import datetime, timedelta, timezone
import requests
import streamlit as st

API_BASE = os.getenv("RAG_API_BASE", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="RAG-Lines",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

st.markdown("""
    <style>
        * {
            margin: 0;
            padding: 0;
        }

        [data-testid="stMainBlockContainer"] {
            padding-top: 2rem;
        }

        .header-container {
            text-align: center;
            margin-bottom: 2rem;
            padding: 1rem;
        }

        .header-container h1 {
            font-size: 2.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            font-weight: 800;
        }

        .header-container p {
            color: #666;
            font-size: 0.95rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 1.5rem;
            border-radius: 8px 8px 0 0;
            background-color: #f5f5f5;
            border: 2px solid transparent;
            font-weight: 500;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #667eea;
        }

        .card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
            border: 1px solid #e8e8e8;
            margin: 1rem 0;
        }

        .input-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            border: 1px solid rgba(102, 126, 234, 0.1);
            margin-bottom: 1.5rem;
        }

        .stMetric {
            background: white;
            padding: 1.25rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
            border-left: 4px solid #667eea;
        }

        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }

        .stTextArea > div > div > textarea {
            border-radius: 8px;
            border: 2px solid #e8e8e8;
            font-family: 'Segoe UI', sans-serif;
            padding: 0.75rem;
        }

        .stTextArea > div > div > textarea:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .stSlider > div {
            padding: 0.5rem 0;
        }

        .answer-box {
            background: linear-gradient(135deg, #f5f7fa 0%, #f0f4ff 100%);
            border-left: 4px solid #667eea;
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        }

        .success-box {
            background: linear-gradient(135deg, #d4fc79 0%, #96f550 100%);
            padding: 1rem;
            border-radius: 8px;
            color: #2d5016;
            font-weight: 500;
        }

        .info-box {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 1rem;
            border-radius: 8px;
            color: #333;
            font-weight: 500;
        }

        .divider {
            margin: 2rem 0;
            border: 0;
            border-top: 2px solid #e8e8e8;
        }

        .status-section {
            background: white;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        }

        [data-testid="stExpander"] {
            background: white;
            border-radius: 8px;
            border: 1px solid #e8e8e8;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        }

        .stCheckbox > label {
            font-weight: 500;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div class="header-container">
                <h1>✨ RAG-Lines</h1>
                <p>Your intelligent document retrieval & questioning system</p>
            </div>
        """, unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Query", "📋 Batch Query", "⏰ Schedule Query", "📤 Upload Documents"])

with tab1:
    st.markdown("### Ask your question")
    with st.container():
        col1, col2 = st.columns([1, 1])
        with col1:
            question = st.text_area("💭 What would you like to know?", height=140, placeholder="Type your question here...")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            top_k = st.slider("🎯 Relevance Results (Top K)", 1, 50, 20, help="How many relevant documents to retrieve")
            include_context = st.checkbox("📚 Show source context", value=True, help="Display the document excerpts used for the answer")

    if st.button("🚀 Get Answer", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question")
        else:
            with st.spinner("✨ Thinking... (may take 30-60s on CPU)"):
                try:
                    resp = requests.post(
                        f"{API_BASE}/query",
                        json={"question": question, "top_k": top_k, "include_context": include_context},
                        timeout=300,
                    )
                    if resp.ok:
                        data = resp.json()
                        st.success("✅ Answer generated successfully!")
                        st.markdown(f"""
                            <div class="answer-box">
                                <strong>📖 Answer:</strong><br><br>
                                {data.get('answer', '')}
                            </div>
                        """, unsafe_allow_html=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("📚 Documents Used", data.get("document_count", 0))
                        with col2:
                            st.metric("🔑 Top-K", top_k)

                        if include_context and data.get("context_used"):
                            with st.expander("📄 View Source Context"):
                                st.text_area("Context used:", data.get("context_used", ""), height=220, disabled=True)
                    else:
                        error_msg = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                        st.error(f"❌ Error: {error_msg}")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out — model is still thinking. Try a shorter question or switch to tinyllama.")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to RAG API. Is the backend running at http://localhost:8000?")

with tab2:
    st.markdown("### Process Multiple Questions")
    st.info("💡 Enter one question per line. Each will be answered separately and efficiently.")

    lines = st.text_area("📝 Questions (one per line)", height=150, placeholder="What is AI?\nExplain machine learning?\nHow does NLP work?")
    b_top_k = st.slider("🎯 Batch Top-K", 1, 50, 10, help="Documents to retrieve per question")

    if st.button("▶️ Process Batch", use_container_width=True):
        questions = [l.strip() for l in lines.splitlines() if l.strip()]
        if not questions:
            st.warning("Please enter at least one question.")
        else:
            with st.spinner(f"⚙️ Processing {len(questions)} question(s)... (this may take a while on CPU)"):
                try:
                    resp = requests.post(
                        f"{API_BASE}/batch-query",
                        json={"questions": questions, "top_k": b_top_k, "include_context": False},
                        timeout=600,
                    )
                    if resp.ok:
                        data = resp.json()
                        st.success(f"✅ {data.get('count', 0)} answers generated!")
                        st.metric("📊 Completion Rate", f"{data.get('count', 0)}/{len(questions)}")

                        for i, result in enumerate(data.get("results", [])):
                            with st.expander(f"📌 Q{i+1}: {questions[i]}", expanded=(i==0)):
                                st.markdown(f"**Answer:** {result.get('answer', '')}")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.caption(f"📚 Docs used: {result.get('document_count', 0)}")
                                with col2:
                                    st.caption(f"✨ Top-K: {b_top_k}")
                    else:
                        error_msg = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                        st.error(f"❌ Error: {error_msg}")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Batch timed out — try fewer questions or switch to tinyllama model.")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to RAG API. Is the backend running?")

with tab3:
    st.markdown("### Schedule Questions for Later")
    st.info("⏰ Schedule a question to run at a future time and we'll auto-check results for you.")

    col1, col2 = st.columns([2, 1])
    with col1:
        s_question = st.text_area("🕐 Question to schedule", height=120, placeholder="Type your question here...")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        delay_min = st.number_input("⏳ Run after (minutes)", min_value=1, max_value=240, value=5)
        st.caption(f"Scheduled for: {(datetime.now(timezone.utc) + timedelta(minutes=int(delay_min))).strftime('%H:%M:%S UTC')}")

    if st.button("📅 Schedule Query", use_container_width=True):
        if not s_question.strip():
            st.warning("Please enter a question")
        else:
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
                    st.success(f"✅ Query scheduled successfully!")

                    with st.expander("📋 Job Details", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🆔 Job ID", job_id[:8] + "...")
                        with col2:
                            st.metric("⏱️ Delay", f"{delay_min}m")
                        with col3:
                            st.metric("📍 Status", "Pending")

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
                                st.markdown(f"""
                                    <div class="answer-box">
                                        <strong>📖 Answer:</strong><br><br>
                                        {result.get('answer', '')}
                                    </div>
                                """, unsafe_allow_html=True)
                                st.metric("📚 Documents Used", result.get('document_count', 0))
                                break
                            elif status == "failed":
                                poll_placeholder.error(f"❌ Job failed: {job.get('error', 'Unknown error')}")
                                break
                else:
                    error_msg = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                    st.error(f"❌ Error: {error_msg}")
            except requests.exceptions.Timeout:
                st.error("⏱️ Schedule request timed out.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to RAG API. Is the backend running?")

with tab4:
    st.markdown("### Upload & Ingest Your Documents")
    st.info("📄 Upload PDF, Markdown, Text, HTML, or JSON files. They'll be automatically ingested into the vector database.")

    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "md", "txt", "html", "json"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) selected")
        col1, col2, col3 = st.columns([1, 1, 1])

        total_size = sum(f.size for f in uploaded_files)
        with col1:
            st.metric("📦 Files", len(uploaded_files))
        with col2:
            st.metric("📊 Total Size", f"{total_size / 1024:.1f} KB")
        with col3:
            st.metric("✅ Ready to Upload", "Yes")

        with st.expander("📋 File Details"):
            for file in uploaded_files:
                st.caption(f"📄 {file.name} ({file.size / 1024:.1f} KB)")

        if st.button("🚀 Ingest Documents", use_container_width=True):
            import tempfile
            from pathlib import Path

            with st.spinner("⚙️ Uploading and processing documents..."):
                try:
                    temp_dir = tempfile.mkdtemp()
                    file_paths = []

                    for uploaded_file in uploaded_files:
                        file_path = Path(temp_dir) / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(str(file_path))

                    resp = requests.post(
                        "http://localhost:8000/api/v1/ingest",
                        json={"file_paths": file_paths},
                        timeout=300,
                    )

                    if resp.ok:
                        data = resp.json()
                        st.success(f"✅ Ingestion complete!")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📚 Documents Ingested", data.get('documents_ingested', 0))
                        with col2:
                            st.metric("🔀 Chunks Created", data.get('chunks_created', 0))
                        with col3:
                            st.metric("📊 Total in DB", data.get('total_in_db', 0))
                    else:
                        st.error(f"❌ Ingestion failed: {resp.text}")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    st.markdown("---")
    st.markdown("### 📊 Vector Database Status")

    try:
        resp = requests.get(f"{API_BASE}/stats", timeout=10)
        if resp.ok:
            stats = resp.json()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📚 Documents in DB", stats.get("document_count", 0))
            with col2:
                st.metric("🔢 Embedding Dimension", stats.get("embedding_dimension", 0))
            with col3:
                db_name = stats.get("db_path", "N/A").split("/")[-1].split("\\")[-1]
                st.metric("💾 Database", db_name)
        else:
            st.warning("⚠️ Could not fetch database stats")
    except Exception as e:
        st.warning(f"⚠️ Could not connect to API: {str(e)}")