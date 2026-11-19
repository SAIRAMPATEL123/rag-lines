import os
import time
from datetime import datetime, timedelta, timezone
import requests
import streamlit as st
import json

API_BASE = os.getenv("RAG_API_BASE", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="RAG-Lines - Premium AI Search",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: 1rem;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }

    [data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 2px solid #e2e8f0;
    }

    .header-premium {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2.5rem 1.5rem;
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        border-radius: 20px;
        border: none;
        box-shadow: 0 12px 40px rgba(6, 182, 212, 0.25);
    }

    .header-premium h1 {
        font-size: 3.2rem;
        color: #ffffff;
        margin-bottom: 0.5rem;
        font-weight: 900;
        letter-spacing: -1px;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .header-premium p {
        color: #f0f9fa;
        font-size: 1.05rem;
        font-weight: 500;
    }

    .search-container {
        background: linear-gradient(135deg, #ffffff 0%, #f0f9fa 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 2px solid #06b6d4;
        margin-bottom: 2rem;
        box-shadow: 0 4px 16px rgba(6, 182, 212, 0.1);
    }

    .result-card {
        background: linear-gradient(135deg, #ffffff 0%, #f7fbfc 100%);
        border-radius: 14px;
        padding: 1.5rem;
        border: 2px solid #e0f2fe;
        margin: 1rem 0;
        color: #0c4a6e;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(6, 182, 212, 0.08);
    }

    .result-card:hover {
        background: linear-gradient(135deg, #f0f9fa 0%, #ffffff 100%);
        border-color: #06b6d4;
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(6, 182, 212, 0.2);
    }

    .answer-premium {
        background: linear-gradient(135deg, #ecf9ff 0%, #f0fafe 100%);
        border-left: 5px solid #06b6d4;
        padding: 2rem;
        border-radius: 14px;
        margin: 2rem 0;
        color: #0c4a6e;
        border: 1px solid #cffafe;
        font-size: 1.05rem;
        line-height: 1.8;
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.12);
    }

    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f9fa 100%);
        padding: 1.5rem;
        border-radius: 14px;
        border: 2px solid #e0f2fe;
        text-align: center;
        color: #0c4a6e;
        box-shadow: 0 2px 8px rgba(6, 182, 212, 0.08);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: #06b6d4;
        box-shadow: 0 4px 16px rgba(6, 182, 212, 0.15);
        transform: translateY(-2px);
    }

    .metric-card-value {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.75rem 0;
    }

    .metric-card-label {
        font-size: 0.85rem;
        color: #0891b2;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 700;
    }

    .stButton > button {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 8px 20px rgba(6, 182, 212, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(6, 182, 212, 0.5);
    }

    .stTextArea > div > div > textarea {
        background: #ffffff;
        border-radius: 10px;
        border: 2px solid #e0f2fe;
        color: #0c4a6e;
        padding: 1rem;
        font-family: 'Segoe UI', sans-serif;
        font-size: 1rem;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #06b6d4;
        box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.2);
        background: #f7fbfc;
    }

    .stTextArea > div > div > textarea::placeholder {
        color: #64748b;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 3px solid #e2e8f0;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 1rem 1.5rem;
        border-radius: 10px 10px 0 0;
        background-color: #f1f5f9;
        border: none;
        font-weight: 700;
        color: #64748b;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        color: white;
        border: none;
    }

    .source-badge {
        display: inline-block;
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin: 0.5rem 0.5rem 0.5rem 0;
        box-shadow: 0 2px 8px rgba(6, 182, 212, 0.25);
    }

    [data-testid="stExpander"] {
        background: #ffffff;
        border-radius: 12px;
        border: 2px solid #e0f2fe;
        box-shadow: 0 2px 8px rgba(6, 182, 212, 0.08);
    }

    .stCheckbox > label {
        color: #0c4a6e;
        font-weight: 600;
    }

    .info-badge {
        background: linear-gradient(135deg, #ecf9ff 0%, #f0fafe 100%);
        border-left: 5px solid #06b6d4;
        padding: 1.2rem;
        border-radius: 10px;
        color: #0c4a6e;
        margin: 1rem 0;
        border: 1px solid #cffafe;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(6, 182, 212, 0.1);
    }

    .success-badge {
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
        border-left: 5px solid #10b981;
        padding: 1.2rem;
        border-radius: 10px;
        color: #065f46;
        margin: 1rem 0;
        border: 1px solid #d1fae5;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.1);
    }

    .error-badge {
        background: linear-gradient(135deg, #fef2f2 0%, #fef8f8 100%);
        border-left: 5px solid #ef4444;
        padding: 1.2rem;
        border-radius: 10px;
        color: #7f1d1d;
        margin: 1rem 0;
        border: 1px solid #fee2e2;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.1);
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <style>
        .sidebar-nav {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            margin-bottom: 2rem;
        }

        .nav-item {
            padding: 1rem;
            border-radius: 10px;
            background: linear-gradient(135deg, #f0f9fa 0%, #ffffff 100%);
            border: 2px solid #e0f2fe;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
            color: #0c4a6e;
            text-align: center;
            font-size: 0.95rem;
        }

        .nav-item:hover {
            background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
            color: white;
            border-color: #06b6d4;
            box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3);
        }

        .status-card {
            background: linear-gradient(135deg, #ecf9ff 0%, #f0fafe 100%);
            border: 2px solid #cffafe;
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 8px rgba(6, 182, 212, 0.1);
        }

        .status-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 0.5rem 0;
            color: #0c4a6e;
            font-weight: 500;
        }

        .status-value {
            background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 1.1rem;
        }

        .sidebar-section-title {
            color: #0891b2;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.8rem;
            margin: 1.5rem 0 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
                padding: 1rem; border-radius: 12px; text-align: center; color: white;
                margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(6, 182, 212, 0.25);">
        <h3 style="margin: 0; font-size: 1.1rem; font-weight: 900;">🧭 Navigation</h3>
    </div>
    """, unsafe_allow_html=True)

    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("📊 Dashboard", use_container_width=True, key="nav_dashboard"):
            st.session_state.current_page = "dashboard"
        if st.button("🔍 Search", use_container_width=True, key="nav_search"):
            st.session_state.current_page = "search"
        if st.button("📤 Documents", use_container_width=True, key="nav_docs"):
            st.session_state.current_page = "documents"

    with col_nav2:
        if st.button("📋 Batch", use_container_width=True, key="nav_batch"):
            st.session_state.current_page = "batch"
        if st.button("⏰ Scheduled", use_container_width=True, key="nav_scheduled"):
            st.session_state.current_page = "scheduled"
        if st.button("⚙️ Settings", use_container_width=True, key="nav_settings"):
            st.session_state.current_page = "settings"

    st.markdown("<p class='sidebar-section-title'>📊 System Status</p>", unsafe_allow_html=True)

    try:
        resp = requests.get(f"{API_BASE}/stats", timeout=5)
        if resp.ok:
            stats = resp.json()
            st.markdown(f"""
            <div class="status-card">
                <div class="status-item">
                    <span>📚 Documents:</span>
                    <span class="status-value">{stats.get('document_count', 0)}</span>
                </div>
                <div class="status-item">
                    <span>🔢 Embedding Dim:</span>
                    <span class="status-value">{stats.get('embedding_dimension', 0)}</span>
                </div>
                <div class="status-item">
                    <span>💾 Database:</span>
                    <span style="color: #0891b2; font-weight: 700;">Chroma</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("❌ Cannot fetch stats")
    except:
        st.warning("❌ API unreachable")

    st.markdown("<p class='sidebar-section-title'>🔧 Settings</p>", unsafe_allow_html=True)

    with st.expander("⚡ Query Options", expanded=True):
        top_k_default = st.slider("Top-K Results", 1, 50, 20, key="sidebar_top_k")
        include_context = st.checkbox("Show sources", value=True, key="sidebar_context")
        show_scores = st.checkbox("Show scores", value=True, key="sidebar_scores")

    st.markdown("<p class='sidebar-section-title'>📝 History</p>", unsafe_allow_html=True)

    if "query_history" not in st.session_state:
        st.session_state.query_history = []

    if len(st.session_state.query_history) > 0:
        st.markdown(f"<small style='color: #64748b;'><strong>{len(st.session_state.query_history)} queries</strong></small>",
                   unsafe_allow_html=True)
        for i, q in enumerate(st.session_state.query_history[-5:]):
            if st.button(f"📌 {q[:25]}...", key=f"history_{i}", use_container_width=True):
                st.session_state.search_query = q
                st.rerun()
    else:
        st.markdown('<small style="color: #cbd5e0; text-align: center; display: block;">No history yet</small>',
                   unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 2rem;">
        <p>RAG-Lines v1.2.0</p>
        <p>API: {API_BASE}</p>
    </div>
    """, unsafe_allow_html=True)

if "current_page" not in st.session_state:
    st.session_state.current_page = "search"

with st.container():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
            <div class="header-premium">
                <h1 style="font-size: 2.5rem; margin-bottom: 0.3rem;">✨ RAG-Lines</h1>
                <p style="font-size: 0.95rem;">Enterprise-Grade Intelligent Document Retrieval & AI Search</p>
            </div>
        """, unsafe_allow_html=True)

if st.session_state.current_page == "search":
    tab_key = 0
elif st.session_state.current_page == "batch":
    tab_key = 1
elif st.session_state.current_page == "scheduled":
    tab_key = 2
elif st.session_state.current_page == "documents":
    tab_key = 3
else:
    tab_key = 0

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Smart Search", "📋 Batch Processing", "⏰ Schedule", "📤 Documents"])

with tab1:
    st.markdown("<div class='search-container'>", unsafe_allow_html=True)
    col1, col2 = st.columns([5, 1])

    with col1:
        if "search_query" not in st.session_state:
            st.session_state.search_query = ""
        question = st.text_area(
            "🔍 Ask anything about your documents",
            height=120,
            placeholder="What would you like to know? e.g., 'Tell me about the job offer' or 'What are the key terms?'",
            value=st.session_state.search_query,
            key="single_query"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        top_k = st.slider("Top-K", 1, 50, top_k_default, key="single_topk")

    st.markdown("</div>", unsafe_allow_html=True)

    col_search, col_settings = st.columns([3, 1])
    with col_search:
        search_btn = st.button("🚀 Search & Analyze", use_container_width=True, key="search_btn")
    with col_settings:
        show_advanced = st.checkbox("Advanced", key="show_advanced_single")

    if show_advanced:
        col_a, col_b = st.columns(2)
        with col_a:
            include_ctx = st.checkbox("Show sources", value=include_context, key="single_context")
        with col_b:
            show_scores = st.checkbox("Show scores", value=show_scores, key="single_scores")
    else:
        include_ctx = include_context
        show_scores = show_scores

    if search_btn:
        if not question.strip():
            st.markdown('<div class="error-badge">❌ Please enter a question</div>', unsafe_allow_html=True)
        else:
            st.session_state.query_history.append(question)

            with st.spinner(""):
                st.markdown('<div style="text-align: center; color: #667eea; font-size: 1.2rem; margin: 2rem 0;"><strong>✨ Analyzing your documents...</strong></div>', unsafe_allow_html=True)
                search_progress = st.progress(0)

                try:
                    search_progress.progress(20)
                    resp = requests.post(
                        f"{API_BASE}/query",
                        json={"question": question, "top_k": top_k, "include_context": include_ctx},
                        timeout=300,
                    )
                    search_progress.progress(80)

                    if resp.ok:
                        data = resp.json()
                        search_progress.progress(100)
                        time.sleep(0.5)
                        search_progress.empty()

                        st.markdown(f"""
                            <div class="answer-premium">
                                <strong style="font-size: 1.2rem; color: #667eea;">💡 AI Response</strong><br><br>
                                {data.get('answer', '')}
                            </div>
                        """, unsafe_allow_html=True)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-card-label">📚 Sources</div>
                                    <div class="metric-card-value">{data.get('document_count', 0)}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-card-label">🎯 Relevance</div>
                                    <div class="metric-card-value">{top_k}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        with col3:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-card-label">⚡ Quality</div>
                                    <div class="metric-card-value">High</div>
                                </div>
                            """, unsafe_allow_html=True)

                        if data.get("retrieved_documents"):
                            st.markdown("### 📑 Source Documents")
                            for i, doc in enumerate(data.get("retrieved_documents", [])[:5]):
                                with st.expander(f"📄 Source {i+1}" + (f" (Score: {doc.get('semantic_score', 0):.2%})" if show_scores else "")):
                                    st.markdown(f"""
                                        <div class="result-card">
                                            {doc.get('document', '')[:500]}...
                                        </div>
                                    """, unsafe_allow_html=True)

                                    col_meta1, col_meta2 = st.columns(2)
                                    with col_meta1:
                                        st.caption(f"Semantic Score: {doc.get('semantic_score', 0):.2%}")
                                    with col_meta2:
                                        if doc.get('metadata'):
                                            st.caption(f"Type: {doc.get('metadata', {}).get('doc_type', 'Unknown')}")

                        if include_ctx and data.get("context_used"):
                            with st.expander("📋 Full Context Used"):
                                st.code(data.get("context_used", ""), language="text")
                    else:
                        error_msg = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                        st.markdown(f'<div class="error-badge">❌ Error: {error_msg}</div>', unsafe_allow_html=True)
                except requests.exceptions.Timeout:
                    st.markdown('<div class="error-badge">⏱️ Request timed out. Try a shorter question.</div>', unsafe_allow_html=True)
                except requests.exceptions.ConnectionError:
                    st.markdown('<div class="error-badge">❌ Cannot connect to API at http://localhost:8000</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("""
        <div class="info-badge">
        💡 <strong>Batch Processing</strong> - Process multiple questions efficiently in one operation
        </div>
    """, unsafe_allow_html=True)

    lines = st.text_area(
        "📝 Questions (one per line)",
        height=150,
        placeholder="What is the job title?\nWho is the hiring manager?\nWhat are the start dates?\nWhen is the deadline?"
    )

    col_bk1, col_bk2 = st.columns(2)
    with col_bk1:
        b_top_k = st.slider("Top-K Results", 1, 50, 10, key="batch_topk")
    with col_bk2:
        batch_type = st.selectbox("Query Type", ["Standard", "Detailed", "Summary"], key="batch_type")

    if st.button("⚡ Process Batch", use_container_width=True, key="batch_btn"):
        questions = [l.strip() for l in lines.splitlines() if l.strip()]
        if not questions:
            st.markdown('<div class="error-badge">❌ Please enter at least one question</div>', unsafe_allow_html=True)
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.markdown(f'<div style="color: #667eea; font-weight: 700;">⚙️ Processing {len(questions)} question(s)...</div>', unsafe_allow_html=True)

                resp = requests.post(
                    f"{API_BASE}/batch-query",
                    json={"questions": questions, "top_k": b_top_k, "include_context": False},
                    timeout=600,
                )

                if resp.ok:
                    data = resp.json()
                    count = data.get('count', 0)

                    status_text.markdown(f'<div class="success-badge">✅ Completed: {count}/{len(questions)} questions answered</div>', unsafe_allow_html=True)
                    progress_bar.progress(100)

                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-card-label">✅ Completed</div>
                                <div class="metric-card-value">{count}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col_m2:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-card-label">📌 Total</div>
                                <div class="metric-card-value">{len(questions)}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col_m3:
                        success_rate = (count / len(questions) * 100) if questions else 0
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-card-label">📊 Rate</div>
                                <div class="metric-card-value">{success_rate:.0f}%</div>
                            </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")

                    for i, result in enumerate(data.get("results", [])):
                        with st.expander(f"❓ Q{i+1}: {questions[i]}", expanded=(i==0)):
                            st.markdown(f"""
                                <div class="result-card">
                                    <strong>Answer:</strong><br>
                                    {result.get('answer', '')}
                                </div>
                            """, unsafe_allow_html=True)

                            col_b1, col_b2 = st.columns(2)
                            with col_b1:
                                st.caption(f"📚 Sources: {result.get('document_count', 0)}")
                            with col_b2:
                                st.caption(f"🎯 Top-K: {b_top_k}")
                else:
                    error_msg = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                    st.markdown(f'<div class="error-badge">❌ Error: {error_msg}</div>', unsafe_allow_html=True)
            except requests.exceptions.Timeout:
                st.markdown('<div class="error-badge">⏱️ Batch processing timed out. Try fewer questions.</div>', unsafe_allow_html=True)
            except requests.exceptions.ConnectionError:
                st.markdown('<div class="error-badge">❌ Cannot connect to API</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("""
        <div class="info-badge">
        📅 <strong>Scheduled Queries</strong> - Schedule queries to run at a specific time with auto-polling
        </div>
    """, unsafe_allow_html=True)

    col_sq1, col_sq2 = st.columns([2, 1])
    with col_sq1:
        s_question = st.text_area(
            "🕐 Question",
            height=120,
            placeholder="Type your question here...",
            key="schedule_q"
        )
    with col_sq2:
        st.markdown("<br>", unsafe_allow_html=True)
        delay_min = st.number_input("⏳ Run after (minutes)", min_value=1, max_value=240, value=5, key="schedule_delay")
        scheduled_time = datetime.now(timezone.utc) + timedelta(minutes=int(delay_min))
        st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.1); padding: 0.75rem; border-radius: 8px; color: #cbd5e0; font-size: 0.9rem;">
            📍 Will run at: <strong>{scheduled_time.strftime('%H:%M:%S UTC')}</strong>
            </div>
        """, unsafe_allow_html=True)

    if st.button("⏱️ Schedule Query", use_container_width=True, key="schedule_btn"):
        if not s_question.strip():
            st.markdown('<div class="error-badge">❌ Please enter a question</div>', unsafe_allow_html=True)
        else:
            try:
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
                    job_id = data["job_id"]

                    st.markdown('<div class="success-badge">✅ Query scheduled successfully!</div>', unsafe_allow_html=True)

                    col_j1, col_j2, col_j3 = st.columns(3)
                    with col_j1:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-card-label">🆔 Job ID</div>
                                <div class="metric-card-value" style="font-size: 1.2rem;">{job_id[:8]}...</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col_j2:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-card-label">⏱️ Delay</div>
                                <div class="metric-card-value">{delay_min}m</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col_j3:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-card-label">📍 Status</div>
                                <div class="metric-card-value" style="font-size: 1.2rem; color: #fbbf24;">Pending</div>
                            </div>
                        """, unsafe_allow_html=True)

                    st.markdown(f"<div style='text-align: center; color: #667eea; margin: 1.5rem 0;'>⏳ Auto-checking in {delay_min} minute(s)...</div>", unsafe_allow_html=True)
                    poll_placeholder = st.empty()
                    deadline = datetime.now(timezone.utc) + timedelta(minutes=int(delay_min) + 3)

                    while datetime.now(timezone.utc) < deadline:
                        time.sleep(10)
                        try:
                            poll_resp = requests.get(f"{API_BASE}/schedule-query/{job_id}", timeout=10)
                            if poll_resp.ok:
                                job = poll_resp.json()
                                status = job.get("status")

                                if status == "completed":
                                    poll_placeholder.empty()
                                    st.markdown('<div class="success-badge">✅ Query completed!</div>', unsafe_allow_html=True)
                                    result = job.get("result", {})
                                    st.markdown(f"""
                                        <div class="answer-premium">
                                            <strong style="color: #667eea;">📖 Scheduled Query Result</strong><br><br>
                                            {result.get('answer', '')}
                                        </div>
                                    """, unsafe_allow_html=True)

                                    col_r1, col_r2 = st.columns(2)
                                    with col_r1:
                                        st.markdown(f"""
                                            <div class="metric-card">
                                                <div class="metric-card-label">📚 Sources</div>
                                                <div class="metric-card-value">{result.get('document_count', 0)}</div>
                                            </div>
                                        """, unsafe_allow_html=True)
                                    with col_r2:
                                        exec_time = job.get("executed_at", "Unknown")
                                        st.markdown(f"""
                                            <div class="metric-card">
                                                <div class="metric-card-label">⏱️ Executed</div>
                                                <div class="metric-card-value" style="font-size: 0.9rem;">Just now</div>
                                            </div>
                                        """, unsafe_allow_html=True)
                                    break

                                elif status == "failed":
                                    poll_placeholder.empty()
                                    st.markdown(f'<div class="error-badge">❌ Job failed: {job.get("error", "Unknown error")}</div>', unsafe_allow_html=True)
                                    break

                                else:
                                    poll_placeholder.markdown(f'<div style="color: #667eea; text-align: center;">⏳ Status: <strong>{status}</strong></div>', unsafe_allow_html=True)
                        except:
                            pass

                else:
                    error_msg = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                    st.markdown(f'<div class="error-badge">❌ Error: {error_msg}</div>', unsafe_allow_html=True)
            except requests.exceptions.Timeout:
                st.markdown('<div class="error-badge">⏱️ Request timed out</div>', unsafe_allow_html=True)
            except requests.exceptions.ConnectionError:
                st.markdown('<div class="error-badge">❌ Cannot connect to API</div>', unsafe_allow_html=True)

with tab4:
    st.markdown("""
        <div class="info-badge">
        📤 <strong>Document Management</strong> - Upload and ingest PDF, Markdown, Text, HTML, or JSON files
        </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "📁 Choose files to upload",
        type=["pdf", "md", "txt", "html", "json"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.markdown('<div class="success-badge">✅ Ready to ingest</div>', unsafe_allow_html=True)

        total_size = sum(f.size for f in uploaded_files)
        col_u1, col_u2, col_u3 = st.columns(3)

        with col_u1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-card-label">📦 Files</div>
                    <div class="metric-card-value">{len(uploaded_files)}</div>
                </div>
            """, unsafe_allow_html=True)
        with col_u2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-card-label">📊 Size</div>
                    <div class="metric-card-value" style="font-size: 1.5rem;">{total_size / 1024:.0f}KB</div>
                </div>
            """, unsafe_allow_html=True)
        with col_u3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-card-label">✅ Status</div>
                    <div class="metric-card-value" style="font-size: 1.2rem; color: #4ade80;">Ready</div>
                </div>
            """, unsafe_allow_html=True)

        with st.expander("📋 File Details"):
            for i, file in enumerate(uploaded_files):
                st.caption(f"{i+1}. {file.name} ({file.size / 1024:.1f} KB)")

        if st.button("⚡ Ingest Now", use_container_width=True, key="ingest_btn"):
            import tempfile
            from pathlib import Path

            progress_bar = st.progress(0)
            status_placeholder = st.empty()

            try:
                status_placeholder.markdown('<div style="color: #667eea; font-weight: 700; text-align: center;">📤 Uploading files...</div>', unsafe_allow_html=True)
                progress_bar.progress(10)

                temp_dir = tempfile.mkdtemp()
                file_paths = []

                for i, uploaded_file in enumerate(uploaded_files):
                    file_path = Path(temp_dir) / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(str(file_path))
                    progress_bar.progress(10 + (i + 1) * 30 // len(uploaded_files))

                status_placeholder.markdown('<div style="color: #667eea; font-weight: 700; text-align: center;">⚙️ Processing documents...</div>', unsafe_allow_html=True)
                progress_bar.progress(50)

                resp = requests.post(
                    f"{API_BASE}/ingest",
                    json={"file_paths": file_paths},
                    timeout=300,
                )

                progress_bar.progress(90)

                if resp.ok:
                    data = resp.json()
                    progress_bar.progress(100)

                    status_placeholder.markdown('<div class="success-badge">✅ Ingestion complete!</div>', unsafe_allow_html=True)

                    col_i1, col_i2, col_i3 = st.columns(3)
                    with col_i1:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-card-label">📚 Ingested</div>
                                <div class="metric-card-value">{data.get('documents_ingested', 0)}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col_i2:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-card-label">🔀 Chunks</div>
                                <div class="metric-card-value">{data.get('chunks_created', 0)}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col_i3:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-card-label">💾 Total DB</div>
                                <div class="metric-card-value">{data.get('total_in_db', 0)}</div>
                            </div>
                        """, unsafe_allow_html=True)

                    time.sleep(1)
                    progress_bar.empty()
                    status_placeholder.empty()

                else:
                    st.markdown(f'<div class="error-badge">❌ Ingestion failed: {resp.text}</div>', unsafe_allow_html=True)
                    progress_bar.empty()

            except Exception as e:
                st.markdown(f'<div class="error-badge">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
                progress_bar.empty()

    st.markdown("---")
    st.markdown("### 📊 Vector Database Status")

    try:
        resp = requests.get(f"{API_BASE}/stats", timeout=10)
        if resp.ok:
            stats = resp.json()
            col_db1, col_db2, col_db3 = st.columns(3)

            with col_db1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-card-label">📚 Documents</div>
                        <div class="metric-card-value">{stats.get('document_count', 0)}</div>
                    </div>
                """, unsafe_allow_html=True)

            with col_db2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-card-label">🔢 Embedding Dim</div>
                        <div class="metric-card-value">{stats.get('embedding_dimension', 0)}</div>
                    </div>
                """, unsafe_allow_html=True)

            with col_db3:
                db_name = stats.get("db_path", "N/A").split("/")[-1].split("\\")[-1]
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-card-label">💾 Database</div>
                        <div class="metric-card-value" style="font-size: 1rem; color: #667eea;">{db_name}</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-badge">⚠️ Could not fetch database stats</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="error-badge">⚠️ API connection failed</div>', unsafe_allow_html=True)