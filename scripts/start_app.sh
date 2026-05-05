#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${RAG_LLM_MODEL:-mistral}"
API_HOST="${RAG_API_HOST:-0.0.0.0}"
API_PORT="${RAG_API_PORT:-8000}"
UI_PORT="${RAG_UI_PORT:-8501}"

cleanup() {
  echo "\n[rag-lines] Stopping services..."
  [[ -n "${API_PID:-}" ]] && kill "$API_PID" 2>/dev/null || true
  [[ -n "${UI_PID:-}" ]] && kill "$UI_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

command -v ollama >/dev/null 2>&1 || { echo "[rag-lines] Error: ollama not installed."; exit 1; }
command -v streamlit >/dev/null 2>&1 || { echo "[rag-lines] Error: streamlit not installed (pip install -r requirements.txt)."; exit 1; }

if ! pgrep -f "ollama serve" >/dev/null 2>&1; then
  echo "[rag-lines] Starting Ollama server..."
  nohup ollama serve >/tmp/rag_ollama.log 2>&1 &
  sleep 2
fi

echo "[rag-lines] Ensuring model is available: ${MODEL_NAME}"
ollama pull "$MODEL_NAME" >/tmp/rag_ollama_pull.log 2>&1 || {
  echo "[rag-lines] Warning: unable to pull model '${MODEL_NAME}'. Check Ollama/network.";
}

echo "[rag-lines] Starting API on ${API_HOST}:${API_PORT} ..."
python main.py api >/tmp/rag_api.log 2>&1 &
API_PID=$!

echo "[rag-lines] Starting Streamlit UI on :${UI_PORT} ..."
RAG_API_BASE="http://localhost:${API_PORT}/api/v1" \
streamlit run src/ui/streamlit_app.py --server.port "${UI_PORT}" >/tmp/rag_ui.log 2>&1 &
UI_PID=$!

echo "[rag-lines] App started"
echo "  UI:  http://localhost:${UI_PORT}"
echo "  API: http://localhost:${API_PORT}/docs"
echo "  Logs: /tmp/rag_api.log, /tmp/rag_ui.log, /tmp/rag_ollama.log"

echo "[rag-lines] Press Ctrl+C to stop all started services."
wait
