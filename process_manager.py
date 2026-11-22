#!/usr/bin/env python3
"""
RAG-Lines Process Manager
Manages FastAPI and Streamlit processes with a web dashboard
"""

import os
import subprocess
import time
import psutil
from flask import Flask, render_template_string, jsonify, request
from threading import Thread
import json

app = Flask(__name__)

class ProcessManager:
    def __init__(self):
        self.api_process = None
        self.ui_process = None
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

    def kill_by_port(self, port):
        """Kill process using specific port"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'connections']):
                try:
                    for conn in proc.connections():
                        if conn.laddr.port == port:
                            proc.kill()
                            print(f"Killed process on port {port}")
                            time.sleep(1)
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            print(f"Error killing process on port {port}: {e}")
        return False

    def is_port_open(self, port):
        """Check if port is listening"""
        try:
            for proc in psutil.process_iter(['pid', 'connections']):
                try:
                    for conn in proc.connections():
                        if conn.laddr.port == port and conn.status == 'LISTEN':
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            print(f"Error checking port {port}: {e}")
        return False

    def start_api(self):
        """Start FastAPI server"""
        if self.is_port_open(8000):
            print("API already running on port 8000")
            return {"status": "already_running", "message": "API already running"}

        try:
            self.api_process = subprocess.Popen(
                [
                    "python", "main.py", "api"
                ],
                cwd=self.base_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            time.sleep(3)

            if self.is_port_open(8000):
                print(f"API started on port 8000 (PID: {self.api_process.pid})")
                return {"status": "started", "message": "API started successfully"}
            else:
                return {"status": "failed", "message": "API failed to start"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def start_ui(self):
        """Start Streamlit UI"""
        if self.is_port_open(8501):
            print("UI already running on port 8501")
            return {"status": "already_running", "message": "UI already running"}

        try:
            self.ui_process = subprocess.Popen(
                [
                    "streamlit", "run", "src/ui/streamlit_app.py",
                    "--logger.level=warning",
                    "--client.showErrorDetails=false"
                ],
                cwd=self.base_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            time.sleep(3)

            if self.is_port_open(8501):
                print(f"UI started on port 8501 (PID: {self.ui_process.pid})")
                return {"status": "started", "message": "UI started successfully"}
            else:
                return {"status": "failed", "message": "UI failed to start"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def stop_all(self):
        """Stop all processes"""
        results = []

        # Kill by port (more reliable)
        if self.kill_by_port(8000):
            results.append("API stopped")
        if self.kill_by_port(8501):
            results.append("UI stopped")

        time.sleep(1)
        self.api_process = None
        self.ui_process = None

        return {"status": "stopped", "message": " | ".join(results) if results else "No processes running"}

    def get_status(self):
        """Get status of all services"""
        api_running = self.is_port_open(8000)
        ui_running = self.is_port_open(8501)

        return {
            "api": {
                "running": api_running,
                "port": 8000,
                "url": "http://localhost:8000" if api_running else None,
                "docs": "http://localhost:8000/docs" if api_running else None
            },
            "ui": {
                "running": ui_running,
                "port": 8501,
                "url": "http://localhost:8501" if ui_running else None
            },
            "all_running": api_running and ui_running
        }

# Create global manager
manager = ProcessManager()

# HTML Dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG-Lines Control Panel</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            padding: 40px;
            max-width: 600px;
            width: 100%;
        }

        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 28px;
        }

        .status-section {
            margin-bottom: 30px;
        }

        .status-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 8px;
            margin-bottom: 15px;
        }

        .status-info {
            flex: 1;
        }

        .status-label {
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }

        .status-detail {
            font-size: 12px;
            color: #666;
        }

        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-left: 15px;
        }

        .status-indicator.running {
            background-color: #4CAF50;
            box-shadow: 0 0 10px rgba(76, 175, 80, 0.7);
        }

        .status-indicator.stopped {
            background-color: #f44336;
        }

        .controls {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 20px;
        }

        .controls-full {
            display: grid;
            grid-template-columns: 1fr;
            gap: 10px;
        }

        button {
            padding: 12px 20px;
            font-size: 14px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
            text-transform: uppercase;
        }

        .btn-start {
            background-color: #4CAF50;
            color: white;
        }

        .btn-start:hover:not(:disabled) {
            background-color: #45a049;
            box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);
        }

        .btn-stop {
            background-color: #f44336;
            color: white;
        }

        .btn-stop:hover:not(:disabled) {
            background-color: #da190b;
            box-shadow: 0 5px 15px rgba(244, 67, 54, 0.4);
        }

        .btn-restart {
            background-color: #2196F3;
            color: white;
            grid-column: 1 / -1;
        }

        .btn-restart:hover:not(:disabled) {
            background-color: #0b7dda;
            box-shadow: 0 5px 15px rgba(33, 150, 243, 0.4);
        }

        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .message {
            padding: 12px;
            border-radius: 5px;
            margin-top: 15px;
            text-align: center;
            font-size: 13px;
        }

        .message.success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }

        .message.error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }

        .message.info {
            background-color: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }

        .links {
            margin-top: 25px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }

        .link-button {
            display: block;
            padding: 12px;
            margin-bottom: 10px;
            background-color: #f0f0f0;
            color: #333;
            text-decoration: none;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
            transition: all 0.3s ease;
        }

        .link-button:hover {
            background-color: #e0e0e0;
            transform: translateY(-2px);
        }

        .link-button.api {
            border-left: 4px solid #2196F3;
        }

        .link-button.ui {
            border-left: 4px solid #FF9800;
        }

        .loading {
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: #f3f3f3;
            border-radius: 100%;
            animation: sk-scaleout 1.0s infinite ease-in-out;
        }

        @keyframes sk-scaleout {
            0% {
                -webkit-transform: scale(0);
                transform: scale(0);
            }
            100% {
                -webkit-transform: scale(1.0);
                opacity: 0;
                transform: scale(1.0);
            }
        }

        .footer {
            text-align: center;
            margin-top: 20px;
            font-size: 12px;
            color: #999;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 RAG-Lines Control Panel</h1>

        <div class="status-section">
            <h3 style="margin-bottom: 15px; color: #333;">Service Status</h3>

            <div class="status-item">
                <div class="status-info">
                    <div class="status-label">FastAPI Backend</div>
                    <div class="status-detail">Port 8000</div>
                </div>
                <div class="status-indicator" id="api-status" class="stopped"></div>
            </div>

            <div class="status-item">
                <div class="status-info">
                    <div class="status-label">Streamlit UI</div>
                    <div class="status-detail">Port 8501</div>
                </div>
                <div class="status-indicator" id="ui-status" class="stopped"></div>
            </div>
        </div>

        <div class="controls">
            <button class="btn-start" onclick="startAPI()">Start API</button>
            <button class="btn-start" onclick="startUI()">Start UI</button>
        </div>

        <div class="controls">
            <button class="btn-stop" onclick="stopAPI()">Stop API</button>
            <button class="btn-stop" onclick="stopUI()">Stop UI</button>
        </div>

        <div class="controls-full">
            <button class="btn-restart" onclick="restartAll()">⚡ Restart All</button>
            <button class="btn-stop" onclick="stopAll()">⏹️ Stop All</button>
            <button class="btn-start" onclick="startAll()">▶️ Start All</button>
        </div>

        <div id="message"></div>

        <div class="links">
            <a href="http://localhost:8000/docs" target="_blank" class="link-button api">📚 API Docs (Swagger)</a>
            <a href="http://localhost:8501" target="_blank" class="link-button ui">🎨 RAG-Lines UI</a>
        </div>

        <div class="footer">
            Last updated: <span id="last-update">never</span>
        </div>
    </div>

    <script>
        // Auto-refresh status every 2 seconds
        setInterval(updateStatus, 2000);

        async function apiCall(endpoint, method = 'GET') {
            try {
                const response = await fetch(`/api${endpoint}`, { method });
                return await response.json();
            } catch (error) {
                console.error('API error:', error);
                return { error: error.message };
            }
        }

        async function updateStatus() {
            const status = await apiCall('/status');
            if (status.error) return;

            document.getElementById('api-status').className = `status-indicator ${status.api.running ? 'running' : 'stopped'}`;
            document.getElementById('ui-status').className = `status-indicator ${status.ui.running ? 'running' : 'stopped'}`;
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
        }

        async function startAPI() {
            await apiCall('/start-api', 'POST');
            showMessage('Starting API...', 'info');
            setTimeout(updateStatus, 2000);
        }

        async function startUI() {
            await apiCall('/start-ui', 'POST');
            showMessage('Starting UI...', 'info');
            setTimeout(updateStatus, 2000);
        }

        async function stopAPI() {
            await apiCall('/stop-api', 'POST');
            showMessage('Stopping API...', 'info');
            setTimeout(updateStatus, 1000);
        }

        async function stopUI() {
            await apiCall('/stop-ui', 'POST');
            showMessage('Stopping UI...', 'info');
            setTimeout(updateStatus, 1000);
        }

        async function startAll() {
            await startAPI();
            await new Promise(resolve => setTimeout(resolve, 2000));
            await startUI();
            showMessage('All services started!', 'success');
        }

        async function stopAll() {
            await apiCall('/stop-all', 'POST');
            showMessage('All services stopped!', 'info');
            setTimeout(updateStatus, 1000);
        }

        async function restartAll() {
            showMessage('Restarting all services...', 'info');
            await stopAll();
            await new Promise(resolve => setTimeout(resolve, 3000));
            await startAll();
            showMessage('All services restarted successfully!', 'success');
        }

        function showMessage(text, type) {
            const msgEl = document.getElementById('message');
            msgEl.textContent = text;
            msgEl.className = `message ${type}`;
            if (type === 'success' || type === 'error') {
                setTimeout(() => msgEl.textContent = '', 5000);
            }
        }

        // Initial status check
        updateStatus();
    </script>
</body>
</html>
"""

# Flask Routes
@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify(manager.get_status())

@app.route('/api/start-api', methods=['POST'])
def start_api():
    return jsonify(manager.start_api())

@app.route('/api/start-ui', methods=['POST'])
def start_ui():
    return jsonify(manager.start_ui())

@app.route('/api/stop-api', methods=['POST'])
def stop_api():
    manager.kill_by_port(8000)
    return jsonify({"status": "stopped", "message": "API stopped"})

@app.route('/api/stop-ui', methods=['POST'])
def stop_ui():
    manager.kill_by_port(8501)
    return jsonify({"status": "stopped", "message": "UI stopped"})

@app.route('/api/stop-all', methods=['POST'])
def stop_all():
    return jsonify(manager.stop_all())

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 RAG-Lines Control Panel")
    print("="*60)
    print("\n📊 Dashboard: http://localhost:5000")
    print("\nOpening in browser...")
    print("\n" + "="*60)

    # Open browser
    import webbrowser
    webbrowser.open('http://localhost:5000')

    # Run Flask
    app.run(debug=False, port=5000, host='127.0.0.1')
