"""
NovaLM Interactive Web Interface / GUI

A zero-dependency local browser dashboard providing:
1. Interactive Chat & Text Generation Playground with parameter sliders.
2. Training Metrics & Loss/Perplexity Visualization Dashboard.
3. Incremental Data Manager (inspect files, trigger processing).
4. Checkpoints Explorer.
"""

from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import json
import mimetypes
import os
import threading
import traceback
import datetime
import math

import torch

from configs import default_config
from inference.generate import load_inference_artifacts, generate, get_device
def run_data_pipeline(force_rebuild=False):
    """Compatibility bridge for the current data-preparation script.

    Older GUI code exposed run_data_pipeline(); the current prepare_data.py
    entry point exposes main() instead. Prefer the newer function when it
    exists, otherwise invoke main() without modifying the existing dataset.
    """
    import importlib
    module = importlib.import_module("scripts.prepare_data")
    pipeline = getattr(module, "run_data_pipeline", None)
    if callable(pipeline):
        return pipeline(force_rebuild=force_rebuild)
    main_fn = getattr(module, "main", None)
    if callable(main_fn):
        return main_fn()
    raise AttributeError("scripts.prepare_data exposes neither run_data_pipeline() nor main()")
from scripts.visualize import plot_training_metrics


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NovaLM Studio</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0f172a;
            --surface: #1e293b;
            --surface-hover: #334155;
            --border: #334155;
            --primary: #3b82f6;
            --primary-hover: #2563eb;
            --accent: #10b981;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --danger: #ef4444;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            display: flex;
            height: 100vh;
            overflow: hidden;
        }
        /* Sidebar */
        .sidebar {
            width: 260px;
            background: var(--surface);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            padding: 24px 16px;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 32px;
            padding: 0 8px;
        }
        .brand-logo {
            width: 38px;
            height: 38px;
            border-radius: 10px;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 18px;
        }
        .brand-text h1 { font-size: 18px; font-weight: 700; letter-spacing: -0.5px; }
        .brand-text span { font-size: 12px; color: var(--text-muted); }
        .nav-links { list-style: none; display: flex; flex-direction: column; gap: 6px; flex: 1; }
        .nav-item button {
            width: 100%;
            background: transparent;
            border: none;
            padding: 12px 14px;
            border-radius: 8px;
            color: var(--text-muted);
            font-size: 14px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 10px;
            cursor: pointer;
            transition: all 0.15s;
            text-align: left;
        }
        .nav-item button:hover { background: var(--surface-hover); color: var(--text); }
        .nav-item button.active { background: var(--primary); color: white; }
        .system-badge {
            background: rgba(15, 23, 42, 0.6);
            padding: 12px;
            border-radius: 8px;
            font-size: 12px;
            color: var(--text-muted);
            border: 1px solid var(--border);
        }
        /* Main Content */
        .main {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }
        .tab-content { display: none; padding: 32px; max-width: 1200px; margin: 0 auto; width: 100%; }
        .tab-content.active { display: block; }
        .page-header {
            margin-bottom: 28px;
        }
        .page-header h2 { font-size: 26px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 6px; }
        .page-header p { color: var(--text-muted); font-size: 14px; }
        /* Cards */
        .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
        .stat-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
        }
        .stat-card .label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; margin-bottom: 8px; }
        .stat-card .value { font-size: 24px; font-weight: 700; color: var(--text); }
        /* Chat Interface */
        .chat-container {
            display: grid;
            grid-template-columns: 1fr 320px;
            gap: 24px;
            height: calc(100vh - 160px);
        }
        .chat-box {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .chat-history {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .message {
            max-width: 80%;
            padding: 14px 18px;
            border-radius: 12px;
            font-size: 14px;
            line-height: 1.6;
        }
        .message.user {
            align-self: flex-end;
            background: var(--primary);
            color: white;
            border-bottom-right-radius: 4px;
        }
        .message.bot {
            align-self: flex-start;
            background: #243247;
            border: 1px solid var(--border);
            border-bottom-left-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            white-space: pre-wrap;
        }
        .chat-input-bar {
            padding: 16px;
            background: #192231;
            border-top: 1px solid var(--border);
            display: flex;
            gap: 10px;
        }
        .chat-input-bar input {
            flex: 1;
            background: var(--surface);
            border: 1px solid var(--border);
            padding: 12px 16px;
            border-radius: 8px;
            color: var(--text);
            font-size: 14px;
            outline: none;
        }
        .chat-input-bar button {
            background: var(--primary);
            color: white;
            border: none;
            padding: 0 24px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.15s;
        }
        .chat-input-bar button:hover { background: var(--primary-hover); }
        /* Settings panel */
        .settings-panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .slider-group { display: flex; flex-direction: column; gap: 8px; }
        .slider-header { display: flex; justify-content: space-between; font-size: 13px; font-weight: 500; }
        .slider-header span:last-child { color: var(--primary); font-family: monospace; font-weight: 600; }
        input[type="range"] {
            accent-color: var(--primary);
            height: 6px;
            background: var(--border);
            border-radius: 3px;
        }
        /* Buttons & Tables */
        .btn {
            background: var(--primary);
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: background 0.15s;
        }
        .btn:hover { background: var(--primary-hover); }
        .btn-secondary { background: var(--surface-hover); color: var(--text); }
        .btn-secondary:hover { background: #475569; }
        table {
            width: 100%;
            border-collapse: collapse;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }
        th, td {
            padding: 14px 18px;
            text-align: left;
            border-bottom: 1px solid var(--border);
            font-size: 14px;
        }
        th { background: #182234; color: var(--text-muted); font-size: 12px; text-transform: uppercase; font-weight: 600; }
        .plot-container {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 24px;
            text-align: center;
        }
        .plot-container img { max-width: 100%; border-radius: 8px; }
        .status-row { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:24px; }
        .mini-card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:16px; }
        .mini-card .label { color:var(--text-muted); font-size:11px; text-transform:uppercase; font-weight:600; }
        .mini-card .value { margin-top:6px; font-size:18px; font-weight:700; }
        .muted { color:var(--text-muted); }
        .ok { color:var(--accent); }
        .warn { color:#f59e0b; }
        .danger { color:var(--danger); }
        .mono { font-family:'JetBrains Mono',monospace; font-size:12px; word-break:break-all; }
        @media (max-width: 900px) { .sidebar { width:210px; } .grid-4,.status-row { grid-template-columns:repeat(2,1fr); } .chat-container { grid-template-columns:1fr; height:auto; } }
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="brand">
            <div class="brand-logo">N</div>
            <div class="brand-text">
                <h1>NovaLM Studio</h1>
                <span>Continuous LM Pipeline</span>
            </div>
        </div>
        <ul class="nav-links">
            <li class="nav-item"><button class="active" onclick="showTab('chat')">💬 Playground</button></li>
            <li class="nav-item"><button onclick="showTab('training')">📊 Training Monitor</button></li>
            <li class="nav-item"><button onclick="showTab('data')">📁 Data Manager</button></li>
            <li class="nav-item"><button onclick="showTab('checkpoints')">💾 Checkpoints</button></li>
        </ul>
        <div class="system-badge">
            <div style="margin-bottom: 4px; font-weight: 600; color: var(--text);" id="system-device">Device: Loading...</div>
            <div id="system-step">Step: Loading...</div>
        </div>
    </div>

    <div class="main">
        <!-- Tab 1: Chat / Generation Playground -->
        <div id="tab-chat" class="tab-content active">
            <div class="page-header">
                <h2>Model Generation Playground</h2>
                <p>Interact with NovaLM directly conditioned on the latest checkpoint.</p>
            </div>
            <div class="chat-container">
                <div class="chat-box">
                    <div class="chat-history" id="chat-history">
                        <div class="message bot">Hello! I am NovaLM. Enter a prompt to start text generation.</div>
                    </div>
                    <div class="chat-input-bar">
                        <input type="text" id="prompt-input" placeholder="Enter prompt (e.g. Once upon a time, a clever dog...)" onkeydown="if(event.key==='Enter') sendPrompt()">
                        <button onclick="sendPrompt()" id="send-btn">Generate</button>
                    </div>
                </div>
                <div class="settings-panel">
                    <h3 style="font-size: 16px;">Sampling Parameters</h3>
                    <div class="slider-group">
                        <div class="slider-header"><span>Temperature</span><span id="temp-val">0.8</span></div>
                        <input type="range" id="temp-slider" min="0.1" max="2.0" step="0.05" value="0.8" oninput="document.getElementById('temp-val').textContent = this.value">
                    </div>
                    <div class="slider-group">
                        <div class="slider-header"><span>Top-K Cutoff</span><span id="topk-val">40</span></div>
                        <input type="range" id="topk-slider" min="1" max="100" step="1" value="40" oninput="document.getElementById('topk-val').textContent = this.value">
                    </div>
                    <div class="slider-group">
                        <div class="slider-header"><span>Max New Tokens</span><span id="tokens-val">60</span></div>
                        <input type="range" id="tokens-slider" min="10" max="200" step="5" value="60" oninput="document.getElementById('tokens-val').textContent = this.value">
                    </div>
                    <div class="slider-group">
                        <div class="slider-header"><span>Repetition Penalty</span><span id="rep-val">1.1</span></div>
                        <input type="range" id="rep-slider" min="1.0" max="1.5" step="0.05" value="1.1" oninput="document.getElementById('rep-val').textContent = this.value">
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab 2: Training Monitor -->
        <div id="tab-training" class="tab-content">
            <div class="page-header" style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2>Continuous Training Dashboard</h2>
                    <p>Live metrics, loss curves, and perplexity tracking across training sessions.</p>
                </div>
                <button class="btn btn-secondary" onclick="refreshPlots()">🔄 Refresh Charts</button>
            </div>
            <div class="grid-4">
                <div class="stat-card"><div class="label">Global Step</div><div class="value" id="stat-step">-</div></div>
                <div class="stat-card"><div class="label">Best Val Loss</div><div class="value" id="stat-val-loss">-</div></div>
                <div class="stat-card"><div class="label">Best Val Perplexity</div><div class="value" id="stat-ppl">-</div></div>
                <div class="stat-card"><div class="label">Latest Train Loss</div><div class="value" id="stat-train-loss">-</div></div>
            </div>
            <div class="status-row">
                <div class="mini-card"><div class="label">Train Sequences</div><div class="value" id="stat-train-seqs">-</div></div>
                <div class="mini-card"><div class="label">Validation Sequences</div><div class="value" id="stat-val-seqs">-</div></div>
                <div class="mini-card"><div class="label">Tokens / Sequence</div><div class="value" id="stat-context">-</div></div>
                <div class="mini-card"><div class="label">Tokenizer</div><div class="value" id="stat-vocab">-</div></div>
            </div>
            <div class="mini-card" style="margin-bottom:24px;">
                <div class="label">Latest Training State</div>
                <div id="training-state" style="margin-top:8px;" class="mono muted">Loading...</div>
            </div>
            <div class="plot-container">
                <img id="training-plot-img" src="/api/plots" alt="Training Progress Plots" onerror="this.alt='No training plots generated yet. Run training to produce metrics.'">
            </div>
        </div>

        <!-- Tab 3: Data Manager -->
        <div id="tab-data" class="tab-content">
            <div class="page-header" style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2>Dataset Manager</h2>
                    <p>Track ingested text files and run incremental preprocessing.</p>
                </div>
                <button class="btn" onclick="triggerPrepareData()" id="prep-btn">⚡ Process New Data</button>
            </div>
            <div id="prep-status" style="margin-bottom: 16px; font-size: 14px; color: var(--accent);"></div>
            <table>
                <thead>
                    <tr>
                        <th>File Name</th>
                        <th>Size</th>
                        <th>Bytes</th>
                        <th>Fingerprint</th>
                        <th>Processed Output</th>
                        <th>Status</th>
                        <th>Last Modified</th>
                    </tr>
                </thead>
                <tbody id="data-table-body">
                    <tr><td colspan="6" style="text-align: center; color: var(--text-muted);">Loading files...</td></tr>
                </tbody>
            </table>
        </div>

        <!-- Tab 4: Checkpoints Explorer -->
        <div id="tab-checkpoints" class="tab-content">
            <div class="page-header">
                <h2>Checkpoints Explorer</h2>
                <p>Saved model weight snapshots and restoration states.</p>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Checkpoint File</th>
                        <th>Type</th>
                        <th>Size</th>
                        <th>Modified Time</th>
                    </tr>
                </thead>
                <tbody id="ckpt-table-body">
                    <tr><td colspan="4" style="text-align: center; color: var(--text-muted);">Loading checkpoints...</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        function showTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.nav-item button').forEach(el => el.classList.remove('active'));
            document.getElementById('tab-' + tabId).classList.add('active');
            event.target.classList.add('active');
            if (tabId === 'training') refreshPlots();
            if (tabId === 'data') loadDataStatus();
            if (tabId === 'checkpoints') loadCheckpoints();
        }

        async function updateStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('system-device').textContent = 'Device: ' + data.device;
                document.getElementById('system-step').textContent = 'Step: ' + Number(data.global_step || 0).toLocaleString();
                document.getElementById('stat-step').textContent = Number(data.global_step || 0).toLocaleString();
                document.getElementById('stat-val-loss').textContent = Number.isFinite(data.best_val_loss) ? data.best_val_loss.toFixed(4) : '-';
                document.getElementById('stat-ppl').textContent = Number.isFinite(data.best_val_ppl) ? data.best_val_ppl.toFixed(2) : '-';
                document.getElementById('stat-train-loss').textContent = Number.isFinite(data.latest_train_loss) ? data.latest_train_loss.toFixed(4) : '-';
                document.getElementById('stat-train-seqs').textContent = Number(data.train_sequences || 0).toLocaleString();
                document.getElementById('stat-val-seqs').textContent = Number(data.val_sequences || 0).toLocaleString();
                document.getElementById('stat-context').textContent = Number(data.sequence_length || 0).toLocaleString();
                document.getElementById('stat-vocab').textContent = Number(data.vocab_size || 0).toLocaleString();
                document.getElementById('training-state').textContent = data.training_state || 'No checkpoint information available.';
            } catch (e) {
                console.error(e);
                document.getElementById('training-state').textContent = 'Unable to reach NovaLM backend: ' + e.message;
            }
        }

        async function sendPrompt() {
            const input = document.getElementById('prompt-input');
            const prompt = input.value.trim();
            if (!prompt) return;

            const chatHistory = document.getElementById('chat-history');
            const userMsg = document.createElement('div');
            userMsg.className = 'message user';
            userMsg.textContent = prompt;
            chatHistory.appendChild(userMsg);

            input.value = '';
            input.disabled = true;
            document.getElementById('send-btn').disabled = true;

            const botMsg = document.createElement('div');
            botMsg.className = 'message bot';
            botMsg.textContent = 'Generating...';
            chatHistory.appendChild(botMsg);
            chatHistory.scrollTop = chatHistory.scrollHeight;

            try {
                const res = await fetch('/api/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        prompt: prompt,
                        temperature: parseFloat(document.getElementById('temp-slider').value),
                        top_k: parseInt(document.getElementById('topk-slider').value),
                        max_new_tokens: parseInt(document.getElementById('tokens-slider').value),
                        repetition_penalty: parseFloat(document.getElementById('rep-slider').value),
                    })
                });
                const data = await res.json();
                botMsg.textContent = data.text || 'Error: ' + data.error;
            } catch (err) {
                botMsg.textContent = 'Network error: ' + err.message;
            } finally {
                input.disabled = false;
                document.getElementById('send-btn').disabled = false;
                input.focus();
                chatHistory.scrollTop = chatHistory.scrollHeight;
            }
        }

        function refreshPlots() {
            const img = document.getElementById('training-plot-img');
            img.src = '/api/plots?t=' + Date.now();
        }

        function escapeHtml(value) {
            const div = document.createElement('div');
            div.textContent = String(value ?? '');
            return div.innerHTML;
        }

        function formatBytes(bytes) {
            if (!Number.isFinite(bytes)) return '-';
            const units = ['B','KB','MB','GB','TB'];
            let n = bytes, i = 0;
            while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
            return `${n.toFixed(i ? 2 : 0)} ${units[i]}`;
        }

        async function loadDataStatus() {
            try {
                const res = await fetch('/api/data-status');
                const data = await res.json();
                const tbody = document.getElementById('data-table-body');
                tbody.innerHTML = '';
                if (!data.files || data.files.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-muted);">No raw .txt files detected in data/raw/</td></tr>';
                    return;
                }
                data.files.forEach(f => {
                    const tr = document.createElement('tr');
                    const statusClass = f.status === 'processed' ? 'ok' : 'warn';
                    tr.innerHTML = `
                        <td><strong>${escapeHtml(f.name)}</strong></td>
                        <td>${formatBytes(f.size_bytes)}</td>
                        <td class="mono">${f.fingerprint ? f.fingerprint.substring(0, 12) + '…' : '-'}</td>
                        <td>${f.sequences ? f.sequences.toLocaleString() + ' seq' : '-'}</td>
                        <td><span class="${statusClass}">${f.status || 'Pending'}</span></td>
                        <td>${f.modified || '-'}</td>`;
                    tbody.appendChild(tr);
                });
            } catch (e) {
                document.getElementById('data-table-body').innerHTML = `<tr><td colspan="6" class="danger">Failed to load data status: ${escapeHtml(e.message)}</td></tr>`;
            }
        }

        async function triggerPrepareData() {
            const btn = document.getElementById('prep-btn');
            const status = document.getElementById('prep-status');
            btn.disabled = true;
            status.textContent = 'Processing data... please wait.';
            try {
                const res = await fetch('/api/prepare-data', { method: 'POST' });
                const result = await res.json();
                status.textContent = '✓ Data preparation completed: ' + result.message;
                loadDataStatus();
                updateStatus();
            } catch (e) {
                status.textContent = 'Error: ' + e.message;
            } finally {
                btn.disabled = false;
            }
        }

        async function loadCheckpoints() {
            const res = await fetch('/api/checkpoints');
            const data = await res.json();
            const tbody = document.getElementById('ckpt-table-body');
            tbody.innerHTML = '';
            if (!data.checkpoints || data.checkpoints.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" style="text-align: center;">No checkpoints found yet.</td></tr>';
                return;
            }
            data.checkpoints.forEach(c => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${c.name}</strong></td>
                    <td>${c.type}</td>
                    <td>${(c.size_bytes / (1024*1024)).toFixed(2)} MB</td>
                    <td>${c.modified}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        updateStatus();
        setInterval(updateStatus, 5000);
        setInterval(() => {
            const trainingTab = document.getElementById('tab-training');
            if (trainingTab && trainingTab.classList.contains('active')) refreshPlots();
        }, 15000);
    </script>
</body>
</html>
"""


# ============================================================
# HTTP REQUEST HANDLER
# ============================================================

class NovaLMStudioHandler(BaseHTTPRequestHandler):

    _model_cache = None
    _tokenizer_cache = None
    _checkpoint_cache = None

    def log_message(self, format, *args):
        # Clean console logging
        pass

    @classmethod
    def get_cached_model(cls):
        if cls._model_cache is None:
            device = get_device()
            model, tok, ckpt, _ = load_inference_artifacts(device=device)
            cls._model_cache = model
            cls._tokenizer_cache = tok
            cls._checkpoint_cache = ckpt
        return cls._model_cache, cls._tokenizer_cache, cls._checkpoint_cache

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        elif path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            global_step = 0
            best_val_loss = None
            best_val_ppl = None
            latest_train_loss = None
            epoch = None
            checkpoint_time = None
            tokenizer_identity = None

            if default_config.latest_checkpoint.exists():
                try:
                    ckpt = torch.load(default_config.latest_checkpoint, map_location="cpu", weights_only=False)
                    global_step = int(ckpt.get("global_step", 0))
                    best_val_loss = ckpt.get("best_validation_loss")
                    best_val_ppl = ckpt.get("best_validation_ppl")
                    epoch = ckpt.get("epoch")
                    tokenizer_identity = ckpt.get("tokenizer_identity")
                    checkpoint_time = datetime.datetime.fromtimestamp(
                        default_config.latest_checkpoint.stat().st_mtime
                    ).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass

            history = []
            if default_config.history_file.exists():
                try:
                    import csv
                    with open(default_config.history_file, "r", encoding="utf-8") as f:
                        history = list(csv.DictReader(f))
                    if history:
                        last = history[-1]
                        latest_train_loss = float(last["train_loss"])
                except Exception:
                    pass

            manifest = {}
            if default_config.data.manifest_file.exists():
                try:
                    with open(default_config.data.manifest_file, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                except Exception:
                    pass

            train = manifest.get("splits", {}).get("train", {})
            val = manifest.get("splits", {}).get("val", {})
            tok = manifest.get("tokenizer", {})
            total_train = int(train.get("sequences", 0))
            total_val = int(val.get("sequences", 0))
            seq_len = int(manifest.get("sequence_length", 0))
            vocab = int(tok.get("vocab_size", manifest.get("vocab_size", 0)))
            checkpoint_tokenizer_sha = (tokenizer_identity or {}).get("sha256", "")
            manifest_tokenizer_sha = tok.get("sha256", "")
            shown_tokenizer_sha = manifest_tokenizer_sha or checkpoint_tokenizer_sha
            training_state = (
                f"step {global_step:,} · epoch {epoch if epoch is not None else '-'} · "
                f"checkpoint {checkpoint_time or '-'} · "
                f"tokenizer {shown_tokenizer_sha[:12] + '…' if shown_tokenizer_sha else '-'}"
            )

            response = {
                "device": str(get_device()),
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "best_val_ppl": best_val_ppl,
                "latest_train_loss": latest_train_loss,
                "train_sequences": total_train,
                "val_sequences": total_val,
                "sequence_length": seq_len,
                "vocab_size": vocab,
                "training_state": training_state,
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
            return

        elif path == "/api/plots":
            # Generate or refresh plot
            plot_training_metrics()
            plot_file = default_config.plot_file

            if plot_file.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(plot_file, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
            return

        elif path == "/api/data-status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            raw_files = sorted(default_config.data.raw_dir.rglob("*.txt")) if default_config.data.raw_dir.exists() else []
            manifest = {}
            if default_config.data.manifest_file.exists():
                try:
                    with open(default_config.data.manifest_file, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                except Exception:
                    pass

            source_map = {}
            for item in manifest.get("source_files", []):
                source_map[item.get("source", "")] = item

            train_sequences = manifest.get("splits", {}).get("train", {}).get("sequences", 0)
            files_info = []
            for rf in raw_files:
                rel = str(rf.relative_to(default_config.data.raw_dir)).replace("\\", "/")
                key = f"data/raw/{rel}"
                info = source_map.get(key, {})
                stat = rf.stat()
                processed = bool(info)
                files_info.append({
                    "name": rel,
                    "size_bytes": stat.st_size,
                    "fingerprint": info.get("fingerprint"),
                    "sequences": train_sequences if processed and len(raw_files) == 1 else None,
                    "status": "processed" if processed else "pending",
                    "modified": datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                })

            self.wfile.write(json.dumps({
                "files": files_info,
                "train_sequences": manifest.get("splits", {}).get("train", {}).get("sequences", 0),
                "val_sequences": manifest.get("splits", {}).get("val", {}).get("sequences", 0),
                "shards": len(manifest.get("splits", {}).get("train", {}).get("shards", [])),
                "sequence_length": manifest.get("sequence_length", 0),
            }).encode("utf-8"))
            return

        elif path == "/api/checkpoints":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            ckpts = []
            if default_config.checkpoint_dir.exists():
                for pt in sorted(default_config.checkpoint_dir.glob("*.pt")):
                    stat = pt.stat()
                    ctype = "Latest" if pt.name == "latest.pt" else ("Best" if pt.name == "best.pt" else "Step Snapshot")
                    ckpts.append({
                        "name": pt.name,
                        "type": ctype,
                        "size_bytes": stat.st_size,
                        "modified": datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    })

            self.wfile.write(json.dumps({"checkpoints": ckpts}).encode("utf-8"))
            return

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"

        if path == "/api/generate":
            try:
                data = json.loads(body)
                prompt = data.get("prompt", "").strip()
                temperature = float(data.get("temperature", 0.8))
                top_k = int(data.get("top_k", 40))
                max_new_tokens = int(data.get("max_new_tokens", 60))
                repetition_penalty = float(data.get("repetition_penalty", 1.1))

                # Reload fresh model if checkpoint updated
                NovaLMStudioHandler._model_cache = None
                model, tokenizer, _, _ = load_inference_artifacts()

                output = generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"text": output}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        elif path == "/api/prepare-data":
            try:
                run_data_pipeline(force_rebuild=False)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"message": "Data preparation completed successfully."}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        else:
            self.send_response(404)
            self.end_headers()


def run_gui(host: str = "0.0.0.0", port: int = 8080):
    server = ThreadingHTTPServer((host, port), NovaLMStudioHandler)
    print("=" * 60)
    print(f"★ NovaLM Studio GUI running at: http://localhost:{port}")
    print("Open the link in your browser to access the interface.")
    print("Press Ctrl+C to stop the GUI server.")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping GUI server...")
        server.server_close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NovaLM Studio GUI Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    args = parser.parse_args()

    run_gui(port=args.port)


if __name__ == "__main__":
    main()
