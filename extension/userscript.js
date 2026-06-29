// ==UserScript==
// @name         YouTube RAG Assistant
// @namespace    https://github.com/GaurangSane/rag-yt-assistant
// @version      1.0.0
// @description  Chat with any YouTube video. AI answers with exact timestamps.
// @author       Gaurang Sane
// @match        *://*.youtube.com/watch*
// @grant        GM_xmlhttpRequest
// @grant        GM_addStyle
// @connect      your-app.railway.app
// @run-at       document-idle
// ==/UserScript==

(function () {
    'use strict';

    const API_BASE = 'https://rag-yt-assistant-production.up.railway.app';

    // ── Inject CSS ──────────────────────────────────────────────────
    GM_addStyle(`
        #yt-rag-panel {
            position        : fixed;
            right           : 0;
            top             : 56px;
            width           : 380px;
            height          : calc(100vh - 56px);
            background      : #0f172a;
            border-left     : 1px solid #1e293b;
            z-index         : 9999;
            display         : flex;
            flex-direction  : column;
            font-family     : -apple-system, BlinkMacSystemFont, sans-serif;
            color           : #f1f5f9;
            transform       : translateX(100%);
            transition      : transform 0.3s ease;
        }
        #yt-rag-panel.open {
            transform: translateX(0);
        }
        #yt-rag-toggle {
            position        : fixed;
            right           : 0;
            top             : 50%;
            transform       : translateY(-50%);
            background      : #6366f1;
            color           : white;
            border          : none;
            border-radius   : 8px 0 0 8px;
            padding         : 12px 8px;
            cursor          : pointer;
            z-index         : 10000;
            font-size       : 18px;
            writing-mode    : vertical-rl;
            letter-spacing  : 2px;
        }
        #yt-rag-toggle:hover { background: #4f46e5; }
        #yt-rag-header {
            display         : flex;
            align-items     : center;
            justify-content : space-between;
            padding         : 12px 14px;
            background      : #1e293b;
            border-bottom   : 1px solid #334155;
            flex-shrink     : 0;
        }
        #yt-rag-header h3 {
            margin   : 0;
            font-size: 13px;
            color    : #f1f5f9;
        }
        #yt-rag-status {
            font-size: 10px;
            color    : #94a3b8;
        }
        #yt-rag-messages {
            flex       : 1;
            overflow-y : auto;
            padding    : 12px;
            display    : flex;
            flex-direction: column;
            gap        : 10px;
        }
        .yt-rag-msg {
            max-width    : 90%;
            padding      : 8px 11px;
            border-radius: 10px;
            font-size    : 12px;
            line-height  : 1.5;
            word-wrap    : break-word;
        }
        .yt-rag-msg.user {
            align-self : flex-end;
            background : #1e3a5f;
        }
        .yt-rag-msg.assistant {
            align-self : flex-start;
            background : #1e293b;
            border     : 1px solid #334155;
        }
        .yt-rag-sources {
            display   : flex;
            flex-wrap : wrap;
            gap       : 4px;
            margin-top: 6px;
        }
        .yt-rag-src-btn {
            font-size    : 10px;
            color        : #10b981;
            background   : rgba(16,185,129,0.1);
            border       : 1px solid rgba(16,185,129,0.2);
            border-radius: 4px;
            padding      : 2px 7px;
            cursor       : pointer;
            text-decoration: none;
        }
        .yt-rag-src-btn:hover { background: rgba(16,185,129,0.2); }
        #yt-rag-input-bar {
            display    : flex;
            gap        : 8px;
            padding    : 10px 12px;
            background : #1e293b;
            border-top : 1px solid #334155;
            flex-shrink: 0;
        }
        #yt-rag-input {
            flex         : 1;
            background   : #334155;
            border       : 1px solid #475569;
            border-radius: 6px;
            color        : #f1f5f9;
            font-size    : 12px;
            padding      : 7px 10px;
            resize       : none;
            outline      : none;
            font-family  : inherit;
            min-height   : 34px;
            max-height   : 80px;
        }
        #yt-rag-input:focus { border-color: #6366f1; }
        #yt-rag-send {
            background   : #6366f1;
            border       : none;
            border-radius: 6px;
            color        : white;
            cursor       : pointer;
            padding      : 0 12px;
            font-size    : 16px;
            height       : 34px;
            align-self   : flex-end;
        }
        #yt-rag-send:hover    { background: #4f46e5; }
        #yt-rag-send:disabled { background: #334155; cursor: not-allowed; }
        .yt-rag-typing {
            display    : flex;
            gap        : 4px;
            padding    : 8px 11px;
            align-items: center;
        }
        .yt-rag-typing span {
            width        : 6px;
            height       : 6px;
            background   : #94a3b8;
            border-radius: 50%;
            animation    : ytRagBounce 1.2s infinite;
        }
        .yt-rag-typing span:nth-child(2) { animation-delay: 0.2s; }
        .yt-rag-typing span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes ytRagBounce {
            0%,80%,100% { transform: translateY(0); }
            40%          { transform: translateY(-5px); }
        }
    `);

    // ── State ───────────────────────────────────────────────────────
    let currentVideoId      = null;
    let currentVideoUrl     = null;
    let conversationHistory = [];
    let isLoading           = false;
    let isIndexed           = false;

    // ── Build Panel HTML ────────────────────────────────────────────
    const panel = document.createElement('div');
    panel.id    = 'yt-rag-panel';
    panel.innerHTML = `
        <div id="yt-rag-header">
            <h3>🎬 RAG Assistant</h3>
            <span id="yt-rag-status">Connecting...</span>
        </div>
        <div id="yt-rag-messages"></div>
        <div id="yt-rag-input-bar">
            <textarea id="yt-rag-input"
                      placeholder="Ask anything about this video..."
                      rows="1"
                      disabled></textarea>
            <button id="yt-rag-send" disabled>↑</button>
        </div>
    `;

    const toggle = document.createElement('button');
    toggle.id          = 'yt-rag-toggle';
    toggle.textContent = '🎬 ASK';
    toggle.title       = 'Open YouTube RAG Assistant';

    document.body.appendChild(panel);
    document.body.appendChild(toggle);

    // ── Toggle panel ────────────────────────────────────────────────
    toggle.addEventListener('click', () => {
        panel.classList.toggle('open');
        if (panel.classList.contains('open') && !isIndexed) {
            initialise();
        }
    });

    // ── DOM helpers ─────────────────────────────────────────────────
    const msgContainer = () => document.getElementById('yt-rag-messages');
    const statusEl     = () => document.getElementById('yt-rag-status');
    const inputEl      = () => document.getElementById('yt-rag-input');
    const sendEl       = () => document.getElementById('yt-rag-send');

    function setStatus(text) {
        statusEl().textContent = text;
    }

    function addMessage(role, content, sources = [], grounded = true) {
        const wrapper       = document.createElement('div');
        wrapper.className   = `yt-rag-msg ${role}`;
        wrapper.textContent = content;

        if (role === 'assistant' && grounded && sources.length > 0) {
            const srcRow = document.createElement('div');
            srcRow.className = 'yt-rag-sources';
            sources.forEach(s => {
                const a       = document.createElement('a');
                a.className   = 'yt-rag-src-btn';
                a.href        = s.youtube_link;
                a.target      = '_blank';
                a.textContent = `▶ ${s.display}`;
                srcRow.appendChild(a);
            });
            wrapper.appendChild(srcRow);
        }

        msgContainer().appendChild(wrapper);
        msgContainer().scrollTop = msgContainer().scrollHeight;
    }

    function showTyping() {
        const t       = document.createElement('div');
        t.className   = 'yt-rag-msg assistant';
        t.id          = 'yt-rag-typing';
        t.innerHTML   = `<div class="yt-rag-typing">
            <span></span><span></span><span></span></div>`;
        msgContainer().appendChild(t);
        msgContainer().scrollTop = msgContainer().scrollHeight;
        return () => { const el = document.getElementById('yt-rag-typing'); if(el) el.remove(); };
    }

    // ── API helpers ─────────────────────────────────────────────────
    function apiPost(path, body) {
        return new Promise((resolve, reject) => {
            GM_xmlhttpRequest({
                method  : 'POST',
                url     : `${API_BASE}${path}`,
                headers : { 'Content-Type': 'application/json' },
                data    : JSON.stringify(body),
                timeout : 120000,
                onload  : r => {
                    if (r.status >= 200 && r.status < 300) {
                        resolve(JSON.parse(r.responseText));
                    } else {
                        reject(new Error(`HTTP ${r.status}`));
                    }
                },
                onerror : () => reject(new Error('Network error')),
                ontimeout: () => reject(new Error('Request timed out')),
            });
        });
    }

    function apiGet(path) {
        return new Promise((resolve, reject) => {
            GM_xmlhttpRequest({
                method  : 'GET',
                url     : `${API_BASE}${path}`,
                timeout : 10000,
                onload  : r => resolve(JSON.parse(r.responseText)),
                onerror : () => reject(new Error('Network error')),
            });
        });
    }

    // ── Main flow ───────────────────────────────────────────────────
    function getVideoId() {
        const params = new URLSearchParams(window.location.search);
        return params.get('v');
    }

    async function initialise() {
        const vid = getVideoId();
        if (!vid) {
            setStatus('No video detected');
            return;
        }

        currentVideoId  = vid;
        currentVideoUrl = `https://www.youtube.com/watch?v=${vid}`;

        setStatus('Checking server...');
        try {
            await apiGet('/health');
        } catch {
            setStatus('Server offline');
            addMessage('assistant',
                '⚠️ Cannot connect to API. Make sure the backend is running.');
            return;
        }

        // Check if already indexed
        try {
            const data = await apiGet('/videos');
            if (data.video_ids.includes(currentVideoId)) {
                isIndexed = true;
                setStatus(`Ready (${currentVideoId})`);
                enableInput();
                addMessage('assistant',
                    '✅ Video ready! Ask me anything about it.');
                return;
            }
        } catch { /* proceed to ingest */ }

        // Ingest
        setStatus('Indexing...');
        addMessage('assistant', '⏳ Indexing video... (~10s, only happens once)');

        try {
            const result = await apiPost('/ingest', { video_url: currentVideoUrl });
            isIndexed = true;
            setStatus(`Ready · ${result.chunk_count} chunks`);
            msgContainer().innerHTML = '';
            addMessage('assistant',
                `✅ Ready! ${result.chunk_count} chunks indexed. Ask anything.`);
            enableInput();
        } catch (e) {
            setStatus('Indexing failed');
            addMessage('assistant', `❌ Failed to index: ${e.message}`);
        }
    }

    function enableInput() {
        inputEl().disabled = false;
        sendEl().disabled  = false;
        inputEl().focus();
    }

    async function handleSend() {
        const question = inputEl().value.trim();
        if (!question || isLoading) return;

        isLoading          = true;
        sendEl().disabled  = true;
        inputEl().value    = '';

        addMessage('user', question);
        const removeTyping = showTyping();

        try {
            const resp = await apiPost('/chat', {
                video_url: currentVideoUrl,
                question : question,
                history  : conversationHistory,
            });

            removeTyping();
            addMessage(
                'assistant',
                resp.answer,
                resp.sources,
                resp.answer_grounded,
            );

            if (resp.answer_grounded) {
                conversationHistory.push({
                    question: question,
                    answer  : resp.answer,
                });
            }
        } catch (e) {
            removeTyping();
            addMessage('assistant', `❌ Error: ${e.message}`);
        }

        isLoading         = false;
        sendEl().disabled = false;
        inputEl().focus();
    }

    sendEl().addEventListener('click', handleSend);
    document.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey &&
            document.activeElement === inputEl()) {
            e.preventDefault();
            handleSend();
        }
    });

    // Detect YouTube SPA navigation
    let lastUrl = location.href;
    new MutationObserver(() => {
        if (location.href !== lastUrl) {
            lastUrl = location.href;
            if (getVideoId()) {
                currentVideoId      = null;
                isIndexed           = false;
                conversationHistory = [];
                msgContainer().innerHTML = '';
                setStatus('New video detected');
                inputEl().disabled  = true;
                sendEl().disabled   = true;
                if (panel.classList.contains('open')) {
                    initialise();
                }
            }
        }
    }).observe(document, { subtree: true, childList: true });

})();