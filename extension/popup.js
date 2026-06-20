/**
 * popup.js
 * ────────
 * Controls the extension popup UI.
 *
 * Responsibilities:
 *   1. Get current YouTube video URL from content.js
 *   2. Check if video is already indexed via GET /videos
 *   3. If not indexed → POST /ingest with progress feedback
 *   4. If indexed    → show chat immediately
 *   5. Handle user questions via POST /chat
 *   6. Render answers with clickable timestamp sources
 *   7. Persist conversation history across popup open/close
 *
 * State machine (what screen shows when):
 *   not-youtube  → user is not on youtube.com
 *   no-video     → on youtube.com but no video playing
 *   indexing     → video found, ingestion running
 *   error        → something failed
 *   chat         → ready to answer questions
 */

// ── Configuration ────────────────────────────────────────────────────
const API_BASE = "http://localhost:8000";

// ── State ────────────────────────────────────────────────────────────
let currentVideoUrl  = null;
let currentVideoId   = null;
let conversationHistory = [];   // [{ question, answer }, ...]
let isLoading        = false;

// ── DOM References ───────────────────────────────────────────────────
// Grab all elements once at startup — faster than querySelector each time
const screens = {
  notYoutube : document.getElementById("not-youtube-screen"),
  noVideo    : document.getElementById("no-video-screen"),
  indexing   : document.getElementById("indexing-screen"),
  error      : document.getElementById("error-screen"),
  chat       : document.getElementById("chat-screen"),
};

const els = {
  statusDot     : document.getElementById("status-dot"),
  statusText    : document.getElementById("status-text"),
  videoBar      : document.getElementById("video-bar"),
  videoIdDisplay: document.getElementById("video-id-display"),
  chunkBadge    : document.getElementById("chunk-badge"),
  progressFill  : document.getElementById("progress-fill"),
  progressLabel : document.getElementById("progress-label"),
  errorMessage  : document.getElementById("error-message"),
  retryBtn      : document.getElementById("retry-btn"),
  messages      : document.getElementById("messages"),
  questionInput : document.getElementById("question-input"),
  sendBtn       : document.getElementById("send-btn"),
  clearBtn      : document.getElementById("clear-btn"),
};


// ════════════════════════════════════════════════════════════════════
// SCREEN MANAGEMENT
// ════════════════════════════════════════════════════════════════════

/**
 * Show exactly one screen, hide all others.
 * This is our simple state machine.
 */
function showScreen(name) {
  Object.entries(screens).forEach(([key, el]) => {
    el.classList.toggle("hidden", key !== name);
  });
}

/**
 * Update the status indicator in the header.
 * color: "green" | "yellow" | "red" | "" (grey)
 */
function setStatus(text, color = "") {
  els.statusText.textContent   = text;
  els.statusDot.className      = `status-dot ${color}`;
}

/**
 * Show the error screen with a specific message.
 * Always gives the user a next action via the retry button.
 */
function showError(message) {
  els.errorMessage.textContent = message;
  showScreen("error");
  setStatus("Error", "red");
}

/**
 * Update the ingestion progress bar and label.
 * percent: 0-100
 */
function setProgress(percent, label) {
  els.progressFill.style.width = `${percent}%`;
  els.progressLabel.textContent = label;
}


// ════════════════════════════════════════════════════════════════════
// API CALLS
// ════════════════════════════════════════════════════════════════════

/**
 * Check if the FastAPI server is reachable.
 * Returns true if healthy, false if down.
 *
 * We do this first so the error message is clear:
 * "Server not running" is more helpful than "fetch failed".
 */
async function checkServerHealth() {
  try {
    const resp = await fetch(`${API_BASE}/health`, {
      signal: AbortSignal.timeout(3000),  // 3 second timeout
    });
    const data = await resp.json();
    return data.pipeline_loaded === true;
  } catch {
    return false;
  }
}

/**
 * Check if a video is already indexed.
 * Returns true if the video_id appears in the /videos list.
 */
async function isVideoIndexed(videoId) {
  try {
    const resp = await fetch(`${API_BASE}/videos`);
    const data = await resp.json();
    return data.video_ids.includes(videoId);
  } catch {
    return false;
  }
}

/**
 * Call POST /ingest to index a YouTube video.
 *
 * Shows live progress updates during the ~10 second wait.
 * Progress is simulated (we don't have real step callbacks from the API)
 * but gives users feedback so they know it's working.
 *
 * Returns { success: true, chunkCount } or { success: false, error }
 */
async function ingestVideo(videoUrl) {
  // Simulated progress stages — realistic timing for a 10s operation
  const stages = [
    { percent: 15, label: "Fetching transcript...",    delay: 0    },
    { percent: 35, label: "Chunking segments...",      delay: 1500 },
    { percent: 60, label: "Generating embeddings...",  delay: 3000 },
    { percent: 80, label: "Storing in vector DB...",   delay: 6000 },
    { percent: 90, label: "Almost done...",            delay: 8000 },
  ];

  // Start progress animation
  stages.forEach(({ percent, label, delay }) => {
    setTimeout(() => setProgress(percent, label), delay);
  });

  try {
    const resp = await fetch(`${API_BASE}/ingest`, {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({ video_url: videoUrl }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      return {
        success: false,
        error  : err.detail || "Ingestion failed",
      };
    }

    const data = await resp.json();
    setProgress(100, "Complete!");

    return {
      success    : true,
      chunkCount : data.chunk_count,
      wasCached  : data.was_cached,
    };

  } catch (err) {
    return {
      success: false,
      error  : `Network error: ${err.message}`,
    };
  }
}

/**
 * Call POST /chat to get an answer about the video.
 *
 * Returns the full response object or null on failure.
 */
async function askQuestion(videoUrl, question, history) {
  const resp = await fetch(`${API_BASE}/chat`, {
    method : "POST",
    headers: { "Content-Type": "application/json" },
    body   : JSON.stringify({
      video_url: videoUrl,
      question : question,
      history  : history,
    }),
  });

  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.detail || "Chat request failed");
  }

  return await resp.json();
}


// ════════════════════════════════════════════════════════════════════
// CONTENT SCRIPT COMMUNICATION
// ════════════════════════════════════════════════════════════════════

/**
 * Ask content.js for the current YouTube video URL.
 *
 * Flow:
 *   1. Get the active tab
 *   2. Send GET_VIDEO_URL message to content.js on that tab
 *   3. Receive reply with videoUrl and videoId
 *
 * Returns null if:
 *   - Not on YouTube
 *   - YouTube but no video playing
 *   - Content script not injected yet
 */
async function getCurrentVideoUrl() {
  return new Promise((resolve) => {
    // Get the currently active tab
    chrome.tabs.query(
      { active: true, currentWindow: true },
      (tabs) => {
        const tab = tabs[0];

        if (!tab) {
          resolve(null);
          return;
        }

        // Check if we're even on YouTube
        const isYouTube = tab.url &&
          (tab.url.includes("youtube.com") ||
           tab.url.includes("youtu.be"));

        if (!isYouTube) {
          resolve({ notYouTube: true });
          return;
        }

        // Send message to content.js injected in this tab
        chrome.tabs.sendMessage(
          tab.id,
          { action: "GET_VIDEO_URL" },
          (response) => {
            // chrome.runtime.lastError fires if content.js not ready
            if (chrome.runtime.lastError) {
              resolve(null);
              return;
            }
            resolve(response);
          }
        );
      }
    );
  });
}


// ════════════════════════════════════════════════════════════════════
// CHAT RENDERING
// ════════════════════════════════════════════════════════════════════

/**
 * Add a message bubble to the chat container.
 *
 * role: "user" | "assistant"
 * For assistant messages with sources, renders clickable
 * timestamp buttons below the answer text.
 */
function addMessage(role, content, sources = [], answerGrounded = true) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  // Label above bubble
  const label = document.createElement("div");
  label.className   = "message-label";
  label.textContent = role === "user" ? "You" : "Assistant";
  wrapper.appendChild(label);

  // Bubble
  const bubble = document.createElement("div");
  bubble.className   = "bubble";
  bubble.textContent = content;
  wrapper.appendChild(bubble);

  // Sources (assistant only, grounded answers only)
  if (role === "assistant") {
    if (answerGrounded && sources && sources.length > 0) {
      const sourcesEl = document.createElement("div");
      sourcesEl.className = "sources";

      sources.forEach(source => {
        const link = document.createElement("a");
        link.className   = "source-btn";
        link.href        = source.youtube_link;
        link.target      = "_blank";
        link.rel         = "noopener noreferrer";
        link.textContent = `▶ ${source.display}`;
        link.title       = `Open video at ${source.start_time}`;
        sourcesEl.appendChild(link);
      });

      wrapper.appendChild(sourcesEl);

    } else if (!answerGrounded) {
      // Bug fix from Stage 4: no sources shown when guard fired
      const notice = document.createElement("div");
      notice.className   = "no-source-notice";
      notice.textContent = "ℹ️ No relevant sections found for this question.";
      wrapper.appendChild(notice);
    }
  }

  els.messages.appendChild(wrapper);

  // Scroll to bottom so new message is visible
  els.messages.scrollTop = els.messages.scrollHeight;
}

/**
 * Show animated typing indicator while waiting for response.
 * Returns a function that removes it when called.
 */
function showTypingIndicator() {
  const typing = document.createElement("div");
  typing.className = "message assistant";
  typing.id        = "typing-indicator";

  const label = document.createElement("div");
  label.className   = "message-label";
  label.textContent = "Assistant";

  const dots = document.createElement("div");
  dots.className = "typing";
  dots.innerHTML = "<span></span><span></span><span></span>";

  typing.appendChild(label);
  typing.appendChild(dots);
  els.messages.appendChild(typing);
  els.messages.scrollTop = els.messages.scrollHeight;

  // Return cleanup function
  return () => {
    const el = document.getElementById("typing-indicator");
    if (el) el.remove();
  };
}

/**
 * Load and render saved conversation from Chrome storage.
 * Called when popup opens — restores history from previous session.
 */
async function loadSavedConversation(videoId) {
  return new Promise((resolve) => {
    const key = `conversation_${videoId}`;
    chrome.storage.local.get([key], (result) => {
      const saved = result[key];
      if (saved && saved.messages) {
        conversationHistory = saved.history || [];
        saved.messages.forEach(msg => {
          addMessage(
            msg.role,
            msg.content,
            msg.sources,
            msg.answerGrounded,
          );
        });
      }
      resolve();
    });
  });
}

/**
 * Save current conversation to Chrome storage.
 * Called after every message so closing popup doesn't lose history.
 */
function saveConversation(videoId, messages) {
  const key  = `conversation_${videoId}`;
  const data = {
    messages: messages,
    history : conversationHistory,
  };
  chrome.storage.local.set({ [key]: data });
}


// ════════════════════════════════════════════════════════════════════
// MAIN FLOW
// ════════════════════════════════════════════════════════════════════

/**
 * Run when popup opens.
 *
 * Steps:
 *   1. Check server health
 *   2. Get current video URL from content.js
 *   3. Check if video already indexed
 *   4. If not → ingest
 *   5. Show chat screen
 *   6. Load saved conversation
 */
async function initialise() {
  setStatus("Connecting...", "yellow");

  // ── Step 1: Server health check ─────────────────────────────
  const serverOk = await checkServerHealth();
  if (!serverOk) {
    showError(
      "Cannot connect to the API server.\n\n" +
      "Make sure it's running:\n" +
      "uvicorn main:app --port 8000"
    );
    setStatus("Server offline", "red");
    return;
  }

  setStatus("Connected", "green");

  // ── Step 2: Get current video URL ───────────────────────────
  const urlResponse = await getCurrentVideoUrl();

  if (!urlResponse) {
    showScreen("noVideo");
    setStatus("No video", "");
    return;
  }

  if (urlResponse.notYouTube) {
    showScreen("notYoutube");
    setStatus("Not YouTube", "");
    return;
  }

  if (!urlResponse.success || !urlResponse.videoUrl) {
    showScreen("noVideo");
    setStatus("No video", "");
    return;
  }

  // We have a video!
  currentVideoUrl = urlResponse.videoUrl;
  currentVideoId  = urlResponse.videoId;

  // Show video bar
  els.videoIdDisplay.textContent = currentVideoId;
  els.videoBar.classList.remove("hidden");

  // ── Step 3: Check if already indexed ────────────────────────
  const alreadyIndexed = await isVideoIndexed(currentVideoId);

  if (!alreadyIndexed) {
    // ── Step 4: Ingest the video ───────────────────────────────
    showScreen("indexing");
    setStatus("Indexing...", "yellow");

    const result = await ingestVideo(currentVideoUrl);

    if (!result.success) {
      showError(
        result.error.includes("captions") || result.error.includes("transcript")
          ? "This video has no captions available. Try a different video."
          : `Indexing failed: ${result.error}`
      );
      return;
    }

    // Show chunk count
    els.chunkBadge.textContent = `${result.chunkCount} chunks`;
    els.chunkBadge.classList.remove("hidden");
  } else {
    // Already indexed — get chunk count from API
    try {
      const resp = await fetch(`${API_BASE}/videos`);
      // Chunk count not in /videos response — we show "Ready" instead
      els.chunkBadge.textContent = "Ready";
      els.chunkBadge.classList.remove("hidden");
    } catch {
      // Non-critical — badge just won't show
    }
  }

  // ── Step 5: Show chat screen ─────────────────────────────────
  showScreen("chat");
  setStatus("Ready", "green");
  els.clearBtn.classList.remove("hidden");

  // ── Step 6: Load saved conversation ─────────────────────────
  await loadSavedConversation(currentVideoId);

  // Show welcome message if no history
  if (els.messages.children.length === 0) {
    addMessage(
      "assistant",
      "Video ready! Ask me anything about it — " +
      "I'll cite the exact timestamps.",
      [],
      true,
    );
  }

  // Focus the input box so user can type immediately
  els.questionInput.focus();
}


// ════════════════════════════════════════════════════════════════════
// EVENT HANDLERS
// ════════════════════════════════════════════════════════════════════

/**
 * Handle question submission.
 * Called by send button click OR Enter key in textarea.
 */
async function handleSubmit() {
  const question = els.questionInput.value.trim();
  if (!question || isLoading || !currentVideoUrl) return;

  isLoading = true;
  els.sendBtn.disabled       = true;
  els.questionInput.value    = "";
  els.questionInput.disabled = true;

  // Track messages for storage
  const sessionMessages = [];

  // Add user message
  addMessage("user", question);
  sessionMessages.push({
    role          : "user",
    content       : question,
    sources       : [],
    answerGrounded: true,
  });

  // Show typing indicator while waiting
  const removeTyping = showTypingIndicator();

  try {
    const response = await askQuestion(
      currentVideoUrl,
      question,
      conversationHistory,
    );

    removeTyping();

    // Render assistant message
    addMessage(
      "assistant",
      response.answer,
      response.sources,
      response.answer_grounded,
    );

    sessionMessages.push({
      role          : "assistant",
      content       : response.answer,
      sources       : response.sources,
      answerGrounded: response.answer_grounded,
    });

    // Update conversation history for follow-ups
    // Only add grounded answers (same fix as Streamlit app)
    if (response.answer_grounded) {
      conversationHistory.push({
        question: question,
        answer  : response.answer,
      });
    }

    // Persist conversation to storage
    saveConversation(currentVideoId, sessionMessages);

  } catch (err) {
    removeTyping();
    addMessage(
      "assistant",
      `Sorry, something went wrong: ${err.message}`,
      [],
      false,
    );
  }

  isLoading                  = false;
  els.sendBtn.disabled       = false;
  els.questionInput.disabled = false;
  els.questionInput.focus();
}

// Send on button click
els.sendBtn.addEventListener("click", handleSubmit);

// Send on Enter (Shift+Enter for new line)
els.questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSubmit();
  }
});

// Auto-resize textarea as user types
els.questionInput.addEventListener("input", () => {
  els.questionInput.style.height = "auto";
  els.questionInput.style.height =
    Math.min(els.questionInput.scrollHeight, 80) + "px";
});

// Clear conversation
els.clearBtn.addEventListener("click", () => {
  els.messages.innerHTML  = "";
  conversationHistory     = [];

  // Clear from storage
  chrome.storage.local.remove(`conversation_${currentVideoId}`);

  // Re-show welcome message
  addMessage(
    "assistant",
    "Conversation cleared! Ask me anything about the video.",
    [],
    true,
  );
});

// Retry after error
els.retryBtn.addEventListener("click", () => {
  initialise();
});

// Listen for video change messages from content.js
// (fires when user navigates to a different YouTube video)
chrome.runtime.onMessage.addListener((message) => {
  if (message.action === "VIDEO_CHANGED") {
    // Reset state and reinitialise for new video
    currentVideoUrl      = null;
    currentVideoId       = null;
    conversationHistory  = [];
    els.messages.innerHTML = "";
    initialise();
  }
});

// ── Start ─────────────────────────────────────────────────────────
// Run initialise() as soon as popup.js loads
initialise();