/**
 * popup.js
 * ────────
 * Fixed version with:
 *   1. Programmatic content script injection as fallback
 *   2. Better error messages for debugging
 *   3. Retry logic for message passing
 *   4. Improved YouTube URL detection
 */

// ── Configuration ─────────────────────────────────────────────────────
const API_BASE = "https://rag-yt-assistant-production.up.railway.app";

// ── State ─────────────────────────────────────────────────────────────
let currentVideoUrl     = null;
let currentVideoId      = null;
let conversationHistory = [];
let isLoading           = false;
let sessionMessages     = [];

// ── DOM References ────────────────────────────────────────────────────
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
// SCREEN + STATUS MANAGEMENT
// ════════════════════════════════════════════════════════════════════

function showScreen(name) {
  Object.entries(screens).forEach(([key, el]) => {
    el.classList.toggle("hidden", key !== name);
  });
  console.log(`[popup] Screen: ${name}`);
}

function setStatus(text, color = "") {
  els.statusText.textContent = text;
  els.statusDot.className    = `status-dot ${color}`;
}

function showError(message) {
  console.error(`[popup] Error: ${message}`);
  els.errorMessage.textContent = message;
  showScreen("error");
  setStatus("Error", "red");
}

function setProgress(percent, label) {
  els.progressFill.style.width  = `${percent}%`;
  els.progressLabel.textContent = label;
}


// ════════════════════════════════════════════════════════════════════
// VIDEO URL DETECTION — FIXED VERSION
// ════════════════════════════════════════════════════════════════════

/**
 * Extract video ID directly from a tab URL string.
 *
 * We do this in popup.js as a PRIMARY method — before even
 * asking content.js. The tab URL is always available to the
 * popup via chrome.tabs.query and does not require message passing.
 *
 * This is more reliable than content.js for initial detection.
 */
function extractVideoIdFromUrl(url) {
  if (!url) return null;

  try {
    const urlObj = new URL(url);

    // Standard: youtube.com/watch?v=VIDEO_ID
    const vParam = urlObj.searchParams.get("v");
    if (vParam && vParam.length === 11) return vParam;

    // Short: youtu.be/VIDEO_ID
    if (urlObj.hostname === "youtu.be") {
      const id = urlObj.pathname.slice(1).split("?")[0];
      if (id && id.length === 11) return id;
    }

    // Embedded: youtube.com/embed/VIDEO_ID
    const embedMatch = urlObj.pathname.match(/\/embed\/([a-zA-Z0-9_-]{11})/);
    if (embedMatch) return embedMatch[1];

    // Shorts: youtube.com/shorts/VIDEO_ID
    const shortsMatch = urlObj.pathname.match(/\/shorts\/([a-zA-Z0-9_-]{11})/);
    if (shortsMatch) return shortsMatch[1];

  } catch (e) {
    console.warn(`[popup] URL parse failed: ${url}`, e);
  }

  return null;
}

function buildVideoUrl(videoId) {
  return `https://www.youtube.com/watch?v=${videoId}`;
}


// ════════════════════════════════════════════════════════════════════
// CONTENT SCRIPT INJECTION — THE KEY FIX
// ════════════════════════════════════════════════════════════════════

/**
 * Inject content.js into a tab programmatically if it is not there.
 *
 * WHY THIS IS NEEDED:
 *   Chrome only auto-injects content scripts into tabs opened AFTER
 *   the extension loads. Tabs already open when the extension was
 *   installed/updated never receive the content script.
 *
 *   This function injects content.js on demand — solving the
 *   "works after refresh but not on first load" problem permanently.
 *
 * HOW IT WORKS:
 *   1. Try sending a ping message to content.js
 *   2. If it replies → already injected, skip
 *   3. If it doesn't reply → inject content.js via chrome.scripting
 *   4. Wait 200ms for injection to complete
 *   5. Now message passing works normally
 */
async function ensureContentScriptInjected(tabId) {
  return new Promise((resolve) => {
    // Test if content.js is already running with a ping
    chrome.tabs.sendMessage(tabId, { action: "PING" }, (response) => {
      if (chrome.runtime.lastError || !response) {
        // content.js not running — inject it now
        console.log("[popup] Content script not found — injecting...");

        chrome.scripting.executeScript(
          {
            target: { tabId: tabId },
            files : ["content.js"],
          },
          () => {
            if (chrome.runtime.lastError) {
              console.error(
                "[popup] Injection failed:",
                chrome.runtime.lastError.message
              );
              resolve(false);
            } else {
              console.log("[popup] Content script injected successfully");
              // Wait for script to initialise before sending messages
              setTimeout(() => resolve(true), 200);
            }
          }
        );
      } else {
        // Already running
        console.log("[popup] Content script already running");
        resolve(true);
      }
    });
  });
}

/**
 * Get the current YouTube video URL.
 *
 * IMPROVED APPROACH — three layers:
 *   Layer 1: Read tab.url directly (no content script needed)
 *            Most reliable — works even if content.js is not injected
 *   Layer 2: Ask content.js via message passing
 *            Fallback for edge cases where tab.url doesn't have ?v=
 *   Layer 3: Inject content.js then ask again
 *            Last resort for tabs that predate the extension
 */
async function getCurrentVideoUrl() {
  return new Promise((resolve) => {
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
      const tab = tabs[0];

      if (!tab || !tab.url) {
        console.log("[popup] No active tab or URL");
        resolve({ error: "no_tab" });
        return;
      }

      console.log(`[popup] Active tab URL: ${tab.url}`);

      // ── Check if on YouTube at all ──────────────────────────
      const isYouTube =
        tab.url.includes("youtube.com") ||
        tab.url.includes("youtu.be");

      if (!isYouTube) {
        console.log("[popup] Not a YouTube tab");
        resolve({ notYouTube: true });
        return;
      }

      // ── Layer 1: Extract from tab URL directly ──────────────
      // This is the most reliable method — no content script needed
      const videoId = extractVideoIdFromUrl(tab.url);

      if (videoId) {
        console.log(`[popup] Video ID from tab URL: ${videoId}`);
        resolve({
          videoUrl : buildVideoUrl(videoId),
          videoId  : videoId,
          success  : true,
          source   : "tab_url",
        });
        return;
      }

      // ── Layer 2: Ask content.js (may not be injected yet) ───
      console.log("[popup] No video ID in tab URL — trying content.js...");

      // Ensure content.js is running
      const injected = await ensureContentScriptInjected(tab.id);

      if (!injected) {
        console.log("[popup] Content script injection failed");
        resolve({ success: false, reason: "injection_failed" });
        return;
      }

      // Now ask content.js for the video URL
      chrome.tabs.sendMessage(
        tab.id,
        { action: "GET_VIDEO_URL" },
        (response) => {
          if (chrome.runtime.lastError) {
            console.error(
              "[popup] Message failed after injection:",
              chrome.runtime.lastError.message
            );
            resolve({ success: false, reason: "message_failed" });
            return;
          }

          console.log("[popup] Content script response:", response);
          resolve(response || { success: false, reason: "no_response" });
        }
      );
    });
  });
}


// ════════════════════════════════════════════════════════════════════
// API CALLS
// ════════════════════════════════════════════════════════════════════

async function checkServerHealth() {
  try {
    const resp = await fetch(`${API_BASE}/health`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!resp.ok) return false;
    const data = await resp.json();
    console.log("[popup] Health check:", data);
    return data.pipeline_loaded === true;
  } catch (e) {
    console.error("[popup] Health check failed:", e.message);
    return false;
  }
}

async function isVideoIndexed(videoId) {
  try {
    const resp = await fetch(`${API_BASE}/videos`, {
    signal: AbortSignal.timeout(10000)
});
    const data = await resp.json();
    console.log("[popup] Indexed videos:", data.video_ids);
    return data.video_ids.includes(videoId);
  } catch (e) {
    console.error("[popup] isVideoIndexed failed:", e.message);
    return false;
  }
}

async function ingestVideo(videoUrl) {
  const stages = [
    { percent: 15, label: "Fetching transcript...",   delay: 0    },
    { percent: 35, label: "Chunking segments...",     delay: 1500 },
    { percent: 60, label: "Generating embeddings...", delay: 3500 },
    { percent: 80, label: "Storing in vector DB...",  delay: 6000 },
    { percent: 90, label: "Almost done...",           delay: 8500 },
  ];

  const timers = stages.map(({ percent, label, delay }) =>
    setTimeout(() => setProgress(percent, label), delay)
  );

  try {
    console.log(`[popup] Ingesting: ${videoUrl}`);
    const resp = await fetch(`${API_BASE}/ingest`, {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({ video_url: videoUrl }),
    });

    timers.forEach(clearTimeout);

    if (!resp.ok) {
      const err = await resp.json();
      return { success: false, error: err.detail || "Ingestion failed" };
    }

    const data = await resp.json();
    setProgress(100, "Complete!");
    console.log("[popup] Ingestion complete:", data);
    return { success: true, chunkCount: data.chunk_count, wasCached: data.was_cached };

  } catch (e) {
    timers.forEach(clearTimeout);
    console.error("[popup] Ingestion error:", e.message);
    return { success: false, error: `Network error: ${e.message}` };
  }
}

async function askQuestion(videoUrl, question, history) {
  console.log(`[popup] Asking: "${question}"`);
  const controller = new AbortController();

    // Allow Railway up to 2 minutes
    const timeout = setTimeout(() => {
        controller.abort();
    }, 120000);

    try {

        const resp = await fetch(`${API_BASE}/chat`, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                video_url: videoUrl,
                question : question,
                history  : history,
            }),

            signal: controller.signal

        });

        clearTimeout(timeout);

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "Chat request failed");
        }

        return await resp.json();

    } catch (e) {

        clearTimeout(timeout);

        if (e.name === "AbortError") {
            throw new Error(
                "The server took too long to respond. Please try again."
            );
        }

        throw e;
    }
}



// ════════════════════════════════════════════════════════════════════
// CHAT RENDERING
// ════════════════════════════════════════════════════════════════════

function addMessage(role, content, sources = [], answerGrounded = true) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const label = document.createElement("div");
  label.className   = "message-label";
  label.textContent = role === "user" ? "You" : "Assistant";
  wrapper.appendChild(label);

  const bubble = document.createElement("div");
  bubble.className   = "bubble";
  bubble.textContent = content;
  wrapper.appendChild(bubble);

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
        link.title       = `Open at ${source.start_time}`;
        sourcesEl.appendChild(link);
      });
      wrapper.appendChild(sourcesEl);
    } else if (!answerGrounded) {
      const notice = document.createElement("div");
      notice.className   = "no-source-notice";
      notice.textContent = "ℹ️ No relevant sections found for this question.";
      wrapper.appendChild(notice);
    }
  }

  els.messages.appendChild(wrapper);
  els.messages.scrollTop = els.messages.scrollHeight;
}

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
  return () => {
    const el = document.getElementById("typing-indicator");
    if (el) el.remove();
  };
}

async function loadSavedConversation(videoId) {
  return new Promise((resolve) => {
    chrome.storage.local.get([`conv_${videoId}`], (result) => {
      const saved = result[`conv_${videoId}`];
      if (saved && saved.messages && saved.messages.length > 0) {
        conversationHistory = saved.history || [];
        saved.messages.forEach(msg =>
          addMessage(msg.role, msg.content, msg.sources, msg.answerGrounded)
        );
        console.log(`[popup] Loaded ${saved.messages.length} saved messages`);
      }
      resolve();
    });
  });
}

function saveConversation(videoId) {
  chrome.storage.local.set({
    [`conv_${videoId}`]: {
      messages: sessionMessages,
      history : conversationHistory,
    }
  });
}


// ════════════════════════════════════════════════════════════════════
// MAIN INITIALISATION FLOW
// ════════════════════════════════════════════════════════════════════

async function initialise() {
  console.log("[popup] Initialising...");
  setStatus("Connecting...", "yellow");

  // ── Step 1: Server health ─────────────────────────────────────
  const serverOk = await checkServerHealth();

  if (!serverOk) {

      showError(
          "Backend is starting or temporarily unavailable.\n\n" +
          "If this is the first request, Railway may be waking up.\n\n" +
          "Wait 30 seconds and click Retry."
      );

      setStatus("Starting backend...", "yellow");

      return;
  }

  setStatus("Connected", "green");
  console.log("[popup] Server healthy");

  // ── Step 2: Get video URL ─────────────────────────────────────
  const urlResponse = await getCurrentVideoUrl();
  console.log("[popup] URL response:", urlResponse);

  if (!urlResponse || urlResponse.error === "no_tab") {
    showScreen("noVideo");
    setStatus("No tab", "");
    return;
  }

  if (urlResponse.notYouTube) {
    showScreen("notYoutube");
    setStatus("Not YouTube", "");
    return;
  }

  if (!urlResponse.success || !urlResponse.videoUrl) {
    // Show helpful message with what we detected
    showScreen("noVideo");
    setStatus("No video", "");
    console.log("[popup] No video detected. Reason:", urlResponse.reason);
    return;
  }

  // Video found!
  currentVideoUrl = urlResponse.videoUrl;
  currentVideoId  = urlResponse.videoId;
  console.log(`[popup] Video: ${currentVideoId}`);

  els.videoIdDisplay.textContent = currentVideoId;
  els.videoBar.classList.remove("hidden");

  // ── Step 3: Check if already indexed ─────────────────────────
  const alreadyIndexed = await isVideoIndexed(currentVideoId);

  if (!alreadyIndexed) {
    // ── Step 4: Ingest ──────────────────────────────────────────
    showScreen("indexing");
    setStatus("Indexing...", "yellow");

    const result = await ingestVideo(currentVideoUrl);

    if (!result.success) {
      const msg = result.error;
      showError(
        msg.includes("transcript") || msg.includes("caption")
          ? "This video has no captions. Try an educational video."
          : `Indexing failed: ${msg}`
      );
      return;
    }

    els.chunkBadge.textContent = `${result.chunkCount} chunks`;
    els.chunkBadge.classList.remove("hidden");

  } else {
    els.chunkBadge.textContent = "Ready";
    els.chunkBadge.classList.remove("hidden");
  }

  // ── Step 5: Show chat ─────────────────────────────────────────
  showScreen("chat");
  setStatus("Ready", "green");
  els.clearBtn.classList.remove("hidden");

  // ── Step 6: Load saved conversation ──────────────────────────
  sessionMessages = [];
  await loadSavedConversation(currentVideoId);

  if (els.messages.children.length === 0) {
    addMessage(
      "assistant",
      "Ready! Ask me anything about this video.",
      [], true,
    );
  }

  els.questionInput.focus();
}


// ════════════════════════════════════════════════════════════════════
// EVENT HANDLERS
// ════════════════════════════════════════════════════════════════════

async function handleSubmit() {
  const question = els.questionInput.value.trim();
  if (!question || isLoading || !currentVideoUrl) return;

  isLoading                  = true;
  els.sendBtn.disabled       = true;
  els.questionInput.value    = "";
  els.questionInput.disabled = true;
  els.questionInput.style.height = "auto";

  addMessage("user", question);
  sessionMessages.push({
    role: "user", content: question,
    sources: [], answerGrounded: true,
  });

  const removeTyping = showTypingIndicator();

  try {
    const response = await askQuestion(
      currentVideoUrl, question, conversationHistory
    );
    removeTyping();

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

    if (response.answer_grounded) {
      conversationHistory.push({
        question: question,
        answer  : response.answer,
      });
    }

    saveConversation(currentVideoId);

  } catch (e) {
    removeTyping();
    addMessage("assistant", `Error: ${e.message}`, [], false);
  }

  isLoading                  = false;
  els.sendBtn.disabled       = false;
  els.questionInput.disabled = false;
  els.questionInput.focus();
}

els.sendBtn.addEventListener("click", handleSubmit);

els.questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSubmit();
  }
});

els.questionInput.addEventListener("input", () => {
  els.questionInput.style.height = "auto";
  els.questionInput.style.height =
    Math.min(els.questionInput.scrollHeight, 80) + "px";
});

els.clearBtn.addEventListener("click", () => {
  els.messages.innerHTML  = "";
  conversationHistory     = [];
  sessionMessages         = [];
  chrome.storage.local.remove(`conv_${currentVideoId}`);
  addMessage("assistant", "Conversation cleared!", [], true);
});

els.retryBtn.addEventListener("click", () => {
  els.messages.innerHTML = "";
  conversationHistory    = [];
  sessionMessages        = [];
  initialise();
});

chrome.runtime.onMessage.addListener((message) => {
  if (message.action === "VIDEO_CHANGED") {
    console.log(`[popup] Video changed to: ${message.videoId}`);
    currentVideoUrl     = null;
    currentVideoId      = null;
    conversationHistory = [];
    sessionMessages     = [];
    els.messages.innerHTML = "";
    initialise();
  }
});
// Add this near the other event listeners at the bottom of popup.js

// Close button — closes the side panel
document.getElementById("close-btn").addEventListener("click", () => {
  window.close();
});

// ── Start ──────────────────────────────────────────────────────────
initialise();