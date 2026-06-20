/**
 * content.js
 * ──────────
 * Injected into every YouTube page by Chrome automatically.
 *
 * Responsibilities:
 *   1. Detect the current video URL from the page
 *   2. Listen for messages from popup.js
 *   3. Reply with the current video URL when asked
 *
 * Why can't popup.js just read the URL itself?
 *   The popup runs in its own isolated context — it cannot
 *   directly access the YouTube page's DOM or URL.
 *   Chrome message passing is the only bridge between them.
 *
 * Message protocol:
 *   Popup sends:   { action: "GET_VIDEO_URL" }
 *   We reply with: { videoUrl: "https://...", videoId: "abc123" }
 *   If no video:   { videoUrl: null, videoId: null }
 */

/**
 * Extract the YouTube video ID from the current page URL.
 *
 * YouTube uses several URL formats:
 *   Standard:  youtube.com/watch?v=VIDEO_ID
 *   Short:     youtu.be/VIDEO_ID
 *   Embedded:  youtube.com/embed/VIDEO_ID
 *
 * Returns null if no video ID found (e.g. YouTube home page,
 * search results, channel pages).
 */
// At the TOP of content.js, before everything else:

/**
 * PING handler — lets popup.js check if we're running
 * without this, ensureContentScriptInjected() can't tell
 * if injection is needed
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "PING") {
    sendResponse({ alive: true });
    return true;
  }

  if (message.action === "GET_VIDEO_URL") {
    const videoId = getVideoId();
    if (videoId) {
      sendResponse({
        videoUrl: buildVideoUrl(videoId),
        videoId : videoId,
        success : true,
      });
    } else {
      sendResponse({
        videoUrl: null,
        videoId : null,
        success : false,
        reason  : "No video found on this page",
      });
    }
  }

  return true;
});
function getVideoId() {
  const url    = window.location.href;
  const urlObj = new URL(url);

  // Standard watch URL: youtube.com/watch?v=VIDEO_ID
  const vParam = urlObj.searchParams.get("v");
  if (vParam) return vParam;

  // Short URL: youtu.be/VIDEO_ID
  if (urlObj.hostname === "youtu.be") {
    return urlObj.pathname.slice(1);  // remove leading "/"
  }

  // Embedded URL: youtube.com/embed/VIDEO_ID
  const embedMatch = urlObj.pathname.match(/\/embed\/([^/?]+)/);
  if (embedMatch) return embedMatch[1];

  return null;
}

/**
 * Build the full canonical YouTube URL from a video ID.
 * We always use the standard format for consistency —
 * even if the page had a different URL format.
 */
function buildVideoUrl(videoId) {
  return `https://www.youtube.com/watch?v=${videoId}`;
}

/**
 * Listen for messages from popup.js.
 *
 * Chrome message passing works like this:
 *   1. Popup calls: chrome.tabs.sendMessage(tabId, message)
 *   2. Content script receives it here via chrome.runtime.onMessage
 *   3. Content script calls sendResponse(data) to reply
 *   4. Popup receives data in its callback
 *
 * The `return true` at the end is CRITICAL for async responses.
 * Without it Chrome closes the message channel before sendResponse
 * is called and the popup never receives the reply.
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "GET_VIDEO_URL") {
    const videoId = getVideoId();

    if (videoId) {
      sendResponse({
        videoUrl: buildVideoUrl(videoId),
        videoId : videoId,
        success : true,
      });
    } else {
      sendResponse({
        videoUrl: null,
        videoId : null,
        success : false,
        reason  : "No video playing on this page",
      });
    }
  }

  return true;  // ← CRITICAL: keeps message channel open for async reply
});

/**
 * Notify the popup when YouTube navigates to a new video.
 *
 * YouTube is a Single Page Application (SPA) — navigating between
 * videos does NOT reload the page. The URL changes but content.js
 * is not re-injected.
 *
 * We watch for URL changes using a MutationObserver on the title
 * element — when the video changes, the page title changes.
 * This fires our listener so the popup can update automatically.
 */
let lastVideoId = getVideoId();

const observer = new MutationObserver(() => {
  const currentVideoId = getVideoId();

  if (currentVideoId && currentVideoId !== lastVideoId) {
    lastVideoId = currentVideoId;

    // Tell the popup a new video is playing
    // The popup may not be open — chrome.runtime.sendMessage
    // silently does nothing if no listener exists
    chrome.runtime.sendMessage({
      action  : "VIDEO_CHANGED",
      videoId : currentVideoId,
      videoUrl: buildVideoUrl(currentVideoId),
    }).catch(() => {
      // Ignore errors — popup may not be open
    });
  }
});

// Watch for title changes (reliable signal of YouTube navigation)
const titleEl = document.querySelector("title");
if (titleEl) {
  observer.observe(titleEl, {
    subtree    : true,
    childList  : true,
    characterData: true,
  });
}