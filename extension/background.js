/**
 * background.js
 * ─────────────
 * Service worker that runs in the background.
 *
 * Sole responsibility: open the side panel when the
 * extension icon is clicked.
 *
 * Why a service worker?
 *   In Manifest V3, you cannot open the side panel directly
 *   from a content script or popup. It must be opened from
 *   the background service worker in response to a user gesture
 *   (clicking the extension icon counts as a user gesture).
 *
 * Service workers are event-driven — this file does nothing
 * until chrome.action.onClicked fires.
 */

chrome.action.onClicked.addListener((tab) => {
  // Open the side panel for the current tab
  // The side panel shows popup.html (set in manifest.json)
  chrome.sidePanel.open({ tabId: tab.id });
});