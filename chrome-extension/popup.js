document.addEventListener("DOMContentLoaded", () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const tab = tabs[0];
        const statusEl = document.getElementById("status");
        const statusText = document.getElementById("statusText");

        if (tab && tab.url && tab.url.includes("youtube.com/watch")) {
            statusEl.className = "status active";
            statusText.textContent = "Chat sidebar is active on this page";
        } else {
            statusEl.className = "status inactive";
            statusText.textContent = "Not on a YouTube video page";
        }
    });
});
