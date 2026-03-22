(() => {
    const API_BASE = "https://youtube-chatbot-api.onrender.com";

    let sidebarInjected = false;
    let currentVideoId = null;
    let chatHistory = [];
    let sidebarVisible = false;

    function getVideoId() {
        const params = new URLSearchParams(window.location.search);
        return params.get("v");
    }

    function createToggleButton() {
        if (document.getElementById("ytchat-toggle")) return;

        const btn = document.createElement("div");
        btn.id = "ytchat-toggle";
        btn.title = "Toggle YouTube Chat AI";
        btn.innerHTML = `
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
            </svg>
        `;
        btn.addEventListener("click", toggleSidebar);
        document.body.appendChild(btn);
    }

    function toggleSidebar() {
        const sidebar = document.getElementById("ytchat-sidebar");
        if (!sidebar) return;

        sidebarVisible = !sidebarVisible;
        sidebar.style.transform = sidebarVisible ? "translateX(0)" : "translateX(100%)";

        const toggle = document.getElementById("ytchat-toggle");
        if (toggle) {
            toggle.style.right = sidebarVisible ? "375px" : "12px";
        }
    }

    function injectSidebar() {
        if (sidebarInjected) return;

        const sidebar = document.createElement("div");
        sidebar.id = "ytchat-sidebar";
        sidebar.innerHTML = `
            <div class="ytchat-header">
                <div class="ytchat-logo">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
                    </svg>
                    <span>YouTube Chat AI</span>
                </div>
                <button class="ytchat-close" id="ytchat-close">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
            <div class="ytchat-status" id="ytchat-status">
                <span class="ytchat-status-text">Click "Load" to analyze this video</span>
                <button class="ytchat-load-btn" id="ytchat-load">Load Video</button>
            </div>
            <div class="ytchat-messages" id="ytchat-messages">
                <div class="ytchat-welcome">
                    <p>Load the video transcript, then ask questions about the content.</p>
                </div>
            </div>
            <div class="ytchat-input-area">
                <textarea id="ytchat-question" placeholder="Ask about this video..." rows="1" disabled></textarea>
                <button id="ytchat-send" class="ytchat-send-btn" disabled>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="22" y1="2" x2="11" y2="13"/>
                        <polygon points="22,2 15,22 11,13 2,9"/>
                    </svg>
                </button>
            </div>
            <div class="ytchat-footer">
                <a href="https://akhilrajs.com/youtube-chat/" target="_blank">Open full app</a>
                <span>&middot;</span>
                <a href="https://akhilrajs.com" target="_blank">By Akhil S</a>
            </div>
        `;

        document.body.appendChild(sidebar);
        sidebarInjected = true;

        document.getElementById("ytchat-close").addEventListener("click", toggleSidebar);
        document.getElementById("ytchat-load").addEventListener("click", loadVideo);
        document.getElementById("ytchat-send").addEventListener("click", askQuestion);

        const questionInput = document.getElementById("ytchat-question");
        questionInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                askQuestion();
            }
        });
        questionInput.addEventListener("input", () => {
            questionInput.style.height = "auto";
            questionInput.style.height = Math.min(questionInput.scrollHeight, 100) + "px";
        });
    }

    async function loadVideo() {
        const videoId = getVideoId();
        if (!videoId) return;

        const statusEl = document.getElementById("ytchat-status");
        const loadBtn = document.getElementById("ytchat-load");
        loadBtn.disabled = true;
        loadBtn.textContent = "Loading...";

        try {
            const res = await fetch(`${API_BASE}/api/load-video`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url: window.location.href }),
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Failed to load");
            }

            const data = await res.json();
            currentVideoId = data.video_id;
            chatHistory = [];

            statusEl.innerHTML = `
                <span class="ytchat-status-text ytchat-success">${data.chunk_count} chunks indexed &middot; ${data.language}</span>
            `;

            document.getElementById("ytchat-question").disabled = false;
            document.getElementById("ytchat-send").disabled = false;

            const messages = document.getElementById("ytchat-messages");
            messages.innerHTML = "";
            addExtMessage("assistant", `Video loaded! Ask me anything about it.`);
        } catch (err) {
            statusEl.innerHTML = `
                <span class="ytchat-status-text ytchat-error">${err.message}</span>
                <button class="ytchat-load-btn" id="ytchat-load" onclick="this.parentElement.querySelector('.ytchat-load-btn')">Retry</button>
            `;
            document.getElementById("ytchat-load").addEventListener("click", loadVideo);
        }
    }

    async function askQuestion() {
        const input = document.getElementById("ytchat-question");
        const question = input.value.trim();
        if (!question || !currentVideoId) return;

        input.value = "";
        input.style.height = "auto";
        document.getElementById("ytchat-send").disabled = true;

        addExtMessage("user", question);
        const typing = addExtTyping();

        chatHistory.push({ role: "user", content: question });

        try {
            const res = await fetch(`${API_BASE}/api/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    video_id: currentVideoId,
                    question,
                    chat_history: chatHistory.slice(-10),
                }),
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Failed to get answer");
            }

            const data = await res.json();
            typing.remove();
            addExtMessage("assistant", data.answer, data.sources);
            chatHistory.push({ role: "assistant", content: data.answer });
        } catch (err) {
            typing.remove();
            addExtMessage("assistant", `Error: ${err.message}`);
        } finally {
            document.getElementById("ytchat-send").disabled = false;
            input.focus();
        }
    }

    function addExtMessage(role, content, sources = []) {
        const container = document.getElementById("ytchat-messages");
        const welcome = container.querySelector(".ytchat-welcome");
        if (welcome) welcome.remove();

        const msg = document.createElement("div");
        msg.className = `ytchat-msg ytchat-msg-${role}`;

        const bubble = document.createElement("div");
        bubble.className = "ytchat-bubble";

        if (role === "assistant") {
            bubble.innerHTML = content
                .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
                .replace(/\n/g, "<br>")
                .replace(/\[(\d{1,2}:)?(\d{1,2}:\d{2})\]/g, (match, h, ms) => {
                    const parts = (h ? h.slice(0, -1) + ":" : "") + ms;
                    const segs = parts.split(":").map(Number);
                    let secs = 0;
                    if (segs.length === 3) secs = segs[0] * 3600 + segs[1] * 60 + segs[2];
                    else secs = segs[0] * 60 + segs[1];
                    return `<span class="ytchat-ts" data-seconds="${secs}">${match}</span>`;
                });
        } else {
            bubble.textContent = content;
        }

        msg.appendChild(bubble);

        if (sources && sources.length > 0) {
            const srcDiv = document.createElement("div");
            srcDiv.className = "ytchat-sources";
            const seen = new Set();
            sources.forEach((s) => {
                const ts = formatTs(s.start_seconds);
                if (seen.has(ts)) return;
                seen.add(ts);
                const chip = document.createElement("span");
                chip.className = "ytchat-src-chip";
                chip.textContent = ts;
                chip.dataset.seconds = Math.floor(s.start_seconds);
                chip.addEventListener("click", () => seekYT(s.start_seconds));
                srcDiv.appendChild(chip);
            });
            msg.appendChild(srcDiv);
        }

        container.appendChild(msg);
        container.scrollTop = container.scrollHeight;

        msg.querySelectorAll(".ytchat-ts").forEach((el) => {
            el.addEventListener("click", () => seekYT(Number(el.dataset.seconds)));
        });
    }

    function addExtTyping() {
        const container = document.getElementById("ytchat-messages");
        const msg = document.createElement("div");
        msg.className = "ytchat-msg ytchat-msg-assistant";
        msg.innerHTML = `<div class="ytchat-bubble"><div class="ytchat-typing"><span></span><span></span><span></span></div></div>`;
        container.appendChild(msg);
        container.scrollTop = container.scrollHeight;
        return msg;
    }

    function seekYT(seconds) {
        const video = document.querySelector("video");
        if (video) {
            video.currentTime = seconds;
            video.play();
        }
    }

    function formatTs(seconds) {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        const pad = (n) => n.toString().padStart(2, "0");
        return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
    }

    function init() {
        if (!getVideoId()) return;
        createToggleButton();
        injectSidebar();
    }

    let lastUrl = location.href;
    const observer = new MutationObserver(() => {
        if (location.href !== lastUrl) {
            lastUrl = location.href;
            sidebarInjected = false;
            sidebarVisible = false;
            currentVideoId = null;
            chatHistory = [];
            const existing = document.getElementById("ytchat-sidebar");
            if (existing) existing.remove();
            const toggle = document.getElementById("ytchat-toggle");
            if (toggle) toggle.remove();
            if (getVideoId()) {
                setTimeout(init, 1000);
            }
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
