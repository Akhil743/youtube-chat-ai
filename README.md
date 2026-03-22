# YouTube Chat AI

Chat with any YouTube video. Paste a link, ask questions, get AI-powered answers with clickable timestamps that jump you to the right moment.

**Live at**: [akhilrajs.com/youtube-chat](https://akhilrajs.com/youtube-chat/)

## How it works

1. You paste a YouTube video URL
2. Backend pulls the transcript, chunks it, and builds a vector index (FAISS)
3. You ask a question — relevant chunks are retrieved and sent to Gemini along with your question
4. You get an answer with `[MM:SS]` timestamps you can click to jump to that part of the video

Basically a RAG pipeline over video transcripts.

## Stack

- **Backend**: FastAPI + LangChain + FAISS + Gemini (free tier)
- **Frontend**: Vanilla HTML/CSS/JS (hosted on GitHub Pages)
- **Chrome Extension**: Manifest V3, injects a chat sidebar on YouTube pages
- **Mobile**: PWA (installable from browser) + Android WebView wrapper

## Run locally

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Create .env with your Gemini API key
cp .env.example .env
# Edit .env and add your key from https://aistudio.google.com/apikey

uvicorn app.main:app --reload --port 8000
```

Frontend is static — just open `index.html` or serve with any static server.

## Project structure

```
backend/           FastAPI + LangChain RAG pipeline
chrome-extension/  Manifest V3 Chrome extension
android-app/       Android WebView wrapper with AdMob
```

## Chrome Extension

Load `chrome-extension/` as an unpacked extension in `chrome://extensions` (enable developer mode). Go to any YouTube video and click the chat toggle.

## Why I built this

I finished [CampusX's GenAI playlist](https://www.youtube.com/playlist?list=PLKnIA16OPWvstUCrUKb_AwA0alj4HWlrT) and wanted to actually build something with LangChain + RAG instead of just watching tutorials. This seemed like a useful tool that I'd actually use myself — and turns out other people want it too.

## License

MIT
