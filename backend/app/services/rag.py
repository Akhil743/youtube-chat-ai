import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from app.services.transcript import TranscriptResult, TranscriptChunk

MAX_CACHED = int(os.getenv("MAX_VIDEOS_CACHED", "50"))

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a YouTube video \
based on its transcript. Use ONLY the provided context to answer. If the answer \
is not in the context, say so honestly.

When referencing specific parts of the video, mention the timestamp so the user \
can find it. Format timestamps as [MM:SS] or [HH:MM:SS].

Previous conversation:
{history}

Video transcript context:
{context}

Question: {question}

Provide a clear, concise answer with relevant timestamp references where applicable."""


@dataclass
class VideoStore:
    video_id: str
    vector_store: FAISS
    title: str
    language: str
    chunk_count: int


class RAGService:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )
        self._llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=api_key,
            temperature=0.3,
        )
        self._cache: OrderedDict[str, VideoStore] = OrderedDict()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    def _evict_if_needed(self):
        while len(self._cache) > MAX_CACHED:
            self._cache.popitem(last=False)

    def is_video_loaded(self, video_id: str) -> bool:
        return video_id in self._cache

    def get_video_store(self, video_id: str) -> Optional[VideoStore]:
        if video_id in self._cache:
            self._cache.move_to_end(video_id)
            return self._cache[video_id]
        return None

    async def ingest_transcript(self, transcript: TranscriptResult) -> VideoStore:
        if existing := self.get_video_store(transcript.video_id):
            return existing

        docs = self._build_documents(transcript.chunks)
        split_docs = self._splitter.split_documents(docs)
        vector_store = await FAISS.afrom_documents(split_docs, self._embeddings)

        store = VideoStore(
            video_id=transcript.video_id,
            vector_store=vector_store,
            title=f"Video {transcript.video_id}",
            language=transcript.language,
            chunk_count=len(split_docs),
        )
        self._cache[transcript.video_id] = store
        self._evict_if_needed()
        return store

    async def query(
        self,
        video_id: str,
        question: str,
        chat_history: Optional[list[dict]] = None,
    ) -> tuple[str, list[dict]]:
        store = self.get_video_store(video_id)
        if store is None:
            raise ValueError(f"Video {video_id} not loaded")

        retriever = store.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5},
        )
        relevant_docs = await retriever.ainvoke(question)

        history_text = ""
        if chat_history:
            history_text = "\n".join(
                f"{m.get('role', '')}: {m.get('content', '')}"
                for m in chat_history[-6:]
            )

        chain = create_stuff_documents_chain(self._llm, self._prompt)
        answer = await chain.ainvoke({
            "context": relevant_docs,
            "question": question,
            "history": history_text,
        })

        sources = [
            {"text": doc.page_content[:150], "start_seconds": doc.metadata.get("start_seconds", 0)}
            for doc in relevant_docs
        ]
        return answer, sources

    def _build_documents(self, chunks: list[TranscriptChunk]) -> list[Document]:
        """Merge small transcript snippets into ~300 char documents with timestamp metadata."""
        documents = []
        buf, start = "", 0.0

        for chunk in chunks:
            if not buf:
                start = chunk.start_seconds
            buf += " " + chunk.text

            if len(buf) > 300:
                documents.append(Document(
                    page_content=buf.strip(),
                    metadata={"start_seconds": start, "end_seconds": chunk.end_seconds},
                ))
                buf = ""

        if buf.strip():
            documents.append(Document(
                page_content=buf.strip(),
                metadata={"start_seconds": start, "end_seconds": chunks[-1].end_seconds if chunks else 0},
            ))
        return documents


rag_service = RAGService()
