"""
Faculty of Computing - Student Handbook Assistant (CPU-only, no LLM).

Run:   python app.py      then open http://127.0.0.1:8000
Host:  see DEPLOY.md (Render free tier; uvicorn app:app --host 0.0.0.0 --port $PORT)
The QA engine lives in handbook_qa.py (structured knowledge base + rules + BM25F).
Conversation layer: dialog_manager.py + small_talk.py + conversation_rules.py (v10).
v11: fuzzy_layer.py (fuzzy rule base), ontology.py (automatic ontology generation from the PDF) and
ontology_query.py + ontology_rules.pl (Prolog reasoning over the generated ontology).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from dialog_manager import DialogManager
from handbook_qa import HandbookQA, build_kb, load_kb

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
KB_FILE = DATA_DIR / "handbook_kb.json"
HTML_FILE = BASE_DIR / "index.html"
DEFAULT_PDF = BASE_DIR / "Student-Handbook-2022.pdf"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
# On a public host set ALLOW_UPLOAD=false so visitors cannot replace the handbook for everyone.
ALLOW_UPLOAD = os.getenv("ALLOW_UPLOAD", "true").strip().lower() not in ("0", "false", "no", "off")
# Extra browser origins allowed to call the API (comma-separated), e.g. a separate front-end site.
EXTRA_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Student Handbook Assistant (no LLM)", version="11.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null", "http://127.0.0.1:8000", "http://localhost:8000"] + EXTRA_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# ------------------------------------------------------------------
# Load an existing knowledge base, or build one from the default PDF.
# ------------------------------------------------------------------
qa: HandbookQA | None = None


dm: DialogManager | None = None


def _activate(kb: dict[str, Any]) -> None:
    global qa, dm
    qa = HandbookQA(kb)
    dm = DialogManager(qa)  # a new handbook resets all conversations


_kb = load_kb(KB_FILE)
if _kb is None and DEFAULT_PDF.is_file():
    print(f"Building knowledge base from {DEFAULT_PDF.name} ...")
    _kb = build_kb(DEFAULT_PDF, KB_FILE)
if _kb is not None:
    _activate(_kb)
    print(f"Ready: {_kb['document']} - {_kb['page_count']} pages, {len(_kb['courses'])} courses, "
          f"{len(_kb['staff'])} staff, {len(_kb['passages'])} passages")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    previous_question: str | None = Field(default=None, max_length=1000)  # enables follow-up questions


class ChatRequest(BaseModel):
    session_id: str = Field(default="default", max_length=100)
    message: str = Field(min_length=1, max_length=1000)


class ResetRequest(BaseModel):
    session_id: str = Field(default="default", max_length=100)


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    """Conversational endpoint: small talk, context, pronouns, clarification questions, graceful failures."""
    if dm is None:
        return {"answer": "Please upload the Student Handbook PDF first.", "kind": "not_ready", "confidence": 0.0}
    return dm.handle(request.session_id, request.message)


@app.post("/api/reset")
async def reset(request: ResetRequest) -> dict[str, Any]:
    if dm is not None:
        dm.reset(request.session_id)
    return {"status": "reset"}


@app.get("/api/status")
async def status() -> dict[str, Any]:
    if qa is None:
        return {"ready": False, "llm_used": False, "upload_enabled": ALLOW_UPLOAD}
    kb = qa.kb
    return {
        "ready": True,
        "document_name": kb["document"],
        "page_count": kb["page_count"],
        "sections": len(kb["sections"]),
        "courses": len(kb["courses"]),
        "staff": len(kb["staff"]),
        "passages": len(kb["passages"]),
        "llm_used": False,
        "gpu_required": False,
        "active_conversations": len(dm.sessions) if dm else 0,
        "upload_enabled": ALLOW_UPLOAD,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    if not ALLOW_UPLOAD:
        raise HTTPException(status_code=403, detail="Uploading a new handbook is disabled on this public site.")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    content = await file.read()
    if not content or not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The PDF exceeds the 50 MB limit.")
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / Path(file.filename).name
        pdf_path.write_bytes(content)
        try:
            kb = build_kb(pdf_path, KB_FILE)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"PDF processing failed: {exc}") from exc
    _activate(kb)
    return {"status": "success", "document_name": kb["document"], "page_count": kb["page_count"],
            "courses": len(kb["courses"]), "staff": len(kb["staff"]), "passages": len(kb["passages"])}


@app.post("/api/ask")
async def ask(request: AskRequest) -> dict[str, Any]:
    if qa is None:
        return {"answer": "Please upload the Student Handbook PDF first.", "kind": "not_ready", "confidence": 0.0}
    return qa.answer(request.question, previous=request.previous_question)


@app.get("/api/ontology")
async def ontology() -> dict[str, Any]:
    """The ontology generated from the current handbook: fact counts per relation and validation issues.
    Files: data/handbook_ontology.pl (Prolog), data/handbook_ontology.ttl (RDF/Turtle, opens in Protege)."""
    if dm is None or getattr(dm, "onto", None) is None:
        return {"ready": False}
    o = dm.onto.onto
    return {"ready": True, "facts": len(o.facts), "by_relation": o.stats(),
            "confirmed_by_two_sources": sum(1 for f in o.facts.values() if len(f["sources"]) > 1),
            "validation_issues": o.issues, "files": ["data/handbook_ontology.pl", "data/handbook_ontology.ttl",
                                                     "data/ontology_report.json"]}


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    """Lightweight health check for the hosting platform."""
    return {"ok": qa is not None}


@app.get("/")
async def home():
    if HTML_FILE.exists():
        return FileResponse(HTML_FILE)
    return HTMLResponse("<p>index.html not found. Place it next to app.py.</p>")


if __name__ == "__main__":
    import uvicorn

    # Hosting platforms set PORT and need the server on all interfaces; locally stay on 127.0.0.1.
    host = os.getenv("HOST") or ("0.0.0.0" if os.getenv("PORT") else "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print("\nStudent Handbook Assistant - CPU only, no LLM, no GPU")
    print(f"Open http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}\n")
    uvicorn.run(app, host=host, port=port, reload=False)
