# Hosting the Student Handbook Assistant for free (Render)

The bot needs about 100 MB of RAM, starts in under a second and needs no GPU, so a free 512 MB instance is
enough. The knowledge base (`data/handbook_kb.json`) is already built, so the PDF does not have to be uploaded.

## What changed for hosting
| file | change |
|---|---|
| `index.html` | calls the API on the same site (it used to call `http://127.0.0.1:8000`, the visitor's own PC) |
| `app.py` | listens on `0.0.0.0` when the host sets `PORT`; `ALLOW_UPLOAD=false` turns off `/api/upload`; `/healthz` health check |
| `render.yaml` | Render Blueprint: free plan, build and start commands, Python version, uploads off |
| `Dockerfile` | the same app as a container, for hosts that run Docker |

Running locally is unchanged: `python app.py`, then open http://127.0.0.1:8000.

## Steps
1. **GitHub.** Create a free account at github.com -> **New repository** (e.g. `kdu-handbook-assistant`).
   Public or private both work. Open the repository -> **Add file -> Upload files** -> drag in **everything inside
   the `handbook_assistant` folder** (including the `data` and `tests` folders and `render.yaml`) -> **Commit**.
   `app.py` must be at the top level of the repository, not inside a sub-folder.
2. **Render.** Sign up at render.com with **GitHub** -> **New -> Blueprint** -> choose the repository ->
   **Apply**. Render reads `render.yaml` and builds the service (about 3-5 minutes the first time).
   - No Blueprint option? Use **New -> Web Service**, pick the repository, and set:
     Runtime `Python 3`, Build `pip install -r requirements.txt`,
     Start `uvicorn app:app --host 0.0.0.0 --port $PORT`, Instance type **Free**,
     environment variables `PYTHON_VERSION=3.12.8` and `ALLOW_UPLOAD=false`.
3. **Open the link** Render shows, e.g. `https://kdu-handbook-assistant.onrender.com`
   (Render adds a suffix if the name is taken). Check `/api/status` and `/api/ontology` too.
4. **Updates.** Upload changed files to GitHub again; Render redeploys automatically.

## What to expect on the free plan
- **Sleeps after 15 minutes without visitors.** The next visitor waits about a minute while it wakes; the page
  then shows "Service unavailable - wait about a minute and refresh". Do one visit before a demo or viva.
- **750 free hours per month per workspace** - enough for one service running all month.
- **512 MB RAM, 0.1 CPU** - fine for this bot (about 100 MB, a few ms per answer).
- **No permanent disk.** Conversations reset when the service restarts or sleeps; the handbook knowledge base
  is part of the code, so it is always there.
- Free-tier terms change; check render.com/pricing before relying on it.

## Before making it public
The handbook includes staff names, emails and phone numbers. It is published by KDU, but a public bot
republishes it. For a research project, ask your supervisor or the faculty before sharing the link widely.

## Other hosts (with the Dockerfile)
- **Koyeb**: one free 512 MB instance; asks for a card for verification. Deploy from GitHub with the Dockerfile.
- **Your own PC / a university server**: `python app.py` on the machine, or `docker build -t handbook . &&
  docker run -p 8000:8000 handbook`.
- **Hugging Face Spaces**: since July 2026 new free accounts can host only static Spaces; Docker/Gradio Spaces on
  CPU need a PRO plan, so it is no longer a free option for this app.
