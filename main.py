# minimal FastAPI backend to transcribe + scrub sensitive extras
import os, re, requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var")

MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")  # or "whisper-1"
app = FastAPI()

def scrub(text: str) -> str:
    # keep operational identifiers; just block high-risk extras
    rules = [
        (r"\b([STFG]\d{7}[A-Z])\b", "[NRIC]"),
        (r"\b(?:\d[ -]?){13,19}\b", "[CARD]"),
        (r"\b\d{9,}\b", "[ACCOUNT]"),
        (r"\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b", "[DOB]"),
    ]
    for pat, repl in rules:
        text = re.sub(pat, repl, text)
    return text

@app.get("/health")
async def health():
    return {"ok": True}

from fastapi import HTTPException

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Empty file upload.")

        files = {
            "file": (file.filename or "audio.mp3", payload, file.content_type or "audio/mpeg")
        }
        data = {
            "model": os.getenv("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe"),
            "response_format": "json"
        }

        resp = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files=files,
            data=data,
            timeout=300
        )

        if resp.status_code != 200:
            # Capture and pass through OpenAI's real error message
            try:
                err = resp.json()
            except Exception:
                err = {"status_code": resp.status_code, "body": resp.text[:500]}
            raise HTTPException(status_code=502, detail={"openai_error": err})

        text = resp.json().get("text", "")
        return JSONResponse({"transcript": scrub(text)})

    except HTTPException:
        raise
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="OpenAI timeout – try a shorter clip.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {type(e).__name__}: {str(e)[:300]}")

