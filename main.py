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

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    files = {"file": (file.filename, await file.read(), file.content_type or "audio/mpeg")}
    data = {"model": MODEL, "response_format": "json"}
    r = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        files=files, data=data, timeout=300
    )
    r.raise_for_status()
    text = r.json().get("text", "")
    return JSONResponse({"transcript": scrub(text)})
