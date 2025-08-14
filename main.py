# backend/main.py
import os
import uuid
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import openai
import requests

# Embeddings & FAISS from langchain-community
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------- Load .env from explicit path ----------
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    logging.warning(f".env file not found at {ENV_PATH}, relying on environment variables.")

# Optional: whisper for server-side STT (if installed)
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

# TTS
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

# ---------- CONFIG (from environment variables) ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # REQUIRED
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")  # optional
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
PORT = int(os.getenv("PORT", 10000))
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")  # set to your frontend origin in production

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agri-backend")

# Validate required config
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable is not set. Exiting.")
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")

# Set OpenAI key for openai package
openai.api_key = OPENAI_API_KEY

app = FastAPI(title="AgriGPT Backend")

# CORS for frontend access
allow_origins_list = [ALLOW_ORIGINS] if ALLOW_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (audio)
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --------- Load embeddings and FAISS vectorstore (best-effort) ----------
logger.info("Initializing embeddings with model: %s", EMBEDDING_MODEL)
try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
except Exception as e:
    logger.exception("Failed to initialize embeddings: %s", e)
    embeddings = None

vectorstore = None
if embeddings is not None:
    try:
        logger.info("Attempting to load FAISS index from: %s", FAISS_INDEX_PATH)
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        logger.info("FAISS index loaded.")
    except Exception as e:
        logger.exception("Failed to load FAISS index. Reason: %s", e)
        vectorstore = None
else:
    logger.warning("Embeddings not initialized; skipping FAISS load.")

# Load whisper model optionally
whisper_model = None
if WHISPER_AVAILABLE:
    try:
        logger.info("Loading Whisper model (may take time)...")
        whisper_model = whisper.load_model("base")
    except Exception as e:
        logger.exception("Failed to load Whisper model: %s", e)
        whisper_model = None
else:
    logger.info("Whisper not available. /voice will return an explanatory error.")

# --------- Request model ----------
class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = "en"  # language code from frontend (optional)

# --------- Utilities ----------
def fetch_weather_for_city(city_name: str) -> Optional[str]:
    if not WEATHER_API_KEY:
        logger.debug("No WEATHER_API_KEY set; skipping weather fetch.")
        return None
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={WEATHER_API_KEY}&units=metric"
        r = requests.get(url, timeout=6)
        data = r.json()
        if data.get("main"):
            return f"Current temperature in {city_name} is {data['main']['temp']}°C, {data['weather'][0]['description']}."
    except Exception as e:
        logger.warning("Weather fetch failed: %s", e)
    return None

def call_openai_chat(system_prompt: str, user_prompt: str) -> str:
    """Call OpenAI ChatCompletion (gpt-3.5-turbo) and return text."""
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.exception("OpenAI call failed: %s", e)
        raise

def generate_tts(text: str, lang: str = "en", filename_prefix: str = "answer") -> str:
    """Generate TTS MP3 using gTTS and return relative static path (e.g. /static/...)."""
    if not GTTS_AVAILABLE:
        raise RuntimeError("gTTS not installed on server.")
    fname = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.mp3"
    out_path = os.path.join("static", fname)
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(out_path)
    return f"/static/{fname}"

# --------- Endpoint: root ----------
@app.get("/")
def root():
    return {"message": "AgriGPT backend running"}

# --------- Endpoint: get_answer ----------
@app.post("/get_answer/")
async def get_answer(req: QueryRequest):
    if vectorstore is None:
        logger.error("FAISS vectorstore not loaded.")
        raise HTTPException(status_code=500, detail="FAISS vectorstore not loaded on server.")

    user_query = (req.query or "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Empty query")

    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(user_query)
        context_text = "\n\n".join([d.page_content for d in docs]) if docs else ""
    except Exception as e:
        logger.exception("Retrieval error: %s", e)
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    realtime_info = ""
    if any(w in user_query.lower() for w in ["weather", "temperature", "rain", "forecast"]):
        w = fetch_weather_for_city("Chennai")
        if w:
            realtime_info = w

    system_prompt = (
        "You are AgriGPT, an expert assistant for farmers. Use the supplied context to answer accurately. "
        "Always answer in the same language as the user's question. Be concise, practical and actionable for farmers."
    )

    user_prompt_parts = [f"Question: {user_query}"]
    if context_text:
        user_prompt_parts.append(f"Context:\n{context_text}")
    if realtime_info:
        user_prompt_parts.append(f"RealTimeData:\n{realtime_info}")

    user_prompt_parts.append("Answer using the context if relevant. If context doesn't contain the answer, use general agricultural knowledge.")
    user_prompt = "\n\n".join(user_prompt_parts)

    try:
        answer_text = call_openai_chat(system_prompt, user_prompt)
    except Exception as e:
        logger.exception("OpenAI error: %s", e)
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

    audio_url = None
    try:
        if GTTS_AVAILABLE:
            lang_code = (req.language or "en")
            fallback = {"mni": "en"}
            tts_lang = fallback.get(lang_code, lang_code)
            try:
                audio_url = generate_tts(answer_text, lang=tts_lang, filename_prefix="answer")
            except Exception as e:
                logger.warning("TTS generation failed: %s", e)
                audio_url = None
    except Exception as e:
        logger.warning("TTS section failed unexpectedly: %s", e)
        audio_url = None

    return {"answer": answer_text, "audio_url": audio_url}

# --------- Endpoint: voice transcription (optional) ----------
@app.post("/voice/")
async def voice_transcribe(file: UploadFile = File(...)):
    if not WHISPER_AVAILABLE or whisper_model is None:
        logger.info("Whisper not installed or model not loaded.")
        return {"error": "Server-side transcription not available. Whisper not installed."}
    try:
        content = await file.read()
        tmp = f"temp_audio_{uuid.uuid4().hex[:8]}.wav"
        with open(tmp, "wb") as f:
            f.write(content)
        result = whisper_model.transcribe(tmp)
        text = result.get("text", "").strip()
        try:
            os.remove(tmp)
        except Exception:
            pass
        return {"text": text}
    except Exception as e:
        logger.exception("Voice transcription failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app" if __package__ else "main:app", host="0.0.0.0", port=PORT)
