# backend/main.py
import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import openai
import requests

# Embeddings & FAISS from langchain-community
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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

# --------- CONFIG (replace placeholders with your keys) ----------
OPENAI_API_KEY = "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"
WEATHER_API_KEY = "17fd2005310756adff0cda284cb08e6e"  # optional; leave blank if not using weather
FAISS_INDEX_PATH = "faiss_index"  # path to your saved vectorstore directory
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agri-backend")

# Set OpenAI key for openai package
openai.api_key = OPENAI_API_KEY

app = FastAPI(title="AgriGPT Backend")

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (audio)
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --------- Load embeddings and FAISS vectorstore (safe) ----------
logger.info("Loading embeddings and FAISS index...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
try:
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    logger.info("FAISS index loaded.")
except Exception as e:
    logger.error("Failed to load FAISS index. Reason: %s", e)
    vectorstore = None

# Load whisper model optionally
if WHISPER_AVAILABLE:
    logger.info("Loading Whisper model (may take time)...")
    whisper_model = whisper.load_model("base")
else:
    whisper_model = None
    logger.info("Whisper not available. /voice will return an explanatory error.")

# --------- Request model ----------
class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = "en"  # language code from frontend (optional)

# --------- Utilities ----------
def fetch_weather_for_city(city_name: str) -> Optional[str]:
    if not WEATHER_API_KEY or WEATHER_API_KEY.startswith("YOUR_"):
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
        logger.exception("OpenAI call failed")
        raise

def generate_tts(text: str, lang: str = "en", filename_prefix: str = "answer") -> str:
    """Generate TTS MP3 using gTTS and return relative static path (e.g. static/...)."""
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
    # Validate vectorstore loaded
    if vectorstore is None:
        raise HTTPException(status_code=500, detail="FAISS vectorstore not loaded on server.")

    user_query = (req.query or "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Empty query")

    # --- retrieve relevant context from FAISS ---
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(user_query)  # returns list of Document with .page_content
        context_text = "\n\n".join([d.page_content for d in docs]) if docs else ""
    except Exception as e:
        logger.exception("Retrieval error")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    # --- optionally attach real-time info (example weather) ---
    # This is a simple example: you can detect location or accept param.
    realtime_info = ""
    # If user mentions weather, attach Chennai weather (customize as needed)
    if any(w in user_query.lower() for w in ["weather", "temperature", "rain", "forecast"]):
        w = fetch_weather_for_city("Chennai")
        if w:
            realtime_info = w

    # --- Build system/user prompts and call OpenAI ---
    system_prompt = (
        "You are AgriGPT, an expert assistant for farmers. Use the supplied context to answer accurately. "
        "Always answer in the same language as the user's question. Be concise, practical and actionable for farmers."
    )

    # Construct user prompt including context and instruction to prefer context if available
    user_prompt_parts = [
        f"Question: {user_query}",
    ]
    if context_text:
        user_prompt_parts.append(f"Context:\n{context_text}")
    if realtime_info:
        user_prompt_parts.append(f"RealTimeData:\n{realtime_info}")

    user_prompt_parts.append("Answer using the context if relevant. If context doesn't contain the answer, use general agricultural knowledge.")
    user_prompt = "\n\n".join(user_prompt_parts)

    try:
        answer_text = call_openai_chat(system_prompt, user_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

    # --- Produce TTS file (best-effort) ---
    audio_url = None
    try:
        # choose gTTS language code from requested language if possible
        # frontend should pass language codes like 'en', 'hi', 'ta', etc.
        lang_code = (req.language or "en")[:2]
        # gTTS supports many languages (en, hi, ta, te, ml, kn, bn, mr, pa, as [approx], etc.)
        # For Manipuri you may fallback to 'en' if not supported.
        # gTTS language list: https://gtts.readthedocs.io
        if GTTS_AVAILABLE:
            # fallback map for some codes:
            fallback = {"mni": "en", "as": "en"}  # adjust if you have better TTS for these langs
            tts_lang = fallback.get(req.language, req.language) if req.language else "en"
            # gTTS expects 2-letter sometimes; trim to first 2
            try:
                audio_url = generate_tts(answer_text, lang=tts_lang, filename_prefix="answer")
            except Exception as e:
                logger.warning("TTS generation failed: %s", e)
                audio_url = None
    except Exception:
        audio_url = None

    return {"answer": answer_text, "audio_url": audio_url}

# --------- Endpoint: voice transcription (optional) ----------
@app.post("/voice/")
async def voice_transcribe(file: UploadFile = File(...)):
    """Accept audio blob uploaded from frontend and transcribe using whisper (server-side)."""
    if not WHISPER_AVAILABLE or whisper_model is None:
        return {"error": "Server-side transcription not available. Whisper not installed."}
    try:
        content = await file.read()
        tmp = "temp_audio.wav"
        with open(tmp, "wb") as f:
            f.write(content)
        result = whisper_model.transcribe(tmp)
        text = result.get("text", "").strip()
        return {"text": text}
    except Exception as e:
        logger.exception("Voice transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
