import os
import tempfile
import whisper
import requests
import torch
from transformers import pipeline
from summarizer import Summarizer 
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables (e.g., Telegram tokens) from .env file
load_dotenv()

# --- Config ---
# Read from .env, using defaults (your provided values) if not found
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8436255248:AAFeNa_MTkzpealBEeyfKwXXuNuwBiEgvOg") 
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1282848803") 
WHISPER_SAMPLE_RATE = whisper.audio.SAMPLE_RATE

# --- FastAPI Setup ---
app = FastAPI(title="AI Lecture Processor API")

# Configure CORS to allow the frontend (index.html) to communicate with the server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# --- Global Model Initialization ---
@app.on_event("startup")
async def startup_event():
    print("Pre-loading models...")
    # BERT Extractive Model
    app.state.bert_summarizer = Summarizer() 
    # BART Abstractive Model. Use GPU (device=0) if available, otherwise CPU (device=-1)
    try:
        app.state.bart_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        print(f"Failed to load BART model: {e}. Abstractive summarization will be skipped if needed.")
        app.state.bart_pipeline = None 
    print("Models loaded successfully.")

# --- Utility Functions ---

def send_telegram_notification(summary_text: str):
    """Sends the final summary via the Telegram Bot API."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    heading = "SUMMARY\n"
    summary_text = heading + summary_text

    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": summary_text}
    
    try:
        r = requests.post(url, data=payload)
        r.raise_for_status() # Raise an exception for bad status codes
        print(f"Telegram notification sent successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error sending Telegram message: {e}")

def run_whisper_transcription(audio_file_path: str, model_size: str, max_minutes: int) -> str:
    """Runs the chunked transcription process using the whisper model."""
    print(f"Loading Whisper model: {model_size}...")
    model = whisper.load_model(model_size)
    
    print("Loading audio...")
    audio = whisper.load_audio(audio_file_path)

    # Convert max minutes to samples and truncate
    max_samples = max_minutes * 60 * WHISPER_SAMPLE_RATE
    audio = audio[:max_samples]

    # Chunked transcription logic 
    CHUNK_DURATION = 60 # seconds
    chunk_samples = CHUNK_DURATION * WHISPER_SAMPLE_RATE
    transcript = []

    print("Transcribing in chunks...")
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        chunk = whisper.pad_or_trim(chunk)
        
        # Use GPU/CPU device
        mel = whisper.log_mel_spectrogram(chunk).to(model.device)
        
        # Use fp16=False for stability on small models or CPU
        result = model.decode(mel, whisper.DecodingOptions(fp16=False))
        transcript.append(result.text)
    
    return " ".join(transcript)

def run_hybrid_summarization(transcript: str) -> str:
    """Runs the two-step extractive (BERT) and abstractive (BART) summarization."""
    print("\nRunning Extractive Summarization (BERT)...")
    
    # 1. Extractive Summarization (BERT)
    bert_model = app.state.bert_summarizer
    extractive_summary = bert_model(transcript, min_length=60, max_length=500)
    
    if not extractive_summary:
        return "The transcription was too short or lacked meaningful content to summarize."

    # 2. Abstractive Summarization (BART)
    summarizer_pipeline = app.state.bart_pipeline
    
    if not summarizer_pipeline:
        print("BART pipeline not available. Returning only extractive summary.")
        return extractive_summary

    print("\nRefining with Abstractive Summarization (BART)...")
    MAX_CHARS = 1000
    final_summary_parts = []
    
    # Chunk the extractive summary if it's too long for BART's context window
    for i in range(0, len(extractive_summary), MAX_CHARS):
        chunk = extractive_summary[i:i + MAX_CHARS]
        summary_result = summarizer_pipeline(
            chunk,
            max_length=52,
            min_length=10,
            do_sample=False
        )
        final_summary_parts.append(summary_result[0]['summary_text'])
    
    final_summary = " ".join(final_summary_parts)
    return final_summary

# --- API Endpoint ---

@app.post("/process")
async def process_lecture(
    audio_file: UploadFile = File(...),
    model_size: str = Form("small"),
    max_minutes: int = Form(5)
):
    """
    Receives an audio file and parameters, runs the transcription and summarization pipeline,
    sends a Telegram notification, and returns the results.
    """
    allowed_types = ["audio/mpeg", "audio/mp3", "audio/wav", "audio/ogg"]
    if audio_file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Must be one of: {', '.join(allowed_types)}.")

    if max_minutes < 1 or max_minutes > 120:
        raise HTTPException(status_code=400, detail="Max minutes must be between 1 and 120.")

    transcript = ""
    final_summary = ""
    
    # Use a temporary file to save the uploaded audio for Whisper to process
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.filename.split('.')[-1]}") as temp_audio:
        try:
            # Write the uploaded content to the temporary file
            content = await audio_file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
            
            # CRUCIAL FIX: Close the file handle now to release the lock on Windows
            temp_audio.close() 
            
            # --- 1. Transcription ---
            transcript = run_whisper_transcription(temp_audio_path, model_size, max_minutes)
            
            # --- 2. Summarization ---
            final_summary = run_hybrid_summarization(transcript)
            
            # --- 3. Notification ---
            send_telegram_notification(final_summary)

        except Exception as e:
            print(f"An error occurred during pipeline execution: {e}")
            raise HTTPException(status_code=500, detail=f"Pipeline Error: {str(e)}")
        finally:
            # Ensure the temporary file is deleted after processing
            # This deletion should now succeed because we explicitly closed the file above
            os.unlink(temp_audio_path)
            
    # Return final results to the frontend
    return {
        "status": "complete",
        "transcript": transcript,
        "summary": final_summary,
        "model_size": model_size,
        "max_minutes": max_minutes
    }

# To run the server: uvicorn backend_app:app --reload
