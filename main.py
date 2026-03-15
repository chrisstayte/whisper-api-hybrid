import os
import logging
import sys
import threading
from dotenv import load_dotenv
import requests
import datetime
import math
from fastapi import FastAPI, BackgroundTasks, HTTPException, Header, Depends
from pydantic import BaseModel
from faster_whisper import WhisperModel
import openai
from pydub import AudioSegment
from typing import Annotated

# 1. Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

load_dotenv()

# 2. Concurrency Control
try:
    MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))
except ValueError:
    logger.warning("Invalid MAX_CONCURRENT_JOBS value, defaulting to 1")
    MAX_CONCURRENT_JOBS = 1
if MAX_CONCURRENT_JOBS == 0:
    transcription_semaphore = None
    logger.info("Concurrency limit: unlimited")
else:
    transcription_semaphore = threading.Semaphore(MAX_CONCURRENT_JOBS)
    logger.info(f"Concurrency limit: {MAX_CONCURRENT_JOBS}")

app = FastAPI()

# Configuration
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
device = "cuda" if USE_GPU else "cpu"

logger.info(f"Loading Whisper model: {MODEL_SIZE} on {device}...")
local_model = WhisperModel(MODEL_SIZE, device=device, compute_type="int8")
logger.info("Model loaded and ready.")

CALLBACK_SECRET = os.getenv("CALLBACK_SECRET")

class TranscriptionRequest(BaseModel):
    file_url: str
    job_id: str
    callback_url: str
    provider: str = "local"

def format_timestamp(seconds: float) -> str:
    td = datetime.timedelta(seconds=int(seconds))
    return str(td).split('.')[0].zfill(8)[3:] if seconds < 3600 else str(td).split('.')[0].zfill(8)

async def verify_secret(x_callback_secret: Annotated[str | None, Header()] = None):
    if not CALLBACK_SECRET:
        return 
    if x_callback_secret != CALLBACK_SECRET:
        logger.warning("Unauthorized request: Secret mismatch.")
        raise HTTPException(status_code=401, detail="Invalid Callback Secret")

def process_transcription(req: TranscriptionRequest):
    def _run():
        temp_file = f"/tmp/{req.job_id}.mp3"
        logger.info(f"--- SLOT ACQUIRED | STARTING JOB: {req.job_id} ---")
        logger.info(f"Callback url: {req.callback_url}")
        
        try:
            # Step 1: Download
            logger.info("Step 1/3: Downloading audio...")
            audio_response = requests.get(req.file_url)
            audio_response.raise_for_status()
            
            with open(temp_file, "wb") as f:
                f.write(audio_response.content)
            logger.info(f"Download complete: {temp_file}")

            payload_content = []

            # Step 2: Transcribe
            if req.provider == "openai":
                logger.info("Step 2/3: Using OpenAI Cloud API...")
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                audio = AudioSegment.from_file(temp_file)
                
                chunk_length_ms = 15 * 60 * 1000 
                duration_ms = len(audio)
                total_chunks = math.ceil(duration_ms / chunk_length_ms)

                for idx, i in enumerate(range(0, duration_ms, chunk_length_ms)):
                    logger.info(f"Processing chunk {idx + 1}/{total_chunks}...")
                    chunk = audio[i : i + chunk_length_ms]
                    chunk_name = f"/tmp/{req.job_id}_chunk_{idx}.mp3"
                    chunk.export(chunk_name, format="mp3")
                    
                    time_offset = i / 1000.0
                    with open(chunk_name, "rb") as f:
                        response = client.audio.transcriptions.create(
                            model="whisper-1", 
                            file=f, 
                            response_format="verbose_json"
                        )
                        
                        for s in response.segments:
                            seg = s if isinstance(s, dict) else s.__dict__
                            start = seg['start'] + time_offset
                            end = seg['end'] + time_offset
                            payload_content.append({
                                "text": seg['text'].strip(),
                                "timestamp": f"{format_timestamp(start)}-{format_timestamp(end)}"
                            })
                    os.remove(chunk_name)
            
            else:
                logger.info("Step 2/3: Using local Faster-Whisper...")
                segments, info = local_model.transcribe(temp_file, beam_size=5)
                logger.info(f"Language: {info.language} ({info.language_probability:.2%})")
                
                for s in segments:
                    payload_content.append({
                        "text": s.text.strip(),
                        "timestamp": f"{format_timestamp(s.start)}-{format_timestamp(s.end)}"
                    })

            # Step 3: Callback
            logger.info(f"Step 3/3: Pushing results to {req.callback_url}")
            payload = {
                    "job_id": req.job_id,
                    "status": "completed",
                    "transcription": payload_content,
                    "secret": CALLBACK_SECRET
            }
       


            cb_res = requests.post(
                req.callback_url, 
                json=payload, 
                headers={"X-Callback-Secret": CALLBACK_SECRET},
                timeout=30
            )
            logger.info(f"Callback status: {cb_res.status_code}")

        except Exception as e:
            logger.error(f"FATAL ERROR in job {req.job_id}: {str(e)}", exc_info=True)
            try:
                requests.post(
                    req.callback_url, 
                    json={
                        "job_id": req.job_id,
                        "status": "failed",
                        "error": str(e),
                        "secret": CALLBACK_SECRET
                    },
                    headers={"X-Callback-Secret": CALLBACK_SECRET},
                    timeout=10
                )
            except:
                logger.error("Callback failed after error.")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.info("Cleanup complete.")
            logger.info(f"--- JOB {req.job_id} FINISHED | RELEASING SLOT ---")

    if transcription_semaphore is not None:
        with transcription_semaphore:
            _run()
    else:
        _run()

@app.post("/transcribe", dependencies=[Depends(verify_secret)])
async def start_transcription(req: TranscriptionRequest, background_tasks: BackgroundTasks):
    logger.info(f"New request for Job: {req.job_id}")
    background_tasks.add_task(process_transcription, req)
    return {"status": "accepted", "job_id": req.job_id}