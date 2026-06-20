import os
import logging
import sys
import threading
import json
from urllib.parse import urlparse, unquote
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
local_model_lock = threading.Lock()
try:
    MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "0"))
except ValueError:
    logger.warning("Invalid MAX_CONCURRENT_JOBS value, defaulting to 0")
    MAX_CONCURRENT_JOBS = 0
if MAX_CONCURRENT_JOBS == 0:
    transcription_semaphore = None
    logger.info("Concurrency limit: unlimited")
else:
    transcription_semaphore = threading.Semaphore(MAX_CONCURRENT_JOBS)
    logger.info(f"Concurrency limit: {MAX_CONCURRENT_JOBS}")

app = FastAPI()

# Job Registry (in-memory; job history is lost on application restart)
jobs_lock = threading.Lock()
jobs: dict[str, dict] = {}

STATUS_QUEUED = "queued"
STATUS_DOWNLOADING = "downloading"
STATUS_TRANSCRIBING = "transcribing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class JobStatus(BaseModel):
    job_id: str
    file_url: str
    file_name: str
    status: str


class JobsResponse(BaseModel):
    jobs: list[JobStatus]


def _extract_file_name(url: str) -> str:
    """Extract the original file name from a URL, sanitized for safety."""
    path = urlparse(url).path
    name = os.path.basename(unquote(path))
    # Remove any directory traversal or path separator characters
    name = name.replace("..", "").replace("/", "").replace("\\", "")
    return name.strip() if name.strip() else "unknown"


def _register_job(req: "TranscriptionRequest") -> None:
    with jobs_lock:
        jobs[req.job_id] = {
            "job_id": req.job_id,
            "file_url": req.file_url,
            "file_name": _extract_file_name(req.file_url),
            "status": STATUS_QUEUED,
        }


def _update_job_status(job_id: str, status: str) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["status"] = status


# Configuration
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
TRANSCRIPTION_PROVIDER = os.getenv("TRANSCRIPTION_PROVIDER", "local").lower()
OPENAI_TRANSCRIPTION_MODEL = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1")
GROQ_TRANSCRIPTION_MODEL = os.getenv("GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3-turbo")
SUPPORTED_PROVIDERS = {"local", "openai", "groq"}
device = "cuda" if USE_GPU else "cpu"
local_model = None

def get_local_model():
    global local_model
    if local_model is None:
        with local_model_lock:
            if local_model is None:
                logger.info(f"Loading Whisper model: {MODEL_SIZE} on {device}...")
                local_model = WhisperModel(MODEL_SIZE, device=device, compute_type="int8")
                logger.info("Model loaded and ready.")
    return local_model

if TRANSCRIPTION_PROVIDER == "local":
    get_local_model()
else:
    logger.info(f"Skipping local Whisper model preload for provider: {TRANSCRIPTION_PROVIDER}")

CALLBACK_SECRET = os.getenv("CALLBACK_SECRET")

class TranscriptionRequest(BaseModel):
    file_url: str
    job_id: str
    callback_url: str
    provider: str | None = None

def format_timestamp(seconds: float) -> str:
    td = datetime.timedelta(seconds=int(seconds))
    return str(td).split('.')[0].zfill(8)[3:] if seconds < 3600 else str(td).split('.')[0].zfill(8)

def get_cloud_client(provider: str):
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when provider is openai")
        return openai.OpenAI(api_key=api_key), OPENAI_TRANSCRIPTION_MODEL

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is required when provider is groq")
        return (
            openai.OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            ),
            GROQ_TRANSCRIPTION_MODEL,
        )

    raise ValueError(f"Unsupported cloud provider: {provider}")

def append_payload_segment(payload_content, text: str, start: float, end: float) -> None:
    text = text.strip()
    if not text:
        logger.info(
            "Skipping empty transcription segment at %s-%s",
            format_timestamp(start),
            format_timestamp(end),
        )
        return

    payload_content.append({
        "text": text,
        "timestamp": f"{format_timestamp(start)}-{format_timestamp(end)}"
    })

def append_segments(payload_content, segments, time_offset: float = 0.0):
    for s in segments:
        seg = s if isinstance(s, dict) else s.__dict__
        start = seg["start"] + time_offset
        end = seg["end"] + time_offset
        append_payload_segment(payload_content, seg["text"], start, end)

def log_callback_payload_for_bad_request(payload: dict, response: requests.Response) -> None:
    payload_for_log = payload.copy()
    if payload_for_log.get("secret"):
        payload_for_log["secret"] = "[REDACTED]"

    logger.error("CMS callback returned 400 Bad Request.")
    logger.error(f"CMS response body: {response.text}")
    logger.error(
        "Callback payload: %s",
        json.dumps(payload_for_log, ensure_ascii=False, default=str),
    )

async def verify_secret(x_callback_secret: Annotated[str | None, Header()] = None):
    if not CALLBACK_SECRET:
        return 
    if x_callback_secret != CALLBACK_SECRET:
        logger.warning("Unauthorized request: Secret mismatch.")
        raise HTTPException(status_code=401, detail="Invalid Callback Secret")

def process_transcription(req: TranscriptionRequest):
    def _run():
        temp_file = f"/tmp/{req.job_id}.mp3"
        provider = (req.provider or TRANSCRIPTION_PROVIDER).lower()
        logger.info(f"--- SLOT ACQUIRED | STARTING JOB: {req.job_id} ---")
        logger.info(f"Callback url: {req.callback_url}")
        logger.info(f"Provider: {provider}")
        
        try:
            if provider not in SUPPORTED_PROVIDERS:
                raise ValueError(
                    f"Unsupported provider '{provider}'. "
                    f"Supported providers: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
                )

            # Step 1: Download
            _update_job_status(req.job_id, STATUS_DOWNLOADING)
            logger.info("Step 1/3: Downloading audio...")
            audio_response = requests.get(req.file_url)
            audio_response.raise_for_status()
            
            with open(temp_file, "wb") as f:
                f.write(audio_response.content)
            logger.info(f"Download complete: {temp_file}")

            payload_content = []

            # Step 2: Transcribe
            _update_job_status(req.job_id, STATUS_TRANSCRIBING)
            if provider in {"openai", "groq"}:
                logger.info(f"Step 2/3: Using {provider.title()} Cloud API...")
                client, model = get_cloud_client(provider)
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
                            model=model,
                            file=f, 
                            response_format="verbose_json",
                        )
                        append_segments(payload_content, response.segments, time_offset)
                    os.remove(chunk_name)
            
            else:
                logger.info("Step 2/3: Using local Faster-Whisper...")
                segments, info = get_local_model().transcribe(temp_file, beam_size=5)
                logger.info(f"Language: {info.language} ({info.language_probability:.2%})")
                
                for s in segments:
                    append_payload_segment(payload_content, s.text, s.start, s.end)

            # Step 3: Callback
            _update_job_status(req.job_id, STATUS_COMPLETED)
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
            if cb_res.status_code == 400:
                log_callback_payload_for_bad_request(payload, cb_res)

        except Exception as e:
            _update_job_status(req.job_id, STATUS_FAILED)
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
    _register_job(req)
    background_tasks.add_task(process_transcription, req)
    return {"status": "accepted", "job_id": req.job_id}


@app.get("/jobs", response_model=JobsResponse, dependencies=[Depends(verify_secret)])
async def list_jobs():
    with jobs_lock:
        snapshot = list(jobs.values())
    job_list = [JobStatus(**job) for job in snapshot]
    return JobsResponse(jobs=job_list)
