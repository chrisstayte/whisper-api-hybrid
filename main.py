import os
import logging
import sys
import threading
import platform
import shutil
import time as time_module
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
from pythonjsonlogger import json as jsonlogger

# Pino-style numeric log levels
PINO_LEVELS = {
    logging.DEBUG: 20,
    logging.INFO: 30,
    logging.WARNING: 40,
    logging.ERROR: 50,
    logging.CRITICAL: 60,
}

class PinoJsonFormatter(jsonlogger.JsonFormatter):
    """Structured JSON formatter matching pino output style."""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["level"] = PINO_LEVELS.get(record.levelno, record.levelno)
        log_record["time"] = int(record.created * 1000)
        log_record["pid"] = record.process
        log_record["hostname"] = platform.node()
        log_record["msg"] = record.getMessage()
        log_record.pop("message", None)
        log_record.pop("taskName", None)

# 1. Setup Logging
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(PinoJsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

load_dotenv()


def _get_system_info() -> dict:
    """Gather comprehensive hardware and runtime information."""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "cpu_count_logical": os.cpu_count(),
        "pid": os.getpid(),
    }

    # Physical CPU info
    try:
        info["processor"] = platform.processor() or "unknown"
    except Exception:
        info["processor"] = "unknown"

    # Memory
    try:
        import resource
        # Soft & hard limits for the process
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        info["rlimit_as_soft"] = soft
        info["rlimit_as_hard"] = hard
    except Exception:
        pass

    # Read /proc/meminfo for total system memory (Linux)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    info["memory_total_mb"] = round(kb / 1024, 1)
                elif line.startswith("MemAvailable"):
                    kb = int(line.split()[1])
                    info["memory_available_mb"] = round(kb / 1024, 1)
    except Exception:
        pass

    # Disk
    try:
        disk = shutil.disk_usage("/")
        info["disk_total_gb"] = round(disk.total / (1024**3), 2)
        info["disk_free_gb"] = round(disk.free / (1024**3), 2)
        info["disk_used_pct"] = round((disk.used / disk.total) * 100, 1)
    except Exception:
        pass

    # GPU / CUDA
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory
            info["cuda_memory_mb"] = round(mem / (1024**2), 1)
    except ImportError:
        info["cuda_available"] = False

    return info


def _log_startup_banner():
    """Emit rich startup diagnostics in structured JSON."""
    sys_info = _get_system_info()
    logger.info("System information collected", extra={"system": sys_info})


_log_startup_banner()


# 2. Concurrency Control
try:
    MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "0"))
except ValueError:
    logger.warning("Invalid MAX_CONCURRENT_JOBS value, defaulting to 0", extra={"raw_value": os.getenv("MAX_CONCURRENT_JOBS")})
    MAX_CONCURRENT_JOBS = 0
if MAX_CONCURRENT_JOBS == 0:
    transcription_semaphore = None
    logger.info("Concurrency limit configured", extra={"max_concurrent_jobs": "unlimited"})
else:
    transcription_semaphore = threading.Semaphore(MAX_CONCURRENT_JOBS)
    logger.info("Concurrency limit configured", extra={"max_concurrent_jobs": MAX_CONCURRENT_JOBS})

app = FastAPI()

# Configuration
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
device = "cuda" if USE_GPU else "cpu"
CALLBACK_SECRET = os.getenv("CALLBACK_SECRET")

logger.info("Application configuration loaded", extra={
    "config": {
        "whisper_model": MODEL_SIZE,
        "device": device,
        "use_gpu": USE_GPU,
        "compute_type": "int8",
        "callback_secret_set": CALLBACK_SECRET is not None,
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS if MAX_CONCURRENT_JOBS > 0 else "unlimited",
    }
})

model_load_start = time_module.monotonic()
logger.info("Loading Whisper model", extra={"model": MODEL_SIZE, "device": device, "compute_type": "int8"})
local_model = WhisperModel(MODEL_SIZE, device=device, compute_type="int8")
model_load_elapsed = round(time_module.monotonic() - model_load_start, 2)
logger.info("Whisper model loaded successfully", extra={"model": MODEL_SIZE, "device": device, "load_time_seconds": model_load_elapsed})

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
        logger.warning("Unauthorized request: secret mismatch", extra={"header_present": x_callback_secret is not None})
        raise HTTPException(status_code=401, detail="Invalid Callback Secret")

def process_transcription(req: TranscriptionRequest):
    def _run():
        temp_file = f"/tmp/{req.job_id}.mp3"
        job_start = time_module.monotonic()
        logger.info("Job started", extra={"job_id": req.job_id, "provider": req.provider, "file_url": req.file_url, "callback_url": req.callback_url})
        
        try:
            # Step 1: Download
            logger.info("Downloading audio file", extra={"job_id": req.job_id, "step": "1/3", "file_url": req.file_url})
            dl_start = time_module.monotonic()
            audio_response = requests.get(req.file_url)
            audio_response.raise_for_status()
            
            with open(temp_file, "wb") as f:
                f.write(audio_response.content)
            file_size_mb = round(len(audio_response.content) / (1024 * 1024), 2)
            dl_elapsed = round(time_module.monotonic() - dl_start, 2)
            logger.info("Audio download complete", extra={
                "job_id": req.job_id,
                "step": "1/3",
                "temp_file": temp_file,
                "file_size_mb": file_size_mb,
                "download_time_seconds": dl_elapsed,
                "http_status": audio_response.status_code,
                "content_type": audio_response.headers.get("content-type"),
            })

            payload_content = []

            # Step 2: Transcribe
            transcribe_start = time_module.monotonic()
            if req.provider == "openai":
                logger.info("Transcribing with OpenAI Cloud API", extra={"job_id": req.job_id, "step": "2/3", "provider": "openai"})
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                audio = AudioSegment.from_file(temp_file)
                
                chunk_length_ms = 15 * 60 * 1000 
                duration_ms = len(audio)
                total_chunks = math.ceil(duration_ms / chunk_length_ms)
                logger.info("Audio file analysed", extra={
                    "job_id": req.job_id,
                    "duration_seconds": round(duration_ms / 1000, 2),
                    "total_chunks": total_chunks,
                    "chunk_length_minutes": 15,
                })

                for idx, i in enumerate(range(0, duration_ms, chunk_length_ms)):
                    chunk_start = time_module.monotonic()
                    logger.info("Processing chunk", extra={"job_id": req.job_id, "chunk": f"{idx + 1}/{total_chunks}"})
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
                        
                        segment_count = 0
                        for s in response.segments:
                            seg = s if isinstance(s, dict) else s.__dict__
                            start = seg['start'] + time_offset
                            end = seg['end'] + time_offset
                            payload_content.append({
                                "text": seg['text'].strip(),
                                "timestamp": f"{format_timestamp(start)}-{format_timestamp(end)}"
                            })
                            segment_count += 1
                    os.remove(chunk_name)
                    chunk_elapsed = round(time_module.monotonic() - chunk_start, 2)
                    logger.info("Chunk processed", extra={
                        "job_id": req.job_id,
                        "chunk": f"{idx + 1}/{total_chunks}",
                        "segments_in_chunk": segment_count,
                        "chunk_time_seconds": chunk_elapsed,
                    })
            
            else:
                logger.info("Transcribing with local Faster-Whisper", extra={"job_id": req.job_id, "step": "2/3", "provider": "local", "model": MODEL_SIZE, "device": device})
                segments, info = local_model.transcribe(temp_file, beam_size=5)
                logger.info("Language detected", extra={
                    "job_id": req.job_id,
                    "language": info.language,
                    "language_probability": round(info.language_probability, 4),
                })
                
                segment_count = 0
                for s in segments:
                    payload_content.append({
                        "text": s.text.strip(),
                        "timestamp": f"{format_timestamp(s.start)}-{format_timestamp(s.end)}"
                    })
                    segment_count += 1
                logger.info("Local transcription segments collected", extra={"job_id": req.job_id, "segment_count": segment_count})

            transcribe_elapsed = round(time_module.monotonic() - transcribe_start, 2)
            logger.info("Transcription complete", extra={
                "job_id": req.job_id,
                "step": "2/3",
                "provider": req.provider,
                "total_segments": len(payload_content),
                "transcription_time_seconds": transcribe_elapsed,
            })

            # Step 3: Callback
            logger.info("Sending callback with results", extra={"job_id": req.job_id, "step": "3/3", "callback_url": req.callback_url, "segment_count": len(payload_content)})
            payload = {
                    "job_id": req.job_id,
                    "status": "completed",
                    "transcription": payload_content,
                    "secret": CALLBACK_SECRET
            }

            cb_start = time_module.monotonic()
            cb_res = requests.post(
                req.callback_url, 
                json=payload, 
                headers={"X-Callback-Secret": CALLBACK_SECRET},
                timeout=30
            )
            cb_elapsed = round(time_module.monotonic() - cb_start, 2)
            logger.info("Callback delivered", extra={
                "job_id": req.job_id,
                "step": "3/3",
                "callback_status": cb_res.status_code,
                "callback_time_seconds": cb_elapsed,
            })

        except Exception as e:
            logger.error("Job failed", extra={"job_id": req.job_id, "error": str(e), "error_type": type(e).__name__}, exc_info=True)
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
            except Exception:
                logger.error("Error callback delivery failed", extra={"job_id": req.job_id})
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.info("Temp file cleaned up", extra={"job_id": req.job_id, "temp_file": temp_file})
            total_elapsed = round(time_module.monotonic() - job_start, 2)
            logger.info("Job finished", extra={"job_id": req.job_id, "total_time_seconds": total_elapsed})

    if transcription_semaphore is not None:
        with transcription_semaphore:
            _run()
    else:
        _run()

@app.post("/transcribe", dependencies=[Depends(verify_secret)])
async def start_transcription(req: TranscriptionRequest, background_tasks: BackgroundTasks):
    logger.info("Transcription request received", extra={"job_id": req.job_id, "provider": req.provider, "file_url": req.file_url})
    background_tasks.add_task(process_transcription, req)
    return {"status": "accepted", "job_id": req.job_id}