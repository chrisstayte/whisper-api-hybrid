# Whisper API Hybrid

A FastAPI-based audio transcription service with a hybrid approach: run [faster-whisper](https://github.com/guillaumekln/faster-whisper) locally for private, cost-effective transcription, or offload to OpenAI or Groq for higher throughput — switchable by environment default or per request.

## Features

- **Hybrid providers** — choose `local`, `openai`, or `groq` by default or per request
- **Async job queue** — requests are accepted immediately and processed in the background
- **Sequential locking** — a configurable concurrency limit controls how many jobs run simultaneously
- **Webhook callbacks** — results are POSTed to your URL when done, no polling needed
- **Large file support** — auto-splits audio into chunks for cloud API file-size limits
- **Optional auth** — secret token validation on both incoming requests and outgoing callbacks
- **Multi-arch Docker image** — prebuilt for `linux/amd64` and `linux/arm64`

## Quick Start

### 1. Pull the image

Prebuilt images are published to GitHub Container Registry on every push to `main`.

```bash
docker pull ghcr.io/chrisstayte/whisper-api-hybrid:latest
```

### 2. Create an environment file

```bash
cp .env.example .env
```

Edit `.env` with your values:

```dotenv
# Required only if using the "openai" provider
OPENAI_API_KEY=sk-proj-your-actual-key

# Required only if using the "groq" provider
GROQ_API_KEY=gsk_your-actual-key

# Default provider: local, openai, or groq
TRANSCRIPTION_PROVIDER=local

# Local whisper model size: tiny, base, small, medium, large-v2
WHISPER_MODEL=small

# Set to "true" if you have an NVIDIA GPU with the NVIDIA Container Toolkit installed
USE_GPU=false

# Optional shared secret for request authentication and callback verification
CALLBACK_SECRET=your_secret_passphrase

# Maximum number of concurrent transcription jobs (0 = unlimited)
MAX_CONCURRENT_JOBS=0
```

### 3. Run it

```bash
docker compose up -d
```

The API is available at `http://localhost:3443`.

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `WHISPER_MODEL` | Local model size (`tiny`, `base`, `small`, `medium`, `large-v2`) | `base` |
| `USE_GPU` | Enable CUDA for local transcription (requires NVIDIA GPU + runtime) | `false` |
| `CALLBACK_SECRET` | Shared secret for authenticating requests and verifying callbacks | _none_ |
| `TRANSCRIPTION_PROVIDER` | Default provider when a request does not specify one (`local`, `openai`, `groq`) | `local` |
| `OPENAI_API_KEY` | OpenAI API key (required only when using the `openai` provider) | _none_ |
| `OPENAI_TRANSCRIPTION_MODEL` | OpenAI transcription model | `whisper-1` |
| `GROQ_API_KEY` | Groq API key (required only when using the `groq` provider) | _none_ |
| `GROQ_TRANSCRIPTION_MODEL` | Groq transcription model | `whisper-large-v3-turbo` |
| `MAX_CONCURRENT_JOBS` | Maximum number of transcription jobs that can run at the same time (`0` = unlimited) | `0` |

## Docker Compose Examples

### CPU-only (simplest setup)

```yaml
services:
  whisper-api:
    image: ghcr.io/chrisstayte/whisper-api-hybrid:latest
    ports:
      - "3443:8000"
    env_file:
      - .env
    volumes:
      - whisper-models:/root/.cache/huggingface
    restart: unless-stopped

volumes:
  whisper-models:
```

### NVIDIA GPU

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```yaml
services:
  whisper-api:
    image: ghcr.io/chrisstayte/whisper-api-hybrid:latest
    ports:
      - "3443:8000"
    env_file:
      - .env
    volumes:
      - whisper-models:/root/.cache/huggingface
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  whisper-models:
```

Set `USE_GPU=true` in your `.env` file.

### Cloud-only (no local model needed)

If `TRANSCRIPTION_PROVIDER` is set to `openai` or `groq`, the local model is not loaded on boot. If a request later overrides the provider to `local`, the model loads on that first local job.

```yaml
services:
  whisper-api:
    image: ghcr.io/chrisstayte/whisper-api-hybrid:latest
    ports:
      - "3443:8000"
    environment:
      - TRANSCRIPTION_PROVIDER=groq
      - GROQ_API_KEY=gsk_your-key
      - WHISPER_MODEL=tiny
    restart: unless-stopped
```

### Build from source

If you prefer to build locally instead of pulling the prebuilt image:

```yaml
services:
  whisper-api:
    build: .
    ports:
      - "3443:8000"
    env_file:
      - .env
    volumes:
      - whisper-models:/root/.cache/huggingface
    restart: unless-stopped

volumes:
  whisper-models:
```

```bash
docker compose up -d --build
```

### With a reverse proxy (Caddy)

```yaml
services:
  whisper-api:
    image: ghcr.io/chrisstayte/whisper-api-hybrid:latest
    env_file:
      - .env
    volumes:
      - whisper-models:/root/.cache/huggingface
    restart: unless-stopped

  caddy:
    image: caddy:2
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile
      - caddy-data:/data
    restart: unless-stopped

volumes:
  whisper-models:
  caddy-data:
```

Example `Caddyfile`:

```
whisper.example.com {
    reverse_proxy whisper-api:8000
}
```

## API Reference

### `POST /transcribe`

Submits an audio file for transcription. The job is processed asynchronously and results are delivered via webhook callback.

**Headers**

| Header | Required | Description |
|---|---|---|
| `X-Callback-Secret` | Only if `CALLBACK_SECRET` is set | Must match the configured secret |

**Request Body**

```json
{
  "file_url": "https://example.com/audio.mp3",
  "job_id": "unique-job-123",
  "callback_url": "https://your-server.com/webhook",
  "provider": "local"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `file_url` | string | yes | Direct URL to the audio file |
| `job_id` | string | yes | Unique identifier for this job |
| `callback_url` | string | yes | URL where results will be POSTed |
| `provider` | string | no | Overrides `TRANSCRIPTION_PROVIDER`; supported values are `local`, `openai`, and `groq` |

Omit `provider` to use the configured `TRANSCRIPTION_PROVIDER`. For example, set `TRANSCRIPTION_PROVIDER=groq` and `GROQ_API_KEY=...` in `.env` to send all default jobs to Groq.

**Response** `200 OK`

```json
{
  "status": "accepted",
  "job_id": "unique-job-123"
}
```

### Callback Payload

When transcription completes, the service sends a `POST` to your `callback_url`.

The callback includes the `X-Callback-Secret` header if `CALLBACK_SECRET` is configured.

Empty transcription segments are discarded before sending the callback payload because
CMS receivers may reject `transcription` items whose `text` value is blank.

**Success**

```json
{
  "job_id": "unique-job-123",
  "status": "completed",
  "transcription": [
    {
      "text": "Hello, this is a test.",
      "timestamp": "00:00-00:02"
    },
    {
      "text": "Welcome to the hybrid whisper API.",
      "timestamp": "00:02-00:05"
    }
  ],
  "secret": "your-configured-callback-secret"
}
```

**Failure**

```json
{
  "job_id": "unique-job-123",
  "status": "failed",
  "error": "Description of what went wrong",
  "secret": "your-configured-callback-secret"
}
```

## How It Works

1. **Request** — you POST to `/transcribe` with an audio URL and callback URL
2. **Queue** — the server accepts immediately and processes in the background
3. **Concurrency control** — a configurable semaphore limits how many jobs run at once (default: 0 = unlimited)
4. **Download** — the audio file is downloaded to a temp location
5. **Transcribe** — either locally via faster-whisper, or via OpenAI/Groq (auto-chunked if needed)
6. **Callback** — results (text + timestamps) are POSTed to your callback URL
7. **Cleanup** — temp files are deleted

## CI/CD

The GitHub Actions workflow (`.github/workflows/docker.yml`) runs on every push to `main`:

1. Auto-bumps the version (semver patch)
2. Builds a multi-arch Docker image (`linux/amd64` + `linux/arm64`) using buildx
3. Pushes to `ghcr.io/chrisstayte/whisper-api-hybrid` with `latest` and versioned tags
4. Creates a GitHub Release with changelog

To trigger a minor or major bump, include `#minor` or `#major` in your commit message.

## Local Development

```bash
# Clone
git clone https://github.com/chrisstayte/whisper-api-hybrid.git
cd whisper-api-hybrid

# Create env file
cp .env.example .env

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --host 0.0.0.0 --port 8000
```

Requires Python 3.10+ and ffmpeg installed on your system.

## Dependencies

- [FastAPI](https://fastapi.tiangolo.com/) — web framework
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) — local transcription engine
- [OpenAI Python](https://github.com/openai/openai-python) — OpenAI and Groq-compatible cloud transcription
- [PyDub](https://github.com/jiaaro/pydub) — audio chunking
- [FFmpeg](https://ffmpeg.org/) — audio processing (installed in Docker image)
