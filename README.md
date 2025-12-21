# Whisper API Hybrid

A robust, FastAPI-based audio transcription service that offers a hybrid approach to transcription. You can choose between running a local instance of `faster-whisper` for cost-effective, private transcription, or offload the task to OpenAI's Whisper API for higher throughput or specific use cases.

## Features

- **Hybrid Providers**: Switch between `local` (on-device) and `openai` (cloud) transcription providers per request.
- **Asynchronous Processing**: Jobs are queued and processed in the background.
- **Sequential Execution**: A global lock ensures that local resources are not overwhelmed by concurrent transcription jobs.
- **Callback System**: Results are pushed to a webhook URL upon completion, so you don't have to poll for status.
- **Large File Support**: Automatically splits large audio files when using the OpenAI provider to adhere to API limits.
- **Secure**: Optional secret token validation for incoming requests and outgoing callbacks.
- **Docker Ready**: Fully containerized with Docker and Docker Compose.

## Prerequisites

- **Docker** and **Docker Compose** installed on your machine.
- **OpenAI API Key** (optional, only if using the `openai` provider).

## Configuration

The application is configured via environment variables. You can set these in a `.env` file or in your `docker-compose.yml`.

| Variable          | Description                                                                                                | Default |
| ----------------- | ---------------------------------------------------------------------------------------------------------- | ------- |
| `WHISPER_MODEL`   | The size of the local Whisper model to load (e.g., `tiny`, `base`, `small`, `medium`, `large-v2`).         | `base`  |
| `USE_GPU`         | Set to `true` to use CUDA for local transcription. Requires a compatible NVIDIA GPU and container runtime. | `false` |
| `CALLBACK_SECRET` | A secret string used to authenticate requests to this API and verify callbacks sent by it.                 | `None`  |
| `OPENAI_API_KEY`  | Your OpenAI API key. Required only if `provider` is set to `openai`.                                       | `None`  |

## Installation & Usage

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd whisper-api-hybrid
    ```

2.  **Set up environment variables:**
    Create a `.env` file or modify `docker-compose.yml` with your desired settings.

3.  **Run with Docker Compose:**
    ```bash
    docker compose up -d
    ```
    The API will be available at `http://localhost:3443`.

## API Reference

### Start Transcription

**Endpoint:** `POST /transcribe`

**Headers:**

- `X-Callback-Secret`: (Optional) Must match `CALLBACK_SECRET` if set in the environment.

**Request Body:**

```json
{
  "file_url": "https://example.com/path/to/audio.mp3",
  "job_id": "unique-job-identifier-123",
  "callback_url": "https://your-server.com/webhook",
  "provider": "local"
}
```

| Field          | Type   | Description                                     |
| -------------- | ------ | ----------------------------------------------- |
| `file_url`     | string | Direct URL to the audio file to be transcribed. |
| `job_id`       | string | A unique identifier for this job.               |
| `callback_url` | string | The URL where the results will be POSTed.       |
| `provider`     | string | `local` (default) or `openai`.                  |

**Response:**

```json
{
  "status": "accepted",
  "job_id": "unique-job-identifier-123"
}
```

## Callback Payload

Once the transcription is finished, the service will send a `POST` request to your `callback_url`.

**Headers:**

- `X-Callback-Secret`: (Optional) Matches `CALLBACK_SECRET` if set in the environment.

### Success Response

```json
{
  "job_id": "unique-job-identifier-123",
  "status": "completed",
  "transcription": [
    {
      "text": "Hello, this is a test.",
      "timestamp": "00:00:00-00:00:02"
    },
    {
      "text": "Welcome to the hybrid whisper API.",
      "timestamp": "00:00:02-00:00:05"
    }
  ],
  "secret": "your-configured-callback-secret"
}
```

### Failure Response

```json
{
  "job_id": "unique-job-identifier-123",
  "status": "failed",
  "error": "Description of what went wrong",
  "secret": "your-configured-callback-secret"
}
```

## How It Works

1.  **Request**: You send a request to `/transcribe` with the audio URL and a callback URL.
2.  **Queue**: The server accepts the request immediately and processes it in the background.
3.  **Lock**: A global lock ensures that only one transcription job runs at a time to prevent server overload (especially important for the local model).
4.  **Download**: The server downloads the audio file to a temporary location.
5.  **Transcribe**:
    - **Local**: Uses `faster-whisper` to process the file.
    - **OpenAI**: Splits the file into chunks (if necessary) and sends them to the OpenAI Whisper API.
6.  **Callback**: The results (text and timestamps) are formatted and sent to your `callback_url`.
7.  **Cleanup**: Temporary files are deleted.

## Dependencies

- [FastAPI](https://fastapi.tiangolo.com/)
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)
- [OpenAI Python Client](https://github.com/openai/openai-python)
- [PyDub](https://github.com/jiaaro/pydub)
- FFmpeg (installed in Docker image)
