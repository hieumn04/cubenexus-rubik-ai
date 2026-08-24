# AI Rubik Service

FastAPI skeleton service for CubeNexus Online Arena Rubik color recognition.

## Phase 2 scope

- `GET /health`
- `POST /ai/pre-check`
- `POST /ai/scramble-check`
- `POST /ai/finish-check`
- `POST /ai/analyze-frame`
- Config-driven model path
- Stub responses that keep .NET integration unblocked

## Run

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8010
```

## Environment variables

- `AI_RUBIK_MODEL_PATH`
- `AI_RUBIK_MODEL_VERSION`
- `AI_RUBIK_SERVICE_NAME`
- `AI_RUBIK_ENABLE_MODEL_LOAD`

## Railway

Set the service root directory to `/ai_rubik_service`. The included
`railway.toml` starts Uvicorn on Railway's `$PORT` and checks `/health`.
