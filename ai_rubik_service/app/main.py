from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    AiCheckRequest,
    AnalyzeFrameRequest,
    CubeScanFaceMapRequest,
    HealthResponse,
    ScannerFaceScanRequest,
    ScannerFrameRequest,
    ScannerSessionStartRequest,
)
from .service import RubikAiService
from .settings import RubikAiSettings


settings = RubikAiSettings.load()
service = RubikAiService(settings)

app = FastAPI(title=settings.service_name, version="0.1.0-phase2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8081",
        "http://127.0.0.1:8081",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _parse_request_payload(request: Request) -> dict:
    content_type = request.headers.get("content-type", "").lower()
    if "multipart/form-data" not in content_type:
        return await request.json()

    form = await request.form()
    payload: dict[str, object] = {}
    metadata: dict[str, object] = {}

    for key, value in form.multi_items():
        if hasattr(value, "filename") and hasattr(value, "read"):
            file_bytes = await value.read()
            payload["imageBytes"] = file_bytes
            payload["imageFileName"] = value.filename
            payload["imageContentType"] = value.content_type
            continue

        text_value = str(value)
        if key in {"scanGeneration", "targetFaceIndex"}:
            try:
                payload[key] = int(text_value)
            except ValueError:
                payload[key] = text_value
        elif key in {"matchId", "playerId", "checkType", "scrambleSequence", "imageUrl", "scanSessionId", "requestId"}:
            payload[key] = text_value
        else:
            metadata[key] = text_value

    if metadata:
        payload["metadata"] = metadata

    return payload


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        serviceName=settings.service_name,
        modelPath=str(settings.model_path),
        modelExists=settings.model_path.exists(),
        modelVersion=settings.model_version,
        modelLoaded=service.model_loaded,
    )


@app.post("/ai/pre-check")
async def pre_check(request: Request):
    payload = await _parse_request_payload(request)
    if "imageBytes" not in payload:
        parsed = AiCheckRequest.model_validate(payload)
        payload = parsed.model_dump(by_alias=True)
    return service.run_check("PRE_CHECK", payload)


@app.post("/ai/scramble-check")
async def scramble_check(request: Request):
    payload = await _parse_request_payload(request)
    if "imageBytes" not in payload:
        parsed = AiCheckRequest.model_validate(payload)
        payload = parsed.model_dump(by_alias=True)
    return service.run_check("SCRAMBLE_CHECK", payload)


@app.post("/ai/finish-check")
async def finish_check(request: Request):
    payload = await _parse_request_payload(request)
    if "imageBytes" not in payload:
        parsed = AiCheckRequest.model_validate(payload)
        payload = parsed.model_dump(by_alias=True)
    return service.run_check("FINISH_CHECK", payload)


@app.post("/ai/analyze-frame")
async def analyze_frame(request: AnalyzeFrameRequest):
    return service.analyze_frame(request.model_dump(by_alias=True))


@app.post("/ai/validate-cube-scan")
async def validate_cube_scan(request: CubeScanFaceMapRequest):
    return service.validate_cube_scan(request.faces)


@app.post("/ai/scanner-test/session/start")
async def start_scanner_test_session(request: ScannerSessionStartRequest):
    return service.start_scanner_test_session(request.metadata)


@app.get("/ai/scanner-test/session/{session_id}")
async def get_scanner_test_session(session_id: str):
    try:
        return service.get_scanner_test_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/ai/scanner-test/session/{session_id}/preview")
async def preview_scanner_test_frame(session_id: str, request: Request):
    try:
        payload = await _parse_request_payload(request)
        if "imageBytes" not in payload:
            parsed = ScannerFrameRequest.model_validate(payload)
            payload = parsed.model_dump(by_alias=True)
        return service.preview_scanner_test_frame(session_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/ai/scanner-test/session/{session_id}/observe")
async def observe_scanner_test_frame(session_id: str, request: Request):
    try:
        payload = await _parse_request_payload(request)
        if "imageBytes" not in payload:
            parsed = ScannerFrameRequest.model_validate(payload)
            payload = parsed.model_dump(by_alias=True)
        return service.observe_scanner_test_frame(session_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/ai/scanner-test/session/{session_id}/scan-face")
async def scan_scanner_test_face(session_id: str, request: ScannerFaceScanRequest):
    try:
        return service.scan_scanner_test_face(session_id, request.model_dump(by_alias=True))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/ai/scanner-test/session/{session_id}/retry-face")
async def retry_scanner_test_face(session_id: str):
    try:
        return service.retry_scanner_test_face(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/ai/scanner-test/session/{session_id}/reset")
async def reset_scanner_test_session(session_id: str):
    try:
        return service.reset_scanner_test_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
