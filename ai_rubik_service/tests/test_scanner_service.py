from __future__ import annotations

import unittest
from pathlib import Path

from ai_rubik_service.app.service import InferenceTimings, RubikAiService
from ai_rubik_service.app.settings import RubikAiSettings
from rubik_scanner.models import FrameDetectionResult


def make_service() -> RubikAiService:
    service = RubikAiService(
        RubikAiSettings(
            service_name="test",
            model_path=Path("missing.pt"),
            model_version="test-model",
            enable_model_load=False,
            scanner_required_stable_observations=2,
            scanner_stability_timeout_seconds=5.0,
            scanner_stable_grid_match_min_cells=7,
        )
    )
    service._model_loaded = True
    service._load_error = None
    service._model = object()
    service._analyze_payload = lambda _payload: {"frame": object(), "reason": None, "timings": InferenceTimings()}  # type: ignore[method-assign]
    return service


def make_frame(center: str = "white", ok: bool = True) -> FrameDetectionResult:
    grid = [
        [center, center, center],
        [center, center, center],
        [center, center, center],
    ]
    confidences = [
        [0.9, 0.9, 0.9],
        [0.9, 0.95, 0.9],
        [0.9, 0.9, 0.9],
    ]
    boxes = [(10 * index, 10, 10 * index + 8, 18) for index in range(9)]
    return FrameDetectionResult(
        ok=ok,
        grid=grid if ok else None,
        confidence_matrix=confidences if ok else None,
        avg_confidence=0.91 if ok else 0.0,
        detected_stickers=9 if ok else 6,
        frame_index=0,
        reason=None if ok else "Need 9 stickers visible in a stable 3x3 layout.",
        ordered_boxes=boxes if ok else [],
    )


class ScannerServiceTests(unittest.TestCase):
    def test_accepts_after_two_consistent_observations(self) -> None:
        service = make_service()
        session = service.start_scanner_test_session({})
        frames = [make_frame("white"), make_frame("white")]

        def fake_infer(_frame, _index):
            return frames.pop(0), InferenceTimings(total_ms=120.0, inference_ms=90.0)

        service._infer_frame_detailed = fake_infer  # type: ignore[method-assign]

        obs1 = service.observe_scanner_test_frame(session.session_id, {"imageBase64": "x"})
        obs2 = service.observe_scanner_test_frame(session.session_id, {"imageBase64": "x"})

        self.assertEqual(obs1.scanner_state, "SCANNING")
        self.assertEqual(obs2.scanner_state, "ACCEPTED")
        current = service.get_scanner_test_session(session.session_id)
        self.assertEqual(current.captured_face_count, 1)

    def test_does_not_accept_from_single_observation(self) -> None:
        service = make_service()
        session = service.start_scanner_test_session({})
        service._infer_frame_detailed = lambda _frame, _index: (make_frame("white"), InferenceTimings())  # type: ignore[method-assign]

        observation = service.observe_scanner_test_frame(session.session_id, {"imageBase64": "x"})
        self.assertNotEqual(observation.scanner_state, "ACCEPTED")
        current = service.get_scanner_test_session(session.session_id)
        self.assertEqual(current.captured_face_count, 0)

    def test_times_out_after_five_seconds(self) -> None:
        service = make_service()
        session = service.start_scanner_test_session({})
        service._infer_frame_detailed = lambda _frame, _index: (make_frame("white"), InferenceTimings())  # type: ignore[method-assign]

        stored = service._require_scanner_session(session.session_id)
        stored.burst.started_at = stored.burst.started_at or 0.0
        stored.burst.started_at = stored.burst.started_at - 10.0

        observation = service.observe_scanner_test_frame(session.session_id, {"imageBase64": "x"})
        self.assertEqual(observation.scanner_state, "RETRY")

    def test_rejects_duplicate_center_color(self) -> None:
        service = make_service()
        session = service.start_scanner_test_session({})
        frames = [make_frame("white"), make_frame("white"), make_frame("white"), make_frame("white")]

        def fake_infer_first(_frame, _index):
            return frames.pop(0), InferenceTimings()

        service._infer_frame_detailed = fake_infer_first  # type: ignore[method-assign]
        service.observe_scanner_test_frame(session.session_id, {"imageBase64": "x"})
        service.observe_scanner_test_frame(session.session_id, {"imageBase64": "x"})

        service.observe_scanner_test_frame(session.session_id, {"imageBase64": "x"})
        duplicate = service.observe_scanner_test_frame(session.session_id, {"imageBase64": "x"})
        self.assertEqual(duplicate.scanner_state, "DUPLICATE_FACE")

    def test_returns_ai_busy_when_inference_gate_is_locked(self) -> None:
        service = make_service()
        session = service.start_scanner_test_session({})
        service._inference_gate.acquire()
        try:
            observation = service.observe_scanner_test_frame(session.session_id, {"imageBase64": "x"})
            self.assertEqual(observation.scanner_state, "AI_BUSY")
        finally:
            service._inference_gate.release()


if __name__ == "__main__":
    unittest.main()
