from __future__ import annotations

import cv2

from .color_utils import DISPLAY_BGR
from .models import CubeScanOutput, FaceScanResult, FrameDetectionResult, ScanState


class CubeNetRenderer:
    def __init__(self, face_order: list[str]) -> None:
        self.face_order = face_order
        self.face_positions = {
            "U": (1, 0),
            "L": (0, 1),
            "F": (1, 1),
            "R": (2, 1),
            "B": (3, 1),
            "D": (1, 2),
        }

    def draw(
        self,
        frame,
        state: ScanState,
        current_face: str,
        faces: dict[str, FaceScanResult],
        frame_result: FrameDetectionResult | None,
        fps: float,
        infer_ms: float,
        status_text: str,
        countdown_text: str,
    ):
        canvas = self._create_canvas(frame)
        camera_region = canvas[0:frame.shape[0], 0:frame.shape[1]]
        camera_region[:] = frame

        self._draw_camera_header(camera_region, fps, infer_ms, status_text, countdown_text, state, current_face)
        if frame_result and frame_result.grid:
            self._draw_grid_preview(camera_region, frame_result)

        self._draw_cube_net(canvas, frame.shape[1] + 20, 30, faces, current_face)
        self._draw_help(canvas, frame.shape[1] + 20, frame.shape[0] - 140)
        return canvas

    def _create_canvas(self, frame):
        height, width = frame.shape[:2]
        canvas = cv2.copyMakeBorder(
            frame,
            0,
            0,
            0,
            520,
            cv2.BORDER_CONSTANT,
            value=(26, 28, 32),
        )
        return canvas

    def _draw_camera_header(self, frame, fps, infer_ms, status_text, countdown_text, state, current_face):
        cv2.putText(frame, f"FPS: {fps:.1f}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"INF: {infer_ms:.0f}ms", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(frame, f"STATE: {state.value}", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FACE: {current_face}", (12, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(frame, status_text, (12, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 220, 255), 2, cv2.LINE_AA)
        if countdown_text:
            cv2.putText(frame, countdown_text, (12, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 220, 120), 2, cv2.LINE_AA)

    def _draw_grid_preview(self, frame, frame_result: FrameDetectionResult):
        for box, color in zip(frame_result.ordered_boxes, [cell for row in frame_result.grid for cell in row]):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            fill = DISPLAY_BGR.get(color, DISPLAY_BGR["unknown"])
            cv2.rectangle(frame, (x1, y1), (x1 + 18, y1 + 18), fill, -1)

    def _draw_cube_net(self, canvas, origin_x: int, origin_y: int, faces: dict[str, FaceScanResult], current_face: str):
        cell = 26
        pad = 4
        face_size = cell * 3 + pad * 4

        for face_name, (col_block, row_block) in self.face_positions.items():
            face_x = origin_x + col_block * (face_size + 14)
            face_y = origin_y + row_block * (face_size + 14)

            border_color = (60, 210, 255) if face_name == current_face else (90, 90, 90)
            cv2.rectangle(canvas, (face_x - 4, face_y - 24), (face_x + face_size, face_y + face_size), border_color, 2)
            cv2.putText(canvas, face_name, (face_x, face_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (250, 250, 250), 2, cv2.LINE_AA)

            result = faces.get(face_name)
            for row in range(3):
                for col in range(3):
                    x1 = face_x + pad + col * (cell + pad)
                    y1 = face_y + pad + row * (cell + pad)
                    x2 = x1 + cell
                    y2 = y1 + cell
                    if result and result.grid:
                        color_name = result.grid[row][col]
                    else:
                        color_name = "unknown"
                    fill = DISPLAY_BGR.get(color_name, DISPLAY_BGR["unknown"])
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 30, 30), 1)

    def _draw_help(self, canvas, origin_x: int, origin_y: int):
        lines = [
            "SPACE: start scan / next face",
            "R: retry current face",
            "B: back to previous face",
            "C: clear all faces",
            "Q or ESC: quit",
        ]
        for idx, text in enumerate(lines):
            cv2.putText(canvas, text, (origin_x, origin_y + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
