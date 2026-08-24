from __future__ import annotations

from collections import Counter

from .models import GridMatrix


VALID_COLORS = ["white", "yellow", "red", "orange", "blue", "green"]

DISPLAY_BGR = {
    "white": (245, 245, 245),
    "yellow": (0, 215, 255),
    "red": (60, 60, 220),
    "orange": (0, 140, 255),
    "blue": (220, 120, 30),
    "green": (60, 180, 60),
    "unknown": (120, 120, 120),
}


def flatten_grid(grid: GridMatrix) -> list[str]:
    return [cell for row in grid for cell in row]


def grid_similarity(grid_a: GridMatrix, grid_b: GridMatrix) -> float:
    flat_a = flatten_grid(grid_a)
    flat_b = flatten_grid(grid_b)
    if len(flat_a) != 9 or len(flat_b) != 9:
        return 0.0
    same = sum(1 for a, b in zip(flat_a, flat_b) if a == b)
    return same / 9.0


def color_count_summary(faces: dict[str, GridMatrix]) -> dict[str, int]:
    counter = Counter()
    for grid in faces.values():
        counter.update(flatten_grid(grid))
    return {color: int(counter.get(color, 0)) for color in VALID_COLORS}


def validate_color_counts(faces: dict[str, GridMatrix]) -> tuple[bool, dict[str, str | int]]:
    counts = color_count_summary(faces)
    total = sum(counts.values())
    report: dict[str, str | int] = {"totalStickers": total}

    valid = total == 54
    for color in VALID_COLORS:
        report[color] = counts[color]
        if counts[color] != 9:
            valid = False

    report["status"] = "VALID_COLOR_COUNT" if valid else "INVALID_COLOR_COUNT"
    return valid, report
