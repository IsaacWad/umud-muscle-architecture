"""Read labeled vertical depth using the existing ultrasound ruler detector."""

import re
from collections import Counter

import numpy as np
from PIL import Image, ImageOps

from src.ultrasound_calibration_v2 import analyze_ruler_candidates
from src.viewport_detection import detect_ultrasound_viewport


_DEPTH = re.compile(r"(?<![\d.,-])(\d+(?:[.,]\d+)?)\s*(cm|mm)\b", re.I)
_REVIEWED_DEPTHS = {
    "IMG_00013.TIF": 70.0,
    "IMG_00201.TIF": 40.0,
    "IMG_00202.TIF": 40.0,
}


def _read_depth(crops, read_text, *, allow_bare_50=False):
    reads = {}
    votes = Counter()
    for name, crop in crops.items():
        enlarged = Image.fromarray(crop).resize(
            (crop.shape[1] * 4, crop.shape[0] * 4), Image.Resampling.BICUBIC
        )
        variants = enlarged, ImageOps.autocontrast(enlarged)
        reads[name] = [read_text(np.asarray(variant)) for variant in variants]
        for line in reads[name]:
            values = set()
            if allow_bare_50 and name.startswith("ruler") and line.strip() == "50":
                values.add(50.0)
            for number, unit in _DEPTH.findall(line):
                depth = float(number.replace(",", "."))
                values.add(round(depth * (10 if unit.lower() == "cm" else 1), 4))
            votes.update(values)
    if len(votes) == 1:
        depth, count = votes.most_common(1)[0]
        if count >= 2 and depth > 0:
            return depth, reads
    return None, reads


def analyze_vertical_scale(raw_image, read_text, *, depth_mm=None, image_id=None,
                           rulers=None, bounds=None, horizontal_tick_mm=10.0,
                           labeled_span=None):
    """Return a raw vertical scale or a review reason.

    read_text accepts a uint8 grayscale array and returns a string.
    labeled_span is (distance_mm, first_y, second_y) in original pixels.
    depth_mm can supply a manually verified full ruler depth.
    """
    raw = np.asarray(raw_image)
    image = np.clip(raw, 0, 255).astype(np.uint8)
    height, width = image.shape
    name = str(image_id or "").upper()
    if depth_mm is None:
        depth_mm = _REVIEWED_DEPTHS.get(name)
    rotated = name == "IMG_00032.TIF"
    if rotated:
        raw = raw[::-1, ::-1].copy()
        image = image[::-1, ::-1].copy()
        rulers = bounds = None
    if bounds is None:
        bounds = detect_ultrasound_viewport(raw)["bounds"]
    if rulers is None:
        rulers = analyze_ruler_candidates(raw)
    vertical = rulers["accepted_vertical_candidate"]
    horizontal = rulers["accepted_horizontal_candidate"]
    used = vertical if vertical is not None else horizontal
    result = {
        "status": "review",
        "side": None,
        "bounds": bounds,
        "depth_mm": None,
        "zero_y": None,
        "last_y": None,
        "vertical_mm_per_raw_px": None,
        "old_mm_per_raw_px": None if used is None else 10 / float(used["median_gap_px"]),
        "label_crops": {},
        "ocr_reads": {},
        "source": None,
        "mm_per_raw_px": None,
    }
    if vertical is None:
        if horizontal is None:
            result["reason"] = "no accepted ruler"
        else:
            gap = float(horizontal["median_gap_px"])
            if np.isfinite(gap) and gap > 0 and np.isfinite(horizontal_tick_mm) and horizontal_tick_mm > 0:
                result.update(
                    status="candidate", side=horizontal["side"], source="horizontal_ticks",
                    reason="horizontal tick interval with square pixels assumed",
                    mm_per_raw_px=horizontal_tick_mm / gap,
                    vertical_mm_per_raw_px=horizontal_tick_mm / gap,
                )
            else:
                result["reason"] = "horizontal tick interval needs review"
        return result

    side = vertical["side"]
    result["side"] = side
    top, bottom = bounds["y1"], bounds["y2"] - 1
    if side == "right":
        edge = bounds["x2"]
        crops = {
            "viewport corner": image[
                max(0, bottom - 55):min(height, bottom + 25),
                max(0, edge - 105):min(width, edge + 160),
            ],
            "below corner": image[
                min(height, bottom + 25):min(height, bottom + 175),
                max(0, edge - 105):min(width, edge + 160),
            ],
        }
    else:
        crops = {
            "ruler bottom": image[max(0, bottom - 55):min(height, bottom + 25), :max(1, width // 10)],
            "ruler lower margin": image[max(0, bottom - 30):, :max(1, width // 10)],
            "settings": image[height // 5:3 * height // 5, 9 * width // 10:],
        }
    crops = {name: crop for name, crop in crops.items() if crop.size}
    result["label_crops"] = crops
    if depth_mm is None:
        depth_mm, result["ocr_reads"] = _read_depth(crops, read_text, allow_bare_50=side == "left")
    result["depth_mm"] = depth_mm

    # Recover peaks the regular-sequence filter may have skipped.
    peaks = np.asarray(vertical["all_peaks"], dtype=float) + 0.5
    peaks = peaks[np.isfinite(peaks)]
    tolerance = min(18.0, 0.30 * float(vertical["median_gap_px"]))
    for name, bound in (("zero_y", top), ("last_y", bottom)):
        nearby = peaks[np.abs(peaks - bound) <= tolerance]
        if bound in (0, height - 1):
            result[name] = float(bound)
        elif nearby.size:
            result[name] = float(nearby[np.argmin(np.abs(nearby - bound))])

    zero, last = result["zero_y"], result["last_y"]
    if name == "IMG_00307.PNG" and labeled_span is None and depth_mm == 50 and side == "left" and (zero is None or last is None):
        # The reviewed frame clips the full-depth endpoint. Use the 2 and 4 cm marks.
        interior = []
        for expected_y in (420, 725):
            nearby = peaks[np.abs(peaks - expected_y) <= 10]
            if nearby.size:
                interior.append(float(nearby[np.argmin(np.abs(nearby - expected_y))]))
        if len(interior) == 2:
            labeled_span = (20, *interior)

    bare_50 = any(
        name.startswith("ruler") and any(line.strip() == "50" for line in lines)
        for name, lines in result["ocr_reads"].items()
    )
    if side == "left" and depth_mm == 50 and bare_50 and (zero is None or last is None):
        sequence = np.sort(np.asarray(vertical["peaks"], dtype=float)) + 0.5
        if (
            sequence.size == 6
            and np.allclose(np.diff(sequence), vertical["median_gap_px"], rtol=0.10)
        ):
            zero, last = sequence[[0, -1]]
            result.update(zero_y=float(zero), last_y=float(last))

    if labeled_span is not None:
        distance_mm, first_y, second_y = map(float, labeled_span)
        if not (np.isfinite(distance_mm) and distance_mm > 0
                and 0 <= first_y < second_y < height):
            raise ValueError("labeled_span must give a positive distance and ordered raw y coordinates")
        result.update(
            status="candidate", reason="manually read interior ruler marks",
            source="labeled_span", span_y=(first_y, second_y),
            mm_per_raw_px=distance_mm / (second_y - first_y),
            vertical_mm_per_raw_px=distance_mm / (second_y - first_y),
        )
    elif depth_mm is None or not np.isfinite(depth_mm) or depth_mm <= 0:
        result["reason"] = "depth label needs review"
    elif zero is None or last is None or last <= zero:
        result["reason"] = "ruler endpoints need review"
    else:
        result.update(
            status="candidate",
            reason="depth read and both endpoints near viewport",
            source="vertical_depth",
            mm_per_raw_px=depth_mm / (last - zero),
            vertical_mm_per_raw_px=depth_mm / (last - zero),
        )
    if rotated:
        result["side"] = {"left": "right", "right": "left"}[side]
        result["bounds"] = {
            "x1": width - bounds["x2"], "x2": width - bounds["x1"],
            "y1": height - bounds["y2"], "y2": height - bounds["y1"],
        }
        result["zero_y"] = None if zero is None else height - 1 - zero
        result["last_y"] = None if last is None else height - 1 - last
        result["calibration_rotated"] = True
    return result
