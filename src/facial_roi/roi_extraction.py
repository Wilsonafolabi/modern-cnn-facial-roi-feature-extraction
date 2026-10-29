"""Face detection and landmark-based ROI cropping."""
from __future__ import annotations

import bz2
import urllib.request
from pathlib import Path

import cv2
import numpy as np

PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"


def load_image(path: str | Path):
    """Read an image with OpenCV and return RGB pixels."""
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _ensure_predictor(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    archive = path.with_suffix(path.suffix + ".bz2")
    urllib.request.urlretrieve(PREDICTOR_URL, archive)
    with bz2.BZ2File(archive, "rb") as src, path.open("wb") as dst:
        dst.write(src.read())
    archive.unlink(missing_ok=True)
    return path


def extract_rois(image_rgb: np.ndarray, predictor_path: str | Path = "models/shape_predictor_68_face_landmarks.dat"):
    """Return the five ROI crops and their boxes using dlib 68-point landmarks.

    Left/right labels follow dlib's image-coordinate landmark groups used in the notebook.
    """
    try:
        import dlib
    except ImportError as exc:
        raise RuntimeError("Install dlib to extract landmark-based ROIs.") from exc

    predictor_path = _ensure_predictor(predictor_path)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    detections = dlib.get_frontal_face_detector()(gray, 1)
    if not detections:
        raise RuntimeError("No face detected by dlib. Try a clear, front-facing image.")
    rect = max(detections, key=lambda r: r.width() * r.height())
    shape = dlib.shape_predictor(str(predictor_path))(gray, rect)
    points = np.asarray([[p.x, p.y] for p in shape.parts()], dtype=int)

    def crop(points_subset, padding):
        h, w = image_rgb.shape[:2]
        lo = points_subset.min(axis=0).astype(float)
        hi = points_subset.max(axis=0).astype(float)
        span = hi - lo
        lo -= span * padding
        hi += span * padding
        x1, y1 = np.maximum(np.floor(lo).astype(int), [0, 0])
        x2, y2 = np.minimum(np.ceil(hi).astype(int) + 1, [w, h])
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid ROI crop produced by landmark coordinates.")
        return image_rgb[y1:y2, x1:x2].copy(), (int(x1), int(y1), int(x2-x1), int(y2-y1))

    x1, y1 = max(0, rect.left()), max(0, rect.top())
    x2, y2 = min(image_rgb.shape[1], rect.right()), min(image_rgb.shape[0], rect.bottom())
    regions = {"Full Face": image_rgb[y1:y2, x1:x2].copy()}
    boxes = {"Full Face": (x1, y1, x2-x1, y2-y1)}
    specs = {
        "Left Eye": (points[36:42], 0.35),
        "Right Eye": (points[42:48], 0.35),
        "Nose": (points[27:36], 0.15),
        "Mouth": (points[48:68], 0.25),
    }
    for name, (subset, padding) in specs.items():
        regions[name], boxes[name] = crop(subset, padding)
    return regions, boxes
