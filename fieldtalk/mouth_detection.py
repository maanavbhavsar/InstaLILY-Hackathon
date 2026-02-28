"""
Mouth detection: MediaPipe Face Mesh (primary) for reliable real-time mouth region,
with Haar cascade fallback when MediaPipe is unavailable or no face detected.
"""
import cv2
import numpy as np
from pathlib import Path
from collections import deque

# MediaPipe mouth landmark indices (Face Mesh 468); outer/inner lip and corners
_MOUTH_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

# Haar fallback
_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"


def _mouth_bbox_from_landmarks(landmarks, frame_shape: tuple, padding: float = 0.15) -> tuple[int, int, int, int] | None:
    """Compute mouth bounding box from MediaPipe face mesh landmarks. Returns (x, y, w, h) or None."""
    h_img, w_img = frame_shape[:2]
    xs = []
    ys = []
    for i in _MOUTH_INDICES:
        if i < len(landmarks):
            lm = landmarks[i]
            xs.append(lm.x * w_img)
            ys.append(lm.y * h_img)
    if not xs or not ys:
        return None
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    pad_w = w * padding
    pad_h = h * padding
    x = max(0, int(x_min - pad_w))
    y = max(0, int(y_min - pad_h))
    w = min(w_img - x, int(w + 2 * pad_w))
    h = min(h_img - y, int(h + 2 * pad_h))
    if w < 10 or h < 10:
        return None
    return (x, y, w, h)


class MouthDetector:
    """Detect face and mouth region: MediaPipe Face Mesh first, Haar cascade fallback."""

    def __init__(self, target_fps: int = 25, use_mediapipe: bool = True, face_scale_factor: float = 1.1, min_neighbors: int = 5):
        self.target_fps = target_fps
        self.use_mediapipe = use_mediapipe
        self._mp_face_mesh = None
        self._face_cascade = cv2.CascadeClassifier(str(_CASCADE_PATH))
        self.face_scale_factor = face_scale_factor
        self.min_neighbors = min_neighbors
        self._last_face_bbox = None
        self._last_mouth_bbox = None

    def _get_mediapipe(self):
        if self._mp_face_mesh is None and self.use_mediapipe:
            try:
                import mediapipe as mp
                self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception:
                self.use_mediapipe = False
        return self._mp_face_mesh

    def _detect_face_haar(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=self.face_scale_factor, minNeighbors=self.min_neighbors, minSize=(30, 30)
        )
        if not len(faces):
            return self._last_face_bbox
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        self._last_face_bbox = (x, y, w, h)
        return self._last_face_bbox

    def _mouth_bbox_from_face(self, face_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, w, h = face_bbox
        mh = max(h // 3, 20)
        my = y + h - mh
        mx = max(0, x - w // 8)
        mw = min(w + (w // 4), w * 2)
        return (mx, my, mw, mh)

    def _detect_mouth_mediapipe(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        mp_face = self._get_mediapipe()
        if mp_face is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)
        if not results.multi_face_landmarks:
            return self._last_mouth_bbox
        landmarks = results.multi_face_landmarks[0].landmark
        bbox = _mouth_bbox_from_landmarks(landmarks, frame.shape)
        if bbox:
            self._last_mouth_bbox = bbox
        return self._last_mouth_bbox

    def _detect_face_and_mouth(self, frame: np.ndarray) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
        """Returns (face_bbox, mouth_bbox). Prefer MediaPipe for mouth."""
        mouth_bbox = None
        if self.use_mediapipe:
            mouth_bbox = self._detect_mouth_mediapipe(frame)
        face_bbox = self._detect_face_haar(frame)
        if mouth_bbox is None and face_bbox is not None:
            mouth_bbox = self._mouth_bbox_from_face(face_bbox)
        if face_bbox is None and mouth_bbox is not None:
            face_bbox = self._last_face_bbox
        return face_bbox, mouth_bbox

    def crop_mouth(self, frame: np.ndarray, mouth_bbox: tuple[int, int, int, int] | None):
        if mouth_bbox is None:
            return None
        mx, my, mw, mh = mouth_bbox
        h_img, w_img = frame.shape[:2]
        x1 = max(0, mx)
        y1 = max(0, my)
        x2 = min(w_img, mx + mw)
        y2 = min(h_img, my + mh)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    def draw_mouth_region(self, frame: np.ndarray, mouth_bbox: tuple[int, int, int, int] | None, color=(0, 255, 0), thickness=2):
        if mouth_bbox is None:
            return
        mx, my, mw, mh = mouth_bbox
        cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), color, thickness)

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray | None, np.ndarray, tuple[int, int, int, int] | None]:
        """
        Detect face and mouth (MediaPipe preferred), crop mouth, draw on frame.
        Returns (mouth_crop or None, frame_with_drawing, face_bbox or None).
        """
        face_bbox, mouth_bbox = self._detect_face_and_mouth(frame)
        mouth_crop = self.crop_mouth(frame, mouth_bbox) if mouth_bbox else None
        display = frame.copy()
        self.draw_mouth_region(display, mouth_bbox)
        return mouth_crop, display, face_bbox


def build_frame_buffer(mouth_frames: list[np.ndarray], size: int = 64) -> np.ndarray | None:
    """Build a single image from up to 16 mouth frames as a 4x4 grid (BGR)."""
    if not mouth_frames or len(mouth_frames) < 1:
        return None
    n = min(len(mouth_frames), size)
    frames = mouth_frames[-n:]
    cell_size = 64
    grid_cols = 4
    target = min(16, n)
    use = frames[-target:] if len(frames) >= target else frames
    rows = (len(use) + grid_cols - 1) // grid_cols
    cols = min(len(use), grid_cols)
    w = cols * cell_size
    h = rows * cell_size
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:] = 128
    for i, f in enumerate(use):
        r, c = i // grid_cols, i % grid_cols
        resized = cv2.resize(f, (cell_size, cell_size), interpolation=cv2.INTER_LINEAR)
        out[r * cell_size : (r + 1) * cell_size, c * cell_size : (c + 1) * cell_size] = resized
    return out
