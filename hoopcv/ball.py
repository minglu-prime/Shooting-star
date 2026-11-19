"""篮球检测（可选，需要 `pip install ultralytics`）。

默认用 COCO 预训练的 YOLO11n 的 "sports ball" 类别；如果你有篮球专用权重（如 Roboflow 上训练的），
用 --ball-weights 指定，并用 --ball-class 指定类别名。
"""
from __future__ import annotations

import cv2
import numpy as np


def detect_ball(video: str, weights: str = "yolo11n.pt", class_name: str = "sports ball",
                conf: float = 0.15, rotate: int = 0, max_frames: int | None = None,
                hand_xy: np.ndarray | None = None) -> np.ndarray:
    """返回 (T, 3)：每帧球心 x, y（像素）与置信度，没检测到为 NaN。

    同一帧多个候选时，优先选离上一帧球位置近的；若提供 hand_xy（T,2），第一帧优先选离手近的。
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:  # pragma: no cover
        raise SystemExit("球检测需要 ultralytics：pip install 'hoopcv[ball]'") from e

    model = YOLO(weights)
    names = model.names
    cls_ids = [k for k, v in names.items() if v == class_name]
    if not cls_ids:
        raise SystemExit(f"权重里没有类别 {class_name!r}，可选：{sorted(names.values())}")
    rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rotate)
    cap = cv2.VideoCapture(video)
    out = []
    prev = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        if rot is not None:
            frame = cv2.rotate(frame, rot)
        res = model.predict(frame, classes=cls_ids, conf=conf, verbose=False)[0]
        best = (np.nan, np.nan, np.nan)
        if len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            cf = res.boxes.conf.cpu().numpy()
            c = np.c_[(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2]
            ref = prev
            if ref is None and hand_xy is not None and i < len(hand_xy) and np.isfinite(hand_xy[i]).all():
                ref = hand_xy[i]
            if ref is not None:
                d = np.linalg.norm(c - ref, axis=1)
                score = cf - d / (frame.shape[0] * 0.5)
                k = int(np.argmax(score))
            else:
                k = int(np.argmax(cf))
            best = (c[k, 0], c[k, 1], cf[k])
            prev = c[k]
        out.append(best)
        i += 1
    cap.release()
    return np.array(out, dtype=float)
