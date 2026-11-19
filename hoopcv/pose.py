"""从视频提取人体关键点（MediaPipe Pose Landmarker）。

输出 PoseSeries：
- img:   (T, 33, 3)  图像像素坐标 x, y 以及可见度 visibility
- world: (T, 33, 3)  以髋中心为原点的 3D 坐标（米），x 向画面右、y 向下、z 远离相机
未检测到人的帧为 NaN。
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


@dataclass
class PoseSeries:
    fps: float
    width: int
    height: int
    img: np.ndarray
    world: np.ndarray

    @property
    def n_frames(self) -> int:
        return self.img.shape[0]

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.n_frames) / self.fps

    def save(self, path: str | os.PathLike) -> None:
        np.savez_compressed(path, fps=self.fps, width=self.width, height=self.height,
                            img=self.img, world=self.world)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "PoseSeries":
        d = np.load(path)
        return cls(float(d["fps"]), int(d["width"]), int(d["height"]), d["img"], d["world"])


def model_path(variant: str = "heavy") -> Path:
    """返回模型文件路径，没有就下载到 ~/.cache/hoopcv。"""
    cache = Path(os.environ.get("HOOPCV_CACHE", Path.home() / ".cache" / "hoopcv"))
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f"pose_landmarker_{variant}.task"
    if not p.exists():
        print(f"下载 MediaPipe 模型 {variant} → {p}")
        urllib.request.urlretrieve(MODEL_URLS[variant], p)
    return p


def extract(video: str, variant: str = "heavy", max_frames: int | None = None,
            rotate: int = 0) -> PoseSeries:
    """逐帧跑姿态估计。rotate 可为 0/90/180/270，用于手机竖拍视频方向不对的情况。"""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"打不开视频：{video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path(variant))),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rotate)
    imgs, worlds = [], []
    w = h = 0
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and i >= max_frames):
                break
            if rot is not None:
                frame = cv2.rotate(frame, rot)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                      int(round(i * 1000.0 / fps)))
            im = np.full((33, 3), np.nan)
            wd = np.full((33, 3), np.nan)
            if res.pose_landmarks:
                for k, p in enumerate(res.pose_landmarks[0]):
                    im[k] = (p.x * w, p.y * h, p.visibility if p.visibility is not None else 1.0)
                for k, p in enumerate(res.pose_world_landmarks[0]):
                    wd[k] = (p.x, p.y, p.z)
            imgs.append(im)
            worlds.append(wd)
            i += 1
    cap.release()
    if not imgs:
        raise RuntimeError("视频里没有读到任何帧")
    return PoseSeries(float(fps), int(w), int(h), np.stack(imgs), np.stack(worlds))
