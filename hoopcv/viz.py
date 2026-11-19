"""可视化：带骨架和角度的标注视频、角度曲线图、突破轨迹图。

图和视频里的文字用英文，避免系统缺中文字体时显示成方块。
"""
from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import landmarks as L  # noqa: E402
from .pose import PoseSeries  # noqa: E402

LABELS = [
    ("elbow_flex_2d", "elbow"), ("knee_flex_2d", "knee"), ("hip_flex_2d", "hip"),
    ("shoulder_flex_2d", "shoulder"), ("trunk_lean_2d", "trunk lean"),
    ("shoulder_ext_rot_3d", "shoulder ER(3D)"),
]


def annotate_video(video: str, out_path: str, ps: PoseSeries, ang: pd.DataFrame,
                   events: dict[int, str] | None = None, ball: np.ndarray | None = None,
                   rotate: int = 0, side: str = "right") -> None:
    cap = cv2.VideoCapture(video)
    rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rotate)
    fs = max(0.4, ps.height / 900)            # 字号随画面大小缩放
    line_h = int(32 * fs) + 4
    panel_w = int(300 * fs) + 20               # 角度面板放在画面右侧，不遮挡人
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), ps.fps, (ps.width + panel_w, ps.height))
    s = L.side_ids(side)
    hi = {s["shoulder"], s["elbow"], s["wrist"], s["hip"], s["knee"], s["ankle"]}
    events = events or {}
    banner, banner_left = "", 0
    i = 0
    while i < ps.n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if rot is not None:
            frame = cv2.rotate(frame, rot)
        pts = ps.img[i]
        for a, b in L.SKELETON:
            if np.isfinite(pts[[a, b], :2]).all():
                col = (0, 200, 255) if (a in hi and b in hi) else (200, 200, 200)
                cv2.line(frame, tuple(int(v) for v in pts[a, :2]), tuple(int(v) for v in pts[b, :2]), col, 2)
        for k in range(33):
            if np.isfinite(pts[k, :2]).all():
                cv2.circle(frame, (int(pts[k, 0]), int(pts[k, 1])), 3, (0, 255, 0), -1)
        if ball is not None and i < len(ball) and np.isfinite(ball[i, :2]).all():
            cv2.circle(frame, (int(ball[i, 0]), int(ball[i, 1])), 12, (0, 128, 255), 2)
        panel = np.zeros((ps.height, panel_w, 3), np.uint8)
        y = line_h
        cv2.putText(panel, f"t = {i / ps.fps:5.2f} s", (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (180, 180, 180), 1, cv2.LINE_AA)
        y += line_h
        for col, name in LABELS:
            v = ang[col].iloc[i]
            txt = f"{name}: {v:6.1f}" if np.isfinite(v) else f"{name}:   --"
            cv2.putText(panel, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
            y += line_h
        if i in events:
            banner, banner_left = events[i], int(0.6 * ps.fps)
        if banner_left > 0:
            cv2.putText(panel, banner, (10, y + line_h), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.3,
                        (0, 0, 255), 2, cv2.LINE_AA)
            banner_left -= 1
        writer.write(np.hstack([frame, panel]))
        i += 1
    cap.release()
    writer.release()


def plot_angles(ang: pd.DataFrame, out_path: str, event_lines: list[tuple[float, str]] | None = None) -> None:
    groups = [
        ("Joint flexion (2D, deg)", ["elbow_flex_2d", "knee_flex_2d", "hip_flex_2d", "shoulder_flex_2d", "wrist_flex_2d"]),
        ("Trunk lean (deg)", ["trunk_lean_2d", "trunk_lean_fwd_3d", "trunk_lean_side_3d"]),
        ("Rotation (3D, approx., deg)", ["shoulder_ext_rot_3d", "forearm_pron_3d", "hip_ext_rot_3d"]),
    ]
    fig, axes = plt.subplots(len(groups), 1, figsize=(11, 9), sharex=True)
    for ax, (title, cols) in zip(axes, groups):
        for c in cols:
            if c in ang:
                ax.plot(ang["t"], ang[c], label=c, lw=1.4)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=3, loc="upper right")
        for t, name in (event_lines or []):
            ax.axvline(t, color="k", lw=0.8, ls="--", alpha=0.6)
    for t, name in (event_lines or []):
        axes[0].text(t, axes[0].get_ylim()[1], name, fontsize=7, rotation=90, va="top", ha="right")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_track(track: pd.DataFrame, summary: dict, cuts: pd.DataFrame, out_path: str) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    sc = a1.scatter(track["x"], track["y"], c=track["t"], s=6, cmap="viridis")
    fig.colorbar(sc, ax=a1, label="time (s)")
    if summary.get("detected"):
        i = int(np.argmin(np.abs(track["t"] - summary["t_start"])))
        a1.plot(track["x"].iloc[i], track["y"].iloc[i], "r^", ms=10, label="drive start")
    for _, c in cuts.iterrows():
        j = int(np.argmin(np.abs(track["t"] - c["t"])))
        a1.plot(track["x"].iloc[j], track["y"].iloc[j], "ks", ms=7)
    a1.set_aspect("equal", adjustable="datalim")
    a1.set_title(f"Feet track ({summary.get('unit')})", fontsize=10)
    a1.legend(fontsize=8)
    a2.plot(track["t"], track["speed"])
    a2.set_xlabel("time (s)")
    a2.set_ylabel(f"speed ({summary.get('unit')}/s)")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
