"""突破分析：启动时刻、第一步方向角、交叉步/顺步、重心高度、躯干倾角、变向切角。

坐标：
- 提供球场标定（4 个画面点 ↔ 4 个球场坐标，单位米）时，脚的位置投到球场平面，距离单位为米；
- 没有标定时用画面像素并除以身高像素，单位为“身高”，方向角只在机位接近俯视或正后方时可信。

推荐拍摄：持球人正后方或高处斜后方固定机位，60 fps 以上，画面内能看到 4 个已知球场点（如罚球线两端、底线与禁区交点）。
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from . import geometry as g
from . import landmarks as L
from .pose import PoseSeries
from .shot import stature_px


def parse_points(s: str | None) -> np.ndarray | None:
    if not s:
        return None
    pts = [tuple(float(v) for v in p.split(",")) for p in s.split(";") if p.strip()]
    return np.array(pts, dtype=float)


class Ground:
    """画面像素 → 地面坐标。"""

    def __init__(self, court_px=None, court_m=None, stature=1.0):
        self.H = None
        self.stature = stature
        if court_px is not None and court_m is not None:
            if len(court_px) < 4 or len(court_px) != len(court_m):
                raise ValueError("球场标定需要至少 4 对点")
            self.H, _ = cv2.findHomography(court_px.astype(np.float32), court_m.astype(np.float32))
        self.unit = "m" if self.H is not None else "stature"

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        if self.H is None:
            out = xy / self.stature
            out[..., 1] *= -1  # 画面 y 向下 → 转为向上为正（远离镜头）
            return out
        p = np.c_[xy.reshape(-1, 2), np.ones(xy.reshape(-1, 2).shape[0])] @ self.H.T
        return (p[:, :2] / p[:, 2:3]).reshape(xy.shape)


def _speed(p, fps):
    v = np.gradient(p, axis=0) * fps
    return v, np.linalg.norm(v, axis=1)


def analyze_drive(ps: PoseSeries, ang: pd.DataFrame, court_px=None, court_m=None,
                  basket=None, start_speed=None, cut_threshold_deg: float = 30.0,
                  feet_m=None, ref_name_m: str = "球场 +y 方向"):
    """返回 (summary dict, cuts DataFrame, track DataFrame)。

    basket：篮筐在地面坐标中的位置（有标定或双机 3D 时为米；单机没标定时为画面像素）。
    feet_m：双机 3D 给出的左右脚踝地面坐标 (lpos, rpos)，各 (T,2) 米；给了就不再用画面像素。
    """
    fps = ps.fps
    im = ps.img
    stature = stature_px(ps)
    ground = Ground(court_px, court_m, stature)
    if feet_m is not None:
        lpos, rpos = (np.asarray(f, float) for f in feet_m)
        ground.unit = "m"
        pos = (lpos + rpos) / 2
    else:
        la, ra = im[:, L.L_ANKLE, :2], im[:, L.R_ANKLE, :2]
        pos = ground((la + ra) / 2)
        lpos, rpos = ground(la), ground(ra)
    pos = g.lowpass(g.fill_short_gaps(pos, int(0.15 * fps)), fps, 4.0)
    v, spd = _speed(pos, fps)

    if start_speed is None:
        start_speed = 1.2 if ground.unit == "m" else 0.6
    hold = max(2, int(0.2 * fps))
    moving = np.nan_to_num(spd > start_speed, nan=False).astype(bool)
    start = None
    for i in range(len(moving) - hold):
        if moving[i:i + hold].all():
            start = i
            break
    track = pd.DataFrame({"t": ps.t, "x": pos[:, 0], "y": pos[:, 1], "speed": spd})
    if start is None:
        return dict(detected=False, unit=ground.unit,
                    note=f"没有检测到速度超过 {start_speed} {ground.unit}/s 的启动"), pd.DataFrame(), track

    # 启动时刻回溯到速度开始上升的位置（速度 < 阈值的 30%）
    s0 = start
    while s0 > 0 and np.isfinite(spd[s0 - 1]) and spd[s0 - 1] > 0.3 * start_speed:
        s0 -= 1

    step_win = int(0.35 * fps)
    e = min(s0 + step_win, len(pos) - 1)
    # 突破方向用启动后 0.6 s（约前两步）的双脚中点位移，避免交叉步单步的横向摆动带偏方向
    e2 = min(s0 + int(0.6 * fps), len(pos) - 1)
    disp = pos[e2] - pos[s0]
    heading0 = np.degrees(np.arctan2(disp[1], disp[0]))

    if basket is not None:
        b = np.asarray(basket, float)
        b = b if (ground.H is not None or feet_m is not None) else ground(b[None])[0]
        to_b = b - pos[s0]
        ref_heading = np.degrees(np.arctan2(to_b[1], to_b[0]))
        ref_name = "篮筐方向"
    else:
        ref_heading = 90.0  # 无篮筐时：没标定 = 画面向上（远离镜头）；有标定 = 球场 +y
        ref_name = ref_name_m if feet_m is not None else ("画面纵深方向" if ground.H is None else "球场 +y 方向")
    drive_angle = float(g.wrap180(heading0 - ref_heading))

    # 第一步：启动后 0.35 s 内位移更大的脚
    dl = np.linalg.norm(lpos[e] - lpos[s0])
    dr = np.linalg.norm(rpos[e] - rpos[s0])
    step_foot = "left" if dl > dr else "right"
    mover, stance = (lpos, rpos) if step_foot == "left" else (rpos, lpos)
    d_unit = disp / (np.linalg.norm(disp) + 1e-9)
    lat = np.array([-d_unit[1], d_unit[0]])  # 前进方向左侧
    before = np.dot(mover[s0] - stance[s0], lat)
    after = np.dot(mover[e] - stance[e], lat)
    # 迈出的脚从支撑脚一侧跨到另一侧 = 交叉步
    step_type = "交叉步" if np.sign(before) != np.sign(after) and abs(after) > (0.05 if ground.unit == "m" else 0.03) else "顺步"
    step_len = float(np.linalg.norm(mover[e] - mover[s0]))

    # 重心：髋中点高度（相对站立）下降多少身高
    hip_y = (im[:, L.L_HIP, 1] + im[:, L.R_HIP, 1]) / 2
    stand = np.nanmedian(hip_y[max(0, s0 - int(1.0 * fps)):s0]) if s0 > 0 else np.nanmedian(hip_y)
    hip_drop = float(np.nanmax(hip_y[s0:e + 1] - stand) / stature)

    lk = 180 - g.joint_angle(im[:, L.L_HIP, :2], im[:, L.L_KNEE, :2], im[:, L.L_ANKLE, :2])
    rk = 180 - g.joint_angle(im[:, L.R_HIP, :2], im[:, L.R_KNEE, :2], im[:, L.R_ANKLE, :2])

    def a(col, i):
        return float(ang[col].iloc[i])

    peak_i = s0 + int(np.nanargmax(spd[s0:min(len(spd), s0 + int(1.0 * fps))]))
    summary = dict(
        detected=True, unit=ground.unit, t_start=s0 / fps,
        drive_angle_deg=drive_angle, drive_angle_ref=ref_name,
        first_step_foot=step_foot, first_step_type=step_type,
        first_step_length=step_len,
        hip_drop_stature=hip_drop,
        trunk_lean_fwd_start=a("trunk_lean_fwd_3d", s0),
        trunk_lean_fwd_first_step=float(np.nanmax(ang["trunk_lean_fwd_3d"].iloc[s0:e + 1])),
        trunk_lean_side_first_step=a("trunk_lean_side_3d", e),
        knee_flex_left_start=float(lk[s0]), knee_flex_right_start=float(rk[s0]),
        peak_speed=float(spd[peak_i]), t_to_peak_speed=(peak_i - s0) / fps,
        accel_0_035s=float((spd[e] - spd[s0]) / ((e - s0) / fps)) if e > s0 else np.nan,
    )

    # 变向：移动中 0.4 s 窗口内前进方向变化超过阈值
    heading = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    w = int(0.4 * fps)
    cuts = []
    i = e2  # 前两步（0.6 s）内的横向摆动属于起步，不算变向
    while i + w < len(heading):
        if moving[i] and moving[i + w]:
            dh = g.wrap180(heading[i + w] - heading[i])
            if abs(dh) >= cut_threshold_deg:
                # 转向往往持续超过窗口：在 2 倍窗口内找到方向变化的最大值作为切角
                hi = min(i + 2 * w + 1, len(heading))
                seg = g.wrap180(heading[i:hi] - heading[i])
                seg = np.where(moving[i:hi] & np.isfinite(seg), seg, 0.0)
                k_end = int(np.argmax(np.abs(seg)))
                dh = seg[k_end]
                j = i + int(np.argmax(np.abs(np.gradient(seg[:k_end + 1])))) if k_end > 1 else i
                cuts.append(dict(t=j / fps, cut_angle_deg=float(dh),
                                 speed_at_cut=float(spd[j]),
                                 trunk_lean_side=a("trunk_lean_side_3d", j),
                                 knee_flex_left=float(lk[j]), knee_flex_right=float(rk[j])))
                i += max(k_end, w)
                continue
        i += 1
    return summary, pd.DataFrame(cuts), track
