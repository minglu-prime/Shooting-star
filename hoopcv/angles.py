"""逐帧关节角度。

2D 列（后缀 _2d）：在图像平面里算，前提是动作平面大致垂直于镜头（如投篮拍正侧面）。
3D 列（后缀 _3d）：用 MediaPipe 的 3D 世界坐标算。屈伸角比 2D 更不受机位影响，但单目深度有误差；
    内外旋、前臂旋前只能在 3D 里定义，单机位下请当作趋势参考，严谨测量请用多机位（Pose2Sim/OpenCap）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import geometry as g
from . import landmarks as L
from .pose import PoseSeries


def smooth_series(ps: PoseSeries, cutoff: float = 8.0, min_vis: float = 0.3) -> PoseSeries:
    """低可见度点置 NaN → 补短缺口（≤0.15 s）→ 零相位低通。"""
    img = ps.img.copy()
    world = ps.world.copy()
    low = img[..., 2] < min_vis
    img[low, :2] = np.nan
    world[low] = np.nan
    gap = max(1, int(round(0.15 * ps.fps)))
    xy = g.lowpass(g.fill_short_gaps(img[..., :2], gap), ps.fps, cutoff)
    world = g.lowpass(g.fill_short_gaps(world, gap), ps.fps, cutoff)
    img = np.concatenate([xy, img[..., 2:3]], axis=-1)
    return PoseSeries(ps.fps, ps.width, ps.height, img, world)


def facing_direction(ps: PoseSeries) -> int:
    """被拍者在画面中朝向：+1 朝右，-1 朝左（用脚尖-脚跟方向，退而用鼻子-肩中点）。"""
    im = ps.img
    toe = np.nanmean(im[:, [L.L_FOOT, L.R_FOOT], 0] - im[:, [L.L_HEEL, L.R_HEEL], 0], axis=1)
    nose = im[:, L.NOSE, 0] - (im[:, L.L_SHOULDER, 0] + im[:, L.R_SHOULDER, 0]) / 2
    v = np.nanmedian(toe)
    if not np.isfinite(v) or abs(v) < 2:
        v = np.nanmedian(nose)
    return 1 if (np.isfinite(v) and v >= 0) else -1


def compute_angles(ps: PoseSeries, side: str = "right") -> pd.DataFrame:
    s = L.side_ids(side)
    P = ps.img[..., :2]
    W = ps.world
    facing = facing_direction(ps)
    out = {"frame": np.arange(ps.n_frames), "t": ps.t}

    # ---------- 2D ----------
    sh, el, wr = P[:, s["shoulder"]], P[:, s["elbow"]], P[:, s["wrist"]]
    hip, kn, an = P[:, s["hip"]], P[:, s["knee"]], P[:, s["ankle"]]
    idx, foot = P[:, s["index"]], P[:, s["foot"]]
    out["elbow_flex_2d"] = g.flexion(sh, el, wr)
    out["knee_flex_2d"] = g.flexion(hip, kn, an)
    out["hip_flex_2d"] = g.flexion(sh, hip, kn)
    out["shoulder_flex_2d"] = g.joint_angle(hip, sh, el)      # 上臂与躯干夹角
    out["wrist_flex_2d"] = g.flexion(el, wr, idx)
    out["ankle_angle_2d"] = g.joint_angle(kn, an, foot)
    mid_sh = (P[:, L.L_SHOULDER] + P[:, L.R_SHOULDER]) / 2
    mid_hip = (P[:, L.L_HIP] + P[:, L.R_HIP]) / 2
    tr = mid_sh - mid_hip
    out["trunk_lean_2d"] = np.degrees(np.arctan2(tr[:, 0] * facing, -tr[:, 1]))  # 前倾为正

    # ---------- 3D ----------
    ls, rs, lh, rh = W[:, L.L_SHOULDER], W[:, L.R_SHOULDER], W[:, L.L_HIP], W[:, L.R_HIP]
    sh3, el3, wr3 = W[:, s["shoulder"]], W[:, s["elbow"]], W[:, s["wrist"]]
    hip3, kn3, an3 = W[:, s["hip"]], W[:, s["knee"]], W[:, s["ankle"]]
    out["elbow_flex_3d"] = g.flexion(sh3, el3, wr3)
    out["knee_flex_3d"] = g.flexion(hip3, kn3, an3)
    out["hip_flex_3d"] = g.flexion((ls + rs) / 2, hip3, kn3)

    mirror = np.array([1.0, 1.0, -1.0 if side == "left" else 1.0])
    o, R = g.thorax_frame(ls, rs, lh, rh)
    loc = lambda p: g.to_local(p, o, R) * mirror  # noqa: E731
    sh_l, el_l, wr_l = loc(sh3), loc(el3), loc(wr3)
    rot = g.limb_rotation(sh_l, el_l, wr_l)
    out["shoulder_elev_3d"] = rot["elevation"]
    out["shoulder_plane_3d"] = rot["plane"]
    out["shoulder_ext_rot_3d"] = rot["ext_rot"]
    out["forearm_pron_3d"] = g.forearm_pronation(sh_l, el_l, wr_l, loc(W[:, s["index"]]), loc(W[:, s["pinky"]]))

    op, Rp = g.pelvis_frame(ls, rs, lh, rh)
    locp = lambda p: g.to_local(p, op, Rp) * mirror  # noqa: E731
    hrot = g.limb_rotation(locp(hip3), locp(kn3), locp(2 * kn3 - an3))
    out["hip_ext_rot_3d"] = hrot["ext_rot"]
    out["hip_abd_plane_3d"] = hrot["plane"]

    # 躯干相对重力的倾角（假设相机水平；MediaPipe 世界坐标 y 向下）
    up = np.array([0.0, -1.0, 0.0])
    Y, X, Z = R[..., 1], R[..., 0], R[..., 2]
    fwd = g.unit(X - np.sum(X * up, -1, keepdims=True) * up)
    lat = g.unit(Z - np.sum(Z * up, -1, keepdims=True) * up)
    ydot = np.sum(Y * up, -1)
    out["trunk_lean_fwd_3d"] = np.degrees(np.arctan2(np.sum(Y * fwd, -1), ydot))
    out["trunk_lean_side_3d"] = np.degrees(np.arctan2(np.sum(Y * lat, -1), ydot)) * (1 if side == "right" else -1)

    df = pd.DataFrame(out)
    df.attrs["facing"] = facing
    df.attrs["side"] = side
    return df


COLUMN_NOTES = {
    "elbow_flex_2d": "肘屈曲角，伸直=0",
    "knee_flex_2d": "膝屈曲角，伸直=0",
    "hip_flex_2d": "髋屈曲角（躯干与大腿），直立=0",
    "shoulder_flex_2d": "上臂与躯干夹角",
    "wrist_flex_2d": "腕屈曲（前臂与手的夹角补角）",
    "ankle_angle_2d": "膝-踝-脚尖夹角",
    "trunk_lean_2d": "躯干前倾角（朝面向方向为正）",
    "shoulder_elev_3d": "上臂抬高角（3D）",
    "shoulder_plane_3d": "抬高平面：0=侧平举（冠状面），90=前平举（矢状面）",
    "shoulder_ext_rot_3d": "肩外旋（+）/内旋（-），单目近似",
    "forearm_pron_3d": "前臂旋前（+）/旋后（-），握手位=0，粗略",
    "hip_ext_rot_3d": "髋外旋（+）/内旋（-），单目近似",
    "trunk_lean_fwd_3d": "躯干相对重力前倾（假设相机水平）",
    "trunk_lean_side_3d": "躯干向分析侧倾斜为正",
}
