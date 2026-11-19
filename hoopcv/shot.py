"""投篮分析：自动切分每次投篮，提取关键时刻的角度、出手角、出手速度和动力链顺序。

关键时刻（每次投篮）：
  dip      下蹲最低点（膝屈曲最大）
  set      举球到位（下蹲后、出手前肘屈曲最大）
  release  出手（有球轨迹时 = 球离手；否则 = 肘伸展角速度峰值）
  follow   出手后 0.3 s 内腕屈曲最大（压腕随挥）

推荐拍摄：投篮手一侧的正侧面，机位与胸同高，120-240 fps，球框与出手点都在画面内。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import landmarks as L
from .pose import PoseSeries

G = 9.81


def _segments(mask: np.ndarray, min_len: int, merge_gap: int):
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    segs = [[s, e] for s, e in zip(idx[::2], idx[1::2])]
    merged = []
    for s, e in segs:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s >= min_len]


def _nanargmax(x, lo, hi):
    lo, hi = max(lo, 0), min(hi, len(x))
    if hi <= lo or np.all(np.isnan(x[lo:hi])):
        return None
    return lo + int(np.nanargmax(x[lo:hi]))


def _deriv(x, fps):
    return np.gradient(x) * fps


def stature_px(ps: PoseSeries) -> float:
    """身高（像素）估计：鼻子到脚跟的竖直距离 × 1.08，取站立帧中位数。"""
    im = ps.img
    heel = np.nanmean(im[:, [L.L_HEEL, L.R_HEEL], 1], axis=1)
    h = (heel - im[:, L.NOSE, 1]) * 1.08
    return float(np.nanpercentile(h, 75))


def fit_ball_flight(ball: np.ndarray, fps: float, start: int, facing: int,
                    max_dur: float = 0.5, min_pts: int = 5, t_eval: float | None = None):
    """对出手后的球心做 x 线性、y 二次拟合。

    返回 dict：release_angle_deg（相对水平，向上为正）、release_speed_mps（用拟合出的重力加速度
    换算像素/米，要求帧率真实）、px_per_m、n_points；点数不够返回 None。
    """
    end = min(len(ball), start + int(max_dur * fps))
    seg = ball[start:end]
    ok = np.isfinite(seg[:, 0])
    if ok.sum() < min_pts:
        return None
    t = np.arange(start, end)[ok] / fps
    t0 = start / fps
    x, y = seg[ok, 0], seg[ok, 1]
    bx = np.polyfit(t - t0, x, 1)
    ay = np.polyfit(t - t0, y, 2)
    te = 0.0 if t_eval is None else t_eval - t0   # 在真实出手时刻求切线方向（可外推到拟合段之前）
    vx, vy = bx[0], ay[1] + 2 * ay[0] * te
    g_px = 2 * ay[0]                       # 图像 y 向下，重力为正
    ang = float(np.degrees(np.arctan2(-vy, vx * facing)))
    res = dict(release_angle_deg=ang, n_points=int(ok.sum()), px_per_m=np.nan, release_speed_mps=np.nan)
    if g_px > 0:
        ppm = g_px / G
        res["px_per_m"] = float(ppm)
        res["release_speed_mps"] = float(np.hypot(vx, vy) / ppm)
    resid = np.sqrt(np.mean((np.polyval(ay, t - t0) - y) ** 2))
    res["fit_rmse_px"] = float(resid)
    return res


def _release_from_ball(ball, hand, s, e, hand_size):
    """球心与手的距离首次超过 2.5 倍手掌尺寸的帧。"""
    if ball is None:
        return None
    d = np.linalg.norm(ball[:, :2] - hand, axis=1)
    for i in range(max(s, 0), min(e, len(d))):
        if np.isfinite(d[i]) and d[i] > 2.5 * hand_size[i]:
            # 要求之后两帧也远离，避免误检
            nxt = d[i + 1:i + 3]
            if np.all(~np.isfinite(nxt) | (nxt > 2.5 * hand_size[i])):
                return i
    return None


def analyze_shots(ps: PoseSeries, ang: pd.DataFrame, side: str = "right",
                  ball: np.ndarray | None = None, made: list[int] | None = None):
    """返回 (shots DataFrame, events dict 列表)。"""
    s = L.side_ids(side)
    fps = ps.fps
    im = ps.img
    facing = ang.attrs.get("facing", 1)
    wrist_y = im[:, s["wrist"], 1]
    nose_y = im[:, L.NOSE, 1]
    hand = (im[:, s["wrist"], :2] + im[:, s["index"], :2]) / 2
    hand_size = np.linalg.norm(im[:, s["wrist"], :2] - im[:, s["index"], :2], axis=1)
    hand_size = np.where(np.isfinite(hand_size), np.maximum(hand_size, 5), np.nan)
    stature = stature_px(ps)
    heel_y = np.nanmean(im[:, [L.L_HEEL, L.R_HEEL], 1], axis=1)
    ground = np.nanpercentile(heel_y, 90)

    above = np.nan_to_num(wrist_y < nose_y, nan=False).astype(bool)
    segs = _segments(above, min_len=max(2, int(0.08 * fps)), merge_gap=int(0.3 * fps))

    knee = ang["knee_flex_2d"].to_numpy()
    elbow = ang["elbow_flex_2d"].to_numpy()
    wristf = ang["wrist_flex_2d"].to_numpy()
    elbow_ext_vel = -_deriv(elbow, fps)

    # 动力链：各关节伸展（或屈曲）角速度
    chain = {
        "knee_ext": -_deriv(knee, fps),
        "hip_ext": -_deriv(ang["hip_flex_2d"].to_numpy(), fps),
        "shoulder_flex": _deriv(ang["shoulder_flex_2d"].to_numpy(), fps),
        "elbow_ext": elbow_ext_vel,
        "wrist_flex": _deriv(wristf, fps),
    }

    rows, events = [], []
    for k, (a, b) in enumerate(segs):
        lookback = int(1.2 * fps)
        dip = _nanargmax(knee, a - lookback, a + 1)
        if dip is None:
            dip = max(a - int(0.5 * fps), 0)
        rel = _release_from_ball(ball, hand, a - int(0.2 * fps), b + int(0.2 * fps), hand_size)
        rel_src = "ball"
        fit_start = rel
        if rel is not None:
            # 球离手判定会晚几帧：回溯到球-手距离开始离开持球距离的那一帧
            d = np.linalg.norm(ball[:, :2] - hand, axis=1)
            hold = np.nanmedian(d[max(rel - int(0.3 * fps), 0):max(rel - int(0.1 * fps), 1)])
            if np.isfinite(hold):
                r = rel
                while r > 0 and np.isfinite(d[r - 1]) and d[r - 1] > hold + 0.5 * hand_size[r]:
                    r -= 1
                rel = r
        else:
            pk = _nanargmax(elbow_ext_vel, a - int(0.2 * fps), b)
            if pk is None:
                continue
            # 出手通常在肘伸展角速度峰值之后、伸展即将结束时：取角速度降到峰值 30% 的帧
            rel = pk
            while rel + 1 < min(b + int(0.2 * fps), len(elbow_ext_vel)) and \
                    elbow_ext_vel[rel + 1] > 0.3 * elbow_ext_vel[pk]:
                rel += 1
            rel_src = "elbow_velocity"
            fit_start = rel
        setp = _nanargmax(elbow, dip, rel + 1)
        setp = dip if setp is None else setp
        follow = _nanargmax(wristf, rel, rel + int(0.3 * fps))
        follow = rel if follow is None else follow
        # 起跳：下蹲后脚跟离地高于地面 2% 身高
        lift = (ground - heel_y) / stature
        to = next((i for i in range(dip, min(rel + int(0.3 * fps), len(lift)))
                   if np.isfinite(lift[i]) and lift[i] > 0.02), None)
        jump = float(np.nanmax(lift[dip:min(len(lift), rel + int(0.5 * fps))])) if stature > 0 else np.nan

        row = dict(shot=k + 1, t_dip=dip / fps, t_set=setp / fps, t_release=rel / fps,
                   release_detected_by=rel_src,
                   dip_to_release_s=(rel - dip) / fps,
                   takeoff_before_release_s=((rel - to) / fps) if to is not None else np.nan,
                   jump_height_stature=jump,
                   release_height_stature=(ground - wrist_y[rel]) / stature)

        def at(col, i):
            return float(ang[col].iloc[i]) if i is not None else np.nan

        row.update({
            "knee_flex_dip": at("knee_flex_2d", dip),
            "hip_flex_dip": at("hip_flex_2d", dip),
            "trunk_lean_dip": at("trunk_lean_2d", dip),
            "elbow_flex_set": at("elbow_flex_2d", setp),
            "shoulder_ext_rot_set": at("shoulder_ext_rot_3d", setp),
            "forearm_pron_set": at("forearm_pron_3d", setp),
            "elbow_flex_release": at("elbow_flex_2d", rel),
            "shoulder_flex_release": at("shoulder_flex_2d", rel),
            "shoulder_ext_rot_release": at("shoulder_ext_rot_3d", rel),
            "forearm_pron_release": at("forearm_pron_3d", rel),
            "trunk_lean_release": at("trunk_lean_2d", rel),
            "knee_flex_release": at("knee_flex_2d", rel),
            "wrist_flex_follow": at("wrist_flex_2d", follow),
        })

        # 出手角：优先球轨迹，否则用出手前 3 帧手部速度方向近似
        fit = fit_ball_flight(ball, fps, fit_start, facing, t_eval=rel / fps) if ball is not None else None
        if fit:
            row.update(release_angle_deg=fit["release_angle_deg"], release_angle_source="ball_fit",
                       release_speed_mps=fit["release_speed_mps"], ball_fit_points=fit["n_points"],
                       ball_fit_rmse_px=fit["fit_rmse_px"])
        else:
            # 手腕在出手瞬间正绕肘/腕转动，其速度方向不能代表球的出手角，因此不做近似
            row.update(release_angle_deg=np.nan,
                       release_angle_source="no_ball_track" if ball is None else "ball_fit_failed",
                       release_speed_mps=np.nan)

        # 动力链：各关节角速度峰值相对出手的时间（负 = 出手前）
        win = (dip, min(rel + int(0.15 * fps), len(knee)))
        for name, vel in chain.items():
            pk = _nanargmax(vel, *win)
            row[f"peak_{name}_t_rel"] = (pk - rel) / fps if pk is not None else np.nan
            row[f"peak_{name}_dps"] = float(vel[pk]) if pk is not None else np.nan
        order = sorted(((row[f"peak_{n}_t_rel"], n) for n in chain if np.isfinite(row[f"peak_{n}_t_rel"])))
        row["chain_order"] = " > ".join(n for _, n in order)
        if made is not None and k < len(made):
            row["made"] = int(made[k])
        rows.append(row)
        events.append(dict(shot=k + 1, dip=dip, set=setp, release=rel, follow=follow, takeoff=to))

    return pd.DataFrame(rows), events


def compare_made_missed(shots: pd.DataFrame) -> pd.DataFrame | None:
    """进球 vs 不进的各指标均值差，按差值相对标准差排序（个体内对比最有训练价值）。"""
    if "made" not in shots or shots["made"].nunique() < 2:
        return None
    num = shots.select_dtypes("number").drop(columns=["shot", "made"], errors="ignore")
    m = num[shots["made"] == 1].mean()
    x = num[shots["made"] == 0].mean()
    sd = num.std().replace(0, np.nan)
    df = pd.DataFrame({"made_mean": m, "missed_mean": x, "diff": m - x, "effect": (m - x) / sd})
    return df.reindex(df["effect"].abs().sort_values(ascending=False).index)
