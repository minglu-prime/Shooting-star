"""用已知运动学合成的关键点序列测试投篮与突破模块（不需要视频和模型）。"""
import numpy as np
import pytest

from hoopcv import landmarks as L
from hoopcv.angles import compute_angles, smooth_series
from hoopcv.drive import analyze_drive
from hoopcv.pose import PoseSeries
from hoopcv.shot import analyze_shots

FPS = 120.0
PX = 200.0  # 像素/米


def _interp(t, pts):
    ts, vs = zip(*pts)
    return np.interp(t, ts, vs)


def _body(x, y, k, e, f, w, z=0.0):
    """矢状面简笔人，x 向前、y 向上（米）。返回 33×3 的世界坐标（x 前、y 上、z 右）。"""
    P = np.zeros((33, 3))
    rk = np.radians(k)
    for side, dz in (("L", -0.12), ("R", 0.12)):
        ankle = np.array([x, y + 0.08, z + dz])
        knee = ankle + 0.45 * np.array([np.sin(rk / 2), np.cos(rk / 2), 0])
        hip = knee + 0.45 * np.array([-np.sin(rk / 2), np.cos(rk / 2), 0])
        heel = ankle + np.array([-0.05, -0.07, 0])
        toe = ankle + np.array([0.15, -0.07, 0])
        sz = dz * 1.6
        sh = hip + np.array([0, 0.55, sz - dz])
        re, rf, rw = np.radians(e), np.radians(e + f), np.radians(e + f - w)
        el = sh + 0.30 * np.array([np.sin(re), -np.cos(re), 0])
        wr = el + 0.27 * np.array([np.sin(rf), -np.cos(rf), 0])
        idx = wr + 0.08 * np.array([np.sin(rw), -np.cos(rw), 0])
        pk = wr + 0.07 * np.array([np.sin(rw), -np.cos(rw), 0]) + np.array([0, 0, 0.03 if side == "R" else -0.03])
        ids = dict(L=dict(sh=L.L_SHOULDER, el=L.L_ELBOW, wr=L.L_WRIST, idx=L.L_INDEX, pk=L.L_PINKY, th=L.L_THUMB,
                          hip=L.L_HIP, kn=L.L_KNEE, an=L.L_ANKLE, he=L.L_HEEL, ft=L.L_FOOT),
                   R=dict(sh=L.R_SHOULDER, el=L.R_ELBOW, wr=L.R_WRIST, idx=L.R_INDEX, pk=L.R_PINKY, th=L.R_THUMB,
                          hip=L.R_HIP, kn=L.R_KNEE, an=L.R_ANKLE, he=L.R_HEEL, ft=L.R_FOOT))[side]
        for key, p in dict(sh=sh, el=el, wr=wr, idx=idx, pk=pk, th=idx, hip=hip, kn=knee, an=ankle,
                           he=heel, ft=toe).items():
            P[ids[key]] = p
    mid_sh = (P[L.L_SHOULDER] + P[L.R_SHOULDER]) / 2
    P[L.NOSE] = mid_sh + np.array([0.08, 0.2, 0])
    P[1:11] = P[L.NOSE]
    return P


def _to_series(frames_world, fps):
    W = np.stack(frames_world)
    img = np.zeros(W.shape)
    img[..., 0] = 400 + PX * W[..., 0]
    img[..., 1] = 1000 - PX * W[..., 1]
    img[..., 2] = 1.0
    # MediaPipe 世界坐标：x 画面右、y 向下、z 远离相机；人朝右站、右侧对镜头
    world = np.stack([W[..., 0], -W[..., 1], -W[..., 2]], axis=-1)
    return PoseSeries(fps, 1200, 1100, img, world)


@pytest.fixture(scope="module")
def shot_case():
    t = np.arange(0, 2.0, 1 / FPS)
    k = _interp(t, [(0, 10), (0.3, 10), (0.6, 60), (1.0, 0), (1.6, 0), (1.9, 10)])
    e = _interp(t, [(0, 20), (0.5, 20), (0.85, 90), (1.05, 150), (1.4, 150), (1.8, 20)])
    f = _interp(t, [(0, 30), (0.5, 30), (0.85, 100), (1.05, 10), (1.8, 30)])
    w = _interp(t, [(0, 0), (1.05, 0), (1.2, 60), (1.5, 60), (1.7, 0)])
    jump = np.where((t > 1.0) & (t < 1.4), 0.2 * (1 - ((t - 1.2) / 0.2) ** 2), 0)
    frames = [_body(0, jump[i], k[i], e[i], f[i], w[i]) for i in range(len(t))]
    ps = _to_series(frames, FPS)

    # 球：出手前随手，出手（1.05 s）后以 52°、7 m/s 抛出
    t_rel, ang, v = 1.05, np.radians(52), 7.0
    ball = np.full((len(t), 3), np.nan)
    i_rel = int(round(t_rel * FPS))
    p0 = None
    for i in range(len(t)):
        hand = (ps.img[i, L.R_WRIST, :2] + ps.img[i, L.R_INDEX, :2]) / 2
        if i <= i_rel:
            ball[i] = (*(hand + [10, -10]), 0.9)
            p0 = ball[i, :2].copy()
        else:
            dt = t[i] - t[i_rel]
            bx = p0[0] + PX * v * np.cos(ang) * dt
            by = p0[1] - PX * (v * np.sin(ang) * dt - 0.5 * 9.81 * dt ** 2)
            ball[i] = (bx, by, 0.9)
    sm = smooth_series(ps)
    df = compute_angles(sm, "right")
    return sm, df, ball, dict(t_rel=t_rel, ang=52.0, v=v)


def test_angles_match_known_kinematics(shot_case):
    sm, df, _, _ = shot_case
    i = int(0.6 * FPS)
    assert df["knee_flex_2d"].iloc[i] == pytest.approx(60, abs=3)
    assert df["knee_flex_3d"].iloc[i] == pytest.approx(60, abs=3)
    i = int(0.85 * FPS)
    assert df["elbow_flex_2d"].iloc[i] == pytest.approx(100, abs=6)  # 尖角被 8 Hz 低通削平几度
    assert df["shoulder_elev_3d"].iloc[i] == pytest.approx(90, abs=4)
    # 前举 + 前臂朝上 = 肩中立位
    assert df["shoulder_ext_rot_3d"].iloc[i] == pytest.approx(0, abs=5)
    assert df.attrs["facing"] == 1


def test_shot_detection_with_ball(shot_case):
    sm, df, ball, truth = shot_case
    shots, events = analyze_shots(sm, df, "right", ball=ball, made=[1])
    assert len(shots) == 1
    r = shots.iloc[0]
    assert r["release_detected_by"] == "ball"
    assert r["t_release"] == pytest.approx(truth["t_rel"], abs=0.04)
    assert r["t_dip"] == pytest.approx(0.6, abs=0.05)
    assert r["knee_flex_dip"] == pytest.approx(60, abs=3)
    assert r["release_angle_deg"] == pytest.approx(truth["ang"], abs=1.0)
    assert r["release_speed_mps"] == pytest.approx(truth["v"], rel=0.05)
    assert r["chain_order"].startswith("knee_ext") or r["chain_order"].startswith("hip_ext")


def test_shot_detection_without_ball(shot_case):
    sm, df, _, _ = shot_case
    shots, _ = analyze_shots(sm, df, "right")
    assert len(shots) == 1
    r = shots.iloc[0]
    assert r["release_detected_by"] == "elbow_velocity"
    assert r["t_release"] == pytest.approx(1.0, abs=0.08)


def test_drive_angle_step_and_cut():
    fps = 60.0
    t = np.arange(0, 3.0, 1 / fps)
    h0, h1 = np.radians(70), np.radians(130)  # 先朝 70°（篮筐在 90°），2.0 s 时变向到 130°
    pos = np.zeros((len(t), 2))
    speed = np.clip((t - 1.0) / 0.5, 0, 1) * 5.0
    for i in range(1, len(t)):
        hd = h0 if t[i] < 2.0 else h1
        pos[i] = pos[i - 1] + speed[i] / fps * np.array([np.cos(hd), np.sin(hd)])
    d = np.array([np.cos(h0), np.sin(h0)])
    lat_right = np.array([d[1], -d[0]])
    lfoot, rfoot = np.zeros_like(pos), np.zeros_like(pos)
    for i, ti in enumerate(t):
        if ti < 1.0:
            lfoot[i], rfoot[i] = (-0.2, 0), (0.2, 0)
        elif ti < 1.35:  # 左脚从右脚前方跨过：交叉步
            a = (ti - 1.0) / 0.35
            rfoot[i] = (0.2, 0)
            target = np.array([0.2, 0]) + 0.7 * d + 0.25 * lat_right
            lfoot[i] = (1 - a) * np.array([-0.2, 0]) + a * target
        else:
            c = pos[i]
            lfoot[i], rfoot[i] = c - 0.15 * lat_right, c + 0.15 * lat_right
    to_px = lambda p: np.c_[640 + 100 * p[:, 0], 900 - 100 * p[:, 1]]  # noqa: E731
    frames = []
    for i in range(len(t)):
        P = _body(0, 0, 20, 20, 30, 0)
        frames.append(P)
    ps = _to_series(frames, fps)
    ps.img[:, L.L_ANKLE, :2] = to_px(lfoot)
    ps.img[:, L.R_ANKLE, :2] = to_px(rfoot)
    court_m = np.array([[0, 0], [5, 0], [5, 5], [0, 5]], float)
    court_px = to_px(court_m)
    df = compute_angles(ps, "right")
    summary, cuts, track = analyze_drive(ps, df, court_px, court_m, basket=(0, 5))
    assert summary["detected"]
    assert summary["unit"] == "m"
    assert summary["t_start"] == pytest.approx(1.0, abs=0.1)
    # 起步方向 70°，篮筐方向 90° → 偏右 20°
    assert summary["drive_angle_deg"] == pytest.approx(-20, abs=5)
    assert summary["first_step_foot"] == "left"
    assert summary["first_step_type"] == "交叉步"
    assert len(cuts) >= 1
    assert cuts.iloc[0]["cut_angle_deg"] == pytest.approx(60, abs=10)
    assert cuts.iloc[0]["t"] == pytest.approx(2.0, abs=0.25)
