"""双机 3D：用已知 3D 运动 + 两台虚拟手机验证时间对齐、标定和三角化。"""
import numpy as np
import pytest

from hoopcv import landmarks as L
from hoopcv import stereo as S
from hoopcv.angles import compute_angles, smooth_series
from hoopcv.pose import PoseSeries

from test_pipeline import _body, _interp

FPS = 120.0
OFFSET = 0.0734          # B 的时间 = A 的时间 + OFFSET
W, H, F = 1920, 1080, 1500.0
RNG = np.random.default_rng(7)


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _axis_rot(v, axis, ang):
    k = axis / np.linalg.norm(axis)
    return v * np.cos(ang) + np.cross(k, v) * np.sin(ang) + k * np.dot(k, v) * (1 - np.cos(ang))


def world_pose(t):
    """世界坐标（x 前、y 上、z 右，米）。先转身走两步，再原地投篮；投篮手前臂带内外旋。"""
    k = _interp(t, [(0, 15), (1.5, 15), (2.1, 60), (2.5, 0), (3.2, 0), (3.6, 15)])
    e = _interp(t, [(0, 30), (1.0, 60), (1.6, 25), (2.0, 20), (2.35, 90), (2.55, 150), (2.9, 150), (3.4, 30)])
    f = _interp(t, [(0, 60), (1.0, 90), (1.6, 40), (2.0, 30), (2.35, 100), (2.55, 10), (3.4, 40)])
    w = _interp(t, [(0, 0), (2.55, 0), (2.7, 60), (3.0, 60), (3.3, 0)])
    er = _interp(t, [(0, -40), (1.0, 30), (1.6, -20), (2.35, 10), (2.55, -15), (3.4, 20)])
    yaw = _interp(t, [(0, -50), (1.2, 25), (1.7, 0)])
    pos = _interp(t, [(0, -0.8), (1.2, 0.2), (1.7, 0)])
    P = _body(0, 0, k, e, f, w)
    # 投篮手（右）前臂与手绕上臂轴旋转：制造真实的 3D 肩内外旋
    sh, el = P[L.R_SHOULDER], P[L.R_ELBOW]
    for j in (L.R_WRIST, L.R_INDEX, L.R_PINKY, L.R_THUMB):
        P[j] = el + _axis_rot(P[j] - el, el - sh, np.radians(er))
    R = _rot_y(np.radians(yaw))
    return P @ R.T + np.array([pos, 0, 0.3 * pos])


def look_at(C, target):
    fwd = target - C
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    return S.Camera(np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]]), R, -R @ C)


CAM_A = look_at(np.array([0, 1.3, 5.0]), np.array([0, 1.3, 0]))            # 侧面、水平
CAM_B = look_at(np.array([3.54, 1.3, 3.54]), np.array([0, 1.0, 0]))        # 斜前 45°


def render(cam, times, noise=1.0):
    img = np.zeros((len(times), 33, 3))
    for i, t in enumerate(times):
        X = world_pose(t)
        img[i, :, :2] = cam.project(X) + RNG.normal(0, noise, (33, 2))
        img[i, :, 2] = 0.9
    drop = RNG.random(img.shape[:2]) < 0.02
    img[drop] = np.nan
    img[drop, 2] = 0.0
    return img


@pytest.fixture(scope="module")
def rig():
    ta = np.arange(0, 4.0, 1 / FPS)
    tb = np.arange(0, 4.2, 1 / FPS)
    pa = PoseSeries(FPS, W, H, render(CAM_A, ta), np.zeros((len(ta), 33, 3)))
    pb = PoseSeries(FPS, W, H, render(CAM_B, tb - OFFSET), np.zeros((len(tb), 33, 3)))
    truth = np.stack([world_pose(t) for t in ta])
    truth = np.stack([truth[..., 0], -truth[..., 1], -truth[..., 2]], -1)  # → x 前、y 下、z 左（右手系）
    gt = PoseSeries(FPS, W, H, pa.img.copy(), truth)
    return pa, pb, gt


def test_motion_sync(rig):
    pa, pb, _ = rig
    r = S.sync_motion(pa, pb)
    assert r.offset == pytest.approx(OFFSET, abs=0.5 / FPS)
    assert r.confidence > 1.2


def test_audio_sync_signals():
    sr = 16000
    n = sr * 6
    xa = RNG.normal(0, 0.002, n)
    clicks = [0.8, 1.9, 2.05, 3.3, 4.75]  # 拍手 + 几次运球
    for c in clicks:
        i = int(c * sr)
        xa[i:i + 200] += RNG.normal(0, 0.5, 200) * np.exp(-np.arange(200) / 40)
    shift = 0.2371
    xb = np.r_[RNG.normal(0, 0.002, int(shift * sr)), xa][:n] * 0.6 + RNG.normal(0, 0.002, n)
    r = S.sync_audio_signals(xa, xb, sr)
    assert r.offset == pytest.approx(shift, abs=0.0015)
    assert r.detail["clap_time_a"] == pytest.approx(0.8, abs=0.01)


def _errors(ps3d, gt, cols):
    a = compute_angles(smooth_series(ps3d), "right")
    b = compute_angles(smooth_series(gt), "right")
    return {c: float(np.nanmedian(np.abs(a[c] - b[c]))) for c in cols}


COLS = ["elbow_flex_3d", "knee_flex_3d", "hip_flex_3d", "shoulder_elev_3d", "shoulder_ext_rot_3d",
        "trunk_lean_fwd_3d"]


@pytest.mark.parametrize("focal_guess", [F, F * 1.1])
def test_auto_calibration_recovers_angles(rig, focal_guess):
    pa, pb, gt = rig
    b_on_a = S.align_b_to_a(pa, pb, OFFSET)
    cal = S.calibrate(pa.img, b_on_a, (W, H), (W, H), focal_a=focal_guess, focal_b=focal_guess, height_m=1.75)
    assert cal.reproj_px["a_median"] < 2.0 and cal.reproj_px["b_median"] < 2.0
    ps3d = S.stereo_series(pa, b_on_a, cal)
    err = _errors(ps3d, gt, COLS)
    print(focal_guess, err, cal.cam_a.K[0, 0], cal.cam_b.K[0, 0])
    for c in COLS:
        assert err[c] < 3.0, (c, err[c])


def test_court_calibration_metric(rig):
    pa, pb, gt = rig
    b_on_a = S.align_b_to_a(pa, pb, OFFSET)
    # 球场坐标 (x, y) ↔ 世界 (x, 0, -y)
    m = np.array([[-3, -2], [3, -2], [3, 2], [-3, 2], [0, 3], [-2, 0]], float)
    Xw = np.c_[m[:, 0], np.zeros(len(m)), -m[:, 1]]
    court = dict(m=m, a=CAM_A.project(Xw) + RNG.normal(0, 1, (6, 2)), b=CAM_B.project(Xw) + RNG.normal(0, 1, (6, 2)))
    cal = S.calibrate(pa.img, b_on_a, (W, H), (W, H), court=court)
    ps3d = S.stereo_series(pa, b_on_a, cal)
    d = np.linalg.norm(ps3d.world - gt.world, axis=-1)
    assert np.nanmedian(d) < 0.03
    la, ra = S.ground_feet(ps3d, cal)
    true_la = gt.world[:, L.L_ANKLE][:, [0, 2]]
    assert np.nanmedian(np.linalg.norm(la - true_la, axis=1)) < 0.03
    err = _errors(ps3d, gt, COLS)
    for c in COLS:
        assert err[c] < 3.0, (c, err[c])


def test_triangulate_exact():
    X = np.array([[0.1, 1.2, 0.3], [-0.2, 0.5, -0.1]])
    obs = [CAM_A.project(X), CAM_B.project(X)]
    Y = S.triangulate([CAM_A, CAM_B], obs)
    assert np.allclose(X, Y, atol=1e-6)
    obs[1][1] = np.nan
    Y = S.triangulate([CAM_A, CAM_B], obs)
    assert np.isnan(Y[1]).all() and np.allclose(Y[0], X[0], atol=1e-6)
