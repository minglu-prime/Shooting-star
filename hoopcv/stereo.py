"""双机位 3D：时间对齐 → 相机标定 → 三角化。

约定
- 手机 A 为参考机（侧面 90°），时间轴、2D 角度和事件切分都以 A 为准。
- 相机坐标系：x 向右、y 向下、z 向前（与 MediaPipe 世界坐标同向），均为右手系。
- 自标定模式下，3D 结果在 A 的相机坐标系里，要求 A 保持水平（app 的水平仪保证），这样 y 轴就是重力方向；
  尺度由球员身高确定，只影响距离和速度，不影响任何角度。
- 球场标定模式下，3D 结果在球场坐标系里（米），再转成 x 右、y 下、z 前 的约定交给角度模块。

时间对齐
- 首选音频：两台手机录到的声音（开拍时拍一下手、运球声）做起音包络互相关，精度约 1 ms。
- 兜底：两台手机里各关节竖直速度的互相关（两台都水平时，竖直运动在两个视角里同步出现）。
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import correlate, correlation_lags

from . import geometry as g
from . import landmarks as L
from .pose import PoseSeries

# 参与对齐和标定的关节：鼻子 + 躯干四肢（不用脸部细节点）
JOINTS = [0] + list(range(11, 33))
# 骨长约束：同一组内（左右对称）共用一个长度参数。人体当“标定物”，消除双视角的深度缩放歧义
BONE_GROUPS = [
    [(L.L_SHOULDER, L.L_ELBOW), (L.R_SHOULDER, L.R_ELBOW)],      # 上臂
    [(L.L_ELBOW, L.L_WRIST), (L.R_ELBOW, L.R_WRIST)],            # 前臂
    [(L.L_HIP, L.L_KNEE), (L.R_HIP, L.R_KNEE)],                  # 大腿
    [(L.L_KNEE, L.L_ANKLE), (L.R_KNEE, L.R_ANKLE)],              # 小腿
    [(L.L_SHOULDER, L.L_HIP), (L.R_SHOULDER, L.R_HIP)],          # 躯干侧边
    [(L.L_SHOULDER, L.R_SHOULDER)],                              # 肩宽
    [(L.L_HIP, L.R_HIP)],                                        # 髋宽
    [(L.L_HEEL, L.L_FOOT), (L.R_HEEL, L.R_FOOT)],                # 脚长
]
SYNC_JOINTS = [L.L_SHOULDER, L.R_SHOULDER, L.L_ELBOW, L.R_ELBOW, L.L_WRIST, L.R_WRIST,
               L.L_HIP, L.R_HIP, L.L_KNEE, L.R_KNEE, L.L_ANKLE, L.R_ANKLE, L.NOSE]


# ============================== 时间对齐 ==============================

@dataclass
class SyncResult:
    offset: float          # B 的时间 = A 的时间 + offset（秒）
    method: str            # "audio" / "motion" / "manual"
    confidence: float      # 互相关主峰与次峰之比，> 1.5 比较可靠
    detail: dict = field(default_factory=dict)


def read_audio(video: str, sr: int = 16000) -> np.ndarray | None:
    """用 ffmpeg 读单声道音频；没有音轨或没有 ffmpeg 时返回 None。"""
    try:
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", str(sr),
                            "-f", "s16le", "-"], capture_output=True, timeout=300)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0 or len(r.stdout) < sr * 2:
        return None
    return np.frombuffer(r.stdout, np.int16).astype(float) / 32768.0


def onset_envelope(x: np.ndarray, sr: int, hop: float = 0.001) -> np.ndarray:
    """起音强度包络（每 hop 秒一个值）：一阶差分高通 → 短时能量 → 对数能量的正向变化。"""
    x = np.diff(x, prepend=x[:1])
    n = max(1, int(round(sr * hop)))
    m = len(x) // n
    e = np.sqrt((x[:m * n].reshape(m, n) ** 2).mean(axis=1) + 1e-10)
    k = max(1, int(round(0.004 / hop)))  # 4 ms 平滑
    le = np.convolve(np.log(e), np.ones(k) / k, mode="same")
    return np.maximum(np.diff(le, prepend=le[:1]), 0.0)


def _xcorr_peak(a: np.ndarray, b: np.ndarray, step: float, max_lag: float, exclude: float):
    """返回 (lag 秒, 置信度)。lag 定义为 b[n + lag] ≈ a[n]。"""
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    c = correlate(b, a, mode="full", method="fft")
    lags = correlation_lags(len(b), len(a), mode="full")
    keep = np.abs(lags * step) <= max_lag
    c, lags = c[keep], lags[keep]
    k = int(np.argmax(c))
    frac = 0.0
    if 0 < k < len(c) - 1:  # 抛物线插值到亚采样精度
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    far = np.abs(lags - lags[k]) * step > exclude
    second = c[far].max() if far.any() else 0.0
    conf = float(c[k] / second) if second > 0 else float("inf")
    return float((lags[k] + frac) * step), conf


def sync_audio_signals(xa: np.ndarray, xb: np.ndarray, sr: int, max_lag: float = 10.0) -> SyncResult:
    hop = 0.001
    ea, eb = onset_envelope(xa, sr, hop), onset_envelope(xb, sr, hop)
    lag, conf = _xcorr_peak(ea, eb, hop, max_lag, exclude=0.05)
    head = ea[: int(10 / hop)]
    # 拍手 = 前 10 秒里第一个达到最强起音一半的起音
    clap_a = float(np.flatnonzero(head >= 0.5 * head.max())[0] * hop) if len(head) and head.max() > 0 else float("nan")
    return SyncResult(lag, "audio", conf, dict(clap_time_a=clap_a))


def sync_audio(video_a: str, video_b: str, time_scale_a: float = 1.0, time_scale_b: float = 1.0,
               max_lag: float = 10.0) -> SyncResult | None:
    """time_scale：慢动作导出视频的音频时间换算到真实时间的系数（= 容器帧率 / 真实帧率）。"""
    sr = 16000
    xa, xb = read_audio(video_a, sr), read_audio(video_b, sr)
    if xa is None or xb is None:
        return None
    if time_scale_a != time_scale_b:
        return None  # 两段慢动作倍率不同时音频对齐不可靠，交给动作互相关
    r = sync_audio_signals(xa, xb, sr, max_lag / time_scale_a)
    r.offset *= time_scale_a
    r.detail["clap_time_a"] *= time_scale_a
    return r


def _resample(img: np.ndarray, fps: float, times: np.ndarray) -> np.ndarray:
    """把关键点序列（T,33,C）线性插值到任意时刻（秒，相对该视频第 0 帧）；范围外为 NaN。"""
    f = times * fps
    i0 = np.floor(f).astype(int)
    u = (f - i0)[:, None, None]
    ok = (i0 >= 0) & (i0 + 1 < len(img))
    out = np.full((len(times),) + img.shape[1:], np.nan)
    a, b = img[i0[ok]], img[i0[ok] + 1]
    out[ok] = a + (b - a) * u[ok]
    if img.shape[-1] == 3:  # 可见度取两帧较小值
        out[ok, :, 2] = np.minimum(a[..., 2], b[..., 2])
    return out


def sync_motion(pa: PoseSeries, pb: PoseSeries, max_lag: float = 5.0) -> SyncResult:
    """各关节竖直速度（按身体尺寸归一化）的互相关，求和后取峰值。"""
    rate = 4.0 * max(pa.fps, pb.fps)
    step = 1.0 / rate

    def sig(ps: PoseSeries):
        t = np.arange(0, (ps.n_frames - 1) / ps.fps, step)
        y = _resample(ps.img[..., :2], ps.fps, t)[:, SYNC_JOINTS, 1]
        y = g.lowpass(g.fill_short_gaps(y, int(0.2 * rate)), rate, 6.0)  # 先平滑，避免关键点抖动主导速度
        size = np.nanmedian(np.abs(ps.img[:, L.NOSE, 1] - np.nanmean(ps.img[:, [L.L_ANKLE, L.R_ANKLE], 1], axis=1)))
        v = np.gradient(y, axis=0) / step / (size if np.isfinite(size) and size > 0 else 1.0)
        return np.nan_to_num(v)

    va, vb = sig(pa), sig(pb)
    total = None
    for j in range(va.shape[1]):
        a, b = va[:, j], vb[:, j]
        if a.std() == 0 or b.std() == 0:
            continue
        a = (a - a.mean()) / a.std()
        b = (b - b.mean()) / b.std()
        c = correlate(b, a, mode="full", method="fft")
        total = c if total is None else total + c
    if total is None:
        raise RuntimeError("两台手机的画面里都没有可用的关节轨迹，无法对齐时间")
    lags = correlation_lags(len(vb), len(va), mode="full")
    keep = np.abs(lags * step) <= max_lag
    total, lags = total[keep], lags[keep]
    k = int(np.argmax(total))
    frac = 0.0
    if 0 < k < len(total) - 1:
        y0, y1, y2 = total[k - 1], total[k], total[k + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    far = np.abs(lags - lags[k]) * step > 0.15
    second = total[far].max() if far.any() else 0.0
    conf = float(total[k] / second) if second > 0 else float("inf")
    return SyncResult(float((lags[k] + frac) * step), "motion", conf)


# ============================== 相机模型 ==============================

@dataclass
class Camera:
    K: np.ndarray                 # 3×3 内参
    R: np.ndarray = field(default_factory=lambda: np.eye(3))   # 参考系 → 相机
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.c_[self.R, self.t]

    def project(self, X: np.ndarray) -> np.ndarray:
        Xc = X @ self.R.T + self.t
        x = Xc @ self.K.T
        return x[..., :2] / x[..., 2:3]

    def to_dict(self):
        return dict(K=self.K.round(3).tolist(), R=self.R.round(6).tolist(), t=self.t.round(6).tolist())


def default_K(w: int, h: int, focal: float | None = None) -> np.ndarray:
    """手机主摄（1×）视频的默认内参：焦距约 0.78 × 长边像素，主点在画面中心。"""
    f = focal or 0.78 * max(w, h)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def _rodrigues(r):
    return cv2.Rodrigues(np.asarray(r, float).reshape(3, 1))[0]


def triangulate(cams: list[Camera], obs: list[np.ndarray], weights: list[np.ndarray] | None = None) -> np.ndarray:
    """加权 DLT 三角化。obs[i] 形状 (..., 2)，NaN 表示该视角没看到；少于两个视角的点为 NaN。"""
    shape = obs[0].shape[:-1]
    n = int(np.prod(shape))
    rows, wts, valid = [], [], []
    for i, (cam, x) in enumerate(zip(cams, obs)):
        x = x.reshape(n, 2)
        w = np.ones(n) if weights is None else np.nan_to_num(weights[i].reshape(n))
        ok = np.isfinite(x).all(axis=1) & (w > 0)
        P = cam.P
        r1 = x[:, 0:1] * P[2] - P[0]
        r2 = x[:, 1:2] * P[2] - P[1]
        r1 = r1 / (np.linalg.norm(r1, axis=1, keepdims=True) + 1e-12)
        r2 = r2 / (np.linalg.norm(r2, axis=1, keepdims=True) + 1e-12)
        rows += [r1, r2]
        wts += [w * ok, w * ok]
        valid.append(ok)
    A = np.stack([np.nan_to_num(r) * wt[:, None] for r, wt in zip(rows, wts)], axis=1)  # (n, 2V, 4)
    _, _, vt = np.linalg.svd(A)
    X = vt[:, -1, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        X = X[:, :3] / X[:, 3:4]
    nview = np.sum(valid, axis=0)
    X[nview < 2] = np.nan
    return X.reshape(shape + (3,))


# ============================== 标定 ==============================

def _correspondences(img_a, img_b, min_vis, max_frames=400):
    """挑出两台都看得清的 (帧, 关节) 对，均匀抽帧以控制规模。"""
    ok = (img_a[..., 2] >= min_vis) & (img_b[..., 2] >= min_vis) \
        & np.isfinite(img_a[..., :2]).all(-1) & np.isfinite(img_b[..., :2]).all(-1)
    ok[:, [j for j in range(33) if j not in JOINTS]] = False
    frames = np.flatnonzero(ok.sum(axis=1) >= 8)
    if len(frames) > max_frames:
        frames = frames[np.linspace(0, len(frames) - 1, max_frames).astype(int)]
    fi, ji = np.nonzero(ok[frames])
    fi = frames[fi]
    return fi, ji, img_a[fi, ji, :2], img_b[fi, ji, :2], np.minimum(img_a[fi, ji, 2], img_b[fi, ji, 2])


def _bone_pairs(fi, ji):
    """在 BA 用到的 (帧, 关节) 点里找出同一帧内的骨段端点对，返回 (K,3)：点 i、点 j、骨长组。"""
    index = {(int(f), int(j)): k for k, (f, j) in enumerate(zip(fi, ji))}
    out = []
    for f in np.unique(fi):
        for gi, group in enumerate(BONE_GROUPS):
            for a, b in group:
                ka, kb = index.get((int(f), a)), index.get((int(f), b))
                if ka is not None and kb is not None:
                    out.append((ka, kb, gi))
    return np.array(out, dtype=int).reshape(-1, 3)


def _refine_cameras(cam_a: Camera, cam_b: Camera, xa, xb, w, refine_focal=True,
                    anchors=None, bones=None, bone_weight=20.0):
    """相机参数优化（变量投影法）：只优化 B 的位姿和两台焦距，3D 点每次由当前相机重新三角化。

    比把几千个 3D 点一起放进优化更不容易陷在初值附近。残差 = 两台相机的重投影误差（Huber）
    + 骨长一致性（同一骨段在各帧长度相同、左右对称） + 焦距弱先验。

    anchors：可选 dict(Xw=(M,3) 球场点, xa=(M,2), xb=(M,2), rw, tw)，世界→A 的变换一起优化，
             此时 B 的平移不归一化，尺度由球场确定。
    """
    f0a, f0b = cam_a.K[0, 0], cam_b.K[0, 0]
    use_anchor = anchors is not None
    t0 = cam_b.t.astype(float)
    tn = np.linalg.norm(t0)
    sw = np.sqrt(np.clip(w, 0.05, 1.0))[:, None]

    if use_anchor:
        x0 = np.r_[cv2.Rodrigues(cam_b.R)[0].ravel(), t0, f0a, f0b, anchors["rw"], anchors["tw"]]
    else:  # 平移只保留方向（球面坐标），长度固定，尺度另定
        d = t0 / tn
        x0 = np.r_[cv2.Rodrigues(cam_b.R)[0].ravel(), np.arctan2(d[1], d[0]), np.arccos(np.clip(d[2], -1, 1)), f0a, f0b]

    def unpack(p):
        rb = p[0:3]
        if use_anchor:
            tb, fa, fb = p[3:6], p[6], p[7]
        else:
            th, ph = p[3], p[4]
            tb = tn * np.array([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)])
            fa, fb = p[5], p[6]
        if not refine_focal:
            fa, fb = f0a, f0b
        Ka, Kb = cam_a.K.copy(), cam_b.K.copy()
        Ka[0, 0] = Ka[1, 1] = fa
        Kb[0, 0] = Kb[1, 1] = fb
        return Camera(Ka), Camera(Kb, _rodrigues(rb), tb)

    def resid(p):
        ca, cb = unpack(p)
        X = triangulate([ca, cb], [xa, xb])
        ra = np.nan_to_num((ca.project(X) - xa) * sw, nan=50.0)
        rb_ = np.nan_to_num((cb.project(X) - xb) * sw, nan=50.0)
        out = [ra.ravel(), rb_.ravel()]
        if use_anchor:
            Rw, tw = _rodrigues(p[8:11]), p[11:14]
            Xa = anchors["Xw"] @ Rw.T + tw
            out += [(ca.project(Xa) - anchors["xa"]).ravel() * 3, (cb.project(Xa) - anchors["xb"]).ravel() * 3]
        if bones is not None and len(bones):
            dl = np.linalg.norm(X[bones[:, 0]] - X[bones[:, 1]], axis=1)
            rel = np.zeros(len(bones))
            for gi in np.unique(bones[:, 2]):
                m = bones[:, 2] == gi
                med = np.nanmedian(dl[m])
                rel[m] = dl[m] / med - 1.0 if med > 0 else 0.0
            out.append(np.nan_to_num(rel) * bone_weight)
        if refine_focal:
            out.append([(ca.K[0, 0] / f0a - 1) / 0.25, (cb.K[0, 0] / f0b - 1) / 0.25])
        return np.concatenate(out)

    res = least_squares(resid, x0, loss="huber", f_scale=2.0, x_scale="jac", method="trf",
                        diff_step=1e-6, max_nfev=300)
    ca, cb = unpack(res.x)
    out = dict(cam_a=ca, cam_b=cb, cost=float(res.cost), nfev=int(res.nfev))
    if use_anchor:
        out["Rw"], out["tw"] = _rodrigues(res.x[8:11]), res.x[11:14]
    return out


def segment_stature(X: np.ndarray) -> float:
    """由 3D 关键点估身高：脚踝→膝→髋→肩→鼻 的节段长度之和 ÷ 0.88（人体测量比例，约 ±5%）。"""
    def seg(a, b):
        return np.linalg.norm(X[:, a] - X[:, b], axis=-1)
    mid = lambda a, b: (X[:, a] + X[:, b]) / 2  # noqa: E731
    legs = []
    for an, kn, hp in ((L.L_ANKLE, L.L_KNEE, L.L_HIP), (L.R_ANKLE, L.R_KNEE, L.R_HIP)):
        legs.append(seg(an, kn) + seg(kn, hp))
    leg = np.nanmedian(np.stack(legs), axis=0)
    trunk = np.linalg.norm(mid(L.L_SHOULDER, L.R_SHOULDER) - mid(L.L_HIP, L.R_HIP), axis=-1)
    head = np.linalg.norm(X[:, L.NOSE] - mid(L.L_SHOULDER, L.R_SHOULDER), axis=-1)
    total = leg + trunk + head
    return float(np.nanmedian(total) / 0.88)


@dataclass
class StereoCalib:
    cam_a: Camera
    cam_b: Camera
    mode: str                          # "auto" / "court"
    scale: float                       # 把三角化结果换算成米的系数（已乘入 cam_b.t）
    reproj_px: dict
    world_from_a: np.ndarray | None = None   # 4×4，A 坐标 → 角度模块用的世界坐标（球场模式）
    notes: list = field(default_factory=list)

    def to_dict(self):
        return dict(mode=self.mode, cam_a=self.cam_a.to_dict(), cam_b=self.cam_b.to_dict(),
                    scale=self.scale, reproj_px=self.reproj_px, notes=self.notes,
                    world_from_a=None if self.world_from_a is None else self.world_from_a.round(6).tolist())


def calibrate(img_a: np.ndarray, img_b: np.ndarray, size_a, size_b, focal_a=None, focal_b=None,
              height_m: float = 1.75, court=None, min_vis: float = 0.6, refine_focal: bool = True) -> StereoCalib:
    """img_a / img_b：已经对齐到同一时间轴的 2D 关键点 (T,33,3)。

    court：可选 dict(m=(M,2) 球场坐标米, a=(M,2) A 画面像素, b=(M,2) B 画面像素)。
    """
    cam_a = Camera(default_K(*size_a, focal_a))
    cam_b = Camera(default_K(*size_b, focal_b))
    fi, ji, xa, xb, w = _correspondences(img_a, img_b, min_vis)
    if len(xa) < 60:
        raise RuntimeError(f"两台手机同时看清的关键点只有 {len(xa)} 个，无法标定。检查两段视频是否拍的是同一时段、同一个人")
    notes = []
    anchors = None
    if court is not None:
        Xw = np.c_[court["m"], np.zeros(len(court["m"]))]  # 球场平面 z=0（z 向上）
        poses = []
        for cam, px in ((cam_a, court["a"]), (cam_b, court["b"])):
            ok, rv, tv = cv2.solvePnP(Xw, px.astype(float), cam.K, None, flags=cv2.SOLVEPNP_IPPE)
            if not ok:
                raise RuntimeError("球场点 PnP 失败，检查点的顺序是否一致")
            poses.append((_rodrigues(rv), tv.ravel()))
        (Ra, ta), (Rb, tb) = poses
        cam_b.R, cam_b.t = Rb @ Ra.T, tb - Rb @ Ra.T @ ta
        anchors = dict(Xw=Xw, xa=court["a"].astype(float), xb=court["b"].astype(float),
                       rw=cv2.Rodrigues(Ra)[0].ravel(), tw=ta)
    else:
        na = cv2.undistortPoints(xa.reshape(-1, 1, 2), cam_a.K, None).reshape(-1, 2)
        nb = cv2.undistortPoints(xb.reshape(-1, 1, 2), cam_b.K, None).reshape(-1, 2)
        thr = 3.0 / cam_a.K[0, 0]
        E, inl = cv2.findEssentialMat(na, nb, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=thr)
        if E is None:
            raise RuntimeError("本质矩阵估计失败：两台手机的视角可能太接近，建议夹角 ≥ 30°")
        E = E[:3]
        _, R, t, _ = cv2.recoverPose(E, na, nb, np.eye(3), mask=inl)
        cam_b.R, cam_b.t = R, t.ravel()
        notes.append(f"自标定：本质矩阵内点 {int(inl.sum())}/{len(inl)}")

    X0 = triangulate([cam_a, cam_b], [xa, xb])
    good = np.isfinite(X0).all(axis=1) & (X0[:, 2] > 0)
    bones = _bone_pairs(fi[good], ji[good])
    ba = _refine_cameras(cam_a, cam_b, xa[good], xb[good], w[good],
                         refine_focal=refine_focal, anchors=anchors, bones=bones)
    cam_a, cam_b = ba["cam_a"], ba["cam_b"]
    world_from_a = None
    if court is not None:
        scale = 1.0
        # A 坐标 → 球场坐标 (x, y, z上) → 角度模块约定 (x, -z, y)
        Rw, tw = ba["Rw"], ba["tw"]
        to_court = np.eye(4)
        to_court[:3, :3], to_court[:3, 3] = Rw.T, -Rw.T @ tw
        conv = np.eye(4)
        conv[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
        world_from_a = conv @ to_court
        notes.append("球场标定：3D 在球场坐标系（米），重力方向由球场平面确定")
    else:
        # 尺度：用身高换算（只影响距离和速度）
        Xall = triangulate([cam_a, cam_b], [img_a[..., :2], img_b[..., :2]])
        est = segment_stature(Xall)
        scale = float(height_m / est) if np.isfinite(est) and est > 0 else 1.0
        cam_b = Camera(cam_b.K, cam_b.R, cam_b.t * scale)
        notes.append(f"尺度按身高 {height_m:.2f} m 换算（只影响距离与速度，不影响角度）")
        notes.append("重力方向取手机 A 的竖直方向：录制时 A 需保持水平")

    # 重投影误差
    X = triangulate([cam_a, cam_b], [xa, xb])
    ea = np.linalg.norm(cam_a.project(X) - xa, axis=1)
    eb = np.linalg.norm(cam_b.project(X) - xb, axis=1)
    rep = dict(a_median=float(np.nanmedian(ea)), a_p95=float(np.nanpercentile(ea, 95)),
               b_median=float(np.nanmedian(eb)), b_p95=float(np.nanpercentile(eb, 95)),
               n_points=int(len(xa)))
    angle_ab = float(np.degrees(np.arccos(np.clip(cam_b.R[2, 2], -1, 1))))
    notes.append(f"两机光轴夹角约 {angle_ab:.0f}°")
    return StereoCalib(cam_a, cam_b, "court" if court is not None else "auto", scale, rep, world_from_a, notes)


# ============================== 总流程 ==============================

def align_b_to_a(pa: PoseSeries, pb: PoseSeries, offset: float) -> np.ndarray:
    """返回在 A 每一帧时刻对应的 B 关键点 (T_A, 33, 3)。"""
    tb = pa.t + offset
    return _resample(pb.img, pb.fps, tb)


def stereo_series(pa: PoseSeries, pb_on_a: np.ndarray, calib: StereoCalib, min_vis: float = 0.5) -> PoseSeries:
    """三角化每一帧，返回 img = A 的 2D、world = 3D（米，x 右 y 下 z 前，右手系）的 PoseSeries。"""
    wa = np.where(pa.img[..., 2] >= min_vis, pa.img[..., 2], 0.0)
    wb = np.where(np.nan_to_num(pb_on_a[..., 2]) >= min_vis, pb_on_a[..., 2], 0.0)
    X = triangulate([calib.cam_a, calib.cam_b], [pa.img[..., :2], pb_on_a[..., :2]], [wa, wb])
    if calib.world_from_a is not None:
        M = calib.world_from_a
        X = X @ M[:3, :3].T + M[:3, 3]
    # 重投影误差过大的点（遮挡、左右腿识别反了）置 NaN
    Xa = X if calib.world_from_a is None else (X - calib.world_from_a[:3, 3]) @ calib.world_from_a[:3, :3]
    ea = np.linalg.norm(calib.cam_a.project(Xa) - pa.img[..., :2], axis=-1)
    eb = np.linalg.norm(calib.cam_b.project(Xa) - pb_on_a[..., :2], axis=-1)
    lim = 6 * max(calib.reproj_px["a_median"], calib.reproj_px["b_median"], 1.5)
    X[(ea > lim) | (eb > lim)] = np.nan
    return PoseSeries(pa.fps, pa.width, pa.height, pa.img.copy(), X)


def ground_feet(ps3d: PoseSeries, calib: StereoCalib):
    """双脚在地面上的位置（米）。球场模式为球场 (x, y)；自标定模式为 A 视角下的 (x 右, z 前)。"""
    # 两种模式下世界坐标都是 (x, y 向下, z)，地面平面即 (x, z)：
    # 球场模式 z = 球场 y；自标定模式 z = A 的纵深方向
    W = ps3d.world
    return W[:, L.L_ANKLE][:, [0, 2]], W[:, L.R_ANKLE][:, [0, 2]]
