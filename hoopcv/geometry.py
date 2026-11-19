"""几何与滤波工具：夹角、坐标系、YXY 欧拉分解、Butterworth 平滑。

所有角度函数都接受形状为 (..., 3) 或 (..., 2) 的数组，可整段时间序列向量化计算。
"""
from __future__ import annotations

import numpy as np


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / n


def wrap180(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """两向量夹角（度），0-180。"""
    c = np.sum(unit(u) * unit(v), axis=-1)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """∠abc（度），b 为顶点。"""
    return angle_between(a - b, c - b)


def flexion(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """关节屈曲角 = 180 - ∠abc。完全伸直为 0。"""
    return 180.0 - joint_angle(a, b, c)


def signed_angle_2d(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """平面内从 ref 转到 v 的有符号角（度），图像坐标系（y 向下）下顺时针为正。"""
    cross = ref[..., 0] * v[..., 1] - ref[..., 1] * v[..., 0]
    dot = np.sum(ref * v, axis=-1)
    return np.degrees(np.arctan2(cross, dot))


def thorax_frame(l_sh, r_sh, l_hip, r_hip):
    """躯干坐标系（ISB 风格）：Y 向上（髋中点→肩中点），Z 指向被拍者右侧，X = Y×Z 指向前方。

    返回 origin (T,3) 与旋转矩阵 R (T,3,3)，R 的三列分别是 X, Y, Z 轴。
    """
    mid_sh = (l_sh + r_sh) / 2
    mid_hip = (l_hip + r_hip) / 2
    y = unit(mid_sh - mid_hip)
    z0 = r_sh - l_sh
    z = unit(z0 - np.sum(z0 * y, axis=-1, keepdims=True) * y)
    x = np.cross(y, z)
    return mid_sh, np.stack([x, y, z], axis=-1)


def pelvis_frame(l_sh, r_sh, l_hip, r_hip):
    """骨盆坐标系：原点髋中点，Z 指向右髋，Y 由躯干向上方向正交化得到，X 向前。"""
    mid_sh = (l_sh + r_sh) / 2
    mid_hip = (l_hip + r_hip) / 2
    z = unit(r_hip - l_hip)
    y0 = mid_sh - mid_hip
    y = unit(y0 - np.sum(y0 * z, axis=-1, keepdims=True) * z)
    x = np.cross(y, z)
    return mid_hip, np.stack([x, y, z], axis=-1)


def to_local(p, origin, R):
    """把全局点转到局部坐标系。"""
    return np.einsum("...ji,...j->...i", R, p - origin)


def euler_yxy(R: np.ndarray):
    """R = Ry(a) Rx(b) Ry(c)，返回 (a, b, c)，单位度，b ∈ [0, 180]。"""
    b = np.degrees(np.arccos(np.clip(R[..., 1, 1], -1, 1)))
    a = np.degrees(np.arctan2(R[..., 0, 1], R[..., 2, 1]))
    c = np.degrees(np.arctan2(R[..., 1, 0], -R[..., 1, 2]))
    return a, b, c


def limb_rotation(prox, mid, dist, min_flex: float = 20.0):
    """近端节段（上臂/大腿）相对母坐标系的 YXY 分解。

    输入为已转到母坐标系（躯干或骨盆）且已镜像成“右侧”的局部坐标：
    prox = 肩/髋，mid = 肘/膝，dist = 用来确定旋转的远端方向点（腕，或膝 - (踝-膝) 的虚拟点）。

    返回 dict：
      elevation   上臂抬高角/大腿与躯干轴的夹角（度）
      plane       抬高平面（0 = 冠状面外展，90 = 矢状面前屈）
      ext_rot     外旋为正、内旋为负（度）；远端关节屈曲 < min_flex 时无法定义，记为 NaN
    """
    yh = unit(prox - mid)
    f = dist - mid
    zh = unit(np.cross(f, yh))
    xh = np.cross(yh, zh)
    H = np.stack([xh, yh, zh], axis=-1)
    a, b, c = euler_yxy(H)
    # 万向节锁（抬高角≈0 或 180）时 a、c 不可分，但 a+c 仍由 R = Ry(a+c) 唯一确定
    lock = np.abs(np.sin(np.radians(b))) < 1e-3
    a_plus_c = np.where(lock, np.degrees(np.arctan2(H[..., 0, 2], H[..., 0, 0])), a + c)
    # 远端方向与近端节段接近共线（肘伸直；髋用虚拟点时为膝伸直）时旋转轴不确定，记为 NaN
    th = angle_between(prox - mid, dist - mid)
    ext = -wrap180(a_plus_c)
    ext = np.where(np.minimum(th, 180.0 - th) >= min_flex, ext, np.nan)
    return dict(elevation=b, plane=wrap180(a + 180.0), ext_rot=ext)


def forearm_pronation(sh, el, wr, index, pinky, min_flex: float = 20.0):
    """前臂旋前角（近似）。握手位（拇指朝上）为 0，掌心朝下为 +90（旋前），掌心朝上为 -90（旋后）。

    输入为躯干局部坐标且已镜像成右侧。MediaPipe 手指点较粗糙，此值仅供趋势参考。
    """
    yf = unit(el - wr)                       # 前臂长轴，指向近端
    zh = unit(np.cross(wr - el, unit(sh - el)))  # 肘屈伸轴，指向外侧（右）
    r0 = unit(sh - el)
    r0 = unit(r0 - np.sum(r0 * yf, axis=-1, keepdims=True) * yf)  # 参考：上臂方向在前臂横截面上的投影
    v = index - pinky
    v = v - np.sum(v * yf, axis=-1, keepdims=True) * yf
    pron = np.degrees(np.arctan2(np.sum(v * -zh, axis=-1), np.sum(v * r0, axis=-1)))
    flex = 180.0 - angle_between(sh - el, wr - el)
    return np.where(flex >= min_flex, pron, np.nan)


# ---------------- 滤波 ----------------

def fill_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """沿第 0 维对长度 ≤ max_gap 的 NaN 段做线性插值，更长的缺口保留 NaN。"""
    x = np.array(x, dtype=float, copy=True)
    flat = x.reshape(x.shape[0], -1)
    idx = np.arange(flat.shape[0])
    for j in range(flat.shape[1]):
        col = flat[:, j]
        bad = np.isnan(col)
        if bad.all() or not bad.any():
            continue
        good = ~bad
        interp = np.interp(idx, idx[good], col[good])
        # 找出每段 NaN 的长度
        run_start = None
        for i in range(len(col) + 1):
            b = i < len(col) and bad[i]
            if b and run_start is None:
                run_start = i
            elif not b and run_start is not None:
                inner = run_start > 0 and i < len(col)
                if inner and (i - run_start) <= max_gap:
                    col[run_start:i] = interp[run_start:i]
                run_start = None
        flat[:, j] = col
    return flat.reshape(x.shape)


def lowpass(x: np.ndarray, fps: float, cutoff: float = 8.0, order: int = 4) -> np.ndarray:
    """零相位 Butterworth 低通（沿第 0 维），保留原 NaN 位置。cutoff ≥ fps/2 时不滤波。"""
    from scipy.signal import butter, filtfilt

    if cutoff <= 0 or cutoff >= fps / 2:
        return x
    b, a = butter(order, cutoff / (fps / 2))
    out = np.array(x, dtype=float, copy=True)
    flat = out.reshape(out.shape[0], -1)
    padlen = 3 * max(len(a), len(b))
    for j in range(flat.shape[1]):
        col = flat[:, j]
        good = ~np.isnan(col)
        if good.sum() <= padlen:
            continue
        # 对每个连续有效段分别滤波
        edges = np.flatnonzero(np.diff(np.r_[0, good.astype(int), 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            if e - s > padlen:
                col[s:e] = filtfilt(b, a, col[s:e])
        flat[:, j] = col
    return flat.reshape(out.shape)
