import numpy as np
import pytest

from hoopcv import geometry as g

X, Y, Z = np.eye(3)


def arm(elbow_dir, forearm_dir):
    sh = np.zeros(3)
    el = sh + elbow_dir
    wr = el + forearm_dir
    return sh, el, wr


def test_flexion():
    assert g.flexion(np.array([0, 1.0]), np.zeros(2), np.array([0, -1.0])) == pytest.approx(0)
    assert g.flexion(np.array([0, 1.0]), np.zeros(2), np.array([1.0, 0])) == pytest.approx(90)


@pytest.mark.parametrize("elbow_dir, forearm_dir, elev, plane, er", [
    (Z, X, 90, 0, 0),      # 侧平举，前臂朝前：中立
    (Z, Y, 90, 0, 90),     # 侧平举，前臂朝上：外旋 90
    (Z, -Y, 90, 0, -90),   # 侧平举，前臂朝下：内旋 90
    (X, Y, 90, 90, 0),     # 前平举（投篮准备姿势），前臂朝上：中立
    (X, -Z, 90, 90, -90),  # 前平举，前臂向内：内旋
    (X, Z, 90, 90, 90),    # 前平举，前臂向外：外旋
    (-Y, X, 0, None, 0),   # 手臂自然下垂、前臂朝前
    (-Y, Z, 0, None, 90),  # 手臂下垂、前臂朝外：外旋 90
])
def test_shoulder_rotation(elbow_dir, forearm_dir, elev, plane, er):
    sh, el, wr = arm(elbow_dir, forearm_dir)
    r = g.limb_rotation(sh, el, wr)
    assert r["elevation"] == pytest.approx(elev, abs=1e-6)
    if plane is not None:
        assert r["plane"] == pytest.approx(plane, abs=1e-6)
    assert r["ext_rot"] == pytest.approx(er, abs=1e-6)


def test_shoulder_rotation_45deg_elevation():
    # 前上举 45°，前臂外旋 30°
    e = np.cos(np.pi / 4) * X - np.sin(np.pi / 4) * Y
    up = np.cos(np.pi / 4) * Y + np.sin(np.pi / 4) * X  # 与上臂垂直、在矢状面内
    a = np.radians(30)
    f = np.cos(a) * up + np.sin(a) * Z
    r = g.limb_rotation(*arm(e, f))
    assert r["elevation"] == pytest.approx(45, abs=1e-6)
    assert r["ext_rot"] == pytest.approx(30, abs=1e-6)


def test_straight_elbow_rotation_undefined():
    r = g.limb_rotation(*arm(X, X))
    assert np.isnan(r["ext_rot"])


def test_hip_rotation_seated():
    hip = np.zeros(3)
    knee = X.copy()
    # 坐姿中立：小腿竖直向下
    r = g.limb_rotation(hip, knee, 2 * knee - (knee - Y))
    assert r["ext_rot"] == pytest.approx(0, abs=1e-6)
    # 脚向外摆 = 髋内旋
    r = g.limb_rotation(hip, knee, 2 * knee - (knee + Z))
    assert r["ext_rot"] == pytest.approx(-90, abs=1e-6)


def test_forearm_pronation():
    sh, el, wr = np.zeros(3), -Y, -Y + X  # 上臂下垂，肘屈 90，前臂朝前
    # 握手位：食指在小指上方
    assert g.forearm_pronation(sh, el, wr, wr + 0.05 * Y, wr) == pytest.approx(0, abs=1e-6)
    # 掌心向下：食指转向内侧（-Z）
    assert g.forearm_pronation(sh, el, wr, wr - 0.05 * Z, wr) == pytest.approx(90, abs=1e-6)
    assert g.forearm_pronation(sh, el, wr, wr + 0.05 * Z, wr) == pytest.approx(-90, abs=1e-6)


def test_thorax_frame_identity():
    ls, rs = np.array([0, 1.0, -0.2]), np.array([0, 1.0, 0.2])
    lh, rh = np.array([0, 0, -0.1]), np.array([0, 0, 0.1])
    o, R = g.thorax_frame(ls, rs, lh, rh)
    assert np.allclose(R, np.eye(3))


def test_fill_and_lowpass_keep_long_gaps():
    x = np.sin(np.linspace(0, 6, 200))[:, None]
    x[50:52] = np.nan
    x[100:150] = np.nan
    f = g.fill_short_gaps(x, 5)
    assert not np.isnan(f[50:52]).any()
    assert np.isnan(f[100:150]).all()
    s = g.lowpass(f, 100, 8)
    assert np.isnan(s[100:150]).all()
    assert np.allclose(s[:40], f[:40], atol=0.05)
