"""命令行 stereo 子命令冒烟测试：用合成的两机关键点代替视频（--reuse 跳过姿态估计）。"""
import json

import numpy as np
import pytest

from hoopcv.cli import main
from hoopcv.pose import PoseSeries

from test_stereo import CAM_A, CAM_B, FPS, OFFSET, W, H, F, render


@pytest.fixture(scope="module")
def outdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("stereo")
    ta = np.arange(0, 4.0, 1 / FPS)
    tb = np.arange(0, 4.2, 1 / FPS)
    PoseSeries(FPS, W, H, render(CAM_A, ta), np.zeros((len(ta), 33, 3))).save(d / "pose_a.npz")
    PoseSeries(FPS, W, H, render(CAM_B, tb - OFFSET), np.zeros((len(tb), 33, 3))).save(d / "pose_b.npz")
    return d


@pytest.mark.parametrize("task", ["shot", "angles", "drive"])
def test_stereo_cli(outdir, task):
    main(["stereo", str(outdir / "a.mp4"), str(outdir / "b.mp4"), "-o", str(outdir), "--reuse", "--no-video",
          "--task", task, "--focal-a", str(F), "--focal-b", str(F)])
    info = json.loads((outdir / "stereo.json").read_text(encoding="utf-8"))
    assert info["sync"]["method"] == "motion"
    assert info["sync"]["offset_s"] == pytest.approx(OFFSET, abs=0.5 / FPS)
    assert info["calibration"]["reproj_px"]["a_median"] < 2.0
    assert info["valid_3d"] > 0.8
    sk = json.loads((outdir / "skeleton3d.json").read_text(encoding="utf-8"))
    assert len(sk["joints"]) == 23 and len(sk["frames"]) == int(4.0 * FPS)
    # three.js 约定 y 向上：头比脚踝高
    fr = next(f for f in sk["frames"] if f[0] is not None and f[sk["joints"].index("l_ankle")] is not None)
    assert fr[0][1] > fr[sk["joints"].index("l_ankle")][1]
    if task == "shot":
        assert (outdir / "shots.csv").exists() and (outdir / "report.md").exists()
        rep = (outdir / "report.md").read_text(encoding="utf-8")
        assert "双机 3D" in rep and "近似" not in rep
    if task == "drive":
        d = json.loads((outdir / "drive.json").read_text(encoding="utf-8"))
        assert d["unit"] == "m"
