"""命令行入口：

  hoopcv angles 视频.mp4 --side right -o 输出目录
  hoopcv shot   视频.mp4 --side right --ball --made 1,0,1,1 -o 输出目录
  hoopcv drive  视频.mp4 --court-px "x,y;x,y;x,y;x,y" --court-m "x,y;..." --basket "x,y" -o 输出目录
  hoopcv stereo 手机A.mp4 手机B.mp4 --task shot --focal-a 1450 --focal-b 1450 --height 1.83 -o 输出目录
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import angles as A
from .pose import PoseSeries, extract

L_CORE = [0] + list(range(11, 29))  # 鼻子 + 躯干四肢（不含手指、脚跟脚尖）


def _common(p: argparse.ArgumentParser):
    p.add_argument("video", help="输入视频")
    p.add_argument("-o", "--out", default=None, help="输出目录（默认：视频同名文件夹）")
    p.add_argument("--side", default="right", choices=["right", "left"], help="分析哪一侧（投篮手）")
    p.add_argument("--model", default="heavy", choices=["lite", "full", "heavy"], help="MediaPipe 模型大小")
    p.add_argument("--fps", type=float, default=None,
                   help="覆盖视频帧率。手机慢动作导出后常被标成 30 fps，此时要填真实拍摄帧率（如 240）")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270], help="旋转画面")
    p.add_argument("--cutoff", type=float, default=8.0, help="低通滤波截止频率 Hz（0 = 不滤波）")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--no-video", action="store_true", help="不输出标注视频")
    p.add_argument("--reuse", action="store_true", help="复用输出目录里已有的 pose.npz，跳过姿态估计")


def _load_pose(video, cache: Path, args, rotate: int, label: str = "") -> PoseSeries:
    if args.reuse and cache.exists():
        return PoseSeries.load(cache)
    print(f"姿态估计中{label}…")
    raw = extract(video, args.model, args.max_frames, rotate)
    raw.save(cache)
    return raw


def _prepare(args):
    out = Path(args.out or Path(args.video).with_suffix(""))
    out.mkdir(parents=True, exist_ok=True)
    raw = _load_pose(args.video, out / "pose.npz", args, args.rotate)
    if args.fps:
        raw.fps = float(args.fps)
    valid = np.isfinite(raw.img[:, 0, 0]).mean()
    print(f"{raw.n_frames} 帧，{raw.fps:.1f} fps，检测到人的帧占 {valid:.0%}")
    if valid < 0.5:
        print("⚠ 一半以上的帧没检测到人，检查机位、光线或 --rotate")
    ps = A.smooth_series(raw, cutoff=args.cutoff)
    ang = A.compute_angles(ps, args.side)
    ang.round(2).to_csv(out / "angles.csv", index=False)
    return out, ps, ang


def _fmt(v, nd=1):
    if isinstance(v, (float, np.floating)):
        return "—" if not np.isfinite(v) else f"{v:.{nd}f}"
    return str(v)


def cmd_angles(args):
    from . import viz
    out, ps, ang = _prepare(args)
    viz.plot_angles(ang, str(out / "angles.png"))
    if not args.no_video:
        viz.annotate_video(args.video, str(out / "annotated.mp4"), ps, ang, rotate=args.rotate, side=args.side)
    desc = ang.drop(columns=["frame", "t"]).describe().T[["mean", "min", "max"]].round(1)
    desc.insert(0, "说明", [A.COLUMN_NOTES.get(c, "") for c in desc.index])
    print(desc.to_string())
    print(f"\n输出：{out}")


SHOT_REPORT_COLS = [
    ("release_angle_deg", "出手角 (°)"), ("release_angle_source", "出手角来源"),
    ("release_speed_mps", "出手速度 (m/s)"), ("release_height_stature", "出手高度 (身高倍数)"),
    ("dip_to_release_s", "下蹲→出手 (s)"), ("jump_height_stature", "起跳高度 (身高倍数)"),
    ("knee_flex_dip", "下蹲膝屈曲 (°)"), ("hip_flex_dip", "下蹲髋屈曲 (°)"),
    ("elbow_flex_set", "举球肘屈曲 (°)"), ("shoulder_ext_rot_set", "举球肩外旋 (°, 近似)"),
    ("elbow_flex_release", "出手肘屈曲 (°)"), ("shoulder_flex_release", "出手上臂-躯干角 (°)"),
    ("forearm_pron_release", "出手前臂旋前 (°, 粗略)"), ("trunk_lean_release", "出手躯干前倾 (°)"),
    ("wrist_flex_follow", "随挥压腕 (°)"), ("chain_order", "动力链峰值顺序"),
]


def cmd_shot(args):
    from . import shot as S
    from . import viz
    out, ps, ang = _prepare(args)
    ball = None
    if args.ball:
        from .ball import detect_ball
        s = __import__("hoopcv.landmarks", fromlist=["side_ids"]).side_ids(args.side)
        print("检测篮球中…")
        ball = detect_ball(args.video, args.ball_weights, args.ball_class, rotate=args.rotate,
                           max_frames=args.max_frames, hand_xy=ps.img[:, s["wrist"], :2])
        np.save(out / "ball.npy", ball)
        print(f"球检出率 {np.isfinite(ball[:, 0]).mean():.0%}")
    made = [int(x) for x in args.made.split(",")] if args.made else None
    shots, events = S.analyze_shots(ps, ang, args.side, ball, made)
    if shots.empty:
        print("没有检测到投篮（判据：投篮手腕高于鼻子）。确认机位能看到完整出手，或换 --side。")
        return
    shots.round(3).to_csv(out / "shots.csv", index=False)
    ev_frames, ev_lines = {}, []
    for e in events:
        for k in ("dip", "set", "release", "follow"):
            ev_frames[e[k]] = f"#{e['shot']} {k}"
            ev_lines.append((e[k] / ps.fps, f"#{e['shot']} {k}"))
    viz.plot_angles(ang, str(out / "angles.png"), ev_lines)
    if not args.no_video:
        viz.annotate_video(args.video, str(out / "annotated.mp4"), ps, ang, ev_frames, ball,
                           rotate=args.rotate, side=args.side)

    _shot_report(shots, Path(args.video).name, ps.fps, args.side, out, stereo=False)


def _shot_report(shots, title, fps, side, out, stereo=False):
    from . import shot as S
    lines = [f"# 投篮分析：{title}", "",
             f"共 {len(shots)} 次投篮，分析侧：{side}，帧率 {fps:.0f} fps，"
             + ("双机 3D。" if stereo else "单机。"), ""]
    hdr = "| 指标 | " + " | ".join(f"#{i}" for i in shots["shot"]) + " | 均值 |"
    lines += [hdr, "|" + " --- |" * (len(shots) + 2)]
    for col, name in SHOT_REPORT_COLS:
        if col not in shots:
            continue
        if stereo:
            name = name.replace("°, 近似", "°").replace("°, 粗略", "°")
        vals = shots[col].tolist()
        mean = shots[col].mean() if pd.api.types.is_numeric_dtype(shots[col]) else ""
        lines.append(f"| {name} | " + " | ".join(_fmt(v) for v in vals) + f" | {_fmt(mean)} |")
    cmp = S.compare_made_missed(shots)
    if cmp is not None:
        lines += ["", "## 进球 vs 不进（按效应量排序，前 8 项）", "",
                  "| 指标 | 进球均值 | 不进均值 | 差值 | 效应量 |", "| --- | --- | --- | --- | --- |"]
        for k, r in cmp.head(8).iterrows():
            lines.append(f"| {k} | {_fmt(r.made_mean)} | {_fmt(r.missed_mean)} | {_fmt(r['diff'])} | {_fmt(r.effect, 2)} |")
    lines += ["", "参考：文献中三分出手角进球约 53°、不进约 48°（Cabarkapa et al. 2022）。"]
    lines.append("肩外旋、前臂旋前来自双机三角化。" if stereo
                 else "肩外旋和前臂旋前来自单目 3D，只适合看同一人多次投篮之间的趋势。")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n输出：{out}")
    return lines


def cmd_drive(args):
    from . import drive as D
    from . import viz
    out, ps, ang = _prepare(args)
    cpx, cm = D.parse_points(args.court_px), D.parse_points(args.court_m)
    basket = D.parse_points(args.basket)
    basket = basket[0] if basket is not None else None
    summary, cuts, track = D.analyze_drive(ps, ang, cpx, cm, basket, args.start_speed, args.cut_angle)
    track.round(4).to_csv(out / "track.csv", index=False)
    if not cuts.empty:
        cuts.round(3).to_csv(out / "cuts.csv", index=False)
    (out / "drive.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    viz.plot_track(track, summary, cuts, str(out / "track.png"))
    ev = []
    if summary.get("detected"):
        ev.append((summary["t_start"], "start"))
    ev += [(c, "cut") for c in cuts.get("t", [])]
    viz.plot_angles(ang, str(out / "angles.png"), ev)
    if not args.no_video:
        fr = {int(round(t * ps.fps)): name for t, name in ev}
        viz.annotate_video(args.video, str(out / "annotated.mp4"), ps, ang, fr, rotate=args.rotate, side=args.side)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    if not cuts.empty:
        print(cuts.round(2).to_string(index=False))
    print(f"\n输出：{out}")


# ============================== 双机 3D ==============================

MONO_COMPARE = ["shoulder_ext_rot_3d", "forearm_pron_3d", "hip_ext_rot_3d", "shoulder_elev_3d", "elbow_flex_3d"]
JOINT_NAMES = {
    0: "nose", 11: "l_shoulder", 12: "r_shoulder", 13: "l_elbow", 14: "r_elbow", 15: "l_wrist", 16: "r_wrist",
    17: "l_pinky", 18: "r_pinky", 19: "l_index", 20: "r_index", 21: "l_thumb", 22: "r_thumb",
    23: "l_hip", 24: "r_hip", 25: "l_knee", 26: "r_knee", 27: "l_ankle", 28: "r_ankle",
    29: "l_heel", 30: "r_heel", 31: "l_foot", 32: "r_foot",
}


def _sync(args, pa, pb, scale_a, scale_b):
    """返回 (SyncResult, 提示列表)。顺序：手动 → 音频 → 动作互相关兜底。"""
    from . import stereo as ST
    if args.offset is not None:
        return ST.SyncResult(float(args.offset), "manual", float("inf")), []
    notes = []
    motion = ST.sync_motion(pa, pb)
    audio = ST.sync_audio(args.video_a, args.video_b, scale_a, scale_b)
    frame = 1.0 / max(pa.fps, pb.fps)
    if audio is None:
        notes.append("没有可用音轨（或两段慢动作倍率不同），用动作互相关对齐")
        r = motion
    elif audio.confidence < 1.3:
        notes.append(f"音频对齐置信度低（{audio.confidence:.2f}），改用动作互相关")
        r = motion
    else:
        r = audio
        if abs(audio.offset - motion.offset) > 2 * frame:
            notes.append(f"⚠ 音频对齐 {audio.offset * 1000:.1f} ms 与动作对齐 {motion.offset * 1000:.1f} ms 相差超过 2 帧，"
                         "检查两段视频是否同一时段；必要时用 --offset 手动指定")
    r.detail = dict(r.detail, motion_offset=motion.offset, motion_confidence=motion.confidence,
                    audio_offset=None if audio is None else audio.offset,
                    audio_confidence=None if audio is None else audio.confidence)
    if r.method == "motion" and r.confidence < 1.2:
        notes.append(f"⚠ 动作对齐置信度低（{r.confidence:.2f}），建议开拍时拍手，或用 --offset 手动指定")
    return r, notes


def _skeleton_json(ps3d: PoseSeries, calib, sync, path: Path):
    """给 app 的 3D 骨架：three.js 约定（x 右、y 上、z 指向手机 A），单位米。"""
    ids = sorted(JOINT_NAMES)
    W = ps3d.world[:, ids] * np.array([1.0, -1.0, -1.0])
    frames = [[None if not np.isfinite(p).all() else [round(float(v), 4) for v in p] for p in fr] for fr in W]
    edges = [[ids.index(a), ids.index(b)] for a, b in __import__("hoopcv.landmarks", fromlist=["SKELETON"]).SKELETON]
    d = dict(fps=ps3d.fps, unit="m", axes="x 右, y 上, z 指向手机 A（three.js 约定）", mode=calib.mode,
             offset_b_s=sync.offset, joints=[JOINT_NAMES[i] for i in ids], edges=edges, frames=frames)
    path.write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def cmd_stereo(args):
    from . import drive as D
    from . import stereo as ST
    from . import viz
    va = Path(args.video_a)
    out = Path(args.out or va.with_name(va.stem + "_stereo"))
    out.mkdir(parents=True, exist_ok=True)

    pa = _load_pose(args.video_a, out / "pose_a.npz", args, args.rotate, "（手机 A）")
    pb = _load_pose(args.video_b, out / "pose_b.npz", args, args.rotate_b, "（手机 B）")
    # 慢动作：容器帧率 → 真实帧率；音频时间按同样比例换算
    scale_a = scale_b = 1.0
    if args.fps:
        scale_a, pa.fps = pa.fps / args.fps, float(args.fps)
    fps_b = args.fps_b or args.fps
    if fps_b:
        scale_b, pb.fps = pb.fps / fps_b, float(fps_b)
    for name, ps in (("A", pa), ("B", pb)):
        valid = np.isfinite(ps.img[:, 0, 0]).mean()
        print(f"手机 {name}：{ps.n_frames} 帧，{ps.fps:.1f} fps，{ps.width}×{ps.height}，检测到人的帧占 {valid:.0%}")

    sync, notes = _sync(args, pa, pb, scale_a, scale_b)
    print(f"时间对齐（{sync.method}）：B 比 A 晚 {sync.offset * 1000:.1f} ms，置信度 {sync.confidence:.2f}")
    b_on_a = ST.align_b_to_a(pa, pb, sync.offset)

    court = None
    if args.court_m:
        court = dict(m=D.parse_points(args.court_m), a=D.parse_points(args.court_a), b=D.parse_points(args.court_b))
        if court["a"] is None or court["b"] is None or not (len(court["m"]) == len(court["a"]) == len(court["b"]) >= 4):
            raise SystemExit("球场标定需要 --court-m、--court-a、--court-b 各至少 4 个点，顺序一致")
    print("相机标定中…")
    calib = ST.calibrate(pa.img, b_on_a, (pa.width, pa.height), (pb.width, pb.height),
                         focal_a=args.focal_a, focal_b=args.focal_b, height_m=args.height, court=court)
    rep = calib.reproj_px
    print(f"标定（{calib.mode}）：重投影误差中位数 A {rep['a_median']:.2f} px / B {rep['b_median']:.2f} px，"
          f"焦距 A {calib.cam_a.K[0, 0]:.0f} / B {calib.cam_b.K[0, 0]:.0f} px")
    if max(rep["a_median"], rep["b_median"]) > 5:
        notes.append("⚠ 重投影误差偏大：检查对齐、两机是否拍同一人、--focal 是否正确")
    angle_ab = float(np.degrees(np.arccos(np.clip(calib.cam_b.R[2, 2], -1, 1))))
    if angle_ab < 25:
        notes.append(f"⚠ 两机夹角只有 {angle_ab:.0f}°，深度不可靠，建议 ≥ 30°")

    ps3d = ST.stereo_series(pa, b_on_a, calib)
    ps = A.smooth_series(ps3d, cutoff=args.cutoff)
    ang = A.compute_angles(ps, args.side)
    mono = A.compute_angles(A.smooth_series(pa, cutoff=args.cutoff), args.side)
    diff = {}
    for c in MONO_COMPARE:
        ang[c + "_mono"] = mono[c].to_numpy()
        diff[c] = float(np.nanmedian(np.abs(ang[c] - mono[c])))
    ang.round(2).to_csv(out / "angles.csv", index=False)
    ok3d = np.isfinite(ps3d.world[:, L_CORE]).all(-1).mean()
    print(f"3D 有效率（躯干四肢关键点）{ok3d:.0%}")

    _skeleton_json(ps, calib, sync, out / "skeleton3d.json")
    info = dict(video_a=str(args.video_a), video_b=str(args.video_b), fps=pa.fps, side=args.side,
                sync=dict(offset_s=sync.offset, method=sync.method,
                          confidence=sync.confidence if np.isfinite(sync.confidence) else None, **sync.detail),
                calibration=calib.to_dict(), valid_3d=float(ok3d),
                mono_vs_stereo_median_abs_diff_deg=diff, notes=notes + calib.notes)
    (out / "stereo.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    ev_frames, ev_lines, ball = {}, [], None
    if args.task == "shot":
        from . import shot as S
        if args.ball:
            from .ball import detect_ball
            from .landmarks import side_ids
            print("检测篮球中（手机 A）…")
            ball = detect_ball(args.video_a, args.ball_weights, args.ball_class, rotate=args.rotate,
                               max_frames=args.max_frames, hand_xy=ps.img[:, side_ids(args.side)["wrist"], :2])
            np.save(out / "ball.npy", ball)
        made = [int(x) for x in args.made.split(",")] if args.made else None
        shots, events = S.analyze_shots(ps, ang, args.side, ball, made)
        if shots.empty:
            print("没有检测到投篮（判据：投篮手腕高于鼻子）。")
        else:
            shots.round(3).to_csv(out / "shots.csv", index=False)
            for e in events:
                for k in ("dip", "set", "release", "follow"):
                    ev_frames[e[k]] = f"#{e['shot']} {k}"
                    ev_lines.append((e[k] / ps.fps, f"#{e['shot']} {k}"))
            _shot_report(shots, f"{va.name} + {Path(args.video_b).name}", ps.fps, args.side, out, stereo=True)
    elif args.task == "drive":
        basket = D.parse_points(args.basket)
        if basket is not None and calib.mode != "court":
            notes.append("自标定模式没有球场坐标，--basket 被忽略；突破角以手机 A 纵深方向为参考")
            basket = None
        ref = "球场 +y 方向" if calib.mode == "court" else "手机 A 纵深方向"
        summary, cuts, track = D.analyze_drive(ps, ang, basket=None if basket is None else basket[0],
                                               start_speed=args.start_speed, cut_threshold_deg=args.cut_angle,
                                               feet_m=ST.ground_feet(ps, calib), ref_name_m=ref)
        track.round(4).to_csv(out / "track.csv", index=False)
        if not cuts.empty:
            cuts.round(3).to_csv(out / "cuts.csv", index=False)
        (out / "drive.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        viz.plot_track(track, summary, cuts, str(out / "track.png"))
        if summary.get("detected"):
            ev_lines.append((summary["t_start"], "start"))
        ev_lines += [(c, "cut") for c in cuts.get("t", [])]
        ev_frames = {int(round(t * ps.fps)): name for t, name in ev_lines}
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    else:
        desc = ang.drop(columns=["frame", "t"]).describe().T[["mean", "min", "max"]].round(1)
        def note(c):
            if c.endswith("_mono"):
                return A.COLUMN_NOTES.get(c.removesuffix("_mono"), "") + "（单机对照）"
            return A.COLUMN_NOTES.get(c, "").replace("，单目近似", "（双机）").replace("，粗略", "（双机）")
        desc.insert(0, "说明", [note(c) for c in desc.index])
        print(desc.to_string())

    viz.plot_angles(ang, str(out / "angles.png"), ev_lines)
    if not args.no_video:
        viz.annotate_video(args.video_a, str(out / "annotated.mp4"), ps, ang, ev_frames, ball,
                           rotate=args.rotate, side=args.side)
    print("\n单机 vs 双机（中位绝对差，°）：" + "，".join(f"{k} {v:.1f}" for k, v in diff.items()))
    for n in notes:
        print(n)
    print(f"\n输出：{out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hoopcv", description="篮球技术动作视觉分析")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("angles", help="逐帧关节角度（含肩/髋内外旋近似）")
    _common(p)
    p.set_defaults(func=cmd_angles)

    p = sub.add_parser("shot", help="投篮分析：自动切分、出手角、动力链")
    _common(p)
    p.add_argument("--ball", action="store_true", help="用 YOLO 检测篮球，得到出手角/出手速度（需 ultralytics）")
    p.add_argument("--ball-weights", default="yolo11n.pt")
    p.add_argument("--ball-class", default="sports ball")
    p.add_argument("--made", default=None, help="每次投篮是否命中，如 1,0,1,1（用于进/不进对比）")
    p.set_defaults(func=cmd_shot)

    p = sub.add_parser("drive", help="突破分析：第一步方向角、交叉步/顺步、重心、变向")
    _common(p)
    p.add_argument("--court-px", default=None, help='画面里 4 个球场点的像素坐标 "x,y;x,y;x,y;x,y"')
    p.add_argument("--court-m", default=None, help='对应的球场坐标（米），顺序一致')
    p.add_argument("--basket", default=None, help='篮筐位置 "x,y"（有标定为米，否则为像素）')
    p.add_argument("--start-speed", type=float, default=None, help="启动速度阈值（默认 1.2 m/s 或 0.6 身高/s）")
    p.add_argument("--cut-angle", type=float, default=30.0, help="判定变向的方向变化阈值（度）")
    p.set_defaults(func=cmd_drive)

    p = sub.add_parser("stereo", help="双机 3D：时间对齐 + 相机自标定 + 三角化，得到真实的内外旋")
    p.add_argument("video_a", help="手机 A（侧面 90°，参考机，需保持水平）")
    p.add_argument("video_b", help="手机 B（斜前 45° 左右）")
    p.add_argument("-o", "--out", default=None, help="输出目录（默认：A 视频名_stereo）")
    p.add_argument("--task", default="shot", choices=["shot", "angles", "drive"], help="3D 之后做哪种分析")
    p.add_argument("--side", default="right", choices=["right", "left"], help="分析哪一侧（投篮手）")
    p.add_argument("--model", default="heavy", choices=["lite", "full", "heavy"])
    p.add_argument("--fps", type=float, default=None, help="A 的真实拍摄帧率（慢动作时必填）")
    p.add_argument("--fps-b", type=float, default=None, help="B 的真实拍摄帧率（默认同 --fps）")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270], help="旋转 A 的画面")
    p.add_argument("--rotate-b", type=int, default=0, choices=[0, 90, 180, 270], help="旋转 B 的画面")
    p.add_argument("--offset", type=float, default=None, help="手动指定 B 比 A 晚多少秒（跳过自动对齐）")
    p.add_argument("--focal-a", type=float, default=None, help="A 的焦距（像素，app 从相机内参读取；不填则估计）")
    p.add_argument("--focal-b", type=float, default=None, help="B 的焦距（像素）")
    p.add_argument("--height", type=float, default=1.75, help="球员身高（米），自标定模式用来确定尺度")
    p.add_argument("--court-m", default=None, help='球场标定：≥4 个球场点坐标（米）"x,y;x,y;..."')
    p.add_argument("--court-a", default=None, help="这些点在 A 画面里的像素坐标，顺序一致")
    p.add_argument("--court-b", default=None, help="这些点在 B 画面里的像素坐标，顺序一致")
    p.add_argument("--cutoff", type=float, default=8.0, help="低通滤波截止频率 Hz")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--no-video", action="store_true", help="不输出标注视频")
    p.add_argument("--reuse", action="store_true", help="复用输出目录里的 pose_a.npz / pose_b.npz")
    p.add_argument("--ball", action="store_true", help="（shot）在 A 画面里检测篮球")
    p.add_argument("--ball-weights", default="yolo11n.pt")
    p.add_argument("--ball-class", default="sports ball")
    p.add_argument("--made", default=None, help="（shot）每次投篮是否命中，如 1,0,1")
    p.add_argument("--basket", default=None, help='（drive，球场标定时）篮筐球场坐标 "x,y"')
    p.add_argument("--start-speed", type=float, default=None)
    p.add_argument("--cut-angle", type=float, default=30.0)
    p.set_defaults(func=cmd_stereo)

    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args.func(args)


if __name__ == "__main__":
    main()
