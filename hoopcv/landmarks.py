"""MediaPipe Pose 33 点索引（注意：LEFT/RIGHT 指被拍者自己的左右）。"""

NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_THUMB, R_THUMB = 21, 22
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

SKELETON = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 19), (16, 20), (15, 17), (16, 18),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32),
]


def side_ids(side: str) -> dict:
    """返回某一侧（'right' / 'left'）的关键点索引，以及对侧肩/髋。"""
    if side not in ("right", "left"):
        raise ValueError("side 必须是 'right' 或 'left'")
    r = side == "right"
    return dict(
        shoulder=R_SHOULDER if r else L_SHOULDER,
        elbow=R_ELBOW if r else L_ELBOW,
        wrist=R_WRIST if r else L_WRIST,
        index=R_INDEX if r else L_INDEX,
        pinky=R_PINKY if r else L_PINKY,
        thumb=R_THUMB if r else L_THUMB,
        hip=R_HIP if r else L_HIP,
        knee=R_KNEE if r else L_KNEE,
        ankle=R_ANKLE if r else L_ANKLE,
        foot=R_FOOT if r else L_FOOT,
        o_shoulder=L_SHOULDER if r else R_SHOULDER,
        o_hip=L_HIP if r else R_HIP,
    )
