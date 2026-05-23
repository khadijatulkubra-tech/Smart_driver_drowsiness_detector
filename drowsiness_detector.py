import cv2
import numpy as np
import mediapipe as mp
import time
import math
from collections import deque
import pygame

# ── Detection Thresholds ──────────────────────────
EAR_THRESHOLD       = 0.25
CONSEC_FRAMES       = 20
MAR_THRESHOLD       = 0.08
YAWN_CONSEC_FRAMES  = 15
HEAD_TILT_THRESHOLD = 20

# ── Eye & Mouth Landmark Points ───────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT   = 78
MOUTH_RIGHT  = 308

# ── EAR Formula ───────────────────────────────────
def eye_aspect_ratio(landmarks, eye_points, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_points]
    A = math.dist(pts[1], pts[5])
    B = math.dist(pts[2], pts[4])
    C = math.dist(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)

# ── MAR Formula (fixed) ───────────────────────────
def mouth_aspect_ratio(landmarks, w, h):
    top    = (landmarks[MOUTH_TOP].x * w,    landmarks[MOUTH_TOP].y * h)
    bottom = (landmarks[MOUTH_BOTTOM].x * w, landmarks[MOUTH_BOTTOM].y * h)
    left   = (landmarks[MOUTH_LEFT].x * w,   landmarks[MOUTH_LEFT].y * h)
    right  = (landmarks[MOUTH_RIGHT].x * w,  landmarks[MOUTH_RIGHT].y * h)
    vertical   = math.dist(top, bottom)
    horizontal = math.dist(left, right)
    return vertical / (horizontal + 1e-6)

# ── Head Tilt ─────────────────────────────────────
def head_tilt(landmarks, w, h):
    nose = landmarks[1]
    chin = landmarks[152]
    dx = (chin.x - nose.x) * w
    dy = (chin.y - nose.y) * h
    return abs(math.degrees(math.atan2(dx, dy)))

# ── Alarm Setup ───────────────────────────────────
pygame.mixer.init()

def play_alarm():
    duration    = 1.0
    freq        = 900
    sample_rate = 44100
    t    = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    wave = np.column_stack([wave, wave])
    sound = pygame.sndarray.make_sound(wave)
    sound.play()

# ── MediaPipe Setup ───────────────────────────────
mp_face   = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── Counters ──────────────────────────────────────
eye_counter  = 0
yawn_counter = 0
nod_counter  = 0

drowsy_episodes = 0
yawn_count      = 0
last_alarm_time = 0
session_start   = time.time()

ear_history = deque(maxlen=10)
mar_history = deque(maxlen=10)

# ── Camera Start ──────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ── Half Screen Window ────────────────────────────
cv2.namedWindow("Driver Drowsiness Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Driver Drowsiness Detection", 960, 540)
cv2.moveWindow("Driver Drowsiness Detection", 160, 100)

print("=" * 45)
print("  Driver Drowsiness Detection — STARTED")
print("  Q dabao band karne ke liye")
print("=" * 45)

# ── Main Loop ─────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    status       = "ALERT"
    status_color = (0, 255, 0)

    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark

        # EAR calculate karo
        left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
        right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        ear = (left_ear + right_ear) / 2.0
        ear_history.append(ear)
        smooth_ear = np.mean(ear_history)

        # MAR calculate karo
        mar = mouth_aspect_ratio(lm, w, h)
        mar_history.append(mar)
        smooth_mar = np.mean(mar_history)

        # Head tilt calculate karo
        tilt = head_tilt(lm, w, h)

        # ── Eye Check ─────────────────────────────
        if smooth_ear < EAR_THRESHOLD:
            eye_counter += 1
        else:
            eye_counter = 0

        # ── DROWSY — sirf yahan alarm bajega ──────
        if eye_counter >= CONSEC_FRAMES:
            drowsy_episodes += 1
            status       = "DROWSY! WAKE UP!"
            status_color = (0, 0, 255)
            if time.time() - last_alarm_time > 3:
                play_alarm()
                last_alarm_time = time.time()

        # ── Yawn Check — alarm nahi ───────────────
        if smooth_mar > MAR_THRESHOLD:
            yawn_counter += 1
        else:
            yawn_counter = 0

        if yawn_counter >= YAWN_CONSEC_FRAMES:
            yawn_count += 1
            if status == "ALERT":
                status       = "YAWNING!"
                status_color = (0, 165, 255)

        # ── Head Nod Check ────────────────────────
        if tilt > HEAD_TILT_THRESHOLD:
            nod_counter += 1
        else:
            nod_counter = 0

        if nod_counter >= 10:
            if status == "ALERT":
                status       = "HEAD NODDING!"
                status_color = (0, 255, 255)

        # ── Warning Zone ──────────────────────────
        if eye_counter > CONSEC_FRAMES // 2 and status == "ALERT":
            status       = "WARNING!"
            status_color = (0, 200, 255)

        # ── Landmarks Draw karo ───────────────────
        for idx in LEFT_EYE + RIGHT_EYE:
            pt = (int(lm[idx].x * w), int(lm[idx].y * h))
            cv2.circle(frame, pt, 2, (0, 255, 255), -1)
        for idx in [MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT]:
            pt = (int(lm[idx].x * w), int(lm[idx].y * h))
            cv2.circle(frame, pt, 3, (255, 100, 0), -1)

        # ── Info Panel ────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (320, 195), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        session_time = int(time.time() - session_start)
        info = [
            f"EAR   : {smooth_ear:.3f}",
            f"MAR   : {smooth_mar:.3f}",
            f"TILT  : {tilt:.1f} deg",
            f"DROWSY: {drowsy_episodes}",
            f"YAWNS : {yawn_count}",
            f"TIME  : {session_time}s",
        ]
        for i, text in enumerate(info):
            color = (0, 255, 100) if i < 3 else (200, 200, 200)
            cv2.putText(frame, text, (10, 35 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    else:
        cv2.putText(frame, "FACE DETECT NAHI HUA", (w//2 - 180, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 255), 2)

    # ── Status Banner ─────────────────────────────
    cv2.rectangle(frame, (0, h - 65), (w, h), (20, 20, 20), -1)
    text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_DUPLEX, 1.4, 3)[0]
    tx = (w - text_size[0]) // 2
    cv2.putText(frame, status, (tx, h - 15),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, status_color, 3)

    # ── Danger Border sirf DROWSY pe ──────────────
    if "DROWSY" in status:
        thickness = int(abs(math.sin(time.time() * 5)) * 8) + 2
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), thickness)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── End ───────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print(f"\nSession khatam!")
print(f"Drowsy episodes : {drowsy_episodes}")
print(f"Yawns           : {yawn_count}")
print(f"Session time    : {int(time.time() - session_start)}s")