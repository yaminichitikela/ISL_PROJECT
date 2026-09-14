"""
Full keypoint extraction script — equivalent to Notebook 01.
Run with: /Library/Developer/CommandLineTools/usr/bin/python3 run_extraction.py
"""
import os, sys, time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path("/Users/yamini/Desktop/projects/ISL PROJECT")
VIDEO_DIR    = PROJECT_ROOT / "include_videos"
MODEL_DIR    = PROJECT_ROOT
DATA_DIR     = PROJECT_ROOT / "data"
RAW_KP_DIR   = DATA_DIR / "raw_keypoints"
RAW_KP_DIR.mkdir(parents=True, exist_ok=True)

HAND_MODEL   = str(MODEL_DIR / "hand_landmarker.task")
POSE_MODEL   = str(MODEL_DIR / "pose_landmarker_full.task")
POSE_INDICES = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
MAX_GAP      = 5

# ── Build detectors ──────────────────────────────────────────────────────────
hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
)
pose_detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
)


def fix_missing_hand(hand_kps, max_gap=MAX_GAP):
    T = hand_kps.shape[0]
    missing = np.all(hand_kps == 0.0, axis=(1, 2))
    if not missing.any():
        return hand_kps
    result = hand_kps.copy()
    i = 0
    while i < T:
        if not missing[i]:
            i += 1
            continue
        j = i
        while j < T and missing[j]:
            j += 1
        gap_len   = j - i
        has_before = (i > 0) and (not missing[i - 1])
        has_after  = (j < T) and (not missing[j])
        if gap_len <= max_gap and has_before and has_after:
            before = result[i - 1]
            after  = result[j]
            for k in range(gap_len):
                alpha = (k + 1) / (gap_len + 1)
                result[i + k] = (1.0 - alpha) * before + alpha * after
        i = j
    return result


def extract_keypoints(video_path):
    cap = cv2.VideoCapture(str(video_path))
    raw_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        hand_result = hand_detector.detect(mp_image)
        left_hand   = np.zeros((21, 3), dtype=np.float32)
        right_hand  = np.zeros((21, 3), dtype=np.float32)
        for idx, handedness_list in enumerate(hand_result.handedness):
            label = handedness_list[0].category_name
            lms   = np.array([[lm.x, lm.y, lm.z]
                               for lm in hand_result.hand_landmarks[idx]], dtype=np.float32)
            if label == "Left":
                left_hand = lms
            else:
                right_hand = lms

        pose_result = pose_detector.detect(mp_image)
        if pose_result.pose_landmarks:
            lms  = pose_result.pose_landmarks[0]
            body = np.array([[lms[i].x, lms[i].y, lms[i].z]
                              for i in POSE_INDICES], dtype=np.float32)
        else:
            body = np.zeros((11, 3), dtype=np.float32)

        raw_frames.append(np.concatenate([left_hand, right_hand, body], axis=0))

    cap.release()
    if not raw_frames:
        return None, {}, {}

    kps = np.stack(raw_frames, axis=0).astype(np.float32)
    miss_left_raw  = int(np.sum(np.all(kps[:, :21,  :] == 0, axis=(1, 2))))
    miss_right_raw = int(np.sum(np.all(kps[:, 21:42, :] == 0, axis=(1, 2))))

    kps[:, :21,  :] = fix_missing_hand(kps[:, :21,  :])
    kps[:, 21:42, :] = fix_missing_hand(kps[:, 21:42, :])

    miss_left_fix  = int(np.sum(np.all(kps[:, :21,  :] == 0, axis=(1, 2))))
    miss_right_fix = int(np.sum(np.all(kps[:, 21:42, :] == 0, axis=(1, 2))))

    return kps, \
           {"missing_left": miss_left_raw,  "missing_right": miss_right_raw}, \
           {"missing_left": miss_left_fix,  "missing_right": miss_right_fix}


# ── Scan all videos ──────────────────────────────────────────────────────────
all_videos = []
for cat_dir in sorted(VIDEO_DIR.iterdir()):
    if not cat_dir.is_dir():
        continue
    for sign_dir in sorted(cat_dir.iterdir()):
        if not sign_dir.is_dir():
            continue
        for video_file in sorted(sign_dir.glob("*.MOV")):
            safe_sign = sign_dir.name.replace("/", "-")
            out_name  = f"{cat_dir.name}__{safe_sign}__{video_file.stem}.npy"
            all_videos.append({
                "category"  : cat_dir.name,
                "sign"      : sign_dir.name,
                "video_id"  : video_file.stem,
                "video_path": str(video_file),
                "rel_path"  : f"{cat_dir.name}/{sign_dir.name}/{video_file.name}",
                "out_name"  : out_name,
            })

total = len(all_videos)
print(f"Total videos: {total}")

metadata = []
errors   = []
t0       = time.time()

for i, v in enumerate(all_videos):
    out_path = RAW_KP_DIR / v["out_name"]

    # Progress every 50 videos
    if i % 50 == 0:
        elapsed = time.time() - t0
        done    = sum(1 for m in metadata if m.get("status") == "ok")
        eta     = (elapsed / max(i, 1)) * (total - i)
        print(f"[{i:4d}/{total}] done={done} errors={len(errors)} "
              f"elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m", flush=True)

    if out_path.exists():
        ex = np.load(str(out_path), mmap_mode="r")
        row = {**v, "n_frames": ex.shape[0], "status": "skipped"}
        metadata.append(row)
        continue

    try:
        kps, stats_raw, stats_fixed = extract_keypoints(v["video_path"])
        if kps is None:
            errors.append({"video": v["video_path"], "error": "No frames"})
            continue
        np.save(str(out_path), kps)
        row = {**v,
               "n_frames"       : kps.shape[0],
               "status"         : "ok",
               "miss_left_raw"  : stats_raw["missing_left"],
               "miss_right_raw" : stats_raw["missing_right"],
               "miss_left_fixed": stats_fixed["missing_left"],
               "miss_right_fixed": stats_fixed["missing_right"]}
        metadata.append(row)
    except Exception as e:
        errors.append({"video": v["video_path"], "error": str(e)})
        print(f"ERROR: {v['video_path']}\n  {e}", flush=True)

# ── Save metadata ────────────────────────────────────────────────────────────
meta_df = pd.DataFrame(metadata)
meta_df.to_csv(DATA_DIR / "keypoints_metadata.csv", index=False)

elapsed = time.time() - t0
print(f"\nDone in {elapsed/60:.1f} minutes.")
print(f"  Processed : {sum(1 for m in metadata if m.get('status') == 'ok')}")
print(f"  Skipped   : {sum(1 for m in metadata if m.get('status') == 'skipped')}")
print(f"  Errors    : {len(errors)}")
print(f"\nFrame stats:")
print(meta_df["n_frames"].describe().round(1).to_string())
