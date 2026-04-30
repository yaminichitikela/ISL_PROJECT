# Paper 5 — BlazePose / MediaPipe Holistic: On-Device Real-Time Body Pose Tracking
**Bazarevsky et al. (Google Research), 2020**

---

## What is this paper about?
This paper introduces **BlazePose** and the **MediaPipe Holistic** pipeline — the tools you use to extract skeleton keypoints from INCLUDE videos. It's the pose estimation backbone of your entire project.

Without this paper, you'd need an expensive depth sensor or OptiTrack motion capture system to get skeleton data. MediaPipe gives you pose estimation from a **regular RGB camera, in real-time, on a mobile device**.

---

## What MediaPipe Holistic Gives You
A single pipeline that outputs three things simultaneously from one RGB video frame:

| Component | Keypoints | What it captures |
|-----------|-----------|-----------------|
| BlazePose (body) | 33 joints | Full body pose |
| Hand landmarks (left) | 21 keypoints | Each finger joint |
| Hand landmarks (right) | 21 keypoints | Each finger joint |
| **Total** | **75 keypoints** | Full body + both hands |

---

## The 33 BlazePose Body Keypoints Include:
Nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles, feet — plus face landmarks

## The 21 Hand Keypoints (per hand):
Wrist + 4 joints per finger (MCP, PIP, DIP, TIP) for all 5 fingers = 21 per hand

---

## Your Project's 53-Joint Subset
From the full 75 keypoints, your roadmap uses **53 joints**:
- 21 left hand keypoints
- 21 right hand keypoints
- 11 upper body: nose, left/right shoulder, left/right elbow, left/right wrist (+ a few more)

Why 53 and not 75? The lower body (hips, knees, feet) is irrelevant for ISL — signs are performed in the upper body space.

---

## Why MediaPipe Instead of Other Pose Estimators?
| Feature | MediaPipe | OpenPose |
|---------|-----------|---------|
| Speed | 30+ fps on mobile | Slow (GPU needed) |
| Hand tracking | Built-in (21 per hand) | Not built-in |
| Free to use | Yes | Yes |
| Accuracy | Good for single person | Very high |

For sign language, **hand tracking** is non-negotiable — MediaPipe is the only widely-used real-time pose estimator that includes detailed hand keypoints.

---

## Known Limitation (relevant to Paper 11)
MediaPipe's hand region-of-interest (ROI) prediction uses a simple heuristic (3 hand points: wrist, index MCP, pinky MCP). When the hand is **rotated** (not parallel to the camera plane), this heuristic fails and the hand bounding box is wrong — leading to incorrect keypoint predictions.

Paper 11 (Moryossef 2024) addresses this exact problem.

---

## What's Important for YOUR Project
- **You use MediaPipe Holistic to extract skeleton sequences from all INCLUDE videos** — this is your data extraction pipeline (Milestone 0)
- The 53-joint graph topology is defined by MediaPipe's output structure
- Graph edges in your ISL GCN = anatomical connections between these 53 joints
- **Inter-hand edge:** Since ISL has many two-handed signs, you need an edge connecting left-hand wrist to right-hand wrist through the shoulder chain (left wrist → left elbow → left shoulder → right shoulder → right elbow → right wrist)
- **Single-hand signs:** For signs using only one hand, you zero out the inactive hand's keypoints during preprocessing

---

## One-line summary
> MediaPipe Holistic is your pose extraction backbone — gives you 75 keypoints (33 body + 21 per hand) from regular video, and your project uses a 53-joint subset focused on the upper body and hands.
