# Paper 11 — Optimizing Hand Region Detection in MediaPipe Holistic
**Moryossef, University of Zurich, arXiv 2024**

---

## What is this paper about?
This is a short technical paper (4 pages) that fixes a specific bug in MediaPipe Holistic's hand detection. It's directly relevant because you use MediaPipe to extract keypoints from INCLUDE videos.

**The bug:** MediaPipe uses a simple heuristic to find where the hand is in the image (the "region of interest" or ROI). This heuristic assumes the hand is always roughly parallel to the camera. When hands are rotated or at unusual angles, the heuristic fails → wrong bounding box → inaccurate hand keypoints.

---

## The Problem (in plain terms)
MediaPipe locates the hand using 3 points: wrist, index MCP, pinky MCP. It calculates the hand center and size from these. But:

- It ignores the `z` depth coordinate (assumes hand is flat in camera plane)
- It ignores shoulder, elbow, and thumb — which also indicate hand orientation
- For ISL signs where hands are rotated (palm facing camera vs. facing away), this breaks

**Result:** When you rotate your hand, MediaPipe's bounding box doesn't rotate with it → hand keypoints land in the wrong positions.

---

## The Fix
Replace the 3-point heuristic with a small trained MLP (multi-layer perceptron):

**Input to MLP:** All 6 right-hand normalized keypoints (wrist, shoulder, elbow, thumb, index, pinky) INCLUDING the z coordinate and image aspect ratio

**Output:** Better prediction of the hand ROI center and size

**Result:** 
| Method | IoU (↑ better) |
|--------|----------------|
| Original MediaPipe heuristic | 57% |
| MLP replacement | **63%** |

The MLP better predicts the ROI center and size (+6% IoU). Rotation prediction still imperfect (MLP only uses linear layers with relu — struggles with angular values).

---

## Why This Matters for YOUR Project

### 1. Data Quality Check
When extracting skeleton sequences from INCLUDE videos, check for keypoint quality:
- Are hand keypoints often landing in wrong positions?
- INCLUDE signers perform in a controlled studio setting — less rotation variability than uncontrolled video
- The bug likely affects you LESS than in-the-wild video, but it's still worth knowing

### 2. Preprocessing Decision
You have two options:
- **Option A:** Use standard MediaPipe (simpler, 57% IoU for hands)
- **Option B:** Apply the MLP fix from this paper (63% IoU) — needs their code: https://github.com/sign-language-processing/mediapipe-hand-crop-fix

For a controlled dataset like INCLUDE, Option A is probably fine.

### 3. Confidence Filtering
Even without fixing the ROI prediction, you can filter out low-confidence keypoints:
- MediaPipe provides a visibility/confidence score per keypoint
- Set a threshold (e.g., confidence > 0.5) and zero out uncertain keypoints
- This prevents bad keypoint detections from polluting your GCN input

### 4. Ablation A5 (Normalization)
Part of your normalization strategy should account for mediapipe keypoint noise. Options:
- Normalize by torso height (shoulder-to-hip distance)
- Normalize by wrist-to-elbow distance (hand-relative normalization)
- Subtract mean position to center the signer

---

## Key Code Available
https://github.com/sign-language-processing/mediapipe-hand-crop-fix

If you find INCLUDE keypoints are noisy (many frames with missing hand detections), check and potentially apply this fix.

---

## One-line summary
> MediaPipe's hand bounding box prediction fails for rotated hands — this paper proposes an MLP fix (+6% IoU) — relevant when extracting INCLUDE keypoints, especially for signs with unusual hand orientations.
