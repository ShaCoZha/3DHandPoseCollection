# Detection, Reliability Weighting, and 3D Consistency

This document explains the implemented WiLoR + Anipose reconstruction pipeline and why an occluded joint should contribute less than a clearly observed joint.

## 1. Terminology and flow

**Block detection** means **hand bounding-box detection**: locating the image region containing a hand before running WiLoR.

**Weighted calibration** is more precisely **weighted, calibrated 3D pose fitting** in this pipeline. The fit updates joint positions using a fixed camera model. It does not refine camera intrinsics, distortion, or extrinsics. Camera calibration is a separate ChArUco procedure.

```mermaid
flowchart TD
    A["Four RGB views + frame timestamps"] --> B["Map clocks and match nearby frames"]
    B --> C["Detect hand boxes and select the target hand"]
    C --> D["WiLoR: 2D joints and predicted hand mesh"]
    D --> E["Initial Anipose triangulation"]
    D --> F["Per-joint reliability: crop stability + mesh visibility + detection score"]
    F --> G["Cross-view consistency weights"]
    E --> H["Joint 3D optimization: weighted reprojection + bones + motion"]
    G --> H
    K["Fixed camera calibration"] --> E
    K --> G
    K --> H
    H --> I["3D skeleton + projection into each RGB view"]
```

## 2. Frame matching and hand detection

Camera hardware timestamps are mapped through the PC clock to the iPhone timeline. The first camera supplies the reference timestamp. Matching uses nearby unused frames, checks the total time span across views, and tries alternative bracketing combinations if individually nearest frames exceed the configured limit.

This is timestamp-based association, not hardware exposure synchronization. A rapidly moving finger may occupy different positions during the four exposures.

For each view, the detector produces hand boxes and a whole-hand score. The baseline selects the largest hand of the configured side. During the reliability pass, selection uses a reference from the other three views when they agree; otherwise it falls back to the baseline selection, then the largest candidate. This association is a heuristic and can still select a bystander's hand.

The reliability pass accepts an optional `detectorImageSize` in the session's
`multicamera/config.json`, such as `1280`. This increases detector input
resolution while keeping predictions in original-image coordinates. The override
is recorded in the inference report and cache manifest, preventing reuse of
observations made at a different detector resolution. Without an override, the
installed detector's default is used (512 for the current checkpoint). Higher
resolution may improve hand detection and handedness classification; it does not
guarantee correct identity or joint locations.

WiLoR predicts a hand mesh and 21 joints from the selected crop. Its projected 2D joints can exist even when a finger is hidden: they are model predictions, not proof that the joint was seen. The baseline's binary keypoint-validity channel is not a learned per-joint confidence score.

## 3. Per-joint reliability

Each camera contributes a separate weight for each joint and time step. A single hand detection score is insufficient because one finger can be hidden while the rest of the hand is visible.

### Crop stability

WiLoR is run on the original crop and three slightly shifted/scaled crops. Let `d` be the joint's RMS displacement in original-image pixels relative to the original-crop prediction, and `s = max(3 px, 0.025 × hand extent)`.

```text
w_stability = 1 / (1 + (d / s)^2)
```

A prediction that moves substantially when the crop changes receives less weight. A stable prediction can still be consistently wrong.

### Mesh self-visibility

The implementation selects a fixed patch of 16 mesh vertices near each joint on the rest-pose hand mesh. It follows those same anatomical vertices in the predicted posed mesh.

From WiLoR's predicted virtual camera, it compares each sample's depth with the nearest intersecting mesh surface along the same ray. A sample is considered visible if it is no more than 1 mm behind that surface. The visible fraction is the number of visible samples divided by 16.

```text
v = visible skin samples / 16
w_mesh = 0.35 + 0.65 × clip(v / 0.45, 0, 1)
```

The test uses skin samples rather than internal skeletal joint centers. It estimates **self-occlusion in the predicted mesh**; it cannot detect an external object covering the hand and can inherit WiLoR's shape or pose mistakes. The 0.35 floor makes it a soft penalty rather than a visibility veto. An unavailable visibility estimate is treated neutrally.

### Whole-hand detection score

The selected hand's detector score is clipped to `[0.3, 1]` and multiplied into the joint weight. Nonfinite or out-of-image 2D joints receive zero image weight.

```text
w_image = w_stability × w_mesh × clipped_hand_detection_score
```

These values are heuristic reliability weights, not calibrated probabilities of correctness or visibility.

## 4. Cross-view consistency

To assess one view, the pipeline constructs a reference using **only the other three views**. It considers pairwise triangulations among those views and requires the resulting point to agree with all three within 12 px. Positive-depth checks also apply.

If that independent reference exists, its projection is compared with the assessed view's 2D prediction. For disagreement `e` in pixels:

```text
w_geometry = max(0.02, 1 / (1 + (e / 15)^2))
w_final = w_image × w_geometry
```

If the other three views do not agree, the geometry factor remains **1**. Lack of a trustworthy reference is not evidence that the assessed view is wrong. In that case, image-based reliability still applies.

Weights are computed before the final fit and then held fixed. The implementation does not repeatedly reduce a view's weight merely because it disagrees with the current fitted pose.

## 5. Weighted 3D fitting

Two modes are available. The legacy mode below refines an existing constrained
3D pose. The new **joint** mode bypasses that pose and is described in section
5.1. Select it explicitly; existing outputs are not automatically migrated.

Anipose first generates candidate 3D joints from subsets containing at least two views. The baseline rejects nonfinite results and points whose selected-subset reprojection error exceeds 5 px. This subset test does not establish agreement across all four cameras.

Baseline regularization estimates bone lengths from supported observations and checks for enough three-view support to establish the capture volume. Gross spatial outliers are rejected; missing joints are not invented. This stage can stop when the calibration and observations do not provide sufficient support.

The weighted fit then optimizes the existing finite joint coordinates. For joint `X`, view `c`, measured prediction `u_c`, and the fixed calibrated projection `π_c`, each image-coordinate residual is:

```text
r_c = sqrt(w_final,c) × (π_c(X) - u_c) / 3 px
```

The objective combines:

| Term | Purpose | Implemented loss |
| --- | --- | --- |
| Weighted 2D reprojection | Make reliable views agree with a shared 3D joint | Cauchy |
| Baseline 3D anchor | Stabilize depth when observations are weak | Soft-L1 |
| Bone lengths | Discourage frame-to-frame finger stretching | Quadratic |
| Timestamp-based acceleration | Reduce rapid, inconsistent motion while allowing fast actions | Soft-L1 |

The robust reprojection loss limits the influence of a badly wrong view. It does not make a geometrically incorrect calibration valid.

### 5.1 Direct joint reconstruction

`pc_receiver/fit_joint_handpose.py SESSION` reads cached `observations.npz`,
camera calibration and matched timestamps. It does **not** read a baseline 3D
pose, its missing-joint mask, its estimated bone lengths or its regularization
report. To run inference as well, use
`pc_receiver/process_weighted_handpose.py SESSION --fit-mode joint`.
Both modes write `multicamera/weighted_handpose`; use a separate session/review
directory to retain comparisons, as done for the CA6A diagnostic run.

```mermaid
flowchart LR
    A["2D joints + per-view joint weights"] --> B["One joint 3D optimization"]
    C["Fixed camera calibration"] --> B
    A --> D["Camera-pair numerical seeds"]
    C --> D
    D -. "initial values only; no 3D anchor" .-> B
    D --> E["Estimate bone lengths from supported seeds"]
    E --> B
    T["Actual timestamps + robust motion"] --> B
    B --> F["Objective check + per-view residuals and support"]
    F --> G["3D joints and RGB projections"]
```

The joint coordinates for a temporal window are the optimization variables:

```text
E(X) = sum(view, time, joint, coordinate)
         weight * soft_L1(((project(X) - measured_2D) / 3 px)^2)
       + sum(bone_length_residual^2)
       + sum(soft_L1(timestamp_acceleration_residual^2))
```

All three terms act in the same solve. There is no penalty for leaving an old
3D skeleton. Reliability multiplies the robust image loss, rather than changing
its pixel transition scale. Image and motion losses use soft-L1; bone lengths
use the same 1.5 mm / 4% scale as the legacy weighted mode. Camera intrinsics,
extrinsics and weights are not optimized.

Initialization tests camera pairs with weights at least 0.15 in both views,
positive depth, at least a 2-degree ray intersection angle and at most 15 px
error in each generating view. The pair candidates are ranked using the
weighted robust image cost across **all** available views. These thresholds
only establish numerical seeds, not final accuracy. Missing single-view joints
remain missing. This mask differs from the legacy 5 px subset rejection mask,
so comparisons must also report errors on the common joints. Bone lengths are
re-estimated from the seeds using robust statistics, preferring three-view
support; insufficient data causes an explicit error.

After window blending, finite coordinates, positive camera depth and the same
full joint objective are checked. A worse or invalid solve reverts to the
numerical seeds, never to the old smoothed skeleton. Conflicting-view errors
and fewer than two supporting views remain diagnostic flags: no whole-frame
P90 rollback is applied. Passing this numerical safeguard is not proof of
accurate 3D or anatomy. The current model has no explicit joint-angle or palm
shape prior and still associates free-running camera frames at one nominal
time; conflicting views can require a compromise.

### 5.2 Camera-time trajectory fitting and short gaps

Use `pc_receiver/fit_joint_handpose.py SESSION --mode trajectory` on an existing
reliability cache, or `pc_receiver/process_weighted_handpose.py SESSION
--fit-mode trajectory` to include reliability inference. As with joint mode,
use a separate session/review directory to preserve previous results.

The optimizer fits piecewise-linear 3D trajectories with knots at the common
reference timestamps. For camera `c`, its original 2D observation at recorded
time `t_c` constrains `project_c(X(t_c))`, rather than `project_c(X(t_reference))`.
Both bracketing 3D knots contribute to the analytic projection Jacobian. Bone
lengths and robust acceleration constrain the same trajectory in one solve;
there is no old 3D anchor. Camera parameters and clock mappings stay fixed.

Recorded per-camera timestamps come from `frame_sets.jsonl`. Their absolute
exposure-start/end/midpoint convention has not been independently verified, so
no exposure/2 correction is guessed. Long exposures integrate motion; the
current model fits a point in time and does not undo that blur or unknown clock
bias. Hardware triggering and shorter exposures remain complementary changes.

For initialization and leave-one-view-out reliability only, 2D predictions are
linearly interpolated to matching times, across gaps no larger than 120 ms.
When aligned 2D seeds are unavailable, a valid native camera-pair seed can
serve as a numerical guess; it is labelled `nativeSeedFallback` and is not a
3D target. This avoids losing otherwise usable observations just because 2D
interpolation is unavailable. The optimization itself uses the **original** 2D
points at their original camera times, not interpolated predictions. Bone lengths are estimated from the
time-aligned seeds. Interior missing 3D seeds are initialized only when valid
seeds bracket the gap within 150 ms (normally up to two missing 20 Hz samples).
Those coordinates are then jointly fitted using available image observations,
bone constraints and neighboring trajectory. Long gaps and sequence endpoints
remain missing; no extrapolation is performed.

Nonfinite or behind-camera output knots are explicitly rejected. They are not
bridged during projection, and cannot cause the entire recording to revert.
The same full objective is compared on the remaining variables and observations
before accepting the blended solution. These checks establish numerical
consistency, not true 3D accuracy. Weakly constrained anatomical configurations
can remain wrong even after convergence.

Outputs add `temporalInferred`, `seedValid`, `perViewJoints`,
`perViewTemporalInferred`, and `viewUnixTimeMs`. The JSONL labels inferred joints
individually. `camerasUsed` counts nearby native observations explained by the
trajectory; it is not proof of simultaneous multiview support at the knot.
The RGB overlays use each camera's trajectory time; the independent 3D panel
uses the reference time. Visualization and clean MP4 export use the session
configured frame rate (30 FPS by default for new 2 ms captures). Pose matching is
limited to half an output-frame interval plus 1 ms; a missing sample
is not silently replaced by a pose a full frame away. Yellow rings mark short-gap estimates, including in
the clean MP4. More displayed joints must not be presented as more measured
joints or demonstrated accuracy. Reports compare reprojection errors on common
observations, and separately count recovered and newly missing joints.

## 6. Bone and temporal consistency

Bone lengths are estimated from the recording, not measured anatomy. In the weighted stage, the bone-length residual scale is `max(1.5 mm, 4% of the estimated bone length)`.

Motion uses actual time differences rather than frame indices:

```text
velocity_before = (X[t]   - X[t-1]) / (time[t]   - time[t-1])
velocity_after  = (X[t+1] - X[t])   / (time[t+1] - time[t])
acceleration    = 2 × (velocity_after - velocity_before)
                  / (time[t+1] - time[t-1])
```

The default fit uses 80-frame windows with 20-frame overlap, blends overlapping solutions, and does not connect motion constraints across gaps longer than 0.12 seconds. Weighted fitting keeps the baseline's finite/missing joint mask unchanged.

Stronger smoothness can suppress genuine fast motion or increase 2D reprojection error. Lower jitter alone does not prove better 3D accuracy.

The weighted stage uses an acceleration residual scale of 10 m/s² with soft-L1
loss (previously 2.5 m/s² with quadratic loss). This reduces the influence of
large accelerations; the initial baseline regularization is unchanged.

After overlapping windows are blended, a quality check uses the same fixed
observations with weights at least 0.5 before and after fitting. With at least
six such observations spanning two cameras, frame median and P90 errors may
increase by at most the larger of 2 px or 10% of their baseline value. Each joint
with at least two high-weight views is also checked: its median error may
increase by at most the larger of 5 px or 25%. A failed check restores the whole
baseline frame, preserving its missing-joint mask. Non-converged windows also
use their input poses. Unchecked frames and fallback frames are recorded in the
report and NPZ/JSONL outputs. The pre-check candidate is saved separately.
This guards agreement with predictions, not ground-truth accuracy, and does not
repair a bad baseline. Frame rollbacks may introduce temporal discontinuities.

## 7. What happens to an occluded finger?

Suppose a joint is hidden in cam01 and visible in cam02/cam03:

1. WiLoR may still predict the joint in cam01 from its learned hand prior.
2. Crop sensitivity or mesh self-occlusion can reduce cam01's image weight; neither signal is guaranteed to identify the mistake.
3. If the other three views agree, their reference can further reduce cam01's geometric weight. If they do not agree, this factor stays neutral.
4. The weighted fit gives reliable observations more influence while preserving plausible bones and motion.
5. If only one view has useful support, depth remains underconstrained. The output then depends more on the baseline, bone lengths and temporal priors; it is not a newly measured depth value.

After fitting, a view counts as support only when its weight is at least 0.15 and its reprojection error is at most 15 px. Joints with fewer than two supported views are marked as prior-dominated. That label is a diagnostic, not a confidence probability.

## 8. Visualization and implementation map

Orange shows WiLoR 2D predictions. Cyan shows the reconstructed 3D skeleton projected through each fixed camera model. The weighted viewer also exposes per-view joint weights and highlights insufficient support. IMU and EIT panels share the replay timeline but do not participate in the 3D fitting objective.

| Implementation | Responsibility |
| --- | --- |
| [align_multicamera.py](pc_receiver/align_multicamera.py) | Clock mapping and frame matching |
| [process_multicamera.py](pc_receiver/process_multicamera.py) | Baseline detection and Anipose triangulation |
| [infer_hand_reliability.py](pc_receiver/infer_hand_reliability.py) | Target association, crop perturbations, mesh cache |
| [hand_reliability.py](pc_receiver/hand_reliability.py) | Stability, mesh visibility, leave-one-view-out geometry |
| [regularize_handpose.py](pc_receiver/regularize_handpose.py) | Bone estimation and timestamp constraints |
| [fit_weighted_handpose.py](pc_receiver/fit_weighted_handpose.py) | Fixed-weight calibrated reprojection optimization |
| [solve_multicamera_calibration.py](pc_receiver/solve_multicamera_calibration.py) | Separate ChArUco camera calibration and validation |
| [visualize_multicamera.py](pc_receiver/visualize_multicamera.py) | Projection audit and interactive replay |

The figures above describe current defaults. Session configuration and processing reports are the source of truth for a particular run. Calibration changes invalidate caches that depend on calibration; changing a calibration file alone does not regenerate poses or videos.
