# Pipeline hardening

RTMO-m remains the default; RTMPose is optional, inference is fully local, and
posture and temporal fall status stay separate. No production temporal model was
trained or fabricated. This document records behavior and compatibility changes,
not evidence of better held-out fall accuracy.

## Identity and temporal evidence

Association uses predicted translation plus IoU, body-relative bbox-center and
joint distances, and log scale consistency only when both scales are reliable.
Matches must pass absolute gates and have clear alternatives on both the track
and detection sides. Near ties are lost/unobserved; no ambiguous detection is
fed to a fall buffer. Each affected track increments a history epoch, clearing
its fall history and state latch before any reacquisition. Retained IDs are
best-effort identities; no ReID claim is made. Expired IDs are never reused.

Short unambiguous misses retain IDs for a duration in seconds. A new or
identity-invalidated track near a lost alerted person enters an alert quarantine,
without inheriting skeletons, probabilities or FALLEN state. It requires stable
upright evidence for recovery plus rearming before it can emit an independent
alert. Continuously identified people have independent alert episodes even
near one another. Spatial quarantine can suppress an unrelated new person's
alert nearby, and cannot recover identity through complete occlusion.

Weak shoulders, hips or ankles suspend heuristic motion measurements. Brief
unreliability preserves valid history/candidates but contributes neither lying
persistence nor state confirmation duration. Prolonged unreliability, timestamp
regression/duplication, a long valid-pose gap, expired ID or uncertain association
resets evidence. Learned inference also suspends confirmation on weak current
root/shoulder/ankle pairs and resets sustained unreliable windows.

Posture smoothing remains consecutive votes with a minimum time hold. Fall and
possible-fall confirmation accumulate **valid prediction duration** in seconds,
independently of inference FPS. Recovery requires valid upright predictions;
rearming requires an additional stable upright period. A relapse before rearming
belongs to the same episode. An independently recovered/rearmed second fall is
allowed without waiting for the old cooldown. Events add `episode_id` while
retaining the existing callback arguments and other JSON fields.

## Heuristic decisions

The rapid path requires root/shoulder descent, velocity, rotation and subsequent
lying near the initial ankle plane. Bent origins can use a smaller additional
rotation. Low origins require less hip descent, but substantially stronger
shoulder descent and speed. Merely kneeling, squatting or bending cannot supply
all that evidence.

The slow path requires substantial cumulative descent within 1.2–2.5 seconds,
mostly downward motion, progression to lying, the same floor proxy and longer
lying persistence. A separate controlled-lowering guard recognizes a recent
straight-knee to flexed-knee lowering phase with an upright torso. It suppresses
a staged intentional floor lie-down through the remaining history window.
Initially kneeling people are still eligible for collapse detection.

These are conservative image-plane policies. Intentional and accidental motions
can produce identical 2D skeletons. An ankle plane is not a measured floor, body
parts can be foreshortened, and a genuine collapse immediately after controlled
kneeling/sitting can be suppressed. Viewpoint, camera motion, occlusion and falls
outside the frame remain difficult. Tune thresholds against actual held-out
falls and hard negatives, not just the projected fixtures in tests.

## Normalization contract

`Pose.normalized` remains per-frame hip centered for posture. Pose objects now
expose the scale measurement source, quality and whether a per-track filter was
applied. Reliable torso-plus-leg measurements update a rolling-median/time-based
EMA. Torso-only and bbox fallback measurements cannot replace a reliable scale
for one missing-joint frame.

Fall model inputs use **`window_root_v1`**. For each causal window:

1. Choose its first confident hip/root center.
2. Choose one median reliable torso-plus-leg-length body scale across the window.
3. Apply that same origin and scale to every frame's pixel joints.
4. Retain the confidence channel, uniformly sample timestamps and interpolate
   only short joint gaps; use zero-confidence padding and no extrapolation.

This preserves global root translation while removing a constant image offset
and body size. Training windows, the ONNX adapter and checkpoint evaluation use
the same function. Training splits identity epochs and long gaps. Extraction
saves raw pixels and epochs using the same VFR timestamp reader as inference.

Old independently centered fall weights require **retraining**. Altering their
metadata does not change what they learned. New exports store normalization,
layout, class order, history/sampling and confidence/gap settings in a sidecar;
the adapter rejects conflicting declared settings. Missing sidecars cannot be
verified automatically: externally supplied models must be checked against the
contract. Posture weights keep their existing per-frame contract.

## New configuration reference

Every value below is in `configs/default.yaml`, with validation in `src/config.py`.
Distances and drops are body relative; velocities are body lengths per second.
Angles are degrees. Association weights are normalized by their sum.

| Section | Setting | Default | Meaning |
|---|---|---:|---|
| tracking | center_weight / skeleton_weight / scale_weight | .25 / .35 / .10 | Additional association terms; IoU weight is now .30 |
| tracking | max_center_distance / max_scale_ratio | .9 / 1.8 | Predicted center and reliable scale gates |
| tracking | minimum_common_joints / minimum_sparse_iou | 4 / .3 | Sparse skeletons require box agreement |
| tracking | ambiguity_margin | .035 | Minimum cost separation on both sides |
| tracking | minimum_association_confidence | .2 | Absolute association quality floor |
| tracking | scale_ema_seconds / scale_window_seconds | .6 / 1 | Reliable scale filter timing |
| tracking | minimum_scale_quality / scale_max_change_ratio | .6 / 1.25 | Scale admission and bounded EMA target |
| posture | confidence_boundary_fraction | .55 | Confidence factor at a rule boundary |
| posture | confidence_angle_margin / confidence_knee_margin | 20 / 25 | Margin saturation widths |
| posture | confidence_height_margin / confidence_aspect_margin | .15 / .5 | Body-height/aspect margin widths |
| fall | max_unreliable_seconds / minimum_pose_quality | .45 / .4 | Suspend briefly, reset sustained weak evidence |
| fall | possible_confirmation_seconds / fall_confirmation_seconds | .08 / .20 | Valid temporal prediction duration |
| fall | maximum_origin_torso_angle / bent_origin_torso_rotation | 75 / 18 | Eligible non-lying bent origins |
| fall | low_origin_hip_ankle_height | .25 | Identify an already low origin |
| fall | low_origin_minimum_hip_drop / low_origin_minimum_shoulder_drop | .14 / .3 | Low-origin descent gates |
| fall | low_origin_minimum_hip_velocity / low_origin_minimum_shoulder_velocity | .35 / .8 | Stronger shoulder motion for low origins |
| fall | maximum_floor_distance | .2 | Final root proximity to the origin ankle plane |
| fall | slow_enabled | true | Enable the independent collapse path |
| fall | slow_minimum_motion_seconds / slow_window_seconds | 1.2 / 2.5 | Slow evidence duration |
| fall | slow_minimum_hip_drop / slow_minimum_shoulder_drop | .58 / .5 | Substantial cumulative descent |
| fall | slow_minimum_hip_velocity / slow_maximum_hip_velocity | .18 / .65 | Collapse velocity range |
| fall | slow_minimum_torso_rotation | 35 | Additional rotation |
| fall | slow_minimum_downward_fraction / slow_jitter_tolerance | .8 / .015 | Monotonic descent with small jitter tolerance |
| fall | slow_lying_persistence_seconds | .6 | Persistent low-lying endpoint |
| fall | controlled_lowering_enabled | true | Guard a staged controlled lie-down |
| fall | controlled_origin_knee_angle / controlled_knee_angle | 155 / 130 | Straight-to-flexed preparatory change |
| fall | controlled_minimum_hip_drop | .18 | Lowering while torso remains upright |
| fall | controlled_confirmation_seconds / controlled_window_seconds | .12 / 1.2 | Sustained preparatory evidence and lookback |
| fall | controlled_hold_seconds | 2.5 | Suppress through old history; must cover history duration |
| fall | normalization | window_root_v1 | Required fall model contract |
| fall | minimum_scale_quality | .6 | Reliable fixed-window scale measurements |
| fall | min_visible_fraction / min_current_pose_fraction | .35 / .5 | Learned temporal/current visibility gates |
| state | rearm_seconds / rearm_postures | 1 / standing,sitting | Stable upright rearming after recovery |
| events | episode_retention_seconds | 60 | Bounded lost-episode alert guard |
| events | episode_match_distance / episode_min_iou | .8 / .05 | Spatial guard for uncertain reacquisition only |
| visualization | max_overlay_frame_lag | 8 | Maximum captured-frame separation for drawing |

Changed defaults: history is 2.5 seconds; candidate timeout is 2 seconds; overlay
age is .25 seconds; `max_missing_frames: 0` disables the legacy frame cap in favor
of `max_missing_seconds`. Explicit nonzero frame caps still work. Extended
history profiles must also extend the controlled hold and fit both motion windows.

Non-default legacy `confirmation_frames` overrides are translated at a fixed
15 Hz reference when no customized duration is supplied: `(frames - 1) / 15`.
Direct state-machine callers can similarly use legacy possible-frame overrides.
Prefer the new seconds keys. An explicitly different seconds value wins; to use
the default .20 seconds with an old non-default count, reset/remove that count.
`event_cooldown_seconds` remains accepted for profile compatibility but does not
replace recovery/rearming. All existing CLI flags remain available. Explicit
unsupported providers, including Azure, are rejected; auto uses only the supported
local CoreML/CUDA/DirectML/OpenVINO/CPU set. RTMO preprocessing is unchanged.

## Regression coverage and manual validation

Projected fixtures cover initially lying, supported fast lying/bed/sofa,
controlled floor lying, rapid rotations in both directions/forward projection,
slow collapse and slow intentional lying, bending/tying shoes, low postures,
bending/kneeling origins, missing samples and weak body pairs, sustained
occlusion, jitter/regression and equivalent 5/10/15/30 Hz outcomes. Tracker tests
cover reorderings, crossings, complete/merged overlaps, quarantine, expiration,
and one falling person with another crossing. Other tests cover VFR timestamps,
unclipped NMS, invalid individual joints, stale result indices, runtime failover,
fixed-window normalization, metadata mismatch and training epoch/gap boundaries.

The original tests remain present. Two obsolete expectations were updated:
ankle occlusion now preserves candidates, and learned input tests use real body
geometry and assert preserved root trajectories rather than independently
centered coordinates. Optional integration tests still report skips explicitly
when models/runtime dependencies are absent.

From this Mac checkout:

```sh
cd /Users/abdullahshahidali/Documents/Fall_test
source .venv/bin/activate
python tools/check_runtime.py
python -m src.app --camera 0 --config configs/mac.yaml --debug
python -m src.app --camera 0 --headless --max-frames 300
python -m src.app --video path/to/video.mp4 --config configs/mac.yaml --debug
python -m src.app --video path/to/video.mp4 --output outputs/result.mp4
python -m src.app --video artifacts/sample_input.mp4 --headless --no-realtime --output outputs/smoke.mp4
python -m pytest -ra --basetemp=outputs/pytest-tmp
ruff check .
ruff format --check .
```

In the debug overlay inspect quality versus confidence, stable scale and match
confidence while crossing or briefly hiding joints. Check that intentional lying
stays LYING and that alerts rearm only after recovery and stable upright evidence.
Use existing annotated recordings for actual fall testing and measure per-person
false alarms and latency; these commands and unit fixtures do not establish
accuracy. Recording writes nominal constant FPS and does not preserve VFR timing
or audio. Windows accelerators require actual target-device testing.

## Changed files

| Files | Change |
|---|---|
| `configs/default.yaml`, `src/config.py` | New validated thresholds and seconds-first defaults |
| `src/tracking/matching.py`, `src/tracking/tracker.py` | Stronger gates, ambiguity abstention, epochs and stable track scale |
| `src/pose/pose_types.py`, `src/pose/normalization.py` | Scale quality/source and robust scale filter |
| `src/pose/rtmo.py` | Unclipped NMS, kept-box clipping, isolated invalid-joint sanitization |
| `src/posture/classifier.py`, `src/posture/geometry.py`, `src/posture/rules.py` | Separate pose quality and rule-margin confidence |
| `src/fall/heuristic.py` | Preserve short unreliable evidence; rapid, low-origin, collapse and controlled-lowering policies |
| `src/fall/normalization.py`, `src/fall/temporal_buffer.py`, `src/fall/stgcn.py` | Fixed-window root contract, interpolation and metadata/current-quality validation |
| `src/state/state_machine.py`, `src/state/episodes.py` | Timed confirmation, recovery/rearming and lost-ID alert quarantine |
| `src/pipeline.py`, `src/events.py` | Association epoch boundaries and episode-aware event payloads |
| `src/capture/camera.py`, `src/app.py`, `src/visualization/renderer.py` | VFR clock, source-frame result indices and stale overlay gates |
| `src/runtime/providers.py` | Reject unsupported/nonlocal explicit providers |
| `training/dataset.py`, `training/extract_skeletons.py`, `training/train_fall.py`, `training/evaluate.py` | Shared normalization, extraction epochs/timestamps, new metadata and incompatible checkpoint rejection |
| `tests/test_config.py`, `tests/test_fall.py`, `tests/test_stgcn.py`, `tests/test_state_machine.py` | Retained/extended regressions and revised obsolete normalization/occlusion expectations |
| `tests/test_hardening_capture.py`, `tests/test_hardening_fall.py`, `tests/test_hardening_normalization.py`, `tests/test_hardening_runtime.py`, `tests/test_hardening_tracking.py` | New deterministic hardening regressions |
| `README.md`, `models/README.md`, `training/README.md`, `docs/HARDENING.md` | Contracts, compatibility, configuration, limits and manual commands |

Fresh local validation results are recorded in `outputs/hardening_validation.json`.
Benchmarks in `outputs/hardening_benchmark_*.json` exclude GUI/encoding, and the
synthetic benchmark excludes pose inference. Historical `artifacts/` results in
the main README predate this hardening and are not new accuracy evidence.

## Final local results

The complete suite passed **216 tests, 0 failures, 0 errors, 0 skips** in 3.55 seconds. Ruff check, Ruff format check and Git whitespace checks passed. There are 31 modified and 8 new source/documentation/test files; no production fall weights were trained.

RTMO-m/CoreML warmed image benchmark, 60 measured frames and one person: pose mean 12.19 ms, total mean 12.68 ms, p95 13.13 ms, p99 13.28 ms, 78.83 FPS. Two-person skeleton-only benchmark, 300 measured frames: mean 0.76 ms, p95 0.82 ms, p99 0.86 ms, 1265 FPS. Capture, GUI and encoding are excluded; the skeleton benchmark also excludes pose inference.

The final 60-frame headless webcam smoke retained CoreML and measured 14.2 pose FPS against a 15 FPS target. The final video smoke decoded 60 annotated output frames at the source's 30 FPS and 640×424 dimensions; its debug overlay was inspected. These checks establish execution, not fall recall or false-alarm rates. No Windows GPU hardware or real held-out fall dataset was evaluated.
