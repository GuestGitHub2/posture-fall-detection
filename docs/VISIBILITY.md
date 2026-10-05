# Visibility-aware posture and fall inference

RTMO-m, local runtime/provider fallbacks, tracking gates, history epochs, CLI
flags, latest-frame capture and stale-overlay gates retain their existing roles.
The default fall detector remains heuristic. No model weights were trained.

## Observability contract

`PostureResult.visibility` contains a `BodyVisibility` assessment with reliable
counts and observed confidence for head, shoulders, hips, knees and ankles;
`torso_observable` and `complete_leg` describe available geometry. Groups use
the existing confidence threshold. Out-of-image estimates are excluded when
`Pose.image_size` is known; `Pipeline.process()` supplies the actual dimensions.
`process_poses()` callers can provide `(width, height)` on each pose.

| Visibility | Evidence | Rule output |
|---|---|---|
| FULL_BODY | Observable torso plus at least one complete hip/knee/ankle chain | Precise posture when rule evidence suffices |
| LOWER_PARTIAL | Observable torso with some knees/ankles, no complete leg | Coarse torso orientation |
| TORSO | Reliable shoulders and hips, no reliable knees/ankles | Coarse torso orientation |
| UPPER_ONLY | Head/shoulders but no observable torso/root | UNKNOWN; insufficient for posture |
| INSUFFICIENT | No useful upper/torso groups | UNKNOWN |

One reliable side can supply a group, consistent with the existing paired-joint
fall setting. Missing groups are never synthesized. A high confidence head alone
does not permit posture or fall confirmation. Pose quality is the mean confidence
of reliably observed joints, independent of coverage; `torso_quality` is the
minimum of observed shoulder and hip group quality. The old `joint_confidence`
feature remains the mean over eight lower/torso joints for compatibility.

## Hierarchical posture and stabilization

The rules emit `standing`, `sitting`, `bending`, `lying` (and optional extra
postures) only with useful lower-body geometry. When that geometry is missing
or inconclusive, the available torso angle supports `upright_partial`,
`bent_partial` or `horizontal_partial`. These are anatomical orientation labels;
upright_partial does not establish standing versus sitting.

Signed rule margins produce overlapping continuous memberships on both sides
of a threshold. Good upright geometry at 28.29 degrees no longer falls directly
to UNKNOWN merely because `standing_torso_angle` is 28. Classification confidence
combines memberships and torso quality, while pose quality stays separate.
Per-track confidence hysteresis stabilizes raw rule selection independently of
the final state machine. Ambiguous association and expiration clear that label
memory with the fall evidence. Full-to-partial transitions immediately downgrade
a previously precise final posture; a later precise promotion uses normal
smoothing. At startup, final posture can be UNKNOWN while Raw is already useful.

The optional posture MLP uses its existing full-body contract when observable;
partial or insufficient inputs use the same visibility-aware rule fallback.
Learned class order and full-body model tensors are unchanged.

## Fall evidence tiers

Full-body analysis retains descent, downward speed, rotation,
floor proxies and endpoint persistence. Knees are not mandatory motion inputs,
but losing a complete leg chain selects the more conservative partial tier.
Torso-visible analysis uses reliable hip/shoulder centers, observed orientation
and a stable body scale. It requires stronger descent/rotation, a consistent
downward path, bounded torso-length changes and a settled horizontal endpoint
for longer. Bbox center and aspect are available as diagnostics, but cropping
can change them without body motion, so they cannot independently confirm falls.

Any available initial ankle plane is retained across later partial observations
and can veto an elevated bed/sofa endpoint. With a torso-only origin there is no
floor proxy; strong temporal evidence can still confirm. Static horizontal
observations never create a candidate. Short knee/ankle losses remain valid torso
measurements; shoulder/root loss suspends measurements and persistence.

After strong descent/rotation creates a candidate, loss of torso or a missing
detection can hold `possible_fall` for a bounded timeout. `measurement_valid=0`
and `held_possible=1` suspend all confirmation. The final state displays FALLING,
never a new FALLEN or event merely because someone disappeared. Weak departure
has no candidate and stays normal. Prolonged unreliability still resets the motion
buffer at the existing duration; a separate pending display timeout does not
preserve measurements past that reset. Clock discontinuities, identity ambiguity
and expiration clear both. Stable UPRIGHT_PARTIAL participates in existing timed
recovery and rearming; horizontal observations cannot rearm an episode.

The optional ST-GCN adapter and `window_root_v1` training normalization retain
their declared full-body scale/current-input requirements. The new heuristic
torso path does not imply that an untrained or full-body-only learned model is
validated for partial observations. Partial-capable learned models require an
explicitly compatible training/evaluation contract and weights; none are supplied.

## Settings

All new keys are in `configs/default.yaml` and validated by `src/config.py`.
Existing thresholds, minimum visible joint count and confidence cutoffs were
not reduced. Torso-only fallback scale uses the existing torso multiplier and
EMA timing, kept separate from reliable full-body scale used by tracking gates.

| Key | Default | Units / purpose |
|---|---:|---|
| posture.label_hysteresis_margin | .08 | Confidence advantage required to change a tracked raw label |
| fall.partial_evidence_multiplier | 1.15 | Stronger descent/rapid rotation and tighter available floor tolerance |
| fall.partial_minimum_downward_fraction | .90 | Minimum mostly-downward path consistency |
| fall.partial_lying_persistence_seconds | .70 | Valid settled horizontal endpoint duration |
| fall.partial_max_torso_scale_ratio | 1.30 | Maximum reciprocal torso-length ratio across a motion window |
| fall.partial_maximum_settling_velocity | .25 | Body lengths/second, absolute endpoint root velocity |
| fall.disappearance_possible_seconds | 1.00 | Bounded possible-only hold, limited also by track lifetime |

Partial persistence must be at least existing rapid/slow persistence. Multipliers
and scale ratios must exceed one. Consistency must be no weaker than the slow
path's configured fraction. The existing `state.recovery_postures` and
`state.rearm_postures` default lists now include `upright_partial`; profile lists
remain overridable and are not merged item-by-item.

## Debugging and saved-input diagnostics

Each person shows ID, State, Raw label/confidence, Final posture and Fall
status/score. Debug adds Visibility, Pose quality, Match, scale source/quality and
geometry. Final posture is smoothed independently of combined fall state. Missing
tracks explicitly mark held raw observations; stale detections are not drawn.

```sh
cd /Users/abdullahshahidali/Documents/Fall_test
source .venv/bin/activate
python -m src.app --camera 0 --config configs/mac.yaml --debug
python -m src.app --video artifacts/sample_input.mp4 --config configs/mac.yaml --debug
python -m src.app --video path/to/video.mp4 --output outputs/result.mp4
python tools/diagnose_posture.py --input outputs/saved_frame.png --config configs/mac.yaml --output outputs/frame_diagnostics.jsonl
python tools/diagnose_posture.py --input path/to/video.mp4 --config configs/mac.yaml --max-frames 300 --output outputs/video_diagnostics.jsonl
```

Diagnostic JSONL includes raw keypoints, groups, geometry, matching, scale,
posture/state and fall scores per frame/person. Run the same saved input with an
explicit alternative pose configuration to compare optional installed adapters;
this never changes the production default or installs another model. A single
still image cannot establish a temporal fall or a smoothed state.

## Changed files

| Files | Change |
|---|---|
| `src/pose/visibility.py` | New group-based observability contract |
| `src/pose/pose_types.py`, `src/pose/normalization.py` | Optional image dimensions, boundary-aware normalization and separate torso fallback EMA |
| `src/posture/geometry.py` | Observed quality, group counts, torso quality/length and box-center features |
| `src/posture/rules.py`, `src/posture/classifier.py` | Hierarchical memberships and per-track classification hysteresis |
| `src/posture/mlp.py` | Observable full-body inference and partial/insufficient rule fallback |
| `src/fall/heuristic.py` | Torso evidence, retained floor proxy, partial persistence and possible-only disappearance hold |
| `src/pipeline.py` | Image dimensions, classifier epoch cleanup, missing-candidate updates and separate smoothed posture |
| `src/state/state_machine.py` | Precise-to-coarse downgrade, bounded possible display, partial recovery/rearming |
| `src/visualization/renderer.py` | Explicit raw/final/state/fall labels and visibility |
| `configs/default.yaml`, `src/config.py` | Seven validated keys and additive upright-partial recovery/rearm defaults |
| `tools/diagnose_posture.py` | Local saved-frame/video diagnostic JSONL |
| `tests/test_visibility_posture.py` | Precise/coarse groups, threshold jitter, hysteresis, scale, boundary and MLP regressions |
| `tests/test_visibility_fall.py` | Full/partial rapid/slow paths, hard negatives, weak/strong disappearance and evidence preservation |
| `tests/test_visibility_pipeline_renderer.py` | Mixed visibility, crossings/ambiguity, state transitions, episodes, diagnostics and renderer age/labels |
| `tests/test_visibility_config.py` | Setting validation and default/profile invariants |
| `tests/test_posture.py`, `tests/test_fall.py`, `tests/test_hardening_fall.py`, `tests/test_hardening_tracking.py` | Preserve existing cases with revised expectations for explicitly changed partial/possible semantics |
| `README.md`, `docs/HARDENING.md`, `docs/VISIBILITY.md` | Current behavior, compatibility, settings, limits and validation |

## Compatibility and limits

New dataclass fields are appended with defaults. Existing `classify(pose)`,
`process_poses`, `missing(timestamp)` and CLI calls remain supported. Consumers
must allow the three additional posture/final-state strings. Pose quality now
describes observed quality, not missing-joint coverage; use visibility/groups
for coverage. Missing-track snapshots no longer repeat a stale positive fall
prediction. Existing hardening tests remain, with explicitly obsolete UNKNOWN,
LYING-with-missing-ankles and ankle-suspension expectations updated.

Torso orientation cannot distinguish sitting and standing without lower-body
evidence. Head/shoulders alone remain UNKNOWN. 2D intentional and accidental
motions may be indistinguishable, especially without a floor reference; the
partial tier can trade recall for caution. Perspective/foreshortening, camera
movement and severe/root occlusion remain difficult. No confirmation is possible
after a wholly unobserved fall endpoint. Crowded identity ambiguity still loses
history rather than transferring it, so a fall may be missed. Unit fixtures and
live smoke tests do not establish real held-out recall or false-alarm rates.


## Final validation on this Mac

The full suite passed **300 tests, 0 failures, 0 errors, 0 skips** in 4.94 seconds.
The four new regression files add 84 parametrized cases: posture 26, fall 33,
pipeline/renderer 11 and configuration 14. Ruff check, Ruff format check (85
Python files) and Git whitespace checks passed. Full results are saved locally
in `outputs/visibility_validation.json`, with pytest JUnit XML and check logs.

The non-headless webcam command completed 900 displayed frames with RTMO-m and
CoreMLExecutionProvider plus CPU graph partitions at 14.2 pose FPS. The observed
camera scene was empty; it did not validate live-person posture or fall accuracy.
Its annotated recording was inspected. The video smoke completed 60 frames at
30 FPS and 640×424, and its explicit raw/final/visibility overlay was inspected.

An actual saved-video frame cropped below the observed hips was inferred again
with RTMO-m: TORSO visibility (two shoulders, two hips, no knees or ankles),
`upright_partial` confidence .758, observed pose quality .924. This verifies
useful coarse output on that saved partial view, not a general accuracy result.
The crop and diagnostic records are under `outputs/visibility_*`.

Warm real one-person image benchmark (60 frames after 10 warmup frames): pose
mean 12.07 ms; inference pipeline mean 12.77 ms, p95 13.07 ms, p99 13.23 ms;
78.28 FPS. Two-person skeleton-only benchmark (300 frames after 30 warmup): mean
1.10 ms, p95 1.15 ms, p99 1.20 ms; 882 FPS. Capture, GUI and encoding are excluded;
the synthetic benchmark also excludes pose inference. Windows hardware and
held-out real fall accuracy were not evaluated.
