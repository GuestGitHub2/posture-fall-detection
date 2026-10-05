# Skeleton datasets, training and evaluation

All tools run locally from the repository root. Install the optional trainer:

```sh
python -m pip install -e '.[training]'
```

## Records and extraction

Organize videos by subject and activity. Annotations are a JSON mapping of
relative video paths to metadata:

```json
{
  "subject01/sideways_fall.mp4": {
    "subject_id": "subject01",
    "label": "normal",
    "fall_onset": 3.2,
    "segments": [{"start": 3.2, "end": 7.1, "label": "fall"}]
  },
  "subject02/lying_on_sofa.mp4": {
    "subject_id": "subject02", "label": "normal"
  }
}
```

```sh
python training/extract_skeletons.py --input datasets/videos --output datasets/skeletons --annotations datasets/annotations.json
```

The extractor reads every decoded video frame and reuses local pose weights,
while creating fresh tracking for each video. No posture/fall inference or event
logging is needed for extraction. Completed JSONL files install atomically;
existing JSONL files are
skipped unless `--overwrite` is passed. A track ID is local to its video;
`subject_id` represents a human participant and must be assigned from actual
dataset annotations, never inferred from a track ID. In multi-person clips,
the supplied video-level labels apply to all visible people; edit extracted
JSONL per track before training if only one person falls. Dataset inspection
and track/label verification are necessary.

Each JSONL record has:

```json
{"sequence_id":"subject01/sideways_fall","frame_number":96,"track_id":1,"timestamp":3.2,"keypoints":[[200,100,0.9]],"label":"fall","subject_id":"subject01","fall_onset":3.2}
```

The abbreviated `keypoints` above must contain **all 17 COCO joints** in a real
record, each `[x_pixels,y_pixels,confidence]`. Bboxes are also saved by extraction
and used for confidence-aware scale measurements. Extraction uses the same
monotonic media timestamps as video inference. Records also include
`history_epoch`, association confidence and scale source/quality. Ambiguous
matches are omitted; epoch changes prevent training windows from bridging an
uncertain identity. Older records without epochs default to zero and should be
audited for ID swaps before reuse.
The loader requires finite coordinates and confidence in `[0,1]`. Posture
training remains per-frame hip centered. Fall windows use `window_root_v1`,
exactly as the ONNX adapter does.

For posture, annotate standing, sitting, bending, lying, squatting, kneeling or
unknown. For fall training, annotate normal or fall, with fall labeling beginning
at the chosen event phase. Falling/fallen are also mapped to the positive class;
recovery maps to normal. Do not label all lying sequences as fall. Exact onset
definition affects measured latency, so use the same definition throughout.

## Split and augment

Training commands connect `subject_id` and recording membership, then split
these independent groups; records without subjects split by whole sequence.
Participants appearing together in a video stay in the same split. At least
two independent groups are required. Neither a track nor a
window from a subject crosses train/validation splits. Keep the test set entirely
separate from both. Verify class representation in each split; a random grouped
split cannot manufacture missing minority classes.

Implemented augmentations: COCO-aware horizontal flipping, normalized coordinate
noise, joint dropout, confidence attenuation, small scale variation, temporal
speed variation/cropping while keeping the causal endpoint. Missing joints are
never revived. A constant image translation disappears under normalization;
time-varying root translation is retained in fall windows. Exercise camera
framing and perspective in collected videos. Do not recenter each fall frame
independently during augmentation, as that would erase descent again.

## Train/export posture

```sh
python training/train_posture.py --data datasets/posture --epochs 40 --output models/posture_mlp.pt --export models/posture_mlp.onnx
```

The small MLP input is float32 `[N,17,3]` normalized x/y/confidence; output is
`[N,C]` logits. A sidecar `.metadata.json` records the class order. To use fewer
classes, pass e.g. `--classes standing sitting bending lying unknown` and keep
that exact ordering in your deployment config. Validation loss selects the saved
checkpoint; `.history.json` stores the actual training history.

Activate the optional MLP with a profile such as `configs/learned_posture.yaml`:

```yaml
posture:
  backend: mlp
  model_path: models/posture_mlp.onnx
  classes: [standing, sitting, bending, lying, squatting, kneeling, unknown]
```

```sh
python -m src.app --camera 0 --config configs/learned_posture.yaml
```

Use the exact class order from the export's sidecar. Rules remain the default
until the trained model and class mapping are supplied.

## Train/export fall

```sh
python training/train_fall.py --data datasets/falls --samples 48 --history-seconds 2.5 --stride-seconds .5 --joint-confidence .4 --max-gap-seconds .5 --epochs 50 --output models/fall_temporal.pt --export models/fall_temporal.onnx
```

This trainer is a compact graph-temporal baseline with normalized COCO adjacency
message passing, residual temporal convolution blocks and global pooling. It is
an equivalent lightweight skeleton temporal model, **not an implementation of
all ST-GCN++ architectural refinements**. There are no pretrained fall weights.
The checkpoint is trainable and the ONNX export is deployable; synthetic smoke
weights test software paths only and must never be treated as a trained detector.

The contract for it and compatible ST-GCN++ exports is:

- float32 input `skeleton`, `[N,C,T,V,M] = [1,3,48,17,1]` at deployment;
- channel order normalized x, normalized y, joint confidence;
- COCO-17 order from `src/pose/pose_types.py`;
- 48 uniform causal samples over the preceding 2.5 seconds by default;
- output `logits`, `[1,2]`, class order normal, fall;
- missing/long-gap joints zero confidence, short gaps interpolated;
- one tracked person's history per invocation.

`window_root_v1` chooses the first confident hip/root center within each window
and the median reliable torso-plus-leg-length measurement across that window.
Every frame uses this **same origin and scale**: `(pixel_xy - origin) / scale`.
The root's y trajectory therefore retains descent. Confidence is a third channel;
unobserved joints and padding have zero confidence. Short joint gaps interpolate
only within the configured maximum gap; no extrapolation is used. Identity-epoch
changes and long processing gaps split windows. Windows without a reliable root
and body scale are excluded from training and cannot trigger model inference.

The checkpoint and ONNX sidecar record `normalization: window_root_v1`, origin,
scale policy, class order, samples, history duration, joint confidence threshold,
minimum scale quality and interpolation gap. Keep those preprocessing values
identical in deployment. The ONNX adapter checks provided metadata, and the
checkpoint evaluator rejects old per-frame-centered fall checkpoints. Old fall
weights must be retrained; renaming their normalization metadata is invalid.
Externally exported weights without a sidecar must be independently verified
against this exact contract. No pretrained or production fall weights were
trained or fabricated for the hardening work.

Create a learned profile:

```yaml
fall:
  detector: stgcn
  model_path: models/fall_temporal.onnx
  sample_count: 48
  history_seconds: 2.5
  min_history_seconds: 2.0
  normalization: window_root_v1
  minimum_joint_confidence: 0.4
  minimum_scale_quality: 0.6
  max_gap_seconds: 0.5
  classes: [normal, fall]
  output_kind: logits
  min_visible_fraction: 0.35
```

```sh
python -m src.app --video path/to/video.mp4 --config configs/learned_fall.yaml
```

For official ST-GCN++, train/fine-tune using
[MMAction2 ST-GCN++](https://github.com/open-mmlab/mmaction2/tree/main/configs/skeleton/stgcnpp)
in a separate compatible training environment. Export the actual trained module
with `torch.onnx.export`, opset 17, a `[1,3,T,17,1]` example and class logits.
Generic NTU models typically use a different joint layout and are not directly
compatible. Remove or wrap framework preprocessing to make normalization,
joint order and input dimensions explicit; validate PyTorch versus ONNX logits
on held-out skeletons before deployment. No conversion invents fall knowledge
from a general action checkpoint.

## Dataset coverage

Falls: forward, backward, sideways, collapse, stumble then fall, chair-related.
Hard negatives: quick/slow sitting, bending, tying shoes, kneeling, squatting,
lying on bed/sofa, getting into/out of bed, sitting/getting up from floor, picking
objects up, exercise, jumping, crawling. Record transitions and extended normal
periods, rather than only short curated positive clips. Include multiple people,
partial frames, occlusion and missing poses. Stage falls only with appropriate
participant supervision and safe physical conditions.

## Evaluate

```sh
python training/evaluate.py --mode posture --data datasets/test_posture --checkpoint models/posture_mlp.pt --output outputs/posture_eval.json
python training/evaluate.py --mode fall --data datasets/test_falls --checkpoint models/fall_temporal.pt --threshold .8 --output outputs/fall_eval.json
```

Posture reports per-class precision/recall/F1, support and confusion matrix.
Fall reports binary sensitivity/specificity/precision/F1, per-sequence detection,
event episodes, false alarms per observed normal hour and onset latency. A fixed
0.25-second causal evaluation stride is used for learned fall checkpoints;
this measures the model before production confirmation/smoothing. To evaluate
the production state machine, supply recorded decisions instead:

```sh
python training/evaluate.py --mode fall --predictions datasets/heldout_predictions.jsonl --cooldown-seconds 10 --max-gap-seconds 1 --output outputs/event_eval.json
```

Prediction records include `sequence_id`, `track_id`, `timestamp`, ground-truth
`label` (normal/fall), `prediction` (normal/fall) and optional `fall_onset`.
A rising fall prediction is one alarm; sustained positives do not create repeated
alarms. `--cooldown-seconds` is the evaluator's legacy prediction-stream protocol,
not the application's recovery/rearming semantics. For actual production
evaluation, retain the event `episode_id` and its source timestamp and count
emitted episodes directly; use `--cooldown-seconds 0` for streams whose rising
edges already represent emitted episode decisions. Latency is first true alarm
minus annotated onset, only when both exist. Missing onset or zero normal
exposure gives null, not a fabricated zero. False alarms/hour requires continuous
annotated normal recordings; intervals exceeding `--max-gap-seconds` do not
count as observed time. Concurrent tracks' normal intervals are merged, so a
two-person scene does not silently double the recording-hours denominator.
Read the JSON protocol notes when comparing metrics.
Report each hard-negative sequence/category separately and avoid selecting
thresholds on the final test set. Checkpoint accuracy alone does not establish
fall-alarm performance.
