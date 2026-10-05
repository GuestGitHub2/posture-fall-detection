# Local models

Inference uses local ONNX files only. It never fetches weights or calls a cloud API.
Large weights, cache files and download provenance are excluded from Git.

## Default: RTMO-m

```sh
python tools/download_models.py
python tools/export_models.py inspect models/rtmo_m.onnx
```

The default is the official **RTMO-m Body7 640 × 640 COCO-17 ONNX SDK** model,
not a fabricated or randomly initialized pose network. It detects every person
in one forward pass. Source: [MMPose RTMO project](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo).
Upstream [Apache-2.0 license](https://github.com/open-mmlab/mmpose/blob/main/LICENSE).
The model was trained on Body7 datasets; their dataset/media rights are distinct
from the code license. Review dataset rights for your intended redistribution.

`manifest.json` records official model URLs and upstream licenses. The downloader
uses an atomic temporary file, extracts only the one ONNX file, retains SDK JSON
metadata in `<name>.provenance.json`, and records archive and ONNX SHA256 hashes.
No upstream SHA256 was published with these downloads; `--sha256 <archive-hash>`
allows a trusted expected hash, and subsequent installations verify against the
recorded ONNX checksum. The URL hash suffix is not represented as a SHA256.

## Optional RTMPose fallback

```sh
python tools/download_models.py --model rtmpose_bundle
```

Configure `pose.model: rtmpose_m`, `pose.model_path: models/rtmpose_m.onnx`, and
`pose.detector_path: models/yolox_tiny.onnx`. RTMPose-m is a top-down COCO-17
SimCC network (256 × 192). A separate official YOLOX-tiny HumanArt+COCO ONNX
person detector (416 × 416) supplies the boxes. This is YOLOX, which uses
[Apache-2.0](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE),
and does not depend on Ultralytics. Top-down pose inference runs once per person,
so latency scales with crowd size. Both adapters were smoke-tested on a real
sample using CoreML and CPU; this is not a crowd or accuracy evaluation.

## Provider compatibility

macOS auto order is CoreML → CPU. Windows auto order is CUDA → DirectML →
OpenVINO → CPU, selecting only installed providers. CoreML model compilation
can make first startup slow. CoreML uses MLProgram by default; configurable
`runtime.provider_options.CoreMLExecutionProvider` entries override this.
Unsupported operators may run on CPU even with an accelerated provider selected.
Initialization and execution errors trigger fallback without network activity.

The default CoreML session fixes the symbolic `batch` dimension to 1 and sets
`RequireStaticInputShapes: "1"`. This keeps the image backbone eligible for
acceleration while dynamic person-count/NMS output subgraphs run on CPU. The
official ONNX weights are unchanged. This combination was tested on alternating
person and empty images: empty detections remain valid without dropping the
whole session to CPU. Allowing dynamic CoreML input shapes (`"0"`) caused a
zero-element tensor failure on empty live-camera scenes in the tested runtime.
`runtime.free_dimension_overrides: {batch: 1}` controls the dimension override;
the video pipeline always sends one image per inference call.

Different official SDK exports can have incompatible layouts. RTMO expects
`[1,N,5]` xyxy/confidence boxes and `[1,N,17,3]` xy/confidence joints in input-image
pixels, after a top-left 114 BGR letterbox. RTMPose expects two SimCC tensors,
`[1,17,2W]` and `[1,17,2H]`. Custom ONNX exports must preserve these contracts or
use a separate adapter.

## Exporting new pose weights

The pre-exported ONNX asset above requires no PyTorch/MMCV installation.
If you retrain pose weights, use a separate compatible MMPose/MMDeploy environment
and the [official RTMO export instructions](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo#onnx-model-export).
From the MMDeploy checkout, the official RTMO deploy configuration is
`configs/mmpose/pose-detection_rtmo_onnxruntime_dynamic-640x640.py`.
The wrapper accepts actual paths and does not install the framework:

```sh
python tools/export_models.py mmdeploy \
  --mmdeploy-root /path/to/mmdeploy \
  --deploy-config /path/to/mmdeploy/configs/mmpose/pose-detection_rtmo_onnxruntime_dynamic-640x640.py \
  --model-config /path/to/mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py \
  --checkpoint /path/to/trained.pth --image /path/to/test.jpg \
  --output-dir models/exported --device cpu
```

## Fall and posture weights

No fall-trained ST-GCN++ weights are supplied or fabricated. The default temporal
fall detector and posture classifier are geometric heuristics. A learned fall
model must be trained on falls and hard negatives, with the exact normalization,
joint layout, sequence duration and output class mapping used by the app. See
`training/README.md` and `src/fall/stgcn.py` for the input contract. An action model
trained on generic activities is not automatically a valid fall detector.

Fall preprocessing now requires `normalization: window_root_v1`: a fixed first
reliable window hip origin and a fixed robust body scale, preserving root descent.
Per-frame hip-centered fall weights are incompatible and require retraining.
Posture MLP preprocessing remains per-frame centered. Keep the trained fall
ONNX `.metadata.json` sidecar with the model; class order, sampling duration,
confidence thresholds and gap interpolation must match deployment. RTMO input
remains BGR, top-left 114 padding, float32 NCHW with no pixel normalization;
NMS uses restored unclipped boxes, clipping only retained boxes.
