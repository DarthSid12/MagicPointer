# Prompt 15: alternative hand detection and pose libraries

Reviewed primary project documentation/papers and first-person Reddit discussions
on 2026-10-21. No alternative was installed or benchmarked locally. “Better” must
separate finding a hand, locating finger joints, 3D geometry, temporal stability
and two-stream latency. The user's logs establish camera-dependent missing
detections, not a ground-truth pose-accuracy ranking.

## Recommendation for this project

Benchmark **RTMDet's hand detector + RTMPose-Hand**, using `rtmlib`/ONNX Runtime
as a possible deployment path, against MediaPipe first. This is an engineering
recommendation based on available hand-specific models and deployment options,
not proof of superior recall at ten feet. If a suitable GPU is available, **WiLoR**
is a second candidate because its design explicitly addresses hand localization
as well as reconstruction. Do not replace the current runtime on reputation alone.

The current uncalibrated gesture/fusion code consumes MediaPipe's per-hand 3D
world landmarks. A 21-point 2D estimator is not a direct substitute: a detector
comparison can evaluate availability first, but production integration would need
appropriate geometry and confidence handling. Calibrated stereo can triangulate
matched 2D joints; it does not supply equivalent uncalibrated per-view 3D output.

## Candidate assessment

| Candidate | Relevant capability | Fit / limitations |
| --- | --- | --- |
| RTMDet hand + RTMPose-Hand / MMPose | Hand-specific localization and 21 2D landmarks | First comparison candidate; separate detector and pose stages, no equivalent per-view 3D world landmarks |
| WiLoR | Learned hand localization plus transformer 3D reconstruction | Interesting for missed hands and challenging views; measure full two-camera latency and resource use |
| HaMeR | Transformer-based 3D hand reconstruction | Strong reconstruction research; obtaining reliable hand crops remains a separate concern |
| OpenPose | Established body/hand keypoint system | Useful baseline, not established as a universal MediaPipe upgrade |
| YOLO pose trained for hands | Custom detector/keypoint pipeline | Potentially adaptable to the actual room/cameras; hand-specific weights/data are required |
| Sapiens / SDPose / whole-body models | Higher-capacity full-person/hand estimates | Relevant research baselines, less attractive first candidates for a lightweight two-camera app |

### RTMPose-Hand and rtmlib

The [official RTMPose model zoo](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose#hand-2d-21-keypoints)
provides a 320x320 RTMDet-nano hand detector and 256x256 RTMPose-m Hand5
estimator. Hand5 combines five public datasets. These hand-specific models are
the relevant comparison; headline COCO body-pose FPS/AP cannot be applied to
two-camera finger tracking. Published hand benchmarks are not a direct MediaPipe
comparison on this room or hardware.

[rtmlib](https://github.com/Tau-J/rtmlib) documents low-level hand RTMPose APIs,
ONNX hand models and CPU/ONNX Runtime deployment without the full MMCV/MMPose/
MMDetection stack. A detector-to-crop-to-keypoint pipeline still needs integration;
it is not sufficient to instantiate the default body model. Windows CPU throughput
for the actual detector plus both hand estimators remains to be measured.

### WiLoR and HaMeR

The [WiLoR paper](https://arxiv.org/abs/2409.12259) explicitly targets the gap in
hand localization and combines it with 3D reconstruction. Its authors describe
training detection on over two million in-the-wild hand images covering varied
conditions. This makes it relevant to our raw-detection bottleneck, without
establishing a guarantee for ten-foot webcams. The [official setup](https://github.com/rolpotamias/WiLoR)
was tested with PyTorch/CUDA; its [demo](https://github.com/rolpotamias/WiLoR/blob/main/demo.py)
also selects CPU when CUDA is unavailable. CPU support does not establish usable
interactive performance, especially on two streams.

[HaMeR](https://arxiv.org/abs/2312.05251) reports strong 3D reconstruction results
using a large transformer and combined training datasets. Its [official demo](https://github.com/geopavlakos/hamer/blob/main/demo.py)
uses additional detection/pose stages to find hand crops. Better reconstruction
of an already-found hand does not automatically remedy a missing hand detector.
The [installation](https://github.com/geopavlakos/hamer) also requires model assets
and additional dependencies; it is not a small drop-in replacement.

### YOLO and OpenPose

[Ultralytics' hand-keypoint documentation](https://docs.ultralytics.com/datasets/pose/hand-keypoints/)
describes training hand-specific pose models. Importantly, its documented dataset
uses MediaPipe-generated landmark annotations. Training on those pseudo-labels
does not independently prove correction of MediaPipe's landmark errors. A custom
model could be evaluated, but would require suitable hand weights and independent
evaluation labels. General human-body pose weights are not finger-keypoint models.

[OpenPose's official project](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
supports hand and body estimation, including Windows demos. The comparison
research below provides no basis for assuming it will outperform our current
detector in the user's room. Its hand localization path and complete pipeline
speed must be evaluated rather than comparing rendered skeletons alone.

## Comparison articles / studies

1. [Pu et al., 2023: robustness of hand pose estimators](https://arxiv.org/abs/2303.04566).
   Compares MediaPipe, BodyHands+OpenPose and NSRM across occlusion, exposure and
   motion-blur transformations on FreiHAND/Panoptic. MediaPipe's baseline detection
   recall exceeds the BodyHands+OpenPose combination on the reported datasets;
   both deteriorate under difficult conditions. This argues against OpenPose as
   an automatic upgrade. These older model versions and synthetic degradations
   do not benchmark the current Windows rig or RTMPose/WiLoR.
2. [O'Brien et al., 2026: pose estimators for sign-language translation](https://arxiv.org/abs/2604.24609).
   Compares eight systems and analyzes missing hand points, jitter and occlusion.
   SDPose/Sapiens outperform the MediaPipe Holistic baseline on downstream
   translation in that study, with substantial throughput tradeoffs in its
   hardware measurements. The occlusion sample is small. Translation quality is
   not mouse-control accuracy; Holistic is not our standalone Hand Landmarker.
   No reported speed should be treated as our laptop's two-camera FPS.

I did not treat body-only comparisons, old wrappers around MediaPipe, or model-zoo
scores on different tasks as proof that a library is better at finding our hands.

## What Reddit users report

These are small, uncontrolled anecdotes, not a benchmark or community consensus:

- [Badminton hand-keypoint discussion, June 2025](https://www.reddit.com/r/computervision/comments/1l7s26d/):
  users describe MediaPipe problems with grip occlusion and recommend trying
  RTMPose through rtmlib. A YOLO suggestion explicitly admits the commenter has
  not compared the two for hand tracking. Useful leads, not measured superiority.
- [Egocentric hand tracking discussion, May 2026](https://www.reddit.com/r/computervision/comments/1tkxjky/hand_tracking_tools_egocentric_videos/):
  one commenter reports decent OpenPose results but calls it less lightweight.
  That use case differs from the user's seated external-camera view.
- [Hand landmark alternatives, August 2024](https://www.reddit.com/r/computervision/comments/1em42sn/):
  suggestions include OpenPose and DeepLabCut; the latter is described as needing
  collected/annotated training data. This is not an off-the-shelf accuracy finding.

## Recommended next evaluation, not implemented here

Compare MediaPipe and RTMDet+RTMPose-Hand on identical live raw frames, first at
the present resolution and camera arrangement. Keep separate counters for both
physical cameras, distant/near conditions, detection gaps, false hands, joint
quality, latency and processing FPS. Include explicit intended poses and no-hand
periods so extra detections cannot masquerade as accuracy. Retain data-only logs
unless the user changes the recording preference. Existing JSON-only archives
cannot be rerun through a different visual model because they contain no images.

The laptop GPU/CPU and end-to-end performance need checking before a model choice.
Focus/lighting/viewpoint differences still matter to every candidate. No library
was found proven universally better for this exact ten-foot two-camera setup.

## Branch removal and validation

User explicitly cancelled feature 14. Returned to main and deleted the unmerged,
unpublished feat/filter-debug-experiment branch (last commit 1878fe5). Remote
branch lookup returned no matching ref. No filter runtime files were merged into
main; the adaptive experiment remains on feat/adaptive-debug-detection. Historical
feature 14 and removal prompt 15 are retained in FEATURES.md. No dependencies or
runtime behavior changed. Branch/ref/file checks and whitespace/ledger validation
apply; model/UI tests were not rerun for this research and branch-removal task.
