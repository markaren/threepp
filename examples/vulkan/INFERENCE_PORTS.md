# Framework-free detector inference on Vulkan compute (YOLOv8n · RT-DETR · RF-DETR)

Umbrella notes for the three object detectors reimplemented as hand-written GLSL
compute shaders on threepp's Vulkan path, sharing one device/memory space with the
renderer. Deep dive for RF-DETR: `rfdetr/PORTING_NOTES.md`.

All numbers: NVIDIA RTX 4070, fp32, end-to-end (preprocess + forward + postprocess),
lean pre/post (not the framework convenience API), `cudnn.benchmark` on for baselines.

---

## Paper-ready methodology abstract (drop-in)

> We evaluate a framework-free inference path in which modern object detectors are
> reimplemented as hand-written GLSL compute shaders on Vulkan, sharing a single GPU
> device and memory space with a real-time renderer and requiring no runtime ML
> framework. Each model is ported under a validation-first protocol: per-layer
> activations captured from the reference PyTorch implementation on a fixed seeded
> input serve as ground truth, and every compute stage is verified element-wise
> (maximum absolute error < 2×10⁻²) before integration. Model architecture is pinned
> by reading the reference source directly rather than inferring it from
> documentation, which proved necessary — several load-bearing details (decoder
> depth, feature-tap indexing, attention windowing) differ from the obvious defaults.
> Performance is measured against optimized PyTorch baselines (TorchScript, and
> TensorRT where available) using a minimal preprocessing/postprocessing pipeline
> rather than the framework's convenience API, with cuDNN convolution autotuning
> enabled; forward-only and end-to-end latency are reported separately.

> Across detectors spanning the CNN (YOLOv8n) and transformer (RT-DETR, RF-DETR)
> families, the hand-written Vulkan path is competitive with optimized PyTorch
> end-to-end on a desktop NVIDIA GPU — at parity for the CNN and within ~10% for the
> transformer — despite not matching vendor-optimized cuBLAS/cuDNN kernels on raw
> forward compute. The contribution is therefore not peak throughput but deployment
> properties: a single dependency-free binary, vendor-portable across any
> Vulkan-capable GPU (including mobile Mali/Adreno and integrated Intel/AMD GPUs,
> where TensorRT is unavailable), with perception co-resident on the rendering device.

(Reword/trim to taste; the claims are calibrated to what the benchmarks actually
support — do not strengthen "competitive" to "faster.")

---

## Cross-port benchmark (RTX 4070, fp32, end-to-end)

| model | Vulkan (this work) | optimized PyTorch (TorchScript, lean e2e) | read |
|---|---|---|---|
| YOLOv8n @640 | 7.9 ms (126.8 FPS) | 8.2 ms (121.5 FPS) | **parity** (within noise) |
| RF-DETR-Nano @384 | ~23.6 ms (42 FPS) | 21.5 ms (46.5 FPS) | within ~10% |
| RT-DETR-L @640 | ~47 ms (21.5 FPS) | *re-run `bench_rtdetr_pytorch.py` with the fair protocol* | unverified — see note |

Why YOLO reaches parity but RF-DETR doesn't: YOLO's forward at 640 is large enough
to saturate the GPU and its e2e (~8 ms) is host-sensitive, where the lean C++
pre/post competes; RF-DETR-Nano's GEMMs are tiny (N=384 → ~60 workgroups on 46 SMs),
so the forward is GPU-underutilized and cuBLAS's advantage shows.

> Note on RT-DETR: its 21.5 FPS was tuned earlier against an ultralytics number that
> predates the fair-baseline rigor in this repo (eager-vs-optimized, `cudnn.benchmark`,
> lean pre/post). Re-measure with `scripts/bench_rtdetr_pytorch.py` before citing a
> comparison.

---

## Per-port summaries

### YOLOv8n (CNN; the first port)
- Pure-CNN backbone + detect head; NMS on the host. 640×640 letterbox.
- The simplest port and the one that reaches parity with optimized PyTorch e2e.
- Optimization sensitivity: **fp16 helps** (convs are compute-bound → tensor cores);
  TorchScript barely helps (few big conv ops, already cuDNN-dispatched).
- Baseline: `scripts/yolo_benchmark.py` (eager / TorchScript / fp16 / TensorRT-if-present;
  `--legacy` reproduces the old unfair per-run-decode benchmark).

### RT-DETR-L (transformer; second port)
- HGNetv2 backbone → AIFI transformer encoder → CCFM neck → deformable decoder + heads.
- Established the shared `VkInfer` harness and the register-blocked GEMM / im2col /
  attention / MSDeformAttn shaders later reused by RF-DETR.
- Validated bit-exact (~1e-5) vs captured PyTorch activations; correct detections on
  bus.jpg. Optimized 198 → 47 ms.
- fp16 and VK_KHR_cooperative_matrix (tensor-core) GEMM both gave **no speedup** —
  the workload is memory-bandwidth-bound on f32 activations, not ALU-bound. Reverted.
- Same top-K-on-noise caveat as RF-DETR: validate the decoder on a real image.

### RF-DETR-Nano (transformer; third port)
- DINOv2-windowed ViT-S backbone + C2f projector + two-stage LW-DETR deformable decoder.
- Full deep-dive, insights, bugs, and optimization log in `rfdetr/PORTING_NOTES.md`.
- Bit-exact through the projector; ~42 FPS after optimization.

---

## Shared optimization & benchmarking lessons (all three)

- **Profile before optimizing.** Small models underutilize the GPU; the bottleneck
  is structural (small-N GEMM occupancy), not fixable by op fusion.
- **fp16 / tensor cores help CNNs, not these transformers** (memory-bound) — a
  consistent, citable cross-architecture finding.
- **Benchmark only against optimized PyTorch**, with `cudnn.benchmark` on and a lean
  pre/post; the framework's `predict()` convenience API carries large host tax and is
  not a deployment path. Getting this wrong flipped conclusions twice during this work.
- **The honest pitch is portability + framework-free + unified-device**, not speed.
  TensorRT (uninstalled here) would widen the raw-compute gap; latency claims stay
  "competitive."

## Reproducibility
- Weights/refs: `scripts/export_*_weights.py`, `scripts/capture_*_activations.py`.
- Validation: `<exe> --validate <weights> <ref.bin>` (per-layer element-wise diff).
- Detection: `<exe> <image> <weights>`.
- PyTorch baselines: `scripts/bench_rfdetr_pytorch.py`, `bench_rtdetr_pytorch.py`,
  `yolo_benchmark.py`.
