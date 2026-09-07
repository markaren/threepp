# RF-DETR-Nano → Vulkan compute: porting log & insights

An engineering record of how the RF-DETR-Nano detector was reimplemented from
scratch on threepp's Vulkan compute path (`examples/vulkan/rfdetr/`), how it was
validated, optimized, and honestly benchmarked. Builds on the earlier YOLOv8n and
RT-DETR ports that established the shared `VkInfer` harness and most shaders.

Target/host for all numbers: NVIDIA RTX 4070, fp32, 384×384 input.

---

## 0. What was built

A framework-free forward pass of RF-DETR-Nano (Roboflow `rfdetr` package; the
LW-DETR architecture = DINOv2-windowed ViT-S backbone + C2f projector + two-stage
deformable decoder) running entirely on hand-written GLSL compute shaders via a
~450-line example-local Vulkan harness. No PyTorch / ONNX Runtime / TensorRT at
runtime; shares the GPU device and memory space with the threepp renderer.

End state: bit-exact vs the PyTorch reference through the projector; correct
detections on a real image (bus.jpg → bus + 4 people); ~42 FPS end-to-end.

---

## 1. Methodology principles (the meta-lessons)

These shaped every phase and are the reason the port converged instead of
turning into an unbounded debugging slog.

1. **Validation-first, layer by layer.** Before writing any GPU code, a Python
   script captured per-layer activations from the real PyTorch model on a *seeded
   random* input (no normalization) into a binary. Every GPU stage was then diffed
   element-wise against its captured counterpart (`maxAbsErr < 2e-2` = OK). This
   turned "the detector is wrong somewhere" into "layer N is wrong" — the single
   highest-leverage decision in the whole effort.

2. **Pin the architecture from source, never from assumptions.** The `rfdetr`
   package was installed and read directly. Several load-bearing details are
   *not* what a reasonable guess would produce (see §3).

3. **Reuse a proven harness.** The `VkInfer` harness, the register-blocked GEMM,
   im2col, attention, and msdeform shaders came from the prior RT-DETR port. The
   RF-DETR-specific work was the windowed ViT, the projector, and the LW-DETR
   decoder wiring. Standing on prior work is what made a multi-thousand-line
   numerical reimplementation tractable in one effort.

4. **Profile before optimizing; trust measurements over intuition.** Every
   optimization was gated on a GPU-timestamp profile, and several "obvious" wins
   turned out neutral or negative (see §6).

---

## 2. Phase order

1. Pin architecture from the `rfdetr` source.
2. Weight export + activation capture scripts (Python).
3. Example dir + harness + shader skeleton (copied from RT-DETR).
4. Backbone: patch embed → windowed embeddings → 12 ViT layers. *(validated)*
5. Projector: de-window taps → C2f → channel LayerNorm. *(validated)*
6. Two-stage deformable decoder + heads. *(validated, modulo §5 caveat)*
7. Example: ImageNet preprocess + sigmoid/threshold postprocess + visualization.
8. Optimization (profile-driven).
9. Honest benchmarking vs PyTorch (and YOLO).

---

## 3. Architecture details that had to be read, not guessed

- **`dec_layers = 2`, not 3.** The base config says 3; `RFDETRNanoConfig` overrides
  to 2. Caught because the decoder weights only contained layers 0 and 1 (an
  `unordered_map::at` throw on `layers.2`). Lesson: trust the checkpoint's actual
  keys over the config defaults.
- **Tap layers are {2, 5, 8, 11}, not {3, 6, 9, 12}.** `out_feature_indexes =
  [3,6,9,12]` are *stage* indices, and `hidden_states[k]` = the output *after layer
  k-1* (the encoder appends the input embedding as `hidden_states[0]`). So the
  0-indexed layer outputs tapped are {2,5,8,11}, each with the final encoder
  LayerNorm applied.
- **Global (full) attention at 0-indexed layers {3, 6, 9}**; windowed elsewhere.
  Derived from `window_block_indexes = range(13) − {3,6,9,12}` and
  `run_full_attention = i not in window_block_indexes`.
- **No register tokens** for Nano (`num_register_tokens = 0`).
- **`lite_refpoint_refine = True`** → decoder reference points are *fixed across
  layers* (computed once), so query_pos is computed once and there is no per-layer
  box refinement. **`bbox_reparam = True`** → boxes via `cxcy = Δ·ref_wh + ref_cxcy;
  wh = exp(Δ)·ref_wh`, never sigmoid.
- **Decoder query content is a learned embedding** (`query_feat`), *not* the
  gathered encoder memory. The two-stage path only supplies the reference boxes.
- **Projector is ConvNeXt-style:** channel-wise LayerNorm (normalize over C at each
  spatial position) + SiLU; convs are bias-free. `projector_scale = ["P4"]` → a
  single stage at scale 1.0 (identity sampling), concat of all 4 taps → C2f → LN.

### The key insight that deleted a class of bugs

The windowed↔global attention reshape in DINOv2 is a **plain row-major `.view()`**.
So a single flat **window-major `[580, 384]` token buffer serves both regimes** —
windowed layers attend with `(B=4, T=145)`, global layers with `(B=1, T=580)`, and
the *only* difference is the `(B, T)` passed to the attention shader. No data
reshuffle between layers. Recognizing this avoided maintaining two layouts and the
attendant bug surface.

---

## 4. The projector bug (and how the torch reference found it)

After the ViT validated bit-exact, the taps validated but `proj0` was 61% wrong
(relL1). A NumPy/torch reproduction of the projector *from the captured taps*
matched the reference at 7e-6 — proving the math/understanding was right and the
bug was purely in the GPU code.

Root cause: `sliceCh_` and the C2f concat buffer used 2-D shapes `{C, HW}`. The
generic conv infers H and W from the tensor shape, so the **3×3 bottleneck convs
saw `H=576, W=1`** (a 576×1 strip) instead of `24×24` — completely wrong neighbor
sampling. The 1×1 convs only use `H·W`, so they were unaffected — which is exactly
why the taps (fed through 1×1-equivalent paths) passed while proj0 broke. Fix:
keep intermediate buffers 3-D `{C, 24, 24}`.

Lesson: when a downstream stage is wrong but its inputs are right, reproduce the
stage in NumPy from the captured inputs first — it cleanly separates "my
understanding is wrong" from "my GPU code is wrong."

---

## 5. The top-K-on-noise caveat (validating the decoder)

`pred_logits`/`pred_boxes` matched the reference except for **3–5 of 300 query
rows** (relL1 ~3e-4). Cause: the two-stage selection does `top-300 of 576`
proposals by max class score; on a *seeded-random* input the scores are
near-uniform, so ~1e-5 numerical drift reorders or reselects a handful of
near-tied proposals, permuting a few output rows. A structural bug would corrupt
*all* rows. Confirmed benign by counting divergent rows and by correct detections
on a real image (where high-confidence detections make the top-K stable). Lesson:
DETR-style top-K decoders cannot be validated bit-exact on noise — use a real
image for the decoder, layer activations for everything upstream.

---

## 6. Optimization (profile-driven)

Profiling used GPU timestamps (`RTDETR_PROFILE`), per-ViT-layer then per-op.

**Findings.** The 12 ViT layers dominate (~14 of ~17 ms backbone); decoder ~2 ms;
~6 ms is CPU/submit overhead. Per-op, only `fc2` (N=384, K=1536) and `dense`
(N=384) exceed 0.3 ms/layer. The tell: `fc1` has *identical* MACs to `fc2` but is
fast because N=1536 → 240 workgroups, whereas N=384 → only ~60 workgroups on 46
SMs. **The bottleneck is small-N GEMM occupancy** — and it is largely inherent:
RF-DETR-Nano is small, so the GPU runs at ~7% of fp32 peak regardless.

**What worked (37.2 → ~42 FPS cumulative):**
- Fused QKV (one `[M,3D]` GEMM instead of three), GELU fused into the FC1 GEMM,
  LayerScale folded into the residual add. *Only ~3%* — the layers are
  compute-bound, not dispatch-bound, so removing cheap ops barely helps.
- `BK` 16→32 in the GEMM (~1%).
- Batched decoder readbacks (`readback2`) + caching the constant `refpoint_embed`
  on the CPU — the **largest single win**; CPU round-trips mattered more than
  kernel fusion.
- Descriptor-set caching (key insight: arena buffers are reused by index, so each
  dispatch's `(pipeline, buffers)` is stable → allocate+write each set once). ~1 ms
  on desktop NVIDIA (near noise — `vkAllocateDescriptorSets` is cheap here); kept
  because it is correct and likely matters more on mobile/edge drivers.

**Negative results (recorded so they aren't re-tried):**
- A narrower 64×32 GEMM tile (more workgroups) *regressed* — lost arithmetic
  intensity > occupancy gain.
- fp16 / cooperative-matrix gave no speedup (memory-bandwidth-bound, established on
  RT-DETR).

**Remaining levers:** merge backbone+enc into one submit (~1 ms); split-K GEMM for
the N=384 occupancy (the real ~17 ms-backbone lever, but high-effort, uncertain,
and still won't beat cuBLAS). The host side is essentially tapped out — descriptor
caching proved the host isn't the bottleneck; the GEMMs are.

---

## 7. Honest benchmarking (the part most likely to be wrong)

This was where rigor mattered most, and where I corrected myself twice.

- **Compare against *optimized* PyTorch, never eager.** `optimize_for_inference()`
  is just TorchScript trace + optional fp16. On the transformer it gave ~1.6×
  (19→12 ms forward) by removing Python per-op dispatch overhead.
- **Strip the framework's convenience API.** RF-DETR's `predict()` goes through
  `supervision` and adds ~24 ms of host tax that no deployment pays. An early
  "Vulkan is 1.4× faster end-to-end" claim was an artifact of comparing against it.
  A *fair lean* PyTorch pre/post (resize + forward + sigmoid/top-k) inverts the
  result: **TorchScript fp32 lean-e2e 21.5 ms / 46.5 FPS vs Vulkan 24.8 ms** at the
  time — PyTorch wins, because its forward is ~2× faster (cuBLAS) and forward is
  ~80% of the budget.
- **`cudnn.benchmark` gotcha (YOLO).** The first YOLO baseline showed TorchScript
  *slower* than eager — impossible if measured right. Cause: `cudnn.benchmark` was
  off, so cuDNN never autotuned its conv algorithms, making eager ~3–4× too slow.
  Enabling it fixed the ordering. Always set it for conv-net baselines.
- **Cross-architecture insight.** TorchScript helps the *transformer* (RF-DETR)
  but barely the *CNN* (YOLO) — its win is Python-dispatch removal, which scales
  with op count. fp16 is the mirror image: it helps the compute-bound CNN (YOLO
  11.5→9.5 ms) but not the memory-bound transformer. Clean framing: "transformer
  detectors are dispatch/memory-bound; CNN detectors are conv-compute-bound."

### Honest standing (same GPU, fp32, end-to-end, lean pre/post)
- **RF-DETR:** ~23.6 ms (42 FPS) vs optimized PyTorch ~21.5 ms (46 FPS) — within
  ~10%, approaching parity.
- **YOLOv8n:** 7.9 ms (126.8 FPS) vs TorchScript 8.2 ms — parity (within noise);
  the lean C++ pre/post offsets the conv compute gap at ~8 ms.
- The Vulkan path is **not** a speed win on NVIDIA (cuBLAS/cuDNN win the compute).
  The defensible contribution is **framework-free, vendor-portable (runs on any
  Vulkan GPU — Mali/Adreno/Intel/AMD), and co-resident with the renderer on one
  device** — none of which the faster PyTorch numbers can claim. TensorRT (not
  installed) would widen the compute gap further, so latency claims must stay
  "competitive," never "faster."

Baselines live in `scripts/bench_rfdetr_pytorch.py`, `scripts/bench_rtdetr_pytorch.py`,
and `scripts/yolo_benchmark.py` (each: eager / TorchScript / fp16 / TensorRT-if-present,
forward / lean-e2e / convenience-`predict()` columns).

---

## 8. Reusable takeaways

- Validation-first against captured activations is the difference between a
  one-pass port and an open-ended debug.
- Read the reference source for the *exact* config; checkpoints reveal overrides
  the docs don't (`dec_layers`, tap indexing).
- Reproduce a wrong stage in NumPy from captured inputs to localize GPU bugs.
- Profile before optimizing; small models are GPU-underutilized by nature, and the
  bottleneck (small-N GEMM occupancy) is structural, not fixable by fusion.
- Benchmark against *optimized* baselines with the *right flags* (`cudnn.benchmark`)
  and a *fair* pre/post; the framework's convenience API is not a deployment path.
- For a hand-written portable compute path, the honest pitch is portability and
  deployment footprint, not beating vendor-optimized inference on the vendor's GPU.
