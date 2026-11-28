# RF-DETR-Nano → Vulkan compute: porting log & insights

An engineering record of how the RF-DETR-Nano detector was reimplemented from
scratch on threepp's Vulkan compute path (`examples/vulkan/rfdetr/`), how it was
validated, optimized, and honestly benchmarked. Builds on the earlier YOLOv8n and
RT-DETR ports that established the shared `VkInfer` harness and most shaders (the
RT-DETR port was deleted on 2026-11-27; git history keeps it).

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

Profiling used GPU timestamps (`RFDETR_PROFILE`), per-ViT-layer then per-op.

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

**Remaining levers (as of May):** merge backbone+enc into one submit (~1 ms); split-K
GEMM for the N=384 occupancy. Split-K has since landed (re-measured below). The belief
that the host side was tapped out did not hold: see the one-submit rewrite below.

### One submit per inference (2026-11-27, RTX 5080)

An inference used to make seven host round trips: the input upload, the backbone frame,
the first-stage head frame, its readback, the query upload, the decoder frame and the
head readback. Between them the host resized and normalized the image, ran the top-300
partial sort, computed 153,600 sin/cos for the query embedding, and allocated a fresh
buffer for the backbone output. On the 5080 that host work was 35 % of Nano's end-to-end
time (5.6 of 16.0 ms) and 29 % of Medium's (9.5 of 33.3 ms).

Now the preprocess (`preprocess.comp`), the proposals (`proposals.comp`), the top-K
(`topk_rank.comp`: each token counts the tokens ranked before it, ties by token index)
and the reference + sine embedding (`query_refs.comp`) are compute shaders; the input and
the head outputs are copied inside the command buffer (`VkInfer::recordUpload` /
`recordReadback`); and the frame ends with one submit and one fence wait.

| bus.jpg, RTX 5080, fp32 | before | after |
|---|---|---|
| Nano end-to-end (median of 3 interleaved runs x 200) | 16.03 ms | 11.81 ms |
| Nano GPU per inference | 10.45 ms | 10.59 ms |
| Medium end-to-end (median of 3 interleaved runs x 100) | 33.33 ms | 26.33 ms |
| Medium GPU per inference | 23.8 ms | 24.8 ms |

GPU time rises a little because the input copy now runs inside the frame. The validation
output is identical to the old path at printed precision, the bus.jpg detections are
identical to 4 decimals, and a run under the validation layer with synchronization
validation logs nothing. What is left on the host (~1.2 ms) is recording ~300 dispatches,
the RGBA copy into staging and the sigmoid/threshold; recording the command buffer once
and replaying it is the next lever.

Split-K re-measured on the 5080: Nano backbone 12.03 → 9.01 ms GPU (end-to-end 19.9 →
16.2 ms), Medium 22.8 → 21.3 ms. It helps both variants on this card, so it stays on for
all of them and the `RF_NOSPLITK` switch is gone.

### Per-kernel profile and fused attention (2026-11-27, RTX 5080)

`RFDETR_PROFILE` now also timestamps every dispatch and sums GPU time per kernel. On Nano
(one-submit build, 10.2 ms of GPU work per inference) it showed attention
(`attn_vit_scores` + `attn_vit_apply`) at 3.8 ms, the GEMMs with their split-K reduce passes
at 3.7 ms, the projector convolutions at 1.0 ms and everything else at ~1.9 ms. LayerNorm
and softmax, though one thread per row, are under 0.1 ms in total. *(Wrong for LayerNorm:
the per-kernel sums misattribute time between dispatches; see §10, where a row-parallel
LayerNorm saved 4.7 ms on the Keypoint variant.)*

The attention kernels wrote the [T,T] scores, read them back for the softmax and again for
the apply, and read Q, K and V straight from global memory. `attn_vit_flash.comp` does all
three in one pass with an online softmax: K and V tiles go through shared memory, four
invocations share a query (each owning every fourth head dim, so their reads hit different
banks), and a clustered subgroup add joins their partial dot products. Nano's attention went
from 3.8 to ~1.2 ms of GPU time; Medium's global layers (T = 1300) gain the most.

| bus.jpg, RTX 5080 (median of 3 interleaved runs) | before | after |
|---|---|---|
| Nano end-to-end | 12.47 ms | 10.24 ms |
| Medium end-to-end | 27.51 ms | 18.85 ms |

The machine was busier than for the one-submit table above, so this "before" is higher
than that table's "after". The validation output stays at fp32 level (taps ≤ 1.6e-4 max
abs error, the same diverging noise rows) and the detections are identical to 4 decimals.
Devices without clustered subgroup operations keep the three-pass kernels.

**Negative result: tensor-core GEMMs.** A `VK_KHR_cooperative_matrix` GEMM (fp16 inputs,
fp32 accumulation, 64x64 tiles, split-K planes into the same reduce) matched the detections
within 0.0015 but was slower: Nano 11.3 vs 10.7 ms and Medium 20.2 vs 18.6 ms with the fused
attention, and worse still without split-K. At M ≈ 580 to 1300 tokens the GEMMs are limited
by how few workgroups fill the GPU, not by arithmetic, so tensor cores have little to speed
up until the kernel itself is restructured. Not kept.

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

Baselines live in `scripts/bench_rfdetr_pytorch.py` and `scripts/yolo_benchmark.py`
(each: eager / TorchScript / fp16 / TensorRT-if-present, forward / lean-e2e /
convenience-`predict()` columns).

---

## 9. Pose: RF-DETR Keypoint (2026-11-27, RTX 5080)

`--variant keypoint` runs Roboflow's RF-DETR Keypoint preview (Apache 2.0; COCO people with
their 17 keypoints) on the same harness. Weights: `scripts/export_rfdetr_weights.py --variant
keypoint --checkpoint rf-detr-keypoint-preview-xlarge.pth`; references:
`scripts/capture_rfdetr_activations.py --variant keypoint --checkpoint <pth> [--image bus.jpg]`.
The architecture was read from the `rfdetr` 1.12.1 source (`models/transformer.py`,
`models/lwdetr.py`, `models/postprocess.py`).

**Same as the detectors:** the backbone, now at patch 12 (576 px input: 48×48 = 2304 patches,
four windows of 577 tokens), the projector, the two-stage selection (100 queries, two logits:
class 0 and person), the four instance decoder layers and the box head.

**New:**
- A second projector (`backbone.0.cross_attn_projector`, same C2f + LayerNorm, own weights) on
  the same taps. Only the keypoint cross-attention reads it.
- 17 keypoint queries per instance. Their initial value depends on learned weights only, so
  the exporter computes it once (`derived.keypoint_queries`).
- After each layer's instance FFN: group self-attention over each instance and its 17
  keypoints (8 heads of 32; the instance token carries no position, keypoint j its learned
  `keypoint_pos_embed[j]`; the instance takes the answer through a learned scalar layer
  scale), keypoint deformable cross-attention into the second projector around the
  instance's reference box (query = keypoint state + the instance's query_pos), and a
  keypoint FFN (256→1024→256).
- A keypoint head (MLP 256→256→256→8): x/y in reference-box units, findable and visible
  logits, the precision-Cholesky factors (log l11, l21, log l22) and a class-logit
  contribution, summed over the 17 keypoints into the person logit.

Three small kernels carry the new data movement: `repeat_rows.comp` (a per-instance row per
keypoint: reference box, query_pos), `group_pack.comp` / `group_unpack.comp` (instance + its
keypoints as one group of 18 rows, and back). The group attention reuses the three-pass ViT
attention kernels at head dim 32.

**Read, not guessed:** the encoder-stage keypoint branch (`keypoint_query_initializer_enc`,
`enc_out_keypoint_embed`) only feeds training losses; the keypoint states get no final norm;
the class mask over a group blocks nothing when one class has keypoints (the loader checks,
and rejects multi-class schemas, which would need masked attention). The postprocess fuses
every selected score with the keypoints' uncertainty, `sigmoid(log s − 0.2·T)` with T the
findability-weighted log mean covariance trace (T = 0 for class 0, which has no keypoints).
There is no NMS: on bus.jpg a half-visible person at the left edge comes back as 36
overlapping detections between 0.26 and 0.43, so the demo keeps `predict()`'s threshold of
0.5 rather than the 0.25 of Roboflow's examples.

**Validation.** Noise reference: patch embedding to both projectors within 1.2e-4 (taps) and
3e-6 (projectors) max abs error; the decoder diverges there as §5 predicts. bus.jpg
reference: decoder reference boxes 9e-6, instance states 3.3e-4, keypoint states 1.8e-3,
logits 4.5e-4, boxes 4e-5; keypoint head x/y ≤ 2.4e-5, the uncertainty channels ≤ 1e-2;
no query row diverges. The host postprocess run on the reference tensor gives rfdetr's
PostProcess output at printed precision: three people at 0.7745, 0.7522 and 0.6694, every
keypoint equal to 0.1 px. From the JPEG (stb_image decode, GPU resize) the third score is
0.6739: the input path differs from PIL + torch, the model does not.

**Cost.** 42 ms end-to-end on bus.jpg (median of 100 runs), 38 ms of it GPU: the ViT ~24 ms,
the two projectors ~6 ms, the decoder ~7.6 ms; GEMMs are ~23 ms of the total. That is twice
Medium at the same input size, because patch 12 gives 2304 patches to Medium's 1296 and the
global attention grows with their square. Roboflow quotes 9.8 ms on a T4 with TensorRT fp16.
§10 brings it to 10.9 ms.

---

## 10. Speed: one GEMM, register-blocked attention, row-parallel LayerNorm (2026-11-28, RTX 5080)

| bus.jpg, RTX 5080, fp32, end to end (median of 3 interleaved runs x 100) | before | after |
|---|---|---|
| Keypoint (576 px, patch 12) | 36.41 ms | 10.94 ms |
| Nano (384 px) | 10.46 ms | 3.75 ms |
| Medium (576 px) | 18.04 ms | 6.54 ms |

Keypoint GPU time went 34.1 → 10.3 ms. The detections and every keypoint are identical at
printed precision (one Nano score moves in the fourth decimal); `--validate` stays at fp32
level on all three variants (Keypoint taps ≤ 2.6e-4, keypoint head ≤ 1e-2 max abs error, the
same noise rows diverge on Nano and Medium); a run under the validation layer with
synchronization validation logs nothing; the synthetic demo's detect went 11.7 → 4.6 ms.

**One GEMM for every linear layer and conv (`gemm.comp`).** The old `linear.comp` computed
4x4 outputs per invocation from scalar, uncoalesced loads; the projector convolutions ran in
`conv2d.comp` at one output per invocation. Now a single kernel, built per B layout
(nn.Linear weight `[N,K]`, a 1x1 conv's CHW input `[K,N]`, or a kxk conv's im2col gathered
while loading, never stored) and per tile, does all of them with float4 loads and
register-blocked micro-tiles, and its epilogue fuses what used to be separate passes: the
bias (per column, or per output channel for a conv), GELU or ReLU, and the residual with its
LayerScale. Split-K writes partial planes that `gemm_reduce.comp` sums with the same
epilogue. Data movement went with it: the C2f branches and the de-windowed taps are written
straight into the buffers that read them, and the decoder's in_proj computes only the rows
each input needs, straight into the fused qkv layout (no splice, a third fewer FLOPs).
Fourteen kernels are gone.

**Two tiles, chosen per GEMM.** Timed per shape (a GEMM repeated 11 times in the frame,
divided out; single-dispatch timestamps can't be trusted, see below):
- 128x64 with 8x8 per invocation, 32 k per tile and *no* register prefetch is fastest on the
  big GEMMs (qkv 82 µs, fc1 102 µs, fc2 split in two 104 µs: ~45-49 % of fp32 peak);
- 64x64 with 4x4 per invocation and prefetch wins where the big tile makes too few
  workgroups (the 2308x384x384 attention out-projection: 25 µs against 87 unsplit on the
  big tile; the decoder's 1700-2304-row GEMMs, split in two to four).
`RfDetrVk::gemm_` estimates each tile and split of 1, 2, 4 or 8 as the busiest SM's whole
workgroups at the tile's measured rate (215 and 170 FMA/ns per SM) plus the split's reduce
traffic, among the candidates that give each SM at least 1.7 workgroups; on the twenty-odd
shapes measured it predicts within a few µs and picks the measured best or within ~3 % of it.
Negative results: 128x128 tiles (256 invocations x 8x8: too many registers, one workgroup
per SM), register prefetch on the big tile (same reason), splits of 3 (342 workgroups on 252
slots leave a ragged second wave).

**Attention (`attn_vit_flash.comp`) rewritten like a GEMM.** A workgroup takes 64 queries of
a head and walks the keys 64 at a time; an invocation holds 4 queries x 8 keys of scores,
then 4 queries x 8 dims of output, so each float4 read from shared memory feeds 8-16 FMAs
(the old kernel did one scalar read per FMA). The row maximum is shared over an 8-lane
cluster per key tile; P goes through shared memory over the K tile it came from. Windowed
layers 0.98 → ~0.6 ms, global 2.3 → 1.15 ms at that point.

**LayerNorm, the biggest single win.** `layernorm.comp` ran one invocation per row: for a
2308x384 token buffer, 2308 invocations each walking a row three times, every warp load
touching 32 cache lines. Now 32 invocations share a row, hold it in registers and read it
once: Keypoint GPU 15.0 → 10.3 ms across its ~60 LayerNorms. The per-kernel table had put
all LayerNorms at 0.045 ms. `layernorm_chw.comp` (the projectors' channel norm) got the
same treatment for ~0.1 ms.

**Record once, replay.** `infer()` records its frame with a key (the image size); while
`VkInfer::replayable` holds (no arena slot, staging buffer or descriptor set replaced since)
the next call only copies the image into the staging and submits the same command buffer.
Host recording was ~2.4 ms per Keypoint inference; now 0.

**Profiling caveat.** On this setup (NVIDIA, Windows, a desktop sharing the GPU) a single
dispatch's timestamp can bill its time to a neighbour: a 1.4 GFLOP fc2 showed 8 µs, a reduce
over 100x256 showed 305 µs. Barriers before the timestamps did not change it. Frame totals
and stage marks are sound; to price one kernel, repeat it in the frame and difference the
totals, or remove a block and difference them. `RFDETR_PROFILE` now also prints the host
time from `beginFrame()` to the submit.

**Left:** the GEMMs at ~45-49 % of fp32 peak; attention at ~13 TFLOPS on windowed layers;
the decoder at 1.7 ms of mostly small dispatches (~200 per inference). Tensor cores would
need the cooperative-matrix features enabled on the renderer's device.

---

## 8. Reusable takeaways

- Validation-first against captured activations is the difference between a
  one-pass port and an open-ended debug.
- Read the reference source for the *exact* config; checkpoints reveal overrides
  the docs don't (`dec_layers`, tap indexing).
- Reproduce a wrong stage in NumPy from captured inputs to localize GPU bugs.
- Profile before optimizing, and check the profiler: per-dispatch timestamps misled this
  port twice (§10). Small-N GEMM occupancy looked structural until the kernel and the tile
  choice were redone; splitting K and a smaller tile fixed it.
- Benchmark against *optimized* baselines with the *right flags* (`cudnn.benchmark`)
  and a *fair* pre/post; the framework's convenience API is not a deployment path.
- For a hand-written portable compute path, the honest pitch is portability and
  deployment footprint, not beating vendor-optimized inference on the vendor's GPU.
