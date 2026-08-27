"""Live torch tensors over the Vulkan renderer's frames — zero copy.

``FrameTensors`` is the one-liner form of ``VulkanRenderer.enable_frame_interop``
plus the CUDA import: it arms the per-frame device-to-device copies, imports
each exported buffer once, and exposes them as torch tensors that STAY VALID
across frames. After a ``render()`` the pixels are one fence wait away, with no
staging buffer, no host memcpy and no ``vkDeviceWaitIdle``.

    import threepp as tp
    from threepp.torch_frames import FrameTensors

    renderer.render(scene, camera)              # exports are sized from a real frame
    with FrameTensors(renderer, channels=("color", "depth", "ids")) as frames:
        for _ in range(steps):
            renderer.render(scene, camera)
            frames.sync()                      # ONE frame fence, not device idle
            obs = frames.color[..., :3].float() / 255.0   # torch, on the GPU
            ...

Sync contract: render() -> sync() -> read -> next render(). Host ordering is the
only cross-API synchronization; there is no shared semaphore.

Single-buffered: the tensors are live views of renderer memory that the next
render() overwrites in place. ``tensor.clone()`` anything that must outlive the
frame — a replay buffer, an async training batch.

Invalidation: a resize, ``set_render_scale``, a G-buffer MSAA or splat-depth-AOV
toggle, or removing the view reallocates the source images, and the renderer
DISABLES interop for that view rather than freeing memory CUDA has imported (one
warning on stderr). Rebuild the FrameTensors afterwards — ``stale`` reports it.

NVIDIA only, by construction. Without a CUDA torch, or on a device with no
Vulkan external-memory extension, ``FrameTensors`` raises
``FrameInteropUnavailable`` and the fallback is the ordinary host readback
(``read_aovs_typed`` / ``read_gbuffer_aov_raw``).
"""

from __future__ import annotations

from .cuda_interop import FRAME_CHANNEL_DTYPES, VkInteropTensor

__all__ = ["FrameTensors", "FrameInteropUnavailable"]

# The channels FrameTensors knows how to shape, in the order the attribute
# properties are declared below.
CHANNELS = ("color", "depth", "normal", "motion", "ids", "albedo", "splat_depth")


class FrameInteropUnavailable(RuntimeError):
    """The renderer exported nothing — no CUDA, no extension, or too early."""


class FrameTensors:
    """Torch tensors over one view's exported frame channels.

    Parameters
    ----------
    renderer : threepp.VulkanRenderer
    view : int
        0 (the primary) or an ``add_view`` handle. A fixed-size secondary view
        is the steady state this is designed for: it never resizes, so an armed
        import survives everything a window does.
    channels : iterable of str
        Any of "color", "depth", "normal", "motion", "ids", "albedo",
        "splat_depth". Unexportable ones are dropped by the renderer (e.g.
        "splat_depth" without ``splat_depth_aov``); check ``.channels``.

    Attributes
    ----------
    color, depth, normal, motion, ids, albedo, splat_depth : torch.Tensor or None
        Live views, shaped (H, W) for the single-component channels and
        (H, W, 4) for the rest. Layouts are the renderer's native ones — see
        ``VulkanRenderer::enableFrameInterop``: depth is REVERSED-Z NDC in
        [0, 1] (1 = near), normals are encoded n*0.5+0.5 in xyz, ids carries
        the visible index in x, the stable instance id in y and flags|class in
        z. Nothing is decoded here; a decode would be a copy.
    bgra : bool
        True when ``color`` is B,G,R,A rather than R,G,B,A (the swapchain
        format; B8G8R8A8 is what threepp asks for and normally gets).
    """

    def __init__(self, renderer, view=0, channels=("color", "depth")):
        self._renderer = renderer
        self._view = int(view)
        self._imports = {}
        self._armed = False
        self.bgra = False

        wanted = [str(c) for c in channels]
        unknown = [c for c in wanted if c not in FRAME_CHANNEL_DTYPES]
        if unknown:
            raise ValueError(f"unknown frame channel(s) {unknown}; "
                             f"expected any of {sorted(FRAME_CHANNEL_DTYPES)}")

        try:
            import torch
        except ImportError as e:  # pragma: no cover - environment-dependent
            raise FrameInteropUnavailable(
                "torch is not installed; frames-out interop needs CUDA torch") from e
        if not torch.cuda.is_available():
            raise FrameInteropUnavailable(
                "torch has no CUDA device; frames-out interop is NVIDIA-only. "
                "The host fallback is renderer.read_aovs_typed(...).")
        # Make torch's primary context current on this thread BEFORE importing:
        # the import binds to whatever context is current, and a mapping in a
        # context torch never uses is an invalid access on first touch. A
        # zero-element allocation is the cheapest way to force initialisation.
        torch.cuda.init()
        torch.empty(0, device="cuda")

        exports = renderer.enable_frame_interop(self._view, wanted)
        if not exports:
            raise FrameInteropUnavailable(
                "the renderer exported no frame channels. Call render() at least "
                "once before arming (the exports are sized from a real frame), "
                "check the view handle is live, and note that a device without "
                "the Vulkan external-memory extension cannot do this at all.")
        self._armed = True

        try:
            for e in exports:
                name = e["channel"]
                typestr, comps = FRAME_CHANNEL_DTYPES[name]
                h, w = int(e["height"]), int(e["width"])
                shape = (h, w) if comps == 1 else (h, w, comps)
                if name == "color":
                    self.bgra = bool(e["bgra"])
                imp = VkInteropTensor(e["handle"], e["size_bytes"], shape, typestr)
                # Materialise the tensor now rather than on first access: a
                # typestr torch refuses should fail HERE, while the object is
                # still being constructed and the teardown below is in scope.
                _ = imp.tensor
                self._imports[name] = imp
        except Exception:
            self.close()
            raise

    # ── The frame ────────────────────────────────────────────────────────────
    @property
    def channels(self):
        """The channel names that were actually exported and imported."""
        return tuple(self._imports)

    @property
    def stale(self):
        """True once these tensors stopped tracking the renderer.

        Either this object was closed, or the renderer disabled interop under it
        — a resize, a render-scale change, a removed view. In the second case
        the tensors still read (the memory is alive while CUDA holds it) but
        NOTHING WRITES THEM ANY MORE: they are a frozen last frame. Poll this in
        a loop that outlives a window, and rebuild the FrameTensors when it
        goes true.
        """
        return not self._armed or not self._renderer.frame_interop_active(self._view)

    def __getitem__(self, name):
        return self._imports[name].tensor

    def __contains__(self, name):
        return name in self._imports

    def get(self, name):
        imp = self._imports.get(name)
        return imp.tensor if imp is not None else None

    color = property(lambda self: self.get("color"))
    depth = property(lambda self: self.get("depth"))
    normal = property(lambda self: self.get("normal"))
    motion = property(lambda self: self.get("motion"))
    ids = property(lambda self: self.get("ids"))
    albedo = property(lambda self: self.get("albedo"))
    splat_depth = property(lambda self: self.get("splat_depth"))

    def sync(self):
        """Wait for the last submitted frame's copies. Returns False before the
        first frame. ONE fence, not a device drain."""
        return self._renderer.sync_frame_interop()

    # ── Lifetime ─────────────────────────────────────────────────────────────
    def close(self):
        """Release the CUDA imports, THEN disarm the renderer's exports.

        That order is the contract, not a preference: the Vulkan allocation is
        the renderer's, and freeing it under a live CUDA mapping is a
        use-after-free that reports as nothing at all. Idempotent.
        """
        for imp in self._imports.values():
            imp.close()
        self._imports.clear()
        if self._armed:
            self._armed = False
            self._renderer.disable_frame_interop(self._view)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter teardown
            pass

    def __repr__(self):
        if not self._imports:
            return "<FrameTensors closed>"
        parts = ", ".join(f"{n}{tuple(i.shape)}" for n, i in self._imports.items())
        return f"<FrameTensors view={self._view} {parts}>"
