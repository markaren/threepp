"""Import a Vulkan external-memory allocation into CUDA, as a Warp array or a torch tensor.

Both directions of threepp's zero-copy interop meet here, because both are the
same four CUDA driver calls over an OS handle the renderer hands out:

  * IN  — ``VulkanRenderer.enable_vertex_interop`` exports a mesh's
    position/normal buffers for a foreign producer to WRITE.
  * OUT — ``VulkanRenderer.enable_frame_interop`` exports the frame's colour /
    G-buffer images for a foreign consumer to READ.

Nothing crosses host memory in either direction.

    import threepp as tp
    from threepp.cuda_interop import VkInteropArray

    renderer.render(scene, camera)                     # the record must exist first
    h = renderer.enable_vertex_interop(mesh, on_frame) # -> ((pos, bytes), (nrm, bytes))
    pos = VkInteropArray(h[0][0], h[0][1], wp.vec3, n_verts, device)
    nrm = VkInteropArray(h[1][0], h[1][1], wp.vec3, n_verts, device)
    ...
    pos.close(); nrm.close()                           # before disable_vertex_interop

Layering. ``VkInteropMemory`` is the whole driver story — import, map, the
context and range guards, the fixed release order — and it depends on NOTHING
but ctypes, so this module imports on a machine with neither Warp nor torch
installed. ``VkInteropArray`` is the Warp wrapper over it (unchanged API) and
``VkInteropTensor`` the torch one; each imports its framework lazily, inside
__init__, so having one does not require the other.

Why raw ctypes against the driver rather than ``cuda-python``: the whole surface
is five driver entry points, and ``cuda-python`` is not a threepp dependency and
should not become one for an optional fast path. Warp 1.16 ships no Vulkan
interop of its own (``RegisteredGLBuffer`` is OpenGL only), so this is ours to
write.

NVIDIA only, by construction — the fallback for everyone else is the ordinary
``geometry.update_attribute`` host route inbound, and ``read_gbuffer_aovs``
outbound, which every demo here keeps.
"""

from __future__ import annotations

import ctypes
import sys
import warnings

# ── CUDA driver constants (cuda.h, CUDA 12.x) ────────────────────────────────
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2
CUDA_EXTERNAL_MEMORY_DEDICATED = 1
CU_POINTER_ATTRIBUTE_CONTEXT = 1
CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12
CUDA_SUCCESS = 0


# ── Struct layouts ───────────────────────────────────────────────────────────
# Transcribed from cuda.h and checked with ctypes.sizeof against the installed
# CUDA 12.x headers. The union's alignment is what makes these easy to get
# subtly wrong: `type` is a 4-byte enum, the union starts 8-byte aligned, so
# there are 4 bytes of padding no field names.


class _Win32Handle(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_void_p), ("name", ctypes.c_void_p)]


class _ExternalMemoryHandle(ctypes.Union):
    _fields_ = [
        ("fd", ctypes.c_int),
        ("win32", _Win32Handle),
        ("nvSciBufObject", ctypes.c_void_p),
    ]


class CUDA_EXTERNAL_MEMORY_HANDLE_DESC(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint),
        ("handle", _ExternalMemoryHandle),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class CUDA_EXTERNAL_MEMORY_BUFFER_DESC(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_ulonglong),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


_driver = None


def _drv():
    """The CUDA driver library, loaded and initialised once.

    Whoever brought a device up — Warp, torch — has already called cuInit, but
    this module no longer imports either of them, so it cannot assume that.
    cuInit is idempotent and cheap, so it is simply called here.
    """
    global _driver
    if _driver is None:
        try:
            drv = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" \
                else ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            raise RuntimeError(
                "CUDA driver library not found (nvcuda.dll / libcuda.so.1); "
                "Vulkan<->CUDA interop needs an NVIDIA driver") from e
        _check(drv.cuInit(0), "cuInit")
        _driver = drv
    return _driver


def _check(code, what):
    if code == CUDA_SUCCESS:
        return
    name = ctypes.c_char_p()
    try:
        # The global, NOT _drv(): this is reachable from inside _drv's own
        # cuInit check, where a re-entrant load would recurse.
        if _driver is not None:
            _driver.cuGetErrorName(ctypes.c_int(code), ctypes.byref(name))
    except Exception:  # pragma: no cover - only if the driver is truly broken
        pass
    tag = name.value.decode() if name.value else f"code {code}"
    raise RuntimeError(f"{what} failed: {tag} ({code})")


class VkInteropMemory:
    """A Vulkan external-memory allocation, imported into CUDA. Framework-free.

    This is the whole driver story — the four calls, the two guards and the
    fixed release order — with no Warp and no torch anywhere in it. The wrappers
    below own the "and now expose it as X" half.

    Parameters
    ----------
    os_handle : int
        The handle from ``enable_vertex_interop`` / ``enable_frame_interop``. On
        Windows it is a Win32 NT handle **owned by the renderer** — CUDA
        duplicates it on import, so do NOT CloseHandle it here; the renderer
        releases it in its disable / teardown path. On Linux it is an fd whose
        ownership transfers to CUDA on import (see the POSIX note below).
    size_bytes : int
        The exported ALLOCATION size, as returned. It is what a dedicated
        ``cuImportExternalMemory`` has to be told about, and it may be larger
        than the bytes actually used (the driver rounds allocations up).
    need_bytes : int
        How much of it the caller intends to address. Checked against both the
        export size and the mapped range, because a short view is out of bounds
        with no error from anyone downstream.
    context : int, optional
        The CUcontext the mapping must belong to. Pass the framework's own
        (``wp.get_device(...).context``); leave it None to adopt whatever
        context is current, which is what the torch path does — torch does not
        publish its CUcontext, and its primary context is current on the thread
        once ``torch.cuda.init()`` has run.

    Attributes
    ----------
    ptr : int
        The mapped device pointer. Valid until ``close()``.
    """

    def __init__(self, os_handle, size_bytes, *, need_bytes=0, context=None):
        self._ext_mem = None
        self.ptr = 0
        self.size_bytes = int(size_bytes)

        # ── Guard 1: the CUDA CONTEXT ────────────────────────────────────────
        # The most likely way to get a silently wrong result.
        # cuImportExternalMemory and cuExternalMemoryGetMappedBuffer both act on
        # the CURRENT context, and neither wp.array(ptr=...) nor
        # torch.as_tensor(__cuda_array_interface__) validates the pointer it is
        # handed. If a context the consumer never uses (or no context at all) is
        # current at import time, every launch against the result is an invalid
        # access at best and someone else's memory at worst.
        cur = ctypes.c_void_p()
        _check(_drv().cuCtxGetCurrent(ctypes.byref(cur)), "cuCtxGetCurrent")
        if not cur.value:
            raise RuntimeError(
                "no CUDA context is current: initialise the consuming framework "
                "first (torch.cuda.init() / wp.init() + device.make_current()) "
                "and import from the thread it runs on")
        if context is not None and cur.value != int(context):
            raise RuntimeError(
                f"CUDA context mismatch before import: current={cur.value:#x} "
                f"expected={int(context):#x}. Import from the thread that owns it.")
        expect_ctx = int(context) if context is not None else cur.value

        need_bytes = int(need_bytes)
        # ── Guard 2: short import ────────────────────────────────────────────
        if need_bytes > self.size_bytes:
            raise ValueError(
                f"a view of {need_bytes} bytes was asked for but the export is "
                f"only {self.size_bytes}")

        hdesc = CUDA_EXTERNAL_MEMORY_HANDLE_DESC()
        hdesc.size = ctypes.c_ulonglong(self.size_bytes)
        # DEDICATED because createExternalBuffer allocates a VkDeviceMemory
        # dedicated to exactly this buffer (VkMemoryDedicatedAllocateInfo).
        hdesc.flags = CUDA_EXTERNAL_MEMORY_DEDICATED
        if sys.platform == "win32":
            hdesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32
            hdesc.handle.win32.handle = ctypes.c_void_p(int(os_handle))
            hdesc.handle.win32.name = None
        else:
            # POSIX branch: WRITTEN BUT UNTESTED — this machine is Windows, and
            # no CI job has both Vulkan and CUDA. The ownership rule differs
            # from Windows: the fd transfers to CUDA on a successful import and
            # must not be closed or re-imported here. The exporter mints a fresh
            # fd per hand-out for exactly that reason.
            hdesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
            hdesc.handle.fd = ctypes.c_int(int(os_handle))

        ext = ctypes.c_void_p()
        _check(_drv().cuImportExternalMemory(ctypes.byref(ext), ctypes.byref(hdesc)),
               "cuImportExternalMemory")
        self._ext_mem = ext

        bdesc = CUDA_EXTERNAL_MEMORY_BUFFER_DESC()
        bdesc.offset = 0
        bdesc.size = ctypes.c_ulonglong(self.size_bytes)
        bdesc.flags = 0
        dptr = ctypes.c_ulonglong()
        try:
            _check(_drv().cuExternalMemoryGetMappedBuffer(
                ctypes.byref(dptr), ext, ctypes.byref(bdesc)),
                "cuExternalMemoryGetMappedBuffer")
            self.ptr = int(dptr.value)

            # Guard 1, second half: the MAPPED POINTER's context, not just the
            # one that happened to be current. And its usable range, which is
            # what a short/rounded import would show up as.
            owner = ctypes.c_void_p()
            _check(_drv().cuPointerGetAttribute(
                ctypes.byref(owner), ctypes.c_int(CU_POINTER_ATTRIBUTE_CONTEXT), dptr),
                "cuPointerGetAttribute(CONTEXT)")
            if owner.value != expect_ctx:
                raise RuntimeError(
                    f"imported pointer belongs to context {owner.value:#x}, not the "
                    f"expected {expect_ctx:#x}")
            rng = ctypes.c_size_t()
            _check(_drv().cuPointerGetAttribute(
                ctypes.byref(rng), ctypes.c_int(CU_POINTER_ATTRIBUTE_RANGE_SIZE), dptr),
                "cuPointerGetAttribute(RANGE_SIZE)")
            if int(rng.value) < need_bytes:
                raise RuntimeError(
                    f"mapped range is {int(rng.value)} bytes, short of the {need_bytes} "
                    f"the requested view uses")
        except Exception:
            self.close()
            raise

        self.context = expect_ctx

    # ── Lifetime ─────────────────────────────────────────────────────────────
    def close(self):
        """Free the mapping and the imported memory object. Idempotent.

        Order matters: the mapped buffer first (cuMemFree_v2), then the external
        memory object. Call this BEFORE the renderer's disable_* — the Vulkan
        allocation is the renderer's, and freeing it under a live CUDA mapping
        is a use-after-free that reports as nothing at all.
        """
        if self.ptr:
            _drv().cuMemFree_v2(ctypes.c_ulonglong(self.ptr))
            self.ptr = 0
        if self._ext_mem is not None:
            _drv().cuDestroyExternalMemory(self._ext_mem)
            self._ext_mem = None

    @property
    def open(self):
        return bool(self.ptr)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        # Warn, then clean up. Interpreter shutdown can run this after ctypes
        # has been torn down, so it must never raise out of __del__.
        if getattr(self, "ptr", 0) or getattr(self, "_ext_mem", None) is not None:
            try:
                warnings.warn(f"{type(self).__name__} was not close()d explicitly; "
                              "releasing at collection time, which is not ordered "
                              "against the renderer's own teardown",
                              ResourceWarning, stacklevel=2)
            except Exception:
                pass
            try:
                self.close()
            except Exception:
                pass

    def __repr__(self):
        state = "closed" if not self.ptr else f"ptr={self.ptr:#x}"
        return f"<{type(self).__name__} {state} bytes={self.size_bytes}>"


class VkInteropArray(VkInteropMemory):
    """A Vulkan external-memory buffer, imported into CUDA and wrapped as a wp.array.

    Parameters
    ----------
    os_handle : int
        The handle from ``VulkanRenderer.enable_vertex_interop``. On Windows it
        is a Win32 NT handle **owned by the renderer** — CUDA duplicates it on
        import, so do NOT CloseHandle it here; the renderer releases it in
        ``disable_vertex_interop`` / teardown. On Linux it is an fd whose
        ownership transfers to CUDA on import (see the POSIX note below).
    size_bytes : int
        The exported ALLOCATION size, as returned. It is what a dedicated
        ``cuImportExternalMemory`` has to be told about, and it may be larger
        than the bytes the mesh actually uses (the driver rounds allocations up).
    dtype, shape : Warp dtype and element count/shape of the view to expose.
    device : Warp device (or name) the import must land on.

    Attributes
    ----------
    array : wp.array
        A view over the imported memory. It does NOT own the allocation — see
        the lifetime note on ``close``.

    Sync contract
    -------------
    Writing this array from a kernel is asynchronous on Warp's stream. The
    renderer's ``on_frame`` callback runs post-fence and pre-record inside
    ``render()``, and host ordering is the ONLY thing sequencing the foreign
    write against the frame that reads it — there is no shared semaphore. So the
    last statement of the callback must be ``wp.synchronize_device(device)``.
    Skipping it does not crash; it renders a torn surface, intermittently.

    Stride
    ------
    The renderer's exported vertex buffers are TIGHTLY PACKED float xyz, so
    ``wp.vec3`` (12-byte stride) is the correct dtype here. This is deliberately
    unlike the soft-body tet path, whose positions are ``PxVec4``-shaped: using
    ``wp.vec3`` against a vec4-padded target shifts every vertex after the first
    by 4 bytes, and the result looks like a sheared, drifting mesh rather than
    an error.

    Stale derived state
    -------------------
    A foreign device write bumps no ``BufferAttribute::version``, so it is
    invisible to the renderer's dirty detection. That is handled renderer-side:
    an interop record is enqueued for a vertex/normal copy + BLAS refit
    unconditionally every frame (and its positions are GPU-sanitized for
    finiteness first, unless ``validate=False``). Nothing is required of the
    producer beyond writing the buffer.
    """

    def __init__(self, os_handle, size_bytes, dtype, shape, device="cuda:0"):
        import warp as wp   # lazy: this module must import without warp

        self.array = None
        dev = wp.get_device(device)
        if not dev.is_cuda:
            raise ValueError(f"VkInteropArray needs a CUDA device, got {dev}")
        # Warp's own (primary) context has to be the current one before the
        # import — the core class then proves it, and proves the mapped pointer
        # landed there too.
        dev.make_current()

        shape_t = (shape,) if isinstance(shape, int) else tuple(shape)
        elems = 1
        for s_ in shape_t:
            elems *= int(s_)
        needed = elems * wp.types.type_size_in_bytes(dtype)

        super().__init__(os_handle, size_bytes, need_bytes=needed, context=dev.context)
        try:
            # ── LIFETIME ─────────────────────────────────────────────────────
            # No `deleter=`, deliberately. Given one, a wp.array frees the
            # mapping whenever it is garbage collected — including a dropped
            # local or a temporary — and would free it twice if the pointer were
            # ever wrapped again. The mapping has to outlive every view of it and
            # be released in a fixed order (mapped buffer, then the external
            # memory object) BEFORE the renderer's disable_vertex_interop. So the
            # array borrows (deleter=None is Warp's "not mine"), this object
            # owns, and close() is the only free.
            self.array = wp.array(ptr=self.ptr, dtype=dtype, shape=shape_t, device=dev)
        except Exception:
            self.close()
            raise
        self._device = dev

    def close(self):
        """Release the wp.array view, then the mapping. See VkInteropMemory.close.

        The array is dropped here rather than left pointing at freed memory: a
        stale view is exactly the silent-wrongness case this class exists to
        prevent.
        """
        self.array = None
        super().close()


# ── dtype table for the frames-out path ──────────────────────────────────────
# One row per VulkanRenderer::FrameChannel: the numpy typestr of the exported
# attachment and the per-pixel component count. Straight from the layout table
# on VulkanRenderer::enableFrameInterop, and the only place Python encodes it.
#
# `ids` is the one row with a caveat: the attachment is RGBA16_UINT and torch's
# uint16 support is partial (2.5 has the dtype, few kernels), so it is exposed
# as int16 with the same bits when torch refuses `<u2`. The values that matter
# — instance ids and class ids — are all well under 32768, so the reinterpret
# is invisible in practice; `.view(torch.uint16)` is a free fix-up where the
# consumer needs the unsigned type.
FRAME_CHANNEL_DTYPES = {
    "color":       ("|u1", 4),
    "albedo":      ("|u1", 4),
    "depth":       ("<f4", 1),
    "splat_depth": ("<f4", 1),
    "normal":      ("<f2", 4),
    "motion":      ("<f2", 4),
    "ids":         ("<u2", 4),
}


class VkInteropTensor(VkInteropMemory):
    """An exported render target, imported into CUDA and exposed to torch.

    Implements ``__cuda_array_interface__`` v3, so ``torch.as_tensor(obj,
    device="cuda")`` yields a tensor that BORROWS the renderer's memory — no
    copy, no host round trip. ``.tensor`` builds (and caches) that tensor.

    Parameters
    ----------
    os_handle, size_bytes : from one entry of ``enable_frame_interop``.
    shape : tuple
        The logical shape, e.g. (H, W, 4) for colour, (H, W) for depth.
    typestr : str
        Numpy typestr of one component ('|u1', '<f4', '<f2', '<u2').

    Sync contract
    -------------
    The renderer fills this buffer with a copy recorded in the frame's own
    command buffer, and host ordering is the ONLY thing sequencing that copy
    against a torch kernel reading it: render() -> sync_frame_interop() -> read.
    Skipping the sync does not crash; it reads a torn frame, intermittently.

    Single-buffered
    ---------------
    There is ONE allocation per channel, so the tensor is a live view the next
    render() overwrites in place. Clone what you need to keep.
    """

    def __init__(self, os_handle, size_bytes, shape, typestr, *, context=None):
        self._tensor = None
        self.shape = tuple(int(s) for s in shape)
        self.typestr = str(typestr)
        itemsize = int(self.typestr[-1])
        elems = 1
        for s_ in self.shape:
            elems *= s_
        super().__init__(os_handle, size_bytes,
                         need_bytes=elems * itemsize, context=context)

    @property
    def __cuda_array_interface__(self):
        if not self.ptr:
            raise RuntimeError("this VkInteropTensor is closed")
        return {
            "shape": self.shape,
            "typestr": self.typestr,
            # read_only False: nothing stops a consumer writing, and a written
            # value simply loses to the next frame's copy.
            "data": (self.ptr, False),
            "strides": None,   # C-contiguous; the copies are tightly packed
            "stream": None,    # no stream ordering — the Vulkan fence is the sync
            "version": 3,
        }

    @property
    def tensor(self):
        """A torch tensor viewing this import. Built once, then cached."""
        if self._tensor is None:
            import torch  # lazy: this module must import without torch
            try:
                self._tensor = torch.as_tensor(self, device="cuda")
            except (TypeError, RuntimeError):
                # torch's __cuda_array_interface__ importer rejects some
                # typestrs depending on version — uint16 is the one that
                # matters. Re-expose the same bits as the signed type; see
                # FRAME_CHANNEL_DTYPES for why that is safe here.
                if self.typestr != "<u2":
                    raise
                signed = VkInteropTensorView(self, "<i2")
                self._tensor = torch.as_tensor(signed, device="cuda")
        return self._tensor

    def close(self):
        self._tensor = None
        super().close()


class VkInteropTensorView:
    """A re-typed view of a VkInteropTensor's memory, for the uint16 fallback.

    Owns nothing: it borrows the parent's pointer and dies with it.
    """

    def __init__(self, parent, typestr):
        self._parent = parent
        self._typestr = typestr

    @property
    def __cuda_array_interface__(self):
        d = dict(self._parent.__cuda_array_interface__)
        d["typestr"] = self._typestr
        return d
