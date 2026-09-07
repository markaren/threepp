"""Import a Vulkan external-memory allocation into CUDA and wrap it as a Warp array.

This is the CUDA half of ``VulkanRenderer.enable_vertex_interop``: the renderer
exports its mesh position/normal buffers and hands back OS handles, and this
module turns one of those handles into a ``wp.array`` a Warp kernel can write
directly. Nothing crosses host memory.

    import threepp as tp
    from threepp.cuda_interop import VkInteropArray

    renderer.render(scene, camera)                     # the record must exist first
    h = renderer.enable_vertex_interop(mesh, on_frame) # -> ((pos, bytes), (nrm, bytes))
    pos = VkInteropArray(h[0][0], h[0][1], wp.vec3, n_verts, device)
    nrm = VkInteropArray(h[1][0], h[1][1], wp.vec3, n_verts, device)
    ...
    pos.close(); nrm.close()                           # before disable_vertex_interop

Why raw ctypes against the driver rather than ``cuda-python``: the whole surface
is four driver entry points, and ``cuda-python`` is not a threepp dependency and
should not become one for an optional fast path. Warp 1.16 ships no Vulkan
interop of its own (``RegisteredGLBuffer`` is OpenGL only), so this is ours to
write.

NVIDIA only, by construction — the fallback for everyone else is the ordinary
``geometry.update_attribute`` host route, which every demo here keeps.
"""

from __future__ import annotations

import ctypes
import sys
import warnings

import warp as wp

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
    """The CUDA driver library, loaded once.

    Warp has already called cuInit by the time anyone gets here (importing this
    module imports warp, and a device query initialises the driver), so this
    only resolves symbols.
    """
    global _driver
    if _driver is None:
        try:
            _driver = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" \
                else ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            raise RuntimeError(
                "CUDA driver library not found (nvcuda.dll / libcuda.so.1); "
                "Vulkan<->CUDA interop needs an NVIDIA driver") from e
    return _driver


def _check(code, what):
    if code == CUDA_SUCCESS:
        return
    name = ctypes.c_char_p()
    try:
        _drv().cuGetErrorName(ctypes.c_int(code), ctypes.byref(name))
    except Exception:  # pragma: no cover - only if the driver is truly broken
        pass
    tag = name.value.decode() if name.value else f"code {code}"
    raise RuntimeError(f"{what} failed: {tag} ({code})")


class VkInteropArray:
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
        self._ext_mem = None
        self._ptr = 0
        self.array = None

        dev = wp.get_device(device)
        if not dev.is_cuda:
            raise ValueError(f"VkInteropArray needs a CUDA device, got {dev}")

        # ── Guard 1: the CUDA CONTEXT ────────────────────────────────────────
        # The most likely way to get a silently wrong result. cuImportExternalMemory
        # and cuExternalMemoryGetMappedBuffer both act on the CURRENT context, and
        # `wp.array(ptr=...)` validates nothing about the pointer it is handed. If
        # torch's context (or no context at all) is current at import time, the
        # mapped pointer belongs to a context Warp never uses, and every kernel
        # launch against it is an invalid access at best and someone else's memory
        # at worst. So: make Warp's own (primary) context current, and prove it.
        dev.make_current()
        cur = ctypes.c_void_p()
        _check(_drv().cuCtxGetCurrent(ctypes.byref(cur)), "cuCtxGetCurrent")
        if not cur.value or cur.value != dev.context:
            raise RuntimeError(
                f"CUDA context mismatch before import: current={cur.value:#x} "
                f"warp={dev.context:#x}. Import from the thread Warp runs on.")

        shape_t = (shape,) if isinstance(shape, int) else tuple(shape)
        elems = 1
        for s in shape_t:
            elems *= int(s)
        itemsize = wp.types.type_size_in_bytes(dtype)
        needed = elems * itemsize
        # ── Guard 4: short import ────────────────────────────────────────────
        # size_bytes is the exporter's allocation size. Asking for a view larger
        # than it is out of bounds with no error from anyone downstream.
        if needed > int(size_bytes):
            raise ValueError(
                f"view of {shape_t} x {dtype.__name__ if hasattr(dtype, '__name__') else dtype} "
                f"needs {needed} bytes but the export is only {size_bytes}")

        hdesc = CUDA_EXTERNAL_MEMORY_HANDLE_DESC()
        hdesc.size = ctypes.c_ulonglong(int(size_bytes))
        # DEDICATED because createExternalBuffer allocates a VkDeviceMemory
        # dedicated to exactly this buffer (VkMemoryDedicatedAllocateInfo).
        hdesc.flags = CUDA_EXTERNAL_MEMORY_DEDICATED
        if sys.platform == "win32":
            hdesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32
            hdesc.handle.win32.handle = ctypes.c_void_p(int(os_handle))
            hdesc.handle.win32.name = None
        else:
            # POSIX branch: WRITTEN BUT UNTESTED — this machine is Windows, and
            # no CI job has both Vulkan and CUDA (see the plan's §6). The
            # ownership rule differs from Windows: the fd transfers to CUDA on a
            # successful import and must not be closed or re-imported here. The
            # exporter mints a fresh fd per hand-out for exactly that reason.
            hdesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
            hdesc.handle.fd = ctypes.c_int(int(os_handle))

        ext = ctypes.c_void_p()
        _check(_drv().cuImportExternalMemory(ctypes.byref(ext), ctypes.byref(hdesc)),
               "cuImportExternalMemory")
        self._ext_mem = ext

        bdesc = CUDA_EXTERNAL_MEMORY_BUFFER_DESC()
        bdesc.offset = 0
        bdesc.size = ctypes.c_ulonglong(int(size_bytes))
        bdesc.flags = 0
        dptr = ctypes.c_ulonglong()
        try:
            _check(_drv().cuExternalMemoryGetMappedBuffer(
                ctypes.byref(dptr), ext, ctypes.byref(bdesc)),
                "cuExternalMemoryGetMappedBuffer")
            self._ptr = int(dptr.value)

            # Guard 1, second half: the MAPPED POINTER's context, not just the
            # one that happened to be current. And its usable range, which is
            # what a short/rounded import would show up as.
            owner = ctypes.c_void_p()
            _check(_drv().cuPointerGetAttribute(
                ctypes.byref(owner), ctypes.c_int(CU_POINTER_ATTRIBUTE_CONTEXT), dptr),
                "cuPointerGetAttribute(CONTEXT)")
            if owner.value != dev.context:
                raise RuntimeError(
                    f"imported pointer belongs to context {owner.value:#x}, not Warp's "
                    f"{dev.context:#x}")
            rng = ctypes.c_size_t()
            _check(_drv().cuPointerGetAttribute(
                ctypes.byref(rng), ctypes.c_int(CU_POINTER_ATTRIBUTE_RANGE_SIZE), dptr),
                "cuPointerGetAttribute(RANGE_SIZE)")
            if int(rng.value) < needed:
                raise RuntimeError(
                    f"mapped range is {int(rng.value)} bytes, short of the {needed} "
                    f"the requested view uses")

            # ── Guards 5 and 6: LIFETIME ─────────────────────────────────────
            # No `deleter=`, deliberately. Given one, a wp.array frees the
            # mapping whenever it is garbage collected — including a dropped
            # local or a temporary — and would free it twice if the pointer were
            # ever wrapped again. The mapping has to outlive every view of it and
            # be released in a fixed order (mapped buffer, then the external
            # memory object) BEFORE the renderer's disable_vertex_interop. So the
            # array borrows (deleter=None is Warp's "not mine"), this object
            # owns, and close() is the only free.
            self.array = wp.array(ptr=self._ptr, dtype=dtype, shape=shape_t, device=dev)
        except Exception:
            self.close()
            raise

        self._device = dev
        self._size_bytes = int(size_bytes)

    # ── Lifetime ─────────────────────────────────────────────────────────────
    def close(self):
        """Free the mapping and the imported memory object. Idempotent.

        Order matters: the mapped buffer first (cuMemFree_v2), then the external
        memory object. Call this BEFORE the renderer's disable_vertex_interop —
        the Vulkan allocation is the renderer's, and freeing it under a live
        CUDA mapping is a use-after-free that reports as nothing at all.

        The wp.array is dropped here rather than left pointing at freed memory:
        a stale view is exactly the silent-wrongness case this class exists to
        prevent.
        """
        self.array = None
        if self._ptr:
            _drv().cuMemFree_v2(ctypes.c_ulonglong(self._ptr))
            self._ptr = 0
        if self._ext_mem is not None:
            _drv().cuDestroyExternalMemory(self._ext_mem)
            self._ext_mem = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        # Warn, then clean up. Interpreter shutdown can run this after ctypes
        # has been torn down, so it must never raise out of __del__.
        if self._ptr or self._ext_mem is not None:
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
        state = "closed" if not self._ptr else f"ptr={self._ptr:#x}"
        return f"<VkInteropArray {state} bytes={getattr(self, '_size_bytes', 0)}>"
