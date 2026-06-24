"""threepp — three.js-style 3D for Python.

The native extension is built alongside this file as the `threepp.threepp` submodule (with its
PhysX DLLs next to it) and re-exported here, so everything is reachable as `threepp.X`:

    import threepp as tp
    scene = tp.Scene(); cam = tp.PerspectiveCamera(); ...

GPU-vectorized RL helpers live in the optional `threepp.rl` subpackage (imported on demand, so
`import threepp` does not pull in torch):

    from threepp.rl import GpuSim, ActorCritic
"""
from .threepp import *  # noqa: F401,F403  (re-export the native module's public API)
from . import threepp as _native  # keep the native module reachable as threepp._native

import contextlib as _contextlib

# Mirror the native module's public names for `from threepp import X` and dir() completeness.
__all__ = [n for n in dir(_native) if not n.startswith("_")]


# --------------------------------------------------------------------------- #
# Backend-aware material flush
#
# The Vulkan backend caches derived per-material GPU state (the MaterialDesc
# SSBO) and re-uploads it only when the material's version changes, so a plain
# field write (`mat.roughness = ...`) stays invisible until `mat.needs_update()`
# bumps the version. The GL backend re-reads material fields every frame, where
# calling needs_update() each frame would needlessly force a program-cache
# lookup. These helpers fire needs_update() *only* on the backend that needs it.
# --------------------------------------------------------------------------- #

def is_vulkan(renderer):
    """True if `renderer` is the Vulkan backend.

    Build-safe: in a GL-only build the VulkanRenderer type is never registered,
    so this is always False (no AttributeError).
    """
    vk = getattr(_native, "VulkanRenderer", None)
    return vk is not None and isinstance(renderer, vk)


def flush_material(renderer, *materials):
    """Push runtime material-property edits to the GPU, only where needed.

    Call after editing a material's scalar/color values:

        mat.roughness = 0.2
        tp.flush_material(renderer, mat)

    No-op on GL (it re-reads material state each frame); calls needs_update()
    on each material under Vulkan. Returns True iff a flush was issued.
    """
    if is_vulkan(renderer):
        for m in materials:
            m.needs_update()
        return True
    return False


@_contextlib.contextmanager
def material_edits(renderer, *materials):
    """Edit materials inside the block; flush once on exit (Vulkan only).

        with tp.material_edits(renderer, hero.material):
            hero.material.metalness = m
            hero.material.roughness = r
    """
    try:
        yield
    finally:
        flush_material(renderer, *materials)


__all__ += ["is_vulkan", "flush_material", "material_edits"]
