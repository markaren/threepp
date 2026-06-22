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

# Mirror the native module's public names for `from threepp import X` and dir() completeness.
__all__ = [n for n in dir(_native) if not n.startswith("_")]
