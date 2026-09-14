"""Sea-drone co-design pilot: the propeller half, differentiable in Warp.

`prop.py` is the chain (Wageningen B5-75 open-water polynomials, thrust and
torque, Burrill cavitation, and an unrolled Newton self-propulsion solve),
written so a Warp tape gives a trustworthy gradient of an endurance objective
with respect to (D, P/D, shaft depth). `optimize.py` is the gates and the run:

    python python/examples/drone_codesign/optimize.py --selftest

Nothing here imports `python/examples/warp_prop_vortex.py` -- that module opens
a window at import. Its polynomials are copied with a provenance comment and
cross-checked against its source with `ast` by `--check-tables`.
"""

from . import prop            # noqa: F401
