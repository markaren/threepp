"""Post-build smoke test cibuildwheel runs against the installed wheel.

Asserts the wheel's feature set rather than printing it: a vcpkg step that
silently failed would otherwise produce a physics-less wheel that tests green
(find_package is QUIET, so configure succeeds either way — the same trap
config.yml's linux job guards against with its EditorConveyor_test assertion).

No GL context is created here — CI runners have no display, and physics is
deliberately CPU-only headless, which is exactly why it CAN be tested here.
"""
import pathlib

import threepp as tp

# Every .py module in the source package must be in the installed one. The
# expectation comes from the source tree next to this script, not from a list:
# python/CMakeLists.txt's install rules are what can drift. Presence is checked
# rather than import, since several modules import torch or warp at top level.
source_pkg = pathlib.Path(__file__).resolve().parents[1] / "threepp"
installed_pkg = pathlib.Path(tp.__file__).resolve().parent
assert installed_pkg != source_pkg, f"imported threepp from the source tree ({source_pkg}), not the wheel"
missing = sorted(
    p.relative_to(source_pkg).as_posix()
    for p in source_pkg.rglob("*.py")
    if "__pycache__" not in p.parts and not (installed_pkg / p.relative_to(source_pkg)).is_file()
)
assert not missing, f"modules missing from the wheel: {missing}"

print("HAS_IMGUI :", tp.HAS_IMGUI)
print("HAS_PHYSX :", tp.HAS_PHYSX)
print("HAS_VULKAN:", tp.HAS_VULKAN)
print("vulkan_available():", tp.vulkan_available())

assert tp.HAS_IMGUI, "wheel built without ImGui — examples/external missing from the sdist?"
assert tp.HAS_PHYSX, "wheel built without PhysX — the vcpkg provisioning step did not take effect"
# Vulkan is COMPILED IN (delay-loaded on Windows; loader vendored by auditwheel
# on Linux), so import must succeed and HAS_VULKAN must be True even on this
# GPU-less runner. vulkan_available() is allowed to be False here — that's the
# graceful-degradation path working; the crash mode this guards against is
# `import threepp` itself failing on machines without a Vulkan runtime.
assert tp.HAS_VULKAN, "wheel built without Vulkan — the SDK/vcpkg feature did not reach the configure"

# A box must fall. One second of CPU simulation, no renderer, no GPU DLLs —
# the wheel ships only the four CPU PhysX DLLs, and this proves they load and
# simulate (PhysXGpu_64.dll is dlopen'd lazily and must NOT be needed here).
world = tp.PhysxWorld()
box = tp.Mesh(tp.BoxGeometry(1, 1, 1), tp.MeshStandardMaterial())
box.position.set(0, 10, 0)
body = world.add(box, density=100)
assert body.is_dynamic
for _ in range(60):
    world.step(1 / 60)
assert box.position.y < 9.0, f"box did not fall (y={box.position.y})"

print(f"physics OK: box fell to y={box.position.y:.2f}")
