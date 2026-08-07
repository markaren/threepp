"""Post-build smoke test cibuildwheel runs against the installed wheel.

Asserts the wheel's feature set rather than printing it: a vcpkg step that
silently failed would otherwise produce a physics-less wheel that tests green
(find_package is QUIET, so configure succeeds either way — the same trap
config.yml's linux job guards against with its EditorConveyor_test assertion).

No GL context is created here — CI runners have no display, and physics is
deliberately CPU-only headless, which is exactly why it CAN be tested here.
"""
import threepp as tp

print("HAS_IMGUI :", tp.HAS_IMGUI)
print("HAS_PHYSX :", tp.HAS_PHYSX)

assert tp.HAS_IMGUI, "wheel built without ImGui — examples/external missing from the sdist?"
assert tp.HAS_PHYSX, "wheel built without PhysX — the vcpkg provisioning step did not take effect"

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
