"""PhysxGpuBatch's link-force write — the only way to push a batched robot.

`ArticulationLink.add_force` is a CPU-path call that PhysX rejects under direct-GPU, so before
`write_link_force` there was no way to perturb a robot at training scale at all. The subject here
is a single free link, because a lone rigid body makes the answer exact: dv = F*dt/m, with no
drives, no articulation reaction and no transient to difference away.
"""
import pytest
import threepp as tp

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not tp.HAS_PHYSX or not torch.cuda.is_available(),
    reason="needs a PhysX build and a CUDA device")

K = 4
DT = 0.01
SIZE = 0.4
DENSITY = 500.0
MASS = DENSITY * SIZE ** 3          # a solid box: PhysX takes mass from density * volume


@pytest.fixture(scope="module")
def batch():
    """K free single-link articulations, gravity off, nothing else acting on them."""
    torch.zeros(1, device="cuda")
    import ctypes
    drv = ctypes.CDLL("nvcuda.dll")
    ctx = ctypes.c_void_p()
    drv.cuCtxGetCurrent(ctypes.byref(ctx))
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, 0), direct_gpu=True, cuda_context=int(ctx.value))
    arts, meshes = [], []
    for i in range(K):
        m = tp.Mesh(tp.BoxGeometry(SIZE, SIZE, SIZE), tp.MeshStandardMaterial())
        m.position.set(0.0, i * 3.0, 1.0)
        art = world.create_articulation(fixed_base=False)
        art.add_link(m, density=DENSITY)
        art.finalize()
        arts.append(art)
        meshes.append(m)
    b = tp.PhysxGpuBatch(world, arts)
    yield b
    del b, arts, meshes, world


def _velocity(b):
    v = torch.zeros(K, 3, device="cuda")
    b.read_root_linvel(v)
    return v


def test_link_force_moves_the_body_by_f_dt_over_m(batch):
    """Exact, because a free single link has nothing else to react against."""
    f = torch.zeros(K, batch.max_links, 3, device="cuda")
    f[:, 0, 0] = 100.0
    v0 = _velocity(batch).clone()
    torch.cuda.synchronize()
    batch.write_link_force(f)
    batch.step(DT)
    dv = _velocity(batch) - v0
    assert dv[:, 0].mean().item() == pytest.approx(100.0 * DT / MASS, rel=0.02)
    assert dv[:, 1].abs().max().item() < 1e-4       # nothing leaks into the other axes
    assert dv[:, 2].abs().max().item() < 1e-4


def test_link_force_is_cleared_after_a_step(batch):
    """PhysX consumes link forces per step. If they persisted, one shove would become a thruster."""
    f = torch.zeros(K, batch.max_links, 3, device="cuda")
    f[:, 0, 0] = 100.0
    torch.cuda.synchronize()
    batch.write_link_force(f)
    batch.step(DT)
    v1 = _velocity(batch).clone()
    batch.step(DT)                                   # no second write
    assert (_velocity(batch) - v1).abs().max().item() < 1e-4


def test_link_force_is_world_frame(batch):
    """Yaw the body 90 degrees and the same force must still push along world +x."""
    import math
    half = math.pi / 4.0
    # Only the SUBSET writers are bound, so address every articulation explicitly. The pose layout
    # is QUAT-FIRST — (qx,qy,qz,qw, x,y,z) — the same trap as eROOT_GLOBAL_POSE elsewhere.
    idx = torch.arange(K, device="cuda", dtype=torch.int32)
    pose = torch.zeros(K, 7, device="cuda")
    pose[:, :4] = torch.tensor([0.0, 0.0, math.sin(half), math.cos(half)], device="cuda")
    pose[:, 5] = torch.arange(K, device="cuda", dtype=torch.float32) * 3.0
    pose[:, 6] = 1.0
    zero = torch.zeros(K, 3, device="cuda")
    batch.write_subset_root_pose(pose, idx)
    batch.write_subset_root_linvel(zero, idx)
    batch.write_subset_root_angvel(zero, idx)
    batch.step(DT)

    f = torch.zeros(K, batch.max_links, 3, device="cuda")
    f[:, 0, 0] = 100.0
    v0 = _velocity(batch).clone()
    torch.cuda.synchronize()
    batch.write_link_force(f)
    batch.step(DT)
    dv = _velocity(batch) - v0
    assert dv[:, 0].mean().item() == pytest.approx(100.0 * DT / MASS, rel=0.05)
    assert dv[:, 1].abs().max().item() < 0.05 * abs(dv[:, 0].mean().item())


def test_wrong_shape_is_rejected(batch):
    """The tensor is handed to PhysX as a device pointer, so a wrong size is silent corruption."""
    with pytest.raises(Exception):
        batch.write_link_force(torch.zeros(K, batch.max_links, 2, device="cuda"))
    with pytest.raises(Exception):
        batch.write_link_force(torch.zeros(K, batch.max_links, 3))          # cpu
