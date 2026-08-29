"""threepp.rl.raycast / .perception — the BVH terrain query and the camera that limits it.

Needs Warp, torch and a CUDA device; no PhysX and no renderer, since the BVH is built from
plain threepp meshes. Skips cleanly everywhere else.
"""
import math

import numpy as np
import pytest
import threepp as tp

torch = pytest.importorskip("torch")
pytest.importorskip("warp")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

from threepp.rl.raycast import CollectedWorld, TerrainRays, object_soup   # noqa: E402
from threepp.rl.perception import PerceivedScan                          # noqa: E402


@pytest.fixture(scope="module")
def step_world():
    """Flat ground at z=0 with a 0.5 m riser occupying x in [2.0, 2.3]."""
    g = tp.Mesh(tp.BoxGeometry(40, 40, 1.0), tp.MeshStandardMaterial())
    g.position.set(0, 0, -0.5)
    w = tp.Mesh(tp.BoxGeometry(0.3, 6, 0.5), tp.MeshStandardMaterial())
    w.position.set(2.15, 0, 0.25)
    return TerrainRays(*object_soup([g, w]), up="z")


def test_heights_match_the_geometry(step_world):
    x = torch.tensor([[0.0, 2.15, 5.0, 100.0]], device="cuda")
    y = torch.zeros_like(x)
    h = step_world.heights(x, y)[0].tolist()
    assert h[0] == pytest.approx(0.0, abs=1e-4)
    assert h[1] == pytest.approx(0.5, abs=1e-4)
    assert h[2] == pytest.approx(0.0, abs=1e-4)
    assert h[3] == pytest.approx(step_world.miss)          # off the terrain entirely


def test_heights_results_do_not_alias(step_world):
    """Two live results must be two tensors.

    heights() used to hand back a view of one reused buffer, so the very common
    `h_here = heights(x, y)` followed by `heights(px, py)` silently repointed h_here at the
    second call's output. Every caller that holds one height while asking for another was
    reading the wrong numbers, and nothing in the shapes or the finiteness gave it away.

    The big query goes FIRST on purpose. A grow-on-demand buffer reallocates when the second
    call is larger, which hands the first result a private allocation and hides the bug —
    exactly why the env only produced garbage from its second control step onward.
    """
    wide = torch.zeros((1, 64), device="cuda")
    step_world.heights(wide, torch.zeros_like(wide))       # size any internal buffer to 64

    x1 = torch.tensor([[2.15]], device="cuda")             # on the riser -> 0.5
    h1 = step_world.heights(x1, torch.zeros_like(x1))
    assert h1.item() == pytest.approx(0.5, abs=1e-4)

    h2 = step_world.heights(wide, torch.zeros_like(wide))  # flat ground -> 0.0, 64 of them
    assert h2.abs().max().item() == pytest.approx(0.0, abs=1e-4)
    assert h1.item() == pytest.approx(0.5, abs=1e-4), "the first result moved under us"
    assert h1.data_ptr() != h2.data_ptr()


def test_out_buffer_is_honoured(step_world):
    x = torch.tensor([[2.15, 0.0]], device="cuda")
    buf = torch.empty(2, device="cuda")
    h = step_world.heights(x, torch.zeros_like(x), out=buf)
    assert h.data_ptr() == buf.data_ptr()
    assert h[0, 0].item() == pytest.approx(0.5, abs=1e-4)
    with pytest.raises(ValueError):
        step_world.heights(x, torch.zeros_like(x), out=torch.empty(3, device="cuda"))


def test_collected_world_records_what_it_forwards():
    class FakeWorld:
        def __init__(self):
            self.got = []

        def add_static(self, mesh, extra=None):
            self.got.append(mesh)
            return "handle"

        def set_gravity(self, g):
            return g

    fake = FakeWorld()
    rec = CollectedWorld(fake)
    m = tp.Mesh(tp.BoxGeometry(1, 1, 1), tp.MeshStandardMaterial())
    assert rec.add_static(m) == "handle"                   # forwards, and returns what it returns
    assert rec.set_gravity(9.81) == 9.81                   # untouched calls pass straight through
    assert rec.meshes == [m] and fake.got == [m]
    assert rec.world is fake


def test_object_soup_takes_unindexed_geometry():
    """A set_from_points soup has no index buffer; consecutive triples are the triangles."""
    pts = [tp.Vector3(0, 0, 0), tp.Vector3(1, 0, 0), tp.Vector3(0, 1, 0)]
    g = tp.BufferGeometry()
    g.set_from_points(pts)
    m = tp.Mesh(g, tp.MeshStandardMaterial())
    verts, faces = object_soup([m])
    assert verts.shape == (3, 3) and faces.shape == (1, 3)
    assert faces.tolist() == [[0, 1, 2]]


def test_a_riser_casts_a_shadow(step_world):
    """The point of the whole exercise: ground behind a step is not visible from a body camera."""
    root = torch.tensor([[0.0, 0.0, 0.55]], device="cuda")
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device="cuda")     # level, facing +x
    x = torch.tensor([[1.5, 2.15, 2.5, 3.0]], device="cuda")       # before / on / behind / behind
    y = torch.zeros_like(x)
    h = step_world.heights(x, y)
    ps = PerceivedScan(step_world, 1, origin_b=torch.zeros(1, device="cuda"),
                       bounds=(-2.0, 12.0, 1.5))
    vis = step_world.visible(root, quat, x, y, h, mount=ps.mount, mount_rot=ps.mount_rot,
                             tan_x=ps.tan_x, tan_y=ps.tan_y).tolist()[0]
    assert vis == [1, 1, 0, 0]


def test_the_map_remembers_what_the_camera_saw(step_world):
    root = torch.tensor([[0.0, 0.0, 0.55]], device="cuda")
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device="cuda")
    x = torch.tensor([[1.5, 2.15, 2.5, 3.0]], device="cuda")
    y = torch.zeros_like(x)
    ps = PerceivedScan(step_world, 1, origin_b=torch.zeros(1, device="cuda"),
                       bounds=(-2.0, 12.0, 1.5))

    ps.read(root, quat, x, y)
    assert ps.last_known.tolist()[0] == [1, 1, 0, 0]       # shadowed cells have no answer yet

    for step in range(1, 120):                             # walk past the riser
        root[0, 0] = step * 0.05
        ps.read(root, quat, x, y)
    root[0, 0] = 0.0
    ahead, _ = ps.read(root, quat, x, y)
    assert ps.last_visible.tolist()[0] == [1, 1, 0, 0]     # still no line of sight...
    assert ps.last_known.tolist()[0] == [1, 1, 1, 1]       # ...but now remembered
    assert ahead[0, 1].item() == pytest.approx(0.5, abs=1e-3)

    ps.forget()
    ps.read(root, quat, x, y)
    assert ps.last_known.tolist()[0] == [1, 1, 0, 0]       # a teleport drops the memory


def test_unknown_cells_read_as_flat(step_world):
    """Never-observed terrain must read as level ground, the way the deploy map does — not as a
    hole, and not as the truth the training env happens to have on hand."""
    root = torch.tensor([[0.0, 0.0, 0.55]], device="cuda")
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device="cuda")
    x = torch.tensor([[2.5]], device="cuda")               # in the riser's shadow, height 0.0
    y = torch.zeros_like(x)
    ps = PerceivedScan(step_world, 1, origin_b=torch.zeros(1, device="cuda"),
                       bounds=(-2.0, 12.0, 1.5))
    ahead, _ = ps.read(root, quat, x, y)
    assert ps.last_known.item() == 0
    assert ahead.item() == 0.0


def test_mount_rotation_points_forward_and_down():
    """The camera views along its own -z; that axis must come out forward and pitched down."""
    from threepp.rl.perception import mount_rotation
    qx, qy, qz, qw = mount_rotation(40.0)
    # rotate (0,0,-1) by the quaternion -> the view direction in body coordinates
    v = np.array([0.0, 0.0, -1.0])
    q = np.array([qx, qy, qz])
    view = v + 2.0 * qw * np.cross(q, v) + 2.0 * np.cross(q, np.cross(q, v))
    assert view[0] == pytest.approx(math.cos(math.radians(40.0)), abs=1e-5)   # forward
    assert view[1] == pytest.approx(0.0, abs=1e-6)                            # no yaw
    assert view[2] == pytest.approx(-math.sin(math.radians(40.0)), abs=1e-5)  # down
