"""flyeye: hex-lattice geometry and the T4/T5 readouts on synthetic fields, CPU torch; plus
one Vulkan-gated smoke test, last in the file.

The package lives under python/examples/flyeye and needs torch and the committed
data/flyeye_model.npz. Only test_vulkan_eye_smoke imports threepp and renders.
"""
import math
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))

from flyeye import MODEL_NPZ  # noqa: E402
from flyeye.lattice import HexLattice  # noqa: E402
from flyeye.readouts import Looming, RotationReadout  # noqa: E402

SIZE, FOV = 403, 90.0


def yaw(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


RIG = np.stack([yaw(45), yaw(-45)])  # two eyes, camera frame -> vehicle frame


@pytest.fixture(scope="module")
def lat():
    return HexLattice()


@pytest.fixture(scope="module")
def rig(lat):
    """Vehicle-frame column rays (2, 721, 3) and image tangents (2, 721, 2, 3) of RIG."""
    R = torch.from_numpy(RIG)
    d = torch.einsum("bij,nj->bni", R, lat.column_rays(SIZE, FOV, torch.float64))
    t = torch.einsum("bij,nkj->bnki", R, lat.column_tangents(SIZE, FOV, torch.float64))
    return d, t


def field(flow3, t):
    """3D flow (2, 721, 3) projected on the image tangents -> (2, 2, 721), x right, y up."""
    return torch.einsum("bni,bnki->bkn", flow3, t).float()


def rotation(w, d, t):
    return field(-torch.linalg.cross(torch.tensor(w, dtype=torch.float64).expand_as(d), d, dim=-1), t)


def translation(T, d, t):
    """Vehicle velocity T inside a unit sphere of static points."""
    T = torch.tensor(T, dtype=torch.float64).expand_as(d)
    return field(-(T - (T * d).sum(-1, keepdim=True) * d), t)


def test_lattice_geometry(lat):
    rays = lat.column_rays(SIZE, FOV, torch.float64)
    tang = lat.column_tangents(SIZE, FOV, torch.float64)
    assert rays.shape == (721, 3) and tang.shape == (721, 2, 3)
    assert torch.allclose(rays.norm(dim=1), torch.ones(721, dtype=torch.float64))
    assert torch.allclose(tang.norm(dim=2), torch.ones(721, 2, dtype=torch.float64))
    c = lat.central_column
    assert torch.allclose(rays[c], torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64), atol=1e-12)
    assert torch.allclose(tang[c], torch.tensor([[1.0, 0, 0], [0, 1.0, 0]], dtype=torch.float64), atol=1e-12)
    assert (tang * rays[:, None]).sum(-1).abs().max() < 1e-12
    # (u, v) = (+-1, 0) sit 13 px straight below / above the centre
    expected = math.atan(2 * 13 * math.tan(math.radians(FOV) / 2) / SIZE)
    for du, sign in ((1, -1), (-1, 1)):
        j = int(torch.nonzero((lat.u == du) & (lat.v == 0))[0, 0])
        assert math.acos(float(rays[c] @ rays[j])) == pytest.approx(expected, abs=1e-9)
        assert sign * rays[j, 1] > 0  # +u is down in the image
    assert torch.equal(lat.pixel_rc(SIZE), torch.from_numpy(np.load(MODEL_NPZ)["boxeye_pixel_rc_403"].astype(np.int64)))
    assert torch.equal(lat.pixel_rc(256)[c], torch.tensor([195, 195]))  # resized 391 frame


@pytest.mark.parametrize("method", ["matched", "lstsq"])
def test_rotation_readout(rig, method):
    d, t = rig
    ro = RotationReadout(SIZE, FOV, RIG, method=method)
    M = torch.stack([ro(rotation(np.eye(3)[k], d, t)) for k in range(3)])  # rows: true pitch, yaw, roll
    assert torch.allclose(M, torch.eye(3), atol=1e-4), M
    w = [0.3, -0.7, 0.5]
    assert torch.allclose(ro(rotation(w, d, t)), torch.tensor(w), atol=1e-4)
    assert torch.allclose(ro(rotation([-x for x in w], d, t)), -torch.tensor(w), atol=1e-4)


def test_looming(rig):
    d, t = rig
    lo = Looming(SIZE, FOV, RIG)

    def value(f):
        r = lo(f / f.norm(dim=-2).mean())  # mean per-column flow magnitude 1
        return r["value"].item(), r["centre"].double()

    v, centre = value(translation([0, 0, -1], d, t))
    assert v > 0.3
    assert torch.dot(centre, torch.tensor([0.0, 0, -1], dtype=torch.float64)) > math.cos(math.radians(5))
    uniform = torch.stack([torch.ones(2, 721), torch.zeros(2, 721)], dim=1)
    assert value(uniform)[0] < 0.01 * v
    assert value(translation([1, 0, 0], d, t))[0] < 0.1 * v  # expansion about +X, at the edge of the rig
    assert value(rotation([0, 1, 0], d, t))[0] < 0.01 * v


def test_vulkan_eye_smoke():
    """10 frames of a bright bar sliding past a 256 px eye view through EyeView + OpticLobe
    + MotionField.

    The only renderer test here, self-contained (its own headless canvas and renderer):
    Vulkan, CUDA torch and frame interop, else skipped.
    """
    tp = pytest.importorskip("threepp")
    if not getattr(tp, "HAS_VULKAN", False) or not torch.cuda.is_available():
        pytest.skip("needs the Vulkan backend and CUDA torch")
    from flyeye.eye import EyeView, configure_sensor_renderer
    from flyeye.optic_lobe import OpticLobe
    from flyeye.readouts import MotionField
    from threepp.torch_frames import FrameInteropUnavailable

    canvas = tp.Canvas("flyeye smoke", width=256, height=256, headless=True, vsync=False)
    r = tp.VulkanRenderer(canvas, flush_frames=1)
    configure_sensor_renderer(r)
    scene = tp.Scene()
    scene.background = tp.Color(0.2, 0.2, 0.2)
    m = tp.MeshBasicMaterial()
    m.color = tp.Color(1, 1, 1)
    bar = tp.Mesh(tp.PlaneGeometry(0.3, 4), m)
    bar.position.set(-0.6, 0, -1)
    scene.add(bar)
    cam = tp.PerspectiveCamera(90, 1.0, 0.1, 10)
    r.render(scene, cam)
    eye = EyeView(r, tp.PerspectiveCamera(90, 1.0, 0.1, 10), size=256)
    try:
        r.render(scene, cam)
        try:
            eye.arm()
        except FrameInteropUnavailable as e:
            pytest.skip(f"frame interop unavailable: {e}")
        r.render(scene, cam)
        lobe = OpticLobe(device="cuda")
        motion = MotionField(lobe)
        x0 = eye.receptors()
        assert x0.shape == (721,) and x0.dtype == torch.float32 and x0.is_cuda
        lobe.fade_in(x0, 0.01)
        t4_start = lobe.by_type("T4b").clone()
        for k in range(10):
            bar.position.x = -0.6 + 0.12 * k
            r.render(scene, cam)
            x = eye.receptors()
            v = lobe.step(x, 0.01)
            f = motion(v)
        assert eye.color.shape == (256, 256, 4) and eye.motion.shape == (256, 256, 4) and eye.depth.shape == (256, 256)
        assert v.shape == (lobe.n_nodes,) and torch.isfinite(v).all()
        assert 0.1 < x.min().item() < 0.9 < x.max().item() <= 1.0  # grey backdrop and the bright bar
        assert (lobe.by_type("T4b") - t4_start).abs().max().item() > 1e-3
        raw = motion.raw(v)
        assert raw.shape == (8, 721) and f.shape == (2, 721) and f.is_cuda and torch.isfinite(f).all()
        assert raw[:4].abs().max().item() > 1e-3 and f.abs().max().item() > 1e-3  # T4 output is non-zero
    finally:
        eye.close()
