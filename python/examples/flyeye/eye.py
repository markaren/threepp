"""One fly eye on a threepp Vulkan secondary view: live BGRA frame -> 721 receptor inputs.

    renderer = tp.VulkanRenderer(canvas, flush_frames=1)
    configure_sensor_renderer(renderer)
    renderer.render(scene, camera)            # add_view needs a rendered frame
    eye = EyeView(renderer, eye_camera, size=403)
    renderer.render(scene, camera)
    eye.arm()                                 # FrameTensors(color, motion, depth)
    while ...:
        renderer.render(scene, camera)
        x = eye.receptors()                   # (721,) float32 CUDA, no host transfer
        lobe.step(x, dt)

Eye camera: a PerspectiveCamera with aspect 1 and vertical FOV fov_deg, looking down -Z,
+X right, +Y up. The view is size x size. Receptors are exactly lattice.luminance
(PIL luma / 255) followed by HexLattice.box_eye, computed on the live colour tensor;
views below 391 px go through box_eye's bilinear resize to 391 first.

Secondary views have no lens or sensor stage, no DLSS/FSR/DoF and no RCAS sharpen, but
the temporal resolve (TAA) always runs on them.
"""

from __future__ import annotations

import torch

from .lattice import HexLattice


def configure_sensor_renderer(renderer):
    """Switch off everything adaptive the Python API exposes. Returns what was set.

    auto_exposure off (secondary views read the primary's exposure), NoToneMapping,
    exposure 1, bloom 0, no physical camera, no DoF, no DLSS/FSR, render scale 1, no
    sensor noise, no lens distortion. Scene choices (AO, GI, fog, shadows) are left alone.
    """
    import threepp as tp

    applied = {}

    def put(name, value):
        try:
            setattr(renderer, name, value)
            applied[name] = getattr(renderer, name)
        except Exception as e:  # property missing in this build
            applied[name] = f"unavailable ({type(e).__name__})"

    put("auto_exposure", False)
    put("physical_camera", False)
    put("tone_mapping", tp.ToneMapping.NoToneMapping)
    put("tone_mapping_exposure", 1.0)
    put("bloom_intensity", 0.0)
    put("depth_of_field", False)
    put("dlss", False)
    put("fsr", False)
    put("render_scale", 1.0)
    for name, call in (("sensor_noise", lambda: renderer.set_sensor_noise(False)),
                       ("lens_distortion", lambda: renderer.set_lens_distortion("none"))):
        try:
            call()
            applied[name] = "off"
        except Exception as e:
            applied[name] = f"unavailable ({type(e).__name__})"
    return applied


def bgra_luma_u8(color: torch.Tensor) -> torch.Tensor:
    """uint8 (..., 4) BGRA -> uint8 (...) luma, bit-identical to lattice.luma_u8 on RGB."""
    c = color[..., :3].to(torch.int32)
    return ((c[..., 2] * 19595 + c[..., 1] * 38470 + c[..., 0] * 7471 + 0x8000) >> 16).to(torch.uint8)


def receptors_from_bgra(color: torch.Tensor, lattice: HexLattice) -> torch.Tensor:
    """uint8 (H, W, 4) BGRA -> (721,) float32 receptor input: luma / 255, then box_eye."""
    return lattice.box_eye(bgra_luma_u8(color).to(torch.float32) / 255)


class EyeView:
    """One persistent size x size secondary view and its live frame tensors."""

    def __init__(self, renderer, camera, size: int = 403, lattice: HexLattice | None = None):
        self.renderer, self.camera, self.size = renderer, camera, int(size)
        self.handle = int(renderer.add_view(camera, self.size, self.size))
        if self.handle == 0:
            raise RuntimeError("add_view returned 0: render() once before creating an EyeView")
        self.lattice = lattice if lattice is not None else HexLattice()
        # box_eye moves the column centres to the frame's device on every call; keep them
        # there so receptors() makes no host transfer
        self.lattice.centers = self.lattice.centers.to("cuda")
        self.frames = None

    def arm(self):
        """Import the view's colour, motion and depth as CUDA tensors. Needs one render()
        after construction. Idempotent."""
        if self.frames is None:
            from threepp.torch_frames import FrameTensors

            self.frames = FrameTensors(self.renderer, self.handle, ("color", "motion", "depth"))
            if not self.frames.bgra:
                raise RuntimeError("colour export is RGBA, expected BGRA")
        return self

    @property
    def fov_deg(self) -> float:
        return float(self.camera.fov)

    color = property(lambda self: self.frames.color)  # (size, size, 4) uint8 BGRA, live
    motion = property(lambda self: self.frames.motion)  # (size, size, 4) float16, prevNDC - currNDC
    depth = property(lambda self: self.frames.depth)  # (size, size) reversed-Z NDC

    def receptors(self) -> torch.Tensor:
        """(721,) float32 CUDA receptor input for the frame last rendered. Waits on the frame
        fence first (render() -> sync -> read); a new tensor, safe to keep.

        Needs arm() and a render() after it: the exported buffers are written by the frames
        rendered while armed, so arming and reading at once gives an all-zero frame."""
        if self.frames is None:
            raise RuntimeError("EyeView.receptors() before arm(): render(), arm(), render(), then read")
        self.frames.sync()
        return receptors_from_bgra(self.frames.color, self.lattice)

    def close(self):
        """Release the frame tensors, then remove the view."""
        if self.frames is not None:
            self.frames.close()
            self.frames = None
        if self.handle:
            self.renderer.remove_view(self.handle)
            self.handle = 0
