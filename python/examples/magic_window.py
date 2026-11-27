"""A colour photograph made of nothing but bent light, live.

Three LEDs (red, green, blue), three panes of clear glass, one white wall. The panes hold
no picture: no pigment, no print, no mask. Each is a flat sheet whose front face carries a
smooth relief a few millimetres deep, and that relief steers every ray of its LED to the
spot on the wall where the photograph needs that much of that colour. The three caustics
land on top of each other and the wall shows the photograph.

The window opens on flat glass: three plain patches of light, one on top of the other, on
the wall. Then the relief rises and the photograph comes out of them, traced as you watch:
the Vulkan path tracer runs in the window, its photons leave the LEDs, refract through the
meshes and pile up where they land. Nothing in the renderer knows about the picture. The
panel switches the LEDs, scales the relief (past 1 the light folds into caustic veins) and
moves the wall in and out of focus. Orbit with the mouse: while the camera moves the window
shows the renderer's realtime frame, which keeps up but lets light through glass unbent, so
the wall shows the plain patch; let go and the photograph develops again.

How the glass is made, all of it in this file:

  1. share out the light: lay a grid on the pane, one cell per pixel, and slide its
     vertices until every cell collects exactly the light its pixel should show
     (optimal transport, solved as a Monge-Ampere equation with a few hundred FFTs);
  2. carve the glass: Snell's law gives the tilt each point of the surface needs to throw
     its ray at its pixel, and one more FFT turns the tilts into heights.

    python magic_window.py                      # the README's Otter X screenshot, out of thin air
    python magic_window.py --image me.jpg       # any picture
    python magic_window.py --mono               # one white LED, one pane
    python magic_window.py --shot still.png     # no window: trace --spp samples and save

Needs a GPU that traces rays, numpy and Pillow.
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import threepp as tp
from threepp import capture

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE = os.path.join(HERE, "..", "..", "doc", "screenshots", "norvasundet_otter.jpg")

IOR = 1.5
PANE_W = 0.10        # m, the pane's width; its height follows the picture's shape
THICK = 0.012        # m of glass
LIGHT = 0.30         # m from an LED's plane to the panes
THROW = 1.20         # m from the panes to the wall
ZOOM = 5.0           # picture width on the wall / pane width
BAR = 0.03           # m of frame between and around the panes
LED_R = 0.0005       # m: the smaller the LED, the sharper the picture
FLOOR = 0.06         # light every pixel gets, as a share of the mean (smooth glass has no true black)
WALL = 0.85          # the wall's albedo
SAMPLES = 8          # camera paths per pixel the trace adds every frame
SETTLED = 16         # frames without a change after which the trace is shown unblended


# ---------------------------------------------------------------- 1. share out the light

def poisson(rhs):
    """lap(u) = rhs on a grid whose rim lets nothing through: mirror the grid, solve it periodic."""
    n, m = rhs.shape
    f = np.block([[rhs, rhs[:, ::-1]], [rhs[::-1], rhs[::-1, ::-1]]])
    lam = (2 * np.cos(np.pi * np.arange(2 * n) / n) - 2)[:, None] + (2 * np.cos(np.pi * np.arange(m + 1) / m) - 2)
    lam[0, 0] = 1.0
    F = np.fft.rfft2(f) / lam
    F[0, 0] = 0.0
    return np.fft.irfft2(F, s=f.shape)[:n, :m]


def laplacian(phi):
    P = np.pad(phi, 1, mode="edge")
    return P[2:, 1:-1] + P[:-2, 1:-1] + P[1:-1, 2:] + P[1:-1, :-2] - 4 * phi


def grid_of(phi):
    """The vertices of the grid, each moved by the gradient of phi (so the map has no curl
    and a surface can realise it). Units: cells. The rim's vertices stay on the rim."""
    n, m = phi.shape
    P = np.pad(phi, 1, mode="edge")
    V = np.stack(np.meshgrid(np.arange(n + 1.0), np.arange(m + 1.0), indexing="ij"), -1)
    V[..., 0] += 0.5 * ((P[1:, 1:] - P[:-1, 1:]) + (P[1:, :-1] - P[:-1, :-1]))
    V[..., 1] += 0.5 * ((P[1:, 1:] - P[1:, :-1]) + (P[:-1, 1:] - P[:-1, :-1]))
    return V


def cell_areas(V):
    d1, d2 = V[1:, 1:] - V[:-1, :-1], V[:-1, 1:] - V[1:, :-1]
    return 0.5 * (d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0])


def share_out(target, irradiance, rounds=300, tol=0.01):
    """Slide the grid on the pane until cell (i, j) collects the light pixel (i, j) shows.

    target: the picture, linear light. irradiance(V): the LED's light per area at grid
    points V. Solves det(D2 psi) = target / irradiance, psi the potential whose gradient
    draws the grid, by the fixed point of Benamou, Froese and Oberman,
    lap(psi) <- sqrt(lap(psi)^2 + 2 (f - det)), with a little momentum."""
    T = target / target.mean()
    phi = before = np.zeros_like(T)
    best = (np.inf, None)
    for it in range(rounds):
        ahead = phi + 0.8 * (phi - before) if it > 20 else phi
        V = grid_of(ahead)
        A = cell_areas(V)
        f = T / irradiance(0.25 * (V[1:, 1:] + V[:-1, :-1] + V[:-1, 1:] + V[1:, :-1]))
        f /= f.mean()
        err = np.abs(A / f - 1).max()
        if err < best[0]:
            best = (err, V)
        if err < tol:
            break
        r = np.sqrt(np.maximum((2 + laplacian(ahead)) ** 2 + 2 * (f - A), 0)) - 2
        before, phi = phi, poisson(r - r.mean())
    return best[1], best[0], it + 1


# ---------------------------------------------------------------------- 2. carve the glass

def into_glass(p, led, back_z):
    """Direction, inside the glass, of the LED's ray that reaches the front face at p
    (it bent once already, entering the flat back face)."""
    def bend(d):
        xy = d[..., :2] / (IOR * np.linalg.norm(d, axis=-1, keepdims=True))
        return np.concatenate([xy, np.sqrt(1 - (xy ** 2).sum(-1, keepdims=True))], -1)
    d = bend(p - led)
    for _ in range(3):   # where on the back face did it enter?
        d = bend(p - d * ((p[..., 2:] - back_z) / d[..., 2:]) - led)
    return d


def carve(xy, marks, led, rounds=3):
    """Heights of the front face so the ray through xy[i, j] leaves for marks[i, j].

    The pane lies in z = 0 with the wall towards +z. Snell: n1 d1 - n2 d2 is parallel to
    the surface normal, which gives the slope wanted at every vertex; the heights that fit
    the slopes best along every edge of the grid are one Poisson solve away."""
    h = np.zeros(xy.shape[:2])
    for _ in range(rounds):   # the slopes depend a little on the heights
        p = np.concatenate([xy, h[..., None]], -1)
        out = marks - p
        N = IOR * into_glass(p, led, -THICK) - out / np.linalg.norm(out, axis=-1, keepdims=True)
        g = -N[..., :2] / N[..., 2:]
        rise_i = (0.5 * (g[1:] + g[:-1]) * (xy[1:] - xy[:-1])).sum(-1)
        rise_j = (0.5 * (g[:, 1:] + g[:, :-1]) * (xy[:, 1:] - xy[:, :-1])).sum(-1)
        div = np.zeros_like(h)
        div[:-1] += rise_i; div[1:] -= rise_i
        div[:, :-1] += rise_j; div[:, 1:] -= rise_j
        h = poisson(div)
        h -= h.max()
    return np.concatenate([xy, h[..., None]], -1)


def surface_normals(P):
    """Smooth vertex normals of the grid's triangles, towards +z."""
    n, m = P.shape[0] - 1, P.shape[1] - 1
    a, b, c, d = P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]
    N = np.zeros_like(P)
    for face, corners in ((np.cross(b - a, c - a), ((0, 0), (1, 0), (1, 1))),
                          (np.cross(c - a, d - a), ((0, 0), (1, 1), (0, 1)))):
        for i, j in corners:
            N[i:i + n, j:j + m] += face
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    return N * np.sign(N[..., 2:])


def misses(P, marks, led):
    """The check that owes the renderer nothing: refract the LED's rays through the finished
    mesh by hand and measure how far from its mark each one lands, in pixels."""
    N, d1 = surface_normals(P), into_glass(P, led, -THICK)
    cos_i = (d1 * N).sum(-1, keepdims=True)
    d2 = IOR * d1 + (np.sqrt(np.maximum(1 - IOR ** 2 * (1 - cos_i ** 2), 0)) - IOR * cos_i) * N
    hit = P + d2 * ((marks[..., 2:] - P[..., 2:]) / d2[..., 2:])
    pixel = np.linalg.norm(marks[0, 1] - marks[0, 0])
    return np.linalg.norm(hit - marks, axis=-1) / pixel


def design(channel, y):
    """One pane: the carved face's vertices for one colour of the picture, thrown from a pane
    y metres above the picture's centre. Pane coordinates: x to the viewer's left, y up, z
    towards the wall; cell (row, col) of the grid is pixel (row, col) of the picture."""
    t0 = time.time()
    rows, cols = channel.shape
    pane_h = PANE_W * rows / cols
    grid = np.stack(np.meshgrid(np.arange(rows + 1.0), np.arange(cols + 1.0), indexing="ij"), -1)

    def on_pane(V):
        return np.stack([-(V[..., 1] / cols - 0.5) * PANE_W, (0.5 - V[..., 0] / rows) * pane_h], -1)

    led = np.array([0.0, y * LIGHT / THROW, -THICK - LIGHT])   # in line with the pane and the picture's centre
    aim = -led / np.linalg.norm(led)

    def irradiance(V):                                      # of the flat back face, from a small flat LED
        d = np.concatenate([on_pane(V), np.full(V.shape[:-1] + (1,), -THICK)], -1) - led
        return (d @ aim) * d[..., 2] / np.linalg.norm(d, axis=-1) ** 4

    target = channel + FLOOR * channel.mean()
    V, err, rounds = share_out(target, irradiance)
    marks = np.concatenate([on_pane(grid) * ZOOM - (0, y), np.full((rows + 1, cols + 1, 1), THROW)], -1)
    P = carve(on_pane(V), marks, led)
    off = misses(P, marks, led)
    # LED radiance that puts the picture's own linear values on the wall: the flux through the
    # pane spread over the picture's area, 8 % of it lost at the two faces
    through = irradiance(grid[:-1, :-1] + 0.5).sum() * (PANE_W / cols) ** 2 * np.pi * LED_R ** 2
    radiance = target.mean() * np.pi * ZOOM ** 2 * PANE_W * pane_h / (WALL * 0.92 * through)
    report = (f"{rounds} FFT rounds (light shared out to {100 * err:.1f} %), relief "
              f"{-1000 * P[..., 2].min():.1f} mm deep, half the rays land within {np.median(off):.2f} px "
              f"of their marks, 95 % within {np.percentile(off, 95):.1f}   [{time.time() - t0:.1f} s]")
    return P, led, float(radiance), report


def pane_geometry(P):
    """A closed block of glass: the carved face, four edge walls and the flat back.
    Returns the geometry and what set_relief needs to reshape it."""
    n, m = P.shape[0] - 1, P.shape[1] - 1
    idx = np.arange((n + 1) * (m + 1)).reshape(n + 1, m + 1)
    pos, nrm = [P.reshape(-1, 3)], [surface_normals(P).reshape(-1, 3)]
    quads = [np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)]
    count = idx.size

    def strip(top, bottom, normal):
        nonlocal count
        k = len(top)
        pos.extend([top, bottom]); nrm.append(np.tile(normal, (2 * k, 1)))
        i = count + np.arange(k - 1)
        quads.append(np.stack([i, i + 1, i + k + 1, i + k], -1))
        count += 2 * k

    def on_back(E):
        return np.concatenate([E[:, :2], np.full((len(E), 1), -THICK)], -1)

    for E in (P[0], P[-1], P[:, 0], P[:, -1]):   # edge walls, with their own vertices: sharp edges
        out = E.mean(0) * (1, 1, 0)
        strip(E, on_back(E), out / np.linalg.norm(out))
    strip(on_back(P[0, [0, -1]]), on_back(P[-1, [0, -1]]), (0, 0, -1))

    pos, nrm, q = np.concatenate(pos), np.concatenate(nrm), np.concatenate(quads)
    tri = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    inside_out = (np.cross(b - a, c - a) * nrm[tri].sum(1)).sum(-1) < 0
    tri[inside_out] = tri[inside_out][:, [0, 2, 1]]
    pos, nrm = pos.astype(np.float32), nrm.astype(np.float32)
    g = tp.BufferGeometry()
    g.set_attribute("position", pos)
    g.set_attribute("normal", nrm)
    g.set_index(tri.reshape(-1).astype(np.uint32))
    return g, pos, nrm, idx.size, (pos[:, 2] > -THICK).astype(np.float32)


def set_relief(pane, depth):
    """Scale the carved face's heights: 0 is flat glass, 1 the relief as designed."""
    g, pos, nrm, face, carved = pane
    pos, nrm = pos.copy(), nrm.copy()
    pos[:, 2] *= 1 + carved * (depth - 1)                   # the back face stays where it is
    nrm[:face, :2] *= depth                                 # a height field's normal, its slopes scaled
    nrm[:face] /= np.sqrt(np.einsum("ij,ij->i", nrm[:face], nrm[:face]))[:, None]
    g.update_attribute("position", pos)
    g.update_attribute("normal", nrm)


# ------------------------------------------------------------------------------- the bench

def load_picture(path, cells):
    """(rows, cols, 3) light, cols = cells, a touch of blur (glass cannot draw a hard edge). Light is
    the pixel value squared: the window shows the square root of the light it traces (close to sRGB
    and quick to compute), so the picture on the wall comes out as the file has it."""
    img = Image.open(path).convert("RGB")
    rows = max(8, round(cells * img.height / img.width))
    img = img.resize((cells, rows), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.8))
    return (np.asarray(img, np.float64) / 255.0) ** 2


def matte(color, roughness=0.9):
    m = tp.MeshStandardMaterial()
    m.color = tp.Color(*color)
    m.roughness, m.metalness = roughness, 0.0
    return m


def glowing(color, radiance):
    m = tp.MeshStandardMaterial()
    m.color = tp.Color(0, 0, 0)
    m.emissive = tp.Color(*color)
    m.emissive_intensity = radiance
    return m


def room_light(height=128):
    """A dark room as an equirectangular map (row 0 is straight down): a faint glow from the
    left so the bench is not a silhouette, and a soft strip light for the glass to mirror."""
    el = ((np.arange(height) + 0.5) / height - 0.5)[:, None] * np.pi
    az = ((np.arange(2 * height) + 0.5) / (2 * height) - 0.5)[None, :] * 2 * np.pi
    d = np.stack([np.cos(el) * np.cos(az), np.sin(el) * np.ones_like(az), np.cos(el) * np.sin(az)], -1)
    glow = 0.004 + 0.15 * np.clip(d @ (-0.8, 0.35, 0.5), 0, 1) ** 3
    off_strip = np.degrees(np.arctan2(d[..., 0], d[..., 2])) + 17.5      # degrees round the bench from the strip
    strip = 1.5 * np.exp(-(off_strip / 3.0) ** 2 - (np.degrees(el) / 30.0) ** 4)
    return tp.float_texture(((glow + strip)[..., None] * (0.8, 0.9, 1.0)).astype(np.float32))


def box(parent, size, at, material):
    mesh = tp.Mesh(tp.BoxGeometry(*size), material)
    mesh.position.set(*at)
    parent.add(mesh)


def build(picture, mono, log=print):
    """The bench: wall, table, the frame with its panes, the LEDs on their post."""
    rows, cols, _ = picture.shape
    pane_h = PANE_W * rows / cols
    pitch = pane_h + BAR
    if mono:
        channels = [("white", (1.0, 1.0, 1.0), picture @ (0.2126, 0.7152, 0.0722), 0)]
    else:
        channels = [(name, tuple(float(c == k) for c in range(3)), picture[..., k], 1 - k)
                    for k, name in enumerate(("red", "green", "blue"))]
    centre_y = 0.09 + 0.5 * (len(channels) * pitch + BAR)

    scene = tp.Scene()
    scene.environment = scene.background = room_light()
    wall = tp.Mesh(tp.PlaneGeometry(4.0, 2.4), matte((WALL,) * 3))
    wall.position.set(0, 1.2, -THROW)
    scene.add(wall)
    table = tp.Mesh(tp.PlaneGeometry(4.0, 5.0), matte((0.05, 0.04, 0.035), roughness=0.25))
    table.rotate_x(-np.pi / 2)
    scene.add(table)

    black = matte((0.02, 0.02, 0.02), roughness=0.6)
    shade = matte((0.02, 0.02, 0.02), roughness=0.6)
    shade.side = tp.Side.Double
    glass = tp.MeshPhysicalMaterial()
    glass.color = tp.Color(1, 1, 1)
    glass.roughness, glass.metalness = 0.0, 0.0
    glass.transmission, glass.ior, glass.thickness = 1.0, IOR, THICK   # thickness > 0: a solid, it refracts

    panes, leds = [], []
    for name, color, channel, slot in channels:
        y = slot * pitch                                    # the pane's height over the middle one
        P, led, radiance, report = design(channel, y)
        log(f"  {name:5s} pane: {report}")

        panes.append(pane_geometry(P))
        pane = tp.Mesh(panes[-1][0], glass)
        pane.rotate_y(np.pi)                                # pane z (to the wall) is the scene's -z
        pane.position.set(0, centre_y + y, 0)
        scene.add(pane)

        # the LED: a disc of light at the back of a short square tube that lets it see its own pane only
        lamp = tp.Group()
        lamp.position.set(0, centre_y + y + led[1], LIGHT + THICK)
        lamp.look_at(0, centre_y + y, THICK)
        leds.append((name, glowing(color, radiance), radiance))
        lamp.add(tp.Mesh(tp.CircleGeometry(LED_R, 24), leds[-1][1]))
        reach, margin, length = np.linalg.norm(led), 0.005, 0.08
        tube = tp.CylinderGeometry(np.sqrt(0.5), np.sqrt(0.5), 1.0, 4, 1, True)
        tube.rotate_y(np.pi / 4)
        tube.translate(0, 0.5, 0)
        tube.rotate_x(np.pi / 2)
        snoot = tp.Mesh(tube, shade)
        snoot.scale.set(2 * (0.5 * PANE_W + margin) * length / reach, 2 * (0.5 * pane_h + margin) * length / reach, length)
        lamp.add(snoot)
        box(lamp, (0.036, 0.03, 0.012), (0, 0, -0.0065), black)
        scene.add(lamp)

    # The frame that holds the panes, and the post that holds the LEDs. The bars bite a hair into the
    # glass: light that walks sideways inside a pane to its edge stops there instead of straying.
    bite, depth, z = 0.0002, THICK + 0.006, 0.5 * THICK
    height = len(channels) * pitch + BAR
    for side in (-1, 1):
        box(scene, (BAR, height, depth), (side * (0.5 * PANE_W + 0.5 * BAR - bite), centre_y, z), black)
        box(scene, (0.012, 0.09, 0.10), (side * 0.5 * (PANE_W + BAR), 0.045, z), black)
    for k in range(len(channels) + 1):
        box(scene, (PANE_W, BAR + 2 * bite, depth), (0, centre_y + (k - 0.5 * len(channels)) * pitch, z), black)
    top = centre_y + (len(channels) // 2) * pitch * (1 + LIGHT / THROW) + 0.03
    box(scene, (0.014, top, 0.014), (0, 0.5 * top, LIGHT + THICK + 0.02), black)
    box(scene, (0.12, 0.008, 0.10), (0, 0.004, LIGHT + THICK + 0.02), black)
    return scene, centre_y, wall, panes, leds


# -------------------------------------------------------------------------------- the show

def main(script=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", default=DEFAULT_IMAGE, help="the picture the glass should throw")
    ap.add_argument("--size", default="1280x720", help="window size")
    ap.add_argument("--cells", type=int, default=384, help="grid columns on each pane")
    ap.add_argument("--mono", action="store_true", help="one white LED and one pane")
    ap.add_argument("--shot", metavar="PNG", help="no window: trace --spp samples of the finished glass and save")
    ap.add_argument("--spp", type=int, default=4096, help="samples per pixel of --shot")
    args = ap.parse_args()
    width, height = (int(v) for v in args.size.lower().split("x"))

    print(f"carving glass for {os.path.basename(args.image)}")
    picture = load_picture(args.image, args.cells)
    canvas = tp.Canvas("magic window", width=width, height=height, vsync=False, headless=bool(args.shot or script))
    renderer = tp.VulkanRenderer(canvas)
    renderer.auto_lod = False
    renderer.tone_mapping = tp.ToneMapping.Linear
    renderer.sim_time = 0.0                                 # nothing here runs on the clock
    ui = tp.ImguiContext(canvas, renderer)
    scene, centre_y, wall, panes, leds = build(picture, args.mono)

    # a long lens from behind the LEDs, off to the side: lamp, panes and picture line up in one frame
    camera = tp.PerspectiveCamera(16, width / height, 0.1, 20)
    camera.position.set(0.5, centre_y + 0.15, LIGHT + 1.2)
    controls = tp.OrbitControls(camera, canvas)
    controls.target.set(-0.2, centre_y - 0.02, -THROW)
    controls.update()

    # the trace is drawn over the frame: a sprite that fills a 2D layer
    layer, lens = tp.Scene(), tp.OrthographicCamera(-1, 1, 1, -1, 0.1, 10)
    lens.position.z = 1
    paper = tp.SpriteMaterial()
    paper.tone_mapped, paper.transparent, paper.depth_test = False, True, False
    sheet = tp.Sprite(paper)
    sheet.scale.set(2, 2, 1)
    layer.add(sheet)

    state = dict(relief=1.0 if args.shot else 0.0, wall=0.0, on=[True] * len(leds), exposure=1.0, trace=True,
                 carving=not args.shot, t0=time.time(), view=None, changed=True, mixed=False, age=0, still=0,
                 fade=0.0, light=None, rgba=None, work=None)

    def resized(w, h):
        camera.aspect = w / max(h, 1)
        camera.update_projection_matrix()
        renderer.set_size(w, h)
        state["changed"] = True
    canvas.on_window_resize(resized)

    def show(light, blend):
        """Put the trace on the sheet. light: its linear picture so far. blend: fold it into what is
        shown instead of replacing it, for a bench that is moving: the trace is then restarted all
        the time and one read of it is a handful of sparse samples."""
        if state["rgba"] is None or state["rgba"].shape[:2] != light.shape[:2]:   # first picture, or a resized window
            state.update(rgba=np.full(light.shape[:2] + (4,), 255, np.uint8), work=np.empty_like(light), light=None)
            paper.map = tp.data_texture(state["rgba"])
            paper.map.min_filter = paper.map.mag_filter = tp.Filter.Nearest
            paper.map.generate_mipmaps = False
        work = state["work"]
        if blend and state["light"] is not None:
            np.subtract(light, state["light"], out=work)
            work *= 0.2
            state["light"] += work
        else:
            state["light"] = light
        np.multiply(state["light"], state["exposure"], out=work)
        np.minimum(work, 1.0, out=work)
        np.sqrt(work, out=work)
        work *= 255.0
        state["rgba"][..., :3] = work[::-1]                 # a texture's first row is its bottom one
        paper.map.update_data(state["rgba"])

    def panel():
        im = tp.imgui
        im.set_next_window_pos(10, 10)
        im.set_next_window_size(330, 0)
        im.begin("magic window")
        im.text(f"{renderer.path_trace_samples} samples per pixel, {im.get_framerate():.0f} fps")
        _, state["trace"] = im.checkbox("path trace (off: the realtime frame)", state["trace"])
        for k, (name, _, _) in enumerate(leds):
            hit, state["on"][k] = im.checkbox(f"{name} LED", state["on"][k])
            state["changed"] |= hit
        for key, label, low, high in (("relief", "relief", 0.0, 1.6), ("wall", "wall (m)", -0.5, 0.5)):
            hit, state[key] = im.slider_float(label, state[key], low, high)
            if hit:
                state.update(changed=True, carving=False)
        _, state["exposure"] = im.slider_float("exposure", state["exposure"], 0.25, 4.0)
        if im.button("carve again"):
            state.update(carving=True, t0=time.time() - 1.0)
        im.end()

    def frame():
        if script:
            script(state, renderer, canvas, camera)
        if state["carving"]:                                # the relief rises over a few seconds
            t = float(np.clip((time.time() - state["t0"] - 1.5) / 6.0, 0, 1))
            state.update(relief=t * t * (3 - 2 * t), carving=t < 1, changed=True)
        controls.enabled = not ui.want_capture_mouse
        controls.update()
        view = (camera.position.x, camera.position.y, camera.position.z,
                camera.quaternion.x, camera.quaternion.y, camera.quaternion.z)
        turning, state["view"] = view != state["view"], view
        settling = state["still"] <= SETTLED
        tracing = renderer.path_trace_active and renderer.path_trace_samples > 0

        # show what the trace has so far: every frame while it is young, now and then once it has settled
        if state["trace"] and tracing and not turning and (settling or state["still"] % 4 == 0):
            show(renderer.path_trace_read(), blend=settling)
            state["fade"] = min(1.0, state["fade"] + 0.125)

        if turning:                                         # orbiting: the realtime frame alone, which keeps up
            renderer.path_trace_stop()
            state.update(light=None, fade=0.0, still=0)
        else:
            if state["changed"]:
                for pane in panes:
                    set_relief(pane, state["relief"])
                wall.position.z = -THROW - state["wall"]
                for (_, glow, radiance), on in zip(leds, state["on"]):
                    if glow.emissive_intensity != (radiance if on else 0.0):
                        glow.emissive_intensity = radiance if on else 0.0
                        glow.needs_update()
            # A trace is of one bench. While the bench moves it is started over every few frames (starting
            # over costs a frame's worth of time), and once more when the bench has come to rest.
            if not renderer.path_trace_active or (state["mixed"] and (state["age"] >= 4 or not state["changed"])):
                renderer.path_trace_begin(samples_per_frame=SAMPLES // 2 if state["changed"] else SAMPLES, max_bounces=8)
                state.update(age=0, mixed=False)
            state["mixed"] |= state["changed"]
            state["still"] = 0 if state["changed"] else state["still"] + 1
            state["age"] += 1
            state["changed"] = False

        renderer.tone_mapping_exposure = state["exposure"]
        renderer.render(scene, camera)                      # the realtime frame, and more samples of the trace
        if state["trace"] and state["fade"] > 0:
            paper.opacity = state["fade"]
            renderer.render(layer, lens)
        ui.render(panel)
        if args.shot and renderer.path_trace_samples >= args.spp:
            show(renderer.path_trace_read(), blend=False)
            capture.write_png(args.shot, state["rgba"][::-1, :, :3])
            print(f"  {renderer.path_trace_samples} samples per pixel -> {args.shot}")
            canvas.close()

    canvas.animate(frame)


if __name__ == "__main__":
    main()
