"""What is threepp?  Part 0: an introduction for new users, about 217 s.

    python threepp_intro.py                               # the film (1080p60), lesson_out/threepp_intro.mp4
    python threepp_intro.py --stills 30,60,95             # individual frames
    python threepp_intro.py --sheet                       # contact sheet, one frame per 3 s
    python threepp_intro.py --preview --out preview.mp4   # 960x540 @ 30 fps

The code on screen is C++. The film assembles the smallest threepp program,
hello.cpp, piece by piece (canvas, renderer, scene, camera, mesh = geometry +
material, light, render loop), shows its Python twin, then shared ownership
(eight cubes on one geometry and one material, with their use_count()), the
scene graph, materials and lights, a robot loaded from URDF and driven by
IkSolver, PhysX rigid bodies, and a DepthSensor, and ends with a CMakeLists.txt
that fetches threepp. All of it is drawn by the OpenGL renderer. The Vulkan
renderer appears only as screenshots from the repository's examples, labelled
as such.

What is on screen was checked against the library:

* every C++ snippet compiles against the headers, and the ownership beat's counts
  (9 and 9 after the loop, 1 and 1 after row->clear(), 0 after material.reset())
  are what that code prints;
* the window after the render-loop step is hello.py's own output: HELLO_PY is run
  in a separate process (headless, and turned a fixed step per frame instead of by
  the clock). A capture build of hello.cpp made the same frames at 1280 x 720 (3
  pixels of 44 million differed by more than one level), so the window is labelled
  as what hello.cpp shows;
* the printed values under the IK, PhysX and depth-camera cards, and the numbers in
  the captions, are measured during the choreography pass (6 significant digits,
  as std::cout prints a float).
"""
from __future__ import annotations

import math
import os
import re
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import lesson  # noqa: E402
from lesson import (DIM, TEXT, Arrow3D, Hud, Keys, Marker3D, OrbitCamera, Cloud, Segments, Stage,  # noqa: E402
                    clamp01, data_file, ease_out, ease_out_back, envelope, fade_in_out, mat4, remap, run, smooth,
                    smoother, standard, tp, turbo)

FPS = 60
REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))

C_ACCENT = 0x4cc9f0
C_WARM = 0xffb347
C_OK = 0x5ee27a
C_BOX = 0x4cc9f0
C_MOON = 0xf4a259
C_PEBBLE = 0xef476f
C_OUT = 0xa6e3a1         # what a program prints

# how the narrator says the names on screen (PhysX and three.js are in lesson.SPOKEN_WORDS)
WORDS = {"glTF": "G L T F", "OBJ": "O B J", "STL": "S T L", "URDF": "U R D F", "C++": "C plus plus",
         "NumPy": "Num Pie", "CMake": "C Make", "FetchContent": "Fetch Content", "OpenGL": "Open G L",
         "create()": "create"}

# ── the script ────────────────────────────────────────────────────────────────
# The narration sets the pace: every caption lasts as long as its spoken line (Kokoro,
# af_heart, measured in threepp_intro.speech.json) plus the voice's lead-in and tail and a
# small margin, and each beat's events hang off its captions. A line whose text changes should
# be re-measured (--remeasure); until then `run` holds the picture for any overrun.
CAPTION_TEXT = {
    "w1": "threepp is a 3D graphics library for C++, modelled on three.js, the most used 3D library on the web.",
    "w2": "It also comes as a Python package, with the same names and the same ideas.",
    "b1": "Every threepp program starts the same way: a canvas, which is the window, and a renderer that draws into it.",
    "b2": "The scene holds everything that will be drawn.",
    "b3": "The camera decides where we look from, and how wide the view is.",
    "b4": "A mesh is two things. Its geometry is the shape: points, joined into triangles.",
    "b5": "Its material decides how the surface looks. Without any light, it stays black.",
    "b6": "So add a light, and the material shades the box: bright where it faces the light.",
    "b7": "Last, the render loop: each frame, update the scene and draw it. That is a complete program.",
    "p1": "The Python version reads almost line for line the same, and draws the very same pixels.",
    "s1": "In C++, every create() returns a shared pointer. So eight cubes can share one geometry and one material.",
    "s2": "Change the shared material once, and all eight cubes change with it. The GPU keeps a single copy.",
    "s3": "Nobody calls delete: when the last reference goes, the memory is freed. Python's references work the same way.",
    "g1": "Objects form a tree, the scene graph. These two moons are children of the box.",
    "g2": "Move or turn a parent, and its children come along. Positions are relative to the parent.",
    "m1": "Materials decide how a surface answers light: from Basic, which ignores light, to Standard and Physical, which follow real physics.",
    "m2": "Lights come in several kinds too: directional like the sun, point like a bulb, spotlights, and soft sky light.",
    "r1": "Loaders turn files into the same kind of tree: glTF, OBJ, STL, and robots from URDF. This arm is {objects} objects.",
    "r2": "Beyond three.js there are tools for robotics. The inverse kinematics solver moves the hand onto the target, frame after frame.",
    "f1": "With PhysX built in, add meshes to a physics world and step it. The meshes follow the simulation.",
    "d1": "Simulated sensors see the same scene. This depth camera returns about {points} points per scan, each a 3D position.",
    "v1": "Everything so far was drawn by the OpenGL renderer, which runs almost anywhere. An optional Vulkan renderer adds ray-traced light, water and particles.",
    "t1": "To start in C++, pull threepp into your CMake project with FetchContent. For Python, it is pip install threepp.",
    "t2": "Then read the getting started guide, and run the examples. Each one is a small program to learn from.",
}
SPEECH = lesson.Speech(CAPTION_TEXT, os.path.join(_HERE, "threepp_intro.speech.json"), words=WORDS)
# beat, lead-in before its first caption, its captions, time after the last one
PLAN = [("open", 0.0, [], 6.0), ("what", 0.4, ["w1", "w2"], 0.3),
        ("build", 0.6, ["b1", "b2", "b3", "b4", "b5", "b6", "b7"], 1.7), ("py", 0.4, ["p1"], 0.8),
        ("share", 0.4, ["s1", "s2", "s3"], 0.5), ("graph", 0.4, ["g1", "g2"], 0.5), ("mat", 0.4, ["m1", "m2"], 0.5),
        ("robot", 0.4, ["r1", "r2"], 0.5), ("physics", 0.4, ["f1"], 0.8), ("sensor", 0.4, ["d1"], 0.5),
        ("vulkan", 0.4, ["v1"], 0.5), ("start", 0.4, ["t1", "t2"], 0.6), ("outro", 0.0, [], 8.0)]
TL, CAP = SPEECH.layout(PLAN)


def cap0(k):
    """When caption k starts."""
    return CAP[k][0]


# the build steps: (name, start, code lines of hello.cpp that belong to it)
STEPS = [("Canvas", cap0("b1") + 0.2, [0, 1, 3, 4, 5]), ("Scene", cap0("b2") + 0.2, [7]),
         ("Camera", cap0("b3") + 0.2, [9, 10, 11]), ("Mesh", cap0("b4") + 0.2, [13, 14, 15, 16]),
         ("Light", cap0("b6") + 0.2, [18, 19, 20]), ("Loop", cap0("b7") + 0.2, [22, 23, 24, 25, 26, 27])]
S = {name: t for name, t, _ in STEPS}
S_MAT = cap0("b5") + 0.2            # the material covers the wireframe
PILLS = ["Canvas", "Renderer", "Scene", "Camera", "Mesh", "Light", "Loop"]
PILL_T = [S["Canvas"], S["Canvas"] + 1.2, S["Scene"], S["Camera"], S["Mesh"], S["Light"], S["Loop"]]

# the ownership beat: one geometry and one material behind eight cubes
T_COLOUR, T_CLEAR, T_RESET = cap0("s2") + 0.7, cap0("s3") + 0.7, cap0("s3") + 4.0
C_SHARED = 0xef476f
SHARE_CPP = ["auto geometry = BoxGeometry::create(0.3f, 0.3f, 0.3f);",
             "auto material = MeshStandardMaterial::create();",
             "auto row = Group::create();",
             "",
             "for (int i = 0; i < 8; i++) {",
             "    auto cube = Mesh::create(geometry, material);",
             "    cube->position.x = i * 0.4f;",
             "    row->add(cube);      // row shares ownership",
             "}",
             "scene.add(row);",
             "",
             "material->color = 0xef476f;   // one change, 8 cubes",
             "row->clear();                 // the cubes are gone",
             "material.reset();             // the last reference"]
SHARE_T = [cap0("s1") + 0.4 + 0.28 * j for j in range(10)] + [0.0, T_COLOUR - 0.5, T_CLEAR - 0.5, T_RESET - 0.5]
N_SHARE = 8

# ── the code on screen ────────────────────────────────────────────────────────
HELLO_PY = """\
import threepp as tp

canvas = tp.Canvas("hello", width=1280, height=720)
renderer = tp.GLRenderer(canvas)

scene = tp.Scene()

camera = tp.PerspectiveCamera(60, canvas.aspect(), 0.1, 100)
camera.position.set(0, 1.5, 4)
camera.look_at(0, 0, 0)

material = tp.MeshStandardMaterial()
material.color = 0x4cc9f0
box = tp.Mesh(tp.BoxGeometry(), material)
scene.add(box)

light = tp.DirectionalLight(0xffffff, 3.0)
light.position.set(3, 6, 4)
scene.add(light)

clock = tp.Clock()
def animate():
    box.rotation.y += clock.get_delta()
    renderer.render(scene, camera)

canvas.animate(animate)""".split("\n")

HELLO_CPP = """\
#include "threepp/threepp.hpp"
using namespace threepp;

int main() {
    Canvas canvas("hello", {{"size", WindowSize{1280, 720}}});
    GLRenderer renderer(canvas);

    Scene scene;

    PerspectiveCamera camera(60, canvas.aspect(), 0.1f, 100);
    camera.position.set(0, 1.5f, 4);
    camera.lookAt(0, 0, 0);

    auto material = MeshStandardMaterial::create();
    material->color = 0x4cc9f0;
    auto box = Mesh::create(BoxGeometry::create(), material);
    scene.add(box);

    auto light = DirectionalLight::create(0xffffff, 3.f);
    light->position.set(3, 6, 4);
    scene.add(light);

    Clock clock;
    canvas.animate([&] {
        box->rotation.y += clock.getDelta();
        renderer.render(scene, camera);
    });
}""".split("\n")

THREE_WAYS = [("three.js", "js", "const box = new THREE.Mesh(new THREE.BoxGeometry(), material);", cap0("w1") + 0.6),
              ("C++", "cpp", "auto box = Mesh::create(BoxGeometry::create(), material);", cap0("w1") + 2.2),
              ("Python", "py", "box = tp.Mesh(tp.BoxGeometry(), material)", cap0("w2") + 0.4)]

GRAPH_CODE = ["auto moon = Mesh::create(SphereGeometry::create(0.2f), moonMaterial);",
              "moon->position.x = 1.2f;   // metres from the box",
              "box->add(moon);            // the moon is now the box's child"]

MAT_CODE = ["MeshBasicMaterial::create();      // flat colour, no light",
            "MeshLambertMaterial::create();    // matte",
            "MeshPhongMaterial::create();      // matte + shiny highlight",
            "MeshStandardMaterial::create();   // physically based",
            "MeshPhysicalMaterial::create();   // + clearcoat, glass, sheen"]

ROBOT_CODE = ["URDFLoader loader;",
              'auto robot = loader.load("fr3.urdf");',
              "scene.add(robot);",
              "IkSolver solver(*robot);",
              "auto result = solver.solve(q, target);   // updates q",
              "robot->setJointValues(q);",
              "std::cout << result.positionError;"]

PHYS_CODE = ["PhysxWorld world;",
             "world.addStatic(*floor);",
             "world.add(*box, 500);     // density, kg/m³",
             "world.step(dt);           // the box follows the simulation",
             "std::cout << box->position.y;"]

SENSOR_CODE = ["DepthSensor sensor(50, 160, 120, 0.1f, 6.f);   // fov, width, height, near, far",
               "scene.addRef(sensor);",
               "std::vector<Vector3> points;",
               "sensor.scan(renderer, scene, points);",
               "std::cout << points.size();"]

START_PY = ["pip install threepp"]
START_CMAKE = ["cmake_minimum_required(VERSION 3.21)",
               "project(hello)",
               "set(CMAKE_CXX_STANDARD 20)",
               "",
               "include(FetchContent)",
               "FetchContent_Declare(threepp",
               "    GIT_REPOSITORY https://github.com/markaren/threepp.git",
               "    GIT_TAG master)      # or a release tag",
               "FetchContent_MakeAvailable(threepp)",
               "",
               "add_executable(hello hello.cpp)",
               "target_link_libraries(hello PRIVATE threepp::threepp)"]

GALLERY = [("sailboat_golden.png", "FFT ocean, cloth sails"), ("sponza.png", "Global illumination"),
           ("vulkan_fire.png", "GPU particle fire"), ("aalesund.png", "Ålesund from open map data"),
           ("warp_fluid.png", "NVIDIA Warp fluid"), ("forest.png", "Procedural trees and grass")]

SUMMARY = [(r"$\mathrm{API}$", "The three.js API in C++, and in Python too."),
           (r"$\mathrm{7\ pieces}$", "Canvas, renderer, scene, camera, mesh, light, loop."),
           (r"$\mathrm{shared\_ptr}$", "Share geometry and materials; memory frees itself."),
           (r"$\mathrm{a\ tree}$", "A scene is a graph: robots, physics and sensors join it.")]

# ── syntax colouring ──────────────────────────────────────────────────────────
KW = {"py": {"import", "as", "def", "return", "for", "in", "if", "else", "True", "False", "None", "from"},
      "cpp": {"using", "namespace", "int", "auto", "return", "const", "new", "float"},
      "js": {"const", "new", "let", "function"}, "cmake": set(), "sh": {"pip", "python"}}
COL = {"kw": 0xc792ea, "str": 0xc3e88d, "num": 0xf78c6c, "fn": 0x82aaff, "type": 0xffcb6b, "mod": 0x89ddff,
       "com": 0x6b7a90, "punc": 0x8fa3bf, "id": 0xe6ecf5, "pre": 0xc792ea}
_TOK = re.compile(r'(//.*|#.*)|("[^"]*")|(\b0x[0-9a-fA-F]+\b|\b\d+\.?\d*f?\b)|([A-Za-z_][A-Za-z_0-9]*)|(\s+)|(.)')


def tokenize(line, lang):
    """(column, text, colour) per token; whitespace is skipped."""
    out = []
    ms = list(_TOK.finditer(line))
    for k, m in enumerate(ms):
        s = m.group(0)
        if m.group(5):
            continue
        if m.group(1):
            if lang == "cpp" and s.startswith("#"):          # #include "..."
                word = s.split()[0]
                out.append((m.start(), word, COL["pre"]))
                rest = s[len(word):]
                if rest.strip():
                    out.append((m.start() + len(word) + (len(rest) - len(rest.lstrip())), rest.strip(), COL["str"]))
                continue
            if (lang == "cpp") != s.startswith("//"):         # a '#' in C++ or '//' elsewhere is not a comment
                out.append((m.start(), s, COL["punc"]))
                continue
            out.append((m.start(), s, COL["com"]))
        elif m.group(2):
            out.append((m.start(), s, COL["str"]))
        elif m.group(3):
            out.append((m.start(), s, COL["num"]))
        elif m.group(4):
            nxt = line[m.end():].lstrip()[:1]
            if s in KW.get(lang, ()):
                c = COL["kw"]
            elif s in ("tp", "THREE", "threepp"):
                c = COL["mod"]
            elif s[0].isupper():
                c = COL["type"]
            elif nxt == "(":
                c = COL["fn"]
            else:
                c = COL["id"]
            out.append((m.start(), s, c))
        else:
            out.append((m.start(), s, COL["punc"]))
    return out


def code_block(ov, x, y, lines, lang, size=20, lh=27, alpha=1.0, line_alpha=None, hl=None, reveal=None):
    """Source lines in a monospace font with syntax colours. line_alpha(j), hl(j) and
    reveal(j) (0..1 of the line's tokens shown) are optional per-line functions."""
    if alpha <= 0.003:
        return
    cw = ov.text_width("0", size, "mono")
    for j, line in enumerate(lines):
        a = alpha * (line_alpha(j) if line_alpha else 1.0)
        if a <= 0.003 or not line.strip():
            continue
        yy = y + j * lh
        h = hl(j) if hl else 0.0
        if h > 0.003:
            ov.panel(x - 16, yy - 2, len(max(lines, key=len)) * cw + 30, lh + 2, radius=5, fill=0x223452,
                     alpha=0.75 * h * alpha)
            ov.panel(x - 16, yy - 2, 4, lh + 2, radius=2, fill=C_ACCENT, alpha=h * alpha)
        toks = tokenize(line, lang)
        n = len(toks) if reveal is None else int(math.ceil(reveal(j) * len(toks) - 1e-9))
        for col, s, c in toks[:n]:
            ov.text(x + col * cw, yy + lh / 2, s, size=size, color=c, alpha=a, kind="mono", anchor="lm")


def card(ov, x, y, w, h, alpha, title=None):
    ov.panel(x, y, w, h, radius=16, alpha=0.74 * alpha, outline=0x8aa0c0, outline_alpha=0.16)
    if title:
        ov.text(x + 26, y + 22, title, size=16, color=DIM, alpha=alpha, kind="semibold", tracking=2.2)


def code_card(ov, t, t_in, t_out, x, y, lines, lang, title, size=20, lh=27, stagger=0.35, min_w=0, out=None,
              out_t=None):
    """A titled card of code whose lines type in one after another from t_in. `out`: lines
    the program prints, shown under the code from `out_t` (default: once the code is in)."""
    a = envelope(t, t_in, t_out, 0.6, 0.6)
    if a <= 0.003:
        return
    cw = ov.text_width("0", size, "mono")
    w = max(max(len(l) for l in lines + (out or [])) * cw + 60, min_w)
    h = 64 + lh * len(lines) + (lh * len(out) + 26 if out else 0)
    card(ov, x, y, w, h, a, title)
    code_block(ov, x + 30, y + 50, lines, lang, size, lh, a,
               reveal=lambda j: remap(t, t_in + 0.3 + stagger * j, t_in + 0.9 + stagger * j))
    if out:
        oa = a * smooth(remap(t, out_t if out_t is not None else t_in + 0.9 + stagger * len(lines),
                              (out_t if out_t is not None else t_in + 0.9 + stagger * len(lines)) + 0.4))
        yy = y + 50 + lh * len(lines) + 10
        ov.panel(x + 12, yy - 4, w - 24, lh * len(out) + 12, radius=8, fill=0x070a10, alpha=0.8 * oa)
        for j, line in enumerate(out):
            ov.text(x + 30, yy + 2 + lh * j + lh / 2, line, size=size, color=C_OUT, alpha=oa, kind="mono",
                    anchor="lm")


# ── choreography ──────────────────────────────────────────────────────────────
DOF = 7
FINGERS = 0.035
Q_READY = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]

# the three sets, in the Y-up world: the robot stands at the origin, facing +x
A = np.array([2.4, 0.0, 0.9])          # the hello box
B = np.array([-2.4, 0.0, 0.9])         # the materials
D = np.array([6.2, 0.0, 0.9])          # eight cubes, one geometry, one material
BOX_Y = 0.9
PILE = np.array([0.12, 0.0, 0.62])     # where the PhysX boxes land
POST = np.array([-0.85, 0.0, 1.45])    # the depth camera's post
SENSOR_AT = np.array([-0.85, 1.3, 1.45])
SENSOR_LOOK = np.array([0.2, 0.3, 0.35])
N_BOXES = 40
SCAN_EVERY = 4                          # frames between depth scans (15 Hz)


def full(q7):
    return list(map(float, q7[:DOF])) + [FINGERS, FINGERS]


def tool_down(p, yaw=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    M = np.eye(4)
    M[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ np.diag([1.0, -1.0, -1.0])
    M[:3, 3] = p
    return M


def target_path(t):
    """The IK target in the robot frame (Z-up): a slow loop in front of the arm."""
    s = 0.55 * (t - cap0("r2") + 0.4)
    return np.array([0.50 + 0.09 * math.sin(s), 0.22 * math.sin(0.5 * s + 0.3), 0.30 + 0.11 * math.cos(s)])


def world_to_zup(p):
    p = np.asarray(p, float)
    return np.array([p[0], -p[2], p[1]])


class Record:
    def __init__(self, n):
        self.n = n
        self.q = np.tile(np.array(Q_READY, float), (n, 1))
        self.target = np.tile(np.eye(4), (n, 1, 1))
        self.ik_err = np.zeros(n)
        self.box_pos = np.zeros((n, N_BOXES, 3))
        self.box_quat = np.zeros((n, N_BOXES, 4))
        self.box_quat[..., 3] = 1.0
        self.spawn = np.zeros(N_BOXES, int)
        self.scans = {}
        self.meta = {}


def choreograph(st, robot, boxes, verbose=True):
    n = int(round(TL.duration * FPS)) + 1
    rec = Record(n)
    idx = lambda t: int(round(t * FPS))  # noqa: E731

    # ---- IK: from the ready pose, glide onto a slow loop, one solve per frame
    opts = tp.IkOptions()
    opts.task = tp.IkTask.Pose
    opts.max_iterations = 100
    opts.position_tolerance = 1e-4
    opts.orientation_tolerance = 1e-3
    opts.rest_pose = full(Q_READY)
    opts.rest_pose_gain = 0.05
    solver = tp.IkSolver(robot, opts)
    home = robot.compute_end_effector_transform(full(Q_READY)).to_numpy().astype(float)
    t_ik = cap0("r2") - 0.2
    q = full(Q_READY)
    for i in range(idx(t_ik), n):
        t = i / FPS
        u = smoother(remap(t, t_ik, t_ik + 2.0))
        p = home[:3, 3] * (1 - u) + target_path(t) * u
        T = tool_down(p, 0.35 * math.sin(0.4 * t) * u)
        q, res = solver.solve(q, mat4(T))
        rec.q[i] = q[:DOF]
        rec.target[i] = T
        rec.ik_err[i] = res.position_error
    rec.target[:idx(t_ik)] = rec.target[idx(t_ik)]
    rec.meta["ik_t"] = t_ik
    if verbose:
        print(f"[ik] {n - idx(t_ik)} solves, worst position error {rec.ik_err.max() * 1000:.2f} mm")

    # ---- PhysX: boxes dropped one after another onto a pile beside the arm
    world = tp.PhysxWorld()
    floor = tp.Mesh(tp.BoxGeometry(8, 0.2, 8), tp.MeshBasicMaterial())
    floor.position.set(0, -0.1, 0)
    floor.update_matrix_world()
    world.add_static(floor)
    rng = np.random.default_rng(7)
    t_p0 = TL.start("physics") + 0.6
    spawn_t = t_p0 + np.sort(rng.uniform(0.0, 5.2, N_BOXES))
    rec.spawn = np.array([idx(s) for s in spawn_t])
    starts = []
    for b in range(N_BOXES):
        p = PILE + np.array([rng.normal(0, 0.07), rng.uniform(0.9, 1.3), rng.normal(0, 0.07)])
        e = rng.uniform(0, math.pi, 3)
        starts.append((p, e))
    added = [False] * N_BOXES
    for i in range(idx(t_p0) - 1, n):
        for b in range(N_BOXES):
            if not added[b] and i >= rec.spawn[b]:
                m = boxes[b]
                p, e = starts[b]
                m.position.set(*p)
                m.rotation.set(*e)
                m.update_matrix_world()
                world.add(m, density=500)
                added[b] = True
        world.step(1.0 / FPS)
        for b in range(N_BOXES):
            m = boxes[b]
            if added[b]:
                rec.box_pos[i, b] = (m.position.x, m.position.y, m.position.z)
                rec.box_quat[i, b] = (m.quaternion.x, m.quaternion.y, m.quaternion.z, m.quaternion.w)
    if verbose:
        print(f"[physx] {N_BOXES} boxes, resting height {rec.box_pos[-1, :, 1].max():.2f} m")
    return rec


def scan_pass(st, sets, rec, sensor, painter, verbose=True):
    """Depth scans of the recorded scene: pose everything as it is at that frame, scan."""
    t0s, t1s = TL.start("sensor") + 0.8, TL.end("sensor") + 0.6
    for i in range(int(t0s * FPS), int(t1s * FPS), SCAN_EVERY):
        painter.pose_frame(i / FPS, i)
        for o in sets:
            o.visible = False
        st.scene.update_matrix_world()
        pts = sensor.scan(st.r, st.scene)
        rec.scans[i] = np.asarray(pts, np.float32).copy()
    counts = [len(p) for p in rec.scans.values()]
    rec.meta["scan_pts"] = int(np.median(counts))
    if verbose:
        print(f"[depth] {len(counts)} scans of 160 x 120, median {np.median(counts):.0f} points")


HELLO_FRAMES = 48       # hello.py's picture over a quarter turn (a cube repeats every 90 degrees)
_HELLO_RUNNER = """
import math, sys
import numpy as np
sys.path.insert(0, {py!r})
src = {src!r}
# the two changes that make it a batch job: no window, and a fixed turn per frame instead of the clock
src = src.replace('height=720)', 'height=720, headless=True)', 1)
src = src.replace('canvas.animate(animate)', '')
ns = {{}}
exec(src, ns)
frames = []
for k in range({n}):
    ns['box'].rotation.y = k * (math.pi / 2) / {n}
    ns['renderer'].render(ns['scene'], ns['camera'])
    frames.append(ns['renderer'].read_pixels()[::2, ::2])
np.save({out!r}, np.stack(frames))
"""


def run_hello(outdir):
    """Run HELLO_PY, as printed on screen, in its own process and collect its frames: what a
    student sees when they run it. Returns (HELLO_FRAMES, 360, 640, 3) uint8."""
    import subprocess
    import tempfile
    out = os.path.join(tempfile.gettempdir(), f"threepp_intro_hello_{os.getpid()}.npy")
    code = _HELLO_RUNNER.format(py=os.path.dirname(os.path.dirname(_HERE)), src="\n".join(HELLO_PY),
                                n=HELLO_FRAMES, out=out)
    subprocess.run([sys.executable, "-c", code], check=True)
    frames = np.load(out)
    os.remove(out)
    return frames


# ── the picture ───────────────────────────────────────────────────────────────
WIDE = (0.0, 0.55, 1.3)
LOOK_BUILD = A + (-1.2, 0.58, 0)
LOOK_SHARE = D + (-1.1, 0.5, 0)
LOOK_GRAPH = A + (-1.15, 0.8, 0)
LOOK_MAT = B + (0.0, 0.55, 0)
LOOK_ROBOT = (-0.38, 0.5, 0.05)
LOOK_PHYS = (0.0, 0.4, 0.4)
LOOK_SENSOR = (0.05, 0.5, 0.65)
CAMERA = OrbitCamera([    # (time, azimuth, elevation, distance, look point, fov)
    (0.0, -90, 17, 7.4, WIDE, 32),
    (TL.end("open") - 1.0, -88, 16, 7.0, WIDE, 32),
    (TL.start("what") + 1.5, -84, 11, 5.0, A + (-1.15, 0.62, 0), 32),
    (TL.end("what") - 0.5, -88, 10, 5.2, LOOK_BUILD, 32),
    (TL.start("build") + 1.0, -90, 10, 5.2, LOOK_BUILD, 32),
    (TL.end("py") - 0.5, -93, 11, 5.2, LOOK_BUILD, 32),
    (TL.start("share") + 1.6, -90, 13, 5.8, LOOK_SHARE, 32),
    (TL.end("share") - 1.0, -93, 13, 5.8, LOOK_SHARE, 32),
    (TL.start("graph") + 1.5, -86, 17, 5.6, LOOK_GRAPH, 32),
    (TL.end("graph") - 0.5, -92, 17, 5.6, LOOK_GRAPH, 32),
    (TL.start("mat") + 2.0, -90, 12, 4.0, LOOK_MAT, 32),
    (TL.end("mat") - 0.5, -92, 13, 4.0, LOOK_MAT, 32),
    (TL.start("robot") + 2.0, -66, 14, 3.0, LOOK_ROBOT, 32),
    (TL.end("robot") - 0.5, -70, 16, 3.0, LOOK_ROBOT, 32),
    (TL.start("physics") + 2.0, -52, 22, 3.0, LOOK_PHYS, 32),
    (TL.end("physics") - 0.5, -56, 23, 3.0, LOOK_PHYS, 32),
    (TL.start("sensor") + 2.0, -78, 24, 3.8, LOOK_SENSOR, 32),
    (TL.end("sensor") - 0.5, -84, 24, 3.8, LOOK_SENSOR, 32),
    (TL.end("vulkan"), -84, 22, 5.0, (0.0, 0.5, 0.8), 32),
    (TL.start("start") + 3.0, -90, 19, 7.4, WIDE, 32),
    (TL.end("outro"), -96, 19, 7.4, WIDE, 32),
], frame="yup", drift=((1.0, 0.19), (0.5, 0.15)))


def share_pop(k):
    """When cube k of the ownership beat appears: one per pass of the loop."""
    return cap0("s1") + 3.4 + 0.22 * k


def lights_on(t):
    """0..1 scale on every light: off from the black of the build until its light step."""
    if t < TL.start("build") + 0.3:
        return 1.0
    if t < S["Light"]:
        return 0.0
    return smooth(remap(t, S["Light"], S["Light"] + 1.6)) if t < S["Light"] + 1.6 else 1.0


def box_angle(t):
    """The box's rotation.y: turning in the preview, still while it is built, then the loop."""
    w = 0.8
    if t < TL.start("build") + 0.4:
        return 0.55 + 0.6 * t
    t0, ramp = S["Loop"] + 1.0, 1.2
    if t < t0:
        return 0.55
    if t < t0 + ramp:
        return 0.55 + w * (t - t0) ** 2 / (2 * ramp)
    return 0.55 + w * (t - t0 - ramp / 2)


class Painter:
    def __init__(self, st, rec, robot, parts, sensor, ov):
        self.st, self.rec, self.robot, self.ov, self.sensor = st, rec, robot, ov, sensor
        self.p = parts
        self.base = dict(hemi=st.hemi.intensity, key=st.key.intensity, rim=st.rim.intensity,
                         env=st.scene.environment_intensity if st.scene.environment is not None else 0.0)
        self.captions = self.make_captions()
        self.textures = []
        for f, _ in GALLERY:
            tex = tp.TextureLoader().load(os.path.join(REPO, "doc", "screenshots", f))
            tex.color_space = tp.ColorSpace.SRGB
            self.textures.append(tex)
        self._cropped = [False] * len(GALLERY)
        self.hello = [tp.data_texture(np.ascontiguousarray(f), True) for f in run_hello(None)]
        print(f"[hello.py] ran it: {len(self.hello)} frames of its own window, "
              f"{self.hello_px[0]} x {self.hello_px[1]} (downsampled from 1280 x 720)")

    hello_px = (640, 360)

    def make_captions(self):
        """The captions, with the measured numbers filled in."""
        m = self.rec.meta
        return SPEECH.captions(CAP, fields={"objects": m["robot_objects"],
                                            "points": f"{int(round(m['scan_pts'], -3)):,}"})

    # 3D ----------------------------------------------------------------------
    def pose_frame(self, t, i):
        """Everything in the 3D scene that depends on t (poses, visibility, lights)."""
        st, p, rec = self.st, self.p, self.rec
        # which sets are on stage
        wide = t < TL.start("what") + 2.0 or t >= TL.start("start")
        on_a = wide or t < TL.start("share") + 1.5 or TL.end("share") - 1.5 <= t < TL.end("graph") + 2.5
        on_b = t >= TL.start("start") or TL.start("mat") - 2.5 <= t < TL.end("mat") + 2.5
        on_c = t >= TL.start("robot") + 0.4
        p["A"].visible = on_a
        p["B"].visible = on_b
        p["D"].visible = TL.start("share") - 2.5 <= t < TL.end("share") + 2.5

        # eight cubes on one material: they arrive with the loop, turn colour together, then go
        for k, m in enumerate(p["share_cubes"]):
            s = ease_out_back(remap(t, share_pop(k), share_pop(k) + 0.45)) * (1 - smooth(remap(t, T_CLEAR, T_CLEAR + 0.5)))
            m.visible = s > 0.002
            m.scale.set(max(s, 1e-3), max(s, 1e-3), max(s, 1e-3))
            m.rotation.y = 0.5 + 0.3 * t
        u = smooth(remap(t, T_COLOUR, T_COLOUR + 0.35))
        c0, c1 = np.array(lesson.hex_rgb(C_BOX)), np.array(lesson.hex_rgb(C_SHARED))
        r_, g_, b_ = (int(round(v)) for v in c0 * (1 - u) + c1 * u)
        p["share_mat"].color = (r_ << 16) | (g_ << 8) | b_
        self.robot.visible = on_c
        p["post"].visible = t >= TL.start("sensor") + 0.2

        # the build: background from black, lights off until their step
        L = lights_on(t)
        building = TL.start("build") + 0.3 <= t < S["Scene"] + 1.5
        bg = smooth(remap(t, S["Scene"], S["Scene"] + 1.2)) if building else 1.0
        r_, g_, b_ = (int(round(c * bg)) for c in lesson.hex_rgb(Stage.BG))
        hexc = (r_ << 16) | (g_ << 8) | b_
        st.scene.background = tp.Background(hexc)
        st.scene.set_fog(tp.Color(hexc), 5.5, 14.0)
        st.hemi.intensity = self.base["hemi"] * L
        st.key.intensity = self.base["key"] * L
        st.rim.intensity = self.base["rim"] * L
        if st.scene.environment is not None:
            st.scene.environment_intensity = self.base["env"] * L
        st.floor.visible = L > 0.003

        # the hello box
        in_build = TL.start("build") + 0.3 <= t
        solid = not in_build or t >= S_MAT
        p["box"].visible = not in_build or t >= S["Mesh"]
        p["box"].material.visible = solid
        wire = 0.0
        if in_build and t < S["Loop"]:
            wire = smooth(remap(t, S["Mesh"], S["Mesh"] + 0.8)) * (1.0 - 0.7 * smooth(remap(t, S_MAT, S_MAT + 0.8)))
            wire *= 1.0 - smooth(remap(t, S["Light"] + 0.4, S["Light"] + 1.6))
        p["wire"].visible = wire > 0.003
        p["wire"].material.opacity = wire
        p["box"].rotation.y = box_angle(t)
        # the graph beat: moons pop in, then the parent goes for a walk
        g0 = TL.start("graph") + 0.8
        pop = [ease_out_back(remap(t, g0 + 0.5 * k, g0 + 0.5 * k + 0.7)) for k in range(3)]
        for k, m in enumerate((p["moon"], p["moon2"], p["pebble"])):
            s = max(pop[k], 1e-3)
            m.scale.set(s, s, s)
            m.visible = pop[k] > 0.002
        p["moon"].rotation.y = 2.0 * t
        u = remap(t, cap0("g2") + 0.4, CAP["g2"][1] - 0.6)
        walk = math.sin(math.pi * smoother(u))
        p["box"].position.set(A[0] - 0.55 * walk, BOX_Y + 0.35 * walk, A[2] + 0.1 * walk)

        # the light's direction, while its step is on
        la = envelope(t, S["Light"] + 0.2, S["Loop"] - 0.3, 0.6, 0.6)
        tip = A + np.array([0, BOX_Y, 0]) + np.array([3, 6, 4]) / np.linalg.norm([3, 6, 4]) * 0.75
        tail = A + np.array([0, BOX_Y, 0]) + np.array([3, 6, 4]) / np.linalg.norm([3, 6, 4]) * 1.9
        p["sun_arrow"].set(tail, tip, opacity=la)

        # materials
        for k, m in enumerate(p["knots"]):
            m.rotation.set(0.35, 0.45 * t + 0.6 * k, 0.0)
        ang = 0.8 * t
        p["bulb"].position.set(*(B + np.array([1.7 * math.cos(ang), 0.95, 0.9 * math.sin(ang)])))
        bulb = envelope(t, cap0("m2") - 0.4, TL.end("mat") + 1.0, 0.8, 0.8)
        p["bulb_light"].intensity = 5.0 * bulb
        p["bulb"].visible = bulb > 0.003

        # the robot: meshes appear link by link while it loads
        r0 = TL.start("robot") + 0.6
        shown = remap(t, r0, r0 + 2.4) * 8.999
        for m, k in p["robot_meshes"]:
            m.visible = k <= shown
        self.robot.set_joint_values(full(rec.q[i]))
        p["target"].place(rec.target[i])
        tv = envelope(t, rec.meta["ik_t"] - 0.4, TL.duration + 5, 0.6, 0.6)
        p["target"].set_opacity(tv, pulse=0.5 + 0.5 * math.sin(4 * t))

        # PhysX boxes
        for b, m in enumerate(p["boxes"]):
            if i >= rec.spawn[b]:
                m.visible = True
                m.position.set(*rec.box_pos[i, b])
                m.quaternion.set(*rec.box_quat[i, b])
            else:
                m.visible = False
        p["cloud"].points.visible = False
        p["frustum"].lines.visible = False

    def set3d(self, t, i):
        self.pose_frame(t, i)
        st, p, rec = self.st, self.p, self.rec
        eye, look, fov = CAMERA(t)
        st.look(eye, look, fov)
        st.follow([look[0], 0.0, look[2]])

        # the depth camera: its frustum and the latest scan
        sa = envelope(t, TL.start("sensor") + 0.6, TL.end("sensor") + 0.8, 0.6, 0.8)
        if sa > 0.003:
            keys = [k for k in rec.scans if k <= i]
            if keys:
                pts = rec.scans[max(keys)]
                rng = np.linalg.norm(pts - SENSOR_AT[None, :], axis=1)
                col = turbo(1.0 - np.clip((rng - 0.8) / 1.9, 0, 1)).astype(np.float32)
                p["cloud"].set(pts, col, opacity=sa)
            self.sensor.update_matrix_world()
            M = self.sensor.matrix_world.to_numpy().astype(float)
            o = M[:3, 3]
            d = 0.6
            hy = d * math.tan(math.radians(25.0))
            hx = hy * 160 / 120
            corners = [(M @ np.array([sx * hx, sy * hy, -d, 1.0]))[:3] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            a_ = [o] * 4 + corners
            b_ = corners + corners[1:] + corners[:1]
            c = np.tile(np.float32([[0.35, 0.8, 1.0]]), (8, 1)) * sa
            p["frustum"].set(np.array(a_, np.float32), np.array(b_, np.float32), c)

    # 2D ----------------------------------------------------------------------
    def draw2d(self, t, i):
        ov, st, rec = self.ov, self.st, self.rec
        ov.title_card(t, "A THREEPP LESSON  ·  PART 0", "What is threepp?",
                      "3D graphics in C++, from a spinning box to a robot", C_ACCENT, t_in=0.6,
                      t_out=5.9)
        self.what_card(t)
        self.build_hud(t)
        self.share_hud(t)
        self.graph_hud(t)
        self.mat_hud(t)
        self.robot_hud(t, i)
        self.physics_hud(t, i)
        self.sensor_hud(t, i)
        self.gallery(t)
        self.start_card(t)
        ov.captions(t, self.captions)
        ov.summary(t, TL.start("outro"), TL.end("outro"), SUMMARY,
                   "Every frame of this film was drawn by threepp's OpenGL renderer", C_ACCENT,
                   C_ACCENT, text_dx=300)

    def what_card(self, t):
        ov = self.ov
        a = envelope(t, TL.start("what") + 0.8, TL.end("what") - 0.2, 0.6, 0.5)
        if a <= 0.003:
            return
        x, y = 70, 300
        cw = ov.text_width("0", 21, "mono")
        w = max(len(c) for _, _, c, _ in THREE_WAYS) * cw + 220
        card(ov, x, y, w, 70 + 3 * 76, a, "ONE LINE, THREE LANGUAGES")
        for k, (name, lang, code, t_in) in enumerate(THREE_WAYS):
            ra = a * smooth(remap(t, t_in, t_in + 0.6))
            yy = y + 62 + 76 * k
            ov.text(x + 28, yy + 14, name, size=22, color=C_ACCENT if lang == "py" else DIM, alpha=ra,
                    kind="semibold", anchor="lm")
            code_block(ov, x + 170, yy, [code], lang, size=21, lh=28, alpha=ra)

    def build_hud(self, t):
        ov = self.ov
        # the black before the build, so the pieces arrive on an empty screen
        pre = TL.start("build")
        ov.fade(smooth(remap(t, pre - 0.4, pre + 0.3)) if pre - 0.4 < t < S["Canvas"] - 0.2 else 0.0)

        cpp_a = envelope(t, S["Canvas"] - 0.3, TL.end("build") + 0.2, 0.5, 0.7)
        py_a = envelope(t, TL.start("py") + 0.2, TL.end("py") - 0.1, 0.7, 0.6)
        x, y, size, lh = 70, 56, 19, 25
        cw = ov.text_width("0", size, "mono")
        if py_a > 0.003 or cpp_a > 0.003:
            w = max(len(l) for l in HELLO_PY + HELLO_CPP) * cw + 60
            ca = max(py_a, cpp_a)
            nlines = len(HELLO_CPP) if cpp_a >= py_a else len(HELLO_PY)
            card(ov, x, y, w, 66 + lh * nlines, ca, "hello.cpp" if cpp_a >= py_a else "hello.py  ·  the same program in Python")

        def step_of(j):
            for k, (_, ts, lines) in enumerate(STEPS):
                if j in lines:
                    return k, ts
            return None, None

        def reveal(j):
            k, ts = step_of(j)
            if k is None:
                return 1.0
            order = STEPS[k][2].index(j)
            return remap(t, ts + 0.15 + 0.45 * order, ts + 0.75 + 0.45 * order)

        def hl(j):
            k, ts = step_of(j)
            if k is None:
                return 0.0
            end = STEPS[k + 1][1] if k + 1 < len(STEPS) else TL.end("build") - 1.0
            return envelope(t, ts, end, 0.3, 0.4)

        code_block(ov, x + 30, y + 48, HELLO_CPP, "cpp", size, lh, cpp_a, hl=hl, reveal=reveal)
        code_block(ov, x + 30, y + 48, HELLO_PY, "py", size, lh, py_a,
                   reveal=lambda j: remap(t, TL.start("py") + 0.2 + 0.07 * j, TL.start("py") + 0.6 + 0.07 * j))

        # the canvas: a window outline around the picture, and the step pills above it
        wa = envelope(t, S["Canvas"], TL.end("py") - 0.2, 0.6, 0.8)
        if wa > 0.003:
            X0, Y0, X1 = 930, 150, 1860
            Y1 = Y0 + 34 + (X1 - X0) * 9 / 16        # a 1280 x 720 window, scaled
            # from the render loop on, the window shows what hello.py itself draws
            oa = wa * smooth(remap(t, S["Loop"] + 1.2, S["Loop"] + 2.0))
            if oa > 0.003:
                ang = max(0.0, t - (S["Loop"] + 1.2))       # rotation.y grows by 1 rad per second
                k = int((ang % (math.pi / 2)) / (math.pi / 2) * HELLO_FRAMES) % HELLO_FRAMES
                ov.panel(X0, Y0, X1 - X0, Y1 - Y0, radius=10, fill=0x000000, alpha=oa)
                ov.image(X0 + 2, Y0 + 36, X1 - X0 - 4, Y1 - Y0 - 38, self.hello[k], alpha=oa)
                la = oa * smooth(remap(t, S["Loop"] + 2.2, S["Loop"] + 2.8))
                ov.text(X0, Y1 + 24, "What running hello.cpp shows: its own window, not this film's studio",
                        size=20, color=C_OUT, alpha=la, anchor="lm")
            ov.outline(X0, Y0, X1 - X0, Y1 - Y0, 0x8aa0c0, 0.55 * wa, width=2.0, radius=10)
            ov.panel(X0 + 2, Y0 + 2, X1 - X0 - 4, 34, radius=8, fill=0x1a2233, alpha=0.9 * wa)
            ov.text(X0 + 20, Y0 + 19, "hello", size=17, color=TEXT, alpha=wa, anchor="lm")
            for k, c in enumerate((0x5ee27a, 0xf4d35e, 0xef476f)):
                ov.circle(X1 - 26 - 22 * k, Y0 + 19, 6, fill=c, alpha=0.8 * wa)
            ra = wa * smooth(remap(t, S["Canvas"] + 1.2, S["Canvas"] + 1.8)) * (1 - smooth(remap(t, S["Scene"], S["Scene"] + 0.8)))
            ov.text((X0 + X1) / 2, (Y0 + Y1) / 2 - 18, "GLRenderer", size=34, color=TEXT, alpha=ra, kind="semibold",
                    anchor="mm")
            ov.text((X0 + X1) / 2, (Y0 + Y1) / 2 + 24, "OpenGL 3.3: Windows, Linux, macOS, and the web", size=22,
                    color=DIM, alpha=ra, anchor="mm")
            # the camera: viewfinder corners
            ca = wa * envelope(t, S["Camera"] + 0.3, S["Mesh"] + 0.4, 0.5, 0.5)
            if ca > 0.003:
                x0, y0, x1, y1, L = X0 + 40, Y0 + 70, X1 - 40, Y1 - 36, 46
                for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x1, y1, -1, -1), (x0, y1, 1, -1)):
                    ov.line([(cx + dx * L, cy), (cx, cy), (cx, cy + dy * L)], C_WARM, 3.0, ca)
                ov.text(x0 + 16, y0 + 22, "PerspectiveCamera  ·  60° field of view", size=20, color=C_WARM,
                        alpha=ca, kind="semibold", anchor="lm")
            # the step pills
            px = X0
            wp = wa * envelope(t, S["Canvas"], S["Loop"] + 4.0, 0.6, 0.9)
            for name, ts in zip(PILLS, PILL_T):
                pa = wp * smooth(remap(t, ts, ts + 0.5))
                on = 0.0
                k = PILLS.index(name)
                t_next = PILL_T[k + 1] if k + 1 < len(PILL_T) else S["Loop"] + 4.0
                on = envelope(t, ts, t_next, 0.3, 0.3)
                wpx = ov.text_width(name, 19, "semibold") + 34
                ov.panel(px, 58, wpx, 36, radius=18, fill=0x223452 if on < 0.5 else 0x2b6f8f,
                         alpha=(0.55 + 0.4 * on) * max(pa, 0.25 * wp))
                ov.text(px + wpx / 2, 76, name, size=19, color=TEXT if pa > 0.5 else DIM,
                        alpha=max(pa, 0.35 * wp), kind="semibold", anchor="mm")
                px += wpx + 10
            # geometry / material labels at the box
            box_px = self.st.project(A + np.array([0, BOX_Y, 0]))
            ga = wa * envelope(t, S["Mesh"] + 0.6, S_MAT + 0.2, 0.4, 0.4)
            ov.callout(box_px[:2] + np.array([-60, 80]), (box_px[0] - 190, box_px[1] + 170),
                       "geometry", 0xffffff, ga, grow=ease_out(remap(t, S["Mesh"] + 0.6, S["Mesh"] + 1.2)))
            ma = wa * envelope(t, S_MAT + 0.2, S["Light"] + 0.2, 0.4, 0.4)
            ov.callout(box_px[:2] + np.array([-60, 80]), (box_px[0] - 190, box_px[1] + 170),
                       "material", C_ACCENT, ma, grow=ease_out(remap(t, S_MAT + 0.2, S_MAT + 0.8)))

    def share_hud(self, t):
        ov, st, p = self.ov, self.st, self.p
        t0 = TL.start("share")
        a = envelope(t, t0 + 0.4, TL.end("share") - 0.1, 0.6, 0.6)
        if a <= 0.003:
            return
        size, lh = 20, 30
        cw = ov.text_width("0", size, "mono")
        card(ov, 70, 64, max(len(l) for l in SHARE_CPP) * cw + 60, 64 + lh * len(SHARE_CPP), a, "SHARING, IN C++")

        def hl(j):
            for ts in (T_COLOUR, T_CLEAR, T_RESET):
                if j == SHARE_T.index(ts - 0.5):
                    return envelope(t, ts - 0.5, ts + 2.8, 0.3, 0.4)
            return 0.0
        code_block(ov, 100, 114, SHARE_CPP, "cpp", size, lh, a, hl=hl,
                   reveal=lambda j: remap(t, SHARE_T[j], SHARE_T[j] + 0.5))

        # who holds what: a line from each shared object to every cube that uses it
        n = sum(t >= share_pop(k) + 0.1 for k in range(N_SHARE))
        gone = t >= T_CLEAR + 0.2
        refs_g = 1 if gone else 1 + n
        refs_m = 0 if t >= T_RESET else 1 if gone else 1 + n
        mat_col = lesson.to_hex(C_SHARED) if t >= T_COLOUR + 0.2 else C_BOX
        labels = [("geometry", 1060, 0xdfe8ff), ("material", 1470, mat_col)]
        fa = a * smooth(remap(t, t0 + 1.0, t0 + 1.6))
        for name, lx, c in labels:
            ly = 300
            freed = name == "material" and t >= T_RESET
            wpx = ov.text_width(name, 24, "mono") + 40
            ov.panel(lx - wpx / 2, ly - 24, wpx, 48, radius=12, fill=0x16202f, alpha=0.9 * fa,
                     outline=c, outline_alpha=0.0 if freed else 0.6)
            ov.text(lx, ly, name, size=24, color=DIM if freed else c, alpha=fa * (0.45 if freed else 1.0),
                    kind="mono", anchor="mm")
            for k, m in enumerate(p["share_cubes"]):
                if not m.visible:
                    continue
                q = st.project(_wp(m) + np.array([0, 0.16, 0]))
                ov.line([(lx, ly + 24), (q[0], q[1])], c, 1.6, fa * 0.55 * min(1.0, m.scale.x))
        ov.readout(1488, 64, 368, "use_count()",
                   [("geometry", f"{refs_g}", None),
                    ("material", "freed" if t >= T_RESET else f"{refs_m}", C_OK if t >= T_RESET else None)],
                   alpha=a * smooth(remap(t, t0 + 1.0, t0 + 1.6)))

    def graph_hud(self, t):
        ov, st, p = self.ov, self.st, self.p
        a = envelope(t, TL.start("graph") + 0.4, TL.end("graph") - 0.1, 0.6, 0.6)
        if a <= 0.003:
            return
        g0 = TL.start("graph") + 0.8
        code_card(ov, t, TL.start("graph") + 0.4, TL.end("graph") - 0.1, 70, 64, GRAPH_CODE, "cpp",
                  "A CHILD OF THE BOX", size=20, lh=30)
        # the hierarchy, as an editor shows it
        rows = [(0, "Scene", 0.0, DIM), (1, "DirectionalLight", 0.0, DIM), (1, "box", 0.0, C_BOX),
                (2, "moon", g0, C_MOON), (3, "pebble", g0 + 1.0, C_PEBBLE), (2, "moon 2", g0 + 0.5, C_MOON)]
        x, y = 70, 290
        card(ov, x, y, 470, 70 + 52 * len(rows), a, "SCENE GRAPH")
        walking = envelope(t, cap0("g2") + 0.4, CAP["g2"][1] - 0.6, 0.5, 0.5)
        for k, (d, name, t_in, c) in enumerate(rows):
            ra = a * smooth(remap(t, t_in, t_in + 0.5)) if t_in else a
            yy = y + 78 + 52 * k
            xx = x + 40 + 44 * d
            if d > 0:
                ov.line([(xx - 28, yy - 40), (xx - 28, yy), (xx - 10, yy)], 0x55627a, 2.0, ra)
            ov.circle(xx + 4, yy, 7, fill=c, alpha=ra)
            glow = walking if name in ("box", "moon", "moon 2", "pebble") else 0.0
            ov.text(xx + 22, yy, name, size=24, color=TEXT, alpha=ra, kind="semibold" if glow > 0.3 else "regular",
                    anchor="lm")
        # parent links in the picture: box to each child
        la = a * envelope(t, g0 + 0.3, TL.end("graph") - 0.3, 0.5, 0.5)
        if la > 0.003:
            st.scene.update_matrix_world()
            for m, parent, c in ((p["moon"], p["box"], C_MOON), (p["moon2"], p["box"], C_MOON),
                                 (p["pebble"], p["moon"], C_PEBBLE)):
                if not m.visible:
                    continue
                a_ = st.project(_wp(parent))
                b_ = st.project(_wp(m))
                ov.dashed(a_[:2], b_[:2], c, 2.0, la * 0.9, dash=8, gap=6)

    def mat_hud(self, t):
        ov, st, p = self.ov, self.st, self.p
        a = envelope(t, TL.start("mat") + 0.8, TL.end("mat") - 0.1, 0.6, 0.6)
        if a <= 0.003:
            return
        names = ["Basic", "Lambert", "Phong", "Standard", "Physical"]
        for k, (m, name) in enumerate(zip(p["knots"], names)):
            q = st.project(_wp(m) + np.array([0, -0.36, 0]))
            ra = a * smooth(remap(t, TL.start("mat") + 1.0 + 0.4 * k, TL.start("mat") + 1.6 + 0.4 * k))
            ov.text(q[0], q[1], name, size=26, color=TEXT, alpha=ra, kind="semibold", anchor="ma")
        code_card(ov, t, TL.start("mat") + 1.2, cap0("m2") - 0.4, 70, 64, MAT_CODE, "cpp", "MATERIALS",
                  size=20, lh=30)
        m0 = cap0("m2")
        la = envelope(t, m0, TL.end("mat") - 0.2, 0.6, 0.6)
        if la > 0.003:
            q = st.project(_wp(p["bulb"]))
            ov.callout(q[:2], (q[0] + 90, q[1] - 80), "PointLight", 0x9fd8ff, la, size=22,
                       grow=ease_out(remap(t, m0, m0 + 0.6)))
            rows = [("DirectionalLight", "the sun: parallel rays"), ("PointLight", "a bulb: all directions"),
                    ("SpotLight", "a cone"), ("HemisphereLight", "soft sky and ground")]
            lx = 1290          # top right: the bulb's orbit passes the top left
            card(ov, lx, 64, 560, 66 + 42 * len(rows), la, "LIGHTS")
            for k, (n, d) in enumerate(rows):
                ra = la * smooth(remap(t, m0 + 0.2 + 0.35 * k, m0 + 0.8 + 0.35 * k))
                ov.text(lx + 26, 64 + 70 + 42 * k, n, size=22, color=0xffcb6b, alpha=ra, kind="mono", anchor="lm")
                ov.text(lx + 260, 64 + 70 + 42 * k, d, size=21, color=DIM, alpha=ra, anchor="lm")

    def robot_hud(self, t, i):
        ov, rec = self.ov, self.rec
        r0 = TL.start("robot") + 0.4
        a = envelope(t, r0, TL.end("robot") - 0.1, 0.6, 0.6)
        if a <= 0.003:
            return
        code_card(ov, t, r0, TL.end("robot") - 0.1, 70, 64, ROBOT_CODE[:3], "cpp", "LOAD A ROBOT", size=20, lh=30)
        tree_a = envelope(t, r0 + 0.6, cap0("r2") + 0.6, 0.6, 0.6)
        if tree_a > 0.003:
            rows = self.p["robot_rows"]
            x, y = 70, 240
            card(ov, x, y, 520, 90 + 34 * len(rows), tree_a, "fr3.urdf AS A TREE")
            for k, (d, name) in enumerate(rows):
                ra = tree_a * smooth(remap(t, r0 + 0.2 + 0.26 * k, r0 + 0.6 + 0.26 * k))
                xx, yy = x + 34 + 18 * d, y + 66 + 34 * k
                if d > 0 and name != "…":
                    ov.line([(xx - 12, yy - 20), (xx - 12, yy), (xx - 2, yy)], 0x55627a, 1.6, ra)
                is_link = "link" in name or name == "fr3"
                ov.text(xx + 4, yy, name, size=19, color=TEXT if is_link else DIM, alpha=ra, kind="mono", anchor="lm")
            ra = tree_a * smooth(remap(t, r0 + 3.2, r0 + 3.8))
            m = rec.meta
            ov.text(x + 26, y + 66 + 34 * len(rows) + 6, f"{m['robot_objects']} objects, {m['robot_meshes']} of them meshes",
                    size=21, color=C_WARM, alpha=ra, kind="semibold", anchor="lm")
        ik_a = envelope(t, rec.meta["ik_t"] - 0.6, TL.end("robot") - 0.1, 0.6, 0.6)
        if ik_a > 0.003:
            err = rec.ik_err[i]
            e_s = rec.ik_err[int(i // 30 * 30)]         # printed twice a second
            code_card(ov, t, rec.meta["ik_t"] - 0.6, TL.end("robot") - 0.1, 70, 250, ROBOT_CODE[3:], "cpp",
                      "INVERSE KINEMATICS", size=20, lh=30, out=[f"{e_s:.6g}"])     # std::cout's default: 6 significant digits
            ov.readout(1488, 64, 368, "ONE SOLVE PER FRAME",
                       [("hand error", f"{err * 1000:.2f} mm", C_OK if err < 1e-3 else None)], alpha=ik_a)

    def physics_hud(self, t, i):
        ov, rec = self.ov, self.rec
        a = envelope(t, TL.start("physics") + 0.3, TL.end("physics") - 0.1, 0.6, 0.6)
        if a <= 0.003:
            return
        j = int(i // 15 * 15)
        y0 = rec.box_pos[j, 0, 1] if j >= rec.spawn[0] else float("nan")
        code_card(ov, t, TL.start("physics") + 0.3, TL.end("physics") - 0.1, 70, 64, PHYS_CODE, "cpp", "PHYSX",
                  size=20, lh=30, out=[f"{y0:.6g}" if y0 == y0 else ""],
                  out_t=rec.spawn[0] / FPS)
        n = int((rec.spawn <= i).sum())
        ov.readout(1488, 64, 368, "RIGID BODIES",
                   [("boxes", f"{n}", None)], alpha=a)

    def sensor_hud(self, t, i):
        ov, rec = self.ov, self.rec
        a = envelope(t, TL.start("sensor") + 0.3, TL.end("sensor") - 0.1, 0.6, 0.6)
        if a <= 0.003:
            return
        keys = [k for k in rec.scans if k <= i]
        npts = len(rec.scans[max(keys)]) if keys else 0
        code_card(ov, t, TL.start("sensor") + 0.3, TL.end("sensor") - 0.1, 70, 64, SENSOR_CODE, "cpp",
                  "A DEPTH CAMERA", size=20, lh=30, out=[f"{npts}" if npts else ""])
        ov.readout(1488, 64, 368, "DEPTH CAMERA",
                   [("resolution", "160 × 120", None), ("points", f"{npts:,}", None)], alpha=a)
        # the colour scale: near is red, far is blue
        ba = a * smooth(remap(t, TL.start("sensor") + 1.4, TL.start("sensor") + 2.0))
        ov.colorbar(1512, 290, 320, 12, lambda u: lesson.turbo_hex(1.0 - u), alpha=ba)
        ov.text(1512, 312, "near", size=16, color=DIM, alpha=ba, anchor="la")
        ov.text(1832, 312, "far", size=16, color=DIM, alpha=ba, anchor="ra")

    def gallery(self, t):
        ov = self.ov
        a = envelope(t, TL.start("vulkan") + 0.2, TL.end("vulkan") + 0.1, 0.8, 0.7)
        if a <= 0.003:
            return
        ov.panel(-20, -20, ov.W + 40, ov.H + 40, radius=0, fill=0x070a10, alpha=0.82 * a)
        ov.text(150, 118, "MORE FROM THE EXAMPLES  ·  MOST USE THE VULKAN RENDERER", size=20, color=C_ACCENT, alpha=a,
                kind="semibold", tracking=3)
        w, h, gx, gy = 520, 292, 30, 76
        for k, ((f, label), tex) in enumerate(zip(GALLERY, self.textures)):
            if not self._cropped[k]:
                _crop(tex, os.path.join(REPO, "doc", "screenshots", f), w / h)
                self._cropped[k] = True
            r, c = divmod(k, 3)
            x, y = 150 + c * (w + gx), 160 + r * (h + gy)
            ia = a * smooth(remap(t, TL.start("vulkan") + 0.5 + 0.35 * k, TL.start("vulkan") + 1.2 + 0.35 * k))
            dy = 14 * (1 - ease_out(remap(t, TL.start("vulkan") + 0.5 + 0.35 * k, TL.start("vulkan") + 1.4 + 0.35 * k)))
            ov.image(x, y + dy, w, h, tex, alpha=ia)
            ov.outline(x, y + dy, w, h, 0x8aa0c0, 0.25 * ia, width=1.5)
            ov.text(x, y + dy + h + 14, label, size=21, color=TEXT, alpha=ia, anchor="la")

    def start_card(self, t):
        ov = self.ov
        a = envelope(t, TL.start("start") + 0.2, TL.end("start") + 0.2, 0.7, 0.6)
        if a <= 0.003:
            return
        ov.panel(-20, -20, ov.W + 40, ov.H + 40, radius=0, fill=0x070a10, alpha=0.55 * a)
        t0 = TL.start("start")
        code_card(ov, t, t0 + 0.4, TL.end("start") + 0.2, 150, 110, START_CMAKE, "cmake", "C++  ·  CMakeLists.txt",
                  size=21, lh=32, stagger=0.25)
        code_card(ov, t, t0 + 4.0, TL.end("start") + 0.2, 150, 590, START_PY, "sh", "PYTHON", size=22, lh=34,
                  min_w=420)
        la = a * smooth(remap(t, cap0("t2") - 0.2, cap0("t2") + 0.5))
        rows = [("doc/getting_started.md", "the ideas, one at a time"),
                ("examples/", "C++ programs, small and runnable"),
                ("python/examples/", "the same in Python")]
        card(ov, 1080, 110, 690, 70 + 48 * len(rows), la, "THEN")
        for k, (pth, d) in enumerate(rows):
            ra = la * smooth(remap(t, cap0("t2") + 0.4 * k, cap0("t2") + 0.6 + 0.4 * k))
            ov.text(1110, 110 + 78 + 48 * k, pth, size=22, color=C_ACCENT, alpha=ra, kind="mono", anchor="lm")
            ov.text(1110 + 330, 110 + 78 + 48 * k, d, size=21, color=DIM, alpha=ra, anchor="lm")


def _wp(o):
    """World position of an object as a numpy vector."""
    v = o.get_world_position()
    return np.array([v.x, v.y, v.z])


def _crop(tex, path, aspect):
    """Centre-crop a texture to `aspect` (w / h) through its uv repeat/offset."""
    import struct
    with open(path, "rb") as f:
        head = f.read(24)
    w, h = struct.unpack(">II", head[16:24])
    ai = w / h
    if ai > aspect:
        tex.repeat.set(aspect / ai, 1.0)
        tex.offset.set((1 - aspect / ai) / 2, 0.0)
    else:
        tex.repeat.set(1.0, ai / aspect)
        tex.offset.set(0.0, (1 - ai / aspect) / 2)
    tex.update_matrix()
    tex.needs_update()


# ── the stage ─────────────────────────────────────────────────────────────────
def build(width, height):
    st = Stage(width, height, renderer="gl", fog=(5.5, 14.0), shadow_extent=2.2, far=40.0)
    parts = {}

    # set A: the hello box (1 m, as BoxGeometry() makes it), its wireframe, the moons
    ga = tp.Group()
    st.scene.add(ga)
    parts["A"] = ga
    bm = standard(C_BOX, roughness=0.42, metalness=0.05)
    box = tp.Mesh(tp.BoxGeometry(), bm)
    box.cast_shadow = True
    box.position.set(A[0], BOX_Y, A[2])
    ga.add(box)
    wm = tp.MeshBasicMaterial()
    wm.wireframe = True
    wm.color = 0xffffff
    wm.transparent = True
    wm.depth_write = False
    wire = tp.Mesh(tp.BoxGeometry(1.002, 1.002, 1.002), wm)
    box.add(wire)
    parts["box"], parts["wire"] = box, wire
    moon = tp.Mesh(tp.SphereGeometry(0.2, 48, 24), standard(C_MOON, roughness=0.5))
    moon.position.set(1.2, 0, 0)
    moon.cast_shadow = True
    box.add(moon)
    moon2 = tp.Mesh(tp.SphereGeometry(0.14, 48, 24), standard(C_MOON, roughness=0.5))
    moon2.position.set(-0.85, 0.2, 0.75)
    moon2.cast_shadow = True
    box.add(moon2)
    pebble = tp.Mesh(tp.SphereGeometry(0.07, 32, 16), standard(C_PEBBLE, roughness=0.4))
    pebble.position.set(0.38, 0, 0)
    pebble.cast_shadow = True
    moon.add(pebble)
    parts["moon"], parts["moon2"], parts["pebble"] = moon, moon2, pebble
    arrow = Arrow3D(0xfff1c4, radius=0.02, parent=st.scene, emissive=1.2)
    arrow.set_opacity(0.0)
    parts["sun_arrow"] = arrow

    # set D: eight cubes that really do share one geometry and one material
    gd = tp.Group()
    st.scene.add(gd)
    parts["D"] = gd
    geo = tp.BoxGeometry(0.3, 0.3, 0.3)
    shared = standard(C_BOX, roughness=0.4, metalness=0.05)
    cubes = []
    for k in range(N_SHARE):
        c = tp.Mesh(geo, shared)
        c.position.set(D[0] + (k - (N_SHARE - 1) / 2) * 0.4, 0.45, D[2])
        c.cast_shadow = True
        gd.add(c)
        cubes.append(c)
    parts["share_cubes"], parts["share_mat"] = cubes, shared

    # set B: one torus knot per material
    gb = tp.Group()
    st.scene.add(gb)
    parts["B"] = gb
    colour = 0xe07a5f
    mats = []
    m = tp.MeshBasicMaterial()
    m.color = colour
    mats.append(m)
    m = tp.MeshLambertMaterial()
    m.color = colour
    mats.append(m)
    m = tp.MeshPhongMaterial()
    m.color = colour
    m.shininess = 90
    m.specular = 0x666666
    mats.append(m)
    mats.append(standard(colour, roughness=0.35, metalness=0.1))
    m = tp.MeshPhysicalMaterial()
    m.color = colour
    m.roughness = 0.55
    m.metalness = 0.3
    m.clearcoat = 1.0
    m.clearcoat_roughness = 0.05
    mats.append(m)
    knots = []
    for k, mat in enumerate(mats):
        kn = tp.Mesh(tp.TorusKnotGeometry(0.15, 0.05, 160, 24), mat)
        kn.position.set(B[0] + (k - 2) * 0.62, 0.62, B[2])
        kn.cast_shadow = True
        gb.add(kn)
        knots.append(kn)
    parts["knots"] = knots
    bulb = tp.Mesh(tp.SphereGeometry(0.035, 24, 12), standard(0x9fd8ff, emissive=0x9fd8ff, emissive_intensity=3.0))
    gb.add(bulb)
    pl = tp.PointLight(tp.Color(0x9fd8ff), 0.0, 4.0, 2.0)
    bulb.add(pl)
    parts["bulb"], parts["bulb_light"] = bulb, pl

    # set C: the robot, its target, the PhysX boxes and the depth camera
    robot = tp.URDFLoader().load(data_file("urdf", "franka", "fr3.urdf"))
    robot.show_colliders(False)
    robot.traverse(lambda o: setattr(o, "cast_shadow", True))
    st.zup.add(robot)
    robot.set_end_effector("fr3_hand_tcp")
    robot.set_joint_values(full(Q_READY))
    robot_meshes = []

    def collect(o, k):
        name = o.name or ""
        mm = re.match(r"fr3_(link(\d)|hand)$", name)
        if mm:
            k = 8 if mm.group(1) == "hand" else int(mm.group(2))
        if type(o).__name__ == "Mesh" and o.visible:
            robot_meshes.append((o, k))
        for c in o.children:
            collect(c, k)
    collect(robot, 0)
    parts["robot_meshes"] = robot_meshes
    counts = {"all": 0, "mesh": 0}

    def count(o):
        counts["all"] += 1
        counts["mesh"] += type(o).__name__ == "Mesh"
    robot.traverse(count)
    names = []

    def chain(o, d):
        if re.match(r"fr3_(link\d|joint\d|hand)$", o.name or "") or o is robot:
            names.append((d, o.name))
        for c in o.children:
            chain(c, d + 1)
    chain(robot, 0)
    # the chain as the tree shows it: robot, links and joints to link 2, then the hand
    keep = [(0, "fr3")] + [(k + 1, n) for k, (_, n) in enumerate(names[1:6])] + [(6, "…")]
    hand = [n for _, n in names if n == "fr3_hand"]
    rows = keep + ([(7, hand[0])] if hand else [])
    parts["robot_rows"] = rows
    target = Marker3D(C_WARM, parent=st.zup)
    parts["target"] = target

    palette = [0xe54b4b, 0x3ca0e5, 0x49c66a, 0xe5c04b, 0xa64be5, 0xe5814b, 0x4cc9f0]
    rng = np.random.default_rng(3)
    boxes = []
    for b in range(N_BOXES):
        s = rng.uniform(0.06, 0.11)
        mb = tp.Mesh(tp.BoxGeometry(s, s, s), standard(palette[b % len(palette)], roughness=0.6))
        mb.cast_shadow = True
        mb.receive_shadow = True
        mb.visible = False
        st.scene.add(mb)
        boxes.append(mb)
    parts["boxes"] = boxes

    post = tp.Group()
    st.scene.add(post)
    pole = tp.Mesh(tp.CylinderGeometry(0.025, 0.025, SENSOR_AT[1] - 0.06, 20), standard(0x3a4150, 0.5, 0.6))
    pole.position.set(POST[0], (SENSOR_AT[1] - 0.06) / 2, POST[2])
    pole.cast_shadow = True
    post.add(pole)
    head = tp.Mesh(tp.BoxGeometry(0.14, 0.06, 0.06), standard(0x1a1d24, 0.4, 0.3))
    head.position.set(*SENSOR_AT)
    head.look_at(*SENSOR_LOOK)
    head.cast_shadow = True
    post.add(head)
    parts["post"] = post
    sensor = tp.DepthSensor(fov_y=50, width=160, height=120, near=0.1, far=6.0)
    sensor.position.set(*SENSOR_AT)
    st.scene.add(sensor)
    sensor.look_at(*SENSOR_LOOK)
    sensor.update_matrix_world()
    parts["cloud"] = Cloud(160 * 120, size=0.016, parent=st.scene)
    parts["frustum"] = Segments(8, parent=st.scene, additive=True)
    parts["robot_objects"], parts["robot_meshes_n"] = counts["all"], counts["mesh"]
    return st, robot, parts, sensor


def setup(width, height):
    st, robot, parts, sensor = build(width, height)
    t0 = time.time()
    rec = choreograph(st, robot, parts["boxes"])
    rec.meta["robot_objects"] = parts["robot_objects"]
    rec.meta["robot_meshes"] = parts["robot_meshes_n"]
    print(f"[urdf] fr3: {parts['robot_objects']} objects, {parts['robot_meshes_n']} meshes")
    rec.meta["scan_pts"] = 0          # counted by scan_pass below
    ov = Hud(1920, 1080, math_cache=os.path.join(_HERE, "threepp_intro.math.json"))
    painter = Painter(st, rec, robot, parts, sensor, ov)
    scan_pass(st, [parts["A"], parts["B"], parts["post"], parts["target"].group], rec, sensor, painter)
    painter.captions = painter.make_captions()     # now with the scan numbers
    print(f"[choreo] {rec.n} frames in {time.time() - t0:.1f}s")

    def render(t):
        i = min(int(round(t * FPS)), rec.n - 1)
        painter.set3d(t, i)
        ov.begin()
        painter.draw2d(t, i)
        ov.fade(fade_in_out(t, TL.duration))
        ov.end()
        return st.frame(t, hud=ov)
    return render, painter.captions


if __name__ == "__main__":
    run("threepp_intro", TL.duration, setup, fps=FPS, speech=SPEECH)
