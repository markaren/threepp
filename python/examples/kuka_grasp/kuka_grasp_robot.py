"""kuka_grasp_robot.py — build the KUKA+gripper arm, the cube, and the table for the grasp env.

The arm is the KUKA LBR iiwa 14 R820 loaded from a **combined URDF** that this module generates at
runtime: the stock `lbr_iiwa_14_r820.urdf` with two prismatic finger links bolted onto `link_7`
(the flange). We must use a combined URDF because `world.load_articulation` finalizes the
articulation internally — you cannot `add_link` a gripper after loading (see SPIKE_FINDINGS.md).

The cube is a free-base single-link articulation (0 joint-DOF, read via its root pose) with a
high-friction "max"-combine material so a parallel-jaw squeeze holds it by friction.

Run `python kuka_grasp_robot.py --inspect` to verify the combined URDF loads as 9 DOF, print the
joint/link order, the finger gap at open/closed, and the flange-tip world pose at DEFAULT_Q
(used to tune DEFAULT_Q + the gripper geometry).
"""
import ctypes
import os
import re
import struct
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_DIR = os.path.dirname(os.path.dirname(_HERE))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

import threepp as tp  # noqa: E402

import kuka_grasp_contract as C  # noqa: E402

# --------------------------------------------------------------------------- #
#  Locate the stock KUKA URDF + generate the combined arm+gripper URDF
# --------------------------------------------------------------------------- #
_REPO = os.path.dirname(_PYTHON_DIR)  # C:\dev\threepp
_URDF_CANDIDATES = [
    os.path.join(_REPO, "build", "_deps", "threepp_data-src", "urdf", "lbr_iiwa_14_r820.urdf"),
    os.path.join(_REPO, "cmake-build-relwithdebinfo", "_deps", "threepp_data-src", "urdf", "lbr_iiwa_14_r820.urdf"),
    os.path.join(_REPO, "cmake-build-release", "_deps", "threepp_data-src", "urdf", "lbr_iiwa_14_r820.urdf"),
    os.path.join(_REPO, "cmake-build-debug", "_deps", "threepp_data-src", "urdf", "lbr_iiwa_14_r820.urdf"),
]


def find_kuka_urdf():
    for p in _URDF_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "lbr_iiwa_14_r820.urdf not found — build threepp once so the data dep is fetched. "
        f"Looked in: {_URDF_CANDIDATES}")


def _finger_xml():
    bx, by, bz = C.FINGER_BOX
    box = f'<box size="{bx} {by} {bz}"/>'
    mx, mz = C.FINGER_MOUNT_X, C.FINGER_MOUNT_Z
    lo, hi = C.FINGER_LOWER, C.FINGER_UPPER
    out = []
    for name, sx, axis in (("finger_left", +1.0, "1 0 0"), ("finger_right", -1.0, "-1 0 0")):
        out.append(f'''  <link name="{name}">
    <visual><origin rpy="0 0 0" xyz="0 0 0"/><geometry>{box}</geometry></visual>
    <collision><origin rpy="0 0 0" xyz="0 0 0"/><geometry>{box}</geometry></collision>
  </link>
  <joint name="{name}_joint" type="prismatic">
    <origin rpy="0 0 0" xyz="{sx * mx} 0 {mz}"/>
    <parent link="link_7"/>
    <child link="{name}"/>
    <axis xyz="{axis}"/>
    <limit effort="200" lower="{lo}" upper="{hi}" velocity="1.0"/>
  </joint>''')
    return "\n".join(out)


def _stl_bbox(path):
    """Axis-aligned bbox (center, size) of a binary STL — the loader's own mesh-collision
    approximation, computed once here so the per-env URDF load needs no file I/O."""
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack_from("<I", data, 80)[0]
    verts = np.frombuffer(data[84:84 + n * 50], dtype=np.uint8).reshape(n, 50)[:, 12:48]
    verts = verts.copy().view("<f4").reshape(n * 3, 3)
    mn, mx = verts.min(0), verts.max(0)
    center = (mn + mx) * 0.5
    size = np.maximum(mx - mn, 1e-3)
    return center, size


def _bake_collisions(text):
    """Replace every <collision> mesh with the equivalent primitive <box> (its bbox). This is what
    the loader does internally anyway, but baking it here means each of the K per-env load_articulation
    calls reads ZERO STL files — the difference between a seconds build and a many-minutes build at
    large K. Visual <mesh> tags are left untouched (only loaded when render_visuals=True)."""
    pat = re.compile(r'<collision>\s*<origin[^>]*/>\s*<geometry>\s*<mesh filename="([^"]+)"\s*/>\s*'
                     r'</geometry>\s*</collision>', re.DOTALL)

    def repl(m):
        c, s = _stl_bbox(m.group(1))
        return (f'<collision><origin rpy="0 0 0" xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}"/>'
                f'<geometry><box size="{s[0]:.6f} {s[1]:.6f} {s[2]:.6f}"/></geometry></collision>')

    return pat.sub(repl, text)


def generate_combined_urdf(dest=None):
    """Read the stock KUKA URDF, rewrite its mesh paths to absolute (so the combined URDF resolves
    from anywhere), BAKE arm collision meshes into primitive boxes (so per-env loads do no file I/O),
    inject the two prismatic finger links onto link_7, and write the result. Returns the path."""
    src = find_kuka_urdf()
    src_dir = os.path.dirname(src).replace("\\", "/")
    with open(src, "r", encoding="utf-8") as f:
        text = f.read()
    # rewrite relative mesh filenames -> absolute (forward slashes) so location is irrelevant
    text = text.replace('filename="lbr_iiwa_14_r820/', f'filename="{src_dir}/lbr_iiwa_14_r820/')
    # bake collision meshes -> boxes (no per-env STL reads); keep visuals as meshes
    text = _bake_collisions(text)
    # inject the gripper subtree just before </robot>
    text = text.replace("</robot>", _finger_xml() + "\n</robot>")
    if dest is None:
        dest = os.path.join(_HERE, "kuka_iiwa_gripper.urdf")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    return dest


# --------------------------------------------------------------------------- #
#  Builders (called per-env by KukaGraspSim)
# --------------------------------------------------------------------------- #
def build_arm(world, env_idx, spacing, urdf_path, render_visuals=False):
    """Load one KUKA+gripper articulation for env `env_idx` (X-spaced). Returns (art, meshes, joints)."""
    ox = env_idx * spacing
    art, meshes, joints = world.load_articulation(
        urdf_path,
        fixed_base=True,
        base_position=(ox, 0.0, 0.0),
        default_density=1200.0,
        stiffness=C.PD_STIFFNESS,
        damping=C.PD_DAMPING,
        max_force=C.PD_MAX_FORCE,
        self_collision=False,          # fingers must not self-collide with the arm or each other
        solver_position_iterations=16,  # firmer contact for the grasp
        render_visuals=render_visuals,
    )
    return art, meshes, joints


def make_cube_material(world):
    """High-friction, no-bounce, MAX-combine material for the cube — wins the contact friction
    against the fingers' default material so a squeeze holds by friction."""
    return world.create_material(
        static_friction=C.CUBE_FRICTION, dynamic_friction=C.CUBE_FRICTION,
        restitution=0.0, friction_combine="max", restitution_combine="average")


def build_cube(world, env_idx, spacing, material, render_visuals=True):
    """One free-base single-link cube (0 joint-DOF; pose read via the cube batch's root)."""
    ox = env_idx * spacing
    mesh = tp.Mesh(tp.BoxGeometry(C.CUBE, C.CUBE, C.CUBE), tp.MeshStandardMaterial())
    mesh.material.color = 0xff8c42
    mesh.position.set(ox + C.TABLE_CX, C.TABLE_CY, C.CUBE_REST_Z)
    art = world.create_articulation(fixed_base=False, solver_position_iterations=16,
                                    disable_self_collision=True)
    art.add_link(mesh, density=C.CUBE_DENSITY, material=material)
    art.finalize()
    return art, mesh


def add_ground_and_tables(world, num_envs, spacing):
    """Static ground + one table slab per env (X-spaced). Built BEFORE any articulation."""
    ground = tp.Mesh(tp.BoxGeometry(200.0, 200.0, 0.1), tp.MeshStandardMaterial())
    ground.material.color = 0x40454d
    ground.position.set(0.0, 0.0, -0.05)
    world.add_static(ground)
    tables = []
    tx, ty = C.TABLE_HALF
    for i in range(num_envs):
        ox = i * spacing
        t = tp.Mesh(tp.BoxGeometry(tx * 2, ty * 2, C.TABLE_THICK), tp.MeshStandardMaterial())
        t.material.color = 0x3b4252
        t.position.set(ox + C.TABLE_CX, C.TABLE_CY, C.TABLE_TOP_Z - C.TABLE_THICK / 2)
        world.add_static(t)
        tables.append(t)
    return ground, tables


# --------------------------------------------------------------------------- #
#  --inspect : verify the combined URDF + tune geometry / DEFAULT_Q
# --------------------------------------------------------------------------- #
def _torch_cuda_context():
    drv = ctypes.CDLL("nvcuda.dll")
    drv.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    drv.cuCtxGetCurrent.restype = ctypes.c_int
    ctx = ctypes.c_void_p()
    if drv.cuCtxGetCurrent(ctypes.byref(ctx)) != 0 or not ctx.value:
        raise RuntimeError("could not read torch CUDA context")
    return int(ctx.value)


def _inspect():
    import torch
    np.set_printoptions(precision=3, suppress=True)
    assert tp.HAS_PHYSX, "needs a PhysX-enabled threepp build"

    urdf = generate_combined_urdf()
    print(f"[inspect] combined URDF: {urdf}")

    torch.zeros(1, device="cuda")
    torch.randn(32, 32, device="cuda").sum().item()
    torch.cuda.synchronize()
    ctx = _torch_cuda_context()
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=C.PHYS_DT,
                          max_substeps=1, direct_gpu=True, cuda_context=ctx)
    add_ground_and_tables(world, 1, 3.0)
    mat = make_cube_material(world)
    arm, _, joints = build_arm(world, 0, 3.0, urdf, render_visuals=False)
    cube, _ = build_cube(world, 0, 3.0, mat, render_visuals=False)

    arm_batch = tp.PhysxGpuBatch(world, [arm])
    cube_batch = tp.PhysxGpuBatch(world, [cube])
    dev = torch.device("cuda")
    arm_dof = arm.dof_order().shape[0]
    print(f"[inspect] arm DOF = {arm_dof}  (expect {C.N_DOF})")
    print(f"[inspect] joint order = {joints}")
    print(f"[inspect] max_links = {arm_batch.max_links}")

    perm = torch.from_numpy(arm.dof_order().astype(np.int64)).to(dev)
    md = arm_batch.max_dofs
    link_pose = torch.zeros(1, arm_batch.max_links * 7, device=dev)

    def drive_to(arm_q9, settle=120):
        tgt = torch.zeros(1, md, device=dev)
        canon = torch.zeros(1, arm_dof, device=dev)
        canon[0, :len(arm_q9)] = torch.tensor(arm_q9, device=dev)
        tgt[:, perm] = canon
        torch.cuda.synchronize()
        arm_batch.write_joint_target_pos(tgt)
        for _ in range(settle):
            arm_batch.step(C.PHYS_DT)
        arm_batch.read_link_pose(link_pose)
        return link_pose.view(1, arm_batch.max_links, 7)[0].cpu().numpy()

    def report(q7, grip=C.GRIP_OPEN, settle=120):
        lp = drive_to(list(q7) + [grip, grip], settle=settle)
        # link order: 0=base_link, 1..7 = link_1..link_7, 8=finger_left, 9=finger_right
        flange, fl, fr = lp[7, 4:7], lp[8, 4:7], lp[9, 4:7]
        gap = float(np.linalg.norm(fl - fr))
        R = _quat_to_R(lp[7, 0:4])
        tip = flange + R[:, 2] * C.TIP_Z          # tip = grasp point, TIP_Z down the approach axis
        return flange, tip, gap, R[:, 2]

    # Target: tip hovering ~0.06 m above the cube top, over the table centre, tool pointing down.
    target = np.array([C.TABLE_CX, C.TABLE_CY, C.CUBE_REST_Z + C.CUBE_HALF + 0.06], np.float32)
    print(f"[inspect] target tip = {target}  (table top {C.TABLE_TOP_Z}, cube top {C.CUBE_REST_Z + C.CUBE_HALF:.3f})")

    if "--q" in sys.argv:
        q7 = [float(v) for v in sys.argv[sys.argv.index("--q") + 1].split(",")]
        flange, tip, gap, approach = report(q7)
        print(f"  cli  flange={flange}  tip={tip}  err={np.linalg.norm(tip - target):.3f}  "
              f"down={-approach[2]:.3f}  gap={gap:.3f}")
        return

    # Random search over shoulder/elbow/wrist (a2,a4,a6) for a posture whose tip is at `target`
    # with the tool pointing down. a1=a3=a5=a7=0 keeps the arm in the x-z plane facing the table.
    rng = np.random.default_rng(0)
    best = None
    for _ in range(260):
        a2 = float(rng.uniform(0.1, 0.9))
        a4 = float(rng.uniform(-1.5, -0.5))
        a6 = float(rng.uniform(1.2, 2.0))
        q7 = [0.0, a2, 0.0, a4, 0.0, a6, 0.0]
        flange, tip, gap, approach = report(q7, settle=60)
        down = float(-approach[2])
        if down < 0.95:                      # tool within ~18° of straight down
            continue
        err = float(np.linalg.norm(tip - target)) + 1.0 * (1.0 - down)
        if best is None or err < best[0]:
            best = (err, q7, tip, down)
            print(f"  better err={err:.3f}  tip={tip}  down={down:.3f}  q={[round(v, 3) for v in q7]}")
    print(f"[inspect] BEST q={[round(v, 3) for v in best[1]]}  tip={best[2]}  down={best[3]:.3f}  err={best[0]:.3f}")


def _quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], np.float32)


if __name__ == "__main__":
    if "--inspect" in sys.argv:
        _inspect()
    else:
        print("combined URDF written to:", generate_combined_urdf())
        print("run with --inspect to verify DOF / geometry / posture")
