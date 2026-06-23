"""Smoke test for the PhysX material API. ONE world (PhysX foundation is a per-process singleton, so
churning many worlds is fragile): friction lanes (vary mu -> slide distance), restitution lanes (vary
rest -> bounce height), runtime mutability (the domain-randomization hot path), and add_link(material=)
with a DISTINCT material per articulation link in the same scene (per-env friction DR feasibility)."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import threepp as tp

MUS = (0.1, 0.5, 1.0)
RESTS = (0.0, 0.4, 0.8)


def main():
    if not tp.HAS_PHYSX:
        print("need PhysX"); return
    w = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=1.0 / 120, max_substeps=8)

    # --- friction lanes: each box on its OWN ground patch with the lane's material (so the contact
    #     coefficient is exactly that mu, not averaged against a shared default), launched at 3 m/s.
    fboxes = []
    for j, mu in enumerate(MUS):
        y = j * 4.0
        mat = w.create_material(static_friction=mu, dynamic_friction=mu, restitution=0.0)
        patch = tp.Mesh(tp.BoxGeometry(40, 3.5, 1.0), tp.MeshStandardMaterial()); patch.position.set(5, y, -0.5)
        w.add_static(patch, material=mat)
        box = tp.Mesh(tp.BoxGeometry(0.2, 0.2, 0.2), tp.MeshStandardMaterial()); box.position.set(0, y, 0.101)
        body = w.add(box, density=500, material=mat)
        body.set_linear_velocity(tp.Vector3(3.0, 0.0, 0.0))
        fboxes.append((mu, box))

    # --- restitution lanes: drop a box from 1.0 m onto its patch; track peak rebound.
    rballs = []
    for j, r in enumerate(RESTS):
        y = -4.0 * (j + 1)
        mat = w.create_material(0.6, 0.6, r)
        patch = tp.Mesh(tp.BoxGeometry(3, 3, 1.0), tp.MeshStandardMaterial()); patch.position.set(0, y, -0.5)
        w.add_static(patch, material=mat)
        ball = tp.Mesh(tp.BoxGeometry(0.2, 0.2, 0.2), tp.MeshStandardMaterial()); ball.position.set(0, y, 1.0)
        w.add(ball, density=500, material=mat)
        rballs.append([r, ball, False, 0.0])    # r, mesh, hit?, peak-after-hit

    # --- a proper free-base 2-link articulation, each link a DISTINCT material (per-env friction DR
    #     feasibility): root + one revolute child, dropped onto its own patch and stepped.
    apatch = tp.Mesh(tp.BoxGeometry(3, 3, 1.0), tp.MeshStandardMaterial()); apatch.position.set(0, 20, -0.5)
    w.add_static(apatch, material=w.create_material(1.0, 0.9, 0.0))
    mat_a = w.create_material(static_friction=0.9, dynamic_friction=0.8, restitution=0.0)
    mat_b = w.create_material(static_friction=0.3, dynamic_friction=0.3, restitution=0.0)
    art = w.create_articulation(fixed_base=False)
    root = tp.Mesh(tp.BoxGeometry(0.3, 0.3, 0.1), tp.MeshStandardMaterial()); root.position.set(0, 20, 0.6)
    rlink = art.add_link(root, parent=None, density=300, material=mat_a)
    child = tp.Mesh(tp.BoxGeometry(0.3, 0.1, 0.1), tp.MeshStandardMaterial()); child.position.set(0.3, 20, 0.6)
    art.add_link(child, parent=rlink, density=300, axis=(0, 1, 0), anchor=(0.15, 20, 0.6),
                 lower=-1.0, upper=1.0, stiffness=8.0, damping=1.0, material=mat_b)
    art.finalize()
    built = 2

    for _ in range(600):                       # 5 s
        w.step(1.0 / 120)
        for rec in rballs:
            z = rec[1].position.z
            if z < 0.13:
                rec[2] = True
            elif rec[2]:
                rec[3] = max(rec[3], z)

    print("FRICTION (box launched at 3 m/s, slide distance):")
    slid = {}
    for mu, box in fboxes:
        slid[mu] = box.position.x
        print(f"  mu={mu:.1f} -> slid {slid[mu]:.2f} m")
    print("RESTITUTION (box dropped from 1.0 m, peak rebound):")
    bounce = {}
    for r, _, _, peak in rballs:
        bounce[r] = peak
        print(f"  restitution={r:.1f} -> rebounded to {peak:.3f} m")

    m = w.create_material(0.5, 0.5, 0.2, friction_combine="min", restitution_combine="min")
    before = (round(m.static_friction, 2), round(m.dynamic_friction, 2), round(m.restitution, 2))
    m.set(1.0, 0.9, 0.0); m.dynamic_friction = 0.75
    after = (round(m.static_friction, 2), round(m.dynamic_friction, 2), round(m.restitution, 2))
    ok_mut = after == (1.0, 0.75, 0.0)
    print(f"MUTABILITY: {before} -> set(1.0,0.9,0.0)+dynamic=0.75 -> {after}")
    print(f"ARTICULATION add_link(material=) 2-link, distinct materials: built {built} links, stepped OK")

    ok_fric = slid[0.1] > slid[1.0] * 2
    ok_rest = bounce[0.8] > bounce[0.0] + 0.05
    print(f"\nVERDICT: friction {'OK' if ok_fric else 'FAIL'}, restitution {'OK' if ok_rest else 'FAIL'}, "
          f"mutability {'OK' if ok_mut else 'FAIL'}, articulation {'OK' if built == 2 else 'FAIL'}")
    print("MATERIAL API SMOKE:", "PASS" if (ok_fric and ok_rest and ok_mut and built == 2) else "FAIL")


if __name__ == "__main__":
    main()
