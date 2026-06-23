"""A cart + single pole (the classic inverted pendulum / cart-pole), built on threepp's
PhysX articulation binding.

Topology: fixed base -> cart (prismatic slider along X, force-controlled, no drive) ->
pole (free, frictionless revolute about Z). Only the cart is actuated; the policy slides
it to swing the passive pole up and balance it. Build configuration is upright (pole at
+Y), so joint angle 0 == balanced, pi == hanging straight down.

DOF order: [cart_x, pole_angle].
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import threepp as tp


def _mat(color):
    m = tp.MeshStandardMaterial()
    m.color = color
    m.roughness = 0.5
    return m


class CartPole:
    def __init__(self, world, x0=0.0, rail_y=1.2, cart=(0.32, 0.18, 0.22),
                 length=0.5, rail=2.2, density=240.0):
        self.world = world
        self.rail_y = rail_y
        self.l = length
        cart_top = rail_y + cart[1] * 0.5
        self.cart_top = cart_top

        art = world.create_articulation(fixed_base=True, solver_position_iterations=8,
                                        disable_self_collision=True)
        # Fixed base: the rail mount (root link, pinned to the world).
        base_mesh = tp.Mesh(tp.BoxGeometry(0.12, 0.12, 0.5), _mat(0x333333))
        base_mesh.position.set(x0, rail_y, 0.0)
        base = art.add_link(base_mesh, density=density)

        # Cart: prismatic slider along world X, no drive (force-controlled), rail-limited.
        cart_mesh = tp.Mesh(tp.BoxGeometry(*cart), _mat(0x2a6fae))
        cart_mesh.position.set(x0, rail_y, 0.0)
        self.cart = art.add_link(cart_mesh, parent=base, density=density,
                                 axis=(1, 0, 0), anchor=(x0, rail_y, 0.0),
                                 lower=-rail, upper=rail, joint_type="prismatic")

        # Pole: free (frictionless) revolute about Z, hinged at the cart top, pointing up.
        pole_mesh = tp.Mesh(tp.BoxGeometry(0.06, length, 0.06), _mat(0xc24d2c))
        pole_mesh.position.set(x0, cart_top + length * 0.5, 0.0)
        self.pole = art.add_link(pole_mesh, parent=self.cart, density=density,
                                 axis=(0, 0, 1), anchor=(x0, cart_top, 0.0))

        art.finalize()
        self.art = art
        self.meshes = [base_mesh, cart_mesh, pole_mesh]

    def add_to_scene(self, scene):
        for m in self.meshes:
            scene.add(m)
