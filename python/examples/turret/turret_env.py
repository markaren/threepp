"""Vision-driven point-defense TURRET — learns from pixels to track and shoot down incoming colliders.

A fixed turret with an onboard camera (the camera IS the aim) must swivel so an incoming projectile is
centered in frame, then FIRE to destroy it. Colliders are hurled in from a dome with randomized shape,
size, speed and trajectory. This is visual servoing (center-the-target) + a trigger — the most reliably
trainable pixel-RL task class — dressed as a missile-defense turret.

  observation: a stack of the last N small GL camera frames -> [K, 3*N, H, W] uint8 (motion is visible
               across frames so the policy can lead moving targets). The threat is a bright color blob.
  action (3):  (d_yaw, d_pitch, fire). fire when action[2] > 0 and the gun is off cooldown.
  reward:      dense — track the nearest threat toward center (analytic angle, NOT pixels) + big hit bonus
               on a centered shot, - small shot/miss cost, - penalty if a collider reaches the turret.

K turrets live in one shared GL scene in regions spaced far apart (far-plane + frustum culling keep each
camera to its own region); the reward/dynamics are vectorized torch on the GPU, only the per-frame render
touches the CPU. Single source of truth: CONFIG (persisted to the checkpoint).
"""
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp

# ---- single source of truth -------------------------------------------------
CONTROL_HZ = 30
DT = 1.0 / CONTROL_HZ
H, W = 48, 64                 # camera frame (small = fast); flat ~7000 renders/s
FRAME_STACK = 3              # last N frames -> motion is observable (lead moving targets)
N_PROJ = 2                  # simultaneous incoming colliders per turret
EPISODE_S = 8.0
SPACING = 200.0             # metres between turret regions (>> far plane -> isolated views)
LAUNCH_LO, LAUNCH_HI = 30.0, 44.0   # colliders are HURLED from FAR (beyond the base), on ballistic arcs
DEFENSE_R = 1.5            # a collider this close to the base = defense fail
FAR = 60.0                # camera far plane (< SPACING/2 so only the local region renders)
GRAVITY = 9.8             # ballistic arc (the velocity is solved so the throw lands on the base)
FLIGHT_LO, FLIGHT_HI = 2.4, 3.6     # time of flight (s) -> sets the launch speed
TARGET_SCATTER = 1.6     # colliders are thrown at the base +/- this (some near-miss)
YAW_RATE = math.radians(7.0)    # rad per step at |action|=1
PITCH_RATE = math.radians(5.0)
PITCH_LO, PITCH_HI = -0.15, 1.25
HIT_CONE = math.radians(6.0)    # a shot within this angle of a collider destroys it
COOLDOWN = 4                  # steps between shots (discrete bolts, not a spam beam)
SIZE_LO, SIZE_HI = 2.4, 3.4     # big in metres but FAR -> appears small; strongly emissive so it
FOV = 50.0                  # resolves at launch range as a bright glowing blob. FOV (deg).
SPAWN_CONE_Y = 0.5          # launched within this yaw/pitch cone of the CURRENT aim ->
SPAWN_CONE_P = 0.3          # on-screen at launch (a trackable visual-servo target)
OBS_C = 3 * FRAME_STACK
ACT_DIM = 3

CONFIG = {"control_hz": CONTROL_HZ, "dt": DT, "h": H, "w": W, "frame_stack": FRAME_STACK,
          "n_proj": N_PROJ, "episode_s": EPISODE_S, "launch_lo": LAUNCH_LO, "launch_hi": LAUNCH_HI,
          "defense_r": DEFENSE_R, "gravity": GRAVITY, "yaw_rate": YAW_RATE, "pitch_rate": PITCH_RATE,
          "hit_cone": HIT_CONE, "cooldown": COOLDOWN}

_SHAPES = ["sphere", "box", "cone", "cyl"]      # shape variety across projectile slots
_THREAT_COLOR = 0xff5522                          # one bright threat color (track-by-brightness)


def aim_dir(yaw, pitch):
    """Unit aim direction from yaw/pitch. yaw about +Y, pitch up. [.,3]."""
    cp = torch.cos(pitch)
    return torch.stack([cp * torch.sin(yaw), torch.sin(pitch), cp * torch.cos(yaw)], dim=-1)


class TurretEnv:
    def __init__(self, num_envs=64, device="cuda", seed=0):
        if not torch.cuda.is_available():
            raise RuntimeError("TurretEnv needs CUDA")
        self.K, self.dt = num_envs, DT
        self.device = torch.device(device)
        self.max_steps = int(EPISODE_S * CONTROL_HZ)
        self.g = torch.Generator(device=self.device).manual_seed(seed)
        dev = self.device

        # region centres (turret bases), spaced along X
        self.center = torch.zeros(self.K, 3, device=dev)
        self.center[:, 0] = torch.arange(self.K, device=dev) * SPACING
        self.cam_y = 1.0

        # state
        self.yaw = torch.zeros(self.K, device=dev)
        self.pitch = torch.full((self.K,), 0.4, device=dev)
        self.ppos = torch.zeros(self.K, N_PROJ, 3, device=dev)
        self.pvel = torch.zeros(self.K, N_PROJ, 3, device=dev)
        self.psize = torch.ones(self.K, N_PROJ, device=dev)
        self.cooldown = torch.zeros(self.K, dtype=torch.long, device=dev)
        self.steps = torch.zeros(self.K, dtype=torch.long, device=dev)
        self.frames = torch.zeros(self.K, FRAME_STACK, 3, H, W, dtype=torch.uint8, device=dev)

        self._build_scene()
        self._cpu_pos = np.zeros((self.K * N_PROJ, 3), np.float32)
        self._spawn(torch.arange(self.K, device=dev).repeat_interleave(N_PROJ),
                    torch.arange(N_PROJ, device=dev).repeat(self.K))

    # --- scene / rendering ----------------------------------------------------
    def _build_scene(self):
        self.canvas = tp.Canvas("turret", width=W, height=H, headless=True)
        self.renderer = tp.GLRenderer(self.canvas)
        self.scene = tp.Scene()
        self.scene.add(tp.HemisphereLight(0x99bbff, 0x223344, 1.3))
        sun = tp.DirectionalLight(0xffffff, 2.0); sun.position.set(0.4, 1.0, 0.3); self.scene.add(sun)
        geos = {"sphere": lambda s: tp.SphereGeometry(s),
                "box": lambda s: tp.BoxGeometry(2 * s, 2 * s, 2 * s),
                "cone": lambda s: tp.ConeGeometry(s, 2.4 * s),
                "cyl": lambda s: tp.CylinderGeometry(s, s, 2.2 * s)}
        self.pmesh = []
        for k in range(self.K):
            for j in range(N_PROJ):
                mat = tp.MeshStandardMaterial(); mat.color = _THREAT_COLOR
                mat.emissive = _THREAT_COLOR; mat.roughness = 0.5   # self-lit -> a bright glow at range
                m = tp.Mesh(geos[_SHAPES[j % len(_SHAPES)]](1.0), mat)
                self.scene.add(m); self.pmesh.append(m)
        self.cams = []
        for k in range(self.K):
            c = tp.PerspectiveCamera(FOV, W / H, 0.1, FAR)
            c.position.set(float(self.center[k, 0]), self.cam_y, 0.0)
            self.cams.append(c)

    @torch.no_grad()
    def _render(self):
        # push GPU projectile positions to the meshes (one CPU sync), then render K cameras
        world = self.ppos + self.center[:, None, :]          # [K,N,3] world position
        self._cpu_pos[:] = world.reshape(-1, 3).cpu().numpy()
        sc = self.psize.reshape(-1).cpu().numpy()
        for i, m in enumerate(self.pmesh):
            p = self._cpu_pos[i]
            m.position.set(float(p[0]), float(p[1]), float(p[2]))
            m.scale.set(float(sc[i]), float(sc[i]), float(sc[i]))
        d = aim_dir(self.yaw, self.pitch)                    # [K,3]
        out = np.empty((self.K, H, W, 3), np.uint8)
        cx = self.center[:, 0].cpu().numpy()
        dn = d.cpu().numpy()
        for k in range(self.K):
            cam = self.cams[k]
            cam.position.set(float(cx[k]), self.cam_y, 0.0)
            cam.look_at(float(cx[k] + dn[k, 0]), self.cam_y + float(dn[k, 1]), float(dn[k, 2]))
            self.renderer.render(self.scene, cam)
            out[k] = self.renderer.read_pixels(True)
        return torch.from_numpy(out).to(self.device).permute(0, 3, 1, 2).contiguous()  # [K,3,H,W]

    def _push_frame(self, frame):
        self.frames = torch.roll(self.frames, shifts=-1, dims=1)
        self.frames[:, -1] = frame

    def _obs(self):
        return self.frames.reshape(self.K, OBS_C, H, W)      # [K, 3*N, H, W] uint8

    # --- dynamics -------------------------------------------------------------
    def _rand(self, n, lo, hi):
        return torch.rand(n, device=self.device, generator=self.g) * (hi - lo) + lo

    def _spawn(self, ke, je):
        """(re)spawn the projectiles indexed by (ke, je): HURLED from far on a ballistic arc aimed
        at the base. Launched within a cone of the env's CURRENT aim so it enters on-screen
        (trackable); the velocity is solved so the throw lands on the base +/- scatter under gravity."""
        n = ke.numel()
        if n == 0:
            return
        syaw = self.yaw[ke] + self._rand(n, -SPAWN_CONE_Y, SPAWN_CONE_Y)
        spit = (self.pitch[ke] + self._rand(n, -SPAWN_CONE_P, SPAWN_CONE_P)).clamp(0.08, 1.2)
        rng = self._rand(n, LAUNCH_LO, LAUNCH_HI)
        p = rng[:, None] * aim_dir(syaw, spit)                               # launch point (turret-local)
        target = torch.stack([self._rand(n, -1, 1), self._rand(n, 0.0, 0.6),
                              self._rand(n, -1, 1)], dim=-1) * TARGET_SCATTER  # near the base
        t = self._rand(n, FLIGHT_LO, FLIGHT_HI)[:, None]
        # ballistic launch velocity: p + v t - 1/2 g t^2 = target  ->  v = (target-p)/t + 1/2 g t
        v = (target - p) / t
        v[:, 1] = v[:, 1] + 0.5 * GRAVITY * t[:, 0]
        self.ppos[ke, je] = p
        self.pvel[ke, je] = v
        self.psize[ke, je] = self._rand(n, SIZE_LO, SIZE_HI)

    def reset(self):
        ke = torch.arange(self.K, device=self.device).repeat_interleave(N_PROJ)
        je = torch.arange(N_PROJ, device=self.device).repeat(self.K)
        self._spawn(ke, je)
        self.yaw.zero_(); self.pitch.fill_(0.4); self.cooldown.zero_(); self.steps.zero_()
        frame = self._render()
        for _ in range(FRAME_STACK):
            self._push_frame(frame)
        return self._obs()

    @torch.no_grad()
    def step(self, action):
        a = action.clamp(-1.0, 1.0)
        self.yaw = self.yaw + a[:, 0] * YAW_RATE
        self.pitch = (self.pitch + a[:, 1] * PITCH_RATE).clamp(PITCH_LO, PITCH_HI)
        self.pvel[:, :, 1] = self.pvel[:, :, 1] - GRAVITY * self.dt   # ballistic arc
        self.ppos = self.ppos + self.pvel * self.dt
        self.steps += 1
        self.cooldown = (self.cooldown - 1).clamp_min(0)

        d = aim_dir(self.yaw, self.pitch)                          # [K,3]
        rel = torch.nn.functional.normalize(self.ppos, dim=-1)     # dir to each projectile (turret-local)
        cosang = (d[:, None, :] * rel).sum(-1).clamp(-1, 1)        # [K,N]
        ang = torch.arccos(cosang)                                 # angular error per projectile
        nearest_ang, nearest = ang.min(dim=1)                      # [K]

        # fire
        want = (a[:, 2] > 0.0) & (self.cooldown == 0)
        hit = want & (nearest_ang < HIT_CONE)
        self.cooldown = torch.where(want, torch.full_like(self.cooldown, COOLDOWN), self.cooldown)

        # defense fail: any projectile within DEFENSE_R of the turret base
        dist = self.ppos.norm(dim=-1)                              # [K,N]
        reached = dist < DEFENSE_R                                 # [K,N]

        # reward (dense): track nearest toward center + hit bonus - shot/miss - defense fail
        r_track = 0.6 * torch.exp(-(nearest_ang / math.radians(8.0)) ** 2)
        r_hit = 3.0 * hit.float()
        r_miss = -0.15 * (want & ~hit).float()
        r_fail = -2.0 * reached.any(dim=1).float()
        rew = r_track + r_hit + r_miss + r_fail - 0.005

        # respawn hit projectiles + any that reached the turret
        resp = reached.clone()
        resp[torch.arange(self.K, device=self.device)[hit], nearest[hit]] = True
        ke, je = torch.nonzero(resp, as_tuple=True)
        self._spawn(ke, je)

        done = self.steps >= self.max_steps                        # timeout only (soft penalties, no fatal terminal)
        self._push_frame(self._render())
        term_obs = self._obs()
        de = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if de.numel() > 0:
            self.yaw[de] = 0.0; self.pitch[de] = 0.4; self.cooldown[de] = 0; self.steps[de] = 0
            ke = de.repeat_interleave(N_PROJ); je = torch.arange(N_PROJ, device=self.device).repeat(de.numel())
            self._spawn(ke, je)
            f = self._render()
            self.frames[de] = f[de].unsqueeze(1).expand(-1, FRAME_STACK, -1, -1, -1)
            obs = self._obs()
        else:
            obs = term_obs
        # expose hit/fail rates for logging
        self.last_hit = hit.float().mean()
        self.last_fail = reached.any(dim=1).float().mean()
        return obs, rew, done, term_obs, done   # is_timeout = done


if __name__ == "__main__":
    env = TurretEnv(num_envs=16)
    obs = env.reset()
    print("obs", tuple(obs.shape), obs.dtype, "finite", bool(torch.isfinite(obs.float()).all()))
    # hand-coded oracle: aim at the nearest projectile, fire when centered -> should score reward+hits
    hits = 0.0
    for _ in range(120):
        d = aim_dir(env.yaw, env.pitch)
        rel = torch.nn.functional.normalize(env.ppos, dim=-1)
        ang = torch.arccos((d[:, None, :] * rel).sum(-1).clamp(-1, 1))
        na, ni = ang.min(dim=1)
        tgt = env.ppos[torch.arange(env.K), ni]                    # nearest projectile local pos
        tyaw = torch.atan2(tgt[:, 0], tgt[:, 2])
        tpitch = torch.arctan2(tgt[:, 1], tgt[:, [0, 2]].norm(dim=1)).clamp(PITCH_LO, PITCH_HI)
        ay = ((tyaw - env.yaw + math.pi) % (2 * math.pi) - math.pi) / YAW_RATE
        ap = (tpitch - env.pitch) / PITCH_RATE
        fire = (na < HIT_CONE).float() * 2 - 1
        act = torch.stack([ay.clamp(-1, 1), ap.clamp(-1, 1), fire], dim=-1)
        obs, rew, done, term, to = env.step(act)
        hits += float(env.last_hit)
    print("oracle controller: mean hit-rate/step=%.3f  (a working env should be clearly >0)" % (hits / 120))
    print("TURRET ENV SELFTEST: PASS" if hits / 120 > 0.02 else "TURRET ENV SELFTEST: SUSPICIOUS")
