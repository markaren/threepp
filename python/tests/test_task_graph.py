"""VecTask's CUDA-graph replay of the per-step torch region.

Uses sim-less tasks (build_robot=None), so this needs torch and a CUDA device but no PhysX: the
machinery under test is VecTask's, not the physics backend's.
"""
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

from threepp.rl import VecTask                                   # noqa: E402


class Counter(VecTask):
    """A task with no physics: the state is a counter the action nudges. Everything it does is
    shape-static and free of host reads, so it is exactly what the capture expects."""
    control_hz = 50
    episode_s = 1.0
    act_dim = 1

    def __init__(self, k=64, **kw):
        super().__init__(k, None, device="cuda", **kw)
        self.x = self.env_state(())
        # registered, so the reset that follows the capture wipes what the warm-up did to it
        self.bias = self.env_state(())

    def simulate(self, a):
        self.x.add_(a[:, 0] * 0.01 + 0.001)

    def on_reset(self, idx):
        self.x[idx] = 0.0

    def on_step(self, s):
        self.bias.add_(0.5)                       # in place: the graph captured this address

    def observe(self, s):
        return torch.stack([self.x, self.bias, torch.sin(self.x * 3.0)], dim=1)

    def reward_terms(self, s, a):
        return {"live": torch.ones(self.K, device=self.device),
                "eff": -0.01 * a[:, 0].pow(2)}

    def terminated(self, s):
        return self.x > 1e9                       # never; the time limit ends episodes


class RebindOnReset(Counter):
    """on_reset rebinds `x` instead of writing through it. The reset path runs eagerly, outside the
    graph, so from then on everything eager uses the new tensor while the replayed observation is
    still reading the address the capture recorded — the classic way a graph goes stale."""
    episode_s = 0.1                               # 5 steps, so a reset lands inside a short check

    def on_reset(self, idx):
        self.x = self.x.clone()                   # NOT in place
        self.x[idx] = 0.0


def test_graph_replay_matches_eager():
    env = Counter(graph=True)
    env.reset()
    a = torch.zeros(env.K, 1, device=env.device)
    for _ in range(5):
        env.step(a)
    assert env.verify_graph(steps=24) == 0.0


def test_graph_and_eager_agree_step_for_step():
    """Same task, same seed, same actions: replay must reproduce the eager trajectory exactly."""
    a = torch.full((64, 1), 0.3, device="cuda")
    out = {}
    for label, use in (("eager", False), ("graph", True)):
        env = Counter(k=64, seed=3, graph=use)
        obs = env.reset()
        traj = [obs.clone()]
        for _ in range(40):
            traj.append(env.step(a)[0].clone())
        out[label] = torch.stack(traj)
    assert torch.equal(out["eager"], out["graph"])


def test_verify_graph_catches_a_frozen_capture():
    """The guard has to fail on a task that varies something the graph froze — otherwise it is
    only ever confirming that two identical code paths agree."""
    env = RebindOnReset(graph=True)
    env.reset()
    a = torch.zeros(env.K, 1, device=env.device)
    for _ in range(5):
        env.step(a)
    with pytest.raises(AssertionError, match="froze"):
        env.verify_graph(steps=12)


def test_verify_graph_refuses_an_uncaptured_env():
    env = Counter(graph=False)
    env.reset()
    with pytest.raises(RuntimeError, match="graph=True"):
        env.verify_graph()


def test_graph_off_is_the_default():
    env = Counter()
    assert env.graph is False
    env.reset()
    env.step(torch.zeros(env.K, 1, device=env.device))
    assert env._cg_hot is None                     # nothing was captured
