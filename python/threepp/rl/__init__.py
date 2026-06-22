"""threepp.rl — GPU-vectorized RL on top of the PhysX direct-GPU backend.

    from threepp.rl import GpuSim, ActorCritic, RunningNorm, compute_gae, save_policy, load_policy

GpuSim hides the GPU plumbing (CUDA context, direct-GPU batch, DOF-order remap, sync); the
rest is a compact, owned PPO. Importing this subpackage pulls in torch, so `import threepp`
stays lightweight — you only pay for torch when you ask for threepp.rl.
"""
from .ppo import ActorCritic, RunningNorm, compute_gae, load_policy, save_policy
from .sim import GpuSim

__all__ = ["GpuSim", "ActorCritic", "RunningNorm", "compute_gae", "save_policy", "load_policy"]
