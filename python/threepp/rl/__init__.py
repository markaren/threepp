"""threepp.rl — GPU-vectorized RL on top of the PhysX direct-GPU backend.

    from threepp.rl import GpuSim, VecTask, PPO, load_policy

GpuSim hides the GPU plumbing (CUDA context, direct-GPU batch, DOF-order remap, sync); VecTask
is the env-authoring base that owns the step/reset/timeout choreography so a task is just its
observation, named reward terms, and reset; PPO is a compact, owned trainer that owns the
rollout+update loop (ActorCritic / RunningNorm / compute_gae are also exported for hand-rolled
loops). Importing this subpackage pulls in torch, so `import threepp` stays lightweight — you
only pay for torch when you ask for threepp.rl.
"""
from .ppo import PPO, ActorCritic, RunningNorm, compute_gae, load_policy, save_policy
from .sim import GpuSim
from .task import RobotState, VecTask, quat_to_frame

__all__ = ["GpuSim", "VecTask", "RobotState", "quat_to_frame", "PPO", "ActorCritic",
           "RunningNorm", "compute_gae", "save_policy", "load_policy"]
