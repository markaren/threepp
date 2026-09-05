# Cart-pole — the smallest end-to-end RL loop

A cart with a single pole, trained to swing up from hanging straight down and hold
the balance. K cart-poles run in one PhysX direct-GPU scene; the observation, reward,
reset and the PPO update are all torch on the GPU. Training takes about a minute and
a half, and the resulting checkpoint is checked in, so `play` and `eval` work on a
fresh clone.

| Script | What it is |
| --- | --- |
| `cartpole.py` | The mechanism on its own: cart + pole as a threepp PhysX articulation, no learning. |
| `cartpole_env.py` | The GPU-vectorized swing-up `VecTask` — obs, reward, resets. |
| `train_cartpole.py` | `python train_cartpole.py --iters 300` — PPO from `threepp.rl`, ~1.5 min. |
| `play_cartpole.py` | Watch the trained policy swing up from hanging and balance. |
| `eval_cartpole.py` | The acceptance test: start hanging straight down, run deterministically, measure whether it swings up and holds. Headless. |
