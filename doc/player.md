# threepp player

`threepp_player` plays a scene the [editor](editor.md) authored. It is the same
play runtime — physics, articulations, sensors, Python scripts — with the
editing machinery taken off: no ImGui, no panels, no undo, no gizmos, no
selection. A window, a camera and the four play sessions, or not even the
window.

It exists to be a **gate**. Point it at a scene whose script drives a trained
policy, give it an episode count and a budget, let the sensors write their CSVs,
and read the exit code. That is something CI can hold, which the editor — an
interactive application whose success condition is "somebody looked at it" — is
not.

```
threepp_player scene.json [options]
```

## What it is not

An **evaluation** vehicle, not a trainer. It runs the document as written, one
instance at a time, at the fidelity the editor would. Batched rollouts across
hundreds of parallel environments are `GpuSim`'s job; there is deliberately no
`scene.json` → `GpuSim` loader here.

## Options

| flag | meaning |
| --- | --- |
| `--episodes=N` | Play the document N times back to back (default 1). |
| `--frames=N` | Stop each episode after N frames. |
| `--seconds=N` | Stop each episode after N simulated seconds. |
| `--headless` | No visible window. |
| `--dt=SECONDS` | Force a fixed simulation step instead of the wall clock. |
| `--record=DIR` | Write the sensor CSVs under `DIR`. |
| `--size=WxH` | Window size (default 1280x720). |
| `--help` | Print all of this. |

The budgets are **per episode**, and they compose: with both, whichever is
reached first ends the episode. Neither, windowed, means play until the window
is closed. `Esc` or closing the window ends the run.

A document that carries `userData["editorView"]` opens at that vantage, in the
same `px,py,pz@tx,ty,tz` spelling the editor's `--shot` takes (the parse is
shared — `extras/editor/ViewSpec.hpp`); `userData["editorFollow"]` names an
object for the camera to chase. Without a view, the player frames the scene's
bounds. `RenderConfig` on the scene root is applied, so a document looks the way
it was saved.

## Exit code

| code | meaning |
| --- | --- |
| 0 | Every episode played clean. |
| 1 | The document would not load or play, an episode failed to stop, or **any script raised**. |
| 2 | Usage error. |

A script that raises is logged once by `ScriptPlaySession`, disabled for the
rest of the episode, and the scene keeps playing — the player does not confuse
"ran" with "worked", and turns that into a nonzero exit.

Nothing running at all is also `1`. A gate that goes green because the scene
never loaded is worse than no gate.

## Episodes

Each episode is a full `PlayController` play → step → stop cycle. Stop restores
the snapshot taken at play, so **episodes are independent by construction** —
not by anybody remembering to reset something. A body that fell in episode 0
starts episode 1 back where it was authored, and the seeded sensor streams
replay identically:

```
$ threepp_player arena.json --headless --episodes=3 --seconds=3 --record=rec
episode 0: 181 frames, 3.01666 s sim, 7 script(s), 0 error(s), 35 bodies, 2 sensor(s), 212 row(s) recorded
episode 1: 181 frames, 3.01666 s sim, 7 script(s), 0 error(s), 35 bodies, 2 sensor(s), 212 row(s) recorded
episode 2: 181 frames, 3.01666 s sim, 7 script(s), 0 error(s), 35 bodies, 2 sensor(s), 212 row(s) recorded
threepp player: 3 episode(s), 0 failed, 0 script error(s)
```

The three `Drone_*.csv` files above are byte-identical, noise included.

Script errors do not leak between episodes: `ScriptPlaySession::start()` clears
the map, so each episode reports its own count and the run fails if any of them
is nonzero.

## Headless

`--headless` does **not** mean "no graphics". The window is created and simply
not shown (`GLFW_VISIBLE` false), so there is still a live GL context and the
vision sensors still scan — a `--headless --record` run of a scene with a lidar
produces its point clouds exactly as a windowed one does. If the context cannot
be created at all, the run continues with no renderer: `SensorPlaySession`
already tolerates that, so the vision sensors are built, say so once and never
scan, while every proprioceptive sensor (IMU, encoders, contact, force/torque)
records unchanged.

Headless also steps at a **fixed** 1/60 s rather than the wall clock, because a
reproducible run is the whole point, and — given neither `--frames` nor
`--seconds` — defaults to 10 simulated seconds per episode rather than hanging a
CI job forever.

## Recording

`--record=DIR` arms `SensorPlaySession`'s CSV recording and points it at `DIR`.
One file per sensor, named for the sensor's label and the first 8 characters of
its uuid, plus a `_cloud.csv` for the point clouds.

Those names are **stable across episodes** — that is what episode independence
means — and the files are opened with `trunc`. So with `--episodes>1` each
episode gets its own `DIR/episode_NNN/`, or a hundred-episode run would leave
exactly one episode's data behind. A single-episode run writes straight into
`DIR`.

## Script debug draw

`threepp.editor.draw_line` and friends push world-space segments into
`scripting::debugDraw()`. `ScriptPlaySession` switches that list on and nothing
switches it off, so **somebody has to drain it every frame** or a long run ends
at the 100000-segment cap.

The windowed player drains it by drawing it: one `LineSegments` under the
player's overlay group, attributes rewritten in place, vertex colours, depth
test off. Headless, the core simply clears the list — the scripts still *run*
their draw calls, because a draw must not behave differently just because nobody
is looking; the segments just go nowhere.

The overlay group is registered with the document as editor-only, so it never
enters a snapshot or an export, and it is handed to the sensor session as
hidden-during-scan, so a lidar cannot range against somebody's debug arrow.

## Layout

| path | what |
| --- | --- |
| `apps/player/PlayerCore.{hpp,cpp}` | the player minus the window: document, sessions, episodes, exit code. Built as `threepp_player_core` so `tests/extras/PlayerCore_test.cpp` drives the code the binary runs |
| `apps/player/PlayerApp.{hpp,cpp}` | canvas, renderer, camera, follow, the episode loop |
| `apps/player/DebugDrawOverlay.{hpp,cpp}` | the drawn lines |
| `apps/player/main.cpp` | the CLI |

The player links the play core in `threepp` and the `threepp_editor_scripting`
static library, and **no part of the editor application**. Its optional halves
are gated exactly as the editor's are: no PhysX means a document plays without
physics, no `THREEPP_EDITOR_WITH_PYTHON` means its scripts are loaded and not
run. It follows `THREEPP_BUILD_EDITOR`, because that is where the scripting
library is declared.
