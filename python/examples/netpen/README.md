# netpen — a net-pen inspection ROV, and the closed sonar loop (E3)

A fish-farm net pen with a torn Warp-cloth net, a BlueROV2 on a tether, camera and sonar
insets and a Warp school of salmon. `--terrain` puts the pen at its real site in
Norddalsfjorden. The same scene carries the paper's E3 experiment: the baked patrol is replaced
by a controller that sees the net only through the scene's imaging sonar.

| Script | What it is |
| --- | --- |
| `warp_netpen.py` | The scene: window, stills (`--shot`), the film (`--film`), the determinism row (`--audit`) and one E3 run (`--e3 SECONDS --e3-seed N`). |
| `netpen_e3.py` | The E3 loop as pure code: sonar echo ranges, wall estimate, seeded ranging noise, controller, vehicle. No scene, no GPU; `--selftest` checks it on its own. |
| `netpen_e3_runs.py` | E3's two numbers from fresh processes: the repeatability row (one seed, `--repeat` runs) and the seed spread (`--seeds` runs), then both figures. |

`--audit` and `--e3` write manifests in the format of `sensor_audit.py`, which lives in
[`../probes/`](../probes); its `--compare` gives the verdict on two of them.

## Run

    python python/examples/netpen/warp_netpen.py --film              # the film -> aaa_caps/netpen/
    python python/examples/netpen/netpen_e3.py --selftest
    python python/examples/netpen/netpen_e3_runs.py --dry            # print the E3 commands and stop
    python python/examples/netpen/netpen_e3_runs.py                  # 10 + 10 runs -> aaa_caps/netpen/e3/

## Snake robot

The ROV retires and an underwater snake robot (9 modules of 0.18 m, 1.62 m; two side thrusters
and two tunnel thrusters, the arrangement of NTNU's swimming manipulator behind Eelume) lives in a
tube cradle hung from the inner collar. It backs out of the cradle, pivots, follows the net on its
head sonar, passes the tear twice ~1.25 m off the cloth with its front two joints turning the head
toward the hole, U-turns, returns and docks into the current through a funnel. Same scene:
`snake_netpen.py` imports `warp_netpen.py` and hooks its head camera, sonar and fish avoidance.

| Script | What it is |
| --- | --- |
| `snake_model.py` | The physics: 9 PhysX capsule links, 8 yaw joints (PD kp 20 / kd 5), the paper's fluid forces per link at 240 Hz, optional thrusters (force at the mounting point, spin-up lag, momentum-theory power). `--selftest`. |
| `snake_sweep.py` | The validation: amplitude / frequency / phase-shift sweeps x 2 gaits x 2 coefficient sets against the paper's trends. |
| `snake_mission.py` | The mission as pure code: sonar wall following, the tear detector, the state machine, docking. `--selftest` (`--physx`). |
| `snake_netpen.py` | The scene: `--mission` (headless, telemetry + summary), `--snake-film` (pass 1 of the film), stills, the window. |
| `snake_panels.py` | Offline HUD panels and the end card from the telemetry (PIL). `--preview`. |
| `snake_film_cut.py` | Pass 2 of the film: the edit (real-time windows cut hard, insets, panels, title, end card). |
| `snake_dock_clearance.py` | The rendered body against the cradle's bars, rings, funnels and pad, per telemetry row; exit 1 on any overlap. |

    python python/examples/netpen/snake_model.py --selftest
    python python/examples/netpen/snake_sweep.py                          # -> D:/dev/snake_out/sweep
    python python/examples/netpen/snake_netpen.py --mission --seed 0 --size 640x360 --render-every 3
    python python/examples/netpen/snake_netpen.py --snake-film --seed 0   # 1920x1080, every frame -> D:/dev/snake_out/film
    python python/examples/netpen/snake_netpen.py --snake-film --film-test   # stills per phase instead
    python python/examples/netpen/snake_film_cut.py                       # -> D:/dev/snake_out/film/snake_film.mp4

What is modelled and what is approximated:

- **Swimming.** Kelasidi, Liljeback, Pettersen, Gravdahl (2015), Robotics and Biomimetics 2:8: linear
  and quadratic drag (tangential and normal), added mass on the normal axes, fluid torques, with the
  coefficients the paper prints (its full set and its control-oriented set). Those come from the
  paper's generic simulation study, not an identification on Mamba. The normal coefficients act on
  both normal axes in 3D; gravity is off with a small righting moment (neutral buoyancy).
- **Validation is trend-level**, as the paper's own is: it tabulates no absolute speeds. The sweep
  reports each trend PASS/FAIL (speed vs amplitude, frequency, phase shift; power vs amplitude).
- **Thrusters, not undulation.** The paper's robot swims by undulating; this one flies the mission
  on thrusters with the joints shaping the body (into a turn's arc, straight in the cradle, the
  neck at the tear). Undulation failed the job: at the tear the 0.15 m/s current pushes inboard and
  the only steady-headed stroke (eel-like, 0.185 m/s in still water) drifted 2 m off the net;
  lateral undulation swings the head, camera and sonar 30+ deg; and a swinging body cannot leave a
  tube cradle without hitting it. The thruster ratings (4 N side, 2.5 N tunnel) and the 50 %
  propeller x motor efficiency are assumptions, not Eelume data.
- **Not the paper's:** the thrusters, the dock's latch line (a capped spring-damper that holds the
  head, and hauls it the last centimetres after capture) and the cradle's wall contact (every
  phase, 2 cm inside the rendered bars; `snake_dock_clearance.py` on the telemetry finds no overlap).
- **Navigation.** Along the net the robot steers on its own sonar images; the return and the
  docking use a USBL fix of the head at 1 Hz (5 cm noise) plus dead reckoning, as a real resident
  vehicle would. The mission is told the tear's reported sector (+-20 deg); the tear detector
  fires on a gap in the sonar's wall return when it sees one, else on the sector. Each run's
  summary json and the film's end card say which mode fired.
- The film (`--snake-film` + `snake_film_cut.py`) is one closed-loop run; every camera and window
  is chosen from that run's own phase events and telemetry. It plays in real time and cuts hard
  between windows (a SKIPPED +N s badge and the mission clock show the jump): a sped-up swimmer
  reads as a wiggle.
