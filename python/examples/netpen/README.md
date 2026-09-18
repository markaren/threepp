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
