# ArduPilot SITL demo

`ardupilot_sitl` (examples/projects/ArduPilotSitl) turns threepp into the
**physics backend and visualizer** for ArduPilot's software-in-the-loop
simulator: ArduCopter's real flight stack runs unmodified (in WSL2 on Windows,
or natively on Linux), while threepp simulates the airframe with PhysX and
renders it. You fly the drone with ordinary MAVProxy commands; landings are
real mesh contacts on the rendered terrain.

## How it talks to SITL

ArduPilot's [JSON backend](https://ardupilot.org/dev/docs/sitl-with-JSON.html)
is strict lock-step over UDP:

1. SITL sends a binary packet to our port **9002**: `uint16 magic` (18458, or
   29569 for 32 channels), `uint16 frame_rate`, `uint32 frame_count`,
   `uint16 pwm[16]` (µs).
2. We run exactly **one PhysX substep** at `1/frame_rate` and reply with one
   newline-terminated JSON line: `timestamp`, `imu.gyro` / `imu.accel_body`
   (body FRD), `position` / `velocity` (NED), `attitude`, `rng_1`, `battery`.
3. Our `timestamp` drives SITL's clock; SITL will not advance until we reply.

Replies go to whatever address sent the last packet, so WSL2's NAT needs no
configuration on the Linux side. `frame_count` going backwards means SITL
restarted — the demo re-homes the vehicle automatically (HUD "resets" counter).
SITL re-sends its packet every 10 s while unanswered, so the two sides can be
started in either order.

The IMU reply comes from `threepp::Imu` with all noise zeroed — SITL adds its
own sensor noise via `SIM_*` parameters.

## One-time WSL2 setup

Any WSL2 Ubuntu (22.04/24.04) works:

```bash
sudo apt update
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git ~/ardupilot
cd ~/ardupilot
Tools/environment_install/install-prereqs-ubuntu.sh -y
# re-login (or `source ~/.profile`) so ~/.local/bin lands on PATH
```

The prereqs script uses `sudo` internally and **dies silently when sudo can't
prompt** (scripted/non-interactive shells). A minimal manual alternative that
only needs the compiler toolchain and a venv:

```bash
sudo apt install -y python3-venv python3-pip python3-dev ccache pkg-config
python3 -m venv ~/venv-ardupilot
source ~/venv-ardupilot/bin/activate
pip install -U pip wheel setuptools
pip install 'empy==3.3.4' pexpect future lxml pymavlink MAVProxy pyserial
cd ~/ardupilot && ./waf configure --board sitl && ./waf copter
```

(`empy` must be 3.3.4 — the 4.x releases break ArduPilot's code generators.
Activate the venv in any shell that runs `sim_vehicle.py`.)

## Running

1. **Windows**: start the demo (any order works):

   ```powershell
   .\bin\ardupilot_sitl.exe          # listens on UDP :9002
   ```

2. **WSL2**: launch SITL pointing at the Windows host:

   ```bash
   HOST_IP=$(ip route show default | cut -d " " -f3)
   cd ~/ardupilot
   Tools/autotest/sim_vehicle.py -v ArduCopter -f JSON:$HOST_IP --console
   ```

   Add `-w` on the first run to wipe parameters. `--map` gets you the MAVProxy
   map. The demo HUD shows *peer / rate / frame* once the link is up.

   The commands above are typed **inside a WSL shell**. To drive them from
   PowerShell as one-liners, you must use `wsl -e bash -c '...'` — plain
   `wsl bash -c` re-joins the arguments through the default shell and destroys
   quoting/`$` expansions (symptom: `JSON control interface set to :9002` with
   an empty IP).

3. **Fly** (in the MAVProxy prompt; wait ~30 s after boot for EKF/AHRS to
   settle — "IMU0 is using GPS"):

   ```
   mode guided
   arm throttle
   takeoff 10
   velocity 3 0 0        # 3 m/s north (visually: toward -Z)
   mode land
   ```

### Rates

The demo runs its PhysX world at a fixed **1200 Hz** and deliberately ignores
the packet's `frame_rate` field: SITL varies that value dynamically while
syncing its speedup, and the protocol readme states the physics backend is
free to ignore it. Lock-step means SITL's clock follows our reply timestamps
either way, so no `SIM_RATE_HZ` tuning is needed.

Why 1200: ArduCopter's INS prearm check requires the gyro sample rate to be at
least 1.8× the 400 Hz loop rate (`PreArm: Gyro 0 rate 400Hz < loop rate*1.8
720Hz`), and 3× the loop rate is what SITL itself requests.

## Windows Firewall

The first launch usually pops the allow dialog — accept it for private
networks. If SITL prints `JSON received:...` never appears / the HUD stays on
"Waiting", add the rule manually (admin PowerShell):

```powershell
New-NetFirewallRule -DisplayName "ArduPilot SITL JSON" -Direction Inbound `
    -Protocol UDP -LocalPort 9002 -Action Allow
```

Note the WSL vEthernet adapter is typically on the **Public** profile.

### Mirrored networking (alternative)

Windows 11 22H2+ can put WSL2 on the host's network directly. In
`%UserProfile%\.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
```

then `wsl --shutdown`, and launch with `-f JSON:127.0.0.1`.

## Verifying without ArduPilot

```powershell
.\bin\ardupilot_sitl.exe --selftest
```

runs an in-process fake SITL over loopback: at-rest specific force, climb,
roll/pitch/yaw sign checks against the ArduCopter X-frame motor table, restart
detection, 32-channel packets. Exit 0 = pass, and `SitlBridge_test` (Catch2)
covers the codec + NED↔threepp frame mapping.

## Automated flight check

`examples/projects/ArduPilotSitl/fly_test.py` (pymavlink) drives the whole
loop without MAVProxy: waits for EKF, arms (feeding RC overrides — ArduPilot
refuses to arm with no RC source), takes off to 10 m, flies 3 m/s north for
5 s, lands, and exits 0 once disarmed. With the demo running on Windows and
`arducopter --model json:$HOST_IP` running in WSL (`--no-mavproxy` style,
serial0 on TCP 5760):

```bash
source ~/venv-ardupilot/bin/activate
python3 fly_test.py tcp:127.0.0.1:5760
```

This exact sequence is the demo's live acceptance test.

## Troubleshooting

- **"link 1 down" / no heartbeat in MAVProxy** — SITL is blocked waiting for
  our reply: the demo isn't running, or the firewall ate the packet (above).
- **"Waiting for SITL" never clears** — same two suspects; check
  `Get-NetUDPEndpoint -LocalPort 9002` shows the demo listening.
- **PreArm: EKF errors right after boot** — normal for ~30 s; the EKF needs
  quiet IMU time on the pad before arming.
- **Vehicle drifts/flips on takeoff** — frame mapping or motor order broke;
  run `--selftest`, whose sign checks localize exactly which axis is wrong.
- **SITL restarted but the drone hovers mid-air** — shouldn't happen: the demo
  detects `frame_count` rollback and re-homes; the HUD resets counter should
  increment.

## Files

| File | Role |
|---|---|
| `SitlBridge.hpp` | UDP + wire protocol, no threepp deps (reusable) |
| `FrameConv.hpp` | the single NED/FRD ↔ threepp axis mapping |
| `QuadSim.hpp` | PhysX lock-step airframe + `threepp::Imu` |
| `DroneVisual.hpp` | primitive-built X-quad, spin-keyed rotors |
| `main.cpp` | scene, HUD, drain-loop, `--selftest` |
| `tests/extras/SitlBridge_test.cpp` | codec + frame-mapping unit tests |
