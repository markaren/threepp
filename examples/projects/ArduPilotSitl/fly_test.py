#!/usr/bin/env python3
"""Automated GUIDED flight test against the threepp SITL backend.

Connects to ArduCopter SITL (started with --no-mavproxy), waits for EKF,
then: GUIDED -> arm -> takeoff 10 m -> 3 m/s north for 5 s -> RTL.
Exit 0 on a completed round trip, 1 on any timeout.

With --tour: uploads a scenic waypoint loop over the valley (climbing to
145 m — into the Vulkan cloud base) and flies it in AUTO at 8 m/s,
ending with RTL onto the pad.
"""
import sys
import time

from pymavlink import mavutil

# Tour waypoints as (north, east, alt-above-home) metres. The terrain rises
# to ~65 m above home outside the apron and waypoint altitudes are relative
# to HOME (no terrain following), so every en-route leg stays >= 90 m; only
# the final fix over the flat pad apron descends below that.
#
# Kept deliberately COMPACT (~650 m of track, ~80 s at 8 m/s, versus ~1520 m
# before) so the loop fits a single continuous take. Shrinking the footprint
# does not buy any altitude margin -- the radial falloff leaves the terrain
# near home at full amplitude, so the >= 90 m floor still applies -- but the
# drone now stays inside a ~130 m radius, which keeps it large in frame from
# a chase camera instead of shrinking to a dot on the long legs.
TOUR = [
    (90, 0, 90),
    (90, 90, 110),
    (0, 130, 145),# peak leg, into the 140 m cloud base
    (-100, 90, 120),
    (-100, -60, 95),
    (0, 0, 35),
]


def upload_tour(m, home_lat, home_lon):
    """Upload TOUR as a mission: takeoff, waypoints, RTL."""
    import math
    dlat = 1.0 / 111111.0
    dlon = 1.0 / (111111.0 * math.cos(math.radians(home_lat / 1e7)))

    def item(seq, cmd, n=0.0, e=0.0, alt=0.0):
        return m.mav.mission_item_int_encode(
                m.target_system, m.target_component, seq,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, cmd,
                0, 1, 0, 0, 0, 0,
                int(home_lat + n * dlat * 1e7),
                int(home_lon + e * dlon * 1e7),
                alt)

    items = [item(0, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT),  # home placeholder
             item(1, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, alt=30)]
    for i, (n, e, alt) in enumerate(TOUR):
        items.append(item(2 + i, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, n, e, alt))
    items.append(item(len(items), mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH))

    # Post the route to the demo's waypoint-marker port so the scene can SHOW
    # the mission (fire-and-forget; harmless if the demo isn't listening).
    #
    # Sent to every plausible address for the Windows host rather than one
    # guess, because which one is right depends on how WSL2 is networked and
    # nothing here can tell: under the default NAT the host is the default
    # gateway, but under `networkingMode=mirrored` that gateway is the physical
    # ROUTER and the datagram vanishes silently (the mission simply never
    # appears, with no error anywhere). These are unconnected UDP sends to a
    # port nothing else uses, so the wrong one costs a discarded packet.
    text = "0,0,30;" + ";".join(f"{n},{e},{alt}" for n, e, alt in TOUR) + ";"
    hosts = ["127.0.0.1"]  # mirrored networking, and native Linux
    try:
        import subprocess
        hosts.append(subprocess.check_output(["sh", "-c", "ip route show default"],
                                             text=True).split()[2])  # NAT
    except Exception:  # pragma: no cover - no default route
        pass
    for host in hosts:
        try:
            import socket
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(
                    text.encode(), (host, 9008))
        except Exception as exc:  # pragma: no cover - cosmetic path
            print(f"[fly] marker send to {host} skipped: {exc}")
    print(f"[fly] waypoint markers sent to {', '.join(hosts)} :9008")

    m.mav.mission_count_send(m.target_system, m.target_component, len(items))
    sent = 0
    t0 = time.time()
    while sent < len(items) and time.time() - t0 < 30:
        msg = m.recv_match(type=["MISSION_REQUEST", "MISSION_REQUEST_INT", "MISSION_ACK"],
                           blocking=True, timeout=5)
        if msg is None:
            continue
        if msg.get_type() == "MISSION_ACK":
            break
        m.mav.send(items[msg.seq])
        sent = max(sent, msg.seq + 1)
    print(f"[fly] mission uploaded ({len(items)} items)")


def wait_for(cond, timeout, what, tick=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            print(f"[fly] {what}: ok ({time.time() - t0:.1f} s)")
            return True
        if tick:
            tick()
        time.sleep(0.2)
    print(f"[fly] {what}: TIMEOUT after {timeout} s")
    return False


def main():
    tour = "--tour" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    url = args[0] if args else "tcp:127.0.0.1:5760"
    m = mavutil.mavlink_connection(url)
    print("[fly] waiting for autopilot heartbeat...")
    t0 = time.time()
    while time.time() - t0 < 120:
        hb = m.wait_heartbeat(timeout=120)
        if hb and m.target_system > 0 and hb.type != mavutil.mavlink.MAV_TYPE_GCS:
            break
    if m.target_system <= 0:
        print("[fly] no autopilot heartbeat")
        return 1
    print(f"[fly] heartbeat from sys {m.target_system} comp {m.target_component}")

    # Nothing sets stream rates on a raw TCP link; ask for them explicitly.
    m.mav.request_data_stream_send(m.target_system, m.target_component,
                                   mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1)

    state = {"alt": 0.0, "armed": False, "ekf_ok": False, "mode": "", "rc_t": 0.0,
             "lat": 0, "lon": 0, "wp": -1}

    def pump():
        # ArduPilot refuses to arm with no RC source ("throttle below
        # failsafe"); feed centred sticks with low throttle like MAVProxy does.
        if time.time() - state["rc_t"] > 0.5:
            m.mav.rc_channels_override_send(m.target_system, m.target_component,
                                            1500, 1500, 1000, 1500,
                                            65535, 65535, 65535, 65535)
            state["rc_t"] = time.time()
        while True:
            msg = m.recv_match(blocking=False)
            if msg is None:
                return
            t = msg.get_type()
            if t == "GLOBAL_POSITION_INT":
                state["alt"] = msg.relative_alt / 1000.0
                if state["lat"] == 0:
                    state["lat"], state["lon"] = msg.lat, msg.lon
            elif t == "MISSION_CURRENT":
                if msg.seq != state["wp"]:
                    state["wp"] = msg.seq
                    print(f"[fly] waypoint {msg.seq}")
            elif t == "HEARTBEAT":
                state["armed"] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                state["mode"] = mavutil.mode_string_v10(msg)
            elif t == "STATUSTEXT":
                print(f"[ap] {msg.text}")
            elif t == "EKF_STATUS_REPORT":
                good = (mavutil.mavlink.EKF_ATTITUDE
                        | mavutil.mavlink.EKF_VELOCITY_HORIZ
                        | mavutil.mavlink.EKF_POS_HORIZ_REL)
                state["ekf_ok"] = (msg.flags & good) == good

    if not wait_for(lambda: state["ekf_ok"], 90, "EKF ready", pump):
        print("[fly] no EKF report stream; relying on arming checks instead")

    # GUIDED
    m.set_mode_apm("GUIDED")
    if not wait_for(lambda: state["mode"] == "GUIDED", 20, "mode GUIDED", pump):
        return 1

    # Arm (retry: prearm checks can lag the EKF report slightly)
    def try_arm():
        m.mav.command_long_send(m.target_system, m.target_component,
                                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                0, 1, 0, 0, 0, 0, 0, 0)

    t_last = [0.0]

    def arm_tick():
        pump()
        if time.time() - t_last[0] > 3:
            try_arm()
            t_last[0] = time.time()

    try_arm()
    if not wait_for(lambda: state["armed"], 60, "armed", arm_tick):
        return 1

    # Takeoff 10 m
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                            0, 0, 0, 0, 0, 0, 0, 10)
    if not wait_for(lambda: state["alt"] > 9.0, 60, "reached 10 m", pump):
        return 1

    if tour:
        # Scenic AUTO loop: mission was flyable-uploaded pre-arm would also
        # work, but uploading now keeps the simple path identical to the
        # basic test. A brisk pace makes the tour watchable.
        if not wait_for(lambda: state["lat"] != 0, 20, "home position", pump):
            return 1
        m.mav.param_set_send(m.target_system, m.target_component,
                             b"WPNAV_SPEED", 800, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        upload_tour(m, state["lat"], state["lon"])
        m.set_mode_apm("AUTO")
        if not wait_for(lambda: state["mode"] == "AUTO", 20, "mode AUTO", pump):
            return 1
        if not wait_for(lambda: not state["armed"] and state["alt"] < 0.5, 600,
                        "tour flown, landed and disarmed", pump):
            return 1
    else:
        # 3 m/s north for 5 s (velocity setpoint in LOCAL_NED)
        print("[fly] velocity 3 m/s north for 5 s")
        t0 = time.time()
        while time.time() - t0 < 5:
            m.mav.set_position_target_local_ned_send(
                    0, m.target_system, m.target_component,
                    mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                    0b0000111111000111,  # velocity only
                    0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0)
            pump()
            time.sleep(0.2)

        # Return to launch (flies back to the pad) + wait for disarm
        m.set_mode_apm("RTL")
        if not wait_for(lambda: not state["armed"] and state["alt"] < 0.5, 180,
                        "returned, landed and disarmed", pump):
            return 1

    print("[fly] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
