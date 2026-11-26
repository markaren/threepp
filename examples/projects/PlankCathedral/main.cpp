// A cathedral of identical wooden planks, designed from one seed, proven to
// stand by PhysX, and knocked down by a steel wrecking ball if you ask for it.
//
// Nothing is loaded from disk: the planks, their wood, the hall and its light
// are all made here. Cathedral.hpp lays out every plank, Stability.hpp settles
// the design in PhysX and rejects one that moves, Look.hpp makes the wood, and
// Hall.hpp the exhibition hall it stands in and the sun through its windows.
//
//   mouse   orbit and zoom
//   B       let the wrecking ball go
//
// Flags:
//   --seed N          which cathedral (default 1)
//   --design N        start from the seed's Nth design (the same kind of building, its sizes drawn again)
//   --scale S         smaller buildings for quick probes (default 1)
//   --ball            let the wrecking ball go at the start
//   --ball-speed V    its speed at the bottom of its swing, m/s (default 4.5)
//   --ball-radius R   metres (default 0.1, steel)
//   --rope L          metres from the pivot (default 6)
//   --ball-aim DX DZ  the direction it swings in (default from the south-west)
//   --ball-height H   of its centre at the bottom of the swing (default 0.16)
//   --ball-offset O   of its path beside the south-west tower's axis, metres
//   --physics         design and settle on the CPU, report, and exit (no window, no GPU)
//   --collapse T      with --physics: also run T seconds after the ball hits, and report
//   --dump-final f    with --collapse: every plank's final pose, one line each
//   --settle T        seconds of awake settling (default 3; 0 skips the proof). A design passes when no
//                     plank moved more than 0.1 mm in the last 0.5 s and none more than 3 mm in all;
//                     otherwise the next design of the seed is tried, up to three
//   --record file     CPU only: prove the design, let the ball go, and write every plank's pose and the
//                     ball's from the release on (a small header, then raw floats per frame)
//   --record-fps F    frames a second in the recording (default 60)
//   --record-time T   seconds recorded from the release (default 7.5)
//   --replay file     render from a recording: no physics at all (with --probe, --shot, --sequence)
//   --threads N       PhysX worker threads
//   --hz N, --iters N the step rate (default 960) and solver position iterations (default 64) of the proof
//   --scene-hz N, --scene-iters N  the same once it is proven, for the ball and the collapse (default 960, 24:
//                     at 480 Hz and 12 iterations a woken wall sags on its own, which would not be honest)
//   --explain         say which planks the generator left out, and why
//   --dump file.csv   write the design, one plank a line: element, course, x, z, angle
//   --test stack|ring|wall|tower|roof N   one element of N courses instead of the cathedral
//   --exposure E      tone mapping exposure
//   --ssaa L          headless: supersample with 2^L samples (default 4; 0: MSAA and FXAA)
//   --sun AZ EL       the sun's azimuth west of north and its elevation, degrees (default 48 24)
//   --beam-x X        the window whose beam the building stands in is centred on the ray from this far east of
//                     the west towers, at 1 m up (default 0.5)
//   --key K, --room R, --fill F, --bounce B   the sun's intensity, the room's light (environment), the room's warm
//                     bounce from the side away from the sun (in the environment), the floor's bounce from below
//
// The hall (Hall.hpp) is placed round the building: its north wall 7.5 m from the building's axis, the windows
// high along it, the floor at y = 0.
//
// Headless:
//   --shot file.png [--time T] [--cam X Y Z] [--target X Y Z] [--fov F] [--size WxH]
//   --sequence dir [--fps F] [--time T]
//   --stills dir      the standard set of views for this seed
//   --hero file.png   the hero view alone
//   --probe dir       the collapse: one wide view before, and 0.25, 0.5, 1, 2, 4 and 6 s after the ball hits
//                     (from a recording with --replay, so that the GPU is held for seconds)
//
// The film (Film.hpp), from a recording of the film's building:
//   --replay rec --film dir --ffmpeg exe [--next-seed 3 --next-design 0] [--size 1920x1080] [--ssaa 4]
//                     draw the film's chunks not yet drawn for this plan into dir/chunks, for at most
//                     --film-budget seconds (default 240); exit 3 while some are left; once every chunk is
//                     drawn, join them into the clean master dir/master.mkv and exit 0
//   --film-chunk K    draw only chunk K (again), and say what it cost
//   --film-finish     no drawing: the master again from the chunks, the words burned in by ffmpeg: dir/film.mp4
//   --film-sheet      the first, middle and last frame of every shot, the words on them: dir/shot_sheet.png
//   --film-clip name T0 T1   the film from T0 to T1 s, clean (no words): dir/clip_name.mp4
//   --film-clip-shots name N M   the same from shot N's start to shot M's end, at most 10 s
//   --film-fresh      forget the chunks drawn so far (their stamps), so that every chunk is drawn again
//   --film-stills d   no film: a few of its frames as PNGs in d (the hook's first and last, the tower's, the
//                     build half done, the swing, the collapse 1 s and 6 s after the hit, the slow motion, the next seed)
// Every invocation writes dir/captions.ass and dir/captions.md from the one table in Film.hpp. The ffmpeg
// executable can also come from the FFMPEG environment variable.

#include "threepp/threepp.hpp"

#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/geometries/RoundedBoxGeometry.hpp"
#include "threepp/geometries/TubeGeometry.hpp"
#include "threepp/postprocessing/postprocessing.hpp"

#include "Cathedral.hpp"
#include "Film.hpp"
#include "Hall.hpp"
#include "Look.hpp"
#include "Stability.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using namespace threepp;
using namespace planks;

namespace {

    using WallClock = std::chrono::steady_clock;

    double since(WallClock::time_point t0) {
        return std::chrono::duration<double>(WallClock::now() - t0).count();
    }

    struct View {
        std::string name;
        Vector3 eye, target;
        float fov = 30;
    };

    // What the building looks like from the floor plan's point of view.
    struct Layout {
        Box3 box;
        Vector3 centre;
        std::vector<Vector3> points;// plank centres, for framing
        float naveX0 = 0, naveX1 = 0, crossingX = 0, eastX = 0, eaves = 0, ridge = 0, zWall = 0;
        Vector3 tower;// the south-west (or the one) west tower's axis, on the floor
        float towerHeight = 0, towerReach = 0, towerHalf = 0, shaftTop = 0;
    };

    Layout layoutOf(const Design& d, const Plan& p) {
        Layout l;
        for (const auto& pl : d.planks) {
            l.box.expandByPoint(Vector3(pl.x - LEN / 2, courseY(pl.course) - THICK / 2, pl.z - LEN / 2));
            l.box.expandByPoint(Vector3(pl.x + LEN / 2, courseY(pl.course) + THICK / 2, pl.z + LEN / 2));
        }
        l.box.getCenter(l.centre);
        for (const auto& pl : d.planks) l.points.emplace_back(pl.x, courseY(pl.course), pl.z);
        l.naveX0 = p.naveStart;
        l.naveX1 = p.naveEnd;
        l.crossingX = p.crossingX;
        l.eastX = p.eastX;
        l.eaves = static_cast<float>(p.naveCourses) * THICK;
        l.ridge = static_cast<float>(p.ridgeCourse) * THICK;
        l.zWall = p.zWall;
        l.tower.set(p.towerX, 0, p.towerZ);
        l.towerHalf = p.towerHalf;
        l.towerReach = p.towerHalf * 1.15f;// the buttresses stand a little proud of the faces
        l.shaftTop = static_cast<float>(p.towerTop) * THICK;
        for (const auto& pl : d.planks) {
            const float dx = pl.x - p.towerX, dz = pl.z - p.towerZ;
            if (std::abs(dx) < p.towerHalf && std::abs(dz) < p.towerHalf) l.towerHeight = std::max(l.towerHeight, courseY(pl.course) + THICK / 2);
        }
        return l;
    }

    // How far back along `from` (a direction away from the target, its height
    // kept) the eye must stand for every point to fall inside the frame, with
    // `margin` of the frame to spare at the edges.
    View fit(const std::string& name, const Vector3& target, Vector3 away, float eyeHeight, float fov, float aspect,
             const std::vector<Vector3>& points, float margin) {
        away.y = 0;
        away.normalize();
        PerspectiveCamera cam(fov, aspect, 0.01f, 200);
        const auto inside = [&](float distance) {
            cam.position.set(target.x + away.x * distance, eyeHeight, target.z + away.z * distance);
            cam.lookAt(target);
            cam.updateMatrixWorld();
            for (auto p : points) {
                p.project(cam);
                if (std::abs(p.x) > 1 - margin || std::abs(p.y) > 1 - margin || p.z > 1) return false;
            }
            return true;
        };
        float lo = 0.3f, hi = 60.f;
        for (int i = 0; i < 40; ++i) {
            const float mid = 0.5f * (lo + hi);
            (inside(mid) ? hi : lo) = mid;
        }
        View v{name, Vector3(target.x + away.x * hi, eyeHeight, target.z + away.z * hi), target, fov};
        return v;
    }

    // A view along `dir` (from the target toward the eye) that holds every point
    // with `margin` of the frame to spare, the points centred in the frame and
    // the eye as close as that allows.
    View frame(const std::string& name, Vector3 target, Vector3 dir, float fov, float aspect,
               const std::vector<Vector3>& points, float margin) {
        dir.normalize();
        PerspectiveCamera cam(fov, aspect, 0.01f, 200);
        const auto place = [&](float distance) {
            cam.position.copy(target).addScaledVector(dir, distance);
            cam.lookAt(target);
            cam.updateMatrixWorld();
        };
        float distance = 5;
        for (int pass = 0; pass < 4; ++pass) {
            float lo = 0.3f, hi = 60.f;
            for (int i = 0; i < 36; ++i) {
                const float mid = 0.5f * (lo + hi);
                place(mid);
                bool inside = true;
                for (auto p : points) {
                    p.project(cam);
                    if (std::abs(p.x) > 1 - margin || std::abs(p.y) > 1 - margin || p.z > 1) {
                        inside = false;
                        break;
                    }
                }
                (inside ? hi : lo) = mid;
            }
            distance = hi;
            // move the target so that what is seen sits in the middle of the frame
            place(distance);
            float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
            for (auto p : points) {
                p.project(cam);
                x0 = std::min(x0, p.x), x1 = std::max(x1, p.x), y0 = std::min(y0, p.y), y1 = std::max(y1, p.y);
            }
            const float halfHeight = std::tan(fov * math::DEG2RAD / 2) * distance;
            Vector3 right, up, forward;
            cam.matrixWorld->extractBasis(right, up, forward);
            target.addScaledVector(right, 0.5f * (x0 + x1) * halfHeight * aspect).addScaledVector(up, 0.5f * (y0 + y1) * halfHeight);
        }
        place(distance);
        return {name, cam.position, target, fov};
    }

    std::vector<Vector3> corners(const Box3& b) {
        std::vector<Vector3> out;
        for (int i = 0; i < 8; ++i)
            out.emplace_back((i & 1) ? b.max().x : b.min().x, (i & 2) ? b.max().y : b.min().y, (i & 4) ? b.max().z : b.min().z);
        return out;
    }

    // Low, from the south-west, about the height of the nave's eaves, the west
    // front on the lit side (the film's hook ends here).
    View heroView(const Layout& l, float aspect) {
        const Vector3 target(l.centre.x - 0.2f, l.box.max().y * 0.40f, l.centre.z);
        return frame("hero", target, Vector3(-0.76f, -0.02f, 0.65f), 30, aspect, l.points, 0.05f);
    }

    std::vector<View> standardViews(const Layout& l, float aspect) {
        std::vector<View> views;
        views.push_back(heroView(l, aspect));
        // the west front, square on, from a little below the eaves
        {
            const Vector3 target(0, l.towerHeight * 0.46f, 0);
            std::vector<Vector3> points;
            for (const auto& p : l.points)
                if (p.x < l.towerHalf + 0.1f) points.push_back(p);
            views.push_back(frame("west_front", target, Vector3(-1, -0.04f, 0.05f), 30, aspect, points, 0.04f));
        }
        // the south-west tower's top: its belfry, parapet and spire, close up
        {
            std::vector<Vector3> points;
            for (const auto& p : l.points)
                if (std::abs(p.x - l.tower.x) < l.towerHalf + 0.02f && std::abs(p.z - l.tower.z) < l.towerHalf + 0.02f && p.y > l.shaftTop - 0.22f) points.push_back(p);
            const Vector3 target(l.tower.x, 0.5f * (l.shaftTop - 0.22f + l.towerHeight), l.tower.z);
            views.push_back(frame("tower_top", target, Vector3(-0.62f, -0.10f, 0.78f), 30, aspect, points, 0.05f));
        }
        // inside the nave, low, looking east along it to the crossing
        views.push_back({"nave_inside", {l.naveX0 + 0.32f, 0.032f, 0.008f}, {l.crossingX, l.eaves * 0.52f, 0.f}, 52});
        // from high up, three quarters from the south-east, looking down at about 38 degrees
        views.push_back(frame("aerial", l.centre, Vector3(0.55f * 0.79f, 0.62f, 0.83f * 0.79f), 30, aspect, l.points, 0.05f));
        return views;
    }

    // The wrecking ball: a pendulum whose lowest point is beside the south-west
    // tower's axis, at the height of its first courses.
    struct Swing {
        Vector3 pivot, start, bottom;
        float radius = 0.1f;
        float rope = 6.f;
    };

    Swing swingFor(const Layout& l, float speed, float radius, float rope, Vector3 aim, float height, float offset) {
        Swing s;
        s.radius = radius;
        s.rope = rope;
        aim.y = 0;
        aim.normalize();
        const Vector3 side(-aim.z, 0, aim.x);
        s.bottom.set(l.tower.x + side.x * offset, height, l.tower.z + side.z * offset);
        s.pivot.set(s.bottom.x, height + rope, s.bottom.z);
        // let go from where it falls far enough to pass the bottom at `speed`
        const float drop = speed * speed / (2 * 9.81f);
        const float angle = std::acos(std::clamp(1 - drop / rope, -1.f, 1.f));
        s.start = Vector3(s.pivot).addScaledVector(aim, -rope * std::sin(angle));
        s.start.y = s.pivot.y - rope * std::cos(angle);
        return s;
    }

    // When the swinging ball first reaches the tower: the pendulum on its own,
    // stepped finely, until its centre is a radius from the tower's outside.
    float predictHit(const Swing& s, const Layout& l) {
        const float g = 9.81f;
        Vector3 out = Vector3(s.start).sub(s.pivot);
        float theta = std::atan2(std::sqrt(out.x * out.x + out.z * out.z), -out.y), omega = 0, t = 0;
        Vector3 back = out;
        back.y = 0;
        back.normalize();
        const float h = 1e-4f;
        for (int i = 0; i < 200000; ++i) {
            const Vector3 at = Vector3(s.pivot).addScaledVector(back, s.rope * std::sin(theta)).add(Vector3(0, -s.rope * std::cos(theta), 0));
            const float dx = at.x - l.tower.x, dz = at.z - l.tower.z;
            if (std::sqrt(dx * dx + dz * dz) <= l.towerReach + s.radius) return t;
            omega += -g / s.rope * std::sin(theta) * h;
            theta += omega * h;
            t += h;
        }
        return t;
    }

    // Wide, from the south-south-west, with room for the ball swinging in.
    View probeView(const Layout& l, const Swing& s, float aspect) {
        std::vector<Vector3> points = l.points;
        Vector3 in = Vector3(s.start).sub(s.bottom);
        in.y = 0;
        in.normalize();
        points.push_back(Vector3(s.bottom).addScaledVector(in, 1.1f).add(Vector3(0, 0.15f, 0)));
        const Vector3 target(l.centre.x - 0.3f, l.box.max().y * 0.38f, l.centre.z + 0.15f);
        return frame("collapse", target, Vector3(-0.50f, 0.02f, 0.87f), 32, aspect, points, 0.04f);
    }

    // ------------------------------------------------------------ the film's plan

    // What one frame of the film shows: which building, how far it has risen
    // or where the recording is, and the camera. Plain floats, so that the
    // chunks' stamps can hash it as it is.
    struct FrameState {
        std::int32_t shot = 0;
        std::int32_t scene = 0;// 0: the film's building (the recording's), 1: the next seed's
        std::int32_t mode = 0; // 0: standing, 1: rising, 2: the recording
        float t = 0;           // rising: seconds into the build; the recording: seconds from the ball's release
        float eye[3]{}, target[3]{};
        float fov = 30;
        float ao = 1;          // how much of the ambient occlusion is laid on: less in the close-up, where it would smudge sunlit planks
        float shadowRadius = 2.5f;// the key's shadow filter, in texels: tighter close up, where its grain would show

        [[nodiscard]] bool same(const FrameState& o) const { return std::memcmp(this, &o, sizeof(FrameState)) == 0; }
    };
    static_assert(sizeof(FrameState) == 52, "FrameState is hashed as raw bytes: no padding");

    // The light, for the film and the stills alike: the late sun low through
    // the hall's north windows from the north-west, so that the west front is
    // lit and the building stands in a beam; the room's warm light in the
    // shadows. All of it is in the film's chunks' stamps.
    struct FilmLook {
        float keyIntensity = 7.0f;    // the sun
        float sunAzimuthDeg = 48.f;   // west of north: it comes through the north wall's windows
        float sunElevationDeg = 24.f;
        float beamX = 0.5f;           // the window's beam is centred on the ray from this far east of the west towers
        float fillIntensity = 0.3f;   // the sunlit floor's warm bounce from the side away from the sun, in the environment
        float bounce = 0.18f;         // the floor's warm bounce from below
        float environment = 0.68f;    // the room's light (Hall.hpp's hallEnvironment): mostly the sky through the windows
        std::uint32_t fog = 0x161617; // the far walls fade into dusk, near neutral
        float fogNear = 7.f, fogFar = 34.f;
    };

    struct FilmInputs {
        const Layout* a = nullptr;
        const Layout* b = nullptr;
        float hit = 1.2f;       // seconds from the release to the hit
        Vector3 ballStart, ballBottom, impact;
        std::vector<Vector3> rubble;// where the planks lie at the end of the real-time collapse
        std::function<Vector3(float)> ballAt;// the recorded ball's centre, seconds from its release
        float aspect = 16.f / 9.f;
        const Hall* hall = nullptr;// every camera stays inside it
    };

    // How far from `from` along `dir` the eye can go and stay `margin` inside the hall's walls and ceiling.
    float roomAlong(const Hall& h, const Vector3& from, Vector3 dir, float margin) {
        dir.normalize();
        float t = 1e9f;
        const auto wall = [&](float p, float d, float lo, float hi) {
            if (d > 1e-6f) t = std::min(t, (hi - margin - p) / d);
            if (d < -1e-6f) t = std::min(t, (lo + margin - p) / d);
        };
        wall(from.x, dir.x, h.x0, h.x1);
        wall(from.y, dir.y, 0.f, h.height);
        wall(from.z, dir.z, h.zN, h.zS);
        return std::max(0.f, t);
    }

    // The hall from its south-west corner, high: the window wall along the left, its patches of sun across the
    // floor, the cathedral in the middle of the room.
    View hallView(const Hall& h, const Layout& l) {
        return {"hall", Vector3(h.x0 + 5.5f, 3.0f, h.zS - 1.4f), Vector3(l.centre.x + 0.8f, 2.2f, l.centre.z - 1.2f), 40};
    }

    // A view along `dir` in which the points span `fraction` of the frame's height, centred; its sides may
    // spill over (`spill` says how far: the widest |x| in normalized device coordinates).
    View fitHeight(const std::string& name, Vector3 target, Vector3 dir, float fov, float aspect, const std::vector<Vector3>& points, float fraction,
                   float* spill = nullptr) {
        dir.normalize();
        PerspectiveCamera cam(fov, aspect, 0.01f, 200);
        const auto place = [&](float distance) {
            cam.position.copy(target).addScaledVector(dir, distance);
            cam.lookAt(target);
            cam.updateMatrixWorld();
        };
        const auto extent = [&](float& x0, float& x1, float& y0, float& y1) {
            x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
            for (auto q : points) {
                q.project(cam);
                x0 = std::min(x0, q.x), x1 = std::max(x1, q.x), y0 = std::min(y0, q.y), y1 = std::max(y1, q.y);
            }
        };
        float distance = 5;
        for (int pass = 0; pass < 4; ++pass) {
            float lo = 0.3f, hi = 60.f;
            for (int i = 0; i < 36; ++i) {
                const float mid = 0.5f * (lo + hi);
                place(mid);
                float x0, x1, y0, y1;
                extent(x0, x1, y0, y1);
                (y1 - y0 <= 2 * fraction ? hi : lo) = mid;
            }
            distance = hi;
            place(distance);
            float x0, x1, y0, y1;
            extent(x0, x1, y0, y1);
            const float halfHeight = std::tan(fov * math::DEG2RAD / 2) * distance;
            Vector3 right, up, forward;
            cam.matrixWorld->extractBasis(right, up, forward);
            target.addScaledVector(right, 0.5f * (x0 + x1) * halfHeight * aspect).addScaledVector(up, 0.5f * (y0 + y1) * halfHeight);
        }
        place(distance);
        if (spill) {
            float x0, x1, y0, y1;
            extent(x0, x1, y0, y1);
            *spill = std::max(std::abs(x0), std::abs(x1));
        }
        return {name, cam.position, target, fov};
    }

    struct FilmPlan {
        std::vector<film::Shot> shots;
        std::vector<FrameState> frames;
        double slowHit = 0;// film seconds: when the ball hits in the slow motion
        double end = 0;
        float buildA = 8.8f, buildB = 2.6f;// seconds each building takes to rise
        float ballInSlow = 0;  // seconds of the recording the ball is in the slow motion's frame before it hits
        float ballInWide = 0;  // seconds of film the ball is in the wide frame before it hits
        float nextSpill = 0;   // how far the next seed's building reaches sideways in its frame (1: the frame's edge)
        View wide;
    };

    // A camera moving from a to b (u from 0 to 1).
    struct Move {
        Vector3 eye0, eye1, target0, target1;
        float fov0 = 30, fov1 = 30;

        void at(float u, FrameState& s) const {
            const Vector3 e = Vector3(eye0).lerp(eye1, u), t = Vector3(target0).lerp(target1, u);
            s.eye[0] = e.x, s.eye[1] = e.y, s.eye[2] = e.z;
            s.target[0] = t.x, s.target[1] = t.y, s.target[2] = t.z;
            s.fov = fov0 + (fov1 - fov0) * u;
        }
    };

    Vector3 horizontal(Vector3 v) {
        v.y = 0;
        return v.normalize();
    }

    // The film, shot by shot (about 46 s at 60 frames a second):
    //  1 the finished cathedral, low, three quarters on its lit side, a slow dolly in
    //  2 a slow move up the south-west tower to its spire, the sun raking across it
    //  3 it rises from an empty floor, course by course; a high camera turns slowly
    //  4 the ball swings in on its rope from the left (wide)
    //  5 the collapse in real time, the same camera
    //  6 the hit and the first seconds after it again, four times slower, low and near
    //  7 an empty floor; the next seed's cathedral rises fast and holds
    //  8 the end card over the last frame (laid over by ffmpeg)
    FilmPlan planFilm(const FilmInputs& in) {
        FilmPlan p;
        const Layout& A = *in.a;
        const Layout& B = *in.b;
        const float aspect = in.aspect;
        // every plank's centre: a spire's last few planks decide where the frame's top is
        const auto& ptsA = A.points;
        const auto& ptsB = B.points;

        // 6: low, beside the ball's path, near where it hits; it starts a moment (0.3 s of film) before the
        // ball comes into its frame
        const Vector3 inDir = horizontal(Vector3(in.ballStart).sub(in.ballBottom));
        const Vector3 aim = Vector3(inDir).negate();
        const Vector3 side(-aim.z, 0, aim.x);
        Vector3 slowEye = Vector3(in.impact).addScaledVector(side, 1.4f).addScaledVector(aim, -0.2f);
        slowEye.y = 0.10f;
        Vector3 slowTarget = Vector3(in.impact).addScaledVector(aim, 0.25f);
        slowTarget.y = 0.28f;
        Move m6{slowEye, Vector3(slowEye).lerp(slowTarget, 0.05f), slowTarget, slowTarget, 42, 42};
        float enters = in.hit;
        {
            PerspectiveCamera cam(42, aspect, 0.01f, 100);
            cam.position.copy(slowEye);
            cam.lookAt(slowTarget);
            cam.updateMatrixWorld();
            for (float t = std::max(0.f, in.hit - 1.5f); t <= in.hit; t += 1.f / 240) {
                Vector3 b = in.ballAt(t);
                b.project(cam);
                if (std::abs(b.x) < 1.04f && std::abs(b.y) < 1.04f && b.z < 1) {
                    enters = t;
                    break;
                }
            }
        }
        p.ballInSlow = in.hit - enters;

        const double hold = 0.25;                                        // shot 4 opens on the ball held back, then let go
        const double slowBefore = p.ballInSlow + 0.3 / 4, slowAfter = 2.2;// the slow motion: seconds of the recording round the hit
        const double lengths[8] = {4.5, 4.5, 10.0, hold + in.hit, 6.0, (slowBefore + slowAfter) * 4, 5.0, 4.0};
        const char* names[8] = {"hook", "tower", "build", "swing", "collapse", "slow motion", "next seed", "end card"};
        double t = 0;
        for (int i = 0; i < 8; ++i) {
            p.shots.push_back({i + 1, names[i], t, t + lengths[i]});
            t += lengths[i];
        }
        p.end = t;
        p.slowHit = p.shots[5].t0 + slowBefore * 4;

        // 1: low, three quarters from the south-west, the west front on the lit side: it opens wide and a little
        // higher on the cathedral in its room (the windows along the left, the sun's patches on the floor) and
        // pushes in, slowing, to the building filling the frame
        const Vector3 hookDir = Vector3(-0.76f, -0.02f, 0.65f).normalize();
        const View hook = frame("hook", Vector3(A.centre.x - 0.2f, A.box.max().y * 0.40f, A.centre.z), hookDir, 30, aspect, ptsA, 0.05f);
        Vector3 hookOut = Vector3(hook.eye).sub(hook.target);
        const float hookNear = hookOut.length();
        hookOut.add(Vector3(0, 0.12f * hookNear, 0)).normalize();
        const float hookFar = std::min(1.6f * hookNear, in.hall ? roomAlong(*in.hall, hook.target, hookOut, 0.9f) : 1.6f * hookNear);
        const Vector3 hookEye0 = Vector3(hook.target).addScaledVector(hookOut, hookFar);
        // looking a little up and a little wider at first, so that the high windows' arches are in the frame
        const Vector3 hookTarget0(hook.target.x, hookEye0.y + 0.05f * hookFar, hook.target.z);
        Move m1{hookEye0, hook.eye, hookTarget0, hook.target, 38, 30};

        // 2: the south-west tower from the south-west, the sun raking across it (its west faces lit, its south face
        // in shade): from its belfry tier the camera rises and tilts until its spire stands against the window wall
        // in shade, the north-west spire to its left and the crossing spire to its right (the tower_top still)
        std::vector<Vector3> spirePts;
        for (const auto& q : ptsA)
            if (std::abs(q.x - A.tower.x) < A.towerHalf + 0.02f && std::abs(q.z - A.tower.z) < A.towerHalf + 0.02f && q.y > A.shaftTop - 0.22f) spirePts.push_back(q);
        const View spire = frame("spire", Vector3(A.tower.x, 0.5f * (A.shaftTop - 0.22f + A.towerHeight), A.tower.z), Vector3(-0.62f, -0.10f, 0.78f), 30, aspect, spirePts, 0.05f);
        const float belfry = A.shaftTop - 0.16f;
        Move m2{Vector3(spire.eye.x, belfry, spire.eye.z).sub(Vector3(spire.target.x, belfry, spire.target.z)).multiplyScalar(1.06f).add(Vector3(spire.target.x, belfry, spire.target.z)),
                spire.eye, Vector3(spire.target.x, belfry + 0.02f, spire.target.z), spire.target, 30, 30};

        // 3: high three quarters, turning slowly round the building
        const auto orbitDir = [](float azimuthDeg, float elevationDeg) {
            const float a = azimuthDeg * math::DEG2RAD, e = elevationDeg * math::DEG2RAD;
            return Vector3(std::cos(e) * std::cos(a), std::sin(e), std::cos(e) * std::sin(a));
        };
        // it turns away from the rim light's side, so the floor never shows that light's sheen
        const float az0 = std::atan2(hookDir.z, hookDir.x) * math::RAD2DEG + 10.f, az1 = az0 - 24.f, el = 31.f;
        // and pulling back as the building rises: framed at twelve moments of the build on what stands half a
        // second later (and the footprint), the distance and aim between those eased, so that the building fills
        // the frame and its spires never rise out of it
        const auto rise = film::makeBuild(ptsA.size(), [&](size_t i) { return static_cast<int>(std::lround(ptsA[i].y / THICK - 0.5f)); }, p.buildA, 0.4f, 0.025f);
        struct Key {
            float t, r;
            Vector3 target;
        };
        std::vector<Key> keys;
        {
            std::vector<Vector3> footprint;
            for (int i = 0; i < 4; ++i) footprint.emplace_back((i & 1) ? A.box.max().x : A.box.min().x, 0.f, (i & 2) ? A.box.max().z : A.box.min().z);
            for (int k = 1; k <= 12; ++k) {
                const float tk = p.buildA * static_cast<float>(k) / 12.f;
                std::vector<Vector3> standing = footprint;
                for (size_t i = 0; i < ptsA.size(); ++i)
                    if (rise.start[i] <= tk + 0.5f) standing.push_back(ptsA[i]);
                const float az = az0 + (az1 - az0) * tk / static_cast<float>(lengths[2]);
                const View v = frame("build", A.centre, orbitDir(az, el), 30, aspect, standing, 0.08f);
                keys.push_back({tk, v.eye.distanceTo(v.target), v.target});
            }
        }
        const int f3a = static_cast<int>(std::ceil(p.shots[2].t0 * film::fps - 1e-6)), f3b = static_cast<int>(std::ceil(p.shots[2].t1 * film::fps - 1e-6));
        std::vector<float> r3(f3b - f3a);
        std::vector<Vector3> t3(f3b - f3a);
        for (int f = f3a; f < f3b; ++f) {
            const float t = static_cast<float>(static_cast<double>(f) / film::fps - p.shots[2].t0);
            auto& r = r3[f - f3a];
            auto& aim = t3[f - f3a];
            r = keys.back().r, aim = keys.back().target;
            if (t <= keys.front().t) r = keys.front().r, aim = keys.front().target;
            else
                for (size_t k = 1; k < keys.size(); ++k)
                    if (t <= keys[k].t) {
                        const float w = (t - keys[k - 1].t) / (keys[k].t - keys[k - 1].t);
                        r = keys[k - 1].r + (keys[k].r - keys[k - 1].r) * w;
                        aim = Vector3(keys[k - 1].target).lerp(keys[k].target, w);
                        break;
                    }
        }
        // eased: three passes of a moving average over half a second
        for (int pass = 0; pass < 3; ++pass) {
            const auto r = r3;
            const auto aim = t3;
            const int n = static_cast<int>(r.size()), half = 15;
            for (int i = 0; i < n; ++i) {
                float sum = 0;
                Vector3 at;
                for (int j = i - half; j <= i + half; ++j) {
                    const int k = std::clamp(j, 0, n - 1);
                    sum += r[k];
                    at.add(aim[k]);
                }
                r3[i] = sum / (2 * half + 1);
                t3[i] = at.multiplyScalar(1.f / (2 * half + 1));
            }
        }

        // 4, 5: wide from the west-south-west, behind the ball's swing: the ball and its rope come in from the left
        // foreground for the whole swing, the west front is lit, and the rubble as it lies at the end of the
        // shot has room (the few planks that flew furthest left out)
        std::vector<Vector3> ptsW = ptsA;
        {
            std::vector<std::pair<float, Vector3>> far;
            for (const auto& r : in.rubble) far.emplace_back(A.box.distanceToPoint(r), r);
            std::sort(far.begin(), far.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
            const size_t keep = far.size() - far.size() / 300;
            for (size_t i = 0; i < keep; ++i) ptsW.push_back(far[i].second);
        }
        for (float t = 0; t <= in.hit; t += 0.05f) {
            const Vector3 b = in.ballAt(t);
            ptsW.push_back(Vector3(b).add(Vector3(0, 0.12f, 0)));
            ptsW.push_back(Vector3(b).add(Vector3(0, -0.12f, 0)));
        }
        View wide = frame("wide", Vector3(A.centre.x - 0.3f, A.box.max().y * 0.38f, A.centre.z + 0.15f), Vector3(-0.85f, 0.06f, 0.53f), 32, aspect, ptsW, 0.07f);
        // the ball in frame from the moment it is let go: back off until it is
        for (int i = 0; i < 60; ++i) {
            PerspectiveCamera cam(wide.fov, aspect, 0.01f, 100);
            cam.position.copy(wide.eye);
            cam.lookAt(wide.target);
            cam.updateMatrixWorld();
            Vector3 b = in.ballAt(0);
            b.project(cam);
            if (std::abs(b.x) < 0.9f && std::abs(b.y) < 0.9f && b.z < 1) break;
            wide.eye.sub(wide.target).multiplyScalar(1.02f).add(wide.target);
        }
        p.wide = wide;
        {
            PerspectiveCamera cam(wide.fov, aspect, 0.01f, 100);
            cam.position.copy(wide.eye);
            cam.lookAt(wide.target);
            cam.updateMatrixWorld();
            for (float t = 0; t <= in.hit; t += 1.f / 240) {
                Vector3 b = in.ballAt(t);
                b.project(cam);
                if (std::abs(b.x) < 1.f && std::abs(b.y) < 1.f && b.z < 1) {
                    p.ballInWide = in.hit - t + (t == 0 ? static_cast<float>(hold) : 0.f);
                    break;
                }
            }
        }

        // 7, 8: the next seed's building from its sunlit north side and close, the hall's sunny floor and its plain
        // south wall behind it: standing, it fills most of the frame's height; a slow push, then still
        // (of the directions from the north-west round to the west-north-west, the one that shows most of its
        // length and still keeps it inside the frame's sides)
        View next;
        for (float az = -166.f; az <= -130.f; az += 4.f) {
            float spill = 0;
            const Vector3 dir(std::cos(az * math::DEG2RAD), -0.02f, std::sin(az * math::DEG2RAD));
            const View v = fitHeight("next", Vector3(B.centre.x - 0.2f, B.box.max().y * 0.45f, B.centre.z), dir, 30, aspect, ptsB, 0.80f, &spill);
            if (spill <= 0.92f && spill > p.nextSpill) next = v, p.nextSpill = spill;
        }
        Move m7{Vector3(next.target).addScaledVector(Vector3(next.eye).sub(next.target), 1.10f), next.eye, next.target, next.target, 30, 30};

        const int frames = static_cast<int>(std::lround(p.end * film::fps));
        p.frames.resize(frames);
        size_t shot = 0;
        for (int f = 0; f < frames; ++f) {
            const double tf = static_cast<double>(f) / film::fps;
            while (shot + 1 < p.shots.size() && tf >= p.shots[shot].t1) ++shot;
            const auto& sh = p.shots[shot];
            const float lt = static_cast<float>(tf - sh.t0);
            const float u = static_cast<float>((tf - sh.t0) / (sh.t1 - sh.t0));
            FrameState& s = p.frames[f];
            s.shot = sh.number;
            s.ao = sh.number == 2 ? 0.45f : 1.f;
            s.shadowRadius = sh.number == 2 || sh.number == 6 ? 1.f : 2.5f;
            switch (sh.number) {
                case 1:
                    s.scene = 0, s.mode = 0;
                    m1.at(0.3f * u + 0.7f * u * (2 - u), s);// slowing as it arrives
                    break;
                case 2:
                    s.scene = 0, s.mode = 0;
                    m2.at(film::smooth(u) * 0.75f + u * 0.25f, s);
                    break;
                case 3: {
                    s.scene = 0, s.mode = 1;
                    s.t = lt;
                    const int k = std::clamp(f - f3a, 0, f3b - f3a - 1);
                    const float az = az0 + (az1 - az0) * u;
                    const Vector3 eye = Vector3(t3[k]).addScaledVector(orbitDir(az, el), r3[k]);
                    Move{eye, {}, t3[k], {}, 30, 30}.at(0, s);
                    break;
                }
                case 4:
                case 5:
                    s.scene = 0, s.mode = 2;
                    s.t = sh.number == 4 ? std::max(0.f, lt - static_cast<float>(hold)) : in.hit + lt;
                    Move{wide.eye, {}, wide.target, {}, wide.fov, wide.fov}.at(0, s);
                    break;
                case 6:
                    s.scene = 0, s.mode = 2;
                    s.t = in.hit - static_cast<float>(slowBefore) + lt / 4.f;
                    m6.at(u, s);
                    break;
                case 7:
                    s.scene = 1, s.mode = 1;
                    s.t = lt - 0.35f;
                    m7.at(u, s);
                    break;
                default:
                    s.scene = 1, s.mode = 1;
                    s.t = 1e3f;// risen
                    m7.at(1, s);
                    break;
            }
        }
        return p;
    }

    // Step timing for the report.
    struct StepStats {
        double asleepMs = 0;
        int asleepSteps = 0;
        double peakMs = 0;
        int peakAwake = 0;
        float peakTime = 0;
        int maxAwake = 0;
        int steps = 0;

        void add(double ms, int awake, float t, bool beforeHit) {
            ++steps;
            if (beforeHit) {
                asleepMs += ms;
                ++asleepSteps;
            } else if (ms > peakMs) {
                peakMs = ms;
                peakAwake = awake;
                peakTime = t;
            }
            maxAwake = std::max(maxAwake, awake);
        }

        void print(float hitAt) const {
            std::printf("physics: %d steps; before the hit %.3f ms/step; collapse peak %.2f ms/step at %.2f s after the hit with %d planks awake (most awake at once: %d)\n",
                        steps, asleepSteps ? asleepMs / asleepSteps : 0.0, peakMs, peakTime - hitAt, peakAwake, maxAwake);
        }
    };

    // A recorded collapse: a small header, then one frame after another, each
    // its time from the release of the ball, the ball's pose and every plank's,
    // as raw little-endian floats: x y z, then the rotation qx qy qz qw.
    struct RecordHeader {
        char magic[8] = {'P', 'L', 'A', 'N', 'K', 'R', 'E', 'C'};
        std::uint32_t version = 1;
        std::uint32_t planks = 0;
        std::uint64_t seed = 1;
        std::int32_t design = 0;
        float scale = 1;
        float fps = 60;
        std::uint32_t frames = 0;
        float hitTime = -1;// seconds from the release, -1 if the ball hit nothing
        float ballRadius = 0.1f;
        float pivot[3] = {0, 0, 0};
        float rope = 6;// from the pivot to the ball's centre
    };

    bool seekTo(std::FILE* f, std::uint64_t offset) {
#ifdef _WIN32
        return _fseeki64(f, static_cast<long long>(offset), SEEK_SET) == 0;
#else
        return fseeko(f, static_cast<off_t>(offset), SEEK_SET) == 0;
#endif
    }

    bool seekToEnd(std::FILE* f) {
#ifdef _WIN32
        return _fseeki64(f, 0, SEEK_END) == 0;
#else
        return fseeko(f, 0, SEEK_END) == 0;
#endif
    }

    class Replay {
    public:
        RecordHeader header;

        ~Replay() {
            if (f_) std::fclose(f_);
        }

        bool open(const std::string& path) {
            f_ = std::fopen(path.c_str(), "rb");
            if (!f_) return false;
            if (std::fread(&header, sizeof(header), 1, f_) != 1) return false;
            if (std::string(header.magic, 8) != "PLANKREC" || header.frames == 0) return false;
            floats_ = 8 + 7 * static_cast<std::size_t>(header.planks);
            return true;
        }

        [[nodiscard]] float duration() const { return static_cast<float>(header.frames - 1) / header.fps; }

        // the poses at time t (from the release), between the two nearest frames
        void at(float t, std::vector<float>& planks, float ball[7]) {
            const float x = std::clamp(t * header.fps, 0.f, static_cast<float>(header.frames - 1));
            const int i0 = static_cast<int>(std::floor(x));
            const int i1 = std::min(i0 + 1, static_cast<int>(header.frames) - 1);
            const float w = x - static_cast<float>(i0);
            load(i0, a_, loadedA_);
            load(i1, b_, loadedB_);
            planks.resize(7 * static_cast<std::size_t>(header.planks));
            const auto blend = [&](const float* p, const float* q, float* out) {
                for (int k = 0; k < 3; ++k) out[k] = p[k] + (q[k] - p[k]) * w;
                const float sign = (p[3] * q[3] + p[4] * q[4] + p[5] * q[5] + p[6] * q[6]) < 0 ? -1.f : 1.f;
                float n = 0;
                for (int k = 3; k < 7; ++k) out[k] = p[k] + (sign * q[k] - p[k]) * w, n += out[k] * out[k];
                n = n > 0 ? 1.f / std::sqrt(n) : 1.f;
                for (int k = 3; k < 7; ++k) out[k] *= n;
            };
            blend(a_.data() + 1, b_.data() + 1, ball);
            for (std::size_t i = 0; i < header.planks; ++i) blend(a_.data() + 8 + 7 * i, b_.data() + 8 + 7 * i, planks.data() + 7 * i);
        }

        // the ball's centre alone at time t: eight floats of the two nearest frames
        Vector3 ballAt(float t) {
            const float x = std::clamp(t * header.fps, 0.f, static_cast<float>(header.frames - 1));
            const int i0 = static_cast<int>(std::floor(x));
            const int i1 = std::min(i0 + 1, static_cast<int>(header.frames) - 1);
            const float w = x - static_cast<float>(i0);
            float a[8], b[8];
            head(i0, a);
            head(i1, b);
            return {a[1] + (b[1] - a[1]) * w, a[2] + (b[2] - a[2]) * w, a[3] + (b[3] - a[3]) * w};
        }

    private:
        std::FILE* f_ = nullptr;
        std::size_t floats_ = 0;

        void head(int frame, float* out) {
            seekTo(f_, sizeof(RecordHeader) + static_cast<std::uint64_t>(frame) * floats_ * sizeof(float));
            if (std::fread(out, sizeof(float), 8, f_) != 8) std::fill(out, out + 8, 0.f);
        }
        std::vector<float> a_, b_;
        int loadedA_ = -1, loadedB_ = -1;

        void load(int frame, std::vector<float>& out, int& loaded) {
            if (loaded == frame) return;
            // the other buffer may hold it already
            if (&out == &a_ && loadedB_ == frame) {
                a_ = b_;
                loaded = frame;
                return;
            }
            out.resize(floats_);
            seekTo(f_, sizeof(RecordHeader) + static_cast<std::uint64_t>(frame) * floats_ * sizeof(float));
            if (std::fread(out.data(), sizeof(float), floats_, f_) != floats_) std::fill(out.begin(), out.end(), 0.f);
            loaded = frame;
        }
    };

    // The recording, for the film's stamps: its header, its size and every 60th frame.
    std::string hashRecording(const std::string& path) {
        film::Hash h;
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return "none";
        RecordHeader header;
        if (std::fread(&header, sizeof(header), 1, f) == 1) {
            h.value(header);
            const std::size_t floats = 8 + 7 * static_cast<std::size_t>(header.planks);
            std::vector<float> frame(floats);
            for (std::uint32_t i = 0; i < header.frames; i += 60) {
                seekTo(f, sizeof(RecordHeader) + static_cast<std::uint64_t>(i) * floats * sizeof(float));
                if (std::fread(frame.data(), sizeof(float), floats, f) != floats) break;
                h.bytes(frame.data(), floats * sizeof(float));
            }
        }
        std::fclose(f);
        std::error_code ec;
        h.value(static_cast<std::uint64_t>(std::filesystem::file_size(path, ec)));
        return h.hex();
    }

    std::string hashDesign(const Design& d) {
        film::Hash h;
        for (const auto& p : d.planks) h.value(p);
        return h.hex();
    }

    // A rope from the pivot to the ball: straight while it is taut, hanging in
    // a curve as long as the rope when the ball is closer than that, never
    // through the floor.
    std::shared_ptr<BufferGeometry> ropeGeometry(const Vector3& pivot, const Vector3& end, float length) {
        const float span = pivot.distanceTo(end);
        std::vector<Vector3> points;
        const int n = 48;
        const auto curve = [&](float sag, std::vector<Vector3>& out) {
            out.clear();
            for (int i = 0; i <= n; ++i) {
                const float t = static_cast<float>(i) / n;
                Vector3 p = Vector3(pivot).lerp(end, t);
                p.y -= 4 * sag * t * (1 - t);
                p.y = std::max(p.y, 0.006f);
                out.push_back(p);
            }
        };
        const auto arc = [&](const std::vector<Vector3>& pts) {
            float l = 0;
            for (size_t i = 1; i < pts.size(); ++i) l += pts[i].distanceTo(pts[i - 1]);
            return l;
        };
        if (span >= length - 0.002f) {
            curve(0, points);
        } else {
            float lo = 0, hi = length;
            for (int i = 0; i < 40; ++i) {
                const float mid = 0.5f * (lo + hi);
                curve(mid, points);
                (arc(points) < length ? lo : hi) = mid;
            }
            curve(lo, points);
        }
        auto path = std::make_shared<CatmullRomCurve3>(points);
        return TubeGeometry::create(path, 96, 0.006f, 8, false);
    }

}// namespace

int run(int argc, char** argv);

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
        return 1;
    }
}

int run(int argc, char** argv) {

    std::uint64_t seed = 1;
    int firstDesign = 0;
    float scale = 1;
    std::string shotPath, sequenceDir, stillsDir, probeDir, heroPath, recordPath, replayPath;
    float shotTime = 0;
    int width = 1920, height = 1080, fps = 30;
    std::optional<Vector3> camArg, targetArg;
    std::optional<float> fovArg;
    bool ballOn = false, physicsOnly = false;
    float ballSpeed = 4.5f, ballRadius = 0.1f, ropeLength = 6.f, ballHeight = 0.16f, ballOffset = 0.f;
    Vector3 ballAim(0.80f, 0, -0.60f);
    float settleSeconds = 3.f, collapseSeconds = 0;
    float recordFps = 60, recordTime = 7.5f;
    float exposure = 1.18f;
    int ssaa = 4;
    std::string test;
    int testCourses = 100;
    int hz = 960, sceneHz = 960, sceneIters = 24;
    bool explain = false;
    std::string dumpPath, dumpFinalPath;
    // the film
    std::string filmDir, ffmpegPath, clipName;
    std::uint64_t nextSeed = 3;
    int nextDesign = 0, filmChunk = -1;
    float filmBudget = 240, clipFrom = 0, clipTo = 0;
    int clipShotFrom = 0, clipShotTo = 0;// --film-clip-shots: the clip's first and last shot
    bool filmFinish = false, filmSheet = false, filmFresh = false;
    std::string filmStillsDir;
    FilmLook look;
    if (const char* env = std::getenv("FFMPEG")) ffmpegPath = env;
    unsigned threads = std::max(2u, std::thread::hardware_concurrency() > 4 ? std::thread::hardware_concurrency() - 4 : 2u);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        if (a == "--film" && more) filmDir = argv[++i];
        else if (a == "--ffmpeg" && more) ffmpegPath = argv[++i];
        else if (a == "--next-seed" && more) nextSeed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--next-design" && more) nextDesign = std::max(0, std::atoi(argv[++i]));
        else if (a == "--film-budget" && more) filmBudget = std::stof(argv[++i]);
        else if (a == "--film-chunk" && more) filmChunk = std::atoi(argv[++i]);
        else if (a == "--film-finish") filmFinish = true;
        else if (a == "--film-sheet") filmSheet = true;
        else if (a == "--film-fresh") filmFresh = true;
        else if (a == "--film-stills" && more) filmStillsDir = argv[++i];
        else if (a == "--sun" && i + 2 < argc) {
            look.sunAzimuthDeg = std::stof(argv[i + 1]);
            look.sunElevationDeg = std::stof(argv[i + 2]);
            i += 2;
        }
        else if (a == "--beam-x" && more) look.beamX = std::stof(argv[++i]);
        else if (a == "--key" && more) look.keyIntensity = std::stof(argv[++i]);
        else if (a == "--room" && more) look.environment = std::stof(argv[++i]);
        else if (a == "--fill" && more) look.fillIntensity = std::stof(argv[++i]);
        else if (a == "--bounce" && more) look.bounce = std::stof(argv[++i]);
        else if (a == "--film-clip" && i + 3 < argc) {
            clipName = argv[i + 1];
            clipFrom = std::stof(argv[i + 2]);
            clipTo = std::stof(argv[i + 3]);
            i += 3;
        }
        else if (a == "--film-clip-shots" && i + 3 < argc) {
            clipName = argv[i + 1];
            clipShotFrom = std::atoi(argv[i + 2]);
            clipShotTo = std::atoi(argv[i + 3]);
            i += 3;
        }
        else if (a == "--seed" && more) seed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--design" && more) firstDesign = std::max(0, std::atoi(argv[++i]));
        else if (a == "--scale" && more) scale = std::stof(argv[++i]);
        else if (a == "--shot" && more) shotPath = argv[++i];
        else if (a == "--hero" && more) heroPath = argv[++i];
        else if (a == "--sequence" && more) sequenceDir = argv[++i];
        else if (a == "--stills" && more) stillsDir = argv[++i];
        else if (a == "--probe" && more) probeDir = argv[++i];
        else if (a == "--record" && more) recordPath = argv[++i];
        else if (a == "--record-fps" && more) recordFps = std::max(1.f, std::stof(argv[++i]));
        else if (a == "--record-time" && more) recordTime = std::stof(argv[++i]);
        else if (a == "--replay" && more) replayPath = argv[++i];
        else if (a == "--time" && more) shotTime = std::stof(argv[++i]);
        else if (a == "--fps" && more) fps = std::max(1, std::atoi(argv[++i]));
        else if (a == "--size" && more) std::sscanf(argv[++i], "%dx%d", &width, &height);
        else if (a == "--fov" && more) fovArg = std::stof(argv[++i]);
        else if (a == "--exposure" && more) exposure = std::stof(argv[++i]);
        else if (a == "--ssaa" && more) ssaa = std::clamp(std::atoi(argv[++i]), 0, 5);
        else if (a == "--ball") ballOn = true;
        else if (a == "--ball-speed" && more) ballSpeed = std::stof(argv[++i]);
        else if (a == "--ball-radius" && more) ballRadius = std::stof(argv[++i]);
        else if (a == "--rope" && more) ropeLength = std::stof(argv[++i]);
        else if (a == "--ball-height" && more) ballHeight = std::stof(argv[++i]);
        else if (a == "--ball-offset" && more) ballOffset = std::stof(argv[++i]);
        else if (a == "--ball-aim" && i + 2 < argc) {
            ballAim.set(std::stof(argv[i + 1]), 0, std::stof(argv[i + 2]));
            i += 2;
        }
        else if (a == "--physics") physicsOnly = true;
        else if (a == "--pgs") tuning.tgs = false;
        else if (a == "--hz" && more) hz = std::max(30, std::atoi(argv[++i]));
        else if (a == "--scene-hz" && more) sceneHz = std::max(30, std::atoi(argv[++i]));
        else if (a == "--scene-iters" && more) sceneIters = std::max(1, std::atoi(argv[++i]));
        else if (a == "--nospec") tuning.speculative = false;
        else if (a == "--nopcm") tuning.pcm = false;
        else if (a == "--explain") explain = true;
        else if (a == "--dump" && more) dumpPath = argv[++i];
        else if (a == "--dump-final" && more) dumpFinalPath = argv[++i];
        else if (a == "--nostab") tuning.stabilization = false;
        else if (a == "--tol" && more) tuning.toleranceLength = std::stof(argv[++i]);
        else if (a == "--offset" && more) tuning.contactOffset = std::stof(argv[++i]);
        else if (a == "--iters" && more) tuning.positionIterations = std::atoi(argv[++i]);
        else if (a == "--test" && i + 2 < argc) {
            test = argv[++i];
            testCourses = std::atoi(argv[++i]);
        }
        else if (a == "--collapse" && more) collapseSeconds = std::stof(argv[++i]);
        else if (a == "--settle" && more) settleSeconds = std::stof(argv[++i]);
        else if (a == "--threads" && more) threads = static_cast<unsigned>(std::max(0, std::atoi(argv[++i])));
        else if ((a == "--cam" || a == "--target") && i + 3 < argc) {
            const Vector3 v(std::stof(argv[i + 1]), std::stof(argv[i + 2]), std::stof(argv[i + 3]));
            (a == "--cam" ? camArg : targetArg) = v;
            i += 3;
        }
    }
    const bool recording = !recordPath.empty();
    const bool replaying = !replayPath.empty();
    Replay replay;
    if (replaying) {
        if (!replay.open(replayPath)) {
            std::printf("cannot read the recording %s\n", replayPath.c_str());
            return 1;
        }
        seed = replay.header.seed;
        firstDesign = replay.header.design;
        scale = replay.header.scale;
        settleSeconds = 0;
        ballOn = true;
        std::printf("replaying %s: seed %llu design %d, %u planks, %u frames at %.0f fps, the ball hit at %.3f s\n", replayPath.c_str(),
                    static_cast<unsigned long long>(seed), firstDesign, replay.header.planks, replay.header.frames, replay.header.fps, replay.header.hitTime);
    }
    const bool filming = !filmDir.empty();
    if (filming && !replaying) {
        std::printf("the film is drawn from a recording of its building: --replay <file>\n");
        return 1;
    }
    if (filming && ffmpegPath.empty()) {
        std::printf("the film needs ffmpeg: --ffmpeg <exe>, or the FFMPEG environment variable\n");
        return 1;
    }
    if (!probeDir.empty() || recording) ballOn = true;
    if (physicsOnly && collapseSeconds > 0) ballOn = true;
    const bool headless = !shotPath.empty() || !sequenceDir.empty() || !stillsDir.empty() || !probeDir.empty() || !heroPath.empty() || filming;
    const float aspect = static_cast<float>(width) / static_cast<float>(height);

    // ------------------------------------------------------------ design + proof

    const float dt = 1.f / static_cast<float>(hz);
    Design design;
    Plan plan{};
    int designIndex = firstDesign;
    std::unique_ptr<Physics> physics;
    for (int attempt = firstDesign; attempt < firstDesign + 3; ++attempt) {
        designIndex = attempt;
        const auto t0 = WallClock::now();
        design = test.empty() ? buildCathedral(seed, attempt, scale, &plan, explain) : buildTest(test, testCourses, explain);
        const double generation = since(t0);
        if (!dumpPath.empty()) {
            std::FILE* f = std::fopen(dumpPath.c_str(), "w");
            for (const auto& p : design.planks) std::fprintf(f, "%d,%d,%.6f,%.6f,%.6f\n", p.element, p.course, p.x, p.z, p.angle);
            std::fclose(f);
        }

        std::map<int, int> perElement;
        for (const auto& p : design.planks) ++perElement[p.element];
        std::printf("seed %llu (design %d): %zu planks, generated in %.1f ms (left out: %d overlapping, %d unsupported)\n",
                    static_cast<unsigned long long>(seed), attempt, design.planks.size(), generation * 1000, design.rejectedOverlap, design.rejectedSupport);
        static const char* crowns[3] = {"spire", "dome", "lantern"};
        if (test.empty())
            std::printf("  %s; nave %.2f m wide, %d bays of %.2f m, eaves at %d courses, ridge at %d, roofs at %.0f deg; %s; towers of %d tiers; crossing %s; %s; %s\n",
                        plan.westTowers == 2 ? "two west towers" : "one west tower", plan.naveInner, plan.bays, plan.bayLength, plan.naveCourses, plan.ridgeCourse, plan.pitch,
                        plan.aisles ? "aisles" : "no aisles", plan.tiers, crowns[plan.crown],
                        plan.armStretchers == 0 ? "no transept" : plan.armStretchers <= 2 ? "a short transept" : "a long transept",
                        plan.roundApse ? "a round apse" : "a flat east end");
        {
            // per element name (the two walls of a pair add up)
            std::map<std::string, std::pair<int, int>> byName;
            for (const auto& [e, n] : perElement) {
                auto& entry = byName[design.elements[e]];
                entry.first += n;
                entry.second += e < static_cast<int>(design.leftOut.size()) ? design.leftOut[e] : 0;
            }
            for (const auto& [name, v] : byName) std::printf("    %-24s %6d  (left out %d)\n", name.c_str(), v.first, v.second);
        }
        if (replaying) break;// no physics at all

        const auto t1 = WallClock::now();
        physics.reset();// PhysX allows one foundation at a time
        physics = std::make_unique<Physics>(design, dt, threads);
        std::printf("  physics built in %.2f s (%u threads)\n", since(t1), threads);
        if (settleSeconds <= 0) break;

        const auto r = physics->settle(settleSeconds);
        const auto nameOf = [&](int i) { return i >= 0 ? design.elements[design.planks[i].element].c_str() : "-"; };
        std::printf("  settled %.1f s awake in %.1f s (%.2f ms/step): in the last 0.5 s at most %.4f mm (%s); in all at most %.3f mm (%s); %d planks moved over 1 mm\n",
                    settleSeconds, r.seconds, r.msPerStep, r.maxLate * 1000, nameOf(r.worstLate), r.maxDisplacement * 1000, nameOf(r.worst), r.movedOverMm);
        if (r.worst >= 0) {
            const auto& w = design.planks[r.worst];
            const auto q = physics->pose(r.worst).p;
            std::printf("  worst plank: course %d at (%.3f, %.3f), moved (%.3f, %.3f, %.3f) mm\n", w.course, w.x, w.z, (q.x - w.x) * 1000, (q.y - courseY(w.course)) * 1000, (q.z - w.z) * 1000);
        }
        // where it moved, element by element: in all, and in the last half second
        const auto moved = physics->displacements();
        const auto late = physics->lateDisplacements();
        std::map<std::string, std::tuple<float, int, float>> worstOf;
        for (size_t i = 0; i < moved.size(); ++i) {
            auto& [m, course, l] = worstOf[design.elements[design.planks[i].element]];
            if (moved[i] >= m) m = moved[i], course = design.planks[i].course;
            l = std::max(l, late[i]);
        }
        for (const auto& [name, m] : worstOf)
            std::printf("    %-24s %8.3f mm (course %d), last 0.5 s %.4f mm\n", name.c_str(), std::get<0>(m) * 1000, std::get<1>(m), std::get<2>(m) * 1000);
        if (r.passed()) {
            std::printf("  design %d PASSED: at rest (%.4f mm in the last 0.5 s) and %.3f mm at most in all\n", attempt, r.maxLate * 1000, r.maxDisplacement * 1000);
            break;
        }
        std::printf("  design %d REJECTED: %s; worst element %s\n", attempt,
                    r.maxDisplacement > 0.003f ? "a plank moved more than 3 mm" : "still moving more than 0.1 mm in the last 0.5 s",
                    r.maxDisplacement > 0.003f ? nameOf(r.worst) : nameOf(r.worstLate));
        if (attempt == firstDesign + 2) std::printf("  no design of this seed passed; keeping the last\n");
    }
    if (physics) {
        physics->sleepAll();
        physics->time = 0;// the scene's clock starts with the cathedral standing
        physics->setRate(1.f / static_cast<float>(sceneHz), sceneIters);
    }
    const float sceneDt = physics ? physics->dt : 1.f / static_cast<float>(sceneHz);
    std::fflush(stdout);

    const Layout layout = layoutOf(design, plan);
    // the hall round the building, which stands in a beam of the low sun
    const Hall hall = makeHall(layout.centre, Vector3(layout.tower.x + look.beamX, 1.f, layout.centre.z), look.sunAzimuthDeg, look.sunElevationDeg);
    Swing swing = swingFor(layout, ballSpeed, ballRadius, ropeLength, ballAim, ballHeight, ballOffset);
    if (replaying) {
        swing.radius = replay.header.ballRadius;
        swing.pivot.set(replay.header.pivot[0], replay.header.pivot[1], replay.header.pivot[2]);
        swing.rope = replay.header.rope;
    }

    // ------------------------------------------------------------ the film: its plan and its words

    Design nextDesignPlanks;
    Plan nextPlan{};
    std::optional<Layout> nextLayout;
    FilmPlan filmPlan;
    std::vector<float> restA;// the film's building standing: the recording's first frame
    float restBall[7]{};
    std::string sceneStamp;
    const std::filesystem::path filmPath = filmDir;
    // a chunk's stamp: everything its pixels depend on
    const auto chunkStamp = [&](int c) {
        film::Hash h;
        h.text(sceneStamp);
        h.value(look);
        h.value(width);
        h.value(height);
        h.value(ssaa);
        h.value(exposure);
        h.value(film::lookCode);
        const size_t f0 = static_cast<size_t>(c) * film::chunkFrames, f1 = std::min(filmPlan.frames.size(), f0 + film::chunkFrames);
        for (size_t f = f0; f < f1; ++f) h.value(filmPlan.frames[f]);
        return h.hex();
    };
    const auto chunkName = [&](int c, const char* ext) {
        char name[32];
        std::snprintf(name, sizeof(name), "chunk_%03d%s", c, ext);
        return filmPath / "chunks" / name;
    };
    const auto chunkDone = [&](int c) {
        return std::filesystem::exists(chunkName(c, ".mkv")) && film::contentsOf(chunkName(c, ".stamp")) == chunkStamp(c);
    };
    // every chunk, all drawn for this plan, joined into the clean master: 0 done, 3 chunks missing, 1 ffmpeg failed
    const auto joinMaster = [&]() -> int {
        const int chunks = static_cast<int>((filmPlan.frames.size() + film::chunkFrames - 1) / film::chunkFrames);
        std::ofstream list(filmPath / "chunks" / "chunks.txt", std::ios::binary);
        int missing = 0;
        for (int c = 0; c < chunks; ++c) {
            if (!chunkDone(c)) ++missing;
            list << "file '" << chunkName(c, ".mkv").filename().string() << "'\n";
        }
        list.close();
        if (missing > 0) {
            std::printf("%d of %d chunks are not drawn for this plan yet (the same --size and --ssaa as the render?)\n", missing, chunks);
            return 3;
        }
        if (film::runCommand(film::inFolder(filmPath / "chunks") + film::q(ffmpegPath) + " -y -loglevel error -f concat -safe 0 -i chunks.txt -c copy " +
                             film::q(filmPath / "master.mkv")) != 0) {
            std::printf("ffmpeg could not join the chunks\n");
            return 1;
        }
        std::printf("wrote the clean master %s\n", (filmPath / "master.mkv").string().c_str());
        return 0;
    };
    if (filming) {
        std::filesystem::create_directories(filmPath / "chunks");
        nextDesignPlanks = buildCathedral(nextSeed, nextDesign, 1, &nextPlan);
        nextLayout = layoutOf(nextDesignPlanks, nextPlan);
        std::printf("the next seed: %llu (design %d), %zu planks\n", static_cast<unsigned long long>(nextSeed), nextDesign, nextDesignPlanks.planks.size());

        replay.at(0, restA, restBall);
        FilmInputs in;
        in.a = &layout;
        in.b = &*nextLayout;
        in.hit = replay.header.hitTime >= 0 ? replay.header.hitTime : 1.2f;
        in.ballStart.set(restBall[0], restBall[1], restBall[2]);
        in.ballBottom.set(swing.pivot.x, swing.pivot.y - swing.rope, swing.pivot.z);
        {
            std::vector<float> planksAtHit;
            float ballAtHit[7];
            replay.at(in.hit, planksAtHit, ballAtHit);
            in.impact.set(ballAtHit[0], ballAtHit[1], ballAtHit[2]);
            replay.at(std::min(in.hit + 6.f, replay.duration()), planksAtHit, ballAtHit);
            for (size_t i = 0; i + 6 < planksAtHit.size(); i += 7) in.rubble.emplace_back(planksAtHit[i], planksAtHit[i + 1], planksAtHit[i + 2]);
        }
        in.aspect = aspect;
        in.hall = &hall;
        in.ballAt = [&](float t) { return replay.ballAt(t); };
        const auto t0 = WallClock::now();
        filmPlan = planFilm(in);
        std::printf("the film: %zu frames, %.2f s, planned in %.2f s; the ball hits %.3f s after its release, %.2f s into the film\n", filmPlan.frames.size(),
                    filmPlan.end, since(t0), in.hit, filmPlan.shots[3].t1);
        for (const auto& s : filmPlan.shots) std::printf("  shot %d %-12s %s - %s\n", s.number, s.name.c_str(), film::mmss(s.t0).c_str(), film::mmss(s.t1).c_str());
        std::printf("  the ball is in the wide frame %.2f s of film before it hits, in the slow motion's %.3f s of the recording (%.2f s of film); "
                    "the next seed's building reaches %.2f of the way to its frame's sides\n",
                    filmPlan.ballInWide, filmPlan.ballInSlow, filmPlan.ballInSlow * 4, filmPlan.nextSpill);
        std::printf("  the south-west tower: half %.3f m, shaft to %.2f m, top %.2f m; the building %.2f x %.2f x %.2f m; the ball hits at (%.2f, %.2f, %.2f)\n",
                    layout.towerHalf, layout.shaftTop, layout.towerHeight, layout.box.max().x - layout.box.min().x, layout.box.max().y,
                    layout.box.max().z - layout.box.min().z, in.impact.x, in.impact.y, in.impact.z);
        {
            // every camera inside the hall
            int outside = 0;
            for (const auto& f : filmPlan.frames)
                if (f.eye[0] < hall.x0 + 0.3f || f.eye[0] > hall.x1 - 0.3f || f.eye[2] < hall.zN + 0.3f || f.eye[2] > hall.zS - 0.3f || f.eye[1] > hall.height - 0.3f) ++outside;
            const auto& first = filmPlan.frames.front();
            std::printf("  the hall: x %.2f to %.2f, z %.2f to %.2f, %.1f m high, windows from x = %.2f every %.1f m, %.2f wide, sill %.2f, spring %.2f; "
                        "the hook opens at (%.2f, %.2f, %.2f); %d frames have the camera within 0.3 m of a wall or outside\n",
                        hall.x0, hall.x1, hall.zN, hall.zS, hall.height, hall.firstX, hall.bay, hall.width, hall.sill, hall.spring, first.eye[0], first.eye[1],
                        first.eye[2], outside);
            for (const auto& sh : filmPlan.shots) {
                const int f0 = static_cast<int>(std::ceil(sh.t0 * film::fps - 1e-6)), f1 = std::min(static_cast<int>(filmPlan.frames.size()), static_cast<int>(std::ceil(sh.t1 * film::fps - 1e-6))) - 1;
                for (const int f : {f0, f1}) {
                    const auto& s = filmPlan.frames[f];
                    std::printf("  camera shot %d frame %d: eye %.3f %.3f %.3f target %.3f %.3f %.3f fov %.1f\n", sh.number, f, s.eye[0], s.eye[1], s.eye[2], s.target[0], s.target[1],
                                s.target[2], s.fov);
                }
            }
        }

        // every word, from the one table
        const auto captions = film::captionTable(filmPlan.shots, static_cast<long long>(design.planks.size()), filmPlan.slowHit);
        std::ofstream(filmPath / "captions.ass", std::ios::binary) << film::assFile(captions, filmPlan.shots);
        std::ofstream(filmPath / "captions.md", std::ios::binary) << film::captionsMarkdown(captions, filmPlan.shots, filmPlan.end);

        // what the pixels depend on besides the frames' plan: the buildings and the recording
        film::Hash h;
        h.text(hashDesign(design));
        h.text(hashDesign(nextDesignPlanks));
        h.text(hashRecording(replayPath));
        sceneStamp = h.hex();
        std::printf("the scene's stamp %s (the designs and the recording)\n", sceneStamp.c_str());

        if (filmFinish) {
            const auto ff = std::filesystem::path(ffmpegPath);
            if (const int joined = joinMaster(); joined != 0) return joined;
            char fades[128];
            std::snprintf(fades, sizeof(fades), "fade=t=in:st=0:d=0.5,fade=t=out:st=%.2f:d=0.8", filmPlan.end - 0.8);
            const std::string burn = film::inFolder(filmPath) + film::q(ff) + " -y -loglevel error -i master.mkv -vf \"ass=captions.ass," + fades +
                                     ",format=yuv420p\" -r 60 -c:v libx264 -preset slow -crf 16 -movflags +faststart film.mp4";
            if (film::runCommand(burn) != 0) {
                std::printf("ffmpeg could not burn the words in\n");
                return 1;
            }
            std::printf("wrote %s (the master: %s)\n", (filmPath / "film.mp4").string().c_str(), (filmPath / "master.mkv").string().c_str());
            return 0;
        }
    }

    StepStats stats;
    bool hung = false;
    std::optional<float> hitTime;// when the ball first woke a plank
    if (replaying && replay.header.hitTime >= 0) hitTime = replay.header.hitTime;
    const auto stepPhysics = [&] {
        if (ballOn && !hung) {
            physics->hangBall(swing.pivot, swing.start, swing.radius);
            hung = true;
        }
        const auto t0 = WallClock::now();
        physics->step();
        const double ms = since(t0) * 1000;
        const bool before = !hitTime;
        const int awake = (stats.steps % 4 == 0 || !before) ? physics->awake() : 0;
        if (ballOn && !hitTime && awake > 0) {
            hitTime = physics->time;
            std::printf("the ball hit at t = %.3f s\n", *hitTime);
        }
        stats.add(ms, awake, physics->time, before);
    };

    // ------------------------------------------------------------ record: CPU only

    if (recording) {
        if (!hung) {
            physics->hangBall(swing.pivot, swing.start, swing.radius);
            hung = true;
        }
        std::FILE* f = std::fopen(recordPath.c_str(), "wb");
        if (!f) {
            std::printf("cannot write %s\n", recordPath.c_str());
            return 1;
        }
        RecordHeader header;
        header.planks = static_cast<std::uint32_t>(design.planks.size());
        header.seed = seed;
        header.design = designIndex;
        header.scale = scale;
        header.fps = recordFps;
        header.ballRadius = swing.radius;
        header.pivot[0] = swing.pivot.x, header.pivot[1] = swing.pivot.y, header.pivot[2] = swing.pivot.z;
        header.rope = swing.pivot.distanceTo(swing.start);
        std::fwrite(&header, sizeof(header), 1, f);
        std::vector<float> frame(8 + 7 * design.planks.size());
        const auto t0 = WallClock::now();
        bool reported6 = false;
        std::uint32_t frames = 0;
        for (int i = 0;; ++i) {
            const float t = static_cast<float>(i) / recordFps;
            if (t > recordTime + 1e-4f) break;
            while (physics->time + 0.5f * physics->dt < t) stepPhysics();
            frame[0] = physics->time;
            const auto b = physics->ball->getGlobalPose();
            const float ball[7] = {b.p.x, b.p.y, b.p.z, b.q.x, b.q.y, b.q.z, b.q.w};
            std::copy(ball, ball + 7, frame.begin() + 1);
            for (size_t k = 0; k < physics->bodies.size(); ++k) {
                const auto p = physics->pose(k);
                float* out = frame.data() + 8 + 7 * k;
                out[0] = p.p.x, out[1] = p.p.y, out[2] = p.p.z, out[3] = p.q.x, out[4] = p.q.y, out[5] = p.q.z, out[6] = p.q.w;
            }
            std::fwrite(frame.data(), sizeof(float), frame.size(), f);
            ++frames;
            if (hitTime && !reported6 && physics->time >= *hitTime + 6.f - 1e-4f) {
                reported6 = true;
                const int m = physics->movedMoreThan(0.05f);
                std::printf("6 s after the hit: %d of %zu planks moved more than 5 cm (%.1f%%)\n", m, design.planks.size(), 100.0 * m / design.planks.size());
            }
            if (i % static_cast<int>(recordFps) == 0) {
                // the header as it stands, so that a recording still being written replays up to here
                header.frames = frames;
                header.hitTime = hitTime ? *hitTime : -1.f;
                std::fflush(f);
                seekTo(f, 0);
                std::fwrite(&header, sizeof(header), 1, f);
                std::fflush(f);
                seekToEnd(f);
                std::printf("  recorded %.1f s (%d awake, %d moved more than 5 cm, %.0f s so far)\n", t, physics->awake(), physics->movedMoreThan(0.05f), since(t0));
            }
        }
        header.frames = frames;
        header.hitTime = hitTime ? *hitTime : -1.f;
        seekTo(f, 0);
        std::fwrite(&header, sizeof(header), 1, f);
        std::fclose(f);
        {
            // which parts came down: planks moved more than 5 cm, element by element
            const auto moved = physics->displacements();
            std::map<std::string, std::pair<int, int>> fell;
            for (size_t k = 0; k < moved.size(); ++k) {
                auto& e = fell[design.elements[design.planks[k].element]];
                ++e.second;
                if (moved[k] > 0.05f) ++e.first;
            }
            std::printf("at the end (%.2f s after the hit), planks moved more than 5 cm by element:\n", physics->time - hitTime.value_or(0.f));
            for (const auto& [name, v] : fell) std::printf("    %-26s %5d of %5d (%3.0f%%)\n", name.c_str(), v.first, v.second, 100.0 * v.first / std::max(1, v.second));
            const int all = physics->movedMoreThan(0.05f);
            std::printf("in all: %d of %zu (%.1f%%)\n", all, design.planks.size(), 100.0 * all / design.planks.size());
        }
        const double bytesPerFrame = static_cast<double>(frame.size() * sizeof(float));
        stats.print(hitTime.value_or(0.f));
        std::printf("recorded %u frames (%.2f s at %.0f fps) to %s: %.1f MB, %.1f MB per second; %.0f s wall clock on %u threads\n", frames, recordTime, recordFps,
                    recordPath.c_str(), bytesPerFrame * frames / 1e6, bytesPerFrame * recordFps / 1e6, since(t0), threads);
        return 0;
    }

    if (physicsOnly) {
        if (collapseSeconds > 0) {
            const auto t0 = WallClock::now();
            while (!hitTime && physics->time < 10.f) stepPhysics();
            const float until = (hitTime ? *hitTime : physics->time) + collapseSeconds;
            float report = hitTime ? *hitTime + 1.f : until;
            while (physics->time < until) {
                stepPhysics();
                if (physics->time >= report) {
                    std::printf("  %.1f s after the hit: %d planks moved more than 5 cm, %d awake (%.0f s so far)\n",
                                physics->time - *hitTime, physics->movedMoreThan(0.05f), physics->awake(), since(t0));
                    report += 1.f;
                }
            }
            stats.print(hitTime ? *hitTime : 0.f);
            std::printf("collapse: %d of %zu planks moved more than 5 cm (%.0f s of CPU for %.1f s)\n",
                        physics->movedMoreThan(0.05f), design.planks.size(), since(t0), physics->time);
            if (!dumpFinalPath.empty()) {
                physics->dump(dumpFinalPath);
                std::printf("wrote %s\n", dumpFinalPath.c_str());
            }
        }
        return 0;
    }

    // ------------------------------------------------------------ the hall

    const bool supersample = headless && ssaa > 0;
    Canvas canvas(Canvas::Parameters()
                          .title("Plank cathedral")
                          .size(width, height)
                          .antialiasing(supersample ? 0 : 4)
                          .headless(headless));
    GLRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = exposure;
    renderer.shadowMap().enabled = true;
    renderer.shadowMap().type = ShadowMap::PFC;

    Scene scene;
    scene.background = Color(0x0d0a08);
    scene.fog = Fog(Color(look.fog), look.fogNear, look.fogFar);
    scene.environment = hallEnvironment(renderer, hall, look.fillIntensity);
    scene.environmentIntensity = look.environment;

    // the hall, and the sun that reaches anything in it only through the windows
    const std::string sun = sunGlsl(hall);
    buildHall(scene, hall, sun);
    std::printf("the hall: x %.2f to %.2f, z %.2f to %.2f, %.1f m high; %d windows from x = %.2f every %.1f m; the sun (%.3f, %.3f, %.3f)\n", hall.x0, hall.x1,
                hall.zN, hall.zS, hall.height, hall.windows, hall.firstX, hall.bay, hall.toSun.x, hall.toSun.y, hall.toSun.z);

    // the planks: one instanced mesh
    const auto plankGeometry = RoundedBoxGeometry::create(LEN, THICK, WIDE, 2, 0.0007f);
    auto planksMesh = InstancedMesh::create(plankGeometry, woodMaterial(sun), design.planks.size());
    planksMesh->castShadow = true;
    planksMesh->receiveShadow = true;
    planksMesh->frustumCulled = false;
    {
        Rng tint(seed * 977 + 11);
        for (size_t i = 0; i < design.planks.size(); ++i) planksMesh->setColorAt(i, plankTint(tint));
        planksMesh->instanceColor()->needsUpdate();
    }
    scene.add(planksMesh);

    // where the planks and the ball are: from PhysX, or from a recording
    std::vector<float> replayPlanks;
    float replayBall[7] = {swing.start.x, swing.start.y, swing.start.z, 0, 0, 0, 1};
    const auto syncPlanks = [&] {
        Matrix4 m;
        for (size_t i = 0; i < design.planks.size(); ++i) {
            if (replaying) {
                const float* p = replayPlanks.data() + 7 * i;
                m.compose(Vector3(p[0], p[1], p[2]), Quaternion(p[3], p[4], p[5], p[6]), Vector3(1, 1, 1));
            } else {
                const auto p = physics->pose(i);
                m.compose(Vector3(p.p.x, p.p.y, p.p.z), Quaternion(p.q.x, p.q.y, p.q.z, p.q.w), Vector3(1, 1, 1));
            }
            planksMesh->setMatrixAt(i, m);
        }
        planksMesh->instanceMatrix()->needsUpdate();
    };
    const auto movedOver = [&](float by) {
        if (!replaying) return physics->movedMoreThan(by);
        int n = 0;
        for (size_t i = 0; i < design.planks.size(); ++i) {
            const auto& d = design.planks[i];
            const float* p = replayPlanks.data() + 7 * i;
            const float dx = p[0] - d.x, dy = p[1] - courseY(d.course), dz = p[2] - d.z;
            n += dx * dx + dy * dy + dz * dz > by * by ? 1 : 0;
        }
        return n;
    };
    if (replaying) replay.at(0, replayPlanks, replayBall);
    syncPlanks();

    // the wrecking ball, dark worn steel, and its rope, in the same sun
    auto ballMaterial = hallMaterial(sun, "ball", Color(0x3c3d3f), 0.42f);
    ballMaterial->metalness = 0.85f;
    auto ballMesh = Mesh::create(SphereGeometry::create(swing.radius, 64, 32), ballMaterial);
    ballMesh->castShadow = true;
    ballMesh->receiveShadow = true;
    ballMesh->visible = false;
    scene.add(ballMesh);
    auto ropeMaterial = hallMaterial(sun, "rope", Color(0x2a241d), 0.9f);
    auto ropeMesh = Mesh::create(BufferGeometry::create(), ropeMaterial);
    ropeMesh->castShadow = true;
    ropeMesh->visible = false;
    ropeMesh->frustumCulled = false;
    scene.add(ropeMesh);

    // The key is the low, warm late sun through the north windows (Hall.hpp lets it through only there): from
    // the north-west, so that the west front is lit and the long shadows run south-east across the floor. Its
    // shadow map is fitted tightly round the building (and round where the rubble and the ball go), so the
    // planks keep crisp shadows; the windows' shapes need no shadow map. The room's light (the environment, with
    // its warm bounce from the side away from the sun) and the floor's warm bounce lift the shadows.
    const View hero = heroView(layout, aspect);
    auto key = DirectionalLight::create(Color(0xffd3a6), look.keyIntensity);
    Object3D keyTarget;
    scene.addRef(keyTarget);
    const Vector3 toLight = hall.toSun;
    // the shadow map fitted round a box, the key 14 m out toward the light
    const auto fitKey = [&](Box3 box) {
        Vector3 centre;
        box.getCenter(centre);
        key->position.copy(centre).addScaledVector(toLight, 14.f);
        fitShadow(*key, keyTarget, box);
        // the shadow camera reaches past the floor where the long shadows fall
        auto* cam = key->shadow->camera->as<OrthographicCamera>();
        cam->farPlane += box.max().y / std::max(0.2f, toLight.y) + 0.5f;
        cam->updateProjectionMatrix();
    };
    {
        key->setTarget(keyTarget);
        key->castShadow = true;
        key->shadow->mapSize.set(4096, 4096);
        key->shadow->bias = -0.00015f;
        key->shadow->normalBias = 0.0012f;
        key->shadow->radius = 2.5f;
        Box3 box = layout.box;
        box.expandByScalar(0.05f);
        if (ballOn && !filming) box.expandByPoint(swing.start).expandByScalar(swing.radius);
        fitKey(box);
    }
    scene.add(key);
    // what the floor throws back into the shadows: warm, from below
    auto bounce = HemisphereLight::create(Color(0x101215), Color(0x7a604a), look.bounce);
    scene.add(bounce);

    // the near plane as far out as the closest shots allow: the depth buffer's precision goes where it is needed, so
    // the hall's far walls never fight their pilasters
    PerspectiveCamera camera(30, aspect, 0.04f, 80);
    camera.position.copy(camArg.value_or(hero.eye));
    OrbitControls controls(camera, canvas);
    controls.target.copy(targetArg.value_or(hero.target));
    controls.update();

    EffectComposer::Options composerOptions;
    composerOptions.samples = supersample ? 0 : 4;
    EffectComposer composer(renderer, composerOptions);
    if (supersample) {
        auto pass = std::make_shared<SSAARenderPass>(scene, camera);
        pass->sampleLevel = ssaa;
        composer.addPass(pass);
    } else {
        composer.addPass(std::make_shared<RenderPass>(scene, camera));
    }
    GTAOParameters aoParameters;
    aoParameters.radius = 0.06f;
    aoParameters.distanceExponent = 1.f;
    aoParameters.thickness = 1.f;
    aoParameters.scale = 1.f;
    aoParameters.samples = 16;
    aoParameters.distanceFallOff = 1.f;
    aoParameters.screenSpaceRadius = false;
    PoissonDenoiseParameters denoiseParameters;
    denoiseParameters.lumaPhi = 10.f;
    denoiseParameters.depthPhi = 2.f;
    denoiseParameters.normalPhi = 3.f;
    denoiseParameters.radius = 4.f;
    denoiseParameters.radiusExponent = 1.f;
    denoiseParameters.rings = 2.f;
    denoiseParameters.samples = 16;
    auto gtaoPass = std::make_shared<GTAOPass>(scene, camera, width, height, aoParameters, denoiseParameters);
    composer.addPass(gtaoPass);
    composer.addPass(std::make_shared<OutputPass>());
    if (!supersample) composer.addPass(std::make_shared<FXAAPass>());

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
        composer.setSize(size.width(), size.height());
    });

    const auto aim = [&](const View& v) {
        camera.position.copy(v.eye);
        camera.fov = fovArg.value_or(v.fov);
        camera.updateProjectionMatrix();
        camera.lookAt(v.target);
    };

    const auto showBall = [&] {
        if (!ballOn) return;
        Vector3 at = swing.start;
        if (replaying) {
            at.set(replayBall[0], replayBall[1], replayBall[2]);
            ballMesh->quaternion.set(replayBall[3], replayBall[4], replayBall[5], replayBall[6]);
        } else if (physics->ball) {
            const auto p = physics->ball->getGlobalPose();
            at.set(p.p.x, p.p.y, p.p.z);
            ballMesh->quaternion.set(p.q.x, p.q.y, p.q.z, p.q.w);
        }
        ballMesh->visible = true;
        ballMesh->position.copy(at);
        // the rope from the pivot to the top of the ball, slack when the ball is nearer the pivot than its length
        const Vector3 top = Vector3(swing.pivot).sub(at).normalize().multiplyScalar(swing.radius).add(at);
        ropeMesh->setGeometry(ropeGeometry(swing.pivot, top, swing.rope - swing.radius));
        ropeMesh->visible = true;
    };

    // ------------------------------------------------------------ the film: drawing

    if (filming) {
        const std::filesystem::path ff = ffmpegPath;

        // the next seed's building: its own planks, its own tints, standing as designed (proven earlier)
        const size_t nB = nextDesignPlanks.planks.size();
        auto nextMesh = InstancedMesh::create(plankGeometry, woodMaterial(sun), nB);
        nextMesh->castShadow = true;
        nextMesh->receiveShadow = true;
        nextMesh->frustumCulled = false;
        nextMesh->visible = false;
        {
            Rng tint(nextSeed * 977 + 11);
            for (size_t i = 0; i < nB; ++i) nextMesh->setColorAt(i, plankTint(tint));
            nextMesh->instanceColor()->needsUpdate();
        }
        scene.add(nextMesh);
        std::vector<float> restB(7 * nB);
        for (size_t i = 0; i < nB; ++i) {
            const auto& p = nextDesignPlanks.planks[i];
            Quaternion r;
            r.setFromAxisAngle(Vector3(0, 1, 0), -p.angle);
            const float pose[7] = {p.x, courseY(p.course), p.z, r.x, r.y, r.z, r.w};
            std::copy(pose, pose + 7, restB.begin() + 7 * static_cast<std::ptrdiff_t>(i));
        }
        const auto riseA = film::makeBuild(design.planks.size(), [&](size_t i) { return design.planks[i].course; }, filmPlan.buildA, 0.4f, 0.025f);
        const auto riseB = film::makeBuild(nB, [&](size_t i) { return nextDesignPlanks.planks[i].course; }, filmPlan.buildB, 0.2f, 0.02f);

        Matrix4 hidden;
        hidden.makeScale(0, 0, 0);
        const auto setPoses = [&](InstancedMesh& mesh, const std::vector<float>& pose, const film::Build* rise, float t) {
            Matrix4 m;
            const size_t n = pose.size() / 7;
            for (size_t i = 0; i < n; ++i) {
                const float* p = pose.data() + 7 * i;
                float lift = 0;
                if (rise) {
                    lift = rise->lift(i, t);
                    if (lift < 0) {
                        mesh.setMatrixAt(i, hidden);
                        continue;
                    }
                }
                m.compose(Vector3(p[0], p[1] + lift, p[2]), Quaternion(p[3], p[4], p[5], p[6]), Vector3(1, 1, 1));
                mesh.setMatrixAt(i, m);
            }
            mesh.instanceMatrix()->needsUpdate();
        };

        // the shadow map round what is in the frame: the building, or the building and where its rubble and the ball go
        const Vector3 ballIn = horizontal(Vector3(restBall[0], 0, restBall[2]).sub(Vector3(swing.pivot.x, 0, swing.pivot.z)));
        int shadowFor = -1;
        const auto fitFor = [&](int sceneIndex, int mode) {
            const int k = sceneIndex * 4 + mode;
            if (k == shadowFor) return;
            shadowFor = k;
            Box3 box = sceneIndex == 0 ? layout.box : nextLayout->box;
            box.expandByScalar(0.05f);
            if (sceneIndex == 0 && mode == 2) {
                box.expandByVector(Vector3(0.6f, 0, 0.6f));
                const Vector3 bottom(swing.pivot.x, swing.pivot.y - swing.rope, swing.pivot.z);
                box.expandByPoint(Vector3(bottom).addScaledVector(ballIn, 1.2f).add(Vector3(0, swing.radius * 2, 0)));
            }
            fitKey(box);
        };

        std::vector<unsigned char> pixels(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
        std::vector<float> poses;
        FrameState drawn;
        bool haveDrawn = false;
        int poseScene = -1, poseMode = -1;
        float poseT = -1e9f;
        double drawSeconds = 0;
        int drawnFrames = 0;
        // one frame of the film into `pixels` (RGB, bottom row first); the same frame again costs nothing
        const auto draw = [&](const FrameState& s) {
            if (haveDrawn && s.same(drawn)) return;
            const auto t0 = WallClock::now();
            planksMesh->visible = s.scene == 0;
            nextMesh->visible = s.scene == 1;
            fitFor(s.scene, s.mode);
            gtaoPass->blendIntensity = s.ao;
            key->shadow->radius = s.shadowRadius;
            if (s.scene != poseScene || s.mode != poseMode || s.t != poseT) {
                if (s.scene == 1) setPoses(*nextMesh, restB, &riseB, s.t);
                else if (s.mode == 0) setPoses(*planksMesh, restA, nullptr, 0);
                else if (s.mode == 1) setPoses(*planksMesh, restA, &riseA, s.t);
                else {
                    replay.at(s.t, poses, replayBall);
                    setPoses(*planksMesh, poses, nullptr, 0);
                }
                poseScene = s.scene, poseMode = s.mode, poseT = s.t;
            }
            if (s.scene == 0 && s.mode == 2) showBall();
            else ballMesh->visible = ropeMesh->visible = false;
            camera.position.set(s.eye[0], s.eye[1], s.eye[2]);
            camera.fov = s.fov;
            camera.updateProjectionMatrix();
            camera.lookAt(Vector3(s.target[0], s.target[1], s.target[2]));
            composer.render();
            renderer.readPixels(Vector2(0, 0), {width, height}, Format::RGB, pixels.data());
            drawn = s;
            haveDrawn = true;
            drawSeconds += since(t0);
            ++drawnFrames;
        };

        const auto& frames = filmPlan.frames;
        const int frameCount = static_cast<int>(frames.size());
        const int chunks = (frameCount + film::chunkFrames - 1) / film::chunkFrames;

        // frames f0..f1-1 into an mkv of the clean master
        const auto drawInto = [&](const std::filesystem::path& out, int f0, int f1) {
            std::FILE* pipe = film::openPipe(film::encoder(ff, out, width, height));
            if (!pipe) return false;
            bool ok = true;
            for (int f = f0; f < f1 && ok; ++f) {
                draw(frames[f]);
                ok = std::fwrite(pixels.data(), 1, pixels.size(), pipe) == pixels.size();
            }
            return film::closePipe(pipe) == 0 && ok;
        };

        canvas.animate([&] {
            int code = 0;
            const auto started = WallClock::now();
            // the first frames compile the shaders: drawn once and dropped
            {
                FrameState warm = frames.front();
                for (int i = 0; i < 2; ++i) {
                    haveDrawn = false;
                    draw(warm);
                }
                warm = frames[static_cast<size_t>(filmPlan.shots[6].t0 * film::fps) + 10];
                haveDrawn = false;
                draw(warm);
                haveDrawn = false;
                drawSeconds = 0;
                drawnFrames = 0;
            }

            if (!filmStillsDir.empty()) {
                // a few of the film's frames as stills, without words
                std::filesystem::create_directories(filmStillsDir);
                const auto frameAt = [&](double t) { return std::clamp(static_cast<int>(std::lround(t * film::fps)), 0, frameCount - 1); };
                const auto& sh = filmPlan.shots;
                const std::pair<const char*, int> stills[] = {
                        {"film_1_hook_first", 0},
                        {"film_1_hook_last", frameAt(sh[0].t1) - 1},
                        {"film_2_tower_first", frameAt(sh[1].t0)},
                        {"film_2_tower_last", frameAt(sh[1].t1) - 1},
                        {"film_3_build_half", frameAt(0.5 * (sh[2].t0 + sh[2].t1))},
                        {"film_4_swing", frameAt(0.5 * (sh[3].t0 + sh[3].t1))},
                        {"film_5_collapse_+1s", frameAt(sh[4].t0 + 1.0)},
                        {"film_5_collapse_+6s", frameAt(sh[4].t1) - 1},
                        {"film_6_slow_motion", frameAt(filmPlan.slowHit + 4 * 0.45)},
                        {"film_7_next_seed", frameAt(sh[6].t1) - 1},
                };
                for (const auto& [name, f] : stills) {
                    draw(frames[f]);
                    const auto path = std::filesystem::path(filmStillsDir) / (std::string(name) + ".png");
                    renderer.writeFramebuffer(path);
                    std::printf("wrote %s (frame %d, %s)\n", path.string().c_str(), f, film::mmss(static_cast<double>(f) / film::fps).c_str());
                }
                std::printf("%d frames drawn, %.1f ms a frame\n", drawnFrames, 1000 * drawSeconds / std::max(1, drawnFrames));
            } else if (filmSheet) {
                // the first, middle and last frame of every shot, the words laid on by ffmpeg, in one sheet
                const auto dir = filmPath / "sheet";
                std::filesystem::create_directories(dir);
                int tile = 0;
                for (const auto& shot : filmPlan.shots) {
                    const int first = static_cast<int>(std::ceil(shot.t0 * film::fps - 1e-6));
                    const int last = std::min(frameCount, static_cast<int>(std::ceil(shot.t1 * film::fps - 1e-6))) - 1;
                    const int which[3] = {first, (first + last) / 2, last};
                    const char* names[3] = {"first", "middle", "last"};
                    for (int k = 0; k < 3; ++k) {
                        const int f = which[k];
                        draw(frames[f]);
                        std::ofstream(dir / "tile.rgb", std::ios::binary).write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
                        char label[160];
                        std::snprintf(label, sizeof(label), "%d %s  |  %s frame  |  %s", shot.number, shot.name.c_str(), names[k], film::mmss(static_cast<double>(f) / film::fps).c_str());
                        std::ofstream(dir / "label.ass", std::ios::binary)
                                << "[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
                                   "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                                   "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
                                   "Style: Label,Consolas,40,&H0000E8FF,&H0000E8FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,3,6,0,7,24,24,20,1\n\n"
                                   "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
                                << "Dialogue: 5,0:00:00.00,9:00:00.00,Label,,0,0,0,," << label << "\n";
                        char png[64];
                        std::snprintf(png, sizeof(png), "tile_%02d.png", tile++);
                        const std::string cmd = film::inFolder(filmPath) + film::q(ff) + " -y -loglevel error -f rawvideo -pix_fmt rgb24 -s " + std::to_string(width) + "x" +
                                                std::to_string(height) + " -framerate " + std::to_string(film::fps) + " -i sheet/tile.rgb -vf \"vflip,setpts=" + std::to_string(f) +
                                                ",ass=captions.ass,ass=sheet/label.ass\" -frames:v 1 sheet/" + png;
                        if (film::runCommand(cmd) != 0) {
                            std::printf("ffmpeg could not compose %s\n", png);
                            code = 1;
                        }
                    }
                }
                const std::string cmd = film::inFolder(filmPath) + film::q(ff) + " -y -loglevel error -framerate 1 -i sheet/tile_%02d.png -vf tile=3x" +
                                        std::to_string(filmPlan.shots.size()) + ":padding=8:color=0x101010 -frames:v 1 shot_sheet.png";
                if (film::runCommand(cmd) != 0) code = 1;
                std::printf("wrote %s (%d frames drawn, %.1f ms a frame)\n", (filmPath / "shot_sheet.png").string().c_str(), drawnFrames, 1000 * drawSeconds / std::max(1, drawnFrames));
            } else if (!clipName.empty()) {
                // a test clip: the film from T0 to T1, clean
                if (clipShotFrom > 0) {
                    // from the first shot's start to the last shot's end, at most 10 s
                    for (const auto& sh : filmPlan.shots) {
                        if (sh.number == clipShotFrom) clipFrom = static_cast<float>(sh.t0);
                        if (sh.number == clipShotTo) clipTo = static_cast<float>(sh.t1);
                    }
                    clipTo = std::min(clipTo, clipFrom + 10.f);
                }
                const int f0 = std::clamp(static_cast<int>(std::lround(clipFrom * film::fps)), 0, frameCount - 1);
                const int f1 = std::clamp(static_cast<int>(std::lround(clipTo * film::fps)), f0 + 1, frameCount);
                const auto dir = filmPath / "clips";
                std::filesystem::create_directories(dir);
                const auto master = dir / (clipName + "_master.mkv");
                const auto t0 = WallClock::now();
                if (!drawInto(master, f0, f1)) {
                    std::printf("could not draw the clip\n");
                    code = 1;
                } else {
                    // clean, as the film goes out: the words are laid on later, in post
                    const std::string cmd = film::inFolder(filmPath) + film::q(ff) + " -y -loglevel error -i " + film::q(master) +
                                            " -vf format=yuv420p -r 60 -c:v libx264 -preset slow -crf 17 -movflags +faststart clip_" + clipName + ".mp4";
                    if (film::runCommand(cmd) != 0) code = 1;
                    std::printf("wrote %s: %.2f to %.2f s, %d frames in %.1f s (%.1f ms a frame drawn and read)\n", (filmPath / ("clip_" + clipName + ".mp4")).string().c_str(),
                                static_cast<double>(f0) / film::fps, static_cast<double>(f1) / film::fps, f1 - f0, since(t0), 1000 * drawSeconds / std::max(1, drawnFrames));
                }
            } else {
                // the film's chunks: those not drawn for this plan, while the budget lasts
                if (filmFresh && filmChunk < 0)
                    for (int c = 0; c < chunks; ++c) {
                        std::error_code ec;
                        std::filesystem::remove(chunkName(c, ".stamp"), ec);
                    }
                int done = 0, left = 0;
                for (int c = 0; c < chunks; ++c) {
                    const bool wanted = filmChunk < 0 || c == filmChunk;
                    if (!wanted) continue;
                    if (filmChunk < 0 && chunkDone(c)) {
                        ++done;
                        continue;
                    }
                    if (filmChunk < 0 && since(started) > filmBudget) {
                        ++left;
                        continue;
                    }
                    const int f0 = c * film::chunkFrames, f1 = std::min(frameCount, f0 + film::chunkFrames);
                    const auto t0 = WallClock::now();
                    const double drawBefore = drawSeconds;
                    const int framesBefore = drawnFrames;
                    std::error_code ec;
                    std::filesystem::remove(chunkName(c, ".stamp"), ec);
                    if (!drawInto(chunkName(c, ".tmp.mkv"), f0, f1)) {
                        std::printf("chunk %d: drawing or encoding failed\n", c);
                        code = 1;
                        break;
                    }
                    std::filesystem::remove(chunkName(c, ".mkv"), ec);
                    std::filesystem::rename(chunkName(c, ".tmp.mkv"), chunkName(c, ".mkv"), ec);
                    std::ofstream(chunkName(c, ".stamp"), std::ios::binary) << chunkStamp(c);
                    ++done;
                    const double wall = since(t0);
                    std::printf("chunk %d of %d (%.2f-%.2f s, shots %d-%d): %d frames in %.1f s, %.1f ms a frame (%d drawn, %.1f ms each)\n", c + 1, chunks,
                                static_cast<double>(f0) / film::fps, static_cast<double>(f1) / film::fps, frames[f0].shot, frames[f1 - 1].shot, f1 - f0, wall,
                                1000 * wall / (f1 - f0), drawnFrames - framesBefore, 1000 * (drawSeconds - drawBefore) / std::max(1, drawnFrames - framesBefore));
                }
                if (filmChunk < 0) {
                    std::printf("%d of %d chunks drawn for this plan, %d left\n", done, chunks, left);
                    code = left > 0 ? 3 : code;
                    if (code == 0) code = joinMaster();
                }
            }
            std::fflush(stdout);
            std::exit(code);
        });
        return 0;
    }

    // ------------------------------------------------------------ headless

    if (headless) {
        struct Capture {
            float time;     // seconds on the scene's clock, or after the hit
            bool afterHit;
            View view;
            std::string path;
        };
        std::vector<Capture> captures;
        const std::string tag = "seed" + std::to_string(seed);
        if (!stillsDir.empty()) {
            std::filesystem::create_directories(stillsDir);
            auto views = standardViews(layout, aspect);
            views.insert(views.begin(), hallView(hall, layout));
            for (const auto& v : views) captures.push_back({shotTime, false, v, (std::filesystem::path(stillsDir) / (tag + "_" + v.name + ".png")).string()});
        }
        if (!heroPath.empty()) captures.push_back({shotTime, false, hero, heroPath});
        if (!shotPath.empty()) {
            View v = hero;
            if (camArg) v.eye = *camArg;
            if (targetArg) v.target = *targetArg;
            captures.push_back({shotTime, false, v, shotPath});
        }
        if (!probeDir.empty()) {
            std::filesystem::create_directories(probeDir);
            const View wide = probeView(layout, swing, aspect);
            captures.push_back({-0.25f, true, wide, (std::filesystem::path(probeDir) / (tag + "_collapse_0_before.png")).string()});
            int index = 1;
            for (const float after : {0.25f, 0.5f, 1.f, 2.f, 4.f, 6.f}) {
                char name[64];
                std::snprintf(name, sizeof(name), "%s_collapse_%d_+%.2fs.png", tag.c_str(), index++, after);
                captures.push_back({after, true, wide, (std::filesystem::path(probeDir) / name).string()});
            }
        }
        if (!sequenceDir.empty()) {
            std::filesystem::create_directories(sequenceDir);
            View v = hero;
            if (camArg) v.eye = *camArg;
            if (targetArg) v.target = *targetArg;
            const int frames = std::max(1, static_cast<int>(shotTime * static_cast<float>(fps)));
            for (int f = 0; f <= frames; ++f) {
                char name[32];
                std::snprintf(name, sizeof(name), "frame_%04d.png", f);
                captures.push_back({static_cast<float>(f) / static_cast<float>(fps), false, v, (std::filesystem::path(sequenceDir) / name).string()});
            }
        }
        std::stable_sort(captures.begin(), captures.end(), [](const Capture& a, const Capture& b) {
            if (a.afterHit != b.afterHit) return !a.afterHit;
            return a.time < b.time;
        });

        const float predicted = replaying ? hitTime.value_or(1.f) : predictHit(swing, layout);
        if (ballOn) std::printf("the ball %s the tower at t = %.3f s\n", replaying ? "reached" : "should reach", predicted);
        size_t next = 0;
        int warm = 0;
        bool timed = false;
        float synced = -1;
        canvas.animate([&] {
            if (next >= captures.size()) {
                if (ballOn && !replaying) stats.print(hitTime.value_or(0.f));
                if (ballOn && !replaying) std::printf("collapse: %d of %zu planks moved more than 5 cm by t = %.2f s\n", physics->movedMoreThan(0.05f), design.planks.size(), physics->time);
                std::fflush(stdout);
                std::exit(0);
            }
            const auto& c = captures[next];
            float when;
            if (replaying) {
                when = !c.afterHit ? c.time : hitTime.value_or(predicted) + c.time;
                if (synced != when) {
                    replay.at(when, replayPlanks, replayBall);
                    syncPlanks();
                    synced = when;
                }
            } else {
                if (c.afterHit && c.time >= 0)
                    while (!hitTime && physics->time < 10.f) stepPhysics();
                when = !c.afterHit ? c.time : c.time < 0 ? predicted + c.time : hitTime.value_or(0.f) + c.time;
                while (physics->time + 0.5f * sceneDt < when) stepPhysics();
                if (synced != physics->time) {
                    syncPlanks();
                    synced = physics->time;
                }
            }
            // the ball and its rope only in the collapse; standing, the building is shown alone
            if (c.afterHit) showBall();
            else ballMesh->visible = ropeMesh->visible = false;
            aim(c.view);

            const auto t0 = WallClock::now();
            composer.render();
            if (warm == 1 && !timed) {
                unsigned char pixel[4];
                renderer.readPixels(Vector2(0, 0), {1, 1}, Format::RGBA, pixel);
                std::printf("one %dx%d frame (warm, to the pixel readback, %d samples): %.1f ms\n", width, height, supersample ? 1 << ssaa : 1, since(t0) * 1000);
                timed = true;
            }
            if (++warm >= 3) {
                renderer.writeFramebuffer(c.path);
                std::printf("wrote %s (t = %.2f s, %d moved over 5 cm)\n", c.path.c_str(), when, movedOver(0.05f));
                std::fflush(stdout);
                warm = 0;
                ++next;
            }
        });
        return 0;
    }

    // ------------------------------------------------------------ windowed

    canvas.onKeyPressed([&](KeyEvent evt) {
        if (evt.key == Key::B && !ballOn) ballOn = true;
    });

    WallClock::time_point last = WallClock::now();
    const WallClock::time_point started = last;
    float pending = 0;
    canvas.animate([&] {
        if (replaying) {
            // the recording over and over, in real time
            const float t = std::fmod(static_cast<float>(since(started)), replay.duration() + 1.f);
            replay.at(t, replayPlanks, replayBall);
            syncPlanks();
        } else {
            const float elapsed = std::min(static_cast<float>(since(last)), 0.1f);
            last = WallClock::now();
            pending += elapsed;
            int steps = 0;
            while (pending >= sceneDt && steps < 16) {
                stepPhysics();
                pending -= sceneDt;
                ++steps;
            }
            if (steps >= 16) pending = 0;
            if (steps > 0) syncPlanks();
        }
        showBall();
        controls.update();
        composer.render();
    });
    return 0;
}
