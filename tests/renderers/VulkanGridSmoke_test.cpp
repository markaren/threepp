// VulkanGridSmoke_test — what the grid smoke solver (ParticleField::GridSmoke,
// src/threepp/renderers/vulkan/SmokeGrid.cpp) must do, measured on its own state
// (VulkanRenderer::readGridSmoke) rather than judged from a picture.
//
//   A  projection   the velocity left after the pressure projection has no more
//                   divergence than the pressure solve itself left behind, and
//                   the solve converges as its iterations rise
//   B  sealed box   with dissipation off, a box walled on every side keeps its smoke
//   C  walls        smoke on one side of a wall never reaches the other: a thick
//                   wall, a wall thinner than a cell, and a thick wall in a wind
//                   faster than a cell per step
//   D  symmetry     a source on the box's centre line makes a mirror-symmetric plume
//   E  still        a still collider and a still box in still air move nothing
//   F  resolution   a plume rises the same at 64^3, 96^3 and 128^3, and in cells
//                   that are not cubes
//
// Every case prints its numbers before its verdict, so a failure says by how much.
// An argument runs only the cases it names: `VulkanGridSmoke_test BD`.
// The host recomputes each quantity with the solver's own stencils and boundary
// rules (shaders/smoke_common.glsl: velAt, pressureAt, solidCell).
//
// Plain exit-code program (not Catch2), like the other Vulkan tests here. Exits 42
// (CTest "Skipped") when no Vulkan device is available.

#include "threepp/threepp.hpp"

#include "threepp/objects/ParticleField.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr int    kSkipCode      = 42;
    constexpr double kDt            = 1.0 / 60.0;
    constexpr int    kStepsPerFrame = 6;// 0.1 s of solver time per rendered frame

    int failures = 0;

    void check(bool ok, const std::string& what) {
        std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
        if (!ok) ++failures;
    }

    void setEnv(const char* k, const char* v) {
#ifdef _WIN32
        _putenv_s(k, v);
#else
        setenv(k, v, 1);
#endif
    }

    using State = VulkanRenderer::GridSmokeState;

    // The solver's state as its shaders see it: the same neighbour rules, on the host.
    struct View {
        const State& s;
        int          n;
        Vector3      lo;       // world minimum of the box
        Vector3      cell;     // cell size per axis (m)
        bool         floor;    // GridSmoke::solidFloor
        bool         hasSolids;// any collider or box (the shaders skip the mask otherwise)

        [[nodiscard]] size_t id(int i, int j, int k) const {
            return size_t(i) + size_t(n) * (size_t(j) + size_t(n) * size_t(k));
        }
        [[nodiscard]] bool in(int i, int j, int k) const {
            return i >= 0 && j >= 0 && k >= 0 && i < n && j < n && k < n;
        }
        [[nodiscard]] bool solid(int i, int j, int k) const {
            if (j < 0 && floor) return true;
            if (!in(i, j, k) || !hasSolids) return false;
            return s.solid[id(i, j, k)] > 0.5f;
        }
        [[nodiscard]] Vector3 world(int i, int j, int k) const {
            return {lo.x + (float(i) + 0.5f) * cell.x, lo.y + (float(j) + 0.5f) * cell.y,
                    lo.z + (float(k) + 0.5f) * cell.z};
        }
        // velAt: a solid's own velocity, the edge value beyond an open face.
        [[nodiscard]] Vector3 vel(int i, int j, int k) const {
            if (solid(i, j, k)) {
                if (!in(i, j, k)) return {};
                const size_t c = id(i, j, k) * 3u;
                return {s.solidVelocity[c], s.solidVelocity[c + 1], s.solidVelocity[c + 2]};
            }
            const size_t c = id(std::clamp(i, 0, n - 1), std::clamp(j, 0, n - 1), std::clamp(k, 0, n - 1)) * 3u;
            return {s.velocity[c], s.velocity[c + 1], s.velocity[c + 2]};
        }
        // pressureAt: a solid is a zero-gradient wall, the open boundary is p = 0.
        [[nodiscard]] float pressure(int i, int j, int k, float centre) const {
            if (solid(i, j, k)) return centre;
            if (!in(i, j, k)) return 0.f;
            return s.pressure[id(i, j, k)];
        }
        // smoke_diverge.comp's divergence, of the velocity as it stands now.
        [[nodiscard]] float divergence(int i, int j, int k) const {
            return (vel(i + 1, j, k).x - vel(i - 1, j, k).x) * 0.5f / cell.x +
                   (vel(i, j + 1, k).y - vel(i, j - 1, k).y) * 0.5f / cell.y +
                   (vel(i, j, k + 1).z - vel(i, j, k - 1).z) * 0.5f / cell.z;
        }
        // What smoke_jacobi.comp's fixed point leaves: L p - div, with L its 7-point Laplacian.
        [[nodiscard]] float solveResidual(int i, int j, int k) const {
            const float  pc = s.pressure[id(i, j, k)];
            const double wx = 1.0 / (cell.x * cell.x), wy = 1.0 / (cell.y * cell.y), wz = 1.0 / (cell.z * cell.z);
            const double lap = wx * (pressure(i + 1, j, k, pc) + pressure(i - 1, j, k, pc) - 2.0 * pc) +
                               wy * (pressure(i, j + 1, k, pc) + pressure(i, j - 1, k, pc) - 2.0 * pc) +
                               wz * (pressure(i, j, k + 1, pc) + pressure(i, j, k - 1, pc) - 2.0 * pc);
            return float(lap - s.divergence[id(i, j, k)]);
        }
        [[nodiscard]] double cellVolume() const { return double(cell.x) * cell.y * cell.z; }
    };

    struct Rig {
        Canvas&            canvas;
        VulkanRenderer&    renderer;
        Scene&             scene;
        PerspectiveCamera& camera;
    };

    // One field in the scene, stepped kStepsPerFrame fixed steps per rendered frame.
    class Sim {
    public:
        Sim(Rig& rig, const Vector3& center, const Vector3& half, std::uint32_t res)
            : rig_(rig), center_(center), half_(half), res_(res) {
            field_ = ParticleField::createGridSmoke(center, half, res);
            auto& gs            = field_->gridSmoke();
            gs.stepDt           = float(kDt);
            gs.maxStepsPerFrame = kStepsPerFrame;
            rig_.scene.add(field_);
        }
        // The renderer keys a field's device state by its address and sweeps a dead one only after the frame's live
        // fields are registered, so the next case's field could land on this address and inherit this solver. One
        // frame without it, after it is gone, sweeps it first.
        ~Sim() {
            rig_.scene.remove(*field_);
            field_.reset();
            render();
        }
        Sim(const Sim&)            = delete;
        Sim& operator=(const Sim&) = delete;

        ParticleField::GridSmoke& gs() { return field_->gridSmoke(); }

        // Run until `steps` solver steps have been taken (a multiple of kStepsPerFrame).
        void runTo(int steps) {
            if (frame_ < 0) {// the first frame only starts the clock
                gs().time = 0.0;
                render();
                frame_ = 0;
            }
            while (frame_ * kStepsPerFrame < steps) {
                ++frame_;
                // Half a step past the boundary, so the owed count never rounds down to one step short.
                gs().time = (double(frame_ * kStepsPerFrame) + 0.5) * kDt;
                render();
            }
        }
        void runToSeconds(double t) { runTo(int(std::lround(t / kDt))); }

        bool read(State& out) {
            if (!rig_.renderer.readGridSmoke(*field_, out)) return false;
            return out.resolution == res_;
        }
        [[nodiscard]] View view(const State& s) {
            const float n = float(res_);
            return View{s,
                        int(res_),
                        Vector3(center_.x - half_.x, center_.y - half_.y, center_.z - half_.z),
                        Vector3(2.f * half_.x / n, 2.f * half_.y / n, 2.f * half_.z / n),
                        gs().solidFloor,
                        !gs().colliders.empty() || !gs().boxes.empty()};
        }

    private:
        void render() {
            rig_.canvas.animateOnce([&] { rig_.renderer.render(rig_.scene, rig_.camera); });
        }

        Rig&                           rig_;
        Vector3                        center_, half_;
        std::uint32_t                  res_;
        std::shared_ptr<ParticleField> field_;
        int                            frame_ = -1;
    };

    // A buoyant plume rising from a soft ball near the floor. noise = 0, so nothing in it is random.
    ParticleField::GridSmoke::Source plumeSource(const Vector3& at, float radius) {
        ParticleField::GridSmoke::Source s;
        s.position      = at;
        s.radius        = radius;
        s.density       = 4.f;
        s.temperature   = 3.f;
        s.velocity      = Vector3(0.f, 1.5f, 0.f);
        s.velocityBlend = 4.f;
        s.noise         = 0.f;
        return s;
    }

    double rms(const std::vector<double>& v) {
        double a = 0.0;
        for (double x : v) a += x * x;
        return v.empty() ? 0.0 : std::sqrt(a / double(v.size()));
    }

    // Total smoke (density x volume) over the open cells that `keep` accepts.
    double smokeIn(const View& v, const std::function<bool(const Vector3&)>& keep) {
        double sum = 0.0;
        for (int k = 0; k < v.n; ++k)
            for (int j = 0; j < v.n; ++j)
                for (int i = 0; i < v.n; ++i) {
                    if (v.solid(i, j, k)) continue;
                    if (keep && !keep(v.world(i, j, k))) continue;
                    sum += v.s.density[v.id(i, j, k)];
                }
        return sum * v.cellVolume();
    }

    // ── A: the projection ────────────────────────────────────────────────────
    // After the projection the velocity's divergence is div_before - G.p, measured with the
    // solver's own divergence stencil. On a discretisation where the divergence of the
    // gradient IS the Laplacian the solve inverts, that equals minus the solve's residual
    // (L p - div_before), so the velocity keeps no divergence the solve did not leave.
    // When the two disagree, more iterations do not help: the solve converges to a
    // pressure whose gradient does not cancel the divergence.
    void caseProjection(Rig& rig) {
        std::printf("\n== A  projection: a buoyant plume, 1 s, 64^3, 2 m box ==\n");
        std::printf("  RMS over the open cells (1/s); 'interior' = 3+ cells from every face of the box\n");
        std::printf("  %6s  %11s  %11s  %11s  %11s  %9s  %15s\n", "iters", "div before", "solve resid", "div after",
                    "fp16 floor", "after/res", "interior a/b");
        struct Row {
            std::uint32_t iters;
            double        before, resid, after, floor;
        };
        std::vector<Row> rows;
        for (std::uint32_t iters : {8u, 24u, 80u, 200u}) {
            Sim  sim(rig, Vector3(0.f, 1.f, 0.f), Vector3(1.f, 1.f, 1.f), 64);
            auto& gs              = sim.gs();
            gs.solidFloor         = true;
            gs.pressureIterations = iters;
            gs.sources.push_back(plumeSource(Vector3(0.f, 0.25f, 0.f), 0.2f));
            sim.runToSeconds(1.0);
            State st;
            if (!sim.read(st)) {
                check(false, "A: the solver state reads back");
                return;
            }
            const View          v = sim.view(st);
            std::vector<double> before, resid, after, floor, inBefore, inAfter;
            for (int k = 0; k < v.n; ++k)
                for (int j = 0; j < v.n; ++j)
                    for (int i = 0; i < v.n; ++i) {
                        if (v.solid(i, j, k)) continue;
                        const double b = st.divergence[v.id(i, j, k)];
                        const double a = v.divergence(i, j, k);
                        before.push_back(b);
                        resid.push_back(v.solveResidual(i, j, k));
                        after.push_back(a);
                        // What rounding the stored velocity to fp16 alone puts into a central difference.
                        const Vector3 u = v.vel(i, j, k);
                        floor.push_back(u.length() * std::ldexp(1.0, -11) * 0.5 / v.cell.x * std::sqrt(2.0));
                        if (std::min({i, j, k, v.n - 1 - i, v.n - 1 - j, v.n - 1 - k}) >= 3) {
                            inBefore.push_back(b);
                            inAfter.push_back(a);
                        }
                    }
            Row r{iters, rms(before), rms(resid), rms(after), rms(floor)};
            rows.push_back(r);
            std::printf("  %6u  %11.4f  %11.4f  %11.4f  %11.4f  %9.2f  %15.3f\n", r.iters, r.before, r.resid, r.after,
                        r.floor, r.after / std::max(r.resid + r.floor, 1e-12),
                        rms(inAfter) / std::max(rms(inBefore), 1e-12));
        }
        check(rows.back().resid < rows.front().resid,
              "A: the pressure solve's residual falls as its iterations rise (8 -> 200)");
        bool consistent = true;
        for (const Row& r : rows) consistent = consistent && r.after <= 1.25 * r.resid + 2.0 * r.floor;
        check(consistent, "A: the projected velocity keeps no more divergence than the solve left "
                          "(div after <= 1.25 x solve residual + 2 x fp16 floor, every iteration count)");
        check(rows.back().after < 0.5 * rows.front().after,
              "A: more iterations leave less divergence in the velocity (200 vs 8: under half)");
    }

    // ── B: a sealed box keeps its smoke ──────────────────────────────────────
    void caseSealedBox(Rig& rig) {
        std::printf("\n== B  sealed box: walls on all six sides, dissipation off, 64^3 ==\n");
        Sim  sim(rig, Vector3(0.f, 1.f, 0.f), Vector3(1.f, 1.f, 1.f), 64);
        auto& gs        = sim.gs();
        gs.solidFloor   = true;
        gs.dissipation  = 0.f;
        gs.boundaryFade = 0.f;
        // 0.1 m walls (3.2 cells) inside the box on the four sides and the top; the solid floor is the sixth.
        const auto wall = [&](Vector3 mn, Vector3 mx) {
            ParticleField::GridSmoke::Box b;
            b.min = mn;
            b.max = mx;
            gs.boxes.push_back(b);
        };
        wall({-1.2f, -0.2f, -1.2f}, {-0.9f, 2.2f, 1.2f});
        wall({0.9f, -0.2f, -1.2f}, {1.2f, 2.2f, 1.2f});
        wall({-1.2f, -0.2f, -1.2f}, {1.2f, 2.2f, -0.9f});
        wall({-1.2f, -0.2f, 0.9f}, {1.2f, 2.2f, 1.2f});
        wall({-1.2f, 1.9f, -1.2f}, {1.2f, 2.2f, 1.2f});
        gs.sources.push_back(plumeSource(Vector3(0.f, 0.3f, 0.f), 0.2f));

        sim.runToSeconds(0.5);
        gs.sources.clear();// emission stops; from here the total may only move by the scheme's error
        // Carried in advective form (d rho/dt = -u . grad rho), the total changes at the density-weighted
        // divergence of the velocity: d/dt sum(rho) = sum(rho div u). This is that rate for the stored velocity.
        const auto weightedDiv = [&](const State& s) {
            const View v = sim.view(s);
            double     rd = 0.0, r = 0.0;
            for (int k = 0; k < v.n; ++k)
                for (int j = 0; j < v.n; ++j)
                    for (int i = 0; i < v.n; ++i) {
                        if (v.solid(i, j, k)) continue;
                        const double d = s.density[v.id(i, j, k)];
                        rd += d * v.divergence(i, j, k);
                        r += d;
                    }
            return rd / std::max(r, 1e-12);
        };
        State st;
        if (!sim.read(st)) {
            check(false, "B: the solver state reads back");
            return;
        }
        const double m0 = smokeIn(sim.view(st), nullptr);
        std::printf("  t = 0.5 s  total %.5f (source off)                 density-weighted div u %+.3f /s\n", m0,
                    weightedDiv(st));
        double worst = 0.0, prev = m0;
        for (double t : {1.5, 2.5, 3.5}) {
            sim.runToSeconds(t);
            if (!sim.read(st)) return;
            const double m = smokeIn(sim.view(st), nullptr);
            worst          = std::max(worst, std::abs(m - m0) / m0);
            std::printf("  t = %.1f s  total %.5f  (%+6.2f %%, %+.3f /s)  density-weighted div u %+.3f /s\n", t, m,
                        100.0 * (m - m0) / m0, std::log(m / prev), weightedDiv(st));
            prev = m;
        }
        check(worst <= 0.05, "B: the sealed box's total smoke stays within 5 % over 3 s");
    }

    // ── C: walls keep their two sides apart ──────────────────────────────────
    // A wall across the whole box centred at x = `at`; the source is on the -x side and blows at it.
    // Returns the smoke beyond the wall (one cell clear of it) over the smoke before it.
    double leakThrough(Rig& rig, float at, float thickness, float windX, const char* label) {
        Sim  sim(rig, Vector3(0.f, 1.f, 0.f), Vector3(1.f, 1.f, 1.f), 64);
        auto& gs      = sim.gs();
        gs.solidFloor = true;
        gs.wind       = Vector3(windX, 0.f, 0.f);
        ParticleField::GridSmoke::Box b;
        b.min = Vector3(at - 0.5f * thickness, -0.2f, -1.2f);
        b.max = Vector3(at + 0.5f * thickness, 2.2f, 1.2f);
        gs.boxes.push_back(b);
        auto s     = plumeSource(Vector3(-0.5f, 0.4f, 0.f), 0.25f);
        s.velocity = Vector3(2.f, 0.5f, 0.f);
        gs.sources.push_back(s);
        sim.runToSeconds(3.0);
        State st;
        if (!sim.read(st)) return 1.0;
        const View v = sim.view(st);
        // How many cells the solver made solid along the x axis through the box's middle.
        int solidCells = 0;
        for (int i = 0; i < v.n; ++i) solidCells += v.solid(i, v.n / 2, v.n / 2) ? 1 : 0;
        const float  clear = 0.5f * thickness + v.cell.x;
        const double far   = smokeIn(v, [&](const Vector3& w) { return w.x > at + clear; });
        const double near  = smokeIn(v, [&](const Vector3& w) { return w.x < at - clear; });
        const double ratio = far / std::max(near, 1e-12);
        std::printf("  %-44s  %d solid cells  near %.4f  far %.6f  far/near %.2e\n", label, solidCells, near, far,
                    ratio);
        return ratio;
    }

    void caseWalls(Rig& rig) {
        std::printf("\n== C  walls: a wall across the box, the source blowing at it, 3 s, 64^3 (cell 3.1 cm) ==\n");
        const float  cell  = 2.f / 64.f;
        const double thick = leakThrough(rig, 0.f, 0.10f, 0.f, "0.10 m wall");
        // Centred on a cell centre, so the solid mask (a cell within half a cell of the wall) takes one cell only.
        const double thin  = leakThrough(rig, 0.5f * cell, 0.016f, 0.f, "0.016 m wall on a cell centre");
        const double windy = leakThrough(rig, 0.f, 0.10f, 3.f, "0.10 m wall, wind 3 m/s (1.6 cells a step)");
        check(thick <= 1e-3, "C: a 0.10 m wall keeps the far side clear (far/near <= 1e-3)");
        check(thin <= 1e-3, "C: a wall one cell thick keeps the far side clear (far/near <= 1e-3)");
        check(windy <= 1e-3, "C: a 0.10 m wall keeps the far side clear in a wind of 1.6 cells a step");
    }

    // ── D: a symmetric set-up stays symmetric ────────────────────────────────
    void caseSymmetry(Rig& rig) {
        std::printf("\n== D  symmetry: a plume on the box's centre line, noise off, 64^3 ==\n");
        Sim  sim(rig, Vector3(0.f, 1.f, 0.f), Vector3(1.f, 1.f, 1.f), 64);
        auto& gs      = sim.gs();
        gs.solidFloor = true;
        gs.sources.push_back(plumeSource(Vector3(0.f, 0.3f, 0.f), 0.2f));
        double worst = 0.0;
        for (double t : {0.5, 1.0, 1.5}) {
            sim.runToSeconds(t);
            State st;
            if (!sim.read(st)) {
                check(false, "D: the solver state reads back");
                return;
            }
            const View v    = sim.view(st);
            float      peak = 0.f;
            for (float d : st.density) peak = std::max(peak, d);
            double ax = 0.0, az = 0.0, vx = 0.0, vpeak = 0.0;
            for (int k = 0; k < v.n; ++k)
                for (int j = 0; j < v.n; ++j)
                    for (int i = 0; i < v.n; ++i) {
                        const float d = st.density[v.id(i, j, k)];
                        ax = std::max(ax, double(std::abs(d - st.density[v.id(v.n - 1 - i, j, k)])));
                        az = std::max(az, double(std::abs(d - st.density[v.id(i, j, v.n - 1 - k)])));
                        // The x velocity mirrors with a sign.
                        const float u  = st.velocity[v.id(i, j, k) * 3u];
                        const float um = st.velocity[v.id(v.n - 1 - i, j, k) * 3u];
                        vx             = std::max(vx, double(std::abs(u + um)));
                        vpeak          = std::max(vpeak, double(v.vel(i, j, k).length()));
                    }
            const double rel = std::max(ax, az) / std::max(double(peak), 1e-12);
            worst            = std::max(worst, rel);
            std::printf("  t = %.1f s  peak %.3f  density asym x %.2e z %.2e (rel %.2e)  u_x asym %.2e of %.2f m/s\n",
                        t, peak, ax, az, rel, vx, vpeak);
        }
        check(worst <= 0.01, "D: the plume stays mirror-symmetric to 1 % of its peak density through 1.5 s");
    }

    // ── E: still solids in still air move nothing ────────────────────────────
    void caseStill(Rig& rig) {
        std::printf("\n== E  still: a resting cloud beside a still sphere and a still block, no buoyancy, 64^3 ==\n");
        Sim  sim(rig, Vector3(0.f, 1.f, 0.f), Vector3(1.f, 1.f, 1.f), 64);
        auto& gs        = sim.gs();
        gs.solidFloor   = true;
        gs.buoyancy     = 0.f;
        gs.weight       = 0.f;
        gs.dissipation  = 0.f;
        gs.boundaryFade = 0.f;
        ParticleField::GridSmoke::Collider ball;
        ball.a = ball.b = Vector3(0.25f, 0.8f, 0.f);
        ball.radius     = 0.25f;
        gs.colliders.push_back(ball);
        ParticleField::GridSmoke::Box block;
        block.min = Vector3(-0.3f, 0.f, -0.6f);
        block.max = Vector3(0.3f, 0.4f, -0.3f);
        gs.boxes.push_back(block);
        // A source that only adds smoke: no heat, no pull on the air.
        ParticleField::GridSmoke::Source s;
        s.position      = Vector3(-0.2f, 0.7f, -0.1f);
        s.radius        = 0.35f;
        s.density       = 4.f;
        s.temperature   = 0.f;
        s.velocityBlend = 0.f;
        s.noise         = 0.f;
        gs.sources.push_back(s);
        sim.runToSeconds(0.5);
        gs.sources.clear();

        const auto summary = [&](const State& st, double& total, Vector3& centroid, double& vmax) {
            const View v = sim.view(st);
            total        = 0.0;
            vmax         = 0.0;
            double cx = 0.0, cy = 0.0, cz = 0.0;
            for (int k = 0; k < v.n; ++k)
                for (int j = 0; j < v.n; ++j)
                    for (int i = 0; i < v.n; ++i) {
                        if (v.solid(i, j, k)) continue;
                        const double  d = st.density[v.id(i, j, k)];
                        const Vector3 w = v.world(i, j, k);
                        total += d;
                        cx += d * w.x;
                        cy += d * w.y;
                        cz += d * w.z;
                        vmax = std::max(vmax, double(v.vel(i, j, k).length()));
                    }
            centroid.set(float(cx / total), float(cy / total), float(cz / total));
            total *= v.cellVolume();
        };
        State st;
        if (!sim.read(st)) {
            check(false, "E: the solver state reads back");
            return;
        }
        double  m0, v0;
        Vector3 c0;
        summary(st, m0, c0, v0);
        sim.runToSeconds(1.5);
        if (!sim.read(st)) return;
        double  m1, v1;
        Vector3 c1;
        summary(st, m1, c1, v1);
        const float cell  = 2.f / 64.f;
        const float shift = c1.distanceTo(c0);
        std::printf("  t = 0.5 s  total %.5f  max |v| %.2e m/s\n", m0, v0);
        std::printf("  t = 1.5 s  total %.5f (%+.3f %%)  max |v| %.2e m/s  centroid moved %.4f cells\n", m1,
                    100.0 * (m1 - m0) / m0, v1, shift / cell);
        check(std::max(v0, v1) <= 1e-3, "E: no air moves (max |v| <= 1 mm/s)");
        check(std::abs(m1 - m0) / m0 <= 0.005, "E: the resting cloud keeps its smoke (within 0.5 %)");
        check(shift <= 0.25f * cell, "E: the resting cloud does not drift (centroid within a quarter cell)");
    }

    // ── F: the plume does not depend on the grid ─────────────────────────────
    struct Plume {
        double total, centroid, top;
    };

    Plume plumeAt(Rig& rig, const Vector3& half, std::uint32_t res, double seconds) {
        // The floor at y = 0 whatever the box's height, so the plumes start in the same place.
        Sim  sim(rig, Vector3(0.f, half.y, 0.f), half, res);
        auto& gs      = sim.gs();
        gs.solidFloor = true;
        gs.sources.push_back(plumeSource(Vector3(0.f, 0.3f, 0.f), 0.25f));
        sim.runToSeconds(seconds);
        State st;
        if (!sim.read(st)) return {0.0, 0.0, 0.0};
        const View          v = sim.view(st);
        std::vector<double> layer(v.n, 0.0);// smoke per metre of height
        double              total = 0.0, my = 0.0;
        for (int k = 0; k < v.n; ++k)
            for (int j = 0; j < v.n; ++j)
                for (int i = 0; i < v.n; ++i) {
                    const double d = st.density[v.id(i, j, k)] * v.cellVolume();
                    total += d;
                    my += d * v.world(i, j, k).y;
                    layer[j] += d / v.cell.y;
                }
        const double peak = *std::max_element(layer.begin(), layer.end());
        double       top  = 0.0;
        for (int j = 0; j < v.n; ++j)
            if (layer[j] >= 0.02 * peak) top = v.world(0, j, 0).y + 0.5 * v.cell.y;
        return {total, my / std::max(total, 1e-12), top};
    }

    void caseResolution(Rig& rig) {
        std::printf("\n== F  resolution: the same plume at 1 s in a 3 m box ==\n");
        const Vector3 cube(1.5f, 1.5f, 1.5f), tall(1.5f, 2.25f, 1.5f);
        struct Run {
            const char*   label;
            Vector3       half;
            std::uint32_t res;
            Plume         p{};
        };
        std::vector<Run> runs = {{"64^3, cubic 4.7 cm cells", cube, 64},
                                 {"96^3, cubic 3.1 cm cells", cube, 96},
                                 {"128^3, cubic 2.3 cm cells", cube, 128},
                                 {"96^3, 3.1 x 4.7 x 3.1 cm cells", tall, 96}};
        for (Run& r : runs) {
            r.p = plumeAt(rig, r.half, r.res, 1.0);
            std::printf("  %-32s  total %.4f  centroid %.3f m  top %.3f m\n", r.label, r.p.total, r.p.centroid,
                        r.p.top);
        }
        const Plume& ref   = runs[2].p;
        const auto   close = [](double a, double b, double tol) { return std::abs(a - b) <= tol * std::abs(b); };
        bool         res   = true;
        for (int i = 0; i < 2; ++i)
            res = res && close(runs[i].p.total, ref.total, 0.10) && close(runs[i].p.centroid, ref.centroid, 0.10) &&
                  close(runs[i].p.top, ref.top, 0.10);
        check(res, "F: 64^3 and 96^3 match 128^3 to 10 % (total smoke, centroid height, plume top)");
        const Plume& c = runs[1].p;
        const Plume& t = runs[3].p;
        check(close(t.total, c.total, 0.10) && close(t.centroid, c.centroid, 0.10) && close(t.top, c.top, 0.10),
              "F: cells 1.5x taller than wide give the same plume as cubic cells to 10 %");
    }

}// namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    setEnv("THREEPP_VULKAN_VALIDATION", "1");// before the instance exists

    std::unique_ptr<Canvas>         canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters().title("VulkanGridSmoke_test").size(96, 96).vsync(false).headless(true));
        rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan GPU unavailable: %s\n", e.what());
        return kSkipCode;
    }
    VulkanRenderer& renderer = *rendererPtr;
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setAutoExposure(false);

    Scene scene;
    scene.add(AmbientLight::create(0xffffff, 1.f));
    auto camera = PerspectiveCamera::create(60.f, 1.f, 0.1f, 100.f);
    camera->position.set(0.f, 1.5f, 7.f);
    camera->lookAt(Vector3(0.f, 1.2f, 0.f));
    Rig rig{*canvasPtr, renderer, scene, *camera};

    const std::uint32_t errorsBefore = vulkan::validationErrorCount();

    // An argument picks cases by letter ("BD" runs B and D); none runs them all.
    const std::string only = argc > 1 ? argv[1] : "";
    const auto        want = [&](char c) { return only.empty() || only.find(c) != std::string::npos; };
    if (want('A')) caseProjection(rig);
    if (want('B')) caseSealedBox(rig);
    if (want('C')) caseWalls(rig);
    if (want('D')) caseSymmetry(rig);
    if (want('E')) caseStill(rig);
    if (want('F')) caseResolution(rig);

    std::printf("\n");
    if (vulkan::validationActive()) {
        const std::uint32_t errs = vulkan::validationErrorCount() - errorsBefore;
        check(errs == 0, "the solver provokes no validation errors (" + std::to_string(errs) + ")");
    } else {
        std::printf("[note] the validation layer is not installed: the zero-error check did not run\n");
    }
    std::printf("\n%s: %d check(s) failed\n", failures ? "FAILED" : "OK", failures);
    return failures ? 1 : 0;
}
