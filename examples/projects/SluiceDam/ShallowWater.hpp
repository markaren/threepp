// The water of the sluice dam example: the shallow-water equations on a regular grid, a depth per cell and a velocity
// per face (a staggered grid), stepped every frame.
//
// Per substep (the classic real-time height-field fluid, with the hydrostatic reconstruction of Audusse et al. at
// every face):
//   1. the face velocities are carried along the flow (semi-Lagrangian: each face looks back along its velocity);
//   2. gravity: each face is pushed by the difference in water level across it, both levels first clamped to the
//      higher of the two beds, so water over a ledge is driven by its own depth at the lip, not by the drop below it,
//      and still water never climbs a dry bank;
//   3. friction: Manning's law, implicit, so a film over a rock slows to nothing instead of overshooting;
//   4. the depths move by the fluxes (velocity times the upwind depth), and no cell gives away more than it holds;
//   5. foam: made where the water is white (piling up, a steep surface, a fast shallow chute), carried, aged.
// The sluice gate is a set of faces whose flux depth is capped by the gate's opening.
//
// Every loop runs over rows (threepp::ThreadPool: a step forks a few dozen short loops, which a pool that stays awake
// between them takes in microseconds) and within a row only over its span of active cells. Each row writes only its own cells and faces, and the rows' sums are kept per row and
// added up serially, so a step gives the same numbers on any number of threads.

#ifndef THREEPP_EXAMPLE_SLUICE_SHALLOW_WATER_HPP
#define THREEPP_EXAMPLE_SLUICE_SHALLOW_WATER_HPP

#include "threepp/utils/ThreadPool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace sluice {

    class ShallowWater {

    public:
        // the scheme's constants
        float gravity = 9.81f;
        float cfl = 0.45f;          // the substep: this share of a cell per step at the fastest wave
        float dryDepth = 1e-4f;     // m: less water than this does not move
        float showDepth = 0.004f;   // m: thinner water is drawn as dry ground
        float tuck = 0.06f;         // m: dry vertices of the surface hide this far under the ground
        float maxDt = 1.f / 30;     // s: a longer frame is simulated as this
        int maxSubsteps = 10;
        // foam: sources per second and its life
        float foamLife = 4.5f;
        float foamPiling = 0.12f;
        float foamSlope = 0.8f;
        float foamSlopeMin = 0.25f;
        float foamChute = 0.45f;
        float foamChuteSpeed = 0.8f;
        float foamMaxRate = 1.8f;

        int nx, nz;
        float dx, x0, z0;

        ShallowWater(int nx, int nz, float dx, float x0, float z0)
            : nx(nx), nz(nz), dx(dx), x0(x0), z0(z0) {
            const size_t cells = static_cast<size_t>(nx) * nz;
            const size_t xFaces = static_cast<size_t>(nx + 1) * nz;
            const size_t zFaces = static_cast<size_t>(nx) * (nz + 1);
            for (auto* v : {&bed_, &floor_, &cover_, &h_, &h2_, &foam_, &foam2_, &source_, &scale_}) v->assign(cells, 0.f);
            manning_.assign(cells, 0.045f);
            sinkLevel_.assign(cells, 0.f);
            sink_.assign(cells, 0);
            active_.assign(cells, 1);
            for (auto* v : {&u_, &u2_, &hu_, &fu_}) v->assign(xFaces, 0.f);
            for (auto* v : {&w_, &w2_, &hw_, &fw_}) v->assign(zFaces, 0.f);
            uGate_.assign(xFaces, -1);
            wGate_.assign(zFaces, -1);
            lo_.assign(nz, 0);
            hi_.assign(nz, nx - 1);
        }

        [[nodiscard]] int cell(int i, int j) const { return j * nx + i; }
        [[nodiscard]] float xOf(int i) const { return x0 + (static_cast<float>(i) + 0.5f) * dx; }
        [[nodiscard]] float zOf(int j) const { return z0 + (static_cast<float>(j) + 0.5f) * dx; }

        // ---- set up ------------------------------------------------------------------------------------------------
        // bed: what the water flows over (the ground, raised by the dam and the stones); floor: the drawn ground (dry
        // vertices of the surface hide under it); cover: the drawn top of each cell (ground, stone or dam: the water's
        // level is carried under a bank only where it lies under this)
        void setBed(const std::vector<float>& bed, const std::vector<float>& floor, const std::vector<float>& cover) {
            bed_ = bed;
            floor_ = floor;
            cover_ = cover;
            surfaced_ = false;
        }

        void setActive(const std::vector<uint8_t>& active) {
            active_ = active;
            for (int j = 0; j < nz; ++j) {
                int lo = nx, hi = -1;
                for (int i = 0; i < nx; ++i) {
                    if (active_[cell(i, j)]) {
                        lo = std::min(lo, i);
                        hi = std::max(hi, i);
                    } else {
                        h_[cell(i, j)] = 0;
                    }
                }
                lo_[j] = lo;
                hi_[j] = hi;
            }
            surfaced_ = false;
        }

        void setSource(int c, float metresPerSecond) { source_[c] = metresPerSecond; }
        // a sink: the cell's water is held at this level (what runs in leaves the grid)
        void setSink(int c, float level) {
            sink_[c] = 1;
            sinkLevel_[c] = level;
        }
        void setDepth(int c, float depth) { h_[c] = active_[c] ? std::max(depth, 0.f) : 0.f; }

        // a gate: the z faces between rows j - 1 and j, columns i0..i1
        void addGate(int j, int i0, int i1) {
            for (int i = i0; i <= i1; ++i) wGate_[wf(i, j)] = 0;
        }
        void setGate(float opening) { gate_ = std::max(opening, 0.f); }

        // ---- run ---------------------------------------------------------------------------------------------------
        // advance by dt (clamped to maxDt) in CFL substeps; returns the substeps taken
        int step(float dt) {
            dt = std::min(dt, maxDt);
            if (!(dt > 0)) return 0;
            const float stable = cfl * dx / std::max(maxWaveSpeed(), 0.5f);
            const int n = std::clamp(static_cast<int>(std::ceil(dt / stable)), 1, maxSubsteps);
            const float sub = std::min(dt / static_cast<float>(n), stable);
            for (int k = 0; k < n; ++k) substep(sub);
            return n;
        }

        // simulate this many seconds at the largest stable substeps (the start)
        void settle(float seconds) {
            for (float t = 0; t < seconds;) {
                const float sub = std::min({cfl * dx / std::max(maxWaveSpeed(), 0.5f), 1.f / 60, seconds - t + 1e-6f});
                substep(sub);
                t += sub;
            }
        }

        [[nodiscard]] float depth(int c) const { return h_[c]; }
        [[nodiscard]] float level(int c) const { return bed_[c] + h_[c]; }
        [[nodiscard]] float volume() const { return volume_; }
        [[nodiscard]] float gateFlux() const { return gateFlux_; }// m^3/s through the gate

        // ---- draw --------------------------------------------------------------------------------------------------
        // Per cell a vertex (x, y, z), its normal, the water's velocity (vx, vz) and (foam, depth). The first call
        // fills every cell; later ones only the rows' active spans (with a margin), since nothing else changes.
        void surface(float* position, float* normal, float* velocity, float* water) {
            const bool all = !surfaced_;
            surfaced_ = true;
            const auto span = [&](int j, int& a, int& b) {
                if (all) {
                    a = 0;
                    b = nx - 1;
                    return;
                }
                int lo = nx, hi = -1;
                for (int jj = std::max(j - 1, 0); jj <= std::min(j + 1, nz - 1); ++jj) {
                    lo = std::min(lo, lo_[jj]);
                    hi = std::max(hi, hi_[jj]);
                }
                a = std::max(lo - 2, 0);
                b = std::min(hi + 2, nx - 1);
            };
            rows(0, nz, [&](int j) {
                int a, b;
                span(j, a, b);
                for (int i = a; i <= b; ++i) {
                    const int c = cell(i, j);
                    const bool wet = active_[c] && h_[c] > showDepth;
                    float y;
                    if (wet) {
                        y = bed_[c] + h_[c];
                    } else {
                        // next to the water its level carries on flat where it lies under what is drawn here (so the
                        // water's edge is where a level surface meets the shore); elsewhere the vertex hides under the
                        // ground
                        float carried = -1e30f;
                        for (int dj = -1; dj <= 1; ++dj) {
                            for (int di = -1; di <= 1; ++di) {
                                const int ii = i + di, jj = j + dj;
                                if ((di == 0 && dj == 0) || ii < 0 || ii >= nx || jj < 0 || jj >= nz) continue;
                                const int n = cell(ii, jj);
                                if (active_[n] && h_[n] > showDepth) carried = std::max(carried, bed_[n] + h_[n]);
                            }
                        }
                        y = carried > -1e29f && carried < cover_[c] ? carried : std::min(bed_[c], floor_[c]) - tuck;
                    }
                    position[3 * c] = xOf(i);
                    position[3 * c + 1] = y;
                    position[3 * c + 2] = zOf(j);
                    velocity[2 * c] = 0.5f * (u_[uf(i, j)] + u_[uf(i + 1, j)]);
                    velocity[2 * c + 1] = 0.5f * (w_[wf(i, j)] + w_[wf(i, j + 1)]);
                    water[2 * c] = std::clamp(foam_[c], 0.f, 1.f);
                    water[2 * c + 1] = wet ? h_[c] : 0.f;
                }
            });
            rows(0, nz, [&](int j) {
                int a, b;
                span(j, a, b);
                for (int i = a; i <= b; ++i) {
                    const int il = std::max(i - 1, 0), ir = std::min(i + 1, nx - 1);
                    const int jb = std::max(j - 1, 0), jt = std::min(j + 1, nz - 1);
                    const float gx = (position[3 * cell(ir, j) + 1] - position[3 * cell(il, j) + 1]) / (static_cast<float>(ir - il) * dx);
                    const float gz = (position[3 * cell(i, jt) + 1] - position[3 * cell(i, jb) + 1]) / (static_cast<float>(jt - jb) * dx);
                    const float inv = 1.f / std::sqrt(gx * gx + 1.f + gz * gz);
                    float* n = normal + 3 * cell(i, j);
                    n[0] = -gx * inv;
                    n[1] = inv;
                    n[2] = -gz * inv;
                }
            });
        }

    private:
        std::vector<float> bed_, floor_, cover_, h_, h2_, foam_, foam2_, source_, manning_, sinkLevel_, scale_;
        std::vector<uint8_t> active_, sink_;
        std::vector<float> u_, u2_, hu_, fu_, w_, w2_, hw_, fw_;
        std::vector<int8_t> uGate_, wGate_;
        std::vector<int> lo_, hi_;
        float gate_ = 0, volume_ = 0, gateFlux_ = 0;
        bool surfaced_ = false;

        // fn(j) for every row j in [j0, j1), the rows spread over the cores
        template<class Fn>
        void rows(int j0, int j1, Fn&& fn) const {
            threepp::ThreadPool::shared().parallelFor(j0, j1, fn);
        }

        [[nodiscard]] int uf(int i, int j) const { return j * (nx + 1) + i; }// x face left of cell (i, j)
        [[nodiscard]] int wf(int i, int j) const { return j * nx + i; }      // z face below cell (i, j)
        // the z faces of row j lie between cell rows j - 1 and j: where both have active cells
        [[nodiscard]] int wLo(int j) const { return (j <= 0 || j >= nz) ? nx : std::max(lo_[j - 1], lo_[j]); }
        [[nodiscard]] int wHi(int j) const { return (j <= 0 || j >= nz) ? -1 : std::min(hi_[j - 1], hi_[j]); }

        static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

        [[nodiscard]] float sampleU(float fx, float fz) const {
            fx = clampf(fx, 0, static_cast<float>(nx));
            fz = clampf(fz, 0, static_cast<float>(nz - 1));
            const int i0 = std::min(static_cast<int>(fx), nx - 1), j0 = std::min(static_cast<int>(fz), std::max(nz - 2, 0));
            const int i1 = std::min(i0 + 1, nx), j1 = std::min(j0 + 1, nz - 1);
            const float tx = fx - static_cast<float>(i0), tz = fz - static_cast<float>(j0);
            const float a = u_[uf(i0, j0)] * (1 - tx) + u_[uf(i1, j0)] * tx;
            const float b = u_[uf(i0, j1)] * (1 - tx) + u_[uf(i1, j1)] * tx;
            return a * (1 - tz) + b * tz;
        }

        [[nodiscard]] float sampleW(float fx, float fz) const {
            fx = clampf(fx, 0, static_cast<float>(nx - 1));
            fz = clampf(fz, 0, static_cast<float>(nz));
            const int i0 = std::min(static_cast<int>(fx), std::max(nx - 2, 0)), j0 = std::min(static_cast<int>(fz), nz - 1);
            const int i1 = std::min(i0 + 1, nx - 1), j1 = std::min(j0 + 1, nz);
            const float tx = fx - static_cast<float>(i0), tz = fz - static_cast<float>(j0);
            const float a = w_[wf(i0, j0)] * (1 - tx) + w_[wf(i1, j0)] * tx;
            const float b = w_[wf(i0, j1)] * (1 - tx) + w_[wf(i1, j1)] * tx;
            return a * (1 - tz) + b * tz;
        }

        [[nodiscard]] float sampleCell(const std::vector<float>& f, float fx, float fz) const {
            fx = clampf(fx, 0, static_cast<float>(nx - 1));
            fz = clampf(fz, 0, static_cast<float>(nz - 1));
            const int i0 = std::min(static_cast<int>(fx), std::max(nx - 2, 0)), j0 = std::min(static_cast<int>(fz), std::max(nz - 2, 0));
            const int i1 = std::min(i0 + 1, nx - 1), j1 = std::min(j0 + 1, nz - 1);
            const float tx = fx - static_cast<float>(i0), tz = fz - static_cast<float>(j0);
            const float a = f[cell(i0, j0)] * (1 - tx) + f[cell(i1, j0)] * tx;
            const float b = f[cell(i0, j1)] * (1 - tx) + f[cell(i1, j1)] * tx;
            return a * (1 - tz) + b * tz;
        }

        [[nodiscard]] float maxWaveSpeed() const {
            std::vector<float> row(nz, 0.f);
            rows(0, nz, [&](int j) {
                float m = 0;
                for (int i = lo_[j]; i <= hi_[j]; ++i) {
                    const float h = h_[cell(i, j)];
                    if (h <= dryDepth) continue;
                    const float vx = 0.5f * (u_[uf(i, j)] + u_[uf(i + 1, j)]);
                    const float vz = 0.5f * (w_[wf(i, j)] + w_[wf(i, j + 1)]);
                    m = std::max(m, std::sqrt(gravity * h) + std::sqrt(vx * vx + vz * vz));
                }
                row[j] = m;
            });
            return *std::max_element(row.begin(), row.end());
        }

        // gravity across a face between cells a and b (velocity v0 carried there), the gate, friction: the new
        // velocity and the depth the flux takes (the upwind one)
        void face(int a, int b, float v0, bool gated, float dt, float& v, float& hFace) const {
            v = 0;
            hFace = 0;
            if (!active_[a] || !active_[b]) return;
            const float top = std::max(bed_[a], bed_[b]);
            const float ha = std::max(0.f, bed_[a] + h_[a] - top);
            const float hb = std::max(0.f, bed_[b] + h_[b] - top);
            if (ha <= dryDepth && hb <= dryDepth) return;
            float vel = v0 - gravity * dt * (hb - ha) / dx;
            float up = vel > 0 ? ha : (vel < 0 ? hb : std::max(ha, hb));
            if (gated) up = std::min(up, gate_);
            if (up <= dryDepth) return;
            const float n = 0.5f * (manning_[a] + manning_[b]);
            const float hh = std::max(up, 1e-3f);
            vel /= 1 + dt * gravity * n * n * std::fabs(vel) / (hh * std::cbrt(hh));
            const float vmax = 0.9f * dx / dt;
            v = clampf(vel, -vmax, vmax);
            hFace = up;
        }

        void substep(float dt) {
            const float k = dt / dx;

            // 1. the face velocities carried along the flow
            rows(0, nz, [&](int j) {
                for (int i = std::max(1, lo_[j] + 1); i <= std::min(nx - 1, hi_[j]); ++i) {
                    const int a = cell(i - 1, j), b = cell(i, j), f = uf(i, j);
                    if (!active_[a] || !active_[b] || (h_[a] <= dryDepth && h_[b] <= dryDepth)) {
                        u2_[f] = 0;
                        continue;
                    }
                    const float vz = 0.25f * (w_[wf(i - 1, j)] + w_[wf(i, j)] + w_[wf(i - 1, j + 1)] + w_[wf(i, j + 1)]);
                    u2_[f] = sampleU(static_cast<float>(i) - u_[f] * k, static_cast<float>(j) - vz * k);
                }
            });
            rows(1, nz, [&](int j) {
                for (int i = wLo(j); i <= wHi(j); ++i) {
                    const int a = cell(i, j - 1), b = cell(i, j), f = wf(i, j);
                    if (!active_[a] || !active_[b] || (h_[a] <= dryDepth && h_[b] <= dryDepth)) {
                        w2_[f] = 0;
                        continue;
                    }
                    const float vx = 0.25f * (u_[uf(i, j - 1)] + u_[uf(i + 1, j - 1)] + u_[uf(i, j)] + u_[uf(i + 1, j)]);
                    w2_[f] = sampleW(static_cast<float>(i) - vx * k, static_cast<float>(j) - w_[f] * k);
                }
            });

            // 2-3. gravity, the gate, friction
            rows(0, nz, [&](int j) {
                for (int i = std::max(1, lo_[j] + 1); i <= std::min(nx - 1, hi_[j]); ++i) {
                    const int f = uf(i, j);
                    face(cell(i - 1, j), cell(i, j), u2_[f], uGate_[f] >= 0, dt, u_[f], hu_[f]);
                }
            });
            rows(1, nz, [&](int j) {
                for (int i = wLo(j); i <= wHi(j); ++i) {
                    const int f = wf(i, j);
                    face(cell(i, j - 1), cell(i, j), w2_[f], wGate_[f] >= 0, dt, w_[f], hw_[f]);
                }
            });

            // 4. the fluxes, scaled so no cell gives away more than it holds
            rows(0, nz, [&](int j) {
                for (int i = std::max(1, lo_[j] + 1); i <= std::min(nx - 1, hi_[j]); ++i) fu_[uf(i, j)] = u_[uf(i, j)] * hu_[uf(i, j)];
            });
            rows(1, nz, [&](int j) {
                for (int i = wLo(j); i <= wHi(j); ++i) fw_[wf(i, j)] = w_[wf(i, j)] * hw_[wf(i, j)];
            });
            rows(0, nz, [&](int j) {
                for (int i = lo_[j]; i <= hi_[j]; ++i) {
                    const int c = cell(i, j);
                    const float out = k * (std::max(0.f, fu_[uf(i + 1, j)]) + std::max(0.f, -fu_[uf(i, j)]) +
                                           std::max(0.f, fw_[wf(i, j + 1)]) + std::max(0.f, -fw_[wf(i, j)]));
                    scale_[c] = out > h_[c] && out > 0 ? h_[c] / out : 1.f;
                }
            });
            rows(0, nz, [&](int j) {
                for (int i = std::max(1, lo_[j] + 1); i <= std::min(nx - 1, hi_[j]); ++i) {
                    float& F = fu_[uf(i, j)];
                    F *= F > 0 ? scale_[cell(i - 1, j)] : scale_[cell(i, j)];
                }
            });
            rows(1, nz, [&](int j) {
                for (int i = wLo(j); i <= wHi(j); ++i) {
                    float& F = fw_[wf(i, j)];
                    F *= F > 0 ? scale_[cell(i, j - 1)] : scale_[cell(i, j)];
                }
            });

            // 5. the depths; a sink's cells held at its level
            std::vector<double> rowVolume(nz, 0.0), rowGate(nz, 0.0);
            rows(0, nz, [&](int j) {
                double vol = 0, gate = 0;
                for (int i = lo_[j]; i <= hi_[j]; ++i) {
                    const int c = cell(i, j);
                    if (!active_[c]) continue;
                    float h = h_[c] + k * (fu_[uf(i, j)] - fu_[uf(i + 1, j)] + fw_[wf(i, j)] - fw_[wf(i, j + 1)]) + dt * source_[c];
                    h = std::max(0.f, h);
                    if (sink_[c]) h = std::max(0.f, sinkLevel_[c] - bed_[c]);
                    h2_[c] = h;
                    vol += static_cast<double>(h) * dx * dx;
                    if (wGate_[wf(i, j)] >= 0) gate += static_cast<double>(fw_[wf(i, j)]) * dx;
                }
                rowVolume[j] = vol;
                rowGate[j] = gate;
            });
            rows(0, nz, [&](int j) {
                for (int i = lo_[j]; i <= hi_[j]; ++i) {
                    const int c = cell(i, j);
                    if (active_[c]) h_[c] = h2_[c];
                }
            });
            double vol = 0, gate = 0;
            for (int j = 0; j < nz; ++j) {
                vol += rowVolume[j];
                gate += rowGate[j];
            }
            volume_ = static_cast<float>(vol);
            gateFlux_ = static_cast<float>(gate);

            // 6. foam: carried, made where the water is white, aged
            const float decay = std::exp(-dt / foamLife);
            const float dryDecay = std::exp(-dt / 0.4f);
            rows(0, nz, [&](int j) {
                for (int i = lo_[j]; i <= hi_[j]; ++i) {
                    const int c = cell(i, j);
                    const float h = h_[c];
                    if (!active_[c] || h <= showDepth) {
                        foam2_[c] = foam_[c] * dryDecay;
                        continue;
                    }
                    const float ul = u_[uf(i, j)], ur = u_[uf(i + 1, j)];
                    const float wb = w_[wf(i, j)], wt = w_[wf(i, j + 1)];
                    const float vx = 0.5f * (ul + ur), vz = 0.5f * (wb + wt);
                    const float speed = std::sqrt(vx * vx + vz * vz);
                    const float carried = sampleCell(foam_, static_cast<float>(i) - vx * k, static_cast<float>(j) - vz * k);
                    const float piling = std::max(0.f, -(ur - ul + wt - wb) / dx);
                    const float here = bed_[c] + h;
                    const auto levelAt = [&](int ii, int jj) {
                        if (ii < 0 || ii >= nx || jj < 0 || jj >= nz) return here;
                        const int n = cell(ii, jj);
                        return active_[n] && h_[n] > showDepth ? bed_[n] + h_[n] : here;
                    };
                    const float gx = (levelAt(i + 1, j) - levelAt(i - 1, j)) / (2 * dx);
                    const float gz = (levelAt(i, j + 1) - levelAt(i, j - 1)) / (2 * dx);
                    const float slope = std::sqrt(gx * gx + gz * gz);
                    const float froude = speed / std::sqrt(gravity * std::max(h, 1e-3f));
                    const float chute = std::max(0.f, froude - 1.3f) * clampf(speed - foamChuteSpeed, 0, 1);
                    // a film a few mm thick on a sloping shore makes none: the sources come in over the first 4 cm
                    const float body = clampf((h - showDepth) / 0.04f, 0, 1);
                    const float rate = body * std::min(foamMaxRate, foamPiling * piling + foamSlope * std::max(0.f, slope - foamSlopeMin) + foamChute * chute);
                    foam2_[c] = std::min(1.f, carried + dt * rate) * decay;
                }
            });
            rows(0, nz, [&](int j) {
                for (int i = lo_[j]; i <= hi_[j]; ++i) foam_[cell(i, j)] = foam2_[cell(i, j)];
            });
        }
    };

}// namespace sluice

#endif//THREEPP_EXAMPLE_SLUICE_SHALLOW_WATER_HPP
