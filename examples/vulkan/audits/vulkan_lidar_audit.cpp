// E1: RT-LIDAR replay audit — the per-modality reproducibility instrument for
// the ray-traced sensor, sibling to vulkan_aov_audit (raster AOVs) and the
// proprioceptive replay_audit under examples/extras/sensors.
//
// The question it answers: is a PathTracedLidarSensor scan seed-replayable
// across fresh processes, and when it is not, HOW BIG is the jitter and WHAT
// FRACTION of beams does it touch. A hash alone cannot answer the second half,
// so unlike the AOV audit this harness dumps every return of every scan and the
// compare works on the point records, not on a digest.
//
//     vulkan_lidar_audit --frames 120 --out runA
//     vulkan_lidar_audit --frames 120 --out runB
//     vulkan_lidar_audit --compare runA runB
//
// Exit codes of --compare: 0 = every scan bit-identical, 3 = differences that
// are all within-jitter (equal point counts, max |Δrange| < 5 cm), 1 =
// structural mismatch or I/O failure. 3 is a RESULT, not a failure: the RT path
// walks the renderer's TLAS, and TLAS builds are scheduling-dependent, so the
// prior on this modality is "not bit-exact". The number the paper needs is the
// size of that non-exactness.
//
// Beam order is the record order. PathTracedLidarSensor lays its returns out as
// (beam * samplesPerBeam + sample) * maxReturns + slot, with the beam table
// built azimuth-major / elevation-minor from the LidarModel; at the defaults
// (1 sample, 1 return) record index IS beam index. Nothing is sorted, because
// nothing needs to be: the layout is a pure function of the model, so the two
// runs align index for index.
//
// Range noise is deliberately OFF (sigma 0). The seed is recorded in the
// manifest and the stream is seeded, but a detector-noise draw replays
// identically in both runs and would only add a constant to both sides of the
// subtraction — measuring the tracer's own range keeps the delta attributable
// to the tracer. --noise-sigma turns it on for a domain-randomization check.
//
// ── --scene-edit: the perturbation refit-only dynamics never reach ──────────
// The baseline script only ever moves instances: same topology, same instance
// order. --scene-edit adds a mesh at frame 45 and removes it at frame 81, and at
// those same two frames pulls an EXISTING mid-list object (the spinner) out and
// later puts it back. The second half is what makes this a reordering test: an
// appended entry removed from the tail shifts nothing behind it, whereas a
// mid-list removal moves every later entry, and the re-add lands at the tail
// rather than in the original slot. The post-revert scene therefore holds the
// same five objects in a DIFFERENT order. All edits are keyed to the scripted
// frame index, so every run performs them at the same points.
//
// In this mode the sensor's yaw is PINNED. A within-run id comparison needs the
// pre-edit and post-revert scans to have been fired from the same pose;
// otherwise the range gate below only selects surfaces whose range happens to
// be rotationally invariant (the ground), and the pillar — the other static
// instance, and the more interesting one — drops out of the sample.
//
// ── --idstability: does a return's instance id survive a rebuild ────────────
// This mode found the defect and now guards the fix. Returns originally carried
// the raw TLAS instance index, so the pillar's static, unmoved surface reported
// id 3 before the churn and id 2 after it, on every one of the ~1.8k beams that
// struck it at an identical range — perception code keying on hitInstanceId
// across a scene edit was keying on a number that moved under it. The renderer
// now translates the index to the stable setObjectInstanceId label at readback,
// so the ids printed below are the harness scene's own labels (ground = 1,
// pillar = 4) and they must not move across the edit.
//
// This mode measures it within one run: take a pre-edit scan and a post-revert
// scan, keep the beams that returned in BOTH and whose |Δrange| < 1e-6 (same
// unmoved surface — mover, spinner and waver are all in motion between the two
// scans, so the range gate is what isolates the static geometry), and count how
// many of those changed hitInstanceId. Any change is a real finding: the beam
// demonstrably struck the same surface at the same distance.
//
// Returns carry hitInstanceId; it is preserved in the dump. The scene has no
// fog, so no volume-scatter sentinels (hitInstanceId == -2) are expected — but
// the compare reports instance-id disagreement separately, because a beam that
// flips which instance it struck is a different failure from one that reports a
// slightly different range off the same instance.
//
// ── --sonar: the imaging sonar joins the same scan ─────────────────────────
// A SonarSensor (Wide130: 256 beams x 16 samples, 512 bins, 20 m) rides the
// lidar's pose and fires on the lidar's schedule. Each echogram is dumped as
// sonar_NNN.bin and hashed into the manifest; --compare reports it bit-exact or
// gives the number of differing bins and the max |Δecho| (echo in [0, 1]). The
// sonar is the same tracer seen through a different fold — a max per bin, which
// is order-independent — so the prior here is "bit-exact whenever the lidar is",
// and the row exists to make that a measurement rather than an argument.

#include "threepp/threepp.hpp"

#include "threepp/helpers/LidarModel.hpp"
#include "threepp/helpers/PathTracedLidarSensor.hpp"
#include "threepp/helpers/SonarSensor.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    // FNV-1a 64, chained: order is part of the contract.
    class Fnv {
    public:
        void bytes(const void* p, std::size_t n) {
            const auto* b = static_cast<const unsigned char*>(p);
            for (std::size_t i = 0; i < n; ++i) {
                h_ ^= b[i];
                h_ *= 0x100000001B3ULL;
            }
        }
        [[nodiscard]] std::uint64_t value() const { return h_; }

    private:
        std::uint64_t h_ = 0xCBF29CE484222325ULL;
    };

    // On-disk scan record. Packed and written field-wise through memcpy of the
    // whole struct: this is a scratch artefact consumed by this binary on the
    // same machine, so host byte order is the contract and is asserted by the
    // magic + a size check on load rather than by a serialization layer.
#pragma pack(push, 1)
    struct DumpHeader {
        char magic[8];// "TPLIDAR1"
        std::uint32_t version;
        std::uint32_t scanIndex;
        std::uint32_t beamCount;
        std::uint32_t elevCount;// rows in the model (minor axis of the beam table)
        std::uint32_t azSteps;  // columns (major axis)
        std::uint32_t pointCount;
        double simTime;
    };
    struct DumpRec {
        float distance;// slant range, m; 0 on miss
        float intensity;
        std::int32_t hitInstanceId;// >=0 surface, -1 miss, -2 volume scatter
        std::int32_t returnNo;     // >0 = real return
        std::int32_t returnKind;   // 0 surface, 1 volume scatter
        float px, py, pz;          // world hit point
    };
    // Echogram dump: header + beams*bins floats, beam-major (SonarImage's own
    // layout, written as-is).
    struct SonarDumpHeader {
        char magic[8];// "TPSONAR1"
        std::uint32_t version;
        std::uint32_t scanIndex;
        std::uint32_t beams;
        std::uint32_t bins;
        std::uint32_t rays;
        std::uint32_t returns;// returns.size() of the scan (rays x maxReturns)
        double simTime;
    };
#pragma pack(pop)
    static_assert(sizeof(DumpHeader) == 40, "header layout is the file format");
    static_assert(sizeof(DumpRec) == 32, "record layout is the file format");
    static_assert(sizeof(SonarDumpHeader) == 40, "sonar header layout is the file format");

    std::string scanFile(const std::string& dir, int idx) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "scan_%03d.bin", idx);
        return dir + "/" + buf;
    }

    std::string sonarFile(const std::string& dir, int idx) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "sonar_%03d.bin", idx);
        return dir + "/" + buf;
    }

    struct SonarDump {
        SonarDumpHeader hdr{};
        std::vector<float> echo;
        std::vector<char> raw;
    };

    bool loadSonar(const std::string& path, SonarDump& out) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        out.raw.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        if (out.raw.size() < sizeof(SonarDumpHeader)) return false;
        std::memcpy(&out.hdr, out.raw.data(), sizeof(SonarDumpHeader));
        if (std::memcmp(out.hdr.magic, "TPSONAR1", 8) != 0) return false;
        const std::size_t cells = std::size_t(out.hdr.beams) * out.hdr.bins;
        if (out.raw.size() != sizeof(SonarDumpHeader) + cells * sizeof(float)) return false;
        out.echo.resize(cells);
        if (cells) std::memcpy(out.echo.data(), out.raw.data() + sizeof(SonarDumpHeader), cells * sizeof(float));
        return true;
    }

    struct Scan {
        DumpHeader hdr{};
        std::vector<DumpRec> recs;
        std::vector<char> raw;// kept so bit-exactness is a memcmp, not a field walk
    };

    bool loadScan(const std::string& path, Scan& out) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        out.raw.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        if (out.raw.size() < sizeof(DumpHeader)) return false;
        std::memcpy(&out.hdr, out.raw.data(), sizeof(DumpHeader));
        if (std::memcmp(out.hdr.magic, "TPLIDAR1", 8) != 0) return false;
        const std::size_t need = sizeof(DumpHeader) +
                                 std::size_t(out.hdr.pointCount) * sizeof(DumpRec);
        if (out.raw.size() != need) return false;
        out.recs.resize(out.hdr.pointCount);
        if (out.hdr.pointCount)
            std::memcpy(out.recs.data(), out.raw.data() + sizeof(DumpHeader),
                        out.hdr.pointCount * sizeof(DumpRec));
        return true;
    }

    std::map<std::string, std::string> loadManifest(const std::string& dir) {
        std::map<std::string, std::string> rows;
        std::ifstream f(dir + "/manifest.txt");
        if (!f) return rows;
        std::string name, rest;
        while (f >> name && std::getline(f, rest)) rows[name] = rest;
        return rows;
    }

    float percentile(const std::vector<float>& sorted, double q) {
        if (sorted.empty()) return 0.f;
        const auto i = std::min(sorted.size() - 1,
                                static_cast<std::size_t>(q * double(sorted.size() - 1) + 0.5));
        return sorted[i];
    }

    // Jitter tolerance: a difference this small is a scheduling artefact of the
    // acceleration structure, not a different scene. 5 cm on a 25 m sensor.
    constexpr float kJitterTol = 0.05f;
    // Below this a delta is float noise in the dump itself, not a measurement.
    constexpr float kDeltaEps = 1e-6f;

    int compare(const std::string& dirA, const std::string& dirB) {
        const auto manA = loadManifest(dirA);
        const auto manB = loadManifest(dirB);
        if (manA.empty() || manB.empty()) {
            std::cout << "IO FAIL: missing manifest in " << (manA.empty() ? dirA : dirB) << "\n";
            return 1;
        }
        int structural = 0;
        {
            const auto a = manA.find("meta"), b = manB.find("meta");
            if (a == manA.end() || b == manB.end() || a->second != b->second) {
                std::cout << "MISMATCH meta\n  " << dirA << ":"
                          << (a == manA.end() ? " <absent>" : a->second) << "\n  " << dirB << ":"
                          << (b == manB.end() ? " <absent>" : b->second) << "\n";
                ++structural;
            } else {
                std::cout << "OK       meta" << a->second << "\n";
            }
            // dyn.geom is REPORTED, never gating: refit bookkeeping is the
            // renderer's, and a divergence there is a finding about the
            // dynamic-geometry path rather than about the sensor.
            const auto da = manA.find("dyn.geom"), db = manB.find("dyn.geom");
            if (da != manA.end() && db != manB.end()) {
                std::cout << (da->second == db->second ? "OK       dyn.geom" : "NOTE     dyn.geom differs")
                          << da->second << (da->second == db->second ? "" : " | ") << "\n";
                if (da->second != db->second) std::cout << "         " << dirB << ":" << db->second << "\n";
            }
            // tlas IS gating: the acceleration structure the beams walked has to
            // have been built the same number of times, the same way. A
            // divergence here means the two runs did not present the tracer with
            // the same structure, and every point statistic below is then a
            // comparison of two different experiments.
            const auto ta = manA.find("tlas"), tb = manB.find("tlas");
            if (ta == manA.end() || tb == manB.end() || ta->second != tb->second) {
                std::cout << "MISMATCH tlas\n  " << dirA << ":"
                          << (ta == manA.end() ? " <absent>" : ta->second) << "\n  " << dirB << ":"
                          << (tb == manB.end() ? " <absent>" : tb->second) << "\n";
                ++structural;
            } else {
                std::cout << "OK       tlas" << ta->second << "\n";
            }
        }

        // Scan count comes from the manifests; a disagreement is structural.
        int nScans = 0;
        for (const auto& [k, v] : manA)
            if (k.rfind("scan_", 0) == 0) ++nScans;
        int nScansB = 0;
        for (const auto& [k, v] : manB)
            if (k.rfind("scan_", 0) == 0) ++nScansB;
        if (nScans != nScansB) {
            std::cout << "MISMATCH scan count " << nScans << " vs " << nScansB << "\n";
            ++structural;
            nScans = std::min(nScans, nScansB);
        }

        int bitExact = 0;
        std::vector<float> allDeltas;// |Δrange| over every beam both runs returned
        std::size_t allCompared = 0, allDiffering = 0, allPresenceFlips = 0, allIdFlips = 0;
        float globalMax = 0.f;

        for (int s = 0; s < nScans; ++s) {
            Scan a, b;
            const std::string pa = scanFile(dirA, s), pb = scanFile(dirB, s);
            if (!loadScan(pa, a) || !loadScan(pb, b)) {
                std::cout << "IO FAIL  scan_" << std::setw(3) << std::setfill('0') << s
                          << std::setfill(' ') << " (" << pa << " / " << pb << ")\n";
                ++structural;
                continue;
            }
            std::cout << "scan_" << std::setw(3) << std::setfill('0') << s << std::setfill(' ') << " ";
            if (a.raw.size() == b.raw.size() &&
                std::memcmp(a.raw.data(), b.raw.data(), a.raw.size()) == 0) {
                // The aggregate denominator is beams BOTH runs returned on, so a
                // bit-identical scan contributes its real returns and not its
                // misses — otherwise the reported fraction would shrink with the
                // miss rate rather than with the jitter.
                std::size_t hits = 0;
                for (const auto& r : a.recs)
                    if (r.returnNo > 0) ++hits;
                std::cout << "OK  bit-identical (" << a.hdr.pointCount << " pts, " << hits << " returns)\n";
                ++bitExact;
                allCompared += hits;
                continue;
            }
            if (a.hdr.pointCount != b.hdr.pointCount || a.hdr.beamCount != b.hdr.beamCount) {
                std::cout << "STRUCTURAL point count " << a.hdr.pointCount << " vs "
                          << b.hdr.pointCount << "\n";
                ++structural;
                continue;
            }

            std::vector<float> d;
            d.reserve(a.recs.size());
            std::size_t presence = 0, idFlip = 0, compared = 0;
            float mx = 0.f;
            for (std::size_t i = 0; i < a.recs.size(); ++i) {
                const auto& ra = a.recs[i];
                const auto& rb = b.recs[i];
                const bool ha = ra.returnNo > 0, hb = rb.returnNo > 0;
                if (ha != hb) {
                    // One run's beam returned and the other's did not. Almost
                    // always a grazing/edge beam flipping across the detector
                    // threshold — counted, but it has no range delta to fold in.
                    ++presence;
                    continue;
                }
                if (!ha) continue;// both missed: nothing to compare
                ++compared;
                if (ra.hitInstanceId != rb.hitInstanceId) ++idFlip;
                const float delta = std::abs(ra.distance - rb.distance);
                mx = std::max(mx, delta);
                if (delta > kDeltaEps) d.push_back(delta);
            }
            allCompared += compared;
            allDiffering += d.size();
            allPresenceFlips += presence;
            allIdFlips += idFlip;
            globalMax = std::max(globalMax, mx);
            allDeltas.insert(allDeltas.end(), d.begin(), d.end());

            std::vector<float> sorted = d;
            std::sort(sorted.begin(), sorted.end());
            double sum = 0.;
            for (float v : sorted) sum += v;
            const double frac = compared ? double(d.size()) / double(compared) : 0.;
            std::cout << "DIFF compared=" << compared
                      << " frac>1e-6=" << std::fixed << std::setprecision(6) << frac
                      << std::setprecision(6)
                      << " mean=" << (sorted.empty() ? 0.0 : sum / double(sorted.size()))
                      << " p50=" << percentile(sorted, 0.50)
                      << " p99=" << percentile(sorted, 0.99)
                      << " max=" << mx
                      << " presenceflips=" << presence
                      << " idflips=" << idFlip
                      << std::defaultfloat << "\n";
        }

        std::sort(allDeltas.begin(), allDeltas.end());
        double sum = 0.;
        for (float v : allDeltas) sum += v;
        std::cout << "BITEXACT " << bitExact << "/" << nScans << "\n";
        std::cout << "AGGREGATE compared=" << allCompared
                  << " differing=" << allDeltas.size()
                  << " frac=" << std::fixed << std::setprecision(6)
                  << (allCompared ? double(allDeltas.size()) / double(allCompared) : 0.)
                  << " mean|d|=" << (allDeltas.empty() ? 0.0 : sum / double(allDeltas.size()))
                  << " p50=" << percentile(allDeltas, 0.50)
                  << " p99=" << percentile(allDeltas, 0.99)
                  << " max=" << globalMax
                  << " presenceflips=" << allPresenceFlips
                  << " idflips=" << allIdFlips << std::defaultfloat << "\n";

        // ---- sonar rows, when the runs carried them --------------------------
        int nSonar = 0, nSonarB = 0;
        for (const auto& [k, v] : manA)
            if (k.rfind("sonar_", 0) == 0) ++nSonar;
        for (const auto& [k, v] : manB)
            if (k.rfind("sonar_", 0) == 0) ++nSonarB;
        if (nSonar != nSonarB) {
            std::cout << "MISMATCH sonar count " << nSonar << " vs " << nSonarB << "\n";
            ++structural;
            nSonar = std::min(nSonar, nSonarB);
        }
        int sonarExact = 0;
        float sonarMax = 0.f;
        std::size_t sonarCells = 0, sonarDiffering = 0;
        for (int s = 0; s < nSonar; ++s) {
            SonarDump a, b;
            const std::string pa = sonarFile(dirA, s), pb = sonarFile(dirB, s);
            if (!loadSonar(pa, a) || !loadSonar(pb, b)) {
                std::cout << "IO FAIL  sonar_" << std::setw(3) << std::setfill('0') << s
                          << std::setfill(' ') << " (" << pa << " / " << pb << ")\n";
                ++structural;
                continue;
            }
            std::cout << "sonar_" << std::setw(3) << std::setfill('0') << s << std::setfill(' ') << " ";
            const std::size_t echoes = std::size_t(std::count_if(a.echo.begin(), a.echo.end(),
                                                                 [](float v) { return v > 0.f; }));
            sonarCells += a.echo.size();
            if (a.raw.size() == b.raw.size() && std::memcmp(a.raw.data(), b.raw.data(), a.raw.size()) == 0) {
                std::cout << "OK  bit-identical (" << a.hdr.beams << "x" << a.hdr.bins << ", "
                          << echoes << " echo bins)\n";
                ++sonarExact;
                continue;
            }
            if (a.hdr.beams != b.hdr.beams || a.hdr.bins != b.hdr.bins || a.hdr.returns != b.hdr.returns) {
                std::cout << "STRUCTURAL " << a.hdr.beams << "x" << a.hdr.bins << "/" << a.hdr.returns
                          << " vs " << b.hdr.beams << "x" << b.hdr.bins << "/" << b.hdr.returns << "\n";
                ++structural;
                continue;
            }
            std::size_t differing = 0;
            float mx = 0.f;
            for (std::size_t i = 0; i < a.echo.size(); ++i) {
                const float d = std::abs(a.echo[i] - b.echo[i]);
                if (d > 0.f) ++differing;
                mx = std::max(mx, d);
            }
            sonarDiffering += differing;
            sonarMax = std::max(sonarMax, mx);
            std::cout << "DIFF bins=" << differing << "/" << a.echo.size()
                      << " max|d|=" << std::scientific << std::setprecision(3) << mx << std::defaultfloat << "\n";
        }
        if (nSonar > 0) {
            std::cout << "SONAR BITEXACT " << sonarExact << "/" << nSonar
                      << " cells=" << sonarCells << " differing=" << sonarDiffering
                      << " max|d|=" << sonarMax << "\n";
        }

        if (structural) {
            std::cout << "RESULT structural mismatch\n";
            return 1;
        }
        if (bitExact == nScans && sonarExact == nSonar) {
            std::cout << "RESULT bit-exact replay\n";
            return 0;
        }
        // A sonar bin is an echo in [0, 1]; a difference at float-rounding
        // level is a scheduling artefact of the tracer, not a different picture.
        constexpr float kSonarTol = 1e-3f;
        if (globalMax < kJitterTol && sonarMax < kSonarTol) {
            std::cout << "RESULT within-jitter (max |d| " << globalMax << " m < " << kJitterTol << " m";
            if (nSonar > 0) std::cout << ", sonar max |d| " << sonarMax << " < " << kSonarTol;
            std::cout << ")\n";
            return 3;
        }
        std::cout << "RESULT jitter exceeds tolerance (max |d| " << globalMax << " m";
        if (nSonar > 0) std::cout << ", sonar max |d| " << sonarMax;
        std::cout << ")\n";
        return 1;
    }

    // Within-run instance-id stability across the edit+revert. Returns 0 when
    // every range-stable beam kept its id.
    int idStability(const std::string& dir, int preIdx, int postIdx) {
        Scan pre, post;
        const std::string pa = scanFile(dir, preIdx), pb = scanFile(dir, postIdx);
        if (!loadScan(pa, pre) || !loadScan(pb, post)) {
            std::cout << "IO FAIL: cannot load " << pa << " / " << pb << "\n";
            return 1;
        }
        if (pre.hdr.pointCount != post.hdr.pointCount) {
            std::cout << "STRUCTURAL: point counts differ (" << pre.hdr.pointCount << " vs "
                      << post.hdr.pointCount << ")\n";
            return 1;
        }
        std::size_t hitBoth = 0, rangeStable = 0, idChanged = 0;
        std::map<std::pair<std::int32_t, std::int32_t>, std::size_t> pairs;
        for (std::size_t i = 0; i < pre.recs.size(); ++i) {
            const auto& a = pre.recs[i];
            const auto& b = post.recs[i];
            if (a.returnNo <= 0 || b.returnNo <= 0) continue;
            ++hitBoth;
            if (std::abs(a.distance - b.distance) >= kDeltaEps) continue;// moved / different surface
            ++rangeStable;
            ++pairs[{a.hitInstanceId, b.hitInstanceId}];
            if (a.hitInstanceId != b.hitInstanceId) ++idChanged;
        }
        std::cout << "idstability " << dir << "  pre=scan_" << std::setw(3) << std::setfill('0')
                  << preIdx << " post=scan_" << std::setw(3) << postIdx << std::setfill(' ')
                  << " (simTime " << pre.hdr.simTime << " -> " << post.hdr.simTime << ")\n"
                  << "  beams=" << pre.hdr.pointCount << " hitBoth=" << hitBoth
                  << " rangeStable=" << rangeStable << " idChanged=" << idChanged << "\n";
        for (const auto& [p, n] : pairs) {
            std::cout << "  id " << p.first << " -> " << p.second << "  " << n
                      << (p.first == p.second ? "" : "   <-- CHANGED") << "\n";
        }
        std::cout << (idChanged ? "RESULT instance ids NOT stable across the rebuild\n"
                                : "RESULT instance ids stable across the rebuild\n");
        return idChanged ? 1 : 0;
    }

}// namespace

int main(int argc, char** argv) {

    int frames = 120;
    std::string outDir;
    // 10 Hz on a 60 fps scripted clock. Integer so the scan schedule is a
    // property of the frame index, not of a float comparison against sim time.
    int scanEvery = 6;
    std::uint64_t seed = 0x9E3779B97F4A7C15ULL;
    float noiseSigma = 0.f;// see file header: OFF by default, on purpose
    bool noLod = false;    // auto-LOD swaps geometry -> different TLAS content
    bool noOccl = false;
    bool sceneEdit = false;
    bool withSonar = false;
    int editAddFrame = 45, editRemoveFrame = 81;
    int idPre = 6, idPost = 15;// scan indices: frame 36 (pre-add), frame 90 (post-remove)
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--frames" && i + 1 < argc) frames = std::atoi(argv[++i]);
        else if (arg == "--out" && i + 1 < argc) outDir = argv[++i];
        else if (arg == "--scene-edit") sceneEdit = true;
        else if (arg == "--sonar") withSonar = true;
        else if (arg == "--edit-add" && i + 1 < argc) editAddFrame = std::atoi(argv[++i]);
        else if (arg == "--edit-remove" && i + 1 < argc) editRemoveFrame = std::atoi(argv[++i]);
        else if (arg == "--pre" && i + 1 < argc) idPre = std::atoi(argv[++i]);
        else if (arg == "--post" && i + 1 < argc) idPost = std::atoi(argv[++i]);
        else if (arg == "--idstability" && i + 1 < argc) {
            const std::string d = argv[++i];
            // Trailing --pre/--post must be parsed before the mode runs.
            for (int j = i + 1; j + 1 < argc; ++j) {
                const std::string f = argv[j];
                if (f == "--pre") idPre = std::atoi(argv[j + 1]);
                else if (f == "--post") idPost = std::atoi(argv[j + 1]);
            }
            return idStability(d, idPre, idPost);
        }
        else if (arg == "--scan-every" && i + 1 < argc) scanEvery = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--seed" && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 0);
        else if (arg == "--noise-sigma" && i + 1 < argc) noiseSigma = float(std::atof(argv[++i]));
        else if (arg == "--no-lod") noLod = true;
        else if (arg == "--no-occl") noOccl = true;
        else if (arg == "--compare" && i + 2 < argc) return compare(argv[i + 1], argv[i + 2]);
    }
    if (outDir.empty()) {
        std::cout << "usage: vulkan_lidar_audit --frames N [--scene-edit] [--sonar] --out <dir>\n"
                  << "       vulkan_lidar_audit --compare <dirA> <dirB>\n"
                  << "       vulkan_lidar_audit --idstability <dir> [--pre N] [--post N]\n";
        return 1;
    }
    std::error_code ec;
    std::filesystem::create_directories(outDir, ec);

    Canvas canvas("vulkan_lidar_audit",
                  {{"vsync", false}, {"size", WindowSize{800, 600}}, {"headless", true}});
    VulkanRenderer renderer(canvas);
    // Pinned for the same reason the AOV audit pins them: anything that adapts
    // across frames from image statistics or from a third-party temporal black
    // box is a confound this harness exists to exclude. None of them feed the
    // LIDAR directly, but they do feed the LOD/culling decisions that shape the
    // TLAS the beams walk.
    renderer.setAutoExposure(false);
    renderer.setDlss(false);
    renderer.setFsr(false);
    if (noLod) renderer.setAutoLod(false);
    if (noOccl) renderer.setOcclusionCulling(false);

    // ---- scene: the AOV audit's, deliberately -------------------------------
    // Same geometry, same scripted motion, so the two audits measure two
    // modalities of ONE scene rather than two scenes. The movers matter here for
    // a second reason: they are the TLAS instances whose per-frame updates the
    // beams have to walk.
    Scene scene;
    scene.background = Color(0x304050);

    auto sun = DirectionalLight::create(Color(0xffffff), 3.f);
    sun->position.set(20.f, 30.f, 15.f);
    scene.add(sun);

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 100.f);
    camera->position.set(0.f, 3.f, 9.f);
    camera->lookAt(Vector3(0.f, 1.f, 0.f));

    auto ground = Mesh::create(BoxGeometry::create(30.f, 0.5f, 30.f),
                               MeshStandardMaterial::create(
                                       MeshStandardMaterial::Params{}.color(Color(0x556b45))));
    ground->position.y = -0.25f;
    scene.add(ground);
    renderer.setObjectInstanceId(*ground, 1);
    renderer.setObjectClassId(*ground, 1);

    auto mover = Mesh::create(BoxGeometry::create(1.f, 1.f, 1.f),
                              MeshStandardMaterial::create(
                                      MeshStandardMaterial::Params{}.color(Color(0xc8783c))));
    scene.add(mover);
    renderer.setObjectInstanceId(*mover, 2);
    renderer.setObjectClassId(*mover, 2);

    auto spinner = Mesh::create(SphereGeometry::create(0.7f, 32, 16),
                                MeshStandardMaterial::create(
                                        MeshStandardMaterial::Params{}.color(Color(0x3c78c8))));
    spinner->position.set(-2.5f, 1.f, 0.f);
    scene.add(spinner);
    renderer.setObjectInstanceId(*spinner, 3);
    renderer.setObjectClassId(*spinner, 2);

    auto pillar = Mesh::create(BoxGeometry::create(0.8f, 4.f, 0.8f),
                               MeshStandardMaterial::create(
                                       MeshStandardMaterial::Params{}.color(Color(0x808890))));
    pillar->position.set(2.5f, 2.f, -1.f);
    scene.add(pillar);
    renderer.setObjectInstanceId(*pillar, 4);
    renderer.setObjectClassId(*pillar, 3);

    // The CPU deformer that graduates to the per-frame dynamic-geometry path
    // (staging upload + frame-cb BLAS refit). Without it the certificate would
    // exclude the refit machinery — and for the LIDAR that machinery is not a
    // side concern: a refit is exactly the kind of GPU-scheduled work whose
    // ordering the beams could be sensitive to.
    auto waver = Mesh::create(PlaneGeometry::create(4.f, 4.f, 20, 20),
                              MeshStandardMaterial::create(
                                      MeshStandardMaterial::Params{}.color(Color(0xb03a48))));
    waver->rotation.x = -math::PI / 2;
    waver->position.set(-0.5f, 0.6f, 3.0f);
    scene.add(waver);
    renderer.setObjectInstanceId(*waver, 5);
    renderer.setObjectClassId(*waver, 4);
    auto* wavPos = waver->geometry()->getAttribute<float>("position");
    auto* wavNrm = waver->geometry()->getAttribute<float>("normal");
    wavPos->setUsage(DrawUsage::Dynamic);
    wavNrm->setUsage(DrawUsage::Dynamic);
    const std::vector<float> wavRest = wavPos->array();
    auto deform = [&](double t) {
        auto& p = wavPos->array();
        auto& n = wavNrm->array();
        for (std::size_t v = 0; v < p.size() / 3; ++v) {
            const float x = wavRest[3 * v + 0];
            const float y = wavRest[3 * v + 1];
            const float ax = 2.0f * x + static_cast<float>(t) * 3.0f;
            const float ay = 2.0f * y + static_cast<float>(t) * 2.0f;
            const float z = 0.25f * std::sin(ax) * std::cos(ay);
            const float dzdx = 0.50f * std::cos(ax) * std::cos(ay);
            const float dzdy = -0.50f * std::sin(ax) * std::sin(ay);
            p[3 * v + 2] = z;
            const float inv = 1.f / std::sqrt(dzdx * dzdx + dzdy * dzdy + 1.f);
            n[3 * v + 0] = -dzdx * inv;
            n[3 * v + 1] = -dzdy * inv;
            n[3 * v + 2] = inv;
        }
        wavPos->needsUpdate();
        wavNrm->needsUpdate();
    };

    // ---- sensor -------------------------------------------------------------
    // VLP-16 at 0.2 deg azimuth = 1800 x 16 = 28800 beams, mounted at 1.5 m and
    // yawing slowly on the scripted clock. NOT added to the scene: the sensor
    // carries no geometry, and scan() updates its own world matrix when it has
    // no parent, so keeping it out of the graph keeps the TLAS identical to the
    // AOV audit's.
    const LidarModel model = LidarModel::VLP16();
    PathTracedLidarSensor sensor(model, /*maxRange=*/25.f);
    sensor.position.set(0.f, 1.5f, 0.f);
    sensor.params.referenceRange = 5.f;
    sensor.params.laserPower = 1.f;
    sensor.params.atmosphericExtinction = 0.f;
    sensor.params.detectorThreshold = 0.005f;
    sensor.rangeNoise = {noiseSigma, 0.f, 0.f, seed};
    sensor.resetNoise();

    // The sonar rides the lidar's pose (same mount, same yaw script) and, like
    // it, stays out of the scene graph. Clean by default for the same reason:
    // the fold is what is being measured, not the speckle stream.
    SonarSensor sonarSensor(SonarModel::Wide130());
    sonarSensor.position.copy(sensor.position);
    sonarSensor.rangeNoise = {noiseSigma, 0.f, 0.f, seed};
    sonarSensor.resetNoise();
    SonarImage sonarImage;
    int sonarScans = 0;

    // The edit subject. Built up front so its geometry upload is not itself the
    // event; the ADD at editAddFrame is what churns the entry list. Placed in
    // clear line of sight at ~4.9 m and 1 m up — well inside the VLP-16's
    // -15 deg row and not overlapping any scripted mover's path.
    auto editBox = Mesh::create(BoxGeometry::create(1.f, 1.f, 1.f),
                                MeshStandardMaterial::create(
                                        MeshStandardMaterial::Params{}.color(Color(0xd0c060))));
    editBox->position.set(3.5f, 1.0f, 3.5f);

    const int elevCount = static_cast<int>(model.elevationAngles.size());
    const int azSteps = elevCount > 0
                                ? static_cast<int>(sensor.beamCount()) / elevCount
                                : 0;

    // ---- render + scan ------------------------------------------------------

    constexpr double kDt = 1.0 / 60.0;// scripted clock — never wall time
    std::vector<LidarReturn> returns;
    std::ostringstream manifest;
    int scans = 0, failures = 0;
    std::uint64_t totalHits = 0;

    for (int f = 0; f < frames; ++f) {
        const double t = f * kDt;
        renderer.setSimTime(t);
        sensor.setSimTime(t);
        sonarSensor.setSimTime(t);

        mover->position.set(static_cast<float>(2.0 * std::sin(t * 1.3)), 1.f,
                            static_cast<float>(1.5 * std::cos(t * 0.9)));
        mover->rotation.y = static_cast<float>(t * 1.7);
        spinner->rotation.x = static_cast<float>(t * 2.3);
        deform(t);
        // Pinned in scene-edit mode: --idstability compares two scans of the
        // same run, and that only isolates static geometry if both were fired
        // from the same pose. See the file header.
        sensor.rotation.y = sceneEdit ? 0.f : static_cast<float>(t * 0.35);
        sonarSensor.rotation.y = sensor.rotation.y;

        if (sceneEdit) {
            if (f == editAddFrame) {
                scene.add(editBox);
                renderer.setObjectInstanceId(*editBox, 6);
                renderer.setObjectClassId(*editBox, 2);
                // The add alone appends, and removing an appended entry later
                // is a TAIL removal — nothing behind it to shift, so it does
                // not test reordering at all. Pulling the spinner out of the
                // MIDDLE of the list is what forces every entry behind it
                // (pillar, waver) to move, and re-adding it at editRemoveFrame
                // sends it to the tail rather than back to its old slot. That
                // asymmetry is the point: after the revert the scene holds the
                // same five objects in a different order.
                scene.remove(*spinner);
            } else if (f == editRemoveFrame) {
                scene.remove(*editBox);
                scene.add(spinner);
                renderer.setObjectInstanceId(*spinner, 3);
                renderer.setObjectClassId(*spinner, 2);
            }
        }

        renderer.render(scene, *camera);

        // The rate gate proper needs a PhysxWorld to drive it; the frame index
        // is the same 10 Hz schedule with one fewer moving part, and it is a
        // property of the script rather than of a float clock comparison.
        if (f % scanEvery != 0) continue;

        sensor.scan(renderer, returns);
        if (returns.empty()) {
            ++failures;
            continue;
        }

        DumpHeader hdr{};
        std::memcpy(hdr.magic, "TPLIDAR1", 8);
        hdr.version = 1;
        hdr.scanIndex = static_cast<std::uint32_t>(scans);
        hdr.beamCount = sensor.beamCount();
        hdr.elevCount = static_cast<std::uint32_t>(elevCount);
        hdr.azSteps = static_cast<std::uint32_t>(azSteps);
        hdr.pointCount = static_cast<std::uint32_t>(returns.size());
        hdr.simTime = sensor.lastScanTime();

        std::vector<DumpRec> recs(returns.size());
        std::uint64_t hits = 0;
        for (std::size_t i = 0; i < returns.size(); ++i) {
            const auto& r = returns[i];
            DumpRec& d = recs[i];
            d.distance = r.distance;
            d.intensity = r.intensity;
            d.hitInstanceId = r.hitInstanceId;
            d.returnNo = r.returnNo;
            d.returnKind = r.returnKind;
            d.px = r.position.x;
            d.py = r.position.y;
            d.pz = r.position.z;
            if (r.returnNo > 0) ++hits;
        }
        totalHits += hits;

        const std::string path = scanFile(outDir, scans);
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            std::cout << "cannot write " << path << "\n";
            return 1;
        }
        out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        out.write(reinterpret_cast<const char*>(recs.data()),
                  static_cast<std::streamsize>(recs.size() * sizeof(DumpRec)));

        Fnv h;
        h.bytes(&hdr, sizeof(hdr));
        h.bytes(recs.data(), recs.size() * sizeof(DumpRec));
        char name[32];
        std::snprintf(name, sizeof(name), "scan_%03d", scans);
        manifest << name << " frame=" << f << " points=" << hdr.pointCount
                 << " hits=" << hits << " fnv=" << std::hex << h.value() << std::dec << "\n";
        ++scans;

        if (withSonar) {
            sonarSensor.scan(renderer, sonarImage);
            const auto& rets = sonarSensor.lastReturns();
            if (rets.empty()) {
                ++failures;
                continue;
            }
            SonarDumpHeader sh{};
            std::memcpy(sh.magic, "TPSONAR1", 8);
            sh.version = 1;
            sh.scanIndex = static_cast<std::uint32_t>(sonarScans);
            sh.beams = sonarImage.beams;
            sh.bins = sonarImage.bins;
            sh.rays = sonarSensor.rayCount();
            sh.returns = static_cast<std::uint32_t>(rets.size());
            sh.simTime = sonarSensor.lastScanTime();

            const std::string spath = sonarFile(outDir, sonarScans);
            std::ofstream sout(spath, std::ios::binary);
            if (!sout) {
                std::cout << "cannot write " << spath << "\n";
                return 1;
            }
            sout.write(reinterpret_cast<const char*>(&sh), sizeof(sh));
            sout.write(reinterpret_cast<const char*>(sonarImage.intensity.data()),
                       static_cast<std::streamsize>(sonarImage.intensity.size() * sizeof(float)));

            std::size_t sonarHits = 0, echoes = 0;
            for (const auto& r : rets)
                if (r.returnNo > 0) ++sonarHits;
            for (float v : sonarImage.intensity)
                if (v > 0.f) ++echoes;
            Fnv sh1;
            sh1.bytes(&sh, sizeof(sh));
            sh1.bytes(sonarImage.intensity.data(), sonarImage.intensity.size() * sizeof(float));
            char sname[32];
            std::snprintf(sname, sizeof(sname), "sonar_%03d", sonarScans);
            manifest << sname << " frame=" << f << " rays=" << sh.rays << " hits=" << sonarHits
                     << " echoes=" << echoes << " fnv=" << std::hex << sh1.value() << std::dec << "\n";
            ++sonarScans;
        }
    }

    // meta is compared field-for-field by --compare: two runs that disagree here
    // were not running the same experiment, and no point statistic computed
    // across them would mean anything.
    std::ostringstream meta;
    meta << "meta frames=" << frames << " scanEvery=" << scanEvery << " scans=" << scans
         << " model=VLP16 beams=" << sensor.beamCount()
         << " elev=" << elevCount << " az=" << azSteps
         << " maxRange=" << sensor.params.maxRange
         << " seed=0x" << std::hex << seed << std::dec
         << " noiseSigma=" << noiseSigma
         << " lod=" << (noLod ? 0 : 1) << " occl=" << (noOccl ? 0 : 1)
         << " sceneEdit=" << (sceneEdit ? 1 : 0)
         << " editAdd=" << (sceneEdit ? editAddFrame : -1)
         << " editRemove=" << (sceneEdit ? editRemoveFrame : -1)
         << " yaw=" << (sceneEdit ? "fixed" : "orbit")
         << " sonar=" << (withSonar ? 1 : 0);
    if (withSonar) {
        const auto& sm = sonarSensor.model();
        meta << " sonarModel=Wide130 sonarBeams=" << sm.beams << " sonarSamples=" << sm.verticalSamples
             << " sonarBins=" << sm.rangeBins << " sonarRange=" << sm.maxRange << " sonarScans=" << sonarScans;
    }
    meta << "\n";

    const auto dyn = renderer.dynamicGeomStats();
    std::ostringstream dynRow;
    dynRow << "dyn.geom graduated=" << dyn.graduated
           << " refits=" << dyn.refitsRecorded
           << " rebuilds=" << dyn.fullRebuilds << "\n";
    // The acceleration-structure counterpart: a scene edit must show up here as
    // extra fullRebuilds, and both runs must agree on the count.
    const auto tl = renderer.tlasStats();
    dynRow << "tlas rebuilds=" << tl.fullRebuilds
           << " updates=" << tl.updates
           << " instances=" << tl.instances << "\n";
    if (dyn.graduated == 0) {
        std::cout << "DYNAMIC-GEOM PATH NEVER ENGAGED (deformer failed to graduate)\n";
        ++failures;
    }

    std::ofstream man(outDir + "/manifest.txt");
    man << meta.str() << manifest.str() << dynRow.str();

    std::cout << "vulkan_lidar_audit: " << frames << " frames, " << scans << " scans, "
              << sensor.beamCount() << " beams/scan, " << totalHits << " total returns"
              << (failures ? ", FAILURES: " : ", clean");
    if (failures) std::cout << failures;
    std::cout << "\n"
              << meta.str() << dynRow.str() << "wrote " << outDir << "\n";
    return failures ? 2 : 0;
}
