// Shared conveyor-system assets: the layout schema (saved/loaded as JSON) plus a
// loader that turns an Isaac Sim conveyor .usd into a placeholder-material model
// standing on the floor. Used by the conveyor designer (to author layouts) and by
// the physics sim (to rebuild them). Deliberately PhysX-free so the designer need
// not link PhysX — belt physics is created on the sim side from the same Layout.
#ifndef THREEPP_CONVEYOR_ASSETS_HPP
#define THREEPP_CONVEYOR_ASSETS_HPP

#include "threepp/threepp.hpp"
#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/loaders/USDLoader.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace conveyor {

    namespace fs = std::filesystem;
    using namespace threepp;

    // Orientation for a belt segment a→b: local +X = travel direction (full 3D, so
    // sloped/vertical segments tilt), local +Z = horizontal width axis, local +Y =
    // belt normal. Shared by the designer (preview planes) and the sim (colliders)
    // so they always match.
    inline Quaternion segmentOrientation(const Vector3& a, const Vector3& b) {
        Vector3 x(b.x - a.x, b.y - a.y, b.z - a.z);
        if (x.length() < 1e-6f) return Quaternion();
        x.normalize();
        const Vector3 up(0, 1, 0);
        Vector3 z;
        if (std::abs(x.dot(up)) > 0.999f) {
            z.set(0, 0, 1);// (near) vertical travel — arbitrary horizontal width axis
        } else {
            z.crossVectors(x, up).normalize();
        }
        Vector3 y;
        y.crossVectors(z, x).normalize();
        Matrix4 m;
        m.makeBasis(x, y, z);
        Quaternion q;
        q.setFromRotationMatrix(m);
        return q;
    }

    // Build one continuous, gap-free belt-surface ribbon of the given width that
    // follows a centerline polyline. A chain of independent per-segment quads fans
    // apart on the OUTSIDE of a bend (leaving triangular holes); this instead emits
    // one left/right vertex pair per centerline point and shares each cross-section
    // edge between neighbouring quads, so curves stay watertight. The width axis is
    // horizontal and bisects turns (miter join): at point i it is cross(tangent, up)
    // with a CENTERED tangent (pts[i+1]-pts[i-1]), matching segmentOrientation's
    // local +Z so the ribbon lines up with the per-segment belt frame; sloped runs
    // tilt with the tangent. UV.v carries cumulative arc length (uniform texturing).
    inline std::shared_ptr<BufferGeometry> ribbonGeometry(const std::vector<Vector3>& centerline,
                                                          float width) {
        auto geom = BufferGeometry::create();
        const size_t n = centerline.size();
        if (n < 2) return geom;

        const Vector3 up(0, 1, 0);
        const float hw = width * 0.5f;

        std::vector<float> pos, nrm, uv;
        pos.reserve(n * 6);
        nrm.reserve(n * 6);
        uv.reserve(n * 4);

        float runLen = 0.f;
        for (size_t i = 0; i < n; ++i) {
            if (i > 0) runLen += centerline[i].distanceTo(centerline[i - 1]);

            Vector3 t;
            if (i == 0) {
                t.subVectors(centerline[1], centerline[0]);
            } else if (i + 1 == n) {
                t.subVectors(centerline[n - 1], centerline[n - 2]);
            } else {
                t.subVectors(centerline[i + 1], centerline[i - 1]);
            }
            if (t.length() < 1e-6f) t.set(1, 0, 0);
            t.normalize();

            Vector3 lat;
            if (std::abs(t.dot(up)) > 0.999f) {
                lat.set(0, 0, 1);
            } else {
                lat.crossVectors(t, up).normalize();
            }
            Vector3 nv;
            nv.crossVectors(lat, t).normalize();

            const Vector3& c = centerline[i];
            pos.insert(pos.end(), {c.x - lat.x * hw, c.y - lat.y * hw, c.z - lat.z * hw});
            pos.insert(pos.end(), {c.x + lat.x * hw, c.y + lat.y * hw, c.z + lat.z * hw});
            nrm.insert(nrm.end(), {nv.x, nv.y, nv.z, nv.x, nv.y, nv.z});
            uv.insert(uv.end(), {0.f, runLen, 1.f, runLen});
        }

        std::vector<unsigned int> idx;
        idx.reserve((n - 1) * 6);
        for (size_t i = 0; i + 1 < n; ++i) {
            const auto l0 = static_cast<unsigned int>(i * 2);
            const unsigned int r0 = l0 + 1, l1 = l0 + 2, r1 = l0 + 3;
            idx.insert(idx.end(), {l0, r0, r1, l0, r1, l1});
        }

        geom->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geom->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        geom->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        geom->setIndex(idx);
        geom->computeBoundingSphere();
        geom->computeBoundingBox();
        return geom;
    }

    // Build a vertical wall ribbon of the given height that stands ON the centerline
    // (base at the centerline, top extruded straight up). Same waypoint system as the
    // belt ribbon — used for separators / guide rails. The face normal is the
    // horizontal lateral (perpendicular to travel), so the wall is visible from both
    // sides with a double-sided material. UV: v = cumulative arc length, u = height.
    inline std::shared_ptr<BufferGeometry> wallGeometry(const std::vector<Vector3>& centerline,
                                                        float height) {
        auto geom = BufferGeometry::create();
        const size_t n = centerline.size();
        if (n < 2) return geom;

        const Vector3 up(0, 1, 0);
        std::vector<float> pos, nrm, uv;
        pos.reserve(n * 6);
        nrm.reserve(n * 6);
        uv.reserve(n * 4);

        float runLen = 0.f;
        for (size_t i = 0; i < n; ++i) {
            if (i > 0) runLen += centerline[i].distanceTo(centerline[i - 1]);

            Vector3 t;
            if (i == 0) t.subVectors(centerline[1], centerline[0]);
            else if (i + 1 == n) t.subVectors(centerline[n - 1], centerline[n - 2]);
            else t.subVectors(centerline[i + 1], centerline[i - 1]);
            if (t.length() < 1e-6f) t.set(1, 0, 0);
            t.normalize();

            Vector3 lat;// horizontal face normal
            if (std::abs(t.dot(up)) > 0.999f) lat.set(0, 0, 1);
            else lat.crossVectors(t, up).normalize();

            const Vector3& c = centerline[i];
            pos.insert(pos.end(), {c.x, c.y, c.z});                                      // base
            pos.insert(pos.end(), {c.x + up.x * height, c.y + up.y * height, c.z + up.z * height});// top
            nrm.insert(nrm.end(), {lat.x, lat.y, lat.z, lat.x, lat.y, lat.z});
            uv.insert(uv.end(), {runLen, 0.f, runLen, 1.f});
        }

        std::vector<unsigned int> idx;
        idx.reserve((n - 1) * 6);
        for (size_t i = 0; i + 1 < n; ++i) {
            const auto b0 = static_cast<unsigned int>(i * 2);
            const unsigned int t0 = b0 + 1, b1 = b0 + 2, t1 = b0 + 3;
            idx.insert(idx.end(), {b0, b1, t1, b0, t1, t0});
        }

        geom->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geom->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        geom->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        geom->setIndex(idx);
        geom->computeBoundingSphere();
        geom->computeBoundingBox();
        return geom;
    }

    // A path waypoint. Normally a point on the centerline; if arcCenter is set, this
    // point is the CENTRE of a circular arc between its two neighbours (start, end) —
    // used for exact 90/180 horizontal bends instead of a spline approximation.
    struct Waypoint {
        Vector3 pos;
        bool arcCenter = false;
    };

    // Circular-arc parameters for a bend defined by an arc-CENTRE waypoint: the arc
    // runs from A (the last centreline point) to B (the next waypoint) around centre C
    // in the horizontal (XZ) plane. `incoming` is the travel direction arriving at A,
    // used only to break the ~180° tie (both ways are equal length). Shared by the
    // resampler (visual polyline) and the sim's rotational bend belt (physics) so the
    // collider matches exactly what is drawn.
    struct Arc {
        bool valid = false;
        float a0 = 0.f;   // start angle atan2(z,x) of (A-C)
        float sweep = 0.f;// signed swept angle from A to B (radians)
        float radA = 0.f, radB = 0.f;
    };

    inline Arc computeArc(const Vector3& A, const Vector3& C, const Vector3& B,
                          const Vector3& incoming) {
        const float PI = static_cast<float>(math::PI);
        Arc r;
        const Vector3 rA(A.x - C.x, 0.f, A.z - C.z);
        const Vector3 rB(B.x - C.x, 0.f, B.z - C.z);
        r.radA = rA.length();
        r.radB = rB.length();
        if (r.radA < 1e-4f || r.radB < 1e-4f) return r;// degenerate (A or B at centre)
        r.a0 = std::atan2(rA.z, rA.x);
        const float a1 = std::atan2(rB.z, rB.x);
        // Shortest signed angular difference, in (-PI, PI].
        float d = a1 - r.a0;
        while (d <= -PI) d += 2.f * PI;
        while (d > PI) d -= 2.f * PI;
        if (std::abs(d) < PI - 0.05f) {
            // Minor (shorter) arc — exact for a 90° turn and any bend < 180°.
            // Direction is fixed by geometry (which side B is on), not flow.
            r.sweep = d;
        } else {
            // ~180°: both directions are equal-length, so pick the side that
            // continues the incoming flow.
            const Vector3 tCCW(-rA.z, 0.f, rA.x);// tangent at A if sweeping CCW
            const bool ccw = incoming.length() > 1e-5f
                                     ? (tCCW.x * incoming.x + tCCW.z * incoming.z) >= 0.f
                                     : (d >= 0.f);
            r.sweep = ccw ? PI : -PI;
        }
        r.valid = true;
        return r;
    }

    // Resample the waypoints into a dense, evenly-spaced point list shared by the
    // designer preview and the sim. Three modes:
    //  - any arc-centre present: straight runs between regular points, and a true
    //    circular arc at each arc-centre node (radius from the start, ending exactly
    //    at the end point, turn direction inferred from the incoming flow);
    //  - else smooth: centripetal Catmull-Rom spline through the points;
    //  - else: the raw polyline.
    inline std::vector<Vector3> resamplePath(const std::vector<Waypoint>& wps, bool smooth,
                                             int samplesPerSegment = 12) {
        const float PI = static_cast<float>(math::PI);
        bool hasArc = false;
        for (const auto& w : wps) {
            if (w.arcCenter) {
                hasArc = true;
                break;
            }
        }

        if (!hasArc) {
            std::vector<Vector3> ctrl;
            ctrl.reserve(wps.size());
            for (const auto& w : wps) ctrl.push_back(w.pos);
            if (!smooth || ctrl.size() < 3) return ctrl;
            CatmullRomCurve3 curve(ctrl);
            const auto divisions = static_cast<unsigned int>(
                    (ctrl.size() - 1) * static_cast<size_t>(std::max(2, samplesPerSegment)));
            return curve.getSpacedPoints(divisions);
        }

        std::vector<Vector3> out;
        const int n = static_cast<int>(wps.size());
        int i = 0;
        while (i < n) {
            if (wps[i].arcCenter) {
                // Need a preceding emitted point (A) and a following regular point (B).
                if (out.empty() || i + 1 >= n || wps[i + 1].arcCenter) {
                    ++i;
                    continue;
                }
                const Vector3 A = out.back();
                const Vector3 C = wps[i].pos;
                const Vector3 B = wps[i + 1].pos;
                Vector3 incoming(0.f, 0.f, 0.f);
                if (out.size() >= 2) {
                    incoming.set(out.back().x - out[out.size() - 2].x, 0.f,
                                 out.back().z - out[out.size() - 2].z);
                }
                const Arc arc = computeArc(A, C, B, incoming);
                if (!arc.valid) {
                    out.push_back(B);
                    i += 2;
                    continue;
                }
                const int steps = std::max(2, static_cast<int>(std::ceil(
                        std::abs(arc.sweep) / (PI * 0.5f) * static_cast<float>(std::max(2, samplesPerSegment)))));
                for (int k = 1; k <= steps; ++k) {
                    const float t = static_cast<float>(k) / static_cast<float>(steps);
                    const float ang = arc.a0 + arc.sweep * t;
                    const float rad = arc.radA + (arc.radB - arc.radA) * t;
                    out.emplace_back(C.x + rad * std::cos(ang), A.y + (B.y - A.y) * t, C.z + rad * std::sin(ang));
                }
                i += 2;
            } else {
                out.push_back(wps[i].pos);
                ++i;
            }
        }
        return out;
    }

    // --- Layout schema (the JSON written by the designer) ----------------------

    struct Piece {
        std::string model;     // USD stem, e.g. "ConveyorBelt_A06"
        Vector3 position{0, 0, 0};
        float yawDeg = 0.f;    // rotation about +Y
        float scale = 1.f;
        float beltSpeed = 0.f; // m/s along the piece's local +X; 0 = static (frame)
    };

    // One belt path: a centerline (spline control points) + its belt settings.
    // Multiple paths run in parallel.
    struct Path {
        std::vector<Waypoint> waypoints;// belt centerline (regular points + arc centres)
        float beltWidth = 1.0f;
        float beltSpeed = 0.6f;
        bool reverse = false;
        bool smooth = true;// spline through the (non-arc) waypoints
        // Separator: a collision-only vertical wall along the centerline (a guide rail /
        // lane divider) instead of a moving belt surface. Uses the same waypoints.
        bool separator = false;
        float wallHeight = 0.5f;
    };

    struct Layout {
        std::vector<Piece> pieces;// visual conveyor models
        std::vector<Path> paths;  // parallel belt paths the fish ride
    };

    inline void saveLayout(const fs::path& path, const Layout& layout) {
        nlohmann::json j;
        j["version"] = 2;
        j["pieces"] = nlohmann::json::array();
        for (const auto& p : layout.pieces) {
            j["pieces"].push_back({
                    {"model", p.model},
                    {"position", {p.position.x, p.position.y, p.position.z}},
                    {"yawDeg", p.yawDeg},
                    {"scale", p.scale},
                    {"beltSpeed", p.beltSpeed},
            });
        }
        j["paths"] = nlohmann::json::array();
        for (const auto& pa : layout.paths) {
            nlohmann::json jp;
            jp["beltWidth"] = pa.beltWidth;
            jp["beltSpeed"] = pa.beltSpeed;
            jp["reverse"] = pa.reverse;
            jp["smooth"] = pa.smooth;
            jp["separator"] = pa.separator;
            jp["wallHeight"] = pa.wallHeight;
            jp["waypoints"] = nlohmann::json::array();
            for (const auto& w : pa.waypoints) {
                if (w.arcCenter) jp["waypoints"].push_back({w.pos.x, w.pos.y, w.pos.z, 1});
                else jp["waypoints"].push_back({w.pos.x, w.pos.y, w.pos.z});
            }
            j["paths"].push_back(jp);
        }
        std::ofstream f(path);
        f << j.dump(2);
    }

    inline std::optional<Layout> loadLayout(const fs::path& path) {
        std::ifstream f(path);
        if (!f) return std::nullopt;
        nlohmann::json j;
        try {
            f >> j;
        } catch (const std::exception&) {
            return std::nullopt;
        }
        Layout out;
        for (const auto& jp : j.value("pieces", nlohmann::json::array())) {
            Piece p;
            p.model = jp.value("model", std::string{});
            if (auto it = jp.find("position"); it != jp.end() && it->is_array() && it->size() == 3) {
                p.position.set((*it)[0].get<float>(), (*it)[1].get<float>(), (*it)[2].get<float>());
            }
            p.yawDeg = jp.value("yawDeg", 0.f);
            p.scale = jp.value("scale", 1.f);
            p.beltSpeed = jp.value("beltSpeed", 0.f);
            out.pieces.push_back(p);
        }
        auto readWaypoints = [](const nlohmann::json& arr, std::vector<Waypoint>& dst) {
            for (const auto& jw : arr) {
                if (jw.is_array() && jw.size() >= 3) {
                    Waypoint w;
                    w.pos.set(jw[0].get<float>(), jw[1].get<float>(), jw[2].get<float>());
                    w.arcCenter = jw.size() >= 4 && jw[3].get<float>() != 0.f;
                    dst.push_back(w);
                }
            }
        };
        if (auto it = j.find("paths"); it != j.end() && it->is_array()) {
            for (const auto& jp : *it) {
                Path p;
                p.beltWidth = jp.value("beltWidth", 1.0f);
                p.beltSpeed = jp.value("beltSpeed", 0.6f);
                p.reverse = jp.value("reverse", false);
                p.smooth = jp.value("smooth", true);
                p.separator = jp.value("separator", false);
                p.wallHeight = jp.value("wallHeight", 0.5f);
                if (auto w = jp.find("waypoints"); w != jp.end() && w->is_array()) readWaypoints(*w, p.waypoints);
                out.paths.push_back(p);
            }
        } else if (auto w = j.find("waypoints"); w != j.end() && w->is_array()) {
            // Back-compat: old single-path layout (top-level waypoints + belt fields).
            Path p;
            p.beltWidth = j.value("beltWidth", 1.0f);
            p.beltSpeed = j.value("beltSpeed", 0.6f);
            p.reverse = j.value("reverse", false);
            p.smooth = j.value("smooth", true);
            readWaypoints(*w, p.waypoints);
            out.paths.push_back(p);
        }
        return out;
    }

    // --- Model templates -------------------------------------------------------

    // The .usd files present in a directory (stems), sorted.
    inline std::vector<std::string> discoverModels(const fs::path& dir) {
        std::vector<std::string> out;
        if (!fs::is_directory(dir)) return out;
        for (const auto& e : fs::directory_iterator(dir)) {
            if (e.path().extension() == ".usd") out.push_back(e.path().stem().string());
        }
        std::ranges::sort(out);
        return out;
    }

    enum class BeltKind { Straight, Curve };

    struct ModelTemplate {
        std::shared_ptr<Group> group;// wrapper at identity; inner node holds the re-origin
        float deckTopY = 0.f;        // conveying-surface height above the piece's base (y=0)
        Vector3 deckSize{1, 0, 1};   // deck footprint: x = length (travel), z = width

        // Belt kind + arc parameters (filled by a circle-fit of the deck mesh). For a
        // Curve the deck is an annular sector; the sim drags fish by rotating about
        // arcCenter rather than translating. arcCenter is in the re-origined template
        // frame (y = deckTopY); angles are degrees about +Y.
        BeltKind kind = BeltKind::Straight;
        Vector3 arcCenter{0, 0, 0};
        float arcRadius = 0.f; // mean radius of the deck annulus
        float arcWidth = 0.f;  // radial width (outer - inner)
        float arcStartDeg = 0.f;
        float arcSweepDeg = 0.f;
    };

    // Classify the deck as straight or curved by least-squares circle-fitting its
    // vertices (XZ). A near-collinear deck (huge radius / tiny angular span) stays
    // Straight; a good fit with a meaningful sweep becomes a Curve and we record its
    // centre, radius, and angular extent.
    inline void analyzeDeckArc(Mesh& deck, float deckTopY, ModelTemplate& t) {
        auto g = deck.geometry();
        if (!g) return;
        auto* pos = g->getAttribute<float>("position");
        if (!pos || pos->count() < 16) return;

        const Matrix4 mw = *deck.matrixWorld;
        const size_t n = pos->count();
        std::vector<float> xs(n), zs(n);
        double xm = 0, zm = 0;
        for (size_t i = 0; i < n; ++i) {
            Vector3 v(pos->getX(i), pos->getY(i), pos->getZ(i));
            v.applyMatrix4(mw);
            xs[i] = v.x;
            zs[i] = v.z;
            xm += v.x;
            zm += v.z;
        }
        xm /= static_cast<double>(n);
        zm /= static_cast<double>(n);

        double Suu = 0, Svv = 0, Suv = 0, Suuu = 0, Svvv = 0, Suvv = 0, Svuu = 0;
        for (size_t i = 0; i < n; ++i) {
            const double u = xs[i] - xm, v = zs[i] - zm;
            Suu += u * u;
            Svv += v * v;
            Suv += u * v;
            Suuu += u * u * u;
            Svvv += v * v * v;
            Suvv += u * v * v;
            Svuu += v * u * u;
        }
        const double det = Suu * Svv - Suv * Suv;
        if (std::fabs(det) < 1e-9) return;// collinear → straight
        const double b1 = 0.5 * (Suuu + Suvv), b2 = 0.5 * (Svvv + Svuu);
        const double uc = (b1 * Svv - b2 * Suv) / det;
        const double vc = (b2 * Suu - b1 * Suv) / det;
        const double cx = xm + uc, cz = zm + vc;
        if (std::sqrt(uc * uc + vc * vc + (Suu + Svv) / double(n)) < 1e-4) return;

        std::vector<double> ang(n);
        double minR = 1e30, maxR = 0;
        for (size_t i = 0; i < n; ++i) {
            const double dx = xs[i] - cx, dz = zs[i] - cz;
            const double ri = std::sqrt(dx * dx + dz * dz);
            minR = std::min(minR, ri);
            maxR = std::max(maxR, ri);
            ang[i] = std::atan2(dz, dx);
        }
        std::sort(ang.begin(), ang.end());
        const double twoPi = 2.0 * static_cast<double>(math::PI);
        double maxGap = (ang.front() + twoPi) - ang.back();// wrap gap
        double startAng = ang.front();
        for (size_t i = 1; i < n; ++i) {
            const double gap = ang[i] - ang[i - 1];
            if (gap > maxGap) {
                maxGap = gap;
                startAng = ang[i];
            }
        }
        const double sweep = twoPi - maxGap;
        const double sweepDeg = sweep * 180.0 / static_cast<double>(math::PI);
        const double meanR = 0.5 * (minR + maxR);

        if (sweepDeg > 40.0 && (maxR - minR) < 0.9 * meanR && meanR > 1e-3) {
            t.kind = BeltKind::Curve;
            t.arcCenter.set(static_cast<float>(cx), deckTopY, static_cast<float>(cz));
            t.arcRadius = static_cast<float>(meanR);
            t.arcWidth = static_cast<float>(maxR - minR);
            t.arcStartDeg = static_cast<float>(startAng * 180.0 / static_cast<double>(math::PI));
            t.arcSweepDeg = static_cast<float>(sweepDeg);
        }
    }

    // Optional per-model kind override (data/models/conveyor/conveyor_types.json):
    // { "ConveyorBelt_A02": "curve", "ConveyorBelt_A08": "straight", ... }.
    inline std::unordered_map<std::string, std::string> loadTypeOverrides(const fs::path& path) {
        std::unordered_map<std::string, std::string> out;
        std::ifstream f(path);
        if (!f) return out;
        nlohmann::json j;
        try {
            f >> j;
        } catch (const std::exception&) {
            return out;
        }
        for (const auto& [k, v] : j.items()) {
            if (v.is_string()) out[k] = v.get<std::string>();
        }
        return out;
    }

    // Load a conveyor .usd, assign placeholder PBR materials (Isaac MDL/OmniPBR
    // materials do not survive the USD->threepp import — every mesh comes in white),
    // and re-origin the model so it stands on the floor: bottom at y=0, centred in XZ.
    // The re-origin lives on an inner node wrapped in an identity Group, so callers
    // can clone the wrapper and freely set position/rotation/scale on it.
    inline ModelTemplate prepareModel(USDLoader& loader, const fs::path& usdPath) {
        ModelTemplate t;
        auto loaded = loader.load(usdPath);
        if (!loaded) return t;

        // Belt deck = thinnest non-trivial sub-mesh (local-bbox aspect, frame-independent).
        Mesh* deck = nullptr;
        float bestThin = 1e9f;
        loaded->traverseType<Mesh>([&](Mesh& m) {
            auto g = m.geometry();
            if (!g) return;
            g->computeBoundingBox();
            if (!g->boundingBox) return;
            Vector3 s;
            g->boundingBox->getSize(s);
            const float mx = std::max({s.x, s.y, s.z});
            const float mn = std::min({s.x, s.y, s.z});
            if (mx < 0.3f) return;
            const float thin = mn / std::max(mx, 1e-6f);
            if (thin < bestThin) {
                bestThin = thin;
                deck = &m;
            }
        });

        loaded->traverseType<Mesh>([&](Mesh& m) {
            auto mat = MeshStandardMaterial::create();
            if (&m == deck) {
                mat->color = Color(0x202020);// dark belt/rubber
                mat->roughness = 0.85f;
                mat->metalness = 0.f;
            } else {
                mat->color = Color(0x9098a0);// brushed metal frame
                mat->roughness = 0.5f;
                mat->metalness = 0.85f;
            }
            m.setMaterial(mat);
        });

        // Re-origin the loaded model: centre XZ, drop bottom to y=0.
        loaded->updateMatrixWorld(true);
        Box3 full;
        full.setFromObject(*loaded, true);
        Vector3 c, sz;
        full.getCenter(c);
        full.getSize(sz);
        loaded->position.set(-c.x, -(c.y - sz.y * 0.5f), -c.z);

        auto wrapper = Group::create();
        wrapper->add(loaded);
        wrapper->updateMatrixWorld(true);

        if (deck) {
            Box3 db;
            db.setFromObject(*deck, true);
            Vector3 dc, ds;
            db.getCenter(dc);
            db.getSize(ds);
            t.deckTopY = dc.y + ds.y * 0.5f;
            t.deckSize = ds;
            analyzeDeckArc(*deck, t.deckTopY, t);
        } else {
            t.deckTopY = sz.y;
            t.deckSize.set(sz.x, 0.f, sz.z);
        }
        t.group = wrapper;
        return t;
    }

}// namespace conveyor

#endif// THREEPP_CONVEYOR_ASSETS_HPP
