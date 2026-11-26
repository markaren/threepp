
#include "threepp/extras/marine/VesselWash.hpp"

#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/DisplacedMesh.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;
using namespace threepp::marine;

namespace {

    // The plan form the ocean's hull footprint has (water_displace.comp
    // hullLocal), as a share of the half beam at s in [-1, 1], stern to bow.
    float planForm(float s) {
        return s >= 0.f ? std::pow(std::max(1.f - s * s, 0.f), 0.6f) : 1.f - 0.25f * s * s;
    }

    // Her sectional area curve, stern to bow (Waves::Thickness): her own
    // sections from the aftmost that holds any water (a table's empty
    // stations behind her are not her: her aftmost wet section is her
    // transom, which update() closes), or a trapezoid over her footprint
    // (Hull::sections). Either way one more station half a spacing ahead of
    // the foremost, where her section is nothing, so the curve ends at her
    // stem.
    std::vector<VesselWash::Section> areaCurve(const VesselWash::Hull& hull) {
        std::vector<VesselWash::Section> s = hull.sections;
        if (s.empty()) {
            if (!(hull.halfLength > 0.f && hull.halfBeam > 0.f && hull.draft > 0.f)) return s;
            const bool  twin    = hull.hullOffset > 0.f;
            const float half    = twin ? std::max(hull.halfBeam - hull.hullOffset, 0.1f) : hull.halfBeam;// one hull's
            const float aMid    = (twin ? 2.f : 1.f) * 0.97f * 2.f * half * hull.draft;
            const float flat    = 2.f * std::clamp(hull.prismatic, 0.5f, 1.f) - 1.f;// the full share of her length
            const int   n       = 4 * VesselWash::kSegments + 1;
            for (int i = 0; i < n; ++i) {
                const float t = -1.f + 2.f * float(i) / float(n - 1);// stern -1 .. bow 1
                const float e = std::abs(t);
                const float a = e <= flat ? 1.f : (1.f - e) / std::max(1.f - flat, 1e-6f);
                s.push_back({hull.centre + t * hull.halfLength, aMid * std::max(a, 0.f), hull.draft, half * planForm(t)});
            }
        } else {
            std::sort(s.begin(), s.end(), [](const auto& a, const auto& b) { return a.x < b.x; });
            const auto wet = std::find_if(s.begin(), s.end(), [](const auto& a) { return a.area > 0.f; });
            s.erase(s.begin(), wet);
        }
        if (s.size() >= 2 && s.back().area > 0.f) {
            VesselWash::Section stem = s.back();
            stem.x += 0.5f * (s.back().x - s[s.size() - 2].x);
            stem.area = 0.f;
            s.push_back(stem);
        }
        return s;
    }

}// namespace

VesselWash::VesselWash(DisplacedMesh& ocean, uint32_t patch, Hull hull, float size, float rippleSize)
    : ocean_(ocean), patch_(patch), hull_(std::move(hull)) {
    size_       = size > 0.f ? size : std::clamp(20.f * hull_.length, 30.f, 120.f);
    rippleSize_ = std::min(size_, rippleSize > 0.f ? rippleSize : std::max(10.f * hull_.length, 16.f));
    if (hull_.strips.empty()) hull_.strips = footprintStrips(hull_);
    if (hull_.waves == Waves::Thickness) {
        thickness_ = thicknessSegments(hull_);
        const auto curve = areaCurve(hull_);
        if (!curve.empty()) transom_ = curve.front();
        // Her bow's lines, as usv_rig's Wash.hull_lines reads them: her stem
        // half a station ahead of her foremost wet section, her shoulder the
        // foremost wet section half her widest, her entry's half angle from
        // the one to the other, her bow's draft the mean ahead of her shoulder.
        float aMax = 0.f, bMax = 0.f;
        for (const auto& s : curve) aMax = std::max(aMax, s.area);
        for (const auto& s : curve) {
            if (s.area > 1e-3f * aMax) bMax = std::max(bMax, s.halfBreadth);
        }
        int stem = -1, shoulder = -1;
        for (int i = static_cast<int>(curve.size()) - 1; i >= 0; --i) {
            if (!(curve[i].area > 1e-3f * aMax)) continue;
            if (stem < 0) stem = i;
            if (curve[i].halfBreadth >= 0.5f * bMax) {
                shoulder = i;
                break;
            }
        }
        if (bMax > 0.f && stem >= 0 && shoulder >= 0) {
            BowLines& b    = bowLines_;
            const size_t n = curve.size();
            const float  dx = stem + 1 < static_cast<int>(n) ? curve[stem + 1].x - curve[stem].x
                                                             : (stem > 0 ? curve[stem].x - curve[stem - 1].x : 0.f);
            b.xStem     = curve[stem].x + 0.5f * dx;
            b.xShoulder = curve[shoulder].x;
            float d = 0.f;
            int   m = 0;
            for (int i = shoulder; i <= stem; ++i) {
                d += std::max(curve[i].draft, 0.f);
                ++m;
            }
            b.draft    = d / float(std::max(m, 1));
            b.tanEntry = curve[shoulder].halfBreadth / std::max(b.xStem - b.xShoulder, 1e-3f);
            b.curve    = curve;
            b.valid    = b.xStem > b.xShoulder && b.draft > 0.f;
        }
    }
}

std::vector<VesselWash::ThicknessSegment> VesselWash::thicknessSegments(const Hull& hull) {
    std::vector<ThicknessSegment> out;
    const std::vector<Section> s = areaCurve(hull);
    if (s.size() < 2) return out;
    // Each interval between two stations gains dA toward her bow, spread
    // evenly over it.
    float total = 0.f;
    for (size_t j = 0; j + 1 < s.size(); ++j) total += std::abs(s[j + 1].area - s[j].area);
    if (!(total > 0.f)) return out;
    const float share = total / float(kSegments);
    // A stretch takes intervals until it holds its share of |dA|, and stops
    // early where her section turns from narrowing to widening (a bulb, her
    // shoulders), so no stretch is a source and a sink at once. It lies at its
    // intervals' |dA|-weighted centroid and is as long as their spread (a
    // uniform line of half length a has variance a^2 / 3).
    float sum = 0.f, w = 0.f, wx = 0.f, wxx = 0.f, wb = 0.f, wd = 0.f;
    int sign = 0;
    const auto flush = [&] {
        if (w > 0.f) {
            const float xc  = wx / w;
            const float var = std::max(wxx / w - xc * xc, 0.f);
            out.push_back({xc, sum, std::sqrt(3.f * var), 0.5f * wb / w, wd / w});
        }
        sum = w = wx = wxx = wb = wd = 0.f;
        sign = 0;
    };
    for (size_t j = 0; j + 1 < s.size(); ++j) {
        const float dA = s[j + 1].area - s[j].area;
        const float a  = std::abs(dA);
        if (!(a > 0.f)) continue;
        const int  sg  = dA < 0.f ? -1 : 1;
        const bool big = a > 0.02f * share;// a sliver of the other sign does not end a stretch
        if (w >= share * (1.f - 1e-4f) || (big && sign != 0 && sg != sign)) flush();
        if (sign == 0 && big) sign = sg;
        const float m = 0.5f * (s[j].x + s[j + 1].x);
        const float l = s[j + 1].x - s[j].x;
        sum += dA;
        w += a;
        wx += a * m;
        wxx += a * (m * m + l * l / 12.f);
        wb += a * 0.5f * (s[j].halfBreadth + s[j + 1].halfBreadth);
        wd += a * 0.5f * (s[j].draft + s[j + 1].draft);
    }
    flush();
    return out;
}

float VesselWash::race(float thrust, float advance, float area, bool jet, float rho) {
    const float t = std::abs(thrust);
    const float v = std::max(advance, 0.f);
    if (t <= 0.f || area <= 0.f) return 0.f;
    if (jet) return 0.5f * (v + std::sqrt(v * v + 4.f * t / (rho * area))) - v;
    return std::sqrt(v * v + 2.f * t / (rho * area)) - v;
}

std::vector<VesselWash::WeightStrip> VesselWash::footprintStrips(const Hull& hull) {
    std::vector<WeightStrip> out;
    if (hull.halfLength <= 0.f || hull.halfBeam <= 0.f) return out;
    const bool twin    = hull.hullOffset > 0.f;
    const float breadth = twin ? 2.f * std::max(hull.halfBeam - hull.hullOffset, 0.1f)// one hull's, at the water
                               : hull.halfBeam;                                      // one side's
    const float stretch = 2.f * hull.halfLength / float(kStrips);
    float b[kStrips];
    float total = 0.f;
    for (int k = 0; k < kStrips; ++k) {
        // The mean of the plan form over the stretch (four points).
        float m = 0.f;
        for (int j = 0; j < 4; ++j) {
            const float s = -1.f + 2.f * (float(k) + (float(j) + 0.5f) / 4.f) / float(kStrips);
            m += planForm(s);
        }
        b[k] = breadth * m / 4.f;
        total += b[k];
    }
    for (int k = 0; k < kStrips; ++k) {
        if (b[k] < 1e-4f * total) continue;
        const float x = hull.centre - hull.halfLength + (float(k) + 0.5f) * stretch;
        const float z = twin ? hull.hullOffset : 0.42f * b[k];
        const float r = std::hypot(0.41f * b[k], 0.8f * hull.draft);
        for (const float side : {1.f, -1.f}) {
            out.push_back({x, side * z, 0.5f * b[k] / total, r, 0.5f * stretch});
        }
    }
    return out;
}

float VesselWash::rippleCell(float length, float cruiseSpeed) {
    const float lambdaT = 2.f * math::PI * cruiseSpeed * cruiseSpeed / kG;
    return std::min(lambdaT / 24.f, length / 10.f);
}

void VesselWash::clear() {
    ocean_.wakePatches.at(patch_).size = 0.f;
    last_.clear();
}

void VesselWash::emit(const std::string& name, float x, float z, float vx, float vz, float radius, float foam,
                      float aeration, float turbulence, float lane, float push, float ax, float az, float taper,
                      float flux, float draft) {
    DisplacedMesh::WakeSource s{};
    s.x0 = s.x1 = x;
    s.z0 = s.z1 = z;
    if (const auto it = last_.find(name); it != last_.end()) {
        // Further than half her patch: she was moved, not sailed.
        if (std::hypot(x - it->second.x, z - it->second.z) <= 0.5f * size_) {
            s.x0 = it->second.x;
            s.z0 = it->second.z;
        }
    }
    s.vx         = vx;
    s.vz         = vz;
    s.radius     = radius;
    s.foam       = foam;
    s.aeration   = aeration;
    s.turbulence = turbulence;
    s.lane       = lane;
    if ((push != 0.f || flux != 0.f) && ripples_) {
        s.push  = push;
        s.flux  = flux;
        s.draft = draft;
        s.ax    = ax;
        s.az    = az;
        s.taper = taper;
        s.patch = static_cast<int32_t>(patch_);
    }
    ocean_.addWakeSource(s);
    last_[name] = {x, z};
}

void VesselWash::put(const std::string& name, const Vector3& at, const Vector3& velocity, float radius, float foam,
                     float aeration, float turbulence, float lane, float push) {
    if (!(ocean_.wakePatches.at(patch_).size > 0.f)) return;
    const float texel = size_ / float(std::max(ocean_.wakeField.resolution, 1u));
    const std::string key = "put:" + name;
    emit(key, at.x, at.z, velocity.x, velocity.z, std::max(radius, 1.25f * texel), foam, aeration, turbulence, lane,
         push);
    putNames_.insert(key);
}

void VesselWash::update(float dt, const Vector3& origin, const Vector3& forward, const Vector3& velocity,
                        const std::vector<Propulsor>& propulsors) {
    auto& patch = ocean_.wakePatches.at(patch_);
    float fx = forward.x, fz = forward.z;
    {
        const float n = std::max(std::hypot(fx, fz), 1e-9f);
        fx /= n;
        fz /= n;
    }
    const float sx = -fz, sz = fx;// to starboard: forward x up
    // She sits a length inside its leading edge; the rest of it lies astern.
    const float astern = 0.5f * size_ - 0.08f * size_ - hull_.length;
    patch.centerX = origin.x - fx * astern;
    patch.centerZ = origin.z - fz * astern;
    patch.size    = size_;
    const float texel = size_ / float(std::max(ocean_.wakeField.resolution, 1u));
    ripples_ = ocean_.wakeField.rippleResolution > 0;
    if (ripples_) {// her waves' window: she sits a fifth of it inside its leading edge
        patch.rippleCenterX = origin.x - fx * 0.3f * rippleSize_;
        patch.rippleCenterZ = origin.z - fz * 0.3f * rippleSize_;
        patch.rippleSize    = rippleSize_;
    }
    t_ += dt;
    for (auto it = last_.begin(); it != last_.end();) {
        // It was not put last frame: its trail has ended.
        if (it->first.rfind("put:", 0) == 0 && putNames_.count(it->first) == 0) it = last_.erase(it);
        else ++it;
    }
    putNames_.clear();

    const float fn3 = std::max(forward.length(), 1e-9f);
    const float u   = forward.dot(velocity) / fn3;// her surge
    const float vbx = velocity.x, vbz = velocity.z;
    float widest    = 0.f;

    for (size_t k = 0; k < propulsors.size(); ++k) {
        const Propulsor& p = propulsors[k];
        const std::string disc = p.name + "_disc";
        const auto silent = [&] {
            last_.erase(p.name);
            last_.erase(disc);
        };
        const float t = p.force.length();
        if (t < 1e-3f) {
            silent();
            continue;
        }
        // The race runs against the force.
        float dx = -p.force.x / t, dz = -p.force.z / t;
        {
            const float n = std::max(std::hypot(dx, dz), 1e-9f);
            dx /= n;
            dz /= n;
        }
        const float r0    = p.radius;
        const float depth = ocean_.sampleHeight(p.position.x, p.position.z, heightMask) - p.position.y;
        if (depth < -r0) {// out of the water: it draws air and makes no race
            silent();
            continue;
        }
        const float w = race(t, -(vbx * dx + vbz * dz), math::PI * r0 * r0, p.jet);
        if (w < 0.02f) {
            silent();
            continue;
        }
        const float cover = std::max(depth - r0, 0.f);// water over the top of the race where it starts
        const float run   = cover / kSpread;          // how far it runs before it has widened to the surface
        float rs          = r0 + kSpread * run;
        const float ws    = w * r0 / rs;              // momentum kept, spread over the wider race
        const float ts    = run / std::max(0.5f * (w + ws), 1e-3f);
        // Where that water is when it gets there.
        float px = p.position.x - vbx * ts + dx * run;
        float pz = p.position.z - vbz * ts + dz * run;
        const float a = 1.7f * t_ + 2.4f * float(k), b = 3.1f * t_ + 4.1f * float(k);
        const float stray = kWander * rs * (std::sin(a) + 0.6f * std::sin(b)) / 1.6f;
        px += -dz * stray;
        pz += dx * stray;
        const float fr   = air * w * w / (kG * std::max(cover, 0.02f));
        const float aer  = 1.f - std::exp(-(fr / 2.5f) * (fr / 2.5f));
        const float over = std::max(fr - 1.5f, 0.f);
        const float foam = 1.f - std::exp(-over * over / 20.f);
        const float turb = kTurbulence * ws;
        rs *= 1.f + 0.25f * std::min(std::sqrt(fr), 6.f);// a hard race breaks the surface wider than it arrived
        widest = std::max(widest, rs);
        emit(p.name, px, pz, dx * ws, dz * ws, std::max(rs, 1.25f * texel), foam, aer, turb,
             std::min(turb / 0.05f, 1.f));
        if (run > 2.f * r0) {// over the disc: its air, through the water above it
            emit(disc, p.position.x, p.position.z, dx * 0.4f * w, dz * 0.4f * w, std::max(1.3f * r0, 1.25f * texel),
                 0.f, 0.6f * aer * std::exp(-1.2f * cover), 0.f, 0.f);
        } else {
            last_.erase(disc);
        }
        report_[p.name] = {t, w, ws, run, fr, aer, foam};
    }

    // Her hulls: the water she drags, a lane from each stern.
    const float hl = hull_.halfLength, zs = hull_.hullOffset;
    const float rh = std::max(hull_.halfBeam - zs, 0.1f);// a hull's half-breadth at the water
    patch.eddy = std::max(kEddy * std::max(widest, rh), 4.f * texel);
    const float froude = std::abs(u) / std::sqrt(kG * 2.f * hl);
    const float lane   = lanes ? math::smoothstep(0.15f, 0.6f, std::abs(u)) : 0.f;
    static const std::string kHullNames[2] = {"hull_0", "hull_1"};
    if (lane > 0.f) {
        const float dry = math::smoothstep(0.35f, 0.85f, froude);// a transom runs dry and white when she is fast for her length
        const float x   = hull_.centre - std::copysign(hl, u);
        const int hulls = zs > 0.f ? 2 : 1;
        for (int k = 0; k < hulls; ++k) {
            const float z = hulls == 2 ? (k == 0 ? -zs : zs) : 0.f;
            emit(kHullNames[k], origin.x + fx * x + sx * z, origin.z + fz * x + sz * z, 0.15f * vbx, 0.15f * vbz,
                 std::max(rh, 1.25f * texel), math::smoothstep(0.45f, 1.05f, froude) * dry, dry, 0.07f * std::abs(u), lane);
        }
    } else {
        last_.erase(kHullNames[0]);
        last_.erase(kHullNames[1]);
    }

    // Her weight on the water, laid out along her hulls: what makes her waves.
    if (ripples_ && hull_.waves == Waves::Line) {
        // One line a hull. The trapezoid is a box of half length a along her
        // axis convolved with one of half length t (the source's taper, t / a):
        // flat over |x| < a - t, nothing past a + t = her half length, its mean
        // a / (a + t) = C_p. Its transform is sinc(k a) sinc(k t), so it ends
        // without a step. The gaussian across it is her half breadth.
        const int   hulls = zs > 0.f ? 2 : 1;
        const float cp    = std::clamp(hull_.prismatic, 0.5f, 1.f);
        const float a     = cp * hl;
        const float t     = (1.f - cp) * hl;
        const float push  = hull_.mass * kG / float(hulls);
        // The entry head at the stem she leads with: K rho U^2 / 2 at the
        // peak of a gaussian of her half breadth, centred half a half
        // breadth aft of the stem.
        // ASSUMED: where it sits and how wide (tidewater spreads the head
        // over the hull's immersion at the entry).
        constexpr float kRho = 1025.f;
        const float entry = hull_.entryHead * 0.5f * kRho * u * u * math::PI * rh * rh;
        const float xe    = hull_.centre + std::copysign(hl - 0.5f * rh, u);
        static const std::string kLineNames[2]  = {"line_0", "line_1"};
        static const std::string kEntryNames[2] = {"entry_0", "entry_1"};
        for (int k = 0; k < hulls; ++k) {
            const float z = hulls == 2 ? (k == 0 ? -zs : zs) : 0.f;
            emit(kLineNames[k], origin.x + fx * hull_.centre + sx * z, origin.z + fz * hull_.centre + sz * z, 0.f,
                 0.f, rh, 0.f, 0.f, 0.f, 0.f, push, fx * a, fz * a, a > 0.f ? t / a : 0.f);
            if (entry > 0.f) {
                emit(kEntryNames[k], origin.x + fx * xe + sx * z, origin.z + fz * xe + sz * z, 0.f, 0.f, rh, 0.f,
                     0.f, 0.f, 0.f, entry);
            } else {
                last_.erase(kEntryNames[k]);
            }
        }
    } else if (ripples_ && hull_.waves == Waves::Thickness) {
        // Her shape: each stretch of her sectional area curve a source of
        // -U dA (a catamaran's split between her hulls), and what her transom
        // leaves open closed by a sink spread over U^2 / g behind it, as if
        // the water that follows her transom were part of her. ASSUMED: that
        // length (the dead water a wet transom drags, and the hollow behind a
        // dry one, are both about that long).
        const int hulls = zs > 0.f ? 2 : 1;
        for (size_t k = 0; k < thickness_.size(); ++k) {
            const ThicknessSegment& g = thickness_[k];
            const float q = -u * g.dA / float(hulls);
            for (int h = 0; h < hulls; ++h) {
                const float z = hulls == 2 ? (h == 0 ? -zs : zs) : 0.f;
                emit("thick_" + std::to_string(k) + "_" + std::to_string(h), origin.x + fx * g.x + sx * z,
                     origin.z + fz * g.x + sz * z, 0.f, 0.f, g.radius, 0.f, 0.f, 0.f, 0.f, 0.f, fx * g.halfLength,
                     fz * g.halfLength, 0.f, q, g.draft);
            }
        }
        if (transom_.area > 0.f) {
            const float l = std::max(u * u / kG, transom_.halfBreadth);
            const float x = transom_.x - 0.5f * l;
            const float q = -u * transom_.area / float(hulls);
            for (int h = 0; h < hulls; ++h) {
                const float z = hulls == 2 ? (h == 0 ? -zs : zs) : 0.f;
                emit("thick_close_" + std::to_string(h), origin.x + fx * x + sx * z, origin.z + fz * x + sz * z, 0.f,
                     0.f, 0.5f * transom_.halfBreadth, 0.f, 0.f, 0.f, 0.f, 0.f, fx * 0.5f * l, fz * 0.5f * l, 0.f, q,
                     transom_.draft);
            }
        }
    } else if (ripples_) {
        for (size_t k = 0; k < hull_.strips.size(); ++k) {
            const WeightStrip& s = hull_.strips[k];
            emit("weight_" + std::to_string(k), origin.x + fx * s.x + sx * s.z, origin.z + fz * s.x + sz * s.z, 0.f,
                 0.f, s.radius, 0.f, 0.f, 0.f, 0.f, hull_.mass * kG * s.share, fx * s.halfLength,
                 fz * s.halfLength);
        }
    }

    // Her bow wave where it breaks (Waves::Thickness, bowFoam): see kBowLand.
    bow_ = {};
    bool broke = false;
    if (bowFoam && hull_.waves == Waves::Thickness && bowLines_.valid && u > 0.f) {
        const BowLines& b  = bowLines_;
        const float     ta = b.tanEntry;
        const float     F  = u / std::sqrt(kG * b.draft);
        const float     zn = 2.2f * ta * std::sqrt(1.f + ta * ta) / (1.f + F);// Z_b g / U^2 (1 / cos a = sqrt(1 + tan^2 a))
        bow_.froude   = F;
        bow_.unsteady = zn > 0.5f;
        bow_.height   = std::min(zn, 0.5f) * u * u / kG;
        bow_.landing  = std::sqrt(2.f * kG * bow_.height);
        const float air   = math::smoothstep(kBowLand[0], kBowLand[1], bow_.landing) * std::min(bow_.landing / kBowLandFull, 1.f);
        const float crest = 0.5f * bow_.height;// the broken crest's width
        const float r     = std::max(crest, 1.25f * texel);
        bow_.breaking     = air * crest / r;
        if (bow_.breaking > 1e-3f) {
            broke = true;
            const float brk = bow_.breaking;
            // her half-breadth at the water at x (her sections', one hull's)
            const auto breadth = [&](float x) {
                const auto& c = b.curve;
                if (x <= c.front().x) return c.front().halfBreadth;
                for (size_t i = 0; i + 1 < c.size(); ++i) {
                    if (x <= c[i + 1].x) {
                        const float t = (x - c[i].x) / std::max(c[i + 1].x - c[i].x, 1e-6f);
                        return c[i].halfBreadth + (c[i + 1].halfBreadth - c[i].halfBreadth) * t;
                    }
                }
                return c.back().halfBreadth;
            };
            const int   hulls    = zs > 0.f ? 2 : 1;
            const float entry    = b.xStem - b.xShoulder;
            const float at[3]    = {b.xStem - 0.2f * entry, b.xStem - 0.5f * entry, b.xShoulder};
            const float share[3] = {0.6f, 0.9f, 1.f};// rising to her crest and past it
            for (int k = 0; k < 3; ++k) {
                const float bx = breadth(at[k]);
                // each side of her bow about her centreline, or (a catamaran)
                // both sides of each hull's bow, the inner ones into the tunnel;
                // the broken water from her side outward
                for (int h = 0; h < hulls; ++h) {
                    const float zc = hulls == 2 ? (h == 0 ? -zs : zs) : 0.f;
                    for (const float side : {1.f, -1.f}) {
                        const float z = zc + side * (bx + r);
                        emit("bow_" + std::to_string(k) + "_" + std::to_string(h) + (side > 0.f ? "s" : "p"),
                             origin.x + fx * at[k] + sx * z, origin.z + fz * at[k] + sz * z,
                             sx * side * kKelvinTan * u, sz * side * kKelvinTan * u, r, kBowFoam * brk * share[k],
                             kBowAir * brk * share[k], kBowTurb * u * brk, 0.5f * brk);
                    }
                }
            }
        }
    }
    if (!broke) {
        for (auto it = last_.begin(); it != last_.end();) {
            if (it->first.rfind("bow_", 0) == 0) it = last_.erase(it);
            else ++it;
        }
    }
}
