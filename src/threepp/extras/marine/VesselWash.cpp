
#include "threepp/extras/marine/VesselWash.hpp"

#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/DisplacedMesh.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;
using namespace threepp::marine;

namespace {

    float smoothstep(float a, float b, float x) {
        const float t = std::clamp((x - a) / (b - a), 0.f, 1.f);
        return t * t * (3.f - 2.f * t);
    }

    // The plan form the ocean's hull footprint has (water_displace.comp
    // hullLocal), as a share of the half beam at s in [-1, 1], stern to bow.
    float planForm(float s) {
        return s >= 0.f ? std::pow(std::max(1.f - s * s, 0.f), 0.6f) : 1.f - 0.25f * s * s;
    }

}// namespace

VesselWash::VesselWash(DisplacedMesh& ocean, uint32_t patch, Hull hull, float size, float rippleSize)
    : ocean_(ocean), patch_(patch), hull_(std::move(hull)) {
    size_       = size > 0.f ? size : std::clamp(20.f * hull_.length, 30.f, 120.f);
    rippleSize_ = std::min(size_, rippleSize > 0.f ? rippleSize : std::max(10.f * hull_.length, 16.f));
    if (hull_.strips.empty()) hull_.strips = footprintStrips(hull_);
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

void VesselWash::clear() {
    ocean_.wakePatches.at(patch_).size = 0.f;
    last_.clear();
}

void VesselWash::emit(const std::string& name, float x, float z, float vx, float vz, float radius, float foam,
                      float aeration, float turbulence, float lane, float push, float ax, float az) {
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
    if (push != 0.f && ripples_) {
        s.push  = push;
        s.ax    = ax;
        s.az    = az;
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
    const float lane   = smoothstep(0.15f, 0.6f, std::abs(u));
    static const std::string kHullNames[2] = {"hull_0", "hull_1"};
    if (lane > 0.f) {
        const float dry = smoothstep(0.35f, 0.85f, froude);// a transom runs dry and white when she is fast for her length
        const float x   = hull_.centre - std::copysign(hl, u);
        const int hulls = zs > 0.f ? 2 : 1;
        for (int k = 0; k < hulls; ++k) {
            const float z = hulls == 2 ? (k == 0 ? -zs : zs) : 0.f;
            emit(kHullNames[k], origin.x + fx * x + sx * z, origin.z + fz * x + sz * z, 0.15f * vbx, 0.15f * vbz,
                 std::max(rh, 1.25f * texel), smoothstep(0.45f, 1.05f, froude) * dry, dry, 0.07f * std::abs(u), lane);
        }
    } else {
        last_.erase(kHullNames[0]);
        last_.erase(kHullNames[1]);
    }

    // Her weight on the water, laid out along her hulls: what makes her waves.
    if (ripples_) {
        for (size_t k = 0; k < hull_.strips.size(); ++k) {
            const WeightStrip& s = hull_.strips[k];
            emit("weight_" + std::to_string(k), origin.x + fx * s.x + sx * s.z, origin.z + fz * s.x + sz * s.z, 0.f,
                 0.f, s.radius, 0.f, 0.f, 0.f, 0.f, hull_.mass * kG * s.share, fx * s.halfLength,
                 fz * s.halfLength);
        }
    }
}
