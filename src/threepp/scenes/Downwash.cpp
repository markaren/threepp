#include "threepp/scenes/Downwash.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace threepp;

namespace {

    float smooth01(float a, float b, float v) {
        const float t = std::clamp((v - a) / (b - a), 0.f, 1.f);
        return t * t * (3.f - 2.f * t);
    }

    // [0, 1) from a row index: displace()'s per-point jitter.
    float hash01(std::uint32_t x) {
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        return float(x >> 8) * (1.f / 16777216.f);
    }

}// namespace

Downwash::Source& Downwash::addJet(const Vector3& center, float thrust, float radius) {
    if (count_ >= kMaxSources) throw std::length_error("Downwash: at most 8 sources");
    Source& s = sources_[count_++];
    s = Source{};
    s.kind = Kind::Jet;
    s.center = center;
    s.thrust = thrust;
    s.radius = radius;
    ++version_;
    return s;
}

Downwash::Source& Downwash::addRings(const Vector3& center, float startTime, float speed, float spacing,
                                     int count, float amplitude) {
    if (count_ >= kMaxSources) throw std::length_error("Downwash: at most 8 sources");
    Source& s = sources_[count_++];
    s = Source{};
    s.kind = Kind::Rings;
    s.center = center;
    s.startTime = startTime;
    s.speed = speed;
    s.spacing = spacing;
    s.count = count;
    s.amplitude = amplitude;
    ++version_;
    return s;
}

Downwash::Source& Downwash::source(std::size_t i) {
    if (i >= count_) throw std::out_of_range("Downwash: no such source");
    return sources_[i];
}

const Downwash::Source& Downwash::source(std::size_t i) const {
    if (i >= count_) throw std::out_of_range("Downwash: no such source");
    return sources_[i];
}

void Downwash::remove(std::size_t i) {
    if (i >= count_) throw std::out_of_range("Downwash: no such source");
    for (std::size_t j = i + 1; j < count_; ++j) sources_[j - 1] = sources_[j];
    --count_;
    ++version_;
}

float Downwash::inducedVelocity(float thrust, float radius, float airDensity) {
    const float area = 3.14159265f * radius * radius;
    if (!(thrust > 0.f) || !(area > 0.f) || !(airDensity > 0.f)) return 0.f;
    return std::sqrt(thrust / (2.f * airDensity * area));
}

float Downwash::jetShare(const Source& s) {
    if (!s.enabled || s.kind != Kind::Jet) return 0.f;
    float share;
    if (s.strength >= 0.f) {
        share = s.strength;
    } else {
        static const float ref = inducedVelocity(kReferenceThrust, kReferenceRadius);
        share = inducedVelocity(s.thrust, std::max(s.radius, 1e-3f)) / ref;
    }
    return std::clamp(share, 0.f, 3.f);
}

float Downwash::heightFade(float h) {
    return 1.f - smooth01(2.f, 9.f, h);
}

float Downwash::peakRadius(float radius, float h) {
    return 0.6f * (radius / kReferenceRadius) + 0.30f * std::max(h, 0.f);
}

float Downwash::decayRadius(float radius, float h) {
    return 1.2f * (radius / kReferenceRadius) + 0.55f * std::max(h, 0.f);
}

// KEEP IN SYNC with dwWarp in shaders/downwash.glsl. The jet strikes the
// ground and turns outward: a point at radius r moves to sqrt(r^2 + c(r)^2),
// with c(r) = c0 exp(-(r / 2Rh)^2). Near the axis that empties a disc of
// radius c0; the mapping keeps the points' order (c0 < 2 Rh), so the cloud
// does not fold, and its tail dies a few Rh out. The rim, where the moved
// points pile up, is lifted a little. Each particle takes its own share of
// c0 (`jitter`), so the hollow's edge is a thinning band, not a cut.
Vector3 Downwash::warp(const Vector3& p0, float jitter) const {
    const float own = 0.45f + 0.55f * std::clamp(jitter, 0.f, 1.f);
    Vector3 p = p0;
    for (std::size_t i = 0; i < count_; ++i) {
        const Source& s = sources_[i];
        const float share = jetShare(s);
        if (share <= 0.f) continue;
        const Vector3& c = s.center;
        const float ground = std::isnan(s.surfaceY) ? p.y : s.surfaceY;
        const float h = std::max(c.y - ground, 0.f);
        const float k = share * heightFade(h) * (1.f - smooth01(c.y - 0.3f, c.y + 0.6f, p.y));
        if (k <= 0.f) continue;
        const float rh = peakRadius(s.radius, h) + 0.8f;
        const float c0 = std::min(hollowRadius * k, 1.8f * rh) * own;
        const float dx = p.x - c.x, dz = p.z - c.z;
        const float r = std::sqrt(dx * dx + dz * dz);
        if (r > 4.f * rh) continue;
        const float ce = c0 * std::exp(-(r / (2.f * rh)) * (r / (2.f * rh)));
        const float r2 = std::sqrt(r * r + ce * ce);
        float ux = 1.f, uz = 0.f;
        if (r > 1e-5f) {
            ux = dx / r;
            uz = dz / r;
        }
        const float q = (r2 - rh) / rh;
        p.x = c.x + ux * r2;
        p.z = c.z + uz * r2;
        p.y += rimLift * c0 * std::exp(-q * q);
    }
    return p;
}

void Downwash::displace(float* xyz, std::size_t n, std::size_t stride) const {
    if (!xyz || stride < 3) return;
    for (std::size_t i = 0; i < n; ++i) {
        float* v = xyz + i * stride;
        const Vector3 w = warp(Vector3(v[0], v[1], v[2]), hash01(static_cast<std::uint32_t>(i) * 0x9e3779b9u + 0x5bd1e995u));
        v[0] = w.x;
        v[1] = w.y;
        v[2] = w.z;
    }
}
