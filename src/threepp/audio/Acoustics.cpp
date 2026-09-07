
#include "threepp/audio/Acoustics.hpp"

#include "threepp/audio/Audio.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Ray.hpp"
#include "threepp/objects/Mesh.hpp"

#include <algorithm>
#include <cmath>
#include <limits>


using namespace threepp;

namespace {

    // Tetrahedral jitter directions. Fixed, so occlusion is stable
    // frame-to-frame (and testable) — an RNG here reads as a flutter.
    const Vector3 _jitterDirs[4]{
            Vector3(1, 1, 1).normalize(),
            Vector3(1, -1, -1).normalize(),
            Vector3(-1, 1, -1).normalize(),
            Vector3(-1, -1, 1).normalize()};

    // Nudge past the surface we just hit, so the next query does not find it again.
    constexpr float _surfaceEps = 1e-3f;

    // Degenerate geometry (coplanar sheets, self-intersecting proxies) can hand
    // out hits forever; stop counting well before that costs a frame.
    constexpr int _maxMarchHits = 8;

}// namespace


void AcousticScene::add(const Mesh& mesh, AcousticSurface surface) {

    const auto geometry = mesh.geometry();
    if (!geometry) return;

    remove(mesh);

    Entry entry{&mesh, BVH{}, surface};
    entry.bvh.build(*geometry);

    entries_.emplace_back(std::move(entry));
}

void AcousticScene::remove(const Mesh& mesh) {

    entries_.erase(
            std::remove_if(entries_.begin(), entries_.end(), [&](const Entry& e) { return e.mesh == &mesh; }),
            entries_.end());
}

float AcousticScene::transmission(const Vector3& from, const Vector3& to,
                                  int extraRays, float jitterRadius,
                                  std::vector<AcousticDebugRay>* debugRays) const {

    if (debugRays) debugRays->clear();

    const int rayCount = 1 + std::max(0, extraRays);

    Vector3 axis;
    axis.subVectors(to, from);
    if (axis.lengthSq() < 1e-12f) {
        axis.set(0, 0, 1);
    } else {
        axis.normalize();
    }

    Matrix4 inverse;
    Vector3 localFrom, localTo, direction, offset, origin, worldHit;

    const auto castRay = [&](const Vector3& target, std::optional<Vector3>& firstHit) {
        float gain = 1.f;
        float firstDistance = std::numeric_limits<float>::infinity();

        for (const auto& entry : entries_) {

            if (entry.surface.transmission >= 1.f) continue;

            // Read the pose fresh: movers work without a refit as long as
            // their geometry is rigid.
            inverse.copy(*entry.mesh->matrixWorld).invert();

            localFrom.copy(from).applyMatrix4(inverse);
            localTo.copy(target).applyMatrix4(inverse);

            direction.subVectors(localTo, localFrom);
            const float length = direction.length();
            if (length < 1e-6f) continue;
            direction.multiplyScalar(1.f / length);

            // March closest-hit rather than asking a single any-hit question: a
            // segment crossing a CLOSED solid enters and exits, so two surface
            // hits are ONE wall of material. A double-walled building gives 4
            // hits = 2 walls; an open single-sided quad gives 1 hit = 1 wall.
            origin.copy(localFrom);
            float travelled = 0.f;
            int hits = 0;

            while (hits < _maxMarchHits) {

                const auto hit = entry.bvh.raycast(Ray(origin, direction), length - travelled);
                if (!hit) break;

                if (hits == 0) {
                    worldHit.copy(hit->point).applyMatrix4(*entry.mesh->matrixWorld);
                    const float distance = worldHit.distanceTo(from);
                    if (distance < firstDistance) {
                        firstDistance = distance;
                        firstHit = worldHit;
                    }
                }

                ++hits;

                travelled += hit->distance + _surfaceEps;
                if (travelled >= length) break;

                origin.copy(hit->point).addScaledVector(direction, _surfaceEps);
            }

            if (hits > 0) {
                gain *= std::pow(entry.surface.transmission, std::ceil(static_cast<float>(hits) / 2.f));
            }
        }

        return gain;
    };

    float total = 0.f;

    for (int i = 0; i < rayCount; ++i) {

        Vector3 target = to;

        if (i > 0) {
            const auto& dir = _jitterDirs[(i - 1) % 4];

            // Spread the jitter across the segment, never along it.
            offset.copy(axis).multiplyScalar(axis.dot(dir));
            offset.subVectors(dir, offset);

            if (offset.lengthSq() < 1e-8f) {
                offset.crossVectors(axis, _jitterDirs[i % 4]);
            }

            target.add(offset.normalize().multiplyScalar(jitterRadius));
        }

        std::optional<Vector3> firstHit;
        const float gain = castRay(target, firstHit);
        total += gain;

        if (debugRays) debugRays->emplace_back(AcousticDebugRay{from, target, gain, firstHit});
    }

    return total / static_cast<float>(rayCount);
}

std::optional<AcousticScene::Hit> AcousticScene::closestHit(const Vector3& origin, const Vector3& direction,
                                                            float maxDistance) const {

    Matrix4 inverse;
    Matrix3 normalMatrix;
    Vector3 localOrigin, localDir, point, normal;

    std::optional<Hit> best;
    float bestDistance = maxDistance;

    for (const auto& entry : entries_) {

        inverse.copy(*entry.mesh->matrixWorld).invert();

        localOrigin.copy(origin).applyMatrix4(inverse);

        // Directions carry no translation, so transform a point one unit along
        // the ray instead; its length is how many local units a world unit is.
        localDir.copy(origin).add(direction).applyMatrix4(inverse).sub(localOrigin);
        const float scale = localDir.length();
        if (scale < 1e-9f) continue;
        localDir.multiplyScalar(1.f / scale);

        const auto hit = entry.bvh.raycast(Ray(localOrigin, localDir), bestDistance * scale);
        if (!hit) continue;

        point.copy(hit->point).applyMatrix4(*entry.mesh->matrixWorld);
        const float distance = point.distanceTo(origin);
        if (distance > bestDistance) continue;

        normalMatrix.getNormalMatrix(*entry.mesh->matrixWorld);
        normal.copy(hit->normal).applyNormalMatrix(normalMatrix);
        if (normal.dot(direction) > 0) normal.negate();

        bestDistance = distance;
        best = Hit{distance, point, normal, &entry};
    }

    return best;
}

AcousticEnvironment AcousticScene::probe(const Vector3& origin, int rayCount, int maxBounces,
                                         float maxDistance) const {

    AcousticEnvironment env;
    if (entries_.empty() || rayCount <= 0 || maxBounces <= 0) return env;

    // Fibonacci sphere: an even, deterministic spread. An RNG here would make
    // the reverb estimate shimmer between frames (and defeat the selftest).
    const float golden = math::PI * (3.f - std::sqrt(5.f));

    int escaped = 0;
    int pathCount = 0;
    double pathTotal = 0;
    double absorptionTotal = 0;

    Vector3 direction, position;

    for (int i = 0; i < rayCount; ++i) {

        const float y = 1.f - 2.f * (static_cast<float>(i) + 0.5f) / static_cast<float>(rayCount);
        const float radius = std::sqrt(std::max(0.f, 1.f - y * y));
        const float theta = golden * static_cast<float>(i);

        direction.set(std::cos(theta) * radius, y, std::sin(theta) * radius);
        position.copy(origin);

        for (int bounce = 0; bounce < maxBounces; ++bounce) {

            const auto hit = closestHit(position, direction, maxDistance);
            if (!hit) {
                ++escaped;
                break;
            }

            pathTotal += hit->distance;
            absorptionTotal += hit->entry->surface.absorption;
            ++pathCount;

            direction.reflect(hit->normal);
            position.copy(hit->point).addScaledVector(direction, _surfaceEps);
        }
    }

    env.escapeFraction = static_cast<float>(escaped) / static_cast<float>(rayCount);

    if (pathCount == 0) return env;

    env.meanFreePath = static_cast<float>(pathTotal / pathCount);

    // A plain mean over hits, not an energy-weighted one: across a handful of
    // bounces the weights barely separate, and this keeps the number legible.
    const float meanAbsorption = std::clamp(static_cast<float>(absorptionTotal / pathCount), 0.f, 0.99f);

    if (meanAbsorption > 0.f) {
        // Eyring, with the V/S term written as meanFreePath / 4.
        env.rt60 = 0.163f * env.meanFreePath / (-4.f * std::log(1.f - meanAbsorption));
    }

    // Open sky drains the tail no matter how reflective what remains is.
    env.rt60 = std::clamp(env.rt60 * (1.f - env.escapeFraction), 0.f, 8.f);
    env.wetLevel = std::clamp((1.f - env.escapeFraction) * (env.rt60 / (env.rt60 + 0.5f)), 0.f, 1.f);

    return env;
}


AcousticsSystem::AcousticsSystem(AcousticScene& scene, AudioListener& listener)
    : scene_(&scene), listener_(&listener) {}

void AcousticsSystem::add(PositionalAudio& audio) {

    sources_.emplace_back(Source{&audio, 0.f});
}

void AcousticsSystem::remove(const PositionalAudio& audio) {

    sources_.erase(
            std::remove_if(sources_.begin(), sources_.end(), [&](const Source& s) { return s.audio == &audio; }),
            sources_.end());
}

void AcousticsSystem::setEnabled(bool flag) {

    enabled_ = flag;
    probeTimer_ = 1e9f;// re-probe as soon as we are back on
}

bool AcousticsSystem::enabled() const {

    return enabled_;
}

void AcousticsSystem::setProbeInterval(float seconds) {

    probeInterval_ = std::max(0.f, seconds);
}

void AcousticsSystem::update(float dt) {

    debugRays_.clear();

    Vector3 listenerPos, sourcePos;
    listener_->getWorldPosition(listenerPos);

    std::vector<AcousticDebugRay> rays;

    // Exponential smoothing keeps the filter from being retuned on every
    // wobble of the listener — a zipper otherwise.
    const float alpha = 1.f - std::exp(-dt / tau_);

    for (auto& source : sources_) {

        float target = 0.f;

        if (enabled_) {
            source.audio->getWorldPosition(sourcePos);
            target = 1.f - scene_->transmission(listenerPos, sourcePos, 4, 0.35f, &rays);

            debugRays_.insert(debugRays_.end(), rays.begin(), rays.end());
        }

        source.occlusion += (target - source.occlusion) * alpha;
        source.audio->setOcclusion(source.occlusion);
    }

    probeTimer_ += dt;

    if (enabled_ && probeTimer_ >= probeInterval_) {
        // Proxy-geometry probes are sub-millisecond; a worker thread would cost
        // more in synchronisation than it saves.
        probed_ = scene_->probe(listenerPos);
        probeTimer_ = 0.f;
    }

    const float targetRt60 = enabled_ ? probed_.rt60 : 0.f;
    const float targetWet = enabled_ ? probed_.wetLevel : 0.f;

    const float beta = 1.f - std::exp(-dt / reverbTau_);

    environment_.rt60 += (targetRt60 - environment_.rt60) * beta;
    environment_.wetLevel += (targetWet - environment_.wetLevel) * beta;
    environment_.meanFreePath = probed_.meanFreePath;
    environment_.escapeFraction = probed_.escapeFraction;

    listener_->setReverb(environment_.rt60, environment_.wetLevel);

    for (auto& source : sources_) {
        source.audio->setReverbSend(environment_.wetLevel);
    }
}

const std::vector<AcousticDebugRay>& AcousticsSystem::debugRays() const {

    return debugRays_;
}

float AcousticsSystem::occlusionOf(const PositionalAudio& audio) const {

    for (const auto& source : sources_) {
        if (source.audio == &audio) return source.occlusion;
    }

    return 0.f;
}

const AcousticEnvironment& AcousticsSystem::environment() const {

    return environment_;
}
