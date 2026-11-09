#include "threepp/helpers/MarineRadar.hpp"

#include "threepp/math/Matrix4.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include <cmath>

using namespace threepp;

namespace {

    std::int64_t floorDiv(std::int64_t a, std::int64_t n) {
        std::int64_t q = a / n;
        if ((a % n) != 0 && ((a < 0) != (n < 0))) --q;
        return q;
    }
    std::size_t ringIndex(std::int64_t slot, unsigned spokes) {
        const std::int64_t m = slot % static_cast<std::int64_t>(spokes);
        return static_cast<std::size_t>(m < 0 ? m + spokes : m);
    }

}// namespace

MarineRadar::MarineRadar(const MarineRadarModel& m)
    : VisionSensor(*this, RangeNoiseModel{0.f, 0.f, 0.f, 0x7AD4ULL}), model(m) {
    params.detectorThreshold = 0.f;
    params.maxReturns = 1;
    params.samplesPerBeam = 1;
    // Keep the scene's visual fog out of the trace (see ImagingSonar).
    params.mediumExtinction = 1e-5f;
    params.mediumSurfaceY = -1e20f;
    params.mediumAlbedo = 0.f;
    anchorRpm_ = model.rpm;
    rebuild();
}

MarineRadar::~MarineRadar() = default;

void MarineRadar::rebuild() {
    synth_ = std::make_unique<MarineRadarSynthesizer>(model);
    const unsigned spokes = std::max(8u, model.spokes);
    const unsigned bins = std::max(1u, model.rangeBins);
    image_.reset(spokes, bins, std::max(1.f, model.minRange), model.maxRange);
    ring_.assign(static_cast<std::size_t>(spokes) * bins, {});
    ringLabel_.assign(static_cast<std::size_t>(spokes) * bins, -1);
    ringAmp_.assign(static_cast<std::size_t>(spokes) * bins, 0.f);
    ringHeading_.assign(spokes, 0.f);
    ringExtent_.assign(spokes, 0u);
    started_ = false;
    // A slot in flight was traced for the old geometry: it is dropped, never
    // folded into the new ring (the renderer reclaims its scan slot).
    handle_ = -1;
    lastSpokes_.clear();
    builtRangeBins_ = model.rangeBins;
    builtMinRange_ = model.minRange;
    builtMaxRange_ = model.maxRange;
}

double MarineRadar::turnOffsetAt(double t) const {
    // The clock set back is a new timeline: the angle is t * rpm / 60 again,
    // as a fresh radar's. Otherwise a new rpm takes over at the last update,
    // from the angle the antenna had then, instead of moving it to where it
    // would be had it always turned at that rate (back, for a lower one).
    if (t < lastTime_) return 0.0;
    return turnOffset_ + lastTime_ * (static_cast<double>(anchorRpm_) - static_cast<double>(model.rpm)) / 60.0;
}

double MarineRadar::turnsNow() const {
    const double t = simTime();
    return t * model.rpm / 60.0 + turnOffsetAt(t);
}

std::int64_t MarineRadar::slotNow() const {
    return static_cast<std::int64_t>(std::floor(turnsNow() * static_cast<double>(image_.spokes)));
}

float MarineRadar::antennaBearingDeg() const {
    const double turns = turnsNow();
    return static_cast<float>(360.0 * (turns - std::floor(turns)));
}

unsigned MarineRadar::update(VulkanRenderer& renderer) {
    // The ring and the picture are laid out by the range scale: a new one
    // (an operator's range switch) rebuilds them before anything is folded.
    // The raw fields are compared, not the image's clamped ones, which a
    // minRange under 1 m would never match.
    if (model.rangeBins != builtRangeBins_ || model.minRange != builtMinRange_ ||
        model.maxRange != builtMaxRange_) {
        rebuild();
    }
    lastSpokes_.clear();
    const unsigned spokes = image_.spokes;
    const std::int64_t K = synth_->kernelHalfSlots();
    const double t = simTime();
    // The raw clock, not the antenna's slot: a lower rpm also moves the slot
    // t * rpm back, which is a slower antenna, not an earlier time.
    const bool rewound = t < lastTime_;
    const std::int64_t now = slotNow();
    turnOffset_ = turnOffsetAt(t);
    anchorRpm_ = model.rpm;
    lastTime_ = t;

    if (!started_ || rewound) {
        // First call, or the clock went back (by any amount: an episode
        // reset): start the sweep here. The picture is of a time that has
        // not happened yet, so it goes too. A trace in flight is of the old
        // timeline: it is dropped as rebuild() drops one, not read back,
        // which would wait on its fence for rows that are traced again.
        handle_ = -1;
        if (started_) image_.reset(image_.spokes, image_.bins, image_.minRange, image_.maxRange);
        started_ = true;
        nextSpoke_ = now;
        nextFire_ = now - K;
        tracedUpTo_ = nextFire_;
    }

    if (handle_ >= 0 && (renderer.scanLidarReady(handle_) || ++framesWaiting_ > 3)) collect(renderer);

    // A stall of more than a revolution: drop what can no longer be shown.
    if (now - nextSpoke_ > static_cast<std::int64_t>(spokes)) {
        nextSpoke_ = now - static_cast<std::int64_t>(spokes) / 2;
        if (nextFire_ < nextSpoke_ - K) {
            nextFire_ = nextSpoke_ - K;
            tracedUpTo_ = std::max(tracedUpTo_, nextFire_);
        }
    }

    // The ring holds one revolution: a spoke whose kernel reaches back past
    // it can no longer be formed.
    nextSpoke_ = std::max(nextSpoke_, tracedUpTo_ - static_cast<std::int64_t>(spokes) + K);

    // Spokes whose whole kernel is traced and that the antenna has passed.
    const auto bins = static_cast<std::size_t>(image_.bins);
    const double slotsPerSecond = model.rpm / 60.0 * spokes;
    while (nextSpoke_ + K < tracedUpTo_ && nextSpoke_ <= now) {
        RadarSpoke sp;
        const std::uint64_t revKey = echo::combineKeys(model.seed, static_cast<std::uint64_t>(floorDiv(nextSpoke_, spokes)));
        synth_->formSpoke(
                model, nextSpoke_,
                [&](std::int64_t s) { return &ring_[ringIndex(s, spokes) * bins]; },
                [&](std::int64_t s) { return &ringLabel_[ringIndex(s, spokes) * bins]; },
                [&](std::int64_t s) { return &ringAmp_[ringIndex(s, spokes) * bins]; },
                [&](std::int64_t s) { return ringExtent_[ringIndex(s, spokes)]; }, revKey, sp);
        // When the antenna passed the slot, at the rate it turns at now; a
        // stopped one gives the clock (it passed them before).
        sp.time = slotsPerSecond > 0.0
                          ? (static_cast<double>(nextSpoke_) - turnOffset_ * spokes) / slotsPerSecond
                          : t;
        sp.bearingDeg = static_cast<float>(360.0 * static_cast<double>(ringIndex(nextSpoke_, spokes)) / spokes);
        float wb = ringHeading_[ringIndex(nextSpoke_, spokes)] + sp.bearingDeg;
        wb = std::fmod(wb, 360.f);
        sp.worldBearingDeg = wb < 0.f ? wb + 360.f : wb;
        image_.store(sp);
        lastSpokes_.push_back(std::move(sp));
        ++nextSpoke_;
    }

    if (handle_ < 0) fire(renderer, now + K + 1);
    return static_cast<unsigned>(lastSpokes_.size());
}

void MarineRadar::fire(VulkanRenderer& renderer, std::int64_t upTo) {
    if (upTo <= nextFire_) return;
    const auto count = static_cast<unsigned>(std::min<std::int64_t>(upTo - nextFire_, std::max(1u, maxSlotsPerBatch)));

    beginScan();
    if (!parent) updateMatrixWorld();
    getWorldPosition(pendingOrigin_);
    // The rotation alone: it is inverted by transposing below, which a scaled
    // parent would turn into a scale of the vertical pattern's argument.
    pendingRotation_.setFromMatrix4(Matrix4().extractRotation(*matrixWorld));
    Vector3 fwd(0.f, 0.f, -1.f);
    fwd.applyMatrix3(pendingRotation_);
    heading_ = static_cast<float>(std::atan2(static_cast<double>(fwd.x), -static_cast<double>(fwd.z)) * 180.0 /
                                  echo::kPi);
    if (heading_ < 0.f) heading_ += 360.f;
    // The rays go out level, turned to the heading; the pattern tilts with her.
    pendingLevel_.setFromMatrix4(Matrix4().makeRotationY(static_cast<float>(-heading_ * echo::kPi / 180.0)));
    Matrix3 sensorFromWorld = pendingRotation_;
    sensorFromWorld.transpose();
    pendingTilt_.multiplyMatrices(sensorFromWorld, pendingLevel_);

    const double antennaHeight = pendingOrigin_.y - model.seaLevel;
    pendingDirs_.clear();
    pendingSolid_.clear();
    for (unsigned i = 0; i < count; ++i) {
        const std::int64_t s = nextFire_ + i;
        synth_->slotRays(s, antennaHeight, pendingDirs_, pendingSolid_);
        ringHeading_[ringIndex(s, image_.spokes)] = heading_;
    }
    beams_.resize(pendingDirs_.size());
    for (std::size_t i = 0; i < pendingDirs_.size(); ++i) {
        Vector3 d = pendingDirs_[i];
        d.applyMatrix3(pendingLevel_).normalize();
        beams_[i].origin = pendingOrigin_;
        beams_[i].direction = d;
    }
    const echo::RangeResponse resp(0.5 * 299792458.0 * model.pulseLength, echo::PulseKind::CW);
    params.minRange = std::max(0.001f, model.housingRange);// the blind zone is the synthesizer's
    params.maxRange = model.maxRange + static_cast<float>(resp.halfSpan());

    handle_ = renderer.scanLidarBegin(beams_, params);
    if (handle_ >= 0) {
        pendingFirst_ = nextFire_;
        pendingCount_ = count;
        nextFire_ += count;
        framesWaiting_ = 0;
    }
}

void MarineRadar::collect(VulkanRenderer& renderer) {
    const int h = handle_;
    handle_ = -1;
    ++scansCollected_;
    returns_.clear();
    if (!renderer.scanLidarCollect(h, returns_)) return;
    const unsigned per = synth_->raysPerSlot();
    if (returns_.size() < static_cast<std::size_t>(per) * pendingCount_) return;
    const unsigned spokes = image_.spokes;
    const auto bins = static_cast<std::size_t>(image_.bins);
    for (unsigned i = 0; i < pendingCount_; ++i) {
        const std::int64_t s = pendingFirst_ + i;
        const std::size_t ri = ringIndex(s, spokes);
        const std::uint64_t revKey = echo::combineKeys(model.seed, static_cast<std::uint64_t>(floorDiv(s, spokes)));
        ringExtent_[ri] = synth_->foldSlot(model, materials, s, &returns_[static_cast<std::size_t>(i) * per],
                         &pendingDirs_[static_cast<std::size_t>(i) * per], &pendingSolid_[static_cast<std::size_t>(i) * per],
                         pendingOrigin_, pendingLevel_, revKey, &ring_[ri * bins], &ringLabel_[ri * bins],
                         &ringAmp_[ri * bins], image_.bins, &pendingTilt_);
    }
    tracedUpTo_ = pendingFirst_ + pendingCount_;
}
