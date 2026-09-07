// Shared base for the ranging vision sensors (depth camera, LIDAR).
//
// Why these are not driven like an Imu: a scan needs a Renderer (a raster
// depth pass or a walk of the path tracer's acceleration structure), and the
// physics step loop has no renderer and runs at a different rate. So the
// vision sensors stay PULLED — the frame loop calls scan() — while everything
// that makes a measurement reproducible comes from the Sensor base:
//
//   * one seeded PRNG per sensor instance. The pre-port code drew range noise
//     from a function-local `static std::mt19937 rng{std::random_device{}()}`,
//     which is three separate reproducibility bugs stacked: the seed came from
//     the OS (a recorded dataset can never be replayed), the stream was shared
//     by every sensor instance in the process (a sensor's noise depended on how
//     many times its neighbours had been scanned), and std::normal_distribution
//     is not specified to produce the same numbers across standard libraries.
//   * a declarative RangeNoiseModel — a value you can log next to the dataset,
//     re-roll per episode for domain randomization, or set to zero for a
//     ground-truth capture.
//   * the sim clock, so a scan is stamped with simulation time. Register the
//     sensor with a PhysxWorld and the clock tracks the physics automatically;
//     otherwise advance it from the frame loop (see Sensor's sim-clock block).
//
// Rate gating comes along for free: give the sensor a rateHz and register it,
// and scanDue() flips true at that rate so a 10 Hz LIDAR in a 60 Hz app scans
// on the right frames.
//
//     DepthSensor sensor(60.f, 320, 240);
//     sensor.rangeNoise = {0.01f, 0.002f, 0.f, /*seed*/ 7};
//     sensor.setRateHz(10);
//     world.registerSensor(&sensor);          // clock + rate gate from physics
//     ...
//     if (sensor.scanDue()) {
//         sensor.scan(renderer, scene, cloud);
//         record(sensor.lastScanTime(), cloud);
//     }

#ifndef THREEPP_SENSORS_VISIONSENSOR_HPP
#define THREEPP_SENSORS_VISIONSENSOR_HPP

#include "threepp/extras/sensors/Sensor.hpp"

#include <cmath>
#include <cstdint>

namespace threepp {

    class VisionSensor: public Sensor {

    public:
        // Per-return range noise. Public so a domain-randomization loop can
        // re-roll it between episodes; changing `seed` re-seeds the stream on
        // the next scan, changing the sigmas takes effect immediately (so an
        // interactive slider does not restart the noise sequence every frame).
        RangeNoiseModel rangeNoise;

        VisionSensor(Object3D& node, const RangeNoiseModel& noise, double rateHz = 0.0)
            : Sensor(node, rateHz), rangeNoise(noise), rng_(noise.seed), seededWith_(noise.seed) {}

        // Sim time stamped on the most recent scan. 0 before the first scan.
        [[nodiscard]] double lastScanTime() const { return lastScanTime_; }

        // True when the rate gate says a scan is due. Only meaningful for a
        // sensor that is both rate-gated (setRateHz) and driven (registered
        // with a PhysxWorld); an ungated sensor is always due, which is the
        // "scan every frame" default the examples use. Cleared by beginScan().
        [[nodiscard]] bool scanDue() const { return rateHz() <= 0.0 || scanDue_; }

        // Re-seed the noise stream from rangeNoise.seed and forget the last
        // scan time. Call after an episode reset so two episodes with the same
        // seed produce the same noise.
        //
        // Virtual because a sensor may not apply the noise itself: DepthSensor
        // on Vulkan pushes its model down into a path-traced back-end that owns
        // the actual stream, and a reset that stopped at the front door would
        // leave the episode unreplayable on exactly one backend.
        virtual void resetNoise() {
            rng_ = SplitMix64{rangeNoise.seed};
            seededWith_ = rangeNoise.seed;
            lastScanTime_ = 0.0;
        }

    protected:
        // Every scan() implementation calls this first: it re-seeds if the
        // caller changed the seed, stamps the scan with the current sim time,
        // and consumes the rate gate.
        void beginScan() {
            if (rangeNoise.seed != seededWith_) {
                rng_ = SplitMix64{rangeNoise.seed};
                seededWith_ = rangeNoise.seed;
            }
            lastScanTime_ = simTime();
            scanDue_ = false;
        }

        // Corrupt one clean range [m]. A zero model returns `range` untouched
        // and draws no random numbers, so a ground-truth capture is bit-exact
        // and costs nothing. Returns of exactly this call order are what the
        // seed reproduces — iterate beams in a stable order.
        [[nodiscard]] float applyRangeNoise(float range) {
            if (!rangeNoise.active()) return range;
            // hypot() guards against intermediate overflow, which range sigmas in
            // metres never come near, and it costs an order of magnitude more than
            // the arithmetic — measurable at 65k returns per scan. Take it only
            // when there is genuinely a second term; hypot(a, 0) is exactly |a|,
            // so the common range-independent model is bit-identical.
            const float perMetre = range * rangeNoise.stddevPerMetre;
            const float sigma = perMetre == 0.f ? std::abs(rangeNoise.stddev)
                                                : std::hypot(rangeNoise.stddev, perMetre);
            float out = range + rangeNoise.bias;
            if (sigma != 0.f) out += sigma * static_cast<float>(rng_.nextGaussian());
            return out;
        }

        // A pulled sensor has nothing to do on a physics substep: tick() has
        // already latched the clock, so all that is left is to arm the rate
        // gate that scanDue() reports. The scan itself happens in the frame
        // loop, where the renderer lives.
        void sample(double /*dt*/, double /*simTime*/) override { scanDue_ = true; }

    private:
        SplitMix64 rng_;
        std::uint64_t seededWith_;
        double lastScanTime_ = 0.0;
        bool scanDue_ = false;
    };

}// namespace threepp

#endif// THREEPP_SENSORS_VISIONSENSOR_HPP
