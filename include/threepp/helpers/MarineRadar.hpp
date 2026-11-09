#ifndef THREEPP_MARINERADAR_HPP
#define THREEPP_MARINERADAR_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/sensors/VisionSensor.hpp"
#include "threepp/helpers/MarineRadarModel.hpp"
#include "threepp/math/Matrix3.hpp"

#include <memory>
#include <vector>

namespace threepp {

    class VulkanRenderer;

    /**
     * A rotating marine radar on the Vulkan renderer's ray tracer.
     *
     * The node is the antenna's mount: it turns about local +Y, clockwise
     * seen from above, relative bearing 0 along local -Z (the bow, when the
     * node is mounted level and facing forward). The rotation is SIMULATED
     * from the sensor's sim clock (rpm), not by spinning the node: drive the
     * clock (setSimTime / advanceClock) and call update() once per frame after
     * renderer.render(). Each call collects the slots fired last frame, forms
     * every spoke the antenna has swept past (see MarineRadarModel.hpp for the
     * physics), stores them in image(), and fires the slots the antenna will
     * sweep next. A spoke therefore shows the world where it was when the
     * beam passed: a moving target leaves a fresh echo each revolution.
     *
     * Radar materials are keyed on stable instance ids (the ocean gets
     * RadarMaterial::Sea()). The renderer's visual fog is kept out of the
     * trace; rain is the model's own RainCells.
     */
    class MarineRadar: public Object3D, public VisionSensor {

    public:
        MarineRadarModel model;
        RadarMaterials materials;
        LidarParams params;
        // At most this many slots per trace (a stall then catches up over a
        // few frames instead of firing a whole revolution at once).
        unsigned maxSlotsPerBatch = 256;

        explicit MarineRadar(const MarineRadarModel& model = MarineRadarModel::OpenArray4ft());
        ~MarineRadar() override;

        [[nodiscard]] bool usesCameraLookAtConvention() const override { return true; }

        // Rebuild ray tables and patterns from `model`'s geometry; clears the picture.
        void rebuild();

        // Once per frame, after render(). Returns the number of spokes formed.
        // A new range scale (model.rangeBins, minRange or maxRange) rebuilds
        // first, by itself; the other geometry fields wait for rebuild().
        unsigned update(VulkanRenderer& renderer);

        [[nodiscard]] const RadarPolarImage& image() const { return image_; }
        [[nodiscard]] const std::vector<RadarSpoke>& lastSpokes() const { return lastSpokes_; }

        // The antenna's relative bearing now (deg, clockwise from the bow).
        [[nodiscard]] float antennaBearingDeg() const;
        // The bow's world bearing (deg from world -Z toward +X) at the last fire.
        [[nodiscard]] float headingDeg() const { return heading_; }
        [[nodiscard]] unsigned raysPerSlot() const { return synth_->raysPerSlot(); }
        [[nodiscard]] const MarineRadarSynthesizer& synthesizer() const { return *synth_; }

    private:
        std::unique_ptr<MarineRadarSynthesizer> synth_;
        RadarPolarImage image_;
        std::vector<RadarSpoke> lastSpokes_;

        // The slot ring: one complex range profile per slot of a revolution.
        std::vector<std::complex<float>> ring_;
        std::vector<std::int32_t> ringLabel_;
        std::vector<float> ringAmp_;
        std::vector<float> ringHeading_;
        std::vector<unsigned> ringExtent_;

        bool started_ = false;
        std::int64_t nextFire_ = 0, nextSpoke_ = 0, tracedUpTo_ = 0;
        // The range scale the ring and the picture were laid out for.
        unsigned builtRangeBins_ = 0;
        float builtMinRange_ = 0.f, builtMaxRange_ = 0.f;

        int handle_ = -1;
        int framesWaiting_ = 0;
        std::int64_t pendingFirst_ = 0;
        unsigned pendingCount_ = 0;
        Vector3 pendingOrigin_;
        Matrix3 pendingRotation_;// world from the antenna
        Matrix3 pendingLevel_;   // world from the antenna's level frame (its heading only)
        Matrix3 pendingTilt_;    // the antenna from its level frame: its pitch and roll
        std::vector<Vector3> pendingDirs_;
        std::vector<float> pendingSolid_;
        std::vector<LidarBeam> beams_;
        std::vector<LidarReturn> returns_;
        float heading_ = 0.f;

        [[nodiscard]] std::int64_t slotNow() const;
        void collect(VulkanRenderer& renderer);
        void fire(VulkanRenderer& renderer, std::int64_t upTo);
    };

}// namespace threepp

#endif//THREEPP_MARINERADAR_HPP
