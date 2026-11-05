#ifndef THREEPP_IMAGINGSONAR_HPP
#define THREEPP_IMAGINGSONAR_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/sensors/VisionSensor.hpp"
#include "threepp/helpers/ImagingSonarModel.hpp"
#include "threepp/math/Matrix3.hpp"

#include <future>
#include <memory>
#include <vector>

namespace threepp {

    class VulkanRenderer;

    /**
     * A forward-looking multibeam imaging sonar on the Vulkan renderer's ray
     * tracer: the ImagingSonarModel's ray table fired through the scene's
     * acceleration structure, folded into a SonarFrame by the acoustic
     * synthesizer (sonar equation, beamforming, speckle, multipath,
     * reverberation, noise; see ImagingSonarModel.hpp).
     *
     * An Object3D whose world frame is the transducer: forward is local -Z,
     * the receive array lies along local X, lookAt() aims it (camera
     * convention). Fire from the frame loop after renderer.render():
     *
     *     sonar.pingBegin(renderer);              // frame N
     *     ...                                     // render, present
     *     if (sonar.pingReady(renderer)) sonar.pingCollect(renderer, frame);
     *
     * The frame describes the pose the sonar held at pingBegin. Acoustic
     * materials are keyed on stable instance ids; acoustic proxies (a solid
     * membrane for a net the camera draws as twine) go on
     * VulkanRenderer::kSensorOnlyLayer with setSensorOnlySurfaces(true), as
     * for every ray-traced sensor.
     *
     * The renderer's visual fog is kept out of the trace (the sonar models
     * its own water column); `params` is the tracer's LidarParams, set up for
     * that and rarely worth touching.
     *
     * Reproducible: noise and reverberation are keyed on the ping count, the
     * scatterer phases on the world and the seed, so the same scene, poses
     * and ping sequence give the same frames bit for bit.
     */
    class ImagingSonar: public Object3D, public VisionSensor {

    public:
        // Physics and display, read every ping. Geometry fields (swath,
        // beams, beamwidths, taper, sampling, seed) take effect on rebuild().
        ImagingSonarModel model;
        AcousticMaterials materials;
        LidarParams params;
        // Fold threads; 0 = hardware concurrency, capped at 8.
        unsigned threads = 0;

        explicit ImagingSonar(const ImagingSonarModel& model = ImagingSonarModel::OculusM750d());
        ~ImagingSonar() override;

        [[nodiscard]] bool usesCameraLookAtConvention() const override { return true; }

        // Rebuild the ray table and beam patterns from `model`'s geometry.
        void rebuild();

        [[nodiscard]] unsigned rayCount() const;
        [[nodiscard]] const ImagingSonarSynthesizer& synthesizer() const { return *synth_; }

        // One ping from the current pose, blocking on the trace.
        void ping(VulkanRenderer& renderer, SonarFrame& out);

        void pingBegin(VulkanRenderer& renderer);
        [[nodiscard]] bool pingFired() const { return handle_ >= 0; }
        [[nodiscard]] bool pingReady(const VulkanRenderer& renderer) const;
        // False when nothing was fired.
        bool pingCollect(VulkanRenderer& renderer, SonarFrame& out);

        /**
         * Take delivery of the trace and fold it on a worker thread, so the frame
         * loop never waits for the acoustics (tens of ms for a 512-beam fan).
         * False when nothing was fired or a fold is still running. The fold reads
         * a copy of model and materials taken here.
         */
        bool pingCollectAsync(VulkanRenderer& renderer);
        // A fold is running, or its frame is waiting to be taken.
        [[nodiscard]] bool folding() const { return asyncPending_; }
        [[nodiscard]] bool frameReady() const;
        // The folded frame; false when none is ready.
        bool takeFrame(SonarFrame& out);

        // The per-ray hits and the pose of the last collected ping.
        [[nodiscard]] const std::vector<LidarReturn>& lastReturns() const { return returns_; }
        [[nodiscard]] const Vector3& lastOrigin() const { return origin_; }
        [[nodiscard]] std::uint64_t pingCount() const { return pings_; }

        void resetNoise() override;

    private:
        std::unique_ptr<ImagingSonarSynthesizer> synth_;
        std::vector<LidarBeam> beams_;
        std::vector<LidarReturn> returns_;
        Vector3 origin_;
        Matrix3 rotation_;
        std::uint64_t pingKey_ = 0;
        std::uint64_t pings_ = 0;
        int handle_ = -1;

        std::future<void> fold_;
        bool asyncPending_ = false;
        std::vector<LidarReturn> asyncReturns_;
        SonarFrame asyncFrame_;

        void syncParams();
        void waitFold();
    };

}// namespace threepp

#endif//THREEPP_IMAGINGSONAR_HPP
