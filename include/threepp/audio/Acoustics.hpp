
#ifndef THREEPP_ACOUSTICS_HPP
#define THREEPP_ACOUSTICS_HPP

#include "threepp/math/Vector3.hpp"
#include "threepp/utils/BVH.hpp"

#include <optional>
#include <vector>


namespace threepp {

    class Mesh;
    class AudioListener;
    class PositionalAudio;

    struct AcousticSurface {
        // Fraction of sound energy transmitted through the surface.
        // 0 = solid concrete, ~0.6 = curtain/foliage.
        float transmission = 0.f;
        // Fraction of incident energy absorbed on reflection; the remainder
        // reflects. Concrete ~0.05-0.1, curtain/soft ~0.6.
        float absorption = 0.3f;
    };

    struct AcousticDebugRay {
        Vector3 from;
        Vector3 to;
        float gain;
        // Where the ray first met a surface, if it met one at all.
        std::optional<Vector3> firstHit;
    };

    // What the space around a point sounds like, from a bounced-ray probe.
    struct AcousticEnvironment {
        float rt60 = 0.f;          // seconds
        float meanFreePath = 0.f;  // meters (over bounce segments that hit)
        float escapeFraction = 1.f;// fraction of rays that left the scene
        float wetLevel = 0.f;      // suggested reverb send [0,1]
    };

    // Geometric occlusion: how much of a sound survives the walls between two
    // points. Depends on BVH + math only, so it is usable without an audio
    // device (headless tests included).
    class AcousticScene {

    public:
        void add(const Mesh& mesh, AcousticSurface surface = {});

        void remove(const Mesh& mesh);

        // Combined transmission gain [0,1] between two world-space points.
        // 1 = unobstructed. Casts one center ray plus `extraRays` jittered rays
        // (deterministic offsets around `to`, radius `jitterRadius`) and averages.
        [[nodiscard]] float transmission(const Vector3& from, const Vector3& to,
                                         int extraRays = 4, float jitterRadius = 0.35f,
                                         std::vector<AcousticDebugRay>* debugRays = nullptr) const;

        // Reverberation estimate around `origin`, from deterministic rays that
        // bounce specularly off the registered surfaces.
        [[nodiscard]] AcousticEnvironment probe(const Vector3& origin,
                                                int rayCount = 128, int maxBounces = 4,
                                                float maxDistance = 100.f) const;

    private:
        struct Entry {
            const Mesh* mesh;
            BVH bvh;
            AcousticSurface surface;
        };

        struct Hit {
            float distance;
            Vector3 point;
            Vector3 normal;// world space, facing the ray origin
            const Entry* entry;
        };

        // Nearest hit across all entries, in world space.
        [[nodiscard]] std::optional<Hit> closestHit(const Vector3& origin, const Vector3& direction,
                                                    float maxDistance) const;

        std::vector<Entry> entries_;
    };

    // Drives Audio::setOcclusion for a set of positional sources from an
    // AcousticScene, once per frame, with smoothing.
    class AcousticsSystem {

    public:
        AcousticsSystem(AcousticScene& scene, AudioListener& listener);

        void add(PositionalAudio& audio);

        void remove(const PositionalAudio& audio);

        // Bypass: drives occlusion back to 0 rather than freezing it.
        void setEnabled(bool flag);

        [[nodiscard]] bool enabled() const;

        // How often the environment probe is re-cast, in seconds.
        void setProbeInterval(float seconds);

        void update(float dt);

        [[nodiscard]] const std::vector<AcousticDebugRay>& debugRays() const;

        [[nodiscard]] float occlusionOf(const PositionalAudio& audio) const;

        // Last smoothed probe result.
        [[nodiscard]] const AcousticEnvironment& environment() const;

    private:
        struct Source {
            PositionalAudio* audio;
            float occlusion{};
        };

        AcousticScene* scene_;
        AudioListener* listener_;

        bool enabled_{true};
        float tau_{0.06f};
        // Room transitions should glide, not snap.
        float reverbTau_{0.3f};

        float probeInterval_{0.5f};
        float probeTimer_{1e9f};
        AcousticEnvironment probed_;
        AcousticEnvironment environment_;

        std::vector<Source> sources_;
        std::vector<AcousticDebugRay> debugRays_;
    };

}// namespace threepp

#endif//THREEPP_ACOUSTICS_HPP
