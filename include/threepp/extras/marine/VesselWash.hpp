// VesselWash — what a vessel leaves ON the water behind her.
//
// A patch of an ocean's wake field (DisplacedMesh::WakeField) that trails one
// vessel, and her sources in it. Two kinds of producer, and neither knows the
// other:
//
//   her propulsors  Each one's race is taken from its thrust (race()), run
//                   down its own line until it has widened to the surface (it
//                   widens by kSpread of the way it has run; the vessel has
//                   moved on meanwhile), and put there as a source: the water
//                   moving at what is left of the race's speed, turbulence a
//                   share of that, and air by how hard the race is for how
//                   little water it has over it (w^2 / g d, a Froude number on
//                   its submergence): none under about 1, bubbles above, a
//                   white film well above, and a race that hard breaks the
//                   surface wider than it came up. A second, fainter source
//                   lies over the disc itself: the race's air seen through the
//                   water it has not yet come up through, so the mark starts
//                   at her stern. A race is not steady, and where it surfaces
//                   wanders a part of its own width (kWander).
//   her hulls       The turbulent water a hull drags astern: a lane a hull's
//                   breadth wide from each stern, with no air in it until she
//                   is fast for her length. And her WAVES, where the wake
//                   field has ripples (WakeField::rippleResolution): she bears
//                   down on the water with her weight, laid out along her
//                   hulls (Hull::strips), and the field's small linear sea
//                   makes of that the waves a weight of that shape makes at
//                   her speed. No wave is drawn: at rest she makes none, at
//                   speed the fan has the angle and the wavelengths of that
//                   speed, and it bends where she turned.
//
// Nothing here moves the vessel or knows how she is driven: the scene hands in
// where she is, her velocity and what her propulsors are doing this frame.
//
// This is the model of python/examples/rigs/usv_rig.py's Wash (the boats of the
// Nørvasundet twin), constant for constant; a change to one belongs in both.
// VesselWash_test holds this to numbers taken from that class.
//
// ── ORDER IN A FRAME ────────────────────────────────────────────────────────
//     ocean->wakeField.resolution = 1024;        // before the first render (latched)
//     ocean->wakeField.rippleResolution = 512;   // her waves (latched with it)
//     marine::VesselWash wash(*ocean, 0, hull);
//     ... each frame, after the vessel has moved:
//     ocean->clearWakeSources();                 // once, before every vessel's wash
//     wash.update(dt, origin, forward, velocity, propulsors);
//     renderer.render(scene, camera);
//
// Vulkan only, as the wake field is: on a sea whose wake field is off
// (resolution 0) the sources are simply never read.

#ifndef THREEPP_VESSELWASH_HPP
#define THREEPP_VESSELWASH_HPP

#include "threepp/math/Vector3.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace threepp {

    class DisplacedMesh;

}

namespace threepp::marine {

    class VesselWash {

    public:
        // A stretch of hull bearing down on the water, in the vessel frame (x
        // ahead of her origin, z to starboard): its share of her weight, the
        // radius of the gaussian with the spread of a load as wide as the hull
        // is at the water there (0.41 x that width), and half its length.
        struct WeightStrip {
            float x          = 0.f;
            float z          = 0.f;
            float share      = 0.f;
            float radius     = 0.f;
            float halfLength = 0.f;
        };

        struct Hull {
            float length     = 0.f;// overall (m)
            // Her footprint at the water, as the ocean's HullExclusion has it:
            // half length, half beam, and how far ahead of her origin its
            // centre lies.
            float halfLength = 0.f;
            float halfBeam   = 0.f;
            float centre     = 0.f;
            // A catamaran's hulls lie this far to either side of her
            // centreline; 0 = one hull.
            float hullOffset = 0.f;
            float mass       = 0.f;// kg: what makes her waves
            // Her weight along her hulls. Empty = laid out from the footprint
            // (footprintStrips()); a vessel that has her sectional areas
            // gives the real thing.
            std::vector<WeightStrip> strips;
        };

        // One propulsor, this frame.
        struct Propulsor {
            std::string name;
            Vector3 position;    // world: the centre of its disc or its nozzle
            Vector3 force;       // world (N): what it puts on the vessel; its race runs against it
            float   radius = 0.f;// of the disc or the nozzle (m)
            bool    jet    = false;
        };

        // A propulsor's figures at the last update, for a readout.
        struct Race {
            float thrust    = 0.f;// N
            float race      = 0.f;// its speed through the water (m/s)
            float atSurface = 0.f;// what is left of it where it surfaces (m/s)
            float run       = 0.f;// how far it ran to get there (m)
            float froude    = 0.f;// w^2 / g d on the water over it
            float aeration  = 0.f;
            float foam      = 0.f;
        };

        static constexpr float kSpread     = 0.3f; // a race's radius grows by this much of the distance it has run
        static constexpr float kTurbulence = 0.45f;// rms turbulent speed of a race where it surfaces, as a share of its speed there
        static constexpr float kEddy       = 2.4f; // her patch's large eddies, in radii of her widest race at the surface
        static constexpr float kWander     = 0.45f;// how far a race's surfacing point strays to either side, in its radii there
        static constexpr int   kStrips     = 8;    // stretches a side her weight is laid out in along her hulls
        static constexpr float kG          = 9.81f;

        // Scales the air her propulsors take down (1 = as computed).
        float air = 1.f;
        // The ocean cascades a propulsor's depth is read against
        // (DisplacedMesh::sampleHeight): swell and the mid band, as her
        // buoyancy has it.
        uint32_t heightMask = 0b011u;

        // `patch` is the patch of the ocean's wake field that is hers. `size`
        // is its side (m): how much water astern she keeps, and, since a patch
        // has the field's resolution whatever its size, how fine her mark is
        // drawn (0 = 20 lengths of her, between 30 and 120 m). `rippleSize` is
        // the side of the window her waves are kept in (0 = 10 lengths of her,
        // at least 16 m; never more than her patch). The ocean must outlive
        // the wash.
        VesselWash(DisplacedMesh& ocean, uint32_t patch, Hull hull, float size = 0.f, float rippleSize = 0.f);

        // After the vessel has moved: place her patch and add this frame's
        // sources. `origin` is her origin at the water (world), `forward` her
        // bow's way, `velocity` hers over the ground (m/s). The scene clears
        // the ocean's sources once a frame (DisplacedMesh::clearWakeSources)
        // before its vessels' updates.
        void update(float dt, const Vector3& origin, const Vector3& forward, const Vector3& velocity,
                    const std::vector<Propulsor>& propulsors = {});

        // A source of any other producer of hers (a paddle's blade, an anchor
        // going down): `at` a point and `velocity` the water velocity it
        // imparts over the ground, both world; `push` the force (N) it bears
        // down on the water with, which is what makes waves. After update(),
        // every frame the producer is at work; a frame without it ends its
        // trail.
        void put(const std::string& name, const Vector3& at, const Vector3& velocity, float radius,
                 float foam = 0.f, float aeration = 0.f, float turbulence = 0.f, float lane = 0.f, float push = 0.f);

        // Empty her patch and forget where her producers were (after she is
        // placed elsewhere, or to take her mark away).
        void clear();

        [[nodiscard]] float size() const { return size_; }
        [[nodiscard]] float rippleSize() const { return rippleSize_; }
        [[nodiscard]] const Hull& hull() const { return hull_; }
        [[nodiscard]] const std::unordered_map<std::string, Race>& report() const { return report_; }

        // The speed of a propulsor's race THROUGH the water (m/s) from
        // momentum theory: what a thrust (N) takes from a disc or a nozzle of
        // `area` (m2) that the water meets at `advance` (m/s). A propeller is
        // an actuator disc, T = 2 rho A (V + v) v, and its race far astern
        // runs 2 v faster than the water round it; a jet takes water in at V
        // and throws it out at Vj, T = rho A Vj (Vj - V), and its race is
        // Vj - V.
        [[nodiscard]] static float race(float thrust, float advance, float area, bool jet = false, float rho = 1025.f);

        // A hull's weight laid out from her footprint alone, for a vessel that
        // has no sectional areas to give: kStrips stretches a side over the
        // plan form the ocean's hull footprint has (a bow that tapers, a stern
        // that keeps three quarters of her beam), each bearing in proportion
        // to its breadth. ASSUMED: an even draft along her, and a half
        // section whose load sits 0.42 of its breadth off the centreline
        // (between a box's 0.5 and a wedge's 0.33).
        [[nodiscard]] static std::vector<WeightStrip> footprintStrips(const Hull& hull);

    private:
        struct Point {
            float x = 0.f, z = 0.f;
        };

        DisplacedMesh& ocean_;
        uint32_t patch_;
        Hull hull_;
        float size_;
        float rippleSize_;
        bool ripples_ = false;// the wake field has ripples (read at each update)
        float t_ = 0.f;
        std::unordered_map<std::string, Point> last_;// each producer's source point at the last update (world x, z)
        std::unordered_set<std::string> putNames_;   // the scene's own producers that put() since the last update
        std::unordered_map<std::string, Race> report_;

        void emit(const std::string& name, float x, float z, float vx, float vz, float radius, float foam,
                  float aeration, float turbulence, float lane, float push = 0.f, float ax = 0.f, float az = 0.f);
    };

}// namespace threepp::marine

#endif//THREEPP_VESSELWASH_HPP
