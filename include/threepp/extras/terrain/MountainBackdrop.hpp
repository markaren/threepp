// Distant mountain ranges round a play area: one ring of terrain from a kilometre or two
// out to the horizon, for scenes whose near ground is something else (a glacier, a sea,
// a TerrainGenerator tile, a heightmap).
//
// The ranges are analytic, so they cost nothing to sample and have no edge:
//   - derivative-damped fBm: each finer octave is weakened where the ground is already
//     steep, which leaves sharp crests, smooth flanks and gullies running down them, the
//     look erosion gives without simulating it;
//   - a domain warp, and a thin ridged layer along the crests;
//   - massifs: big peaks placed where a shot needs them.
// Its rings are spaced as far apart as its segments are wide, so nothing smears toward the
// centre, and every vertex carries a cavity value ("aCavity": hollow > 0, crest < 0).
//
// Shading::PerPixel (GLRenderer only): a ShaderMaterial on the engine's physical shader
// chunks decides snow and rock per pixel, from a normal that carries relief finer than the
// mesh. Snow holds on gentle ground, higher up and in hollows; crests and steep faces go
// bare; rock is banded by strata, with snow on the ledges between bands. An exponential
// aerial haze fades range behind range into the sky's horizon colour, warmer toward the
// sun; it replaces scene fog for this mesh.
// Shading::Textured: the same rule baked once into a colour and a normal map over the ring
// (texelsPerSegment texels across each segment, so a few metres at five kilometres, finer
// nearer) on a MeshStandardMaterial, with scene fog for the haze. It works under every
// renderer; VulkanRenderer's deferred path does not shade a ShaderMaterial, so use this one
// there. It costs the two maps (48 MB each at the defaults) and a second or so to bake.
//
//   terrain::MountainBackdropOptions o;
//   o.ground = [](float x, float z) { return glacierHeight(x, z); };
//   o.massifs.push_back({Vector2(4000, 2500), 1600, 1100});
//   o.hazeColor = horizon;   // the sky's colour at the horizon
//   o.sunDirection = sunDir;
//   auto ranges = terrain::MountainBackdrop::create(o);
//   scene->add(ranges);

#ifndef THREEPP_MOUNTAINBACKDROP_HPP
#define THREEPP_MOUNTAINBACKDROP_HPP

#include "threepp/math/Color.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace threepp::terrain {

    // A big peak placed on purpose: a Gaussian bump (radius = sigma) shaped by the ranges'
    // own noise, so it reads as part of them.
    struct Massif {
        Vector2 centre;// world x, z (m)
        float radius = 1600;
        float height = 1100;// added at the centre (m)
    };

    struct MountainBackdropOptions {
        enum class Shading {
            PerPixel,
            Textured
        };

        float innerRadius = 1250;// the ranges begin to rise here (m from the world origin)
        float fullRadius = 2800; // full height from here outward
        float outerRadius = 9000;// edge of the mesh
        int segments = 2048;     // around the ring; the ring spacing follows

        unsigned int seed = 70;
        float height = 1050;      // typical height of a range (m)
        float crestHeight = 260;  // the thin ridged layer along the crests (m)
        float featureScale = 2600;// metres per base-noise period
        std::vector<Massif> massifs;

        // The surface the ranges stand on; empty = flat at 0. Inside innerRadius the
        // backdrop sits `sink` metres under it, out of sight below the play area's ground.
        std::function<float(float, float)> ground;
        float sink = 4;

        Shading shading = Shading::PerPixel;
        int texelsPerSegment = 3;// Textured: map resolution
        float snowSlope = 0.16f;// slope (1 - normal.y) up to which snow lies on open ground
        Color snowColor{0.86f, 0.89f, 0.95f};
        Color rockColor{0.018f, 0.019f, 0.022f};// the darker strata
        Color rockBandColor{0.07f, 0.066f, 0.062f};// the lighter strata

        // Aerial haze (PerPixel): match hazeColor to the sky at the horizon.
        float hazeDistance = 9000;// e-folding distance (m)
        Color hazeColor{0.62f, 0.68f, 0.76f};
        Color hazeSunColor{1.f, 0.64f, 0.28f};// tint toward the sun
        Vector3 sunDirection{0, 1, 0};        // toward the sun
    };

    class MountainBackdrop: public Group {
    public:
        explicit MountainBackdrop(MountainBackdropOptions options = {});

        static std::shared_ptr<MountainBackdrop> create(MountainBackdropOptions options = {});

        // Height of the ranges at world (x, z); the mesh samples exactly this.
        [[nodiscard]] float heightAt(float x, float z) const;

        // The ring itself (share its geometry and material to put the ranges in a second
        // scene, e.g. one baked into an environment map).
        [[nodiscard]] const std::shared_ptr<Mesh>& mesh() const;

        [[nodiscard]] const MountainBackdropOptions& options() const;

        // PerPixel only: retune the haze (no-op for Textured, which takes scene fog).
        void setHaze(const Color& color, const Color& sunColor, const Vector3& sunDirection, float distance);

    private:
        MountainBackdropOptions options_;
        std::shared_ptr<Mesh> mesh_;
    };

}// namespace threepp::terrain

#endif//THREEPP_MOUNTAINBACKDROP_HPP
