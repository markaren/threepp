// Ocean — Vulkan PT demo of FFT-displaced water using DisplacedMesh.
//
// Single-cascade Phillips spectrum + GPU IFFT chain feeds vertex positions
// directly into the BLAS each frame; the path tracer's existing transmission
// BSDF handles refraction, Beer-Lambert absorption, and reflections. A simple
// sandy floor sits below the surface so caustics from the photon-mapping
// pass become visible as the sun moves.
//
// Phase 1 of the WebTide-style ocean integration. Multi-cascade + foam +
// procedural sky come later; for now a single 40 m tile + an HDRI sky is
// enough to validate the geometry pipeline and BLAS-rebuild-per-frame.

#include "threepp/audio/Audio.hpp"
#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/helpers/LidarWaveform.hpp"
#include "threepp/helpers/PathTracedLidarSensor.hpp"
#include "threepp/input/KeyListener.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/objects/DisplacedMesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

using namespace threepp;

namespace {

    // Boat input state — captured by KeyListener, polled each frame.
    struct BoatInput : KeyListener {
        bool W = false, A = false, S = false, D = false;
        bool shotRequest = false;// F12: dump the next frame to aaa_caps/ (artifact reports)
        void onKeyPressed(KeyEvent e) override {
            if (e.key == Key::F12) shotRequest = true;
            update(e.key, true);
        }
        void onKeyReleased(KeyEvent e) override { update(e.key, false); }
        void update(Key k, bool down) {
            if (k == Key::W || k == Key::UP)    W = down;
            if (k == Key::S || k == Key::DOWN)  S = down;
            if (k == Key::A || k == Key::LEFT)  A = down;
            if (k == Key::D || k == Key::RIGHT) D = down;
        }
    };

    // Persistent boat state. Position is world, yaw is rotation around +Y
    // (heading); pitch and roll are read each frame from wave-surface tilt
    // and aren't integrated. Forward speed is along +heading; max ~14 kn
    // (~7 m/s) for a research vessel of Gunnerus's size.
    struct BoatState {
        Vector3 position{0.f, 0.f, 0.f};
        float   yaw          = 0.f;       // radians
        float   forwardSpeed = 0.f;       // m/s along +heading
        float   smoothPitch  = 0.f;       // radians, low-passed from wave tilt
        float   smoothRoll   = 0.f;       // radians
        float   y            = 0.f;       // metres, spring-damped toward wave height
        float   vY           = 0.f;       // m/s, heave velocity (state for the spring-damper)
    };
}// namespace

namespace {

    constexpr float kTileSize    = 1000.0f;  // metres — full mesh extent and cascade-0 tile
    constexpr uint32_t kFftSize  = 1024;     // FFT resolution per cascade — drives wave detail, NOT mesh density.
    constexpr float kPlaneEdge   = kTileSize;  // mesh extends one full FFT tile in X and Z
    // Mesh density is decoupled from FFT size: the water_displace.comp samples
    // the height texture via normalised UVs (u = i / (gridDim-1)) so the mesh
    // can be any tessellation while the wave field stays at kFftSize². Halving
    // the subdivision from kFftSize-1 to kFftSize/2-1 drops the vertex count
    // from ~1 M to ~262 K — a 4× win on BLAS rebuild/refit and the per-vertex
    // displace dispatch, while wave geometry stays crisp (vertex spacing ~2 m
    // still resolves λ ≥ 4 m, and Phillips 1/k⁴ puts most energy above that).
    constexpr int   kSubdiv      = static_cast<int>(kFftSize) / 2 - 1;

    auto makeOceanMaterial() {
        auto mat = MeshPhysicalMaterial::create();
        // Pure water has no diffuse pigment — the blue comes from Beer-Lambert
        // absorption through the medium, not albedo.
        mat->color = Color::white;
        // Small roughness simulates the sub-pixel chop the FFT can't resolve.
        // 0.04 broadens the specular lobe just enough that each highlight
        // covers multiple pixels — converges fast under TAA, avoids the
        // salt-and-pepper sparkle that 0.01 + a tight-mip normal map gives
        // on distant water.
        mat->roughness = 0.04f;
        mat->metalness = 0.0f;
        mat->setIor(1.33f);
        mat->transmission = 1.0f;
        // doubleSided + thickness opts this surface into the path tracer's
        // thin-shell transmission path: every transmission crossing applies
        // Beer-Lambert for `thickness` metres of in-medium depth. The down-
        // crossing (camera → water) tints the refracted ray; the up-crossing
        // (sand → camera, after bounce) tints again. 2 m × 2 ≈ 4 m of
        // effective tint — a tropical-ocean blue that still shows refracted
        // sand under the brightest crests. Without doubleSided the BSDF
        // would need to use the actual ray distance through the medium
        // (~12 m here), which over-saturates to near-black.
        mat->side = Side::Double;
        mat->thickness = 2.0f;
        // Opt this surface into the path tracer's thin-shell BSDF: a single
        // FFT-displaced plane has no closed interior, so both faces should
        // refract as entries and Beer-Lambert applies per-crossing using
        // `thickness` as the in-medium proxy. Without this flag, the back-
        // face hit (ray bouncing off sand) would refract using eta=ior with
        // gl_HitTEXT = full water column → opaque deep blue, no see-through.
        mat->thinWalled = true;
        mat->attenuationColor = Color(0.10f, 0.45f, 0.55f);
        mat->attenuationDistance = 3.0f;
        mat->clearcoat = 0.1;

        // Sub-mesh-resolution wave detail comes from the FFT fine cascade
        // sampled directly in closest_hit (binding 32 → cascade-2 height,
        // gated on `thinWalled`). That animates with the wave field for free
        // and replaces the procedural normal map this example used to ship.
        return mat;
    }

    auto makeSandMaterial() {
        return MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0.02,0.02,0.02)).roughness(1.0f));
    }

}// namespace

// ── Procedural enclosing archipelago ────────────────────────────────────────
// A ring of rocky islands around the play area (r ≈ 385–495 m) so the scene
// reads as a sheltered Norwegian skerry bay instead of bare open horizon.
// Deterministic value-noise FBM drives everything: an angular "mass plan"
// (periodic by construction — noise sampled on a circle) picks where islands
// rise and where passes stay open sea; a radial bump profile dives both
// shores below the sand floor so the rims bury cleanly; ridged FBM adds the
// rocky relief. Surface detail comes from three maps baked in the mesh's own
// polar parameterisation (the Vulkan PT has no vertex-colour path for
// meshes): an albedo map layering two granite tones, strata banding, scree,
// grass / heather / lichen niches and a wet waterline; a tangent-space
// normal map carrying creased-slab relief far below the vertex grid; and a
// roughness map that turns the wet band glossy — all derived from the same
// height field.
namespace island {

    constexpr float kInnerR = 385.f;// boat waypoints reach ~320 m — keep clear water
    constexpr float kOuterR = 495.f;// stays inside the 1 km ocean/sand tile
    constexpr float kPeakH  = 55.f; // tallest summits (m)
    constexpr float kSkirt  = 7.f;  // rim depth — below the sand floor (-5 m)

    float smoothstepf(float e0, float e1, float x) {
        const float t = std::clamp((x - e0) / (e1 - e0), 0.f, 1.f);
        return t * t * (3.f - 2.f * t);
    }

    float hashf(int xi, int zi) {
        uint32_t h = static_cast<uint32_t>(xi) * 374761393u + static_cast<uint32_t>(zi) * 668265263u;
        h = (h ^ (h >> 13)) * 1274126177u;
        h ^= h >> 16;
        return static_cast<float>(h) * (1.f / 4294967296.f);
    }

    // Value noise with quintic fade, range [0,1].
    float vnoise(float x, float z) {
        const float fx = std::floor(x), fz = std::floor(z);
        const int xi = static_cast<int>(fx), zi = static_cast<int>(fz);
        float tx = x - fx, tz = z - fz;
        tx = tx * tx * tx * (tx * (tx * 6.f - 15.f) + 10.f);
        tz = tz * tz * tz * (tz * (tz * 6.f - 15.f) + 10.f);
        const float a = hashf(xi, zi), b = hashf(xi + 1, zi);
        const float c = hashf(xi, zi + 1), d = hashf(xi + 1, zi + 1);
        return a + (b - a) * tx + (c - a) * tz + (a - b - c + d) * tx * tz;
    }

    float fbm(float x, float z, int octaves) {
        float sum = 0.f, amp = 0.5f, norm = 0.f;
        for (int o = 0; o < octaves; ++o) {
            sum += amp * vnoise(x, z);
            norm += amp;
            amp *= 0.5f;
            // irrational-ish lacunarity + offset decorrelates the octave lattices
            const float nx = x * 1.93f + 19.7f, nz = z * 2.11f + 7.3f;
            x = nx;
            z = nz;
        }
        return sum / norm;
    }

    float heightAt(float x, float z) {
        const float r = std::sqrt(x * x + z * z);
        // ring coordinate, warped so the coastlines wander instead of circling
        float w = (r - kInnerR) / (kOuterR - kInnerR);
        w += 0.20f * (2.f * fbm(x * 0.011f, z * 0.011f, 2) - 1.f);
        if (w <= 0.f || w >= 1.f) return -kSkirt;
        const float prof = std::pow(std::sin(math::PI * w), 1.5f);

        // mass plan: low-frequency noise on a circle, thresholded — sections
        // below the band stay a submerged sill (the passes between islands)
        const float a = std::atan2(z, x);
        const float m0 = 0.65f * vnoise(7.3f + std::cos(a) * 2.9f, 3.1f + std::sin(a) * 2.9f) +
                         0.35f * vnoise(13.7f + std::cos(a) * 5.3f, 23.9f + std::sin(a) * 5.3f);
        const float m = smoothstepf(0.27f, 0.60f, m0);

        // ridged FBM (fold-over of signed noise) = sharp rocky crests
        const float ridge = 1.f - std::abs(2.f * fbm(x * 0.035f, z * 0.035f, 4) - 1.f);
        // meso relief: creased-slab ridges (λ ≈ 11 m down to ~3.5 m) — real
        // geometry now that the vertex grid resolves it; the baked normal map
        // carries only the finer scales (≤ 3 m). Scaled by the mass plan so
        // the submerged passes stay smooth sills.
        const float slab = 1.f - std::abs(2.f * fbm(x * 0.09f, z * 0.09f, 3) - 1.f);
        const float meso = 3.5f * (slab * slab - 0.45f);
        const float crest = (kPeakH * (0.45f + 0.55f * ridge * ridge) + meso) * m;
        // prof=1 mid-ring: passes top out 2 m underwater, islands rise to crest
        return -kSkirt + (crest + kSkirt - 2.f) * prof;
    }

    // Analytic finite-difference normal — seam-consistent (the height field is
    // continuous in angle) and adds sub-vertex shading detail for free. The
    // 1.2 m radius matches the ~1.4 m vertex grid so the meso slabs shade
    // correctly instead of being averaged away.
    Vector3 normalAt(float x, float z, float eps = 1.2f) {
        const float dhdx = (heightAt(x + eps, z) - heightAt(x - eps, z)) / (2.f * eps);
        const float dhdz = (heightAt(x, z + eps) - heightAt(x, z - eps)) / (2.f * eps);
        Vector3 n(-dhdx, 1.f, -dhdz);
        return n.normalize();
    }

    // High-frequency rock relief for the baked normal map — creased slabs
    // (λ ≈ 3 m) plus value-noise grain (λ ≈ 1 m): only the scales below what
    // the vertex grid carries (the λ ≥ 3.5 m slabs live in heightAt now).
    float detailHeight(float x, float z) {
        const float r2 = 1.f - std::abs(2.f * fbm(x * 0.31f + 53.f, z * 0.31f + 17.f, 3) - 1.f);
        const float g  = fbm(x * 0.9f + 9.f, z * 0.9f + 27.f, 2);
        return 0.55f * r2 * r2 + 0.10f * g;
    }

    struct BakedMaps {
        std::shared_ptr<DataTexture> albedo;
        std::shared_ptr<DataTexture> normal;
        std::shared_ptr<DataTexture> rough;
    };

    // Albedo + tangent-space normal + roughness in the mesh's polar
    // parameterisation (u = angle, v = radius): one texel ≈ 0.7 m of
    // coastline, 0.4 m radially. Each texel re-evaluates the height field —
    // centre + 4 neighbours give the slope normal AND the Laplacian
    // (crest/hollow) from the same five probes — so every mask tracks the
    // actual geometry. Rows bake in parallel; the pass is a one-off at startup.
    BakedMaps bakeMaps() {
        const int W = 4096, H = 256;
        std::vector<unsigned char> albPx(static_cast<size_t>(W) * H * 4);
        std::vector<unsigned char> nrmPx(static_cast<size_t>(W) * H * 4);
        std::vector<unsigned char> rghPx(static_cast<size_t>(W) * H * 4);
        auto toByte = [](float v) {
            return static_cast<unsigned char>(std::lround(std::clamp(v, 0.f, 1.f) * 255.f));
        };

        std::vector<int> rows(H);
        std::iota(rows.begin(), rows.end(), 0);
        std::for_each(std::execution::par, rows.begin(), rows.end(), [&](int y) {
            const float r = kInnerR + (kOuterR - kInnerR) * ((y + 0.5f) / H);
            for (int x = 0; x < W; ++x) {
                const float a = 2.f * math::PI * ((x + 0.5f) / W);
                const float ca = std::cos(a), sa = std::sin(a);
                const float wx = ca * r, wz = sa * r;

                const float eps = 2.5f;
                const float h   = heightAt(wx, wz);
                const float hxp = heightAt(wx + eps, wz), hxm = heightAt(wx - eps, wz);
                const float hzp = heightAt(wx, wz + eps), hzm = heightAt(wx, wz - eps);
                const float dhdx = (hxp - hxm) / (2.f * eps);
                const float dhdz = (hzp - hzm) / (2.f * eps);
                const float ny   = 1.f / std::sqrt(1.f + dhdx * dhdx + dhdz * dhdz);
                const float lap  = (hxp + hxm + hzp + hzm - 4.f * h) / (eps * eps);
                // convexity: exposed crests bleach, concave seams collect dirt
                const float crest = std::clamp(-lap * 0.8f, -1.f, 1.f);

                const float tone   = fbm(wx * 0.0045f, wz * 0.0045f, 2); // per-face rock tone, λ ≈ 220 m
                const float mottle = fbm(wx * 0.05f, wz * 0.05f, 3);     // λ ≈ 20 m
                const float grain  = fbm(wx * 0.7f + 5.f, wz * 0.7f + 13.f, 2);// λ ≈ 1.4 m
                const float vegL   = fbm(wx * 0.013f + 31.f, wz * 0.013f, 3);  // veg patches, λ ≈ 75 m
                const float vegS   = fbm(wx * 0.11f + 7.f, wz * 0.11f + 3.f, 2);// ragged veg edges, λ ≈ 9 m
                const float warp   = fbm(wx * 0.03f + 71.f, wz * 0.03f + 11.f, 2);

                // two-tone base: pale weathered granite vs darker gneiss,
                // blended at island scale so each face reads as its own mass
                const float t = smoothstepf(0.35f, 0.65f, tone);
                float cr = 0.26f + 0.20f * t;
                float cg = 0.245f + 0.195f * t;
                float cb = 0.235f + 0.175f * t;
                const float speck = (mottle - 0.5f) * 0.14f + (grain - 0.5f) * 0.09f;
                cr += speck;
                cg += speck;
                cb += speck * 0.9f;
                // wandering sub-horizontal strata bands on steep faces
                const float strata = 1.f + 0.09f * std::sin(h * 0.75f + 6.f * warp) *
                                                 smoothstepf(0.85f, 0.55f, ny);
                // crest/hollow shading from the Laplacian
                const float cav = 1.f + 0.14f * crest;
                cr *= strata * cav;
                cg *= strata * cav;
                cb *= strata * cav;

                // scree aprons: gentle low benches at the cliff feet collect debris
                const float scree = smoothstepf(0.60f, 0.78f, ny) * smoothstepf(1.2f, 2.6f, h) *
                                    (1.f - smoothstepf(5.f, 13.f, h)) *
                                    smoothstepf(0.35f, 0.65f, vegS) * 0.55f;
                cr += (0.41f - cr) * scree;
                cg += (0.375f - cg) * scree;
                cb += (0.315f - cb) * scree;

                // heather/shrub on mid slopes, its own patch noise
                const float hePatch = fbm(wx * 0.019f + 57.f, wz * 0.019f + 91.f, 2);
                const float heather = smoothstepf(0.45f, 0.62f, ny) * smoothstepf(1.5f, 3.5f, h) *
                                      (1.f - smoothstepf(22.f, 34.f, h)) *
                                      smoothstepf(0.45f, 0.62f, hePatch);
                cr += (0.205f + 0.05f * hePatch - cr) * heather;
                cg += (0.17f + 0.05f * hePatch - cg) * heather;
                cb += (0.105f - cb) * heather;

                // grass/moss on flat low benches; hollows accumulate soil, so
                // the Laplacian feeds the patch threshold
                const float gPatch = 0.55f * vegL + 0.30f * vegS +
                                     0.15f * std::clamp(lap * 1.5f, 0.f, 1.f);
                const float grass = smoothstepf(0.60f, 0.80f, ny) * smoothstepf(0.8f, 2.6f, h) *
                                    (1.f - smoothstepf(24.f, 38.f, h)) *
                                    smoothstepf(0.42f, 0.58f, gPatch);
                cr += (0.13f + 0.07f * vegS - cr) * grass;
                cg += (0.27f + 0.08f * vegL - cg) * grass;
                cb += (0.085f - cb) * grass;

                // pale lichen crusts on exposed high rock
                const float lich = smoothstepf(8.f, 20.f, h) *
                                   smoothstepf(0.55f, 0.75f, fbm(wx * 0.15f + 13.f, wz * 0.15f + 29.f, 2)) *
                                   std::clamp(0.5f + 0.5f * crest, 0.f, 1.f) * 0.30f;
                cr += (0.50f - cr) * lich;
                cg += (0.51f - cg) * lich;
                cb += (0.46f - cb) * lich;
                // summits bleach toward bare washed rock
                const float alt = 1.f + 0.08f * smoothstepf(28.f, 52.f, h);
                cr *= alt;
                cg *= alt;
                cb *= alt;

                // algae film straddling the waterline, then the dark wet band
                const float algae = (1.f - smoothstepf(0.6f, 1.4f, std::abs(h - 0.3f))) * 0.5f;
                cr += (0.10f - cr) * algae;
                cg += (0.15f - cg) * algae;
                cb += (0.10f - cb) * algae;
                const float wet = (1.f - smoothstepf(0.4f, 2.2f, h)) * 0.85f;
                cr += (0.095f - cr) * wet;
                cg += (0.095f - cg) * wet;
                cb += (0.09f - cb) * wet;

                // detail normal: world-plane gradient of the relief field,
                // projected onto the polar tangent frame (T = +u = angular,
                // B = +v = radial — matches the shader's derivative TBN).
                // Damped under vegetation: soil and moss smooth micro-relief.
                const float de = 0.5f;
                const float damp = 0.8f * (1.f - 0.65f * std::max(grass, heather));
                const float gx = (detailHeight(wx + de, wz) - detailHeight(wx - de, wz)) / (2.f * de) * damp;
                const float gz = (detailHeight(wx, wz + de) - detailHeight(wx, wz - de)) / (2.f * de) * damp;
                const float st = -gx * sa + gz * ca;// slope along +u (angular)
                const float sb = gx * ca + gz * sa; // slope along +v (radial)
                const float inv = 1.f / std::sqrt(st * st + sb * sb + 1.f);

                // roughness (.g multiplies material roughness): matte dry
                // granite, matte vegetation, water-slicked rock turns glossy
                float rough = 0.86f + 0.10f * (mottle - 0.5f) - 0.06f * crest;
                rough += (0.95f - rough) * std::max(grass, heather);
                rough += (0.45f - rough) * wet;

                const size_t i = (static_cast<size_t>(y) * W + x) * 4;
                albPx[i + 0] = toByte(cr);
                albPx[i + 1] = toByte(cg);
                albPx[i + 2] = toByte(cb);
                albPx[i + 3] = 255;
                nrmPx[i + 0] = toByte(-st * inv * 0.5f + 0.5f);
                nrmPx[i + 1] = toByte(-sb * inv * 0.5f + 0.5f);
                nrmPx[i + 2] = toByte(inv * 0.5f + 0.5f);
                nrmPx[i + 3] = 255;
                rghPx[i + 0] = 255;
                rghPx[i + 1] = toByte(rough);
                rghPx[i + 2] = 0;
                rghPx[i + 3] = 255;
            }
        });

        auto makeTex = [&](std::vector<unsigned char>&& px, bool srgb) {
            auto tex = DataTexture::create(ImageData{std::move(px)},
                                           static_cast<unsigned>(W), static_cast<unsigned>(H));
            if (srgb) tex->colorSpace = ColorSpace::sRGB;// normal/rough stay raw UNORM
            tex->magFilter = Filter::Linear;
            tex->minFilter = Filter::LinearMipmapLinear;
            tex->generateMipmaps = true;
            tex->needsUpdate();
            return tex;
        };
        return {makeTex(std::move(albPx), true),
                makeTex(std::move(nrmPx), false),
                makeTex(std::move(rghPx), false)};
    }

    std::shared_ptr<Mesh> build() {
        // 2048 angular columns ≈ 1.35 m spacing at mid-ring, 64 radial rows
        // ≈ 1.7 m — fine enough to resolve the λ ≥ 3.5 m meso slabs in
        // heightAt. The seam column is duplicated (u = 0 and u = 1) so UVs
        // never wrap. ~133 K verts / ~262 K tris, static BLAS built once;
        // rows fill in parallel (≈ 670 K height-field probes).
        const int NA = 2048, NR = 64;
        std::vector<float> pos(static_cast<size_t>(NA + 1) * (NR + 1) * 3);
        std::vector<float> nrm(static_cast<size_t>(NA + 1) * (NR + 1) * 3);
        std::vector<float> uv(static_cast<size_t>(NA + 1) * (NR + 1) * 2);
        std::vector<int> rows(NR + 1);
        std::iota(rows.begin(), rows.end(), 0);
        std::for_each(std::execution::par, rows.begin(), rows.end(), [&](int j) {
            const float r = kInnerR + (kOuterR - kInnerR) * (static_cast<float>(j) / NR);
            for (int i = 0; i <= NA; ++i) {
                const float a = 2.f * math::PI * (static_cast<float>(i) / NA);
                const float x = std::cos(a) * r, z = std::sin(a) * r;
                const Vector3 n = normalAt(x, z);
                const size_t v = static_cast<size_t>(j) * (NA + 1) + i;
                pos[v * 3 + 0] = x;
                pos[v * 3 + 1] = heightAt(x, z);
                pos[v * 3 + 2] = z;
                nrm[v * 3 + 0] = n.x;
                nrm[v * 3 + 1] = n.y;
                nrm[v * 3 + 2] = n.z;
                uv[v * 2 + 0] = static_cast<float>(i) / NA;
                uv[v * 2 + 1] = static_cast<float>(j) / NR;
            }
        });
        std::vector<unsigned int> idx;
        idx.reserve(static_cast<size_t>(NA) * NR * 6);
        for (int j = 0; j < NR; ++j)
            for (int i = 0; i < NA; ++i) {
                const unsigned a0 = j * (NA + 1) + i;// (i, j)
                const unsigned b0 = a0 + 1;          // (i+1, j)
                const unsigned a1 = a0 + (NA + 1);   // (i, j+1)
                const unsigned b1 = a1 + 1;          // (i+1, j+1)
                idx.insert(idx.end(), {a0, b0, b1});
                idx.insert(idx.end(), {a0, b1, a1});
            }

        auto geo = BufferGeometry::create();
        geo->setIndex(idx);
        geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        geo->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        geo->computeBoundingBox();
        geo->computeBoundingSphere();

        auto mat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                        .roughness(1.f)// baked map carries the variation
                                                        .metalness(0.f));
        const BakedMaps maps = bakeMaps();
        mat->map          = maps.albedo;
        mat->normalMap    = maps.normal;
        mat->roughnessMap = maps.rough;
        auto mesh = Mesh::create(geo, mat);
        mesh->frustumCulled = false;// the ring surrounds the camera — always partly in view
        return mesh;
    }

}// namespace island

// ── Procedural looping audio (engine + ocean/wind ambience) ─────────────────
// Same temp-WAV approach as the Shooter example: the Audio API loads files,
// so the loops are synthesised once at startup and written to the temp dir.
// Seamless looping: every deterministic component (engine harmonics, swell /
// gust LFOs) is given an exact integer number of cycles over the loop length,
// then the synth renders an extra tail whose start is crossfaded back onto
// the head — periodic terms pass through the wrap unchanged while the noise
// and one-pole filter states blend across it.
namespace {

    struct OnePole {
        float y = 0.f;
        float operator()(float x, float a) {
            y += a * (x - y);
            return y;
        }
    };
    float lpAlpha(float cutoffHz, int sr) {
        return 1.f - std::exp(-2.f * math::PI * cutoffHz / static_cast<float>(sr));
    }

    std::vector<float> normalized(std::vector<float> s, float peak) {
        float m = 0.f;
        for (float x : s) m = std::max(m, std::abs(x));
        if (m > 1e-6f)
            for (float& x : s) x *= peak / m;
        return s;
    }

    // Fold the `extra`-sample overhang back onto the head (linear crossfade).
    // out[0] == s[n] so the n-1 → 0 junction is the continuation of the tail;
    // by i == extra the signal is back on the head verbatim.
    std::vector<float> loopable(const std::vector<float>& s, int n, int extra) {
        std::vector<float> out(s.begin(), s.begin() + n);
        for (int i = 0; i < extra; ++i) {
            const float w = static_cast<float>(i) / static_cast<float>(extra);
            out[i] = s[n + i] * (1.f - w) + s[i] * w;
        }
        return out;
    }

    // 16-bit mono PCM WAV writer (verbatim from the Shooter example).
    void writeWav(const std::filesystem::path& path, const std::vector<float>& samples, int sr = 44100) {
        std::ofstream f(path, std::ios::binary);
        auto u32 = [&](uint32_t v) { f.write(reinterpret_cast<char*>(&v), 4); };
        auto u16 = [&](uint16_t v) { f.write(reinterpret_cast<char*>(&v), 2); };
        const uint32_t dataBytes = static_cast<uint32_t>(samples.size()) * 2u;
        f.write("RIFF", 4);
        u32(36 + dataBytes);
        f.write("WAVE", 4);
        f.write("fmt ", 4);
        u32(16);
        u16(1);// PCM
        u16(1);// mono
        u32(sr);
        u32(sr * 2);
        u16(2);
        u16(16);
        f.write("data", 4);
        u32(dataBytes);
        for (float x : samples) {
            const auto q = static_cast<int16_t>(std::lround(std::clamp(x, -1.f, 1.f) * 32767.f));
            f.write(reinterpret_cast<const char*>(&q), 2);
        }
    }

    // Marine diesel at mid RPM, 2 s loop. Firing rate f0 = 27 Hz (54 exact
    // cycles): harmonic stack for the tonal drone, a |sin|³ "chug" envelope
    // gating low-passed exhaust noise, and a faint band-passed mechanical
    // clatter. Played at rate 0.7 (idle) … 1.6 (full ahead) by the updater.
    std::vector<float> synthEngineLoop(int sr = 44100) {
        const float dur = 2.0f;
        const int n     = static_cast<int>(sr * dur);
        const int extra = sr / 4;
        std::mt19937 r(7);
        auto rn = [&] { return std::uniform_real_distribution<float>(-1.f, 1.f)(r); };
        const float f0 = 27.f;
        OnePole lpExhaust, lpClatHi, lpClatLo;
        const float aExhaust = lpAlpha(170.f, sr);
        const float aClatHi  = lpAlpha(1300.f, sr);
        const float aClatLo  = lpAlpha(450.f, sr);
        std::vector<float> s(n + extra);
        for (int i = 0; i < n + extra; ++i) {
            const float t    = static_cast<float>(i) / sr;
            const float chug = std::pow(0.55f + 0.45f * std::abs(std::sin(math::PI * f0 * t)), 3.f);
            float tone = 0.f;
            tone += std::sin(2.f * math::PI * f0 * t) * 0.55f;
            tone += std::sin(2.f * math::PI * 2.f * f0 * t) * 0.30f;
            tone += std::sin(2.f * math::PI * 3.f * f0 * t) * 0.16f;
            tone += std::sin(2.f * math::PI * 4.f * f0 * t) * 0.09f;
            const float w       = rn();
            const float exhaust = lpExhaust(w, aExhaust) * chug * 1.7f;
            const float clatter = (lpClatHi(w, aClatHi) - lpClatLo(w, aClatLo)) * (0.4f + 0.6f * chug) * 0.45f;
            s[i] = tone * (0.7f + 0.3f * chug) + exhaust + clatter;
        }
        return normalized(loopable(s, n, extra), 0.7f);
    }

    // Rolling sea, 8 s loop: deep low-passed noise swelling on three loop-
    // exact LFOs (k/8 Hz), plus a brighter band-passed "wash" that peaks on
    // its own sharper envelope — the crest-breaking hiss over the rumble.
    std::vector<float> synthOceanLoop(int sr = 44100) {
        const float dur = 8.0f;
        const int n     = static_cast<int>(sr * dur);
        const int extra = sr;
        std::mt19937 r(11);
        auto rn = [&] { return std::uniform_real_distribution<float>(-1.f, 1.f)(r); };
        OnePole lpDeep, lpWashHi, lpWashLo;
        const float aDeep   = lpAlpha(240.f, sr);
        const float aWashHi = lpAlpha(1500.f, sr);
        const float aWashLo = lpAlpha(500.f, sr);
        std::vector<float> s(n + extra);
        for (int i = 0; i < n + extra; ++i) {
            const float t = static_cast<float>(i) / sr;
            float swell = 0.6f * std::sin(2.f * math::PI * 0.125f * t)
                        + 0.3f * std::sin(2.f * math::PI * 0.375f * t + 1.7f)
                        + 0.1f * std::sin(2.f * math::PI * 0.625f * t + 4.1f);
            swell = 0.55f + 0.45f * swell;
            const float washEnv = std::pow(0.5f + 0.5f * std::sin(2.f * math::PI * 0.25f * t + 2.6f), 3.f);
            const float w    = rn();
            const float deep = lpDeep(w, aDeep) * swell * 1.0f;
            const float wash = (lpWashHi(w, aWashHi) - lpWashLo(w, aWashLo)) * washEnv * 0.55f;
            s[i] = deep + wash;
        }
        return normalized(loopable(s, n, extra), 0.6f);
    }

    // Wind, 8 s loop. NOT a flat noise band — that reads as TV static. The
    // "whoosh" character comes from (a) a NARROW low band whose cutoff SWEEPS
    // upward with the gust envelope (the rising pitch of a building gust),
    // (b) 12 dB/oct edges — cascaded one-poles; a single pole leaks so much
    // above cutoff that the leak IS the white-noise hiss — and (c) a hard
    // lull↔gust amplitude swing (gust², near-silent lulls) so it reads as
    // weather, not a constant carrier. A faint flutter band rides only the
    // gust peaks (gust⁴). Gust LFOs are loop-exact (k/8 Hz).
    std::vector<float> synthWindLoop(int sr = 44100) {
        const float dur = 8.0f;
        const int n     = static_cast<int>(sr * dur);
        const int extra = sr;
        std::mt19937 r(13);
        auto rn = [&] { return std::uniform_real_distribution<float>(-1.f, 1.f)(r); };
        OnePole hi1, hi2, lo1, lo2, fl1, fl2;
        const float aFlHi = lpAlpha(1000.f, sr);
        const float aFlLo = lpAlpha(450.f, sr);
        std::vector<float> s(n + extra);
        for (int i = 0; i < n + extra; ++i) {
            const float t = static_cast<float>(i) / sr;
            float gust = 0.55f * std::sin(2.f * math::PI * 0.25f * t)
                       + 0.30f * std::sin(2.f * math::PI * 0.5f * t + 1.3f)
                       + 0.15f * std::sin(2.f * math::PI * 0.875f * t + 4.0f);
            gust = std::clamp(0.5f + 0.5f * gust, 0.f, 1.f);
            // Swept band: lulls murmur at ~60–180 Hz, full gusts open to
            // ~140–620 Hz. The per-sample alpha is driven by the loop-exact
            // LFOs, so the sweep itself wraps seamlessly too.
            const float aHi = lpAlpha(180.f + 440.f * gust, sr);
            const float aLo = lpAlpha(60.f + 80.f * gust, sr);
            const float w   = rn();
            const float band    = hi2(hi1(w, aHi), aHi) - lo2(lo1(w, aLo), aLo);
            const float whoosh  = band * (0.10f + 0.90f * gust * gust);
            const float flutter = (fl1(w, aFlHi) - fl2(w, aFlLo)) * gust * gust * gust * gust * 0.18f;
            s[i] = whoosh + flutter;
        }
        return normalized(loopable(s, n, extra), 0.5f);
    }

    // Engine (spatialised at the stern) + ocean/wind ambience loops, with the
    // listener following the camera. Degrades to a no-op when no audio device
    // is available; never constructed in headless --shot capture runs.
    struct OceanSounds {
        std::unique_ptr<AudioListener> listener;
        std::unique_ptr<PositionalAudio> engine;
        std::unique_ptr<Audio> waves, wind;
        bool ok    = false;
        float rpm_ = 0.f;// smoothed RPM proxy ∈ [0,1] — the engine spools, it doesn't snap

        void init() {
            try {
                const auto dir = std::filesystem::temp_directory_path() / "threepp_ocean_sounds";
                std::filesystem::create_directories(dir);
                const auto enginePath = dir / "engine_loop.wav";
                const auto wavesPath  = dir / "waves_loop.wav";
                const auto windPath   = dir / "wind_loop.wav";
                writeWav(enginePath, synthEngineLoop());
                writeWav(wavesPath, synthOceanLoop());
                writeWav(windPath, synthWindLoop());

                listener = std::make_unique<AudioListener>();
                // Engine: full volume within ~10 m (the side/deck camera),
                // shallow inverse falloff so the chase cam still hears it and
                // the far buoy cam gets only a faint distant throb.
                engine = std::make_unique<PositionalAudio>(*listener, enginePath);
                engine->setDistanceModel(PositionalAudio::DistanceModel::Inverse);
                engine->setMinDistance(10.f);
                engine->setRolloffFactor(0.5f);
                engine->setLooping(true);
                engine->setVolume(0.f);
                engine->play();
                waves = std::make_unique<Audio>(*listener, wavesPath);
                waves->setLooping(true);
                waves->setVolume(0.f);
                waves->play();
                wind = std::make_unique<Audio>(*listener, windPath);
                wind->setLooping(true);
                wind->setVolume(0.f);
                wind->play();
                ok = true;
            } catch (const std::exception& e) {
                std::cerr << "[audio] disabled: " << e.what() << "\n";
            }
        }

        // sternWorld: engine mount position. thrusting: throttle is open
        // (autopilot under way, or W/S held) — bumps the RPM floor so the
        // engine revs as thrust is applied, before boat speed builds.
        // uw: smoothed submersion ∈ [0,1] — above-surface sound ducks under
        // water (wind almost fully, waves partially, engine least: hull noise
        // carries through the water).
        void update(float dt, const Vector3& sternWorld, float forwardSpeed,
                    bool thrusting, float uw, const PerspectiveCamera& cam,
                    float masterVolume) {
            if (!ok) return;
            listener->setMasterVolume(masterVolume);

            const float speedNorm = std::clamp(std::abs(forwardSpeed) / 8.f, 0.f, 1.f);
            float target = 0.18f + 0.82f * speedNorm;
            if (thrusting) target = std::max(target, 0.45f);
            // Spool up faster than the wind-down coast (turbo lag vs. inertia).
            const float tau = target > rpm_ ? 0.9f : 1.8f;
            rpm_ += (target - rpm_) * (1.f - std::exp(-dt / tau));

            engine->setPlaybackRate((0.7f + 0.9f * rpm_) * (1.f - 0.10f * uw));
            engine->setVolume((0.25f + 0.65f * rpm_) * (1.f - 0.35f * uw));
            engine->position.copy(sternWorld);
            engine->updateMatrixWorld(true);// push the source position to the audio engine

            // Slight speed bump on the ambience = apparent wind over the deck.
            waves->setVolume((0.45f + 0.10f * speedNorm) * (1.f - 0.55f * uw));
            wind->setVolume((0.15f + 0.08f * speedNorm) * (1.f - 0.85f * uw));

            listener->position.copy(cam.position);
            listener->quaternion.copy(cam.quaternion);
            listener->updateMatrixWorld(true);
        }
    };

}// namespace

int main(int argc, char** argv) {

    // ── Headless capture (dev iteration loop) ───────────────────────────────
    //   vulkan_ocean --shot <name.png> [--frames N] [--night] [--pt] [--vista]
    // Fixed aerial camera, N warm-up frames (TAA/denoiser converge), one PNG
    // into <project>/aaa_caps/, exit. --night starts in night mode; --pt
    // captures the path-traced reference instead of the deferred default;
    // --vista frames a high oblique overview (archipelago ring + lighthouse).
    std::string shotPath;
    int  shotFrames = 240;
    bool startNight = false;
    bool shotPT     = false;
    bool shotVista  = false;
    bool shotClose  = false;// near-surface grazing view — surface-artifact hunting
    int  toggleNightAt = 0;// --toggle: start in day, flip to night mid-run (exercises the runtime toggle path)
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shot") == 0 && i + 1 < argc) shotPath = argv[++i];
        else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--night") == 0) startNight = true;
        else if (std::strcmp(argv[i], "--pt") == 0) shotPT = true;
        else if (std::strcmp(argv[i], "--vista") == 0) shotVista = true;
        else if (std::strcmp(argv[i], "--close") == 0) shotClose = true;
        else if (std::strcmp(argv[i], "--toggle") == 0) toggleNightAt = 60;
    }
    const bool capturing = !shotPath.empty();
    int shotFrame = 0;

    Canvas canvas("Vulkan PT  Ocean", {{"vsync", false}, {"size", WindowSize{1600, 900}}});
    VulkanRenderer renderer(canvas);
    renderer.setDenoise(true);
    renderer.setRestirDIEnabled(true);
    renderer.setFireflyClamp(6.0f);
    renderer.setMaxBounces(2);
    // Trace PT at lower resolution; TAA upsamples to full swapchain by
    // accumulating jittered low-res samples into the full-res history.
    renderer.setRenderScale(0.9f);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 0.7f;
    if (shotPT) renderer.setRenderMode(VulkanRenderer::RenderMode::ReferencePT);// --pt: capture the PT reference

    RGBELoader rgbe;
    auto env = rgbe.load(std::string(DATA_FOLDER) +
                         "/textures/env/autumn_field_puresky_2k.hdr");

    Scene scene;
    scene.background = env;
    scene.environment = env;

    // Sun-like directional light. The HDR env already contains a sun (env
    // CDF + MIS will importance-sample it), so the directional is mostly
    // here to drive the photon-mapping caustics pass — kept gentle so it
    // doesn't double up with the env's own sun on the surface.
    auto sun = DirectionalLight::create(Color(1.0f, 0.95f, 0.85f), 2.0f);
    sun->position.set(2.f, 1.f, 2.f);
    Object3D sunTarget;
    sunTarget.position.set(0.f, 0.f, 0.f);
    sun->setTarget(sunTarget);
    scene.add(sun);

    // Sand floor sits directly under the ocean tile and matches its extent:
    // making the floor larger leaves a visible sand frame around the water
    // when viewed from above (the open-ocean illusion breaks). At the
    // edges, rays going past the water plane just hit the env sky, which
    // sells "horizon" better than visible beach.
    auto floor = Mesh::create(PlaneGeometry::create(kPlaneEdge, kPlaneEdge),
                              makeSandMaterial());
    floor->rotation.x = -math::PI / 2.f;
    floor->position.y = -5.f;
    scene.add(floor);

    // Enclosing archipelago ring — see namespace island above. Static mesh,
    // one BLAS build; the lighthouse beam grazes its cliffs at night.
    scene.add(island::build());

    // Ocean surface. PlaneGeometry with kSubdiv segments → kFftSize²
    // vertices. The DisplacedMesh detects the grid dimension at first-frame
    // init and runs the FFT/displace pipeline against it.
    auto oceanGeo = PlaneGeometry::create(kPlaneEdge, kPlaneEdge, kSubdiv, kSubdiv);
    oceanGeo->rotateX(-math::PI / 2.f);
    auto oceanMat = makeOceanMaterial();
    auto ocean = DisplacedMesh::create(oceanGeo, oceanMat);
    // Three-cascade FFT:
    //   tileSize0 = 1000 m → big swells, dominant macro shapes.
    //   tileSize1 =  100 m → mid-frequency waves filling each swell face.
    //   tileSize2 =    8 m → fine chop in the 4–8 m range (the rest aliases
    //                        at 1.95 m mesh spacing, but Phillips 1/k⁴ puts
    //                        most energy in the resolvable end).
    // The band-pass scheme (PhillipsSpectrum.kMin/kMax in the renderer)
    // keeps each cascade in its own k-range so they stack cleanly without
    // double-counting wavelengths the adjacent band already covers.
    ocean->params.tileSize0   = kTileSize;
    ocean->params.tileSize1   = 100.0f;
    ocean->params.tileSize2   = 8.0f;
    ocean->params.windTheta   = 0.6f;       // wind slightly off the X axis
    // windSpeed scales wave amplitude as V⁴ in Phillips, so it's the
    // dominant lever for "how big is the sea": 20 m/s = gale (10 m mountain
    // crests, dwarfs the boat), 8–10 = Beaufort 4–5 moderate sea (1–2 m
    // waves, visible chop without overpowering geometry). waveScale should
    // stay near 1 (physical) — keeping it at 0.1 just attenuates the entire
    // multi-cascade detail and reads as a glassy lake.
    ocean->params.windSpeed   = shotClose ? 3.5f : 10.0f;// --close: calm glassy sea — the artifact-hunting state
    ocean->params.waveScale   = 1.0f;
    ocean->params.choppiness  = 0.55f;      // sharper crests, more visible wave-folding
    // Per-cascade FFT resolution. Cascade-0 (big swells, 1 km tile) needs the
    // full kFftSize to keep macro-shape fidelity. Cascades 1 and 2 carry
    // shorter wavelengths whose resolvable detail saturates well below the
    // cascade-0 resolution — halving them cuts FFT cost ~2× with no visible
    // loss.
    ocean->params.textureSize0 = kFftSize;
    ocean->params.textureSize1 = kFftSize / 2;
    ocean->params.textureSize2 = kFftSize / 2;
    scene.add(ocean);

    // ── Lighthouse (scene centre) ───────────────────────────────────────────
    // Rock base + tapered white tower + red gallery + emissive lamp room. The
    // LAMP is the night-mode hero: an emissive mesh (area light for the PT /
    // deferred emissive paths) + a rotating SpotLight whose beam the deferred
    // volumetric march renders as the classic sweeping lighthouse fan.
    auto lampMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
            .color(Color(1.f, 0.95f, 0.8f)).roughness(0.4f).metalness(0.f));
    lampMat->emissive = Color(1.f, 0.85f, 0.55f);
    lampMat->emissiveIntensity = 0.f;// day: off — night toggle raises it
    {
        auto rockMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                .color(Color(0.22f, 0.21f, 0.20f)).roughness(0.95f).metalness(0.f));
        auto rock = Mesh::create(CylinderGeometry::create(7.f, 10.f, 6.f, 24), rockMat);
        rock->position.set(0.f, -1.f, 0.f);
        scene.add(rock);

        auto towerMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                .color(Color(0.92f, 0.90f, 0.86f)).roughness(0.6f).metalness(0.f));
        auto tower = Mesh::create(CylinderGeometry::create(1.9f, 2.8f, 16.f, 24), towerMat);
        tower->position.set(0.f, 10.f, 0.f);
        scene.add(tower);

        auto bandMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                .color(Color(0.75f, 0.12f, 0.10f)).roughness(0.6f).metalness(0.f));
        auto gallery = Mesh::create(CylinderGeometry::create(2.4f, 2.4f, 1.2f, 24), bandMat);
        gallery->position.set(0.f, 18.6f, 0.f);
        scene.add(gallery);

        auto lamp = Mesh::create(CylinderGeometry::create(1.4f, 1.4f, 2.2f, 16), lampMat);
        lamp->position.set(0.f, 20.3f, 0.f);
        scene.add(lamp);

        auto roof = Mesh::create(CylinderGeometry::create(0.1f, 1.8f, 1.6f, 16), bandMat);
        roof->position.set(0.f, 22.2f, 0.f);
        scene.add(roof);
    }

    // Rotating beam — narrow long-throw spot, aimed slightly below horizontal
    // so the far end grazes the swells. decay 2 = physical inverse-square; a
    // real lighthouse lamp is O(10⁵–10⁶ cd), which is what it takes to light
    // water hundreds of metres out. Off by day (intensity 0).
    auto beam = SpotLight::create(Color(1.f, 0.92f, 0.72f), 0.f,
                                  /*distance=*/600.f, /*angle=*/math::PI / 40.f,
                                  /*penumbra=*/0.45f, /*decay=*/2.f);
    beam->position.set(0.f, 20.3f, 0.f);
    Object3D beamTarget;
    beamTarget.position.set(300.f, -4.f, 0.f);
    beam->setTarget(beamTarget);
    scene.add(beam);

    // Dim blue moonlight — direction matches the procedural night env's moon
    // disc so shadows and the bright sky spot agree.
    auto moon = DirectionalLight::create(Color(0.65f, 0.75f, 1.0f), 0.f);
    moon->position.set(-0.55f, 0.60f, 0.35f);
    Object3D moonTarget;
    moonTarget.position.set(0.f, 0.f, 0.f);
    moon->setTarget(moonTarget);
    scene.add(moon);

    // ── Procedural night sky (equirect, RGBA float) ─────────────────────────
    // Deep-blue elevation gradient + faint horizon glow + star field + a moon
    // disc bright enough that the PT's env CDF importance-samples it (it acts
    // as the night "sun"). Built once; the night toggle swaps scene.environment
    // / background and the renderer re-runs PMREM + descriptor rewrites.
    std::shared_ptr<Texture> nightEnv;
    {
        // 2048×1024: one texel ≈ 0.18° ≈ ~4 screen px at this FOV — stars stay
        // point-like. At 512² a texel was ~19 px and bilinear magnification
        // rendered every star as a big square tent.
        const int W = 2048, H = 1024;
        std::vector<float> data(static_cast<size_t>(W) * H * 4, 0.f);
        // sampleEnvLod maps dir.y=+1 → v=1.0 (zenith = last row).
        const Vector3 moonDir = Vector3(-0.55f, 0.60f, 0.35f).normalize();
        for (int y = 0; y < H; ++y) {
            const float v    = (y + 0.5f) / H;
            const float elev = (v - 0.5f) * math::PI;// >0 = above horizon
            for (int x = 0; x < W; ++x) {
                const float u  = (x + 0.5f) / W;
                const float az = (u - 0.5f) * 2.f * math::PI;
                const Vector3 dir(std::cos(elev) * std::cos(az), std::sin(elev),
                                  std::cos(elev) * std::sin(az));
                // Sky gradient: near-black zenith → faint blue horizon band;
                // below the horizon a dark sea-glow so reflections aren't void.
                float r, g, b;
                if (elev >= 0.f) {
                    const float horizon = std::exp(-elev * 4.5f);
                    r = 0.004f + 0.020f * horizon;
                    g = 0.006f + 0.028f * horizon;
                    b = 0.012f + 0.050f * horizon;
                } else {
                    const float fade = std::exp(elev * 6.f);
                    r = 0.003f * fade; g = 0.004f * fade; b = 0.007f * fade;
                }
                // Moon disc (~1.7° radius) + soft glow halo.
                const float cosToMoon = std::clamp(dir.dot(moonDir), -1.f, 1.f);
                const float angTo     = std::acos(cosToMoon);
                if (angTo < 0.03f) {
                    r += 28.f; g += 32.f; b += 40.f;
                } else {
                    const float glow = 0.35f * std::exp(-angTo * angTo * 90.f);
                    r += glow * 0.65f; g += glow * 0.75f; b += glow;
                }
                const size_t i = (static_cast<size_t>(y) * W + x) * 4;
                data[i + 0] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = 1.f;
            }
        }
        // NO baked stars: a star is sub-texel at ANY practical env resolution,
        // so after bilinear magnification (+ TAA upscale) every baked star
        // renders as a ~14 px blob. Stars come from the renderer's procedural
        // direction-space star field instead (setDeferredStarfield) — crisp
        // points at every resolution/FOV. The env keeps what magnifies well:
        // gradient, horizon glow, moon (which also feeds reflections + PT CDF).
        Image img{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0};
        nightEnv = Texture::create(img);
        nightEnv->format = Format::RGBA;
        nightEnv->type = Type::Float;
        nightEnv->colorSpace = ColorSpace::Linear;
        nightEnv->mapping = Mapping::EquirectangularReflection;
        nightEnv->needsUpdate();
    }

    // ── Day/night toggle ────────────────────────────────────────────────────
    bool  night       = startNight;
    float beamSpeed   = 0.45f;// rad/s — ~14 s revolution, classic lighthouse cadence
    float beamAngle   = 0.f;
    float hazeDensity = 0.018f;// σ (1/m) for the deferred volumetric beams
    auto applyMode = [&] {
        if (night) {
            scene.background  = nightEnv;
            scene.environment = nightEnv;
            sun->intensity  = 0.f;
            moon->intensity = 0.30f;
            beam->intensity = 150000.f;// inverse-square: bright enough to read at 300 m
            lampMat->emissiveIntensity = 25.f;
            renderer.setDeferredVolumetrics(hazeDensity, 0.6f);
            renderer.setDeferredStarfield(1.0f);
            renderer.toneMappingExposure = 1.15f;
        } else {
            scene.background  = env;
            scene.environment = env;
            sun->intensity  = 2.0f;
            moon->intensity = 0.f;
            beam->intensity = 0.f;
            lampMat->emissiveIntensity = 0.f;
            renderer.setDeferredVolumetrics(0.f, 0.6f);
            renderer.setDeferredStarfield(0.f);
            renderer.toneMappingExposure = 0.7f;
        }
        // Material PBR values live in a GPU MaterialDesc refreshed on version
        // bump — emissiveIntensity is a plain field, so without this the
        // day↔night toggle leaves the GPU-side lamp at the OLD emissive:
        // no glow, no lamp area light, and (since strongly-emissive housings
        // are what shadow rays skip) the housing blocks the beam entirely.
        lampMat->needsUpdate();
    };
    applyMode();

    constexpr float kBoatLength = 28.0f;
    constexpr float kBoatBeam   = 9.0f;
    GLTFLoader gltfLoader;
    auto boat = loadAsync([&gltfLoader]() -> std::shared_ptr<Group> {
        auto gltf = gltfLoader.load(std::string(DATA_FOLDER) + "/models/gltf/Gunnerus.glb");
        if (!gltf || !gltf->scene) {
            std::cerr << "Failed to load Gunnerus.glb" << std::endl;
            auto fallback = Group::create();
            fallback->add(Mesh::create(BoxGeometry::create(kBoatBeam, 5.f, kBoatLength),
                                       MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color::red))));
            return fallback;
        }

        auto innerAsset = gltf->scene;
        {
            Box3 bbox;
            bbox.setFromObject(*innerAsset);
            Vector3 size; bbox.getSize(size);
            if (size.x > size.z) {
                innerAsset->rotateY(-math::PI / 2.f);
                bbox.setFromObject(*innerAsset);
                bbox.getSize(size);
            }
            innerAsset->rotateY(math::PI);
            const float maxExtent = std::max({size.x, size.y, size.z});
            if (maxExtent > 0.f) {
                const float s = kBoatLength / maxExtent;
                innerAsset->scale.set(s, s, s);
            }
        }

        // Material re-grade. The GLB exports EVERY material with the same
        // metallic-roughness pair (rough 0.71, metal 0.40) — half-metallic
        // everything is exactly the "grey plastic toy" look, and the PBR
        // ocean now exposes it hard. Painted marine steel is a DIELECTRIC
        // with a fairly glossy topcoat; deck paint is matte non-slip; only
        // railings/stairs/exhaust keep a metallic component; glass is
        // near-mirror. Albedo + normal maps stay — only the response
        // changes. Keyed by the GLB's material names (see git history for
        // the dumped inventory).
        struct Grade { float rough, metal; };
        static const std::unordered_map<std::string, Grade> grades = {
                {"HullWhite",            {0.38f, 0.00f}},// painted steel topsides
                {"VesselBlue",           {0.35f, 0.00f}},
                {"BridgeWhite",          {0.42f, 0.00f}},
                {"BridgeWhite.001",      {0.42f, 0.00f}},
                {"BridgeBlue",           {0.40f, 0.00f}},
                {"whitePaint_notBridge", {0.42f, 0.00f}},
                {"VesselBottom",         {0.55f, 0.15f}},// antifouling; bronze props share it
                {"Deck",                 {0.85f, 0.00f}},// matte non-slip paint
                {"Interior",             {0.80f, 0.00f}},
                {"Decals",               {0.55f, 0.00f}},
                {"Glass",                {0.06f, 0.00f}},// near-mirror, mostly sky reflection
                {"VesselGlass",          {0.06f, 0.00f}},
                {"Railing",              {0.45f, 0.35f}},// galvanized/painted metalwork
                {"stairs",               {0.50f, 0.35f}},
                {"Material #315",        {0.50f, 0.35f}},// stairs sibling
                {"Exhaust",              {0.60f, 0.30f}},
                {"VesselSafe",           {0.55f, 0.00f}},// life rings: actual plastic
                {"BridgeSafe",           {0.55f, 0.00f}},
                {"vesselShared",         {0.50f, 0.10f}},
        };
        innerAsset->traverseType<Mesh>([](Mesh& m) {
            auto* mat = m.material()->as<MeshStandardMaterial>();
            if (!mat) return;
            const auto it = grades.find(mat->name);
            // Unknown names: sane dielectric default, far better than the
            // export's uniform half-metal.
            const Grade g = (it != grades.end()) ? it->second : Grade{0.55f, 0.05f};
            mat->roughness = g.rough;
            mat->metalness = g.metal;
        });

        auto group = Group::create();
        group->add(innerAsset);
        return group;
    });
    scene.add(boat);
    BoatState bs;
    BoatInput bi;
    canvas.addKeyListener(bi);

    // ── Auto-pilot waypoints ───────────────────────────────────────────────
    // Closed Catmull-Rom loop the vessel follows when autopilot is on.
    // Five waypoints, one per buoy variant loaded from the GLB below.
    // Points are world-space (x, 0, z); y=0 — boat heave still comes from
    // wave sampling so it rides crests naturally along the path.
    // Centripetal parameterization avoids cusps/loops in non-uniform spacing;
    // closed=true makes the curve wrap so reaching u=1 seamlessly resets to
    // u=0 (continuous tangent at the seam).
    std::vector<Vector3> waypoints = {
            Vector3(-200.f, 0.f, -150.f),
            Vector3( 250.f, 0.f, -200.f),
            Vector3( 320.f, 0.f,  100.f),
            Vector3(   0.f, 0.f,  280.f),
            Vector3(-300.f, 0.f,   50.f),
    };
    CatmullRomCurve3 navCurve(waypoints, /*closed=*/true,
                              CatmullRomCurve3::centripetal, 0.5f);
    const float navCurveLength = navCurve.getLength();

    // ── Buoys at each waypoint ─────────────────────────────────────────────
    // Loaded from a GLB containing 5 distinct buoy meshes as top-level
    // children. We use the GLB's scene root as the buoy group directly
    // (Object3D doesn't expose the owning shared_ptr for children, so the
    // cleanest reparent is to keep them where they are and just reposition
    // each in place). Root transform is reset so the children's local
    // positions ARE their world positions.
    //
    // Fallback path: if the GLB is missing or doesn't expose ≥5 children,
    // emissive spheres take over. Path tracer's emissive NEE picks them up
    // (low intensity so they don't double as light sources).
    auto buoyFallbackMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0.95f, 0.25f, 0.1f)).emissive(Color(0.95f, 0.25f, 0.1f)).emissiveIntensity(0.6f).roughness(0.8f));
    auto makeFallbackBuoy = [&] {
        return Mesh::create(SphereGeometry::create(1.0f, 16, 12), buoyFallbackMat);
    };

    std::shared_ptr<Object3D> buoyRoot;
    try {
        auto buoyGltf = gltfLoader.load(std::string(DATA_FOLDER) +
                                        "/models/gltf/ocean_buoy/ocean_buoy_v.2_tao_buoy.glb");
        if (buoyGltf && buoyGltf->scene) {
            auto obj = buoyGltf->scene->getObjectByName("TAO_L.obj.cleaner.materialmerger.gles");
            buoyRoot = obj->clone();

        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to load buoy GLB: " << e.what() << std::endl;
    }

    // Per-buoy state captured across frames for buoyancy. Filled in the
    // GLB-success branch below; the animate loop sees an empty vector if
    // the fallback sphere path took over (and skips buoyancy then).
    struct BuoyState {
        float   worldX = 0.f, worldZ = 0.f;  // fixed waypoint position
        Vector3 baseLocal;                   // anchored "rest" local position
        float   worldYOffset = 0.f;          // current bob (world units)
        float   vY = 0.f;                    // bob velocity (world units / s)
    };
    std::vector<BuoyState> buoyStates;
    Matrix3 buoyInvRotScale;// captures buoyRoot's inverse rotation+scale for
                            // converting world-Y bob deltas back to local.

    if (buoyRoot && buoyRoot->children.size() >= waypoints.size()) {
        // GLB is Z-up at ~10× our scene scale; the rotation + uniform 0.1
        // scale lives on buoyRoot. Because of that parent transform, placing
        // each child at the waypoint requires going through invRootMat —
        // setting child.position directly in local space would put buoys at
        // (0.1·wp) in world coords, and a 15 m underwater spot in Y because
        // the X-axis rotation maps local Z to world Y.
        buoyRoot->position.set(0.f, 0.f, 0.f);
        buoyRoot->rotation.x = -math::PI / 2.f;
        buoyRoot->scale *= 0.1f;
        buoyRoot->updateMatrixWorld(true);
        Matrix4 invRootMat;
        invRootMat.copy(*buoyRoot->matrixWorld).invert();
        buoyInvRotScale.setFromMatrix4(invRootMat);

        // Per-buoy Y anchoring: GLB variants have different origin
        // conventions (one is centred way below its body, hence the
        // hovering buoy in the first screenshot). Computing each child's
        // world bbox and shifting so bbox.min().y sits at a uniform depth
        // below the waterline puts all five buoys at the same visual height.
        constexpr float kBuoyBottomBelowWater = 0.8f;
        buoyStates.resize(waypoints.size());
        for (size_t i = 0; i < waypoints.size(); ++i) {
            auto* child = buoyRoot->children[i];
            // Step 1: put the buoy at (waypoint.x, 0, waypoint.z) in world.
            Vector3 wantWorld(waypoints[i].x, 0.f, waypoints[i].z);
            Vector3 wantLocal = wantWorld;
            wantLocal.applyMatrix4(invRootMat);
            child->position.copy(wantLocal);
            child->updateMatrixWorld(true);

            // Step 2: compute world bbox, derive a vertical shift that
            // anchors the body's bottom at -kBuoyBottomBelowWater (slightly
            // submerged so the float ring intersects the chop naturally).
            Box3 bbox;
            bbox.setFromObject(*child);
            const float deltaWorldY = -kBuoyBottomBelowWater - bbox.min().y;
            Vector3 dLocal(0.f, deltaWorldY, 0.f);
            dLocal.applyMatrix3(buoyInvRotScale);
            child->position.add(dLocal);

            // Snapshot the "rest" local position so the per-frame buoyancy
            // step can offset cleanly without drifting.
            buoyStates[i].worldX = waypoints[i].x;
            buoyStates[i].worldZ = waypoints[i].z;
            buoyStates[i].baseLocal = child->position;
        }
        scene.add(buoyRoot);
    } else {
        if (buoyRoot) {
            std::cerr << "Buoy GLB has " << buoyRoot->children.size()
                      << " children (need " << waypoints.size() << "); using fallback spheres."
                      << std::endl;
        }
        auto buoyGroup = Group::create();
        for (const Vector3& wp : waypoints) {
            auto buoy = makeFallbackBuoy();
            buoy->position.set(wp.x, 1.5f, wp.z);
            buoyGroup->add(buoy);
        }
        scene.add(buoyGroup);
    }

    // Start the vessel at the first waypoint, heading along the initial tangent.
    bool  autoPilot      = true;
    float autoU          = 0.f;   // arc-length parameter in [0, 1)
    float autoCruiseSpeed = 6.f;  // m/s (~12 kn) — moderate cruise
    {
        Vector3 p0, t0;
        navCurve.getPointAt(0.f, p0);
        navCurve.getTangentAt(0.f, t0);
        bs.position.set(p0.x, 0.f, p0.z);
        bs.yaw          = std::atan2(t0.x, t0.z);
        bs.forwardSpeed = autoCruiseSpeed;
    }

    // Far-clip raised so the horizon doesn't get cut at the elevated/distant
    // camera. Initial position is offset from the boat; OrbitControls'
    // target is updated each frame to follow the boat so orbiting always
    // happens around the vessel.
    PerspectiveCamera camera(45.f, canvas.aspect(), 0.1f, 2000.f);
    camera.position.set(40.f, 18.f, 60.f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, 1.f, 0.f);
    controls.update();

    // ── Camera modes ──────────────────────────────────────────────────────
    // C cycles: Free orbit → Side → Chase → Underwater → NearestBuoy.
    // SideFixed/Chase/Underwater/NearestBuoy override camera.position +
    // lookAt each frame; OrbitControls state is left alone so switching
    // back to Free restores whatever orbit angle the user left it at.
    enum class CamMode : int {
        Free        = 0,
        SideFixed   = 1,
        Chase       = 2,
        Underwater  = 3,
        NearestBuoy = 4,
    };
    constexpr int kCamModeCount = 5;
    CamMode camMode = CamMode::Free;
    const char* camModeNames[] = {
            "Free (orbit)", "Side (forward)", "Chase", "Underwater", "Nearest buoy"};

    // Auto-cam: cycles through every CamMode in turn at a fixed cadence.
    // Lets the demo show all viewpoints without the user touching anything.
    // Free is skipped while auto is on so OrbitControls input doesn't fight
    // the timer.
    bool  autoCam        = false;
    float autoCamHold    = 5.0f;  // seconds per camera before switching
    float autoCamElapsed = 0.f;

    KeyAdapter camKey(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent ev) {
        if (ev.key == Key::C) {
            camMode = static_cast<CamMode>(
                    (static_cast<int>(camMode) + 1) % kCamModeCount);
            autoCamElapsed = 0.f;  // a manual switch restarts the auto timer
        }
    });
    canvas.addKeyListener(camKey);

    // ── Sound: engine + ocean/wind ambience ─────────────────────────────────
    // Skipped in --shot capture runs (headless iteration shouldn't bleep).
    OceanSounds sounds;
    if (!capturing) sounds.init();
    bool  audioOn  = true;
    float audioVol = 0.3f;

    float waveScale = ocean->params.waveScale;
    float choppiness = ocean->params.choppiness;
    float windSpeed = ocean->params.windSpeed;
    float windTheta = ocean->params.windTheta;
    float exposure  = renderer.toneMappingExposure;
    int   toneMode  = static_cast<int>(renderer.toneMapping);
    int   spp       = renderer.samplesPerPixel();
    float renderScale = renderer.renderScale();
    float fps = 0.f, fpsAccum = 0.f;
    int   fpsFrames = 0;

    // ── Primary-trace cost measurement ─────────────────────────────────
    // Toggle `setMeasurePrimaryTraceOnly` and watch the delta on
    // pathTraceMs. EMA-smoothed so the readouts don't dance frame-to-frame;
    // both numbers persist across toggle changes so the comparison stays
    // visible after flipping back to "full".
    bool measurePrimaryOnly = renderer.measurePrimaryTraceOnly();
    float fullPtMs = 0.f;
    float primaryOnlyMs = 0.f;
    constexpr float ptEmaAlpha = 0.10f;

    // ── Underwater fog parameters ─────────────────────────────────────────
    // Controlled by ImGui; applied per-frame when the camera is submerged.
    // fogDensity = σ_t (extinction per metre, uniform across channels).
    // fogColor = single-scattering albedo σ_s/σ_t — blue-green for ocean.
    // fogAnisotropy = Henyey-Greenstein g: 0.85 → strong forward scatter
    //   (sun shafts / god rays when looking toward the light).
    float uwFogDensity   = 0.06f;
    float uwFogColor[3]  = {0.10f, 0.55f, 0.65f};
    float uwFogAniso     = 0.85f;
    float uwDepthSmooth  = 0.f;

    // ── Radar state ────────────────────────────────────────────────────────
    // Heading-up scope, 500 m range, 4 s sweep period (15 RPM — realistic
    // small-craft). Per-target intensity is set to 1 when the sweep crosses
    // its bearing, then decays exponentially (3 s time constant ≈ slightly
    // less than one full rotation). Cap intensity by range so distant blips
    // look weaker than nearby ones — closer to a real CRT.
    bool  radarOn          = true;
    float radarRangeM      = 500.f;
    float radarSweepAngle  = 0.f;        // [0, 2π), advances clockwise
    constexpr float kRadarSweepPeriod = 4.f;
    constexpr float kRadarBlipTau     = 3.f;
    constexpr int   kMaxRadarTargets  = 16;
    std::array<float, kMaxRadarTargets> radarBlipIntensity{};

    // ── LIDAR mast (mounted above the Gunnerus bow, forward-facing 180°) ──
    // OS0-128 beam pattern restricted to the forward hemisphere — matches
    // typical maritime "bow-mounted" sensor coverage, and avoids the
    // beams that would otherwise have to crawl through the wheelhouse and
    // hull bbox behind the sensor origin. Intensity is physically correct
    // via the same closest-hit BSDF the path tracer uses.
    constexpr float kLidarMountHeight  = 11.0f;      // m above waterline — clears wheelhouse + mast roof
    constexpr float kLidarMountForward = 8.0f;       // m forward of boat origin (near the bow)
    constexpr int   kLidarMaxBeams     = 1100000;    // OS0-128 forward × maxReturns 4 × samples 4

    // Forward-only scan (azimuth ∈ [-90°, +90°]). Sensor convention puts
    // azimuth = 0 along the sensor's local -Z (forward), so this carves
    // out the rear half-sphere.
    LidarModel lidarModel = LidarModel::OS0_128();
    lidarModel.azimuthMin = -90.f;
    lidarModel.azimuthMax =  90.f;

    auto lidarSensor = std::make_unique<PathTracedLidarSensor>(lidarModel);
    lidarSensor->params.maxRange = 120.f;            // ~half a buoy spacing
    lidarSensor->params.referenceRange = 30.f;       // calibrate intensity for ocean scale
    lidarSensor->params.laserPower = 2.0f;           // bump so distant buoys still register
    lidarSensor->params.atmosphericExtinction = 0.f;
    lidarSensor->params.detectorThreshold = 0.005f;

    // Dedicated LIDAR water column. Attenuates beams crossing into the
    // ocean regardless of where the camera is, so the seafloor doesn't
    // pop in unattenuated when the eye floats above the waterline.
    // mediumSurfaceY is updated each frame to follow the live wave
    // surface height under the sensor.
    lidarSensor->params.mediumExtinction = 0.30f;
    lidarSensor->params.mediumAlbedo     = 0.85f;
    lidarSensor->params.mediumAnisotropy = 0.85f;
    lidarSensor->params.mediumSurfaceY   = 0.f;
    scene.addRef(*lidarSensor);
    bool lidarEnabled = false;
    bool lidarShowPanel = true;

    // Live stats reflected in the UI panel.
    int   lidarLastReturns = 0;
    float lidarLastScanMs  = 0.f;

    // ── LIDAR point-cloud overlay ──────────────────────────────────────────
    // One vertex per return, intensity-mapped colour. Vulkan PT auto-detects
    // Points objects and routes them through the POINT_LIST overlay pipeline,
    // excluded from the TLAS so beams don't see their own visualisation.
    auto lidarCloudGeom = BufferGeometry::create();
    lidarCloudGeom->setAttribute("position",
                                  FloatBufferAttribute::create(std::vector<float>(kLidarMaxBeams * 3), 3));
    lidarCloudGeom->setAttribute("color",
                                  FloatBufferAttribute::create(std::vector<float>(kLidarMaxBeams * 3), 3));
    lidarCloudGeom->getAttribute<float>("position")->setUsage(DrawUsage::Dynamic);
    lidarCloudGeom->getAttribute<float>("color")->setUsage(DrawUsage::Dynamic);
    lidarCloudGeom->setDrawRange(0, 0);

    auto lidarCloudMat = PointsMaterial::create(PointsMaterial::Params{}.size(3.f).vertexColors(true));
    auto lidarCloud = Points::create(lidarCloudGeom, lidarCloudMat);
    lidarCloud->frustumCulled = false;
    scene.add(lidarCloud);

    // ── LIDAR readout panel (screen-space sprite, top-left below FPS) ──────
    // 720×128 RGBA8 grid laid out as (azimuth × elevation); same colormap as
    // the cloud. We tag it sRGB so the sampler decodes correctly and the
    // bytes we write display vibrantly post-pipeline.
    constexpr unsigned int kLidarPanelW = 720;
    constexpr unsigned int kLidarPanelH = 128;
    constexpr float        kLidarPanelDispW = 360.f;
    constexpr float        kLidarPanelDispH = 64.f;

    auto lidarPanelTex = DataTexture::create(
            ImageData{std::vector<unsigned char>(kLidarPanelW * kLidarPanelH * 4, 0u)},
            kLidarPanelW, kLidarPanelH);
    lidarPanelTex->colorSpace = ColorSpace::sRGB;

    auto lidarPanelMat = SpriteMaterial::create();
    lidarPanelMat->map = lidarPanelTex;
    auto lidarPanel = Sprite::create(lidarPanelMat);
    lidarPanel->scale.set(kLidarPanelDispW, kLidarPanelDispH, 1.f);
    lidarPanel->screenSpace = true;
    lidarPanel->screenAnchor.set(1.f, 1.f);   // viewport top-right
    lidarPanel->center.set(1.f, 1.f);
    lidarPanel->position.set(-10.f, -10.f, 0.f);
    scene.add(lidarPanel);

    std::vector<LidarReturn> lidarReturns;

    // ── Time-of-flight waveform synthesis ──────────────────────────────
    // The discrete returns above are what the host gets from the GPU
    // pipeline; a real LIDAR detector sees a continuous waveform that's
    // the Gaussian-pulse convolution of those returns. We synthesise
    // that waveform CPU-side and plot a single representative beam in
    // ImGui so the demo shows both views (point cloud + analog signal).
    LidarWaveformParams lidarWavParams;
    lidarWavParams.maxRange   = 120.f;
    lidarWavParams.bins       = 256;
    lidarWavParams.pulseFWHM  = 0.75f;
    lidarWavParams.noiseFloor = 0.f;
    bool  lidarShowWave    = true;
    int   lidarWaveBeamIdx = 0;
    std::vector<float> lidarWaveform;

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({340, 0});
        ImGui::Begin("Vulkan PT - Ocean");
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Separator();
        ImGui::TextWrapped(
            "FFT-displaced surface (multi-cascade Phillips, %u² IFFT, "
            "%.0f m tile). Path-traced refraction + photon-map caustics. "
            "WASD = steer the Gunnerus.",
            kFftSize, kTileSize);
        ImGui::Text("Speed: %.1f m/s   Heading: %.0f°", bs.forwardSpeed,
                    bs.yaw * 180.f / 3.14159f);
        ImGui::Text("Pos: %7.1f, %7.1f", bs.position.x, bs.position.z);
        ImGui::Text("Keys  W:%d  A:%d  S:%d  D:%d   (C = cycle camera)",
                    bi.W ? 1 : 0, bi.A ? 1 : 0, bi.S ? 1 : 0, bi.D ? 1 : 0);
        ImGui::Text("Camera: %s", camModeNames[static_cast<int>(camMode)]);
        if (ImGui::Checkbox("Auto-cam (cycle every N s)", &autoCam))
            autoCamElapsed = 0.f;
        if (autoCam) {
            ImGui::SliderFloat("Hold (s)", &autoCamHold, 1.f, 30.f, "%.1f");
            ImGui::Text("Next in %.1fs", std::max(0.f, autoCamHold - autoCamElapsed));
        }
        ImGui::Separator();

        ImGui::TextUnformatted("Autopilot");
        ImGui::Checkbox("Follow waypoint loop", &autoPilot);
        if (autoPilot) {
            ImGui::SliderFloat("Cruise speed (m/s)", &autoCruiseSpeed, 1.f, 12.f, "%.1f");
            const int wpCount = static_cast<int>(waypoints.size());
            const int currentWp = std::min(wpCount - 1,
                                           static_cast<int>(autoU * wpCount));
            ImGui::Text("Waypoint %d / %d   (u = %.2f)", currentWp + 1, wpCount, autoU);
        } else {
            ImGui::TextDisabled("WASD to steer manually.");
        }
        ImGui::Separator();

        if (ImGui::SliderFloat("Wave scale", &waveScale, 0.f, 3.f, "%.2f")) {
            ocean->params.waveScale = waveScale;
        }
        if (ImGui::SliderFloat("Choppiness", &choppiness, 0.f, 1.0f, "%.2f")) {
            ocean->params.choppiness = choppiness;
        }

        ImGui::TextDisabled("Wind changes apply on scene reload.");
        ImGui::Separator();
        ImGui::TextUnformatted("Night & lighthouse");
        if (ImGui::Checkbox("Night mode", &night)) {
            applyMode();
            exposure = renderer.toneMappingExposure;
        }
        if (night) {
            ImGui::SliderFloat("Beam speed (rad/s)", &beamSpeed, 0.f, 2.f, "%.2f");
            if (ImGui::SliderFloat("Haze density (1/m)", &hazeDensity, 0.f, 0.08f, "%.3f")) {
                renderer.setDeferredVolumetrics(hazeDensity, 0.6f);
            }
        }
        ImGui::Separator();
        ImGui::TextUnformatted("Audio");
        if (sounds.ok) {
            ImGui::Checkbox("Enable##audio", &audioOn);
            ImGui::SliderFloat("Volume##audio", &audioVol, 0.f, 1.f, "%.2f");
        } else {
            ImGui::TextDisabled("Audio unavailable.");
        }
        ImGui::Separator();
        if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 5.0f, "%.2f"))
            renderer.toneMappingExposure = exposure;
        const char* toneItems[] = {"None", "Linear", "Reinhard", "Cineon", "ACESFilmic"};
        if (ImGui::Combo("Tone mapping", &toneMode, toneItems, IM_ARRAYSIZE(toneItems))) {
            renderer.toneMapping = static_cast<ToneMapping>(toneMode);
        }
        bool restirDI = renderer.restirDIEnabled();
        if (ImGui::Checkbox("ReSTIR DI", &restirDI)) {
            renderer.setRestirDIEnabled(restirDI);
        }
        bool restirGI = renderer.restirGIEnabled();
        if (ImGui::Checkbox("ReSTIR GI", &restirGI)) {
            renderer.setRestirGIEnabled(restirGI);
        }
        if (ImGui::SliderInt("Samples / pixel", &spp, 1, 16)) {
            renderer.setSamplesPerPixel(spp);
        }
        // Silhouette MSAA: extra primary rays at edge pixels only.
        // 0 disables; default 7 → 8× MSAA at edges.
        int edgeMsaa = static_cast<int>(renderer.silhouetteMsaaExtra());
        if (ImGui::SliderInt("Silhouette MSAA extras", &edgeMsaa, 0, 15)) {
            renderer.setSilhouetteMsaaExtra(static_cast<uint32_t>(edgeMsaa));
        }
        // Path-trace render scale: < 1 traces fewer pixels, then upscales.
        if (ImGui::SliderFloat("Render scale", &renderScale, 0.25f, 1.0f, "%.2f")) {
            renderer.setRenderScale(renderScale);
        }
        ImGui::Separator();


        ImGui::TextUnformatted("Underwater fog");
        ImGui::SliderFloat("Density (1/m)", &uwFogDensity, 0.01f, 0.20f, "%.3f");
        ImGui::ColorEdit3("Inscatter tint", uwFogColor,
                          ImGuiColorEditFlags_Float | ImGuiColorEditFlags_NoInputs);
        ImGui::SliderFloat("Anisotropy (g)", &uwFogAniso, -0.95f, 0.95f, "%.2f");
        if (uwDepthSmooth > 0.001f) {
            ImGui::Text("Submerged: %.1f%%", uwDepthSmooth * 100.f);
        } else {
            ImGui::TextDisabled("Camera above water.");
        }

        ImGui::Separator();
        if (ImGui::CollapsingHeader("LIDAR mast (OS0-128, path-traced)")) {
            ImGui::Checkbox("Enable LIDAR",    &lidarEnabled);
            ImGui::SameLine();
            ImGui::Checkbox("Range panel",     &lidarShowPanel);
            ImGui::SliderFloat("Max range (m)##lidar",      &lidarSensor->params.maxRange, 10.f, 250.f);
            ImGui::SliderFloat("Reference range (m)##lidar",&lidarSensor->params.referenceRange, 1.f, 80.f);
            ImGui::SliderFloat("Laser power##lidar",        &lidarSensor->params.laserPower, 0.1f, 10.f);
            ImGui::SliderFloat("Atmospheric ext##lidar",    &lidarSensor->params.atmosphericExtinction, 0.f, 0.05f, "%.4f");
            ImGui::SliderFloat("Detector thresh##lidar",    &lidarSensor->params.detectorThreshold, 0.f, 0.02f, "%.4f");
            {
                // Multi-return through the water surface: maxReturns=2 lets the
                // sensor see the seafloor below the wave crests where the
                // surface Fresnel return doesn't fully attenuate the beam.
                int maxRet = static_cast<int>(lidarSensor->params.maxReturns);
                if (ImGui::SliderInt("Max returns##lidar", &maxRet, 1, 4)) {
                    lidarSensor->params.maxReturns = static_cast<uint32_t>(std::max(1, maxRet));
                }
            }
            {
                // Monte Carlo samples per beam: jitters direction within the
                // beam-divergence cone and uses a fresh RNG stream per sample
                // for the fog scatter. >1 averages volumetric noise and shows
                // the beam's true angular footprint.
                int samples = static_cast<int>(lidarSensor->params.samplesPerBeam);
                if (ImGui::SliderInt("Samples / beam##lidar", &samples, 1, 8))
                    lidarSensor->params.samplesPerBeam = static_cast<uint32_t>(std::max(1, samples));
            }
            ImGui::SliderFloat("Beam divergence (mrad)##lidar",
                               &lidarSensor->params.beamDivergenceMrad, 0.f, 10.f, "%.2f");
            ImGui::TextDisabled("Water column (dedicated LIDAR medium)");
            ImGui::SliderFloat("Extinction (1/m)##wat",
                               &lidarSensor->params.mediumExtinction, 0.f, 1.f, "%.3f");
            ImGui::SliderFloat("Albedo##wat",
                               &lidarSensor->params.mediumAlbedo, 0.f, 1.f, "%.2f");
            ImGui::SliderFloat("Anisotropy g##wat",
                               &lidarSensor->params.mediumAnisotropy, -0.95f, 0.95f, "%.2f");
            ImGui::Text("Returns: %d / %u beams   Scan: %.2f ms",
                        lidarLastReturns,
                        lidarSensor->beamCount(),
                        static_cast<double>(lidarLastScanMs));

            ImGui::Separator();
            ImGui::Checkbox("Show ToF waveform", &lidarShowWave);
            if (lidarShowWave) {
                ImGui::SliderFloat("Pulse FWHM (m)##lidar", &lidarWavParams.pulseFWHM, 0.1f, 5.f, "%.2f");
                const int maxBeam = std::max(0, static_cast<int>(lidarSensor->beamCount()) - 1);
                ImGui::SliderInt("Beam##wave", &lidarWaveBeamIdx, 0, maxBeam);
                // Plot the selected beam's analog waveform.
                const size_t bins = lidarWavParams.bins;
                if (!lidarWaveform.empty()) {
                    const size_t off = static_cast<size_t>(lidarWaveBeamIdx) * bins;
                    if (off + bins <= lidarWaveform.size()) {
                        ImGui::PlotLines("Intensity",
                                         &lidarWaveform[off],
                                         static_cast<int>(bins),
                                         0,
                                         nullptr,
                                         0.f, 1.f,
                                         ImVec2(0, 80));
                        ImGui::TextDisabled("X = distance 0 .. %.1f m", static_cast<double>(lidarWavParams.maxRange));
                    }
                }
            }
        }
        ImGui::End();

        // ── Radar HUD ─────────────────────────────────────────────────────
        // Separate floating window, top-right of the viewport. Heading-up
        // CRT-green scope drawn entirely with ImDrawList primitives over a
        // dark phosphor disc. Range slider lives inside the radar window.
        if (radarOn) {
            const ImGuiViewport* vp = ImGui::GetMainViewport();
            const float radarSize = 260.f;
            ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + vp->WorkSize.x - radarSize - 16.f,
                                           vp->WorkPos.y + 16.f));
            ImGui::SetNextWindowSize(ImVec2(radarSize, 0.f));
            ImGui::Begin("Radar", &radarOn,
                         ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImVec2 cursor = ImGui::GetCursorScreenPos();
            const float scopeDim = radarSize - 16.f;
            const ImVec2 center(cursor.x + scopeDim * 0.5f,
                                cursor.y + scopeDim * 0.5f);
            const float R = scopeDim * 0.5f - 4.f;

            // Bearing → scope screen position. Negated sin keeps the radar's
            // left/right in sync with the chase/side camera (this scene uses
            // left-handed-style boatRight so world +X is on the LEFT of any
            // forward-looking view).
            auto scopePos = [&](float angle, float radius) {
                return ImVec2(center.x - std::sin(angle) * radius,
                              center.y - std::cos(angle) * radius);
            };

            // Background disc (dark CRT phosphor)
            dl->AddCircleFilled(center, R, IM_COL32(8, 24, 8, 235), 64);
            // Range rings — 4 concentric, at 25/50/75/100% of range
            for (int k = 1; k <= 4; ++k) {
                const float r = R * (float)k / 4.f;
                dl->AddCircle(center, r, IM_COL32(0, 130, 0, 200), 64, 1.0f);
            }
            // Cross-hairs (vertical = vessel forward; horizontal = abeam)
            dl->AddLine(ImVec2(center.x, center.y - R), ImVec2(center.x, center.y + R),
                        IM_COL32(0, 90, 0, 160), 1.0f);
            dl->AddLine(ImVec2(center.x - R, center.y), ImVec2(center.x + R, center.y),
                        IM_COL32(0, 90, 0, 160), 1.0f);

            // Sweep wedge: 16 segments, each step wide arc, fading toward
            // the trail end. Correct math: segment k spans [sweep - k·step,
            // sweep - (k+1)·step] where step = trailArc / trailSteps. The
            // earlier double-divide bug packed every triangle on top of the
            // leading edge — fixed now.
            constexpr int   trailSteps = 16;
            constexpr float trailArc   = 1.4f;
            constexpr float step       = trailArc / trailSteps;
            for (int k = 0; k < trailSteps; ++k) {
                const float a0 = radarSweepAngle - (float)k       * step;
                const float a1 = radarSweepAngle - (float)(k + 1) * step;
                const int alpha = static_cast<int>(140.f * (1.f - (float)k / trailSteps));
                dl->AddTriangleFilled(center, scopePos(a0, R), scopePos(a1, R),
                                      IM_COL32(40, 220, 40, alpha));
            }
            // Sweep leading edge — bright line
            dl->AddLine(center, scopePos(radarSweepAngle, R),
                        IM_COL32(140, 255, 140, 255), 1.6f);

            // Blips — halo glow + bright core for the CRT look.
            for (size_t i = 0; i < buoyStates.size() && i < kMaxRadarTargets; ++i) {
                const float a = radarBlipIntensity[i];
                if (a <= 0.01f) continue;
                const float dxw = buoyStates[i].worldX - bs.position.x;
                const float dzw = buoyStates[i].worldZ - bs.position.z;
                const float dist = std::sqrt(dxw * dxw + dzw * dzw);
                if (dist > radarRangeM) continue;
                const float bearing = std::atan2(dxw, dzw) - bs.yaw;
                const float r       = (dist / radarRangeM) * R;
                const ImVec2 pos    = scopePos(bearing, r);
                dl->AddCircleFilled(pos, 7.f,
                                    IM_COL32(120, 255, 120, (int)(90.f * a)), 14);
                dl->AddCircleFilled(pos, 3.5f,
                                    IM_COL32(200, 255, 200, (int)(255.f * a)), 12);
            }

            // Own-ship marker (triangle pointing forward = up)
            const float ts = 7.f;
            dl->AddTriangleFilled(
                    ImVec2(center.x,             center.y - ts),
                    ImVec2(center.x - ts * 0.6f, center.y + ts * 0.55f),
                    ImVec2(center.x + ts * 0.6f, center.y + ts * 0.55f),
                    IM_COL32(220, 255, 220, 255));

            // Reserve the drawn area + readouts below.
            ImGui::Dummy(ImVec2(scopeDim, scopeDim));
            ImGui::Text("Range: %.0f m   Heading: %.0f°",
                        radarRangeM, bs.yaw * 180.f / 3.14159f);
            ImGui::SliderFloat("##rng", &radarRangeM, 100.f, 1000.f, "Range %.0f m");
            ImGui::End();
        }
    });

    IOCapture ioCapture;
    ioCapture.preventMouseEvent    = []() -> bool { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture.preventScrollEvent   = []() -> bool { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture.preventKeyboardEvent = []() -> bool { return ImGui::GetIO().WantCaptureKeyboard; };
    canvas.setIOCapture(&ioCapture);

    canvas.onWindowResize([&](const WindowSize& ns) {
        renderer.setSize(ns);
        camera.aspect = canvas.aspect();
        camera.updateProjectionMatrix();
    });

    Clock clock;
    canvas.animate([&] {
        const float dt = std::min(clock.getDelta(), 0.1f);  // clamp dt — pause / breakpoints shouldn't teleport the boat
        fpsAccum += dt;
        ++fpsFrames;
        if (fpsAccum >= 0.5f) {
            fps = fpsFrames / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        // === Boat steering integration ===
        // Autopilot path: advance arc-length parameter u by cruise speed × dt
        // along the closed nav curve and lift position + heading directly
        // from the curve. Yaw aligns with the tangent so the boat always
        // faces "forward along the path". forwardSpeed is set to the cruise
        // speed so the wake reads correctly. closed=true on the curve means
        // u wraps seamlessly — fmod handles the [0, 1) wrap, and the loop
        // restarts without a heading jump.
        float cosY, sinY;
        if (autoPilot && navCurveLength > 1e-3f) {
            autoU += (autoCruiseSpeed * dt) / navCurveLength;
            autoU = std::fmod(autoU, 1.f);
            if (autoU < 0.f) autoU += 1.f;
            Vector3 p, tan;
            navCurve.getPointAt(autoU, p);
            navCurve.getTangentAt(autoU, tan);
            bs.position.set(p.x, 0.f, p.z);
            bs.yaw          = std::atan2(tan.x, tan.z);
            bs.forwardSpeed = autoCruiseSpeed;
            cosY = std::cos(bs.yaw);
            sinY = std::sin(bs.yaw);
        } else {
            // Manual: W = forward thrust, S = reverse. Speed clamped to ±vMax;
            // a linear drag (0.7/s) gives a coast-down once thrust released.
            const float thrust = (bi.W ? 6.0f : 0.f) - (bi.S ? 4.0f : 0.f);
            bs.forwardSpeed += thrust * dt;
            bs.forwardSpeed -= bs.forwardSpeed * 0.5f * dt;
            bs.forwardSpeed = std::clamp(bs.forwardSpeed, -4.f, 8.f);
            // Yaw: A = left, D = right. Rate scales with speed (a stationary
            // hull doesn't yaw easily). Min factor 0.1 keeps the rudder usable
            // when nearly stopped.
            const float yawInput = (bi.A ? 1.0f : 0.f) - (bi.D ? 1.0f : 0.f);
            const float speedFactor = std::clamp(std::abs(bs.forwardSpeed) / 5.f, 0.1f, 1.0f);
            bs.yaw += yawInput * 0.6f * speedFactor * dt;
            // Position: boat moves along its heading. Convention: yaw=0 → +Z,
            // matching the OrbitControls default forward.
            cosY = std::cos(bs.yaw);
            sinY = std::sin(bs.yaw);
            bs.position.x += sinY * bs.forwardSpeed * dt;
            bs.position.z += cosY * bs.forwardSpeed * dt;
        }

        // === Hydrodynamics from sampled wave surface ===
        // Sample 5 points: centre + four corners of the bounding rectangle.
        // Heave (Y) follows the centre; pitch comes from fore/aft height
        // diff over hull length; roll from port/starboard diff over beam.
        //
        // Cascade mask 0b011 = swells + mid-frequency only; skip cascade 2
        // (4-8 m chop). A 28 m hull doesn't physically pitch on wavelengths
        // shorter than itself — those waves wash under harmlessly. Reading
        // them at bow and stern gave the boat a "car on rocks" feel.
        // Visuals are unaffected — the GPU still renders all three cascades.
        constexpr uint32_t kBuoyancyMask = 0b011u;
        const float halfL = kBoatLength * 0.5f;
        const float halfB = kBoatBeam   * 0.5f;
        auto sampleH = [&](float dx, float dz) {
            // Local (forward, right) → world via yaw rotation
            const float wx = bs.position.x + sinY * dz + cosY * dx;
            const float wz = bs.position.z + cosY * dz - sinY * dx;
            return ocean->sampleHeight(wx, wz, kBuoyancyMask);
        };
        const float hCentre = sampleH(0.f,    0.f);
        const float hBow    = sampleH(0.f,  +halfL);
        const float hStern  = sampleH(0.f,  -halfL);
        const float hPort   = sampleH(-halfB, 0.f);
        const float hStbd   = sampleH(+halfB, 0.f);
        // Positive pitch = bow up (right-hand rotation around local X).
        const float pitch = std::atan2(hBow  - hStern, kBoatLength);
        // Positive roll = starboard down (right-hand rotation around local Z,
        // which under +Z forward convention means starboard side dips).
        const float roll  = std::atan2(hStbd - hPort,  kBoatBeam);

        // Pitch / roll: temporal low-pass at ~1 Hz so attitude rides the
        // swells but doesn't snap to wave-pumping at hull-traversal rates.
        // alpha = 1 − exp(−2π · cutoff · dt). With cascade 2 masked out
        // the input is already smoother; 1 Hz cutoff (was 3 Hz) is the
        // additional damping that finishes off the rock-terrain feel.
        const float alpha = 1.f - std::exp(-2.f * 3.14159f * 1.f * dt);
        bs.smoothPitch += (pitch - bs.smoothPitch) * alpha;
        bs.smoothRoll  += (roll  - bs.smoothRoll)  * alpha;

        // Heave: spring-damped follower instead of a 1:1 height tracker.
        // A real hull's vertical motion is bounded by buoyancy + mass +
        // hydrodynamic drag; tracking the wave surface exactly looked like
        // a yo-yo riding the crests. Tuning: ω ≈ 0.8 Hz natural frequency
        // (k = (2π·f)²), damping ratio ζ ≈ 0.7 (slightly under-damped → one
        // gentle overshoot then settles). Yields a believable "settling on
        // a wave" rhythm where the boat lags the surface a beat.
        const float omega = 2.f * 3.14159f * 0.8f;
        const float k     = omega * omega;
        const float c     = 2.f * 0.7f * omega;
        const float yErr  = hCentre - bs.y;
        const float accel = yErr * k - bs.vY * c;
        bs.vY += accel * dt;
        bs.y  += bs.vY * dt;

        // Apply transforms. Euler order YXZ — yaw first, then pitch about
        // the post-yaw local X, then roll about the post-yaw-pitch local Z;
        // i.e. the standard ship-attitude convention.
        // Pitch is NEGATED: in three.js, positive Euler.x rotation around
        // local +X tilts +Z forward toward −Y (bow DOWN) by right-hand
        // rule. Our `pitch` is positive when bow is on a crest, so we want
        // bow UP — flip the sign here.
        // The +0.5 m bias raises the hull origin above the waterline so
        // the deck reads above the wave surface (Gunnerus's bbox centres
        // around the waterline).
        boat->position.set(bs.position.x, bs.y - 0.2f, bs.position.z);
        boat->rotation.set(-bs.smoothPitch, bs.yaw, bs.smoothRoll, Euler::YXZ);

        // LIDAR mast: tracks the boat with a vertical offset for wheelhouse
        // clearance + a forward offset so the sensor sits near the bow
        // (otherwise the wheelhouse occludes the rear half of the scan and
        // the forward 180° starts blocked by the hull's superstructure).
        // Inherits pitch + yaw so the beam pattern sweeps with the hull;
        // roll is intentionally damped so the range panel stays readable
        // when the boat heels.
        //
        // The +π yaw offset reconciles two competing forward conventions:
        // the boat moves along (sinY, 0, +cosY) at yaw=0, but the
        // LidarModel puts azimuth=0 along sensor-local -Z so its forward
        // is (-sinY, 0, -cosY) before the offset. Rotating by π makes the
        // sensor look in the direction the boat is heading.
        lidarSensor->position.set(bs.position.x + sinY * kLidarMountForward,
                                  bs.y + kLidarMountHeight,
                                  bs.position.z + cosY * kLidarMountForward);
        lidarSensor->rotation.set(-bs.smoothPitch * 0.5f,
                                  bs.yaw + math::PI,
                                  0.f, Euler::YXZ);

        // Track the live wave height under the boat so the LIDAR water
        // column clamp matches the actual sea surface. Beams crossing
        // this Y get water-column extinction; segments above stay in
        // air and pass through unattenuated.
        lidarSensor->params.mediumSurfaceY = ocean->sampleHeight(bs.position.x, bs.position.z);
        lidarCloud->visible = lidarEnabled;
        lidarPanel->visible = lidarEnabled && lidarShowPanel;

        ocean->hullExclusion.centerX    = bs.position.x;
        ocean->hullExclusion.centerZ    = bs.position.z;
        ocean->hullExclusion.halfLength = kBoatLength * 0.5f;
        ocean->hullExclusion.halfBeam   = kBoatBeam * 0.5f;
        ocean->hullExclusion.sinYaw     = sinY;
        ocean->hullExclusion.cosYaw     = cosY;

        // Adaptive vertex density follows the vessel. The 1 km × 512² grid
        // is uniform-natural at ~1.96 m spacing; the cubic warp packs it
        // toward the boat to roughly 0.15 m at the centre while edge
        // spacing only grows to ~2.7 m. Vertex history (foam) is in a
        // world-space texture now, so vertices reflowing per frame is
        // realism-safe — wake trails stay where they were deposited.
        ocean->warp.centerX   = bs.position.x;
        ocean->warp.centerZ   = bs.position.z;
        ocean->warp.halfRange = kPlaneEdge * 0.5f;
        ocean->warp.coefA     = 0.1f;

        // Vessel wake — analytical foam trail + bow bump + Kelvin V-wake
        // injected in water_displace.comp from the same pose plus speed.
        // The Kelvin V-wake additionally uses a historical sample trail
        // so it traces the boat's actual sailed curve through turns,
        // instead of snapping to the current heading every frame. Trail
        // is maintained below: emit at 4 Hz, drop samples older than 10 s,
        // cap at the renderer's kMaxWakeSamples (32) — that's ~8 s of
        // path at 4 m/s ≈ 32 m of wake trail, comfortably longer than
        // the Kelvin 5·λ visible decay (~50 m at v=4 m/s).
        ocean->wake.forwardSpeed = bs.forwardSpeed;
        {
            constexpr float kEmitInterval     = 0.10f;     // 10 Hz cadence
            constexpr float kEmitDistance     = 1.0f;      // also emit on > 1 m travel
            constexpr float kMaxAge           = 6.0f;      // seconds
            constexpr size_t kMaxSamples      = 64;
            static float emitAccum = 0.f;
            static float lastEmitX = std::numeric_limits<float>::quiet_NaN();
            static float lastEmitZ = std::numeric_limits<float>::quiet_NaN();

            // Age all samples and drop expired ones.
            auto& trail = ocean->wake.trail;
            for (auto& s : trail) s.age += dt;
            trail.erase(std::remove_if(trail.begin(), trail.end(),
                                       [](const DisplacedMesh::WakeSample& s){
                                           return s.age > kMaxAge;
                                       }),
                        trail.end());

            // Emit a new sample on whichever fires first: cadence elapsed
            // OR the boat has travelled > kEmitDistance since the last
            // emission. The distance trigger gives an immediate sample
            // when the boat speeds up from rest (no 100 ms gap of missing
            // wake while emitAccum builds), and the cadence keeps the
            // trail well-sampled during steady-state cruising.
            emitAccum += dt;
            const float spd = std::abs(bs.forwardSpeed);
            const float dxLast = std::isnan(lastEmitX) ? 1e9f : (bs.position.x - lastEmitX);
            const float dzLast = std::isnan(lastEmitZ) ? 1e9f : (bs.position.z - lastEmitZ);
            const float distLast = std::sqrt(dxLast * dxLast + dzLast * dzLast);
            const bool shouldEmit = spd > 0.3f &&
                                    (emitAccum >= kEmitInterval ||
                                     distLast >= kEmitDistance);
            if (shouldEmit) {
                emitAccum = 0.f;
                lastEmitX = bs.position.x;
                lastEmitZ = bs.position.z;
                if (trail.size() >= kMaxSamples) {
                    trail.erase(trail.begin());// drop oldest
                }
                DisplacedMesh::WakeSample s{};
                s.worldX = bs.position.x;
                s.worldZ = bs.position.z;
                s.sinYaw = sinY;
                s.cosYaw = cosY;
                s.speed  = bs.forwardSpeed;
                s.age    = 0.f;
                trail.push_back(s);
            }
        }

        // Hull-contact foam disturbances. Splats gaussian foam blobs along
        // the port + starboard hull perimeter (where the hull pushes water
        // aside at the waterline) and an extra one off the stern for prop
        // wash when the boat is moving. Decays naturally via the existing
        // foam decay (~1.4 s half-life), so a fast-moving boat leaves a
        // visible trail of agitated water in addition to the analytical
        // wake foam. Always-on perimeter base intensity even at zero speed
        // — the hull still displaces water, just less dramatically.
        ocean->clearFoamDisturbances();
        {
            const float L = kBoatLength * 0.5f;
            const float B = kBoatBeam   * 0.5f;
            const float spd = std::abs(bs.forwardSpeed);
            const float speedNorm = std::clamp(spd / 4.0f, 0.0f, 1.0f);
            const float baseI     = 0.4f + 0.55f * speedNorm;
            // Analytical hull plan-form half-width at fraction t ∈ [0,1]
            // along the length (0 = bow, 1 = stern). Bow is sharp (≈0 at
            // tip) so foam pinches to the centreline at the prow rather
            // than sitting on the bbox front corners; stern keeps ~75 %
            // beam (Gunnerus has a roughly transom aft form). Without
            // this taper the perimeter splats below sit on the corners
            // of the 28×9 bounding box and read as a rectangular foam
            // outline that doesn't match the hull silhouette.
            auto hullHalfWidth = [B](float t) -> float {
                const float u = 2.0f * t - 1.0f;          // -1 at bow, +1 at stern
                if (u <= 0.0f) {
                    // Elliptical bow with sharpening exponent < 1 — pinches
                    // tighter toward the tip than a pure ellipse would.
                    const float k = 1.0f - u * u;          // 0 at bow tip, 1 amidships
                    return B * std::pow(k, 0.6f);
                }
                // Stern: stays near full beam, fading slightly toward the transom.
                return B * (1.0f - 0.25f * u * u);
            };
            // 8 points each side, bow → stern. Splats are positioned just
            // outside the tapered hull silhouette (+0.2 m offset) so their
            // gaussian halos paint visible foam on the surrounding ocean
            // rather than on the flat hull-excluded zone below the boat.
            // Spacing ≈ 4 m at midship × radius 1.6 m gives partial overlap
            // along the length so the foam reads as a continuous band.
            constexpr int   kSamples = 8;
            constexpr float kRadius  = 1.6f;
            for (int side = -1; side <= 1; side += 2) {
                for (int i = 0; i < kSamples; ++i) {
                    const float t       = float(i) / float(kSamples - 1);
                    const float localZ  = L - 2.0f * L * t;       // +L (bow) → -L (stern)
                    const float localX  = float(side) * (hullHalfWidth(t) + 0.2f);
                    const float worldX  = bs.position.x + cosY * localX + sinY * localZ;
                    const float worldZ  = bs.position.z - sinY * localX + cosY * localZ;
                    ocean->addFoamDisturbance(worldX, worldZ, kRadius, baseI);
                }
            }
            // Prop wash: larger, brighter splat just behind the stern.
            // Only active when actually moving — a docked boat shouldn't
            // be churning. The +0.5m localZ offset puts it just aft of
            // the hull rectangle so it doesn't fight hull exclusion.
            if (speedNorm > 0.1f) {
                const float localZ  = -L - 1.0f;
                const float worldX  = bs.position.x + sinY * localZ;
                const float worldZ  = bs.position.z + cosY * localZ;
                const float propI   = std::min(0.7f + 0.4f * speedNorm, 1.0f);
                ocean->addFoamDisturbance(worldX, worldZ, 3.0f, propI);
            }
        }

        // ── Buoy buoyancy ──────────────────────────────────────────────────
        // Each buoy spring-damps toward the wave height at its fixed waypoint
        // position. Stiffer than the boat (1.6 Hz natural freq vs 0.8) so the
        // small bodies track chop visibly; under-damped (ζ=0.55) so they get
        // a subtle bob after a crest passes through. Same buoyancy cascade
        // mask as the boat (swells + mid; cascade-2 chop is too fine to push
        // a buoy meaningfully).
        if (!buoyStates.empty() && buoyRoot) {
            constexpr float kBuoyOmega = 2.f * 3.14159f * 1.6f;
            constexpr float kBuoyK     = kBuoyOmega * kBuoyOmega;
            constexpr float kBuoyC     = 2.f * 0.55f * kBuoyOmega;
            for (size_t i = 0; i < buoyStates.size(); ++i) {
                BuoyState& s = buoyStates[i];
                const float h = ocean->sampleHeight(s.worldX, s.worldZ, kBuoyancyMask)
                              + ocean->sampleWakeHeight(s.worldX, s.worldZ);
                const float err = h - s.worldYOffset;
                const float accel = err * kBuoyK - s.vY * kBuoyC;
                s.vY += accel * dt;
                s.worldYOffset += s.vY * dt;
                Vector3 dLocal(0.f, s.worldYOffset, 0.f);
                dLocal.applyMatrix3(buoyInvRotScale);
                auto* child = buoyRoot->children[i];
                child->position.copy(s.baseLocal).add(dLocal);
            }
        }

        // ── Radar sweep + blip update ─────────────────────────────────────
        // Sweep advances clockwise (positive angle from "up" = forward).
        // For each target (buoys only here), compute heading-up bearing and
        // check if the sweep just crossed it this frame. When it crosses,
        // set the blip intensity = range-attenuated strength; otherwise
        // decay toward zero. The ImGui block below renders the scope.
        if (radarOn) {
            const float prevSweep = radarSweepAngle;
            const float twoPi     = 2.f * 3.14159265f;
            const float omega     = twoPi / kRadarSweepPeriod;
            radarSweepAngle = std::fmod(radarSweepAngle + omega * dt, twoPi);
            const float decay     = std::exp(-dt / kRadarBlipTau);
            // Helper: does the angle `tgt` ∈ [0, 2π) fall in the swept arc
            // (prev → curr)? Handles wraparound (prev > curr).
            auto inSweep = [&](float prev, float curr, float tgt) {
                if (prev <= curr) return tgt >= prev && tgt <= curr;
                return tgt >= prev || tgt <= curr;
            };
            for (size_t i = 0; i < buoyStates.size() && i < kMaxRadarTargets; ++i) {
                const float dxw = buoyStates[i].worldX - bs.position.x;
                const float dzw = buoyStates[i].worldZ - bs.position.z;
                const float dist = std::sqrt(dxw * dxw + dzw * dzw);
                radarBlipIntensity[i] *= decay;
                if (dist > radarRangeM) continue;
                // Heading-up bearing: world-bearing − yaw, wrapped to [0, 2π).
                float bearing = std::atan2(dxw, dzw) - bs.yaw;
                bearing = std::fmod(bearing, twoPi);
                if (bearing < 0.f) bearing += twoPi;
                if (inSweep(prevSweep, radarSweepAngle, bearing)) {
                    // Range-attenuated echo: 1 close, ~0.3 at the edge.
                    const float r01 = dist / radarRangeM;
                    const float strength = std::clamp(1.f - 0.7f * r01, 0.3f, 1.f);
                    radarBlipIntensity[i] = std::max(radarBlipIntensity[i], strength);
                }
            }
        }

        // ── Auto-cam timer ────────────────────────────────────────────────
        // When enabled, cycle through every camera mode every `autoCamHold`
        // seconds. Skip Free — its position is owned by OrbitControls and
        // re-entering it mid-cycle just shows the user's last orbit. Start
        // the cycle at SideFixed (1) so the first auto switch from Free
        // doesn't snap straight back to Free.
        if (autoCam) {
            autoCamElapsed += dt;
            if (autoCamElapsed >= autoCamHold) {
                autoCamElapsed = 0.f;
                int next = static_cast<int>(camMode) + 1;
                if (next >= kCamModeCount) next = 1;
                if (next == 0) next = 1;
                camMode = static_cast<CamMode>(next);
            }
        }

        // ── Camera pose per mode ──────────────────────────────────────────
        // Boat basis vectors (yaw-only — pitch/roll come from wave tilt but
        // we deliberately ignore them for camera framing so the horizon
        // doesn't wobble in side/chase modes; that's how real-world doc /
        // chase footage is shot too).
        const Vector3 boatFwd{sinY, 0.f, cosY};
        const Vector3 boatRight{cosY, 0.f, -sinY};
        const Vector3 boatPos{bs.position.x, bs.y, bs.position.z};
        switch (camMode) {
            case CamMode::Free: {
                // Existing behavior: orbit target tracks the boat; controls
                // own the position. User drags / scrolls to compose.
                controls.target.set(boatPos.x, boatPos.y, boatPos.z);
                controls.update();
                break;
            }
            case CamMode::SideFixed: {
                // Starboard rail, just above the deck, looking forward.
                // Offset: half-beam + a bit outboard, ~3 m above mean water,
                // 2 m forward of midships so the bow + bow-wave are visible.
                const Vector3 camPos = boatPos
                                      + boatRight * (kBoatBeam * 0.5f + 1.0f)
                                      + Vector3(-2.f, 5.0f, -2.f)
                                      + boatFwd * 2.0f;
                // Look ~50 m ahead along the heading, slightly down so the
                // horizon sits in the upper third (cinematic framing).
                const Vector3 lookAt = boatPos
                                       + boatFwd * 50.0f
                                       + Vector3(0.f, -2.0f, 0.f);
                camera.position.copy(camPos);
                camera.lookAt(lookAt);
                break;
            }
            case CamMode::Chase: {
                // Behind + above, looking at the boat. Tuned so the boat
                // occupies ~⅓ of the vertical frame at this FOV.
                const Vector3 camPos = boatPos
                                      - boatFwd * 32.0f
                                      + Vector3(0.f, 12.0f, 0.f);
                const Vector3 lookAt = boatPos + Vector3(0.f, 2.0f, 0.f);
                camera.position.copy(camPos);
                camera.lookAt(lookAt);
                break;
            }
            case CamMode::Underwater: {
                // Below the surface near the boat, looking forward and
                // slightly up so god rays from the sun are visible and
                // the caustic pattern on the sand floor is in frame.
                const Vector3 camPos = boatPos
                                      - boatFwd * 15.0f
                                      + Vector3(0.f, -3.0f, 0.f);
                const Vector3 lookAt = boatPos
                                      + boatFwd * 20.0f
                                      + Vector3(0.f, -1.0f, 0.f);
                camera.position.copy(camPos);
                camera.lookAt(lookAt);
                break;
            }
            case CamMode::NearestBuoy: {
                // Find the buoy closest to the current boat position and
                // perch the camera right next to it, looking at the boat.
                // As the boat moves through the waypoint loop this naturally
                // hands off from one buoy to the next — the camera tracks
                // whichever marker is currently nearest.
                int   nearest    = -1;
                float nearestSq  = 1e30f;
                for (size_t i = 0; i < buoyStates.size(); ++i) {
                    const float dx = buoyStates[i].worldX - bs.position.x;
                    const float dz = buoyStates[i].worldZ - bs.position.z;
                    const float d2 = dx * dx + dz * dz;
                    if (d2 < nearestSq) {
                        nearestSq = d2;
                        nearest   = static_cast<int>(i);
                    }
                }
                if (nearest >= 0) {
                    const float bx = buoyStates[nearest].worldX;
                    const float bz = buoyStates[nearest].worldZ;
                    // Offset 3 m behind the buoy (toward the boat) and 2.5 m
                    // up so the buoy sits in the foreground with the boat
                    // visible past it. "Behind" = direction from boat to buoy
                    // continued past the buoy by a small amount.
                    const float dxw = bx - bs.position.x;
                    const float dzw = bz - bs.position.z;
                    const float dlen = std::max(std::sqrt(dxw * dxw + dzw * dzw), 1e-3f);
                    const float ux = dxw / dlen, uz = dzw / dlen;
                    const Vector3 camPos(bx + ux * 3.0f,
                                         2.5f,
                                         bz + uz * 3.0f);
                    const Vector3 lookAt(bs.position.x, bs.y + 1.0f, bs.position.z);
                    camera.position.copy(camPos);
                    camera.lookAt(lookAt);
                } else {
                    // No buoys (fallback path took over earlier) — degrade
                    // to a Chase-style framing rather than leaving the camera
                    // stranded at its last pose.
                    const Vector3 camPos = boatPos
                                          - boatFwd * 32.0f
                                          + Vector3(0.f, 12.0f, 0.f);
                    const Vector3 lookAt = boatPos + Vector3(0.f, 2.0f, 0.f);
                    camera.position.copy(camPos);
                    camera.lookAt(lookAt);
                }
                break;
            }
        }

        // ── Lighthouse beam sweep ─────────────────────────────────────────
        if (night) {
            if (capturing) {
                // Deterministic capture aim: steeper + into the FRAMED water so
                // the lit pool lands in-shot (~135 m out, centre-right).
                beamAngle = -1.1f;
                beamTarget.position.set(std::cos(beamAngle) * 150.f, -2.f,
                                        std::sin(beamAngle) * 150.f);
            } else {
                beamAngle += beamSpeed * dt;
                beamTarget.position.set(std::cos(beamAngle) * 300.f, -4.f,
                                        std::sin(beamAngle) * 300.f);
            }
            // Spot origin at the lamp glass's OUTER face along the beam — never
            // inside the housing, so no self-geometry can occlude the beam's
            // shadow rays regardless of emissive thresholds.
            const Vector3 lampPos(0.f, 20.3f, 0.f);
            Vector3 beamDir = beamTarget.position;
            beamDir.sub(lampPos).normalize();
            beam->position.copy(lampPos).addScaledVector(beamDir, 1.7f);
        }

        // ── Underwater fog activation ─────────────────────────────────────
        // Sample the wave height at the camera's XZ position. If the camera
        // is below the surface, enable homogeneous fog (participating media)
        // for the path tracer — this activates the existing volumeInscatter
        // pipeline in raygen.rgen (single-scattering NEE with HG phase).
        // A 0.5 m ramp smooths the transition at the waterline.
        {
            const float waveH = ocean->sampleHeight(camera.position.x,
                                                    camera.position.z);
            const float submerge = waveH - camera.position.y;
            const float target = std::clamp(submerge / 0.5f, 0.f, 1.f);
            // Temporal smoothing so the fog doesn't flash when waves wash
            // over the camera near the waterline (~4 Hz cutoff).
            const float fogAlpha = 1.f - std::exp(-2.f * 3.14159f * 4.f * dt);
            uwDepthSmooth += (target - uwDepthSmooth) * fogAlpha;

            if (uwDepthSmooth > 0.001f) {
                const float d = uwFogDensity * uwDepthSmooth;
                scene.fog = FogExp2(
                        Color(uwFogColor[0], uwFogColor[1], uwFogColor[2]), d);
                renderer.setFogAnisotropy(uwFogAniso);
                renderer.setFogWaterSurfaceY(waveH);
            } else {
                scene.fog = std::nullopt;
                renderer.setFogWaterSurfaceY(1e30f);
            }
        }

        // ── Sound update ──────────────────────────────────────────────────
        // After the camera pose (listener follows it) and the underwater fog
        // (uwDepthSmooth ducks the above-surface ambience). Engine source
        // sits at the stern, just above the waterline by the prop wash.
        {
            const Vector3 sternWorld = boatPos - boatFwd * (kBoatLength * 0.45f) + Vector3(0.f, 1.0f, 0.f);
            const bool thrusting = autoPilot ? std::abs(autoCruiseSpeed) > 0.1f
                                             : (bi.W || bi.S);
            sounds.update(dt, sternWorld, bs.forwardSpeed, thrusting,
                          uwDepthSmooth, camera, audioOn ? audioVol : 0.f);
        }

        if (capturing) {
            if (shotClose) {
                // Near-surface grazing view beside the boat, looking across
                // its wake — the high-foam-coverage region where close-up
                // world-grid surface artifacts (foam texels, normal facets)
                // are magnified to screen-filling size.
                const Vector3 side = Vector3(boatFwd.z, 0.f, -boatFwd.x);
                // Keep the camera MOVING (slow sideways strafe) — temporal
                // history (TAA / reflection accumulation) never converges,
                // matching interactive flying where the tile artifact shows.
                camera.position.copy(boatPos)
                        .addScaledVector(boatFwd, -10.f)
                        .addScaledVector(side, 9.f + 0.04f * static_cast<float>(shotFrame))
                        .add(Vector3(0.f, 2.5f, 0.f));
                Vector3 tgt = boatPos;
                tgt.addScaledVector(boatFwd, -45.f);// far into the wake — near-horizontal grazing
                tgt.y = 0.5f;
                camera.lookAt(tgt);
            } else if (shotVista) {
                // High oblique overview: lighthouse centre, archipelago ring,
                // passes and the far shore all in frame.
                camera.position.set(300.f, 220.f, 520.f);
                camera.lookAt(Vector3(0.f, 0.f, -60.f));
            } else if (night) {
                // Night framing: low camera toward the lighthouse + horizon —
                // sky (stars), beam fan, and lit water all in shot.
                camera.position.set(70.f, 7.f, 110.f);
                camera.lookAt(Vector3(0.f, 14.f, 0.f));
            } else {
                // Day framing (≈ the reported-artifact view): boat centre,
                // mostly water in frame, horizon out of shot.
                camera.position.set(10.f, 32.f, 50.f);
                camera.lookAt(Vector3(0.f, 0.f, -20.f));
            }
            camera.updateMatrixWorld();
        }

        renderer.render(scene, camera);

        // F12: interactive screenshot → aaa_caps/usershot_N.png. Native-res
        // ground truth for artifact reports — screen captures of the window
        // get rescaled/compressed and hide texel-scale structure.
        if (bi.shotRequest) {
            bi.shotRequest = false;
            static int userShotN = 0;
            const auto path = std::filesystem::path(PROJECT_FOLDER) / "aaa_caps" /
                              ("usershot_" + std::to_string(userShotN++) + ".png");
            try {
                renderer.writeFramebuffer(path);
                std::printf("wrote %s\n", path.string().c_str());
            } catch (const std::exception& e) {
                std::cerr << "[shot] failed: " << e.what() << "\n";
            }
        }

        if (capturing && toggleNightAt > 0 && shotFrame == toggleNightAt) {
            night = true;// runtime day→night flip — the path the UI checkbox takes
            applyMode();
        }
        if (capturing && ++shotFrame >= shotFrames) {
            const auto path = std::filesystem::path(PROJECT_FOLDER) / "aaa_caps" / shotPath;
            renderer.writeFramebuffer(path);// creates parent dirs; throws on failure
            std::printf("wrote %s\n", path.string().c_str());
            std::exit(0);
        }

        // ── LIDAR scan + visualisation update ─────────────────────────────
        // Must follow render() so the TLAS is built; the cloud/panel show
        // last frame's data when re-rendered next frame.
        if (lidarEnabled) {
            const auto t0 = std::chrono::steady_clock::now();
            lidarSensor->scan(renderer, lidarReturns);
            const auto t1 = std::chrono::steady_clock::now();
            lidarLastScanMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

            // Cloud refresh.
            auto* cloudPos = lidarCloudGeom->getAttribute<float>("position");
            auto* cloudCol = lidarCloudGeom->getAttribute<float>("color");
            int vi = 0;
            Color col;
            int validHits = 0;
            for (const auto& r : lidarReturns) {
                if (r.returnNo <= 0) continue;// miss / sub-threshold
                ++validHits;
                if (vi >= cloudPos->count()) break;
                const float t = std::clamp(r.intensity * 3.f, 0.f, 1.f);
                col.setHSL((1.f - t) * 0.66f, 1.f, 0.5f);
                cloudPos->setXYZ(vi, r.position.x, r.position.y, r.position.z);
                cloudCol->setXYZ(vi, col.r, col.g, col.b);
                ++vi;
            }
            lidarCloudGeom->setDrawRange(0, vi);
            cloudPos->needsUpdate();
            cloudCol->needsUpdate();
            lidarLastReturns = validHits;

            // Panel refresh — (azimuth × elevation) grid filled in blocks so
            // the linear sprite sampler doesn't blend bright cells with
            // empty rows. The Ouster pattern uses 1028 azimuth × 128
            // elevation, so the panel naturally renders dense even though
            // the source texture is sparse-by-design.
            if (lidarShowPanel) {
                const int numAz = std::max(1, static_cast<int>(std::round(
                                                  (lidarModel.azimuthMax - lidarModel.azimuthMin) /
                                                  lidarModel.azimuthResolution)));
                const int numElev = static_cast<int>(lidarModel.elevationAngles.size());
                const int blockW = std::max(1, static_cast<int>(kLidarPanelW) / numAz);
                const int blockH = std::max(1, static_cast<int>(kLidarPanelH) / numElev);

                auto& panelBytes = lidarPanelTex->image().data<unsigned char>();
                std::fill(panelBytes.begin(), panelBytes.end(), 0u);

                // Panel shows the FIRST sample's FIRST return per beam
                // (real LIDAR sensors' default debug view). Other samples
                // / multi-returns appear in the 3-D cloud where they don't
                // overlap spatially.
                const uint32_t maxRet  = std::max(1u, lidarSensor->params.maxReturns);
                const uint32_t samples = std::max(1u, lidarSensor->params.samplesPerBeam);
                const uint32_t slotsPerBeam = maxRet * samples;
                for (size_t beam = 0; beam < lidarSensor->beamCount(); ++beam) {
                    const size_t b = beam * slotsPerBeam;
                    if (b >= lidarReturns.size()) break;
                    const auto& r = lidarReturns[b];
                    if (r.returnNo <= 0) continue;// miss
                    const int ai = static_cast<int>(beam) / numElev;
                    const int ei = static_cast<int>(beam) % numElev;
                    const int px0 = std::clamp(ai * static_cast<int>(kLidarPanelW) / numAz,
                                                0, static_cast<int>(kLidarPanelW) - 1);
                    const int py0 = std::clamp(ei * static_cast<int>(kLidarPanelH) / numElev,
                                                0, static_cast<int>(kLidarPanelH) - 1);
                    const float t = std::clamp(r.intensity * 3.f, 0.f, 1.f);
                    col.setHSL((1.f - t) * 0.66f, 1.f, 0.5f);
                    const unsigned char rByte = static_cast<unsigned char>(col.r * 255.f);
                    const unsigned char gByte = static_cast<unsigned char>(col.g * 255.f);
                    const unsigned char bByte = static_cast<unsigned char>(col.b * 255.f);
                    for (int dy = 0; dy < blockH; ++dy) {
                        const int y = std::min(py0 + dy, static_cast<int>(kLidarPanelH) - 1);
                        size_t row = static_cast<size_t>(y) * kLidarPanelW * 4;
                        for (int dx = 0; dx < blockW; ++dx) {
                            const int x = std::min(px0 + dx, static_cast<int>(kLidarPanelW) - 1);
                            const size_t idx = row + static_cast<size_t>(x) * 4;
                            panelBytes[idx + 0] = rByte;
                            panelBytes[idx + 1] = gByte;
                            panelBytes[idx + 2] = bByte;
                            panelBytes[idx + 3] = 255u;
                        }
                    }
                }
                lidarPanelTex->needsUpdate();
            }

            // Synthesize the per-beam analog waveform from the discrete
            // returns. Cheap (<1 ms) for one beam per frame.
            if (lidarShowWave) {
                const uint32_t mr = std::max(1u, lidarSensor->params.maxReturns);
                const uint32_t sp = std::max(1u, lidarSensor->params.samplesPerBeam);
                lidarWavParams.maxRange = lidarSensor->params.maxRange;
                synthesizeLidarWaveform(lidarReturns,
                                        lidarSensor->beamCount(),
                                        static_cast<size_t>(mr) * sp,
                                        lidarWavParams, lidarWaveform);
            }
        } else {
            lidarCloudGeom->setDrawRange(0, 0);
        }

        if (!capturing) ui.render();// capture reads the bare frame (no HUD)
    });

    return 0;
}
