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

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/input/KeyListener.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/objects/DisplacedMesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace threepp;

namespace {

    // Boat input state — captured by KeyListener, polled each frame.
    struct BoatInput : KeyListener {
        bool W = false, A = false, S = false, D = false;
        void onKeyPressed(KeyEvent e) override { update(e.key, true); }
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

    // Procedural water-surface normal map. Multi-octave sum of sine waves in
    // tangent space, with each octave at a different angle and double the
    // frequency. Periodic on [0,1]² so it tiles cleanly when wrapped. Used
    // as `mat->normalMap` on the ocean to add sub-mesh-resolution shading
    // detail (the FFT cascades produce ~1 m mesh-resolvable waves; the
    // visible "real ocean" has cm-scale chop that this fakes via the BRDF
    // microfacet path rather than geometry).
    //
    // Returns a 256² RGBA8 DataTexture with tangent-space normals encoded
    // [0..1] in RGB. Caller wraps + tiles it; `repeat = (20, 20)` over a
    // 1000 m ocean gives ~5 m / texel = ~2 cm normal-detail granularity.
    std::shared_ptr<DataTexture> makeProceduralWaterNormalMap() {
        constexpr unsigned int N = 256;
        std::vector<unsigned char> data(N * N * 4);

        // Tiling 2D value noise — hash of integer lattice points + smooth
        // bilinear interpolation. The hash wraps modulo `period` so the
        // resulting noise field tiles cleanly at integer multiples of
        // `period` per [0,1]² UV. Used as the base of an FBM that gives
        // the surface a non-regular, water-y look (rather than the regular
        // grid pattern that sum-of-sines at power-of-two frequencies
        // produces — that reads as fabric weave).
        auto hash = [](int x, int y) {
            uint32_t h = uint32_t(x) * 73856093u ^ uint32_t(y) * 19349663u;
            h ^= h >> 16; h *= 0x7feb352du;
            h ^= h >> 15; h *= 0x846ca68bu;
            h ^= h >> 16;
            return float(h) / 4294967296.0f;// [0..1]
        };
        auto valueNoise = [&hash](float u, float v, int period) {
            const float U = u * float(period);
            const float V = v * float(period);
            const int x0 = int(std::floor(U)), y0 = int(std::floor(V));
            const float fx = U - float(x0), fy = V - float(y0);
            // Smoothstep for C1 continuity (otherwise see grid lines).
            const float sx = fx * fx * (3.f - 2.f * fx);
            const float sy = fy * fy * (3.f - 2.f * fy);
            // Wrap lattice coords mod `period` for tileability.
            const int X0 = ((x0 % period) + period) % period;
            const int Y0 = ((y0 % period) + period) % period;
            const int X1 = (X0 + 1) % period;
            const int Y1 = (Y0 + 1) % period;
            const float a = hash(X0, Y0);
            const float b = hash(X1, Y0);
            const float c = hash(X0, Y1);
            const float d = hash(X1, Y1);
            const float ab = a * (1.f - sx) + b * sx;
            const float cd = c * (1.f - sx) + d * sx;
            return (ab * (1.f - sy) + cd * sy) * 2.f - 1.f;// → [-1, 1]
        };
        // FBM: sum noise octaves at increasing period (= higher frequency)
        // with halving amplitude. Periods are integers so each octave wraps,
        // and we mix non-power-of-two periods (3, 7, 13, …) to avoid the
        // regular dyadic-grid look.
        // Period choices: skip the lowest periods (3, 5) — they generate
        // noise blobs that span ~⅓ of a tile, which after wrap-tiling read
        // as obvious large-scale repetition. Starting at 11/19/31… makes
        // the largest features ~9 % of a tile, too small to register as a
        // tile-boundary cue when viewed across many tiles.
        auto height = [&valueNoise](float u, float v) {
            float h = 0.f;
            h += 1.000f * valueNoise(u, v, 11);
            h += 0.500f * valueNoise(u, v, 19);
            h += 0.250f * valueNoise(u, v, 37);
            h += 0.125f * valueNoise(u, v, 71);
            h += 0.062f * valueNoise(u, v, 137);
            return h;
        };

        const float eps = 1.f / float(N);
        const float bumpStrength = 1.5f;// tuning knob for "how rippled"
        for (unsigned y = 0; y < N; ++y) {
            for (unsigned x = 0; x < N; ++x) {
                const float u = float(x) / float(N);
                const float v = float(y) / float(N);
                const float hL = height(u - eps, v);
                const float hR = height(u + eps, v);
                const float hD = height(u, v - eps);
                const float hU = height(u, v + eps);
                const float dx = (hR - hL) * 0.5f / eps * bumpStrength;
                const float dy = (hU - hD) * 0.5f / eps * bumpStrength;
                const float invLen = 1.f / std::sqrt(dx * dx + dy * dy + 1.f);
                const float nx = -dx * invLen;
                const float ny = -dy * invLen;
                const float nz = invLen;
                const unsigned i = (y * N + x) * 4;
                data[i + 0] = static_cast<unsigned char>((nx * 0.5f + 0.5f) * 255.f);
                data[i + 1] = static_cast<unsigned char>((ny * 0.5f + 0.5f) * 255.f);
                data[i + 2] = static_cast<unsigned char>((nz * 0.5f + 0.5f) * 255.f);
                data[i + 3] = 255;
            }
        }
        auto tex = DataTexture::create(data, N, N);
        tex->magFilter = Filter::Linear;
        tex->minFilter = Filter::Linear;
        tex->wrapS = TextureWrapping::Repeat;
        tex->wrapT = TextureWrapping::Repeat;
        return tex;
    }

    constexpr float kTileSize    = 1000.0f;  // metres — full mesh extent and cascade-0 tile
    constexpr uint32_t kFftSize  = 1024;     // 4× cost vs 512² → 0.98 m vertex spacing, resolves λ ≥ 2 m
    constexpr float kPlaneEdge   = kTileSize;  // mesh extends one full FFT tile in X and Z
    constexpr int   kSubdiv      = static_cast<int>(kFftSize) - 1;  // 1024² verts ≈ 1 M; BLAS rebuild scales with this

    auto makeOceanMaterial() {
        auto mat = MeshPhysicalMaterial::create();
        // Pure water has no diffuse pigment — the blue comes from Beer-Lambert
        // absorption through the medium, not albedo.
        mat->color = Color::white;
        // Small roughness simulates the sub-pixel chop the FFT can't resolve.
        // 0.02 keeps the glitter tight (sharper specular highlights, fewer
        // distributed sparkles) — denoiser-off output reads as "specular
        // glints on water" rather than "scattered noise".
        mat->roughness = 0.01f;
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
        // mat->clearcoat = 0.1;

        // Procedural water-surface normal map. The FFT cascades only resolve
        // waves down to the mesh resolution (~1 m at 1024² over a 1 km tile),
        // but real ocean has cm-scale chop the eye reads as "wet, textured
        // water". The normal map perturbs the BRDF normal at sub-pixel scale
        // → glittery sun reflections + breaks up the otherwise mirror-clean
        // wave faces. Tile factor 30× across the ocean = each texture tile
        // covers ~33 m, so chop features land at the right perceptual size.
        // normalScale tunes how strong the bump reads in the BRDF.
        mat->normalMap = makeProceduralWaterNormalMap();
        // Higher repeat = smaller tile = less obvious repetition. 60× over
        // 1000 m → each tile covers ~17 m of ocean. Combined with the
        // texture's smallest feature (~0.12 m at period 137), normal-map
        // detail spans ~10 cm – 1.5 m. Reads as chop, not tiled fabric.
        mat->normalMap->repeat.set(60.f, 60.f);
        mat->normalScale.set(0.6f, 0.6f);
        return mat;
    }

    auto makeSandMaterial() {
        return MeshStandardMaterial::create({
                {"color", Color::black},
                {"roughness", 1.0f},
        });
    }

}// namespace

int main() {

    Canvas canvas("Vulkan PT  Ocean", {{"vsync", false}, {"size", WindowSize{1600, 900}}});
    VulkanRenderer renderer(canvas);
    renderer.setDenoise(true);
    renderer.setHybridEnabled(true);
    renderer.setHybridDebugView(0);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    // puresky_2k.hdr is a very bright daylight env. ACES desaturates strongly
    // at high luminance so the sand goes white-ish even at 0.5; 0.3 keeps
    // the sand in the saturated regime so its tan colour shows through and
    // the water's blue-tinted depth gradient can be read against it.
    renderer.toneMappingExposure = 0.5f;

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
    sun->position.set(2.f, 5.f, 2.f);
    Object3D sunTarget;
    sunTarget.position.set(0.f, 0.f, 0.f);
    sun->setTarget(sunTarget);
    scene.add(sun);

    // auto ambient = AmbientLight::create(Color(0.55f, 0.72f, 0.95f), 0.25f);
    // scene.add(ambient);

    // Sand floor sits directly under the ocean tile and matches its extent:
    // making the floor larger leaves a visible sand frame around the water
    // when viewed from above (the open-ocean illusion breaks). At the
    // edges, rays going past the water plane just hit the env sky, which
    // sells "horizon" better than visible beach.
    auto floor = Mesh::create(PlaneGeometry::create(kPlaneEdge, kPlaneEdge),
                              makeSandMaterial());
    floor->rotation.x = -math::PI / 2.f;
    floor->position.y = -50.f;
    scene.add(floor);

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
    ocean->params.windSpeed   = 10.0f;
    ocean->params.waveScale   = 1.0f;
    ocean->params.choppiness  = 0.55f;      // sharper crests, more visible wave-folding
    ocean->params.textureSize = kFftSize;
    scene.add(ocean);

    // Gunnerus research vessel (~28 m × 9 m). Loaded synchronously at
    // startup — 31 MB binary glTF takes ~1 s. Re-scaled to a known length so
    // the hydrodynamic sample points (fore/aft/port/starboard) are
    // physically meaningful regardless of the asset's source unit.
    constexpr float kBoatLength = 28.0f;
    constexpr float kBoatBeam   = 9.0f;
    GLTFLoader gltfLoader;
    auto boat = Group::create();
    auto gltf = gltfLoader.load(std::string(DATA_FOLDER) + "/models/gltf/Gunnerus.glb");
    if (!gltf || !gltf->scene) {
        std::cerr << "Failed to load Gunnerus.glb" << std::endl;

        boat->add(Mesh::create(BoxGeometry::create(kBoatBeam, 5.f, kBoatLength),
                              MeshStandardMaterial::create({{"color", Color::red}})));

    } else {
        auto innerAsset = gltf->scene;

        boat->add(innerAsset);
        {
            Box3 bbox;
            bbox.setFromObject(*innerAsset);
            Vector3 size; bbox.getSize(size);
            // If the asset is longer in X than Z, its longitudinal axis is +X
            // not +Z. Rotate so its long axis aligns with Z to match our heading
            // convention (yaw = 0 → translate along +Z).
            if (size.x > size.z) {
                innerAsset->rotateY(-math::PI / 2.f);
                bbox.setFromObject(*innerAsset);
                bbox.getSize(size);
            }
            // Bow direction can't be inferred from bbox alone — both ends of
            // the longest axis look the same from outside. Empirically Gunnerus
            // ends up with bow at −Z after the auto-rotation, so W key (which
            // translates +Z) reads as "go backward". Add a 180° flip so the
            // visual nose points along the heading direction. If you load a
            // different asset whose bow ends up at +Z naturally, drop this.
            innerAsset->rotateY(math::PI);
            const float maxExtent = std::max({size.x, size.y, size.z});
            if (maxExtent > 0.f) {
                const float s = kBoatLength / maxExtent;
                innerAsset->scale.set(s, s, s);
            }
        }
    }

    scene.add(boat);
    BoatState bs;
    BoatInput bi;
    canvas.addKeyListener(bi);

    // Far-clip raised so the horizon doesn't get cut at the elevated/distant
    // camera. Initial position is offset from the boat; OrbitControls'
    // target is updated each frame to follow the boat so orbiting always
    // happens around the vessel.
    PerspectiveCamera camera(45.f, canvas.aspect(), 0.1f, 2000.f);
    camera.position.set(40.f, 18.f, 60.f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, 1.f, 0.f);
    controls.update();

    float waveScale = ocean->params.waveScale;
    float choppiness = ocean->params.choppiness;
    float windSpeed = ocean->params.windSpeed;
    float windTheta = ocean->params.windTheta;
    float exposure  = renderer.toneMappingExposure;
    int   toneMode  = static_cast<int>(renderer.toneMapping);
    int   spp       = renderer.samplesPerPixel();
    float fps = 0.f, fpsAccum = 0.f;
    int   fpsFrames = 0;

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({340, 0});
        ImGui::Begin("Vulkan PT — Ocean");
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
        ImGui::Text("Keys  W:%d  A:%d  S:%d  D:%d",
                    bi.W ? 1 : 0, bi.A ? 1 : 0, bi.S ? 1 : 0, bi.D ? 1 : 0);
        ImGui::Separator();
        if (ImGui::SliderFloat("Wave scale", &waveScale, 0.f, 3.f, "%.2f"))
            ocean->params.waveScale = waveScale;
        if (ImGui::SliderFloat("Choppiness", &choppiness, 0.f, 1.0f, "%.2f"))
            ocean->params.choppiness = choppiness;
        if (ImGui::SliderFloat("Wind speed (m/s)", &windSpeed, 1.f, 30.f, "%.1f"))
            ocean->params.windSpeed = windSpeed;
        if (ImGui::SliderFloat("Wind direction (rad)", &windTheta, -3.14f, 3.14f, "%.2f"))
            ocean->params.windTheta = windTheta;
        ImGui::TextDisabled("Wind changes apply on scene reload.");
        ImGui::Separator();
        if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 5.0f, "%.2f"))
            renderer.toneMappingExposure = exposure;
        const char* toneItems[] = {"None", "Linear", "Reinhard", "Cineon", "ACESFilmic"};
        if (ImGui::Combo("Tone mapping", &toneMode, toneItems, IM_ARRAYSIZE(toneItems)))
            renderer.toneMapping = static_cast<ToneMapping>(toneMode);
        if (ImGui::SliderInt("Samples / pixel", &spp, 1, 16))
            renderer.setSamplesPerPixel(spp);
        ImGui::End();
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
        // Linear: W = forward thrust, S = reverse. Speed clamped to ±vMax;
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
        const float cosY = std::cos(bs.yaw);
        const float sinY = std::sin(bs.yaw);
        bs.position.x += sinY * bs.forwardSpeed * dt;
        bs.position.z += cosY * bs.forwardSpeed * dt;

        // === Hydrodynamics from sampled wave surface ===
        // Sample 5 points: centre + four corners of the bounding rectangle.
        // Heave (Y) follows the centre; pitch comes from fore/aft height
        // diff over hull length; roll from port/starboard diff over beam.
        const float halfL = kBoatLength * 0.5f;
        const float halfB = kBoatBeam   * 0.5f;
        auto sampleH = [&](float dx, float dz) {
            // Local (forward, right) → world via yaw rotation
            const float wx = bs.position.x + sinY * dz + cosY * dx;
            const float wz = bs.position.z + cosY * dz - sinY * dx;
            return ocean->sampleHeight(wx, wz);
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

        // Pitch / roll: temporal low-pass at ~3 Hz so attitude rides the
        // swells but ignores sub-metre chop. alpha = 1 − exp(−2π · cutoff · dt).
        const float alpha = 1.f - std::exp(-2.f * 3.14159f * 3.f * dt);
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

        // No camera follow — was making the boat read as "stuck" at the
        // camera centre, since camera and target moved with it. Let the
        // user orbit / zoom manually; the boat translates freely in world
        // and the Pos readout in the UI shows actual coordinates.
        controls.target.set(bs.position.x, bs.y, bs.position.z);
        controls.update();

        renderer.render(scene, camera);
        ui.render();
    });

    return 0;
}
