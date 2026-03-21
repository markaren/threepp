// WebTide: FFT-based ocean simulation using threepp's WgpuRenderer (WebGPU).
// Ported from the WebTide BabylonJS/TypeScript project.
//
// Credits & References:
//   - Original project: WebTide by Barth Paleologue
//     https://github.com/BarthPaleologue/WebTide
//   - "Simulating Ocean Water" by Jerry Tessendorf
//   - GPU-based FFT from BabylonJS Ocean Demo by Popov72
//   - Specular coefficients from Shadertoy by afl_ext
//   - Acerola's video breakdown of Tessendorf's paper
//   - Tangent calculations by Rikard Olajos
//   - Preetham sky shader ported from three.js (Preetham et al. 1999)
//   - Sand texture by Engin Akyurt

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/materials/MeshLambertMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/threepp.hpp"

#include "threepp/renderers/wgpu/WgpuTexture.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"

#include "BuoyantSampler.hpp"
#include "DynamicSpectrum.hpp"
#include "IFFT.hpp"
#include "OceanFoam.hpp"
#include "PhillipsSpectrum.hpp"
#include "shaders.hpp"

#include "threepp/geometries/SphereGeometry.hpp"

#include <chrono>

using namespace threepp;

int main() {

    constexpr uint32_t textureSize = 256;
    constexpr float C0_TILE = 5.0f;    // ripples
    constexpr float C1_TILE = 40.0f;   // main chop (= reference tile for mesh)
    constexpr float C2_TILE = 400.0f;  // long swell
    constexpr int tileRadius = 6;      // 13x13 grid (-6..6)

    // Band-pass boundaries from Nyquist wavenumbers
    const float kNyq1 = math::PI * textureSize / C1_TILE;  // ~20.1
    const float kNyq2 = math::PI * textureSize / C2_TILE;  // ~2.01

    // Create window and renderer
    Canvas::Parameters params;
    params.title("WebTide Ocean")
            .size(1280, 720)
            .graphicsApi(GraphicsAPI::WebGPU);

    Canvas canvas(params);
    WgpuRenderer renderer(canvas);

    // Camera
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 2000.f);
    camera->position.set(0, 25, 100);
    camera->lookAt({0, 0, 0});

    OrbitControls controls(*camera, canvas);
    controls.target.set(0, 1.5f, 0);
    controls.minDistance = 2;
    controls.update();

    // Scene
    auto scene = Scene::create();

    // Lights
    auto ambient = AmbientLight::create(Color(0x404040));
    scene->add(ambient);

    auto dirLight = DirectionalLight::create(Color(0xffffff), 0.8f);
    dirLight->position.set(-1, 1, -3);
    scene->add(dirLight);

    // Initialize ocean compute pipelines — 3 cascades with band-pass filtering
    // PhillipsSpectrum(renderer, textureSize, tileSize, kMin, kMax, smallWaveCutoff)
    webtide::PhillipsSpectrum spec0(renderer, textureSize, C0_TILE, kNyq1, 0.0f,  0.002f);  // kMax=0 means no upper limit
    webtide::PhillipsSpectrum spec1(renderer, textureSize, C1_TILE, kNyq2, kNyq1, 0.01f);
    webtide::PhillipsSpectrum spec2(renderer, textureSize, C2_TILE, 0.0f,  kNyq2, 0.05f);

    webtide::DynamicSpectrum dynSpec0(renderer, spec0, textureSize, C0_TILE);
    webtide::DynamicSpectrum dynSpec1(renderer, spec1, textureSize, C1_TILE);
    webtide::DynamicSpectrum dynSpec2(renderer, spec2, textureSize, C2_TILE);

    webtide::IFFT ifft0(renderer, textureSize);
    webtide::IFFT ifft1(renderer, textureSize);
    webtide::IFFT ifft2(renderer, textureSize);

    // Output spatial-domain textures (3 per cascade: height, gradient, displacement).
    // height0/1/2 and displacement1 need CopySrc so the BuoyantSampler can read them back.
    constexpr uint32_t kReadable = WgpuTexture::Storage | WgpuTexture::TextureBinding |
                                   WgpuTexture::CopyDst | WgpuTexture::CopySrc;
    constexpr uint32_t kDefault  = WgpuTexture::Storage | WgpuTexture::TextureBinding |
                                   WgpuTexture::CopyDst;

    WgpuTexture height0(renderer, textureSize, textureSize, WgpuTexture::Format::RG32Float, kReadable);
    WgpuTexture gradient0(renderer, textureSize, textureSize, WgpuTexture::Format::RG32Float, kDefault);
    WgpuTexture displacement0(renderer, textureSize, textureSize, WgpuTexture::Format::RG32Float, kReadable);

    WgpuTexture height1(renderer, textureSize, textureSize, WgpuTexture::Format::RG32Float, kReadable);
    WgpuTexture gradient1(renderer, textureSize, textureSize, WgpuTexture::Format::RG32Float, kReadable);
    WgpuTexture displacement1(renderer, textureSize, textureSize, WgpuTexture::Format::RG32Float, kReadable);

    WgpuTexture height2(renderer, textureSize, textureSize, WgpuTexture::Format::RG32Float, kReadable);
    WgpuTexture gradient2(renderer, textureSize, textureSize, WgpuTexture::Format::RG32Float, kReadable);
    WgpuTexture displacement2(renderer, textureSize, textureSize, WgpuTexture::Format::RG32Float, kReadable);

    // Foam system driven by cascade 1 Jacobian
    webtide::OceanFoam oceanFoam(renderer, textureSize);

    // -------------------------------------------------------------------------
    // Buoyant object — physics-driven sphere that rides the waves accurately.
    //
    // Position Y: spring-mass system (mass, spring, damper) so the buoy bobs
    //             with realistic inertia rather than snapping to the surface.
    // Rotation:   pitch/roll follow the wave-surface normal (from gradient
    //             textures), smoothed with an exponential lag filter.
    // XZ:         Gerstner horizontal displacement from GPU readback.
    //
    // All wave values come from GPU via BuoyantSampler (CopyTextureToBuffer).
    // -------------------------------------------------------------------------
    webtide::BuoyantSampler buoySampler(renderer);

    constexpr float kBuoyRadius = 2.0f;  // metres

    // Anchor = undisplaced XZ reference point.
    float buoyAnchorX = 5.0f;
    float buoyAnchorZ = 5.0f;

    // Spring-mass physics — physically correct buoyancy for a 2 m sphere at 50 % submersion.
    //
    //  Buoyancy stiffness:  k = ρ_w · g · π · r²  = 1025·9.81·π·4  ≈ 126 000 N/m
    //  Natural period 2 s:  m = k / ω₀²            = 126000 / π²   ≈ 12 760 kg
    //  Damping ratio ζ=0.4: c = 2·ζ·√(k·m)                         ≈ 32 100 N·s/m
    //
    // ω₀(buoy) ≈ 3.14 rad/s  >>  ω(ocean wave) ≈ 0.5–1.0 rad/s
    // → buoy responds ~3× faster than the waves, so it tracks tightly with
    //   slight underdamped oscillation when a crest passes quickly.
    constexpr float kBuoyMass    = 12760.f;
    constexpr float kBuoySpring  = 126000.f;
    constexpr float kBuoyDamping = 32100.f;

    bool  buoyFirstFrame = true; // initialise physY from GPU on first sample
    float buoyPhysY  = 0.f;     // sphere-centre Y (spring state)
    float buoyVelY   = 0.f;     // vertical velocity m/s
    float buoyWaveY  = 0.f;     // GPU-sampled wave surface height (shown in UI)

    // Smoothed orientation (exponential lag, τ ≈ 0.2 s)
    float buoyPitch  = 0.f;   // rotation around X (driven by dH/dz)
    float buoyRoll   = 0.f;   // rotation around Z (driven by dH/dx)

    auto buoyGeo = SphereGeometry::create(kBuoyRadius, 24, 18);
    auto buoyMat = MeshLambertMaterial::create();
    buoyMat->color = Color(0xff4400); // classic orange-red marine buoy
    auto buoyMesh = Mesh::create(buoyGeo, buoyMat);
    buoyMesh->position.set(buoyAnchorX, 0.f, buoyAnchorZ);
    scene->add(buoyMesh);

    // Create water ShaderMaterial with custom WGSL shaders
    auto waterMaterial = ShaderMaterial::create();
    waterMaterial->vertexShader = webtide::waterVertexWGSL;
    waterMaterial->fragmentShader = webtide::waterFragmentWGSL;
    waterMaterial->side = Side::Double;
    waterMaterial->transparent = true;
    waterMaterial->depthWrite = false;  // don't occlude geometry behind the surface

    // Set custom uniforms (ocean params at binding 2)
    // Uniform names are alphabetical to match WgpuRenderer's packing order:
    //   foamStrength, foamThreshold, fogDensity, seaColor, tileSize, waveScale
    waterMaterial->uniforms["tileSize"] = Uniform(C1_TILE);

    // Bind GPU textures — MUST be alphabetical for deterministic binding numbers.
    // cascade0Displacement(3,4), cascade0Gradient(5,6), cascade0Height(7,8)
    // cascade1Displacement(9,10), cascade1Gradient(11,12), cascade1Height(13,14)
    // cascade2Displacement(15,16), cascade2Gradient(17,18), cascade2Height(19,20)
    // foamMap(21,22)  [reflectionMap removed — sky is analytical]
    waterMaterial->customTextures["cascade0Displacement"] = &displacement0;
    waterMaterial->customTextures["cascade0Gradient"]     = &gradient0;
    waterMaterial->customTextures["cascade0Height"]       = &height0;
    waterMaterial->customTextures["cascade1Displacement"] = &displacement1;
    waterMaterial->customTextures["cascade1Gradient"]     = &gradient1;
    waterMaterial->customTextures["cascade1Height"]       = &height1;
    waterMaterial->customTextures["cascade2Displacement"] = &displacement2;
    waterMaterial->customTextures["cascade2Gradient"]     = &gradient2;
    waterMaterial->customTextures["cascade2Height"]       = &height2;
    waterMaterial->customTextures["foamMap"]              = &oceanFoam.currentFoam();

    // LOD geometries — inner tiles keep detail, outer tiles are coarser.
    // All share the same material and FFT textures; only vertex density differs.
    // Chebyshev distance from centre determines LOD:
    //   dist <= 1 : 256x256  (9 tiles,  full detail)
    //   dist <= 3 : 128x128  (40 tiles, half detail)
    //   dist <= 6 : 64x64    (120 tiles, quarter detail)
    auto makeWaterGeo = [&](int subdiv) {
        auto g = PlaneGeometry::create(C1_TILE, C1_TILE, subdiv, subdiv);
        g->rotateX(-math::PI / 2.0f);
        return g;
    };
    auto geoInner = makeWaterGeo(256);   // dist <= 1 :  9 tiles, full detail
    auto geoMid   = makeWaterGeo(128);   // dist <= 3 : 40 tiles, half detail
    auto geoOuter = makeWaterGeo(64);    // dist <= 6 :120 tiles, quarter detail

    // Tile meshes stored so we can reposition them each frame as the camera moves.
    // Each tile lives at offset (x,z) from the camera's snapped tile position.
    struct TileEntry { std::shared_ptr<Mesh> mesh; int ox, oz; };
    std::vector<TileEntry> waterTiles;
    waterTiles.reserve((2 * tileRadius + 1) * (2 * tileRadius + 1));

    for (int x = -tileRadius; x <= tileRadius; x++) {
        for (int z = -tileRadius; z <= tileRadius; z++) {
            int dist = std::max(std::abs(x), std::abs(z));
            auto& geo = dist <= 1 ? geoInner : dist <= 3 ? geoMid : geoOuter;
            auto waterMesh = Mesh::create(geo, waterMaterial);
            scene->add(waterMesh);
            waterTiles.push_back({waterMesh, x, z});
        }
    }

    // Snap tile grid to the orbit target (where the camera is looking on the water surface).
    // This keeps the high-detail inner ring centred in front of the viewer, not behind.
    auto updateTilePositions = [&] {
        int snapX = static_cast<int>(std::round(controls.target.x / C1_TILE));
        int snapZ = static_cast<int>(std::round(controls.target.z / C1_TILE));
        for (auto& t : waterTiles) {
            t.mesh->position.set(
                static_cast<float>(snapX + t.ox) * C1_TILE,
                0,
                static_cast<float>(snapZ + t.oz) * C1_TILE);
        }
    };
    updateTilePositions();

    // Ground plane (sand)
    auto groundGeo = PlaneGeometry::create((1 + tileRadius * 2.0f) * C1_TILE, (1 + tileRadius * 2.0f) * C1_TILE);
    groundGeo->rotateX(-math::PI / 2.0f);
    auto groundMat = MeshLambertMaterial::create();
    groundMat->color = Color(0xc2b280);// Sand color

    // Try to load sand texture
    TextureLoader textureLoader;
    try {
        auto sandTex = textureLoader.load(std::string(DATA_FOLDER) + "/textures/sand.jpg");
        if (sandTex) {
            groundMat->map = sandTex;
        }
    } catch (...) {
        // Sand texture not found, use solid color
    }

    auto ground = Mesh::create(groundGeo, groundMat);
    ground->position.y = -10.0f;
    scene->add(ground);

    // Analytical sky — Preetham atmospheric scattering rendered into a large box.
    // The sky box follows the camera each frame so it is always visible.
    // Sun direction comes from LightData (binding 1) so sky and water agree.
    auto skyMaterial = ShaderMaterial::create();
    skyMaterial->vertexShader   = webtide::skyVertexWGSL;
    skyMaterial->fragmentShader = webtide::skyFragmentWGSL;
    skyMaterial->side       = Side::Back;
    skyMaterial->depthWrite = false;

    auto skyMesh = Mesh::create(BoxGeometry::create(1.0f, 1.0f, 1.0f), skyMaterial);
    skyMesh->scale.setScalar(2000.0f);
    skyMesh->renderOrder = -1;   // render before opaque objects
    scene->add(skyMesh);

    // -------------------------------------------------------------------------
    // Ocean visual settings — exposed via ImGui sliders.
    // Water uniform names MUST be alphabetical to match WgpuRenderer's packing:
    //   choppiness,
    //   contactObj0, contactObj1, contactObj2, contactObj3,  ← xy=worldXZ z=radius
    //   detailStrength,                                       ← x=micro-ripple strength
    //   foamStrength, foamThreshold, fogDensity, fresnelScale,
    //   mieCoeff, mieG, normalStrength, rayleigh, seaColor, specShininess,
    //   tileSize, turbidity, waveScale, wireframe
    // Sky uniform names (alphabetical): mieCoeff, mieG, rayleigh, turbidity
    // -------------------------------------------------------------------------
    bool  uWireframe     = false;
    float uChoppiness    = 0.5f;
    float uDetailStrength= 0.8f;  // detail micro-ripple strength (0 = off, ~1 = nice)
    float uFoamStrength  = 0.35f;
    float uFoamThreshold = 0.30f;
    float uFogDensity    = 0.004f;
    float uFresnelScale  = 0.45f;
    float uNormalStrength= 1.0f;
    float uSeaColor[3]   = {0.10f, 0.19f, 0.22f};
    float uSpecShininess = 120.0f;
    float uWaveScale     = 0.25f;
    float uTimeScale     = 1.0f;   // C++ only — not a shader uniform
    float uLambda        = 1.2f;
    float uFoamDecay     = 1.5f;
    float uSunAzimuth    = 210.0f; // degrees, 0=+X, CCW in XZ plane
    float uSunElevation  = 35.0f;  // degrees above horizon
    // Atmospheric scattering parameters (Preetham model)
    float uTurbidity     = 10.0f;  // 1=clear, 20=very hazy
    float uRayleigh      = 1.0f;   // Rayleigh scattering coefficient
    float uMieCoeff      = 0.005f; // Mie scattering coefficient
    float uMieG          = 0.8f;   // Mie phase asymmetry (0=isotropic, 1=forward)

    // Compute sun direction from azimuth/elevation — used in multiple lambdas.
    auto sunDirection = [&]() -> Vector3 {
        const float az = uSunAzimuth  * math::PI / 180.0f;
        const float el = uSunElevation * math::PI / 180.0f;
        return {std::cos(el) * std::cos(az), std::sin(el), std::cos(el) * std::sin(az)};
    };

    // Push sky params to sky mesh material.
    auto pushSkyUniforms = [&] {
        auto sd = sunDirection();
        skyMaterial->uniforms["mieCoeff"]  = Uniform(uMieCoeff);
        skyMaterial->uniforms["mieG"]      = Uniform(uMieG);
        skyMaterial->uniforms["rayleigh"]  = Uniform(uRayleigh);
        skyMaterial->uniforms["turbidity"] = Uniform(uTurbidity);
        (void)sd; // sky sun dir comes from LightData, not a custom uniform
    };

    auto pushUniforms = [&] {
        waterMaterial->uniforms["choppiness"]    = Uniform(uChoppiness);
        // contactObj0 is updated every frame in the animate loop (buoy follows waves).
        // contactObj1-3 are static placeholders — set non-zero radius to activate.
        waterMaterial->uniforms["contactObj0"]   = Uniform(Color(buoyAnchorX, buoyAnchorZ, kBuoyRadius));
        waterMaterial->uniforms["contactObj1"]   = Uniform(Color(0.f, 0.f, 0.f));
        waterMaterial->uniforms["contactObj2"]   = Uniform(Color(0.f, 0.f, 0.f));
        waterMaterial->uniforms["contactObj3"]   = Uniform(Color(0.f, 0.f, 0.f));
        waterMaterial->uniforms["detailStrength"]= Uniform(uDetailStrength);
        waterMaterial->uniforms["foamStrength"]  = Uniform(uFoamStrength);
        waterMaterial->uniforms["foamThreshold"] = Uniform(uFoamThreshold);
        waterMaterial->uniforms["fogDensity"]    = Uniform(uFogDensity);
        waterMaterial->uniforms["fresnelScale"]  = Uniform(uFresnelScale);
        waterMaterial->uniforms["mieCoeff"]      = Uniform(uMieCoeff);
        waterMaterial->uniforms["mieG"]          = Uniform(uMieG);
        waterMaterial->uniforms["normalStrength"]= Uniform(uNormalStrength);
        waterMaterial->uniforms["rayleigh"]      = Uniform(uRayleigh);
        waterMaterial->uniforms["seaColor"]      = Uniform(Color(uSeaColor[0], uSeaColor[1], uSeaColor[2]));
        waterMaterial->uniforms["specShininess"] = Uniform(uSpecShininess);
        waterMaterial->uniforms["tileSize"]      = Uniform(C1_TILE);
        waterMaterial->uniforms["turbidity"]     = Uniform(uTurbidity);
        waterMaterial->uniforms["waveScale"]     = Uniform(uWaveScale);
        waterMaterial->uniforms["wireframe"]     = Uniform(uWireframe ? 1.0f : 0.0f);
    };
    pushUniforms();
    pushSkyUniforms();

    auto updateSunDirection = [&] {
        auto sd = sunDirection();
        dirLight->position.set(sd.x, sd.y, sd.z);
        // Sky mesh sun dir is read from LightData (dirDirection0) automatically,
        // but sky atmosphere params need an explicit push when sliders change.
        pushSkyUniforms();
    };
    updateSunDirection();

    // Animation state
    float elapsedSeconds = 60.0f;
    auto lastTime = std::chrono::high_resolution_clock::now();

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Once);
        ImGui::SetNextWindowSize({300, 0}, ImGuiCond_Once);
        ImGui::Begin("Ocean Settings");

        bool changed = false;
        bool sunChanged = false;

        if (ImGui::CollapsingHeader("Waves", ImGuiTreeNodeFlags_DefaultOpen)) {
            changed |= ImGui::SliderFloat("Height",      &uWaveScale,   0.0f,  1.5f);
            changed |= ImGui::SliderFloat("Choppiness",  &uChoppiness,  0.0f,  2.0f);
            changed |= ImGui::SliderFloat("Speed",       &uTimeScale,   0.0f,  3.0f);
        }

        if (ImGui::CollapsingHeader("Appearance", ImGuiTreeNodeFlags_DefaultOpen)) {
            changed |= ImGui::ColorEdit3 ("Sea Colour",    uSeaColor);
            changed |= ImGui::SliderFloat("Normal Strength",&uNormalStrength,  0.0f, 4.0f);
            changed |= ImGui::SliderFloat("Micro-ripples",  &uDetailStrength,  0.0f, 3.0f);
            changed |= ImGui::SliderFloat("Fresnel",        &uFresnelScale,    0.0f, 1.5f);
            changed |= ImGui::SliderFloat("Specular",       &uSpecShininess,   4.0f, 256.0f);
        }

        if (ImGui::CollapsingHeader("Sun & Sky", ImGuiTreeNodeFlags_DefaultOpen)) {
            sunChanged |= ImGui::SliderFloat("Azimuth°",   &uSunAzimuth,   0.0f,  360.0f);
            sunChanged |= ImGui::SliderFloat("Elevation°", &uSunElevation, 5.0f,   85.0f);
            changed    |= ImGui::SliderFloat("Turbidity",  &uTurbidity,    1.0f,   20.0f);
            changed    |= ImGui::SliderFloat("Rayleigh",   &uRayleigh,     0.0f,    4.0f);
            changed    |= ImGui::SliderFloat("Mie Coeff",  &uMieCoeff,     0.001f,  0.1f);
            changed    |= ImGui::SliderFloat("Mie G",      &uMieG,         0.0f,    0.99f);
        }

        if (ImGui::CollapsingHeader("Foam", ImGuiTreeNodeFlags_DefaultOpen)) {
            changed |= ImGui::SliderFloat("Threshold", &uFoamThreshold, 0.0f, 1.0f);
            changed |= ImGui::SliderFloat("Strength",  &uFoamStrength,  0.0f, 1.0f);
            changed |= ImGui::SliderFloat("Lambda",    &uLambda,        0.3f, 4.0f);
            changed |= ImGui::SliderFloat("Decay",     &uFoamDecay,     0.1f, 5.0f);
        }

        if (ImGui::CollapsingHeader("Atmosphere")) {
            changed |= ImGui::SliderFloat("Fog Density", &uFogDensity, 0.0f, 0.02f);
        }

        if (ImGui::CollapsingHeader("Buoy")) {
            ImGui::SliderFloat("Anchor X", &buoyAnchorX, -80.f, 80.f);
            ImGui::SliderFloat("Anchor Z", &buoyAnchorZ, -80.f, 80.f);
            ImGui::LabelText("Wave Y", "%.3f m", buoyWaveY);
        }

        if (ImGui::CollapsingHeader("Debug")) {
            // Wireframe drawn inside the WGSL fragment shader using fwidth() grid lines.
            // The displaced surface itself is shown — grid animates with the waves.
            if (ImGui::Checkbox("Wireframe", &uWireframe)) { pushUniforms(); }
        }

        if (changed)    { pushUniforms(); pushSkyUniforms(); }
        if (sunChanged) updateSunDirection();
        ImGui::End();
    });

    IOCapture capture;
    capture.preventMouseEvent = [&] {
        // Prevent canvas orbit controls from interfering with ImGui interactions
        return ImGui::GetIO().WantCaptureMouse;
    };
    capture.preventScrollEvent = [&] {
        // Prevent canvas orbit controls from zooming when ImGui scrolls
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&capture);

    canvas.animate([&] {
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;
        elapsedSeconds += dt * uTimeScale;

        // Generate dynamic spectra for all 3 cascades
        dynSpec0.generate(elapsedSeconds);
        dynSpec1.generate(elapsedSeconds);
        dynSpec2.generate(elapsedSeconds);

        // IFFT all cascade outputs to spatial domain
        ifft0.applyToTexture(dynSpec0.ht,          height0);
        ifft0.applyToTexture(dynSpec0.dht,         gradient0);
        ifft0.applyToTexture(dynSpec0.displacement, displacement0);

        ifft1.applyToTexture(dynSpec1.ht,          height1);
        ifft1.applyToTexture(dynSpec1.dht,         gradient1);
        ifft1.applyToTexture(dynSpec1.displacement, displacement1);

        ifft2.applyToTexture(dynSpec2.ht,          height2);
        ifft2.applyToTexture(dynSpec2.dht,         gradient2);
        ifft2.applyToTexture(dynSpec2.displacement, displacement2);

        // GPU readback — all 3 cascade heights, displacements, and gradients.
        // Runs after IFFT passes so spatial textures are fully written.
        {
            auto b = buoySampler.sample(
                height0, height1, height2,
                displacement0, displacement1, displacement2,
                gradient1, gradient2,
                buoyAnchorX, buoyAnchorZ,
                C0_TILE, C1_TILE, C2_TILE,
                uChoppiness, uWaveScale);

            buoyWaveY = b.y;

            // First frame: snap physics to GPU value to avoid catch-up pop.
            if (buoyFirstFrame) {
                buoyPhysY    = b.y;
                buoyFirstFrame = false;
            }

            // Clamp dt to 50 ms — avoids instability on first frames or after
            // a window pause where dt can be arbitrarily large.
            const float pdt = std::min(dt, 0.05f);

            // --- Vertical spring-mass (semi-implicit Euler) ---
            // restY = wave surface = sphere centre for 50 % submersion.
            const float springF = kBuoySpring  * (b.y    - buoyPhysY);
            const float dampF   = kBuoyDamping * buoyVelY;
            buoyVelY  += ((springF - dampF) / kBuoyMass) * pdt;
            buoyPhysY += buoyVelY * pdt;

            // --- Orientation: align with wave-surface normal, smooth lag ---
            // Normal = normalize(-gx, 1, -gz)  →  pitch=atan(-gz), roll=atan(-gx)
            const float alpha = 1.f - std::exp(-8.f * pdt); // τ ≈ 0.125 s
            buoyPitch += (std::atan(-b.gz) - buoyPitch) * alpha;
            buoyRoll  += (std::atan(-b.gx) - buoyRoll)  * alpha;

            // XZ position: anchor + full 3-cascade displaced offset (matches vertex shader)
            buoyMesh->position.set(buoyAnchorX + b.dx, buoyPhysY, buoyAnchorZ + b.dz);
            buoyMesh->rotation.set(buoyPitch, 0.f, buoyRoll);

            // Keep contact foam ring centred on the actual (displaced) buoy position.
            waterMaterial->uniforms["contactObj0"] = Uniform(Color(
                buoyMesh->position.x, buoyMesh->position.z, kBuoyRadius));
        }

        // Update Jacobian foam — combined 3-cascade Jacobian for realistic wave breaking.
        // J_total = J0 + J1 + J2 captures breaking driven by swell modulating chop.
        oceanFoam.update(dynSpec0, ifft0, dynSpec1, ifft1, dynSpec2, ifft2,
                         dt, uLambda, uFoamDecay,
                         C0_TILE, C1_TILE, C2_TILE);

        // Update foamMap pointer after ping-pong swap (before renderer.render())
        waterMaterial->customTextures["foamMap"] = &oceanFoam.currentFoam();

        controls.update();

        // Snap tile grid to orbit target — high-detail tiles always surround the
        // point the viewer is looking at, not where the camera physically is.
        updateTilePositions();

        // Sky box follows camera so it is always around the viewer.
        skyMesh->position.copy(camera->position);

        ui.render();
        renderer.render(*scene, *camera);
    });

    return 0;
}
