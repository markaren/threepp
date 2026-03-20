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
//   - Tropical sunny day skybox from the BabylonJS Asset Library
//   - Sand texture by Engin Akyurt

#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/loaders/CubeTextureLoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
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

    constexpr uint32_t textureSize = 512;
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
    renderer.setClearColor(Color::black);// Will be fully covered by skybox

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

    // Set custom uniforms (ocean params at binding 2)
    // Uniform names are alphabetical to match WgpuRenderer's packing order:
    //   foamStrength, foamThreshold, fogDensity, seaColor, tileSize, waveScale
    waterMaterial->uniforms["tileSize"] = Uniform(C1_TILE);

    // Bind GPU textures — MUST be alphabetical for deterministic binding numbers.
    // cascade0Displacement(3,4), cascade0Gradient(5,6), cascade0Height(7,8)
    // cascade1Displacement(9,10), cascade1Gradient(11,12), cascade1Height(13,14)
    // cascade2Displacement(15,16), cascade2Gradient(17,18), cascade2Height(19,20)
    // foamMap(21,22), reflectionMap(23,24)
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

    for (int x = -tileRadius; x <= tileRadius; x++) {
        for (int z = -tileRadius; z <= tileRadius; z++) {
            int dist = std::max(std::abs(x), std::abs(z));// Chebyshev distance
            auto& geo = dist <= 1 ? geoInner : dist <= 3 ? geoMid
                                                         : geoOuter;
            auto waterMesh = Mesh::create(geo, waterMaterial);
            waterMesh->position.set(
                    static_cast<float>(x) * C1_TILE,
                    0,
                    static_cast<float>(z) * C1_TILE);
            scene->add(waterMesh);
        }
    }

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

    // Skybox (Tropical Sunny Day) – 6 individual textures mapped to box faces
    std::filesystem::path skyboxPath(std::string(DATA_FOLDER) + "/textures/skybox");

    TextureLoader skyTexLoader;
    auto texPx = skyTexLoader.load((skyboxPath / "TropicalSunnyDay_px.jpg").string());
    auto texNx = skyTexLoader.load((skyboxPath / "TropicalSunnyDay_nx.jpg").string());
    auto texPy = skyTexLoader.load((skyboxPath / "TropicalSunnyDay_py.jpg").string());
    auto texNy = skyTexLoader.load((skyboxPath / "TropicalSunnyDay_ny.jpg").string());
    auto texPz = skyTexLoader.load((skyboxPath / "TropicalSunnyDay_pz.jpg").string());
    auto texNz = skyTexLoader.load((skyboxPath / "TropicalSunnyDay_nz.jpg").string());

    auto skyboxGeo = BoxGeometry::create(1800.f, 1800.f, 1800.f);

    auto matPx = MeshBasicMaterial::create();
    matPx->map = texPx;
    matPx->side = Side::Back;

    auto matNx = MeshBasicMaterial::create();
    matNx->map = texNx;
    matNx->side = Side::Back;

    auto matPy = MeshBasicMaterial::create();
    matPy->map = texPy;
    matPy->side = Side::Back;

    auto matNy = MeshBasicMaterial::create();
    matNy->map = texNy;
    matNy->side = Side::Back;

    auto matPz = MeshBasicMaterial::create();
    matPz->map = texPz;
    matPz->side = Side::Back;

    auto matNz = MeshBasicMaterial::create();
    matNz->map = texNz;
    matNz->side = Side::Back;

    std::vector<std::shared_ptr<Material>> skyboxMats{matPx, matNx, matPy, matNy, matPz, matNz};
    auto skybox = Mesh::create(skyboxGeo, skyboxMats);
    scene->add(skybox);

    // Create GPU cubemap from skybox images for water reflections
    auto& faceImg = texPx->image();
    WgpuTexture reflectionMap(renderer, faceImg.width, faceImg.height,
                              WgpuTexture::Format::RGBA8Unorm, WgpuTexture::Dimension::Cube);
    // Write each face (+X, -X, +Y, -Y, +Z, -Z)
    auto& dataPx = texPx->image().data<unsigned char>();
    auto& dataNx = texNx->image().data<unsigned char>();
    auto& dataPy = texPy->image().data<unsigned char>();
    auto& dataNy = texNy->image().data<unsigned char>();
    auto& dataPz = texPz->image().data<unsigned char>();
    auto& dataNz = texNz->image().data<unsigned char>();
    reflectionMap.writeFace(0, dataPx.data(), dataPx.size());
    reflectionMap.writeFace(1, dataNx.data(), dataNx.size());
    reflectionMap.writeFace(2, dataPy.data(), dataPy.size());
    reflectionMap.writeFace(3, dataNy.data(), dataNy.size());
    reflectionMap.writeFace(4, dataPz.data(), dataPz.size());
    reflectionMap.writeFace(5, dataNz.data(), dataNz.size());

    waterMaterial->customTextures["reflectionMap"] = &reflectionMap;

    // -------------------------------------------------------------------------
    // Ocean visual settings — exposed via ImGui sliders.
    // Uniform names MUST be alphabetical to match the WgpuRenderer's packing order:
    //   choppiness, foamStrength, foamThreshold, fogDensity, fresnelScale,
    //   normalStrength, seaColor, specShininess, tileSize, waveScale
    // -------------------------------------------------------------------------
    float uChoppiness    = 0.5f;   // horizontal displacement scale
    float uFoamStrength  = 0.35f;
    float uFoamThreshold = 0.30f;  // Jacobian foam threshold (lower = more foam)
    float uFogDensity    = 0.004f;
    float uFresnelScale  = 0.45f;  // Fresnel reflection multiplier
    float uNormalStrength= 1.0f;   // overall normal map intensity
    float uSeaColor[3]   = {0.10f, 0.19f, 0.22f};// seascape deep-water blue
    float uSpecShininess = 120.0f;  // Blinn-Phong exponent
    float uWaveScale     = 0.25f;   // vertical amplitude (matches old hardcoded 0.5)
    float uTimeScale     = 1.0f;   // C++ only — not a shader uniform
    float uLambda        = 1.2f;   // Jacobian choppiness multiplier
    float uFoamDecay     = 1.5f;   // foam fade per second
    float uSunAzimuth    = 210.0f; // degrees, 0=+X, CCW in XZ plane
    float uSunElevation  = 35.0f;  // degrees above horizon

    auto pushUniforms = [&] {
        waterMaterial->uniforms["choppiness"]    = Uniform(uChoppiness);
        waterMaterial->uniforms["foamStrength"]  = Uniform(uFoamStrength);
        waterMaterial->uniforms["foamThreshold"] = Uniform(uFoamThreshold);
        waterMaterial->uniforms["fogDensity"]    = Uniform(uFogDensity);
        waterMaterial->uniforms["fresnelScale"]  = Uniform(uFresnelScale);
        waterMaterial->uniforms["normalStrength"]= Uniform(uNormalStrength);
        waterMaterial->uniforms["seaColor"]      = Uniform(Color(uSeaColor[0], uSeaColor[1], uSeaColor[2]));
        waterMaterial->uniforms["specShininess"] = Uniform(uSpecShininess);
        waterMaterial->uniforms["tileSize"]      = Uniform(C1_TILE);
        waterMaterial->uniforms["waveScale"]     = Uniform(uWaveScale);
    };
    pushUniforms();

    auto updateSunDirection = [&] {
        const float az  = uSunAzimuth  * math::PI / 180.0f;
        const float el  = uSunElevation * math::PI / 180.0f;
        dirLight->position.set(
            std::cos(el) * std::cos(az),
            std::sin(el),
            std::cos(el) * std::sin(az));
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
            changed |= ImGui::SliderFloat("Normal Strength",&uNormalStrength, 0.0f, 4.0f);
            changed |= ImGui::SliderFloat("Fresnel",       &uFresnelScale,   0.0f, 1.5f);
            changed |= ImGui::SliderFloat("Specular",      &uSpecShininess,  4.0f, 256.0f);
        }

        if (ImGui::CollapsingHeader("Sun", ImGuiTreeNodeFlags_DefaultOpen)) {
            sunChanged |= ImGui::SliderFloat("Azimuth°",   &uSunAzimuth,   0.0f, 360.0f);
            sunChanged |= ImGui::SliderFloat("Elevation°", &uSunElevation, 5.0f,  85.0f);
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

        if (changed)    pushUniforms();
        if (sunChanged) updateSunDirection();
        ImGui::End();
    });

    IOCapture capture;
    capture.preventMouseEvent = [&] {
        // Prevent canvas orbit controls from interfering with ImGui interactions
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
        }

        // Update Jacobian foam (cascade 1 drives the breaking waves)
        oceanFoam.update(dynSpec1, ifft1, dt, uLambda, uFoamDecay);

        // Update foamMap pointer after ping-pong swap (before renderer.render())
        waterMaterial->customTextures["foamMap"] = &oceanFoam.currentFoam();

        controls.update();
        ui.render();
        renderer.render(*scene, *camera);
    });

    return 0;
}
