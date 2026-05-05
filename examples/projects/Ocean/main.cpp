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
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/DisplacedMesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    constexpr float kTileSize    = 1000.0f;  // metres — full mesh extent and cascade-0 tile
    constexpr uint32_t kFftSize  = 512;      // 4× FFT compute vs 256² but halves vertex spacing
    constexpr float kPlaneEdge   = kTileSize;  // mesh extends one full FFT tile in X and Z
    constexpr int   kSubdiv      = static_cast<int>(kFftSize) - 1;  // 512² verts → 1.95 m spacing at 1000 m

    auto makeOceanMaterial() {
        auto mat = MeshPhysicalMaterial::create();
        // Pure water has no diffuse pigment — the blue comes from Beer-Lambert
        // absorption through the medium, not albedo.
        mat->color = Color::white;
        // Small roughness simulates the sub-pixel chop the FFT can't resolve.
        // 0.02 keeps the glitter tight (sharper specular highlights, fewer
        // distributed sparkles) — denoiser-off output reads as "specular
        // glints on water" rather than "scattered noise".
        mat->roughness = 0.02f;
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
        mat->attenuationColor = Color(0.10f, 0.45f, 0.55f);
        mat->attenuationDistance = 3.0f;
        // mat->clearcoat = 0.1;
        // No clearcoat: with roughness=0 the base specular is already a clean
        // delta lobe and Fresnel handles reflect/refract via the transmission
        // BSDF. Stacking a clearcoat lobe on top was the dominant noise
        // source on the surface (per-pixel variance from sampling both lobes).
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

    Canvas canvas("Vulkan PT — Ocean", {{"vsync", false}, {"size", WindowSize{1600, 900}}});
    VulkanRenderer renderer(canvas);
    renderer.setDenoise(false);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    // puresky_2k.hdr is a very bright daylight env. ACES desaturates strongly
    // at high luminance so the sand goes white-ish even at 0.5; 0.3 keeps
    // the sand in the saturated regime so its tan colour shows through and
    // the water's blue-tinted depth gradient can be read against it.
    renderer.toneMappingExposure = 0.5f;

    RGBELoader rgbe;
    auto env = rgbe.load(std::string(DATA_FOLDER) +
                         "/textures/env/citrus_orchard_road_puresky_2k.hdr");

    Scene scene;
    scene.background = env;
    scene.environment = env;

    // Sun-like directional light. The HDR env already contains a sun (env
    // CDF + MIS will importance-sample it), so the directional is mostly
    // here to drive the photon-mapping caustics pass — kept gentle so it
    // doesn't double up with the env's own sun on the surface.
    auto sun = DirectionalLight::create(Color(1.0f, 0.95f, 0.85f), 1.0f);
    sun->position.set(2.f, 5.f, 2.f);
    Object3D sunTarget;
    sunTarget.position.set(0.f, 0.f, 0.f);
    sun->setTarget(sunTarget);
    scene.add(sun);

    auto ambient = AmbientLight::create(Color(0.55f, 0.72f, 0.95f), 0.25f);
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
    // Two-cascade FFT (cascade 2 disabled):
    //   tileSize0 = 1000 m → big swells, dominant macro shapes.
    //   tileSize1 =  100 m → mid-frequency waves filling each swell face.
    // Cascade 2 was emitting wavelengths below the 1.95 m mesh resolving
    // limit, which displayed as random per-vertex displacement noise that
    // the denoiser couldn't filter (legitimately high-frequency, edge-
    // preserving). Dropping it gives a cleaner raw image; the lost detail
    // band is below where the mesh could honestly display it anyway.
    ocean->params.tileSize0   = kTileSize;
    ocean->params.tileSize1   = 100.0f;
    ocean->params.tileSize2   = 0.0f;
    ocean->params.windTheta   = 0.6f;       // wind slightly off the X axis
    ocean->params.windSpeed   = 20.0f;      // fresh breeze tuned to the larger 1 km extent
    ocean->params.waveScale   = 1.0f;
    ocean->params.choppiness  = 0.55f;      // sharper crests, more visible wave-folding
    ocean->params.textureSize = kFftSize;
    scene.add(ocean);

    // Far-clip raised so the horizon doesn't get cut at the elevated/distant
    // camera. Position is "drone over open water" — well above the surface
    // and far enough out that the rim of the water tile sits beyond the
    // frustum, leaving sky/horizon to fill the upper half of the frame.
    PerspectiveCamera camera(45.f, canvas.aspect(), 0.1f, 2000.f);
    camera.position.set(60.f, 12.f, 100.f);
    OrbitControls controls{camera, canvas};
    controls.target.set(-300.f, 1.f, -300.f);// look outward across the surface
    controls.update();

    float waveScale = ocean->params.waveScale;
    float choppiness = ocean->params.choppiness;
    float windSpeed = ocean->params.windSpeed;
    float windTheta = ocean->params.windTheta;
    float exposure  = renderer.toneMappingExposure;
    int   toneMode  = static_cast<int>(renderer.toneMapping);
    float fps = 0.f, fpsAccum = 0.f;
    int   fpsFrames = 0;

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({340, 0});
        ImGui::Begin("Vulkan PT — Ocean");
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Separator();
        ImGui::TextWrapped(
            "FFT-displaced surface (single-cascade Phillips, %u² IFFT, "
            "%.0f m tile). Path-traced refraction + photon-map caustics.",
            kFftSize, kTileSize);
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
        const float dt = clock.getDelta();
        fpsAccum += dt;
        ++fpsFrames;
        if (fpsAccum >= 0.5f) {
            fps = fpsFrames / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }
        controls.update();
        renderer.render(scene, camera);
        ui.render();
    });

    return 0;
}
