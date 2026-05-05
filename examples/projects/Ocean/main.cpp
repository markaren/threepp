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
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/DisplacedMesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    constexpr float kTileSize    = 40.0f;   // metres — matches default DisplacedMesh::Params.tileSize0
    constexpr uint32_t kFftSize  = 256;
    constexpr float kPlaneEdge   = kTileSize;  // mesh extends one full FFT tile in X and Z
    constexpr int   kSubdiv      = static_cast<int>(kFftSize) - 1;

    auto makeOceanMaterial() {
        auto mat = MeshPhysicalMaterial::create();
        mat->color = Color(0.05f, 0.20f, 0.30f);
        mat->roughness = 0.05f;
        mat->metalness = 0.0f;
        mat->setIor(1.33f);
        mat->transmission = 1.0f;
        mat->thickness = 0.5f;
        // Aggressive absorption — open ocean visibility ~3 m. Without this the
        // path tracer sees right through to the floor and the surface looks
        // like a wet sand dune. Ocean-blue tint via attenuationColor (Beer-
        // Lambert): light has to travel ~3 m through water before being tinted
        // to half intensity at this wavelength.
        mat->attenuationColor = Color(0.10f, 0.45f, 0.55f);
        mat->attenuationDistance = 3.0f;
        mat->clearcoat = 1.0f;
        mat->clearcoatRoughness = 0.05f;
        return mat;
    }

    auto makeSandMaterial() {
        return MeshStandardMaterial::create({
                {"color", Color(0.78f, 0.68f, 0.45f)},
                {"roughness", 0.95f},
        });
    }

}// namespace

int main() {

    Canvas canvas("Vulkan PT — Ocean", {{"vsync", false}, {"size", WindowSize{1600, 900}}});
    VulkanRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.0f;

    Scene scene;
    scene.background = Color(0.55f, 0.72f, 0.95f);

    // Sun-like directional light. Photon mapping in the PT will produce focused
    // caustics on the sand floor as light refracts through wave crests.
    auto sun = DirectionalLight::create(Color(1.0f, 0.95f, 0.85f), 4.0f);
    sun->position.set(2.f, 5.f, 2.f);
    Object3D sunTarget;
    sunTarget.position.set(0.f, 0.f, 0.f);
    sun->setTarget(sunTarget);
    scene.add(sun);

    auto ambient = AmbientLight::create(Color(0.55f, 0.72f, 0.95f), 0.25f);
    scene.add(ambient);

    // Sand floor several metres below the surface — gives the refracted path
    // something to land on AND enough water column for the absorption tint to
    // build up. With 3 m attenuation distance and a 12 m column, light loses
    // ~98% of its red component before hitting the floor → ocean blue.
    // Slightly larger than the ocean tile so the rim doesn't escape.
    auto floor = Mesh::create(PlaneGeometry::create(kPlaneEdge * 2.f, kPlaneEdge * 2.f),
                              makeSandMaterial());
    floor->rotation.x = -math::PI / 2.f;
    floor->position.y = -12.f;
    scene.add(floor);

    // Ocean surface. PlaneGeometry with kSubdiv segments → kFftSize²
    // vertices. The DisplacedMesh detects the grid dimension at first-frame
    // init and runs the FFT/displace pipeline against it.
    auto oceanGeo = PlaneGeometry::create(kPlaneEdge, kPlaneEdge, kSubdiv, kSubdiv);
    oceanGeo->rotateX(-math::PI / 2.f);
    auto oceanMat = makeOceanMaterial();
    auto ocean = DisplacedMesh::create(oceanGeo, oceanMat);
    ocean->params.tileSize0   = kTileSize;
    ocean->params.windTheta   = 0.6f;       // wind slightly off the X axis
    ocean->params.windSpeed   = 14.0f;      // moderate breeze (m/s)
    ocean->params.waveScale   = 1.0f;
    ocean->params.textureSize = kFftSize;
    scene.add(ocean);

    PerspectiveCamera camera(45.f, canvas.aspect(), 0.1f, 200.f);
    camera.position.set(8.f, 4.f, 12.f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, 0.f, 0.f);
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
