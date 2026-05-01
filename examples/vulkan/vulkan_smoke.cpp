// Vulkan renderer smoke test (Phase 7: env map background + mirror IBL).
//
// Walks a real threepp Scene, builds one BLAS per BufferGeometry, and a TLAS
// with one instance per Mesh. Closest-hit reads per-mesh material plus the
// scene's AmbientLight + DirectionalLight from a per-frame UBO. Phase 7 adds
// an HDR equirect environment: the primary miss samples it for backgrounds
// and the closest-hit fires one mirror-reflection probe ray for spec IBL,
// so the smooth metal box now reflects the sky.

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/canvas/Canvas.hpp"
#include "threepp/controls/OrbitControls.hpp"
#include "threepp/core/Clock.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/scenes/Scene.hpp"

#include <iostream>

using namespace threepp;

int main() {

    Canvas::Parameters params;
    params.title("Vulkan smoke (Phase 10: + alpha-test cutout)")
            .size(800, 600);

    Canvas canvas(params);

    VulkanRenderer renderer(canvas);

    auto scene = Scene::create();

    // Phase 7 — load an HDR equirect and assign it to both background (so
    // primary misses sample it) and environment (so closest-hit's mirror IBL
    // probe sees it). Same convention as wgpu_raytracer.cpp.
    RGBELoader rgbe;
    auto envMap = rgbe.load(std::string(DATA_FOLDER) +
                            "/textures/env/citrus_orchard_road_puresky_2k.hdr");
    scene->background = envMap;
    scene->environment = envMap;

    // One geometry, two instances at different world positions: confirms
    // both BLAS reuse (cache hit) and the per-instance transform path.
    // Distinct materials exercise the binding-4 MaterialDesc path: a rough
    // dielectric red vs a smooth blue metal makes the GGX/Schlick contrast
    // obvious vs the Phase 5a Lambert-only output.
    auto boxGeom = BoxGeometry::create(1.0f, 1.0f, 1.0f);

    auto matA = MeshStandardMaterial::create();
    matA->color.setHex(0xcc3333);
    matA->roughness = 0.85f;
    matA->metalness = 0.0f;
    auto boxA = Mesh::create(boxGeom, matA);
    boxA->position.set(-1.0f, 0.0f, 0.0f);
    scene->add(boxA);

    auto matB = MeshStandardMaterial::create();
    matB->color.setHex(0x3366cc);
    matB->roughness = 0.2f;
    matB->metalness = 1.0f;
    auto boxB = Mesh::create(boxGeom, matB);
    boxB->position.set(1.0f, 0.0f, 0.0f);
    boxB->rotation.set(0.4f, 0.8f, 0.0f);
    scene->add(boxB);

    // Ground plane catches shadows from both boxes so Phase 6b's shadow-ray
    // path is obvious. Rough light-grey dielectric so contact shadows read clearly.
    auto groundGeom = PlaneGeometry::create(20.0f, 20.0f);
    auto groundMat = MeshStandardMaterial::create();
    groundMat->color.setHex(0xaaaaaa);
    groundMat->roughness = 0.9f;
    groundMat->metalness = 0.0f;
    auto ground = Mesh::create(groundGeom, groundMat);
    ground->rotation.x = -math::PI / 2.0f;
    ground->position.y = -1.5f;
    scene->add(ground);

    // Textured box behind the others — exercises the UV ingest path
    // + bindless albedo sampler array. Loaded with SRGBColorSpace so the
    // VK_FORMAT_R8G8B8A8_SRGB view does the linear decode in hardware.
    TextureLoader texLoader;
    auto albedoTex = texLoader.load(
            std::string(DATA_FOLDER) + "/textures/uv_grid_opengl.jpg",
            SRGBColorSpace);
    auto texturedMat = MeshStandardMaterial::create();
    texturedMat->color.setHex(0xffffff);// neutral so the texture reads cleanly
    texturedMat->roughness = 0.6f;
    texturedMat->metalness = 0.0f;
    texturedMat->map = albedoTex;
    auto texturedBox = Mesh::create(boxGeom, texturedMat);
    texturedBox->position.set(0.0f, 0.0f, -1.5f);
    scene->add(texturedBox);

    // Small emissive cube floating above the ground — self-lit bright object
    // independent of sun/env. Validates closest_hit's emissive contribution.
    auto emissiveGeom = BoxGeometry::create(0.4f, 0.4f, 0.4f);
    auto emissiveMat = MeshStandardMaterial::create();
    emissiveMat->color.setHex(0x222222);
    emissiveMat->roughness = 1.0f;
    emissiveMat->metalness = 0.0f;
    emissiveMat->emissive = Color::green;
    emissiveMat->emissiveIntensity = 6.0f;
    auto emissiveBox = Mesh::create(emissiveGeom, emissiveMat);
    emissiveBox->position.set(0.0f, 0.6f, 0.0f);
    scene->add(emissiveBox);

    // Alpha-test cutout — upright plane with a transparent-PNG texture
    // exercises the any-hit shader. Cutout material drops every texel below
    // alphaTest, so the three.js logo (alpha-blended in source) becomes a
    // hard-edged silhouette through which both primary rays and shadow rays
    // pass cleanly. Test the plane's shadow on the ground to confirm
    // shadow-ray any-hit also fires.
    auto cutoutTex = texLoader.load(
            std::string(DATA_FOLDER) + "/textures/three.png",
            SRGBColorSpace);
    auto cutoutMat = MeshStandardMaterial::create();
    cutoutMat->color.setHex(0xffffff);
    cutoutMat->roughness = 0.6f;
    cutoutMat->metalness = 0.0f;
    cutoutMat->map = cutoutTex;
    cutoutMat->alphaTest = 0.5f;
    cutoutMat->side = Side::Double;// any-hit must fire from both sides
    auto cutoutGeom = PlaneGeometry::create(1.6f, 1.6f);
    auto cutoutPlane = Mesh::create(cutoutGeom, cutoutMat);
    cutoutPlane->position.set(2.4f, -0.2f, 0.5f);
    cutoutPlane->rotation.y = -0.4f;
    scene->add(cutoutPlane);

    auto sun = DirectionalLight::create(Color(0xffffff), 3.0f);
    sun->position.set(0.4f, 1.0f, 0.3f);// matches the prior hard-coded sun
    scene->add(sun);

    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 100.f);
    camera->position.set(0, 1.5f, 4.f);
    camera->lookAt(0, 0, 0);

    OrbitControls controls{*camera, canvas};

    canvas.onWindowResize([&](const WindowSize&) {
        camera->aspect = canvas.aspect();
        camera->updateProjectionMatrix();
    });

    Clock clock;
    canvas.animate([&] {
        const auto dt = clock.getDelta();
        // boxA->rotation.y += 0.1f * dt;
        renderer.render(*scene, *camera);
    });

    return 0;
}
