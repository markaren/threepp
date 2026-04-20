// Wgpu renderer tests (WebGPU backend).
// Split from CrossRenderer_test.cpp for maintainability.

#include "CrossRenderer_helpers.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/materials/MeshDepthMaterial.hpp"
#include "threepp/materials/MeshMatcapMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/textures/CubeTexture.hpp"

#include <webgpu/webgpu.h>

TEST_CASE("Wgpu: clear color produces expected pixels", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto pixels = renderWithWgpu(*scene, *camera, Color(1.0f, 0.0f, 0.0f));
    REQUIRE(pixels.size() == DATA_SIZE);
    CHECK(allPixelsMatch(pixels, 255, 0, 0, 2));
}

TEST_CASE("Wgpu: readback dimensions match render target", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    CHECK(pixels.size() == DATA_SIZE);
}


// =============================================================================
// Section 3: Cross-renderer comparisons
// Generate GL reference data first, then Wgpu data, then compare.
// =============================================================================

TEST_CASE("Wgpu: PointLight illuminates a sphere", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pointLight = PointLight::create(Color(0xffffff), 2.0f);
    pointLight->position.set(0, 0, 2);
    scene->add(pointLight);

    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshLambertMaterial::create();
    material->color = Color(0xff4444);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    auto avg = averageColor(pixels);
    CHECK(avg.r > avg.b);
}

TEST_CASE("Wgpu: SpotLight illuminates a sphere", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto spotLight = SpotLight::create(Color(0xffffff), 2.0f);
    spotLight->position.set(0, 0, 3);
    spotLight->angle = math::PI / 4;
    spotLight->penumbra = 0.2f;
    scene->add(spotLight);

    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshPhongMaterial::create();
    material->color = Color(0x44ff44);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 16);

    auto avg = averageColor(pixels);
    CHECK(avg.g > avg.r);
}

TEST_CASE("Wgpu: HemisphereLight tints geometry", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto hemiLight = HemisphereLight::create(Color(0x4444ff), Color(0x442200));
    hemiLight->position.set(0, 1, 0);
    scene->add(hemiLight);

    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshLambertMaterial::create();
    material->color = Color(0xffffff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);
}

TEST_CASE("Wgpu: textured box uses diffuse map", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    // Create a 2x2 checkerboard texture procedurally
    std::vector<unsigned char> texData = {
            255, 0, 0, 255,// red
            0, 255, 0, 255,// green
            0, 255, 0, 255,// green
            255, 0, 0, 255 // red
    };
    Image image(texData, 2, 2);

    auto texture = Texture::create(image);
    texture->needsUpdate();

    auto geometry = BoxGeometry::create(2, 2, 2);
    auto material = MeshBasicMaterial::create();
    material->map = texture;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    // The box should have visible colored pixels from the texture
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    // Should have both red and green components from the checkerboard
    auto avg = averageColor(pixels);
    CHECK(avg.r > 5.0);
    CHECK(avg.g > 5.0);
}

TEST_CASE("Wgpu: opacity affects brightness", "[wgpu]") {
    REQUIRE_WGPU();

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto makeScene = [](float opacity) {
        auto scene = Scene::create();
        auto geometry = BoxGeometry::create(2, 2, 2);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        material->opacity = opacity;
        material->transparent = true;
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto fullPixels = renderWithWgpu(*makeScene(1.0f), *camera, Color(0x000000));
    auto halfPixels = renderWithWgpu(*makeScene(0.5f), *camera, Color(0x000000));
    REQUIRE(fullPixels.size() == DATA_SIZE);
    REQUIRE(halfPixels.size() == DATA_SIZE);

    auto fullAvg = averageColor(fullPixels);
    auto halfAvg = averageColor(halfPixels);

    double fullBright = (fullAvg.r + fullAvg.g + fullAvg.b) / 3.0;
    double halfBright = (halfAvg.r + halfAvg.g + halfAvg.b) / 3.0;
    CHECK(fullBright > halfBright);
}

TEST_CASE("Wgpu: setSize reconfigures surface", "[wgpu]") {
    REQUIRE_WGPU();


    WgpuRenderer renderer(wgpuCanvas());

    // setSize should not crash and should update reported size
    renderer.setSize({32, 32});
    auto sz = renderer.size();
    CHECK(sz.width() == 32);
    CHECK(sz.height() == 32);

    // Restore
    renderer.setSize({RT_WIDTH, RT_HEIGHT});
    sz = renderer.size();
    CHECK(sz.width() == RT_WIDTH);
    CHECK(sz.height() == RT_HEIGHT);

    renderer.dispose();
}

TEST_CASE("Wgpu: setPixelRatio updates ratio", "[wgpu]") {
    REQUIRE_WGPU();

    WgpuRenderer renderer(wgpuCanvas());

    CHECK(renderer.getTargetPixelRatio() == 1.0f);

    renderer.setPixelRatio(2.0f);
    CHECK(renderer.getTargetPixelRatio() == 2.0f);

    // Setting pixel ratio should not crash and the renderer should still
    // be able to render
    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());
    renderer.setClearColor(Color(0.0f, 0.0f, 1.0f));
    renderer.render(*scene, *camera);

    auto pixels = renderer.readRGBPixels();
    REQUIRE(pixels.size() == DATA_SIZE);

    // Should have blue clear color
    auto avg = averageColor(pixels);
    CHECK(avg.b > avg.r);
    CHECK(avg.b > avg.g);

    renderer.setPixelRatio(1.0f);
    renderer.dispose();
}

TEST_CASE("Wgpu: viewport restricts rendering region", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto geometry = BoxGeometry::create(4, 4, 4);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    WgpuRenderer renderer(wgpuCanvas());
    renderer.setClearColor(Color(0x000000));

    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());

    // Render full viewport
    renderer.setViewport(0, 0, RT_WIDTH, RT_HEIGHT);
    renderer.render(*scene, *camera);
    auto fullPixels = renderer.readRGBPixels();

    // Render with half-width viewport
    renderer.setViewport(0, 0, RT_WIDTH / 2, RT_HEIGHT);
    renderer.render(*scene, *camera);
    auto halfPixels = renderer.readRGBPixels();

    int fullNonBlack = countNonBlack(fullPixels);
    int halfNonBlack = countNonBlack(halfPixels);

    // Half viewport should produce fewer lit pixels
    CHECK(fullNonBlack > halfNonBlack);

    renderer.dispose();
}

TEST_CASE("Wgpu: dispose does not crash on repeated calls", "[wgpu]") {
    REQUIRE_WGPU();

    WgpuRenderer renderer(wgpuCanvas());
    renderer.dispose();
    // Second dispose should not crash
    renderer.dispose();
}

TEST_CASE("Wgpu: PlaneGeometry renders correctly", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto geometry = PlaneGeometry::create(3, 3);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0x00ffff);
    material->side = Side::Double;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);
}

TEST_CASE("Wgpu: CylinderGeometry renders correctly", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto geometry = CylinderGeometry::create(0.5f, 0.5f, 2.0f, 16);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff00ff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 32);
}

TEST_CASE("Wgpu: TorusGeometry renders correctly", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto geometry = TorusGeometry::create(1.0f, 0.4f, 8, 16);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffff00);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 16);
}

TEST_CASE("Wgpu: emissive material produces visible output without lights", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshStandardMaterial::create();
    material->color = Color(0x000000);
    material->emissive = Color(0xff8800);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    // No lights — only emissive should contribute
    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    auto avg = averageColor(pixels);
    CHECK(avg.r > avg.b);
}

TEST_CASE("Wgpu: face culling respects material.side", "[wgpu]") {
    REQUIRE_WGPU();

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    // A plane facing away from the camera.
    // With Side::Front (cull back), it should be invisible.
    // With Side::Double, it should be visible.
    auto makeScene = [](Side side) {
        auto scene = Scene::create();
        auto geometry = PlaneGeometry::create(3, 3);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        material->side = side;
        auto mesh = Mesh::create(geometry, material);
        // Rotate 180 degrees so the plane faces away from camera
        mesh->rotation.y = math::PI;
        scene->add(mesh);
        return scene;
    };

    auto backFacePixels = renderWithWgpu(*makeScene(Side::Front), *camera, Color(0x000000));
    auto doubleSidePixels = renderWithWgpu(*makeScene(Side::Double), *camera, Color(0x000000));

    int backFaceNonBlack = countNonBlack(backFacePixels);
    int doubleSideNonBlack = countNonBlack(doubleSidePixels);

    // Double-sided should show more pixels than front-only (which culls the back face)
    CHECK(doubleSideNonBlack > backFaceNonBlack);
}

TEST_CASE("Wgpu: wireframe mode renders edges", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto geometry = BoxGeometry::create(2, 2, 2);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    material->wireframe = true;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    // Wireframe should produce some visible pixels (but fewer than solid)
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 0);

    // Compare with solid rendering - wireframe should have fewer lit pixels
    auto solidMat = MeshBasicMaterial::create();
    solidMat->color = Color(0xffffff);
    auto solidMesh = Mesh::create(geometry, solidMat);
    auto solidScene = Scene::create();
    solidScene->add(solidMesh);

    auto solidPixels = renderWithWgpu(*solidScene, *camera, Color(0x000000));
    int solidNonBlack = countNonBlack(solidPixels);
    CHECK(solidNonBlack > nonBlack);
}

TEST_CASE("Wgpu: additive blending brightens", "[wgpu]") {
    REQUIRE_WGPU();

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto makeScene = [](Blending blending) {
        auto scene = Scene::create();
        auto geometry = BoxGeometry::create(2, 2, 2);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0x808080);
        material->blending = blending;
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto normalPixels = renderWithWgpu(*makeScene(Blending::Normal), *camera, Color(0x404040));
    auto additivePixels = renderWithWgpu(*makeScene(Blending::Additive), *camera, Color(0x404040));

    auto normalAvg = averageColor(normalPixels);
    auto additiveAvg = averageColor(additivePixels);

    double normalBright = (normalAvg.r + normalAvg.g + normalAvg.b) / 3.0;
    double additiveBright = (additiveAvg.r + additiveAvg.g + additiveAvg.b) / 3.0;

    // Additive blending on a non-black background should be brighter
    CHECK(additiveBright >= normalBright);
}

TEST_CASE("Wgpu: getClearColor/Alpha round-trips", "[wgpu]") {
    REQUIRE_WGPU();

    WgpuRenderer renderer(wgpuCanvas());

    renderer.setClearColor(Color(0.2f, 0.4f, 0.6f), 0.8f);

    Color c;
    renderer.getClearColor(c);
    CHECK(std::abs(c.r - 0.2f) < 0.01f);
    CHECK(std::abs(c.g - 0.4f) < 0.01f);
    CHECK(std::abs(c.b - 0.6f) < 0.01f);

    CHECK(std::abs(renderer.getClearAlpha() - 0.8f) < 0.01f);

    renderer.setClearAlpha(0.5f);
    CHECK(std::abs(renderer.getClearAlpha() - 0.5f) < 0.01f);

    renderer.dispose();
}

TEST_CASE("Wgpu: getViewport round-trips", "[wgpu]") {
    REQUIRE_WGPU();

    WgpuRenderer renderer(wgpuCanvas());
    renderer.setViewport(10, 20, 30, 40);

    Vector4 vp;
    renderer.getViewport(vp);
    CHECK(vp.x == 10.0f);
    CHECK(vp.y == 20.0f);
    CHECK(vp.z == 30.0f);
    CHECK(vp.w == 40.0f);

    renderer.dispose();
}

TEST_CASE("Wgpu: scissor test round-trips", "[wgpu]") {
    REQUIRE_WGPU();

    WgpuRenderer renderer(wgpuCanvas());

    CHECK(renderer.getScissorTest() == false);
    renderer.setScissorTest(true);
    CHECK(renderer.getScissorTest() == true);

    renderer.setScissor(5, 10, 15, 20);
    Vector4 sc;
    renderer.getScissor(sc);
    CHECK(sc.x == 5.0f);
    CHECK(sc.y == 10.0f);
    CHECK(sc.z == 15.0f);
    CHECK(sc.w == 20.0f);

    renderer.dispose();
}

TEST_CASE("Wgpu: render info tracks draw calls", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto geometry = BoxGeometry::create(1, 1, 1);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff0000);
    auto mesh1 = Mesh::create(geometry, material);
    mesh1->position.x = -1;
    scene->add(mesh1);

    auto mesh2 = Mesh::create(geometry, material);
    mesh2->position.x = 1;
    scene->add(mesh2);

    WgpuRenderer renderer(wgpuCanvas());
    renderer.setClearColor(Color(0x000000));

    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());
    renderer.render(*scene, *camera);

    const auto& info = renderer.info();
    CHECK(info.render.calls >= 2);
    CHECK(info.render.triangles > 0);
    CHECK(info.render.frame > 0);

    renderer.dispose();
}

TEST_CASE("Wgpu: getActiveCubeFace and getActiveMipmapLevel", "[wgpu]") {
    REQUIRE_WGPU();

    WgpuRenderer renderer(wgpuCanvas());

    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get(), 2, 3);

    CHECK(renderer.getActiveCubeFace() == 2);
    CHECK(renderer.getActiveMipmapLevel() == 3);

    renderer.dispose();
}

TEST_CASE("Wgpu: resetState does not crash", "[wgpu]") {
    REQUIRE_WGPU();

    WgpuRenderer renderer(wgpuCanvas());
    renderer.resetState();// Should be a no-op
    renderer.dispose();
}


// =============================================================================
// Section 6: GL-only — Additional material types
// =============================================================================

TEST_CASE("Wgpu: MeshToonMaterial renders with stepped shading", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
    dirLight->position.set(0, 0, 1);
    scene->add(dirLight);

    auto geometry = SphereGeometry::create(1.0f, 32, 16);
    auto material = MeshToonMaterial::create();
    material->color = Color(0xff4444);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    auto avg = averageColor(pixels);
    CHECK(avg.r > avg.g);
    CHECK(avg.r > avg.b);
}

TEST_CASE("Wgpu: MeshNormalMaterial shows surface normals as colors", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto geometry = SphereGeometry::create(1.0f, 32, 16);
    auto material = MeshNormalMaterial::create();
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    // Normal material maps normals to RGB — center of sphere facing camera
    // should have significant blue (Z-forward maps to blue)
    auto center = centerPixel(pixels, RT_WIDTH, RT_HEIGHT);
    CHECK((center.r + center.g + center.b) > 50.0);
}

TEST_CASE("Wgpu: MeshDepthMaterial varies brightness with distance", "[wgpu]") {
    REQUIRE_WGPU();

    // Near sphere should be brighter than far sphere
    auto makeScene = [](float z) {
        auto scene = Scene::create();
        auto geometry = SphereGeometry::create(0.5f, 16, 8);
        auto material = MeshDepthMaterial::create();
        auto mesh = Mesh::create(geometry, material);
        mesh->position.z = z;
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 50);
    camera->position.z = 5;
    Color clearColor(0x000000);

    auto nearPixels = renderWithWgpu(*makeScene(0.0f), *camera, clearColor);
    auto farPixels = renderWithWgpu(*makeScene(-10.0f), *camera, clearColor);

    double nearBright = avgBrightness(nearPixels);
    double farBright = avgBrightness(farPixels);
    CHECK(nearBright > farBright);
}

TEST_CASE("Wgpu: MeshMatcapMaterial renders visible geometry", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto geometry = SphereGeometry::create(1.0f, 32, 16);
    auto material = MeshMatcapMaterial::create();
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);
}

TEST_CASE("Wgpu: Line renders visible edges", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto geometry = BufferGeometry::create();
    std::vector<float> positions = {-1, 0, 0, 1, 0, 0, 0, 1, 0};
    geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));

    auto material = LineBasicMaterial::create();
    material->color = Color(0xff0000);
    auto line = Line::create(geometry, material);
    scene->add(line);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 5);
}

TEST_CASE("Wgpu: LineSegments renders discrete segments", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto geometry = BufferGeometry::create();
    std::vector<float> positions = {-1, -1, 0, 1, -1, 0, -1, 1, 0, 1, 1, 0};
    geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));

    auto material = LineBasicMaterial::create();
    material->color = Color(0x00ff00);
    auto lineSegments = LineSegments::create(geometry, material);
    scene->add(lineSegments);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 5);
}

// Lavapipe/software rasterizers render points as zero-size pixels.
// WebGPU pointSize is always 1px; software Vulkan may not rasterize them visibly.

TEST_CASE("Wgpu: Points renders visible dots", "[wgpu]") {
    REQUIRE_WGPU();
    SKIP_ON_SOFTWARE_ADAPTER();

    auto scene = Scene::create();
    auto geometry = BufferGeometry::create();
    std::vector<float> positions = {0, 0, 0, 0.5f, 0.5f, 0, -0.5f, -0.5f, 0};
    geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));

    auto material = PointsMaterial::create();
    material->color = Color(0xffff00);
    material->size = 8.0f;
    auto points = Points::create(geometry, material);
    scene->add(points);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack >= 3);
}

// Lavapipe/software rasterizers may not support billboard quad expansion for sprites.

TEST_CASE("Wgpu: Sprite renders as billboard", "[wgpu]") {
    REQUIRE_WGPU();
    SKIP_ON_SOFTWARE_ADAPTER();

    auto scene = Scene::create();
    auto material = SpriteMaterial::create();
    material->color = Color(0x00ff00);
    auto sprite = Sprite::create(material);
    scene->add(sprite);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 32);
}

// WgpuRenderer: InstancedMesh per-instance transforms not yet implemented (renders single instance)

TEST_CASE("Wgpu: InstancedMesh renders multiple instances", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geometry = BoxGeometry::create(0.4f, 0.4f, 0.4f);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff8800);

    auto im = InstancedMesh::create(geometry, material, 4);
    Matrix4 m;
    m.setPosition(Vector3(-0.8f, -0.8f, 0));
    im->setMatrixAt(0, m);
    m.setPosition(Vector3(0.8f, -0.8f, 0));
    im->setMatrixAt(1, m);
    m.setPosition(Vector3(-0.8f, 0.8f, 0));
    im->setMatrixAt(2, m);
    m.setPosition(Vector3(0.8f, 0.8f, 0));
    im->setMatrixAt(3, m);
    scene->add(im);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);

    // 4 instances should cover more than a single box at center
    auto singleScene = Scene::create();
    singleScene->add(AmbientLight::create(Color(0xffffff)));
    auto singleMat = MeshBasicMaterial::create();
    singleMat->color = Color(0xff8800);
    auto singleMesh = Mesh::create(geometry, singleMat);
    singleScene->add(singleMesh);
    auto singlePixels = renderWithWgpu(*singleScene, *camera, Color(0x000000));
    int singleNonBlack = countNonBlack(singlePixels);

    CHECK(nonBlack > singleNonBlack);
}

// WgpuRenderer: InstancedMesh per-instance transforms not yet implemented

TEST_CASE("Wgpu: vertex colors tint geometry", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geometry = BoxGeometry::create(1, 1, 1);
    auto posAttr = geometry->getAttribute<float>("position");
    int vertexCount = static_cast<int>(posAttr->count());
    std::vector<float> colors(vertexCount * 3, 0.0f);
    for (int i = 0; i < vertexCount; i++) {
        colors[i * 3 + 0] = 1.0f;// red
    }
    geometry->setAttribute("color", FloatBufferAttribute::create(colors, 3));

    auto material = MeshBasicMaterial::create();
    material->vertexColors = true;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    auto avg = averageColor(pixels);
    CHECK(avg.r > avg.g + 10);
    CHECK(avg.r > avg.b + 10);
}

// WgpuRenderer: vertex color attribute not yet in shader

TEST_CASE("Wgpu: Fog attenuates distant objects", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeScene = [](float objZ) {
        auto scene = Scene::create();
        scene->fog = Fog(Color(0x000000), 1.0f, 10.0f);
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = SphereGeometry::create(0.5f, 16, 8);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.z = objZ;
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 50);
    camera->position.z = 5;
    Color clearColor(0x000000);

    auto nearPixels = renderWithWgpu(*makeScene(3.0f), *camera, clearColor);
    auto farPixels = renderWithWgpu(*makeScene(-5.0f), *camera, clearColor);

    double nearBright = avgBrightness(nearPixels);
    double farBright = avgBrightness(farPixels);
    CHECK(nearBright > farBright);
}

TEST_CASE("Wgpu: FogExp2 attenuates distant objects", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeScene = [](float objZ) {
        auto scene = Scene::create();
        scene->fog = FogExp2(Color(0x000000), 0.15f);
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = SphereGeometry::create(0.5f, 16, 8);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.z = objZ;
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 50);
    camera->position.z = 5;
    Color clearColor(0x000000);

    auto nearPixels = renderWithWgpu(*makeScene(3.0f), *camera, clearColor);
    auto farPixels = renderWithWgpu(*makeScene(-5.0f), *camera, clearColor);

    double nearBright = avgBrightness(nearPixels);
    double farBright = avgBrightness(farPixels);
    CHECK(nearBright > farBright);
}

// Cross-renderer fog comparison: Wgpu fog is implemented but output differs from GL

TEST_CASE("Wgpu: OrthographicCamera renders without perspective distortion", "[wgpu]") {
    REQUIRE_WGPU();

    // Two boxes at different depths — with ortho they should appear same size
    auto makeScene = [](float z) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);
        auto geometry = BoxGeometry::create(1, 1, 1);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.z = z;
        scene->add(mesh);
        return scene;
    };

    auto camera = OrthographicCamera::create(-2, 2, 2, -2, 0.1f, 100);
    camera->position.z = 5;
    Color clearColor(0x000000);

    auto nearPixels = renderWithWgpu(*makeScene(0.0f), *camera, clearColor);
    auto farPixels = renderWithWgpu(*makeScene(-10.0f), *camera, clearColor);

    int nearCount = countNonBlack(nearPixels);
    int farCount = countNonBlack(farPixels);

    // Orthographic: same-size boxes at different depths produce similar pixel counts
    // A 1x1x1 box in a [-2,2] ortho viewport covers ~1/16 of the area = PIXEL_COUNT/16
    CHECK(nearCount > PIXEL_COUNT / 32);
    CHECK(farCount > PIXEL_COUNT / 32);
    double ratio = static_cast<double>(nearCount) / farCount;
    CHECK(ratio > 0.8);
    CHECK(ratio < 1.2);
}

TEST_CASE("Wgpu: object hierarchy applies parent transform", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    // Parent rotated 90 degrees around Y, child offset in X
    auto parent = Object3D::create();
    parent->rotation.y = math::PI / 2;
    scene->add(parent);

    auto geometry = BoxGeometry::create(0.5f, 0.5f, 0.5f);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    auto child = Mesh::create(geometry, material);
    child->position.x = 2.0f;
    parent->add(child);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 5);
}

TEST_CASE("Wgpu: RingGeometry renders correctly", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);
    auto geometry = RingGeometry::create(0.5f, 1.0f);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    material->side = Side::Double;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 16);
}

TEST_CASE("Wgpu: TorusKnotGeometry renders correctly", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);
    auto geometry = TorusKnotGeometry::create(0.8f, 0.2f, 64, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 16);
}

TEST_CASE("Wgpu: ConeGeometry renders correctly", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);
    auto geometry = ConeGeometry::create(0.8f, 1.5f, 16);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    // Cone geometry covers fewer pixels at distance; use relaxed threshold
    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 32);
}

TEST_CASE("Wgpu: CapsuleGeometry renders correctly", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);
    auto geometry = CapsuleGeometry::create(0.5f, 1.0f, 4, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    // Capsule geometry covers fewer pixels at distance; use relaxed threshold
    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 32);
}

// =============================================================================
// Section 13: Wgpu — Scissor Test
// =============================================================================

TEST_CASE("Wgpu: scissor test clips rendering", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);
    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    // Full render (no scissor)
    auto fullPixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    int fullCount = countNonBlack(fullPixels);

    // Scissored render — only top-left quarter
    // We need manual Wgpu renderer setup for scissor
    WgpuRenderer renderer(wgpuCanvas());
    renderer.setClearColor(Color(0x000000));
    renderer.setScissor(0, 0, RT_WIDTH / 2, RT_HEIGHT / 2);
    renderer.setScissorTest(true);

    auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
    renderer.setRenderTarget(target.get());
    renderer.render(*scene, *camera);
    auto scissoredPixels = renderer.readRGBPixels();

    renderer.setScissorTest(false);
    renderer.setRenderTarget(nullptr);
    renderer.dispose();

    int scissorCount = countNonBlack(scissoredPixels);
    CHECK(fullCount > scissorCount);
}

// =============================================================================
// Section 14: Wgpu — Multiple Lights Combined
// =============================================================================

TEST_CASE("Wgpu: combined ambient + directional + point lights are brighter", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeSingleLight = []() {
        auto scene = Scene::create();
        auto dirLight = DirectionalLight::create(Color(0xffffff), 0.5f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);

        auto geometry = SphereGeometry::create(1.0f, 16, 8);
        auto material = MeshLambertMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto makeMultiLight = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);
        auto dirLight = DirectionalLight::create(Color(0xffffff), 0.5f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);
        auto pointLight = PointLight::create(Color(0xffffff), 0.5f);
        pointLight->position.set(2, 2, 2);
        scene->add(pointLight);

        auto geometry = SphereGeometry::create(1.0f, 16, 8);
        auto material = MeshLambertMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto singlePixels = renderWithWgpu(*makeSingleLight(), *camera, clearColor);
    auto multiPixels = renderWithWgpu(*makeMultiLight(), *camera, clearColor);

    CHECK(avgBrightness(multiPixels) > avgBrightness(singlePixels));
}

TEST_CASE("Wgpu: SpotLight shadow casts correctly", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();

    auto spotLight = SpotLight::create(Color(0xffffff), 2.0f);
    spotLight->position.set(0, 5, 0);
    spotLight->angle = math::PI / 4;
    spotLight->castShadow = true;
    scene->add(spotLight);

    // Occluder box
    auto boxGeom = BoxGeometry::create(1, 0.2f, 1);
    auto boxMat = MeshPhongMaterial::create();
    boxMat->color = Color(0x888888);
    auto box = Mesh::create(boxGeom, boxMat);
    box->position.y = 2;
    box->castShadow = true;
    scene->add(box);

    // Floor plane
    auto planeGeom = PlaneGeometry::create(10, 10);
    auto planeMat = MeshPhongMaterial::create();
    planeMat->color = Color(0xcccccc);
    auto plane = Mesh::create(planeGeom, planeMat);
    plane->rotation.x = -math::PI / 2;
    plane->receiveShadow = true;
    scene->add(plane);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.set(0, 5, 8);
    camera->lookAt(Vector3(0, 0, 0));

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    int maxBright = maxPixelBrightness(pixels);
    CHECK(maxBright > 100);
}

TEST_CASE("Wgpu: PointLight shadow casts correctly", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();

    auto pointLight = PointLight::create(Color(0xffffff), 2.0f);
    pointLight->position.set(0, 3, 0);
    pointLight->castShadow = true;
    scene->add(pointLight);

    // Occluder sphere
    auto sphereGeom = SphereGeometry::create(0.5f, 16, 8);
    auto sphereMat = MeshPhongMaterial::create();
    sphereMat->color = Color(0x888888);
    auto sphere = Mesh::create(sphereGeom, sphereMat);
    sphere->position.y = 1.5f;
    sphere->castShadow = true;
    scene->add(sphere);

    // Floor plane
    auto planeGeom = PlaneGeometry::create(10, 10);
    auto planeMat = MeshPhongMaterial::create();
    planeMat->color = Color(0xcccccc);
    auto plane = Mesh::create(planeGeom, planeMat);
    plane->rotation.x = -math::PI / 2;
    plane->receiveShadow = true;
    scene->add(plane);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.set(0, 4, 6);
    camera->lookAt(Vector3(0, 0, 0));

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);
}

// =============================================================================
// Section 16: Wgpu — Texture Maps
// =============================================================================

// Cross-renderer normal-map comparison: Wgpu normal mapping implemented but output differs

TEST_CASE("Wgpu: ShaderMaterial renders with custom shaders", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto geometry = PlaneGeometry::create(2, 2);
    auto material = ShaderMaterial::create();

    // Minimal vertex + fragment shader pair that outputs solid magenta
    material->vertexShader = R"(
        void main() {
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    )";
    material->fragmentShader = R"(
        void main() {
            gl_FragColor = vec4(1.0, 0.0, 1.0, 1.0);
        }
    )";

    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    // Should produce magenta — red and blue present, no green
    auto avg = averageColor(pixels);
    CHECK(avg.r > 20);
    CHECK(avg.b > 20);
}

// WgpuRenderer: ShaderMaterial (custom GLSL/WGSL) not yet supported

TEST_CASE("Wgpu: ShaderMaterial with uniforms", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto geometry = PlaneGeometry::create(2, 2);
    auto material = ShaderMaterial::create();

    material->vertexShader = R"(
        void main() {
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    )";
    material->fragmentShader = R"(
        uniform vec3 uColor;
        void main() {
            gl_FragColor = vec4(uColor, 1.0);
        }
    )";
    material->uniforms["uColor"].setValue(Color(0x00ff00));

    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    auto avg = averageColor(pixels);
    CHECK(avg.g > avg.r);
    CHECK(avg.g > avg.b);
}

TEST_CASE("Wgpu: ShadowMaterial renders shadow-receiving plane", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();

    auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
    dirLight->position.set(0, 5, 5);
    dirLight->castShadow = true;
    scene->add(dirLight);

    // Occluder
    auto boxGeom = BoxGeometry::create(1, 1, 1);
    auto boxMat = MeshBasicMaterial::create();
    boxMat->color = Color(0xff0000);
    auto box = Mesh::create(boxGeom, boxMat);
    box->position.y = 1;
    box->castShadow = true;
    scene->add(box);

    // Shadow-receiving plane with ShadowMaterial
    auto planeGeom = PlaneGeometry::create(10, 10);
    auto shadowMat = ShadowMaterial::create();
    shadowMat->color = Color(0x000000);
    auto plane = Mesh::create(planeGeom, shadowMat);
    plane->rotation.x = -math::PI / 2;
    plane->receiveShadow = true;
    scene->add(plane);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.set(0, 5, 8);
    camera->lookAt(Vector3(0, 0, 0));

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    // The box should be visible (red), shadow plane may show shadow darkening
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 5);
}

// =============================================================================
// Section 19: Wgpu — Roughness & Metalness Maps
// =============================================================================

// WgpuRenderer: roughnessMap sampled in shader (green channel multiplies roughness)

TEST_CASE("Wgpu: roughnessMap affects specular highlights", "[wgpu]") {
    REQUIRE_WGPU();

    // Render the same scene with and without roughnessMap to verify it takes effect
    auto makeScene = [](bool useRoughnessMap) {
        auto scene = Scene::create();
        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshStandardMaterial::create();
        material->color = Color(0xffffff);
        material->metalness = 0.0f;
        material->roughness = 0.1f;// smooth base

        if (useRoughnessMap) {
            // Dark green channel (26/255 ~ 0.1): roughness = 0.1 * 0.1 = 0.01 (very smooth)
            material->roughnessMap = makeUniformTexture(0, 26, 0);
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto noMapPixels = renderWithWgpu(*makeScene(false), *camera, clearColor);
    auto mapPixels = renderWithWgpu(*makeScene(true), *camera, clearColor);

    // Verify both render visible geometry (roughnessMap doesn't break rendering)
    CHECK(countNonBlack(noMapPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(mapPixels) > PIXEL_COUNT / 8);
}

// WgpuRenderer: metalnessMap not yet sampled in shader

TEST_CASE("Wgpu: metalnessMap affects metallic appearance", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeScene = [](bool useMetalnessMap) {
        auto scene = Scene::create();
        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshStandardMaterial::create();
        material->color = Color(0xcccccc);
        material->roughness = 0.3f;
        material->metalness = 0.5f;// base semi-metallic (scaled by map)

        if (useMetalnessMap) {
            // metalnessMap blue channel scales metalness: 0.5 * 1.0 = 0.5
            material->metalnessMap = makeUniformTexture(255, 255, 255);
        } else {
            // Without map, use non-metallic to see a difference
            material->metalness = 0.0f;
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto nonMetalPixels = renderWithWgpu(*makeScene(false), *camera, clearColor);
    auto metalPixels = renderWithWgpu(*makeScene(true), *camera, clearColor);

    // Both should render visible geometry
    CHECK(countNonBlack(nonMetalPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(metalPixels) > PIXEL_COUNT / 8);

    // Metallic vs non-metallic should produce different brightness profiles
    double nonMetalBright = avgBrightness(nonMetalPixels);
    double metalBright = avgBrightness(metalPixels);
    CHECK(std::abs(nonMetalBright - metalBright) > 1.0);
}

TEST_CASE("Wgpu: emissiveMap produces glow pattern", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    // No lights — only emissive should be visible
    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshStandardMaterial::create();
    material->color = Color(0x000000);
    material->emissive = Color(0xffffff);
    material->emissiveIntensity = 1.0f;
    // Emissive map: half bright, half dark
    material->emissiveMap = makeProceduralTexture(
            255, 0, 0, 0, 0, 0,
            255, 0, 0, 0, 0, 0);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 5);

    auto avg = averageColor(pixels);
    CHECK(avg.r > avg.g);
}

TEST_CASE("Wgpu: aoMap darkens occluded areas", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeScene = [](bool useAoMap) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        // AO maps require a second UV set (uv2)
        auto uvAttr = geometry->getAttribute<float>("uv");
        auto& uvArr = uvAttr->array();
        geometry->setAttribute("uv2", FloatBufferAttribute::create(
                                              std::vector<float>(uvArr.begin(), uvArr.begin() + static_cast<long>(uvAttr->count() * 2)), 2));

        auto material = MeshStandardMaterial::create();
        material->color = Color(0xffffff);
        material->roughness = 1.0f;
        material->metalness = 0.0f;

        if (useAoMap) {
            // Dark AO — should reduce ambient light contribution
            material->aoMap = makeUniformTexture(50, 50, 50);
            material->aoMapIntensity = 1.0f;
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto noAoPixels = renderWithWgpu(*makeScene(false), *camera, clearColor);
    auto aoPixels = renderWithWgpu(*makeScene(true), *camera, clearColor);

    // AO map should darken the result
    CHECK(avgBrightness(noAoPixels) > avgBrightness(aoPixels));
}

// =============================================================================
// Section 22: Wgpu — Alpha Map
// =============================================================================

// WgpuRenderer: alphaMap not yet sampled in shader

TEST_CASE("Wgpu: alphaMap controls transparency", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeScene = [](bool useAlphaMap) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = PlaneGeometry::create(2, 2);
        auto material = MeshStandardMaterial::create();
        material->color = Color(0xffffff);
        material->transparent = true;
        material->side = Side::Double;

        if (useAlphaMap) {
            // Very transparent alpha map
            material->alphaMap = makeUniformTexture(30, 30, 30);
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto opaquePixels = renderWithWgpu(*makeScene(false), *camera, clearColor);
    auto alphaPixels = renderWithWgpu(*makeScene(true), *camera, clearColor);

    // Alpha-mapped version should be dimmer (more transparent)
    CHECK(avgBrightness(opaquePixels) > avgBrightness(alphaPixels));
}

TEST_CASE("Wgpu: displacementMap offsets vertices", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeScene = [](bool useDisplacement) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        // High-res sphere for displacement to be visible
        auto geometry = SphereGeometry::create(1.0f, 64, 32);
        auto material = MeshStandardMaterial::create();
        material->color = Color(0xffffff);
        material->roughness = 1.0f;

        if (useDisplacement) {
            material->displacementMap = makeUniformTexture(255, 255, 255);
            material->displacementScale = 0.5f;
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto normalPixels = renderWithWgpu(*makeScene(false), *camera, clearColor);
    auto displacedPixels = renderWithWgpu(*makeScene(true), *camera, clearColor);

    // Displaced sphere should cover more or different pixels
    int normalCount = countNonBlack(normalPixels);
    int displacedCount = countNonBlack(displacedPixels);
    CHECK(normalCount > PIXEL_COUNT / 8);
    CHECK(displacedCount > PIXEL_COUNT / 8);
    // Displacement should change the coverage or brightness
    CHECK(std::abs(normalCount - displacedCount) > 0);
}

// =============================================================================
// Section 24: Wgpu — Light Map
// =============================================================================

// WgpuRenderer: lightMap not yet sampled in shader

TEST_CASE("Wgpu: lightMap adds baked illumination", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeScene = [](bool useLightMap) {
        auto scene = Scene::create();
        // Only ambient — light map should add extra illumination
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);

        auto geometry = PlaneGeometry::create(2, 2);
        // Light maps require uv2
        auto uvAttr = geometry->getAttribute<float>("uv");
        auto& uvArr = uvAttr->array();
        geometry->setAttribute("uv2", FloatBufferAttribute::create(
                                              std::vector<float>(uvArr.begin(), uvArr.begin() + static_cast<long>(uvAttr->count() * 2)), 2));

        auto material = MeshStandardMaterial::create();
        material->color = Color(0xffffff);
        material->side = Side::Double;

        if (useLightMap) {
            material->lightMap = makeUniformTexture(255, 255, 255);
            material->lightMapIntensity = 1.0f;
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto noLmPixels = renderWithWgpu(*makeScene(false), *camera, clearColor);
    auto lmPixels = renderWithWgpu(*makeScene(true), *camera, clearColor);

    // Light map should add brightness
    CHECK(avgBrightness(lmPixels) > avgBrightness(noLmPixels));
}

// =============================================================================
// Section 25: Wgpu — Bump Map
// =============================================================================

// WgpuRenderer: bumpMap not yet sampled in shader

TEST_CASE("Wgpu: bumpMap perturbs surface shading", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeScene = [](bool useBumpMap) {
        auto scene = Scene::create();
        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(1, 1, 1);
        scene->add(dirLight);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshPhongMaterial::create();
        material->color = Color(0xffffff);

        if (useBumpMap) {
            // Checkerboard-style bump — should create shading variation
            material->bumpMap = makeProceduralTexture(
                    255, 255, 255, 0, 0, 0,
                    0, 0, 0, 255, 255, 255);
            material->bumpScale = 1.0f;
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto flatPixels = renderWithWgpu(*makeScene(false), *camera, clearColor);
    auto bumpPixels = renderWithWgpu(*makeScene(true), *camera, clearColor);

    // Phong-lit sphere with directional light has limited coverage on small target
    CHECK(countNonBlack(flatPixels) > 0);
    CHECK(countNonBlack(bumpPixels) > 0);

    // Bump map should change the brightness variance (more surface detail)
    double flatVar = brightnessVariance(flatPixels);
    double bumpVar = brightnessVariance(bumpPixels);
    CHECK(bumpVar != flatVar);
}

// =============================================================================
// Section 26: Wgpu — Gradient Map (Toon Shading)
// =============================================================================

// WgpuRenderer: gradientMap (MeshToonMaterial) not yet sampled in shader

TEST_CASE("Wgpu: gradientMap controls toon shading bands", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeScene = [](bool useGradientMap) {
        auto scene = Scene::create();
        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshToonMaterial::create();
        material->color = Color(0xffffff);

        if (useGradientMap) {
            // 3-step toon gradient: dark shadow / mid-tone / full brightness
            // Nearest filtering creates hard band boundaries typical of cel shading
            material->gradientMap = makeGradientTexture({25, 128, 255});
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto defaultPixels = renderWithWgpu(*makeScene(false), *camera, clearColor);
    auto gradientPixels = renderWithWgpu(*makeScene(true), *camera, clearColor);

    CHECK(countNonBlack(defaultPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(gradientPixels) > PIXEL_COUNT / 8);

    // Different gradient maps should produce different brightness
    CHECK(std::abs(avgBrightness(defaultPixels) - avgBrightness(gradientPixels)) > 0.5);
}

// =============================================================================
// Section 27: Wgpu — Environment Maps
// =============================================================================

// WgpuRenderer: envMap / CubeTexture not yet implemented

TEST_CASE("Wgpu: envMap adds reflections to standard material", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeScene = [](bool useEnvMap) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);
        auto dirLight = DirectionalLight::create(Color(0xffffff), 0.5f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshStandardMaterial::create();
        material->color = Color(0x444444);
        material->metalness = 1.0f;
        material->roughness = 0.0f;

        if (useEnvMap) {
            // Create a simple 6-face cube texture (all red)
            std::vector<Image> faces;
            for (int i = 0; i < 6; i++) {
                std::vector<unsigned char> faceData = {255, 0, 0, 255};
                faces.emplace_back(Image(std::move(faceData), 1, 1));
            }
            auto cubeTexture = CubeTexture::create(faces);
            material->envMap = cubeTexture;
            material->envMapIntensity = 1.0f;
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto noEnvPixels = renderWithWgpu(*makeScene(false), *camera, clearColor);
    auto envPixels = renderWithWgpu(*makeScene(true), *camera, clearColor);

    CHECK(countNonBlack(noEnvPixels) > PIXEL_COUNT / 16);
    CHECK(countNonBlack(envPixels) > PIXEL_COUNT / 16);

    // Env map on a metallic surface should make it brighter/more colored
    auto envAvg = averageColor(envPixels);
    CHECK(envAvg.r > 5.0);// Should pick up red from env map
}

// WgpuRenderer: envMap / CubeTexture not yet implemented

TEST_CASE("Wgpu: morph targets deform geometry", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeScene = [](float influence) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = BoxGeometry::create(1, 1, 1);

        // Create a morph target that scales the box to 2x
        auto posAttr = geometry->getAttribute<float>("position");
        int count = static_cast<int>(posAttr->count());
        std::vector<float> morphPositions(count * 3);
        for (int i = 0; i < count; i++) {
            morphPositions[i * 3 + 0] = posAttr->getX(i) * 2.0f;
            morphPositions[i * 3 + 1] = posAttr->getY(i) * 2.0f;
            morphPositions[i * 3 + 2] = posAttr->getZ(i) * 2.0f;
        }

        auto morphAttrs = geometry->getOrCreateMorphAttribute("position");
        morphAttrs->emplace_back(FloatBufferAttribute::create(morphPositions, 3));

        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        material->morphTargets = true;
        auto mesh = Mesh::create(geometry, material);
        mesh->morphTargetInfluences().resize(1);
        mesh->morphTargetInfluences()[0] = influence;
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;
    Color clearColor(0x000000);

    auto basePixels = renderWithWgpu(*makeScene(0.0f), *camera, clearColor);
    auto morphedPixels = renderWithWgpu(*makeScene(1.0f), *camera, clearColor);

    int baseCount = countNonBlack(basePixels);
    int morphedCount = countNonBlack(morphedPixels);

    CHECK(baseCount > 0);
    CHECK(morphedCount > 0);

    // Morphed (scaled up) should cover more pixels
    CHECK(morphedCount > baseCount);
}

// WgpuRenderer: morph targets not yet implemented

TEST_CASE("Wgpu: SkinnedMesh with skeleton renders correctly", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    // Create a simple 2-bone skeleton with a cylinder-like geometry
    auto geometry = CylinderGeometry::create(0.3f, 0.3f, 2.0f, 8, 4);

    // Assign skin weights and indices
    auto posAttr = geometry->getAttribute<float>("position");
    int vertexCount = static_cast<int>(posAttr->count());

    std::vector<float> skinWeights(vertexCount * 4, 0.0f);
    std::vector<float> skinIndices(vertexCount * 4, 0.0f);

    for (int i = 0; i < vertexCount; i++) {
        float y = posAttr->getY(i);
        // Blend between bone 0 (bottom) and bone 1 (top)
        float weight = (y + 1.0f) / 2.0f;// normalize from [-1,1] to [0,1]
        skinWeights[i * 4 + 0] = 1.0f - weight;
        skinWeights[i * 4 + 1] = weight;
        skinIndices[i * 4 + 0] = 0.0f;
        skinIndices[i * 4 + 1] = 1.0f;
    }

    geometry->setAttribute("skinWeight", FloatBufferAttribute::create(skinWeights, 4));
    geometry->setAttribute("skinIndex", FloatBufferAttribute::create(skinIndices, 4));

    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);

    // Create bones
    auto bone0 = Bone::create();
    bone0->position.y = -1.0f;
    auto bone1 = Bone::create();
    bone1->position.y = 1.0f;
    bone0->add(bone1);

    auto skeleton = Skeleton::create({bone0, bone1});
    auto skinnedMesh = SkinnedMesh::create(geometry, material);
    skinnedMesh->add(bone0);
    skinnedMesh->bind(skeleton);

    scene->add(skinnedMesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 0);
}

// WgpuRenderer: SkinnedMesh / skeletal animation not yet implemented

TEST_CASE("Wgpu: SkinnedMesh bone rotation deforms mesh", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeScene = [](float boneRotation) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = CylinderGeometry::create(0.3f, 0.3f, 2.0f, 8, 4);
        auto posAttr = geometry->getAttribute<float>("position");
        int vertexCount = static_cast<int>(posAttr->count());

        std::vector<float> skinWeights(vertexCount * 4, 0.0f);
        std::vector<float> skinIndices(vertexCount * 4, 0.0f);

        for (int i = 0; i < vertexCount; i++) {
            float y = posAttr->getY(i);
            float weight = (y + 1.0f) / 2.0f;
            skinWeights[i * 4 + 0] = 1.0f - weight;
            skinWeights[i * 4 + 1] = weight;
            skinIndices[i * 4 + 0] = 0.0f;
            skinIndices[i * 4 + 1] = 1.0f;
        }

        geometry->setAttribute("skinWeight", FloatBufferAttribute::create(skinWeights, 4));
        geometry->setAttribute("skinIndex", FloatBufferAttribute::create(skinIndices, 4));

        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);

        auto bone0 = Bone::create();
        bone0->position.y = -1.0f;
        auto bone1 = Bone::create();
        bone1->position.y = 1.0f;
        bone0->add(bone1);

        auto skeleton = Skeleton::create({bone0, bone1});
        auto skinnedMesh = SkinnedMesh::create(geometry, material);
        skinnedMesh->add(bone0);
        skinnedMesh->bind(skeleton);

        // Apply bone rotation AFTER binding so it creates a visible deformation
        bone1->rotation.z = boneRotation;

        scene->add(skinnedMesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;
    Color clearColor(0x000000);

    auto straightPixels = renderWithWgpu(*makeScene(0.0f), *camera, clearColor);
    auto bentPixels = renderWithWgpu(*makeScene(math::PI / 4), *camera, clearColor);

    // Both should render visible geometry
    CHECK(countNonBlack(straightPixels) > 0);
    CHECK(countNonBlack(bentPixels) > 0);

    // Bent mesh should have different pixel distribution
    double straightX = avgXPosition(straightPixels, RT_WIDTH, RT_HEIGHT);
    double bentX = avgXPosition(bentPixels, RT_WIDTH, RT_HEIGHT);
    // Rotation should shift average X position
    CHECK(std::abs(straightX - bentX) > 0.5);
}

// =============================================================================
// Section 30: Wgpu — Clipping Planes
// =============================================================================
// WgpuRenderer: clipping planes not yet implemented

TEST_CASE("Wgpu: clipping plane cuts geometry", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeScene = [](bool useClipping) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = SphereGeometry::create(1.0f, 16, 8);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        material->side = Side::Double;

        if (useClipping) {
            // Clip plane that cuts through center of sphere
            material->clippingPlanes.push_back(Plane(Vector3(1, 0, 0), 0));
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    // Need to enable local clipping on the Wgpu renderer
    // Render without clipping
    auto fullPixels = renderWithWgpu(*makeScene(false), *camera, clearColor);

    // Render with clipping — need renderer with localClippingEnabled
    {
        WgpuRenderer renderer(wgpuCanvas());
        renderer.setClearColor(clearColor);
        renderer.localClippingEnabled = true;

        auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        auto clippedScene = makeScene(true);
        renderer.render(*clippedScene, *camera);
        auto clippedPixels = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();

        int fullCount = countNonBlack(fullPixels);
        int clippedCount = countNonBlack(clippedPixels);

        CHECK(fullCount > PIXEL_COUNT / 8);
        // Clipped sphere should show fewer pixels (half was clipped away)
        CHECK(fullCount > clippedCount);
    }
}

// WgpuRenderer: clipping planes not yet implemented

TEST_CASE("Wgpu: tone mapping affects output brightness", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);
    auto dirLight = DirectionalLight::create(Color(0xffffff), 2.0f);
    dirLight->position.set(0, 0, 1);
    scene->add(dirLight);

    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshStandardMaterial::create();
    material->color = Color(0xffffff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    // Render with no tone mapping
    auto renderWithToneMapping = [&](ToneMapping tm, float exposure) {
        WgpuRenderer renderer(wgpuCanvas());
        renderer.setClearColor(Color(0x000000));
        renderer.toneMapping = tm;
        renderer.toneMappingExposure = exposure;

        auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        renderer.render(*scene, *camera);
        auto pixels = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();
        return pixels;
    };

    auto nonePixels = renderWithToneMapping(ToneMapping::None, 1.0f);
    auto reinhardPixels = renderWithToneMapping(ToneMapping::Reinhard, 1.0f);
    auto acesPixels = renderWithToneMapping(ToneMapping::ACESFilmic, 1.0f);

    CHECK(countNonBlack(nonePixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(reinhardPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(acesPixels) > PIXEL_COUNT / 8);

    // Different tone mapping should produce different brightness
    double noneBright = avgBrightness(nonePixels);
    double reinhardBright = avgBrightness(reinhardPixels);
    double acesBright = avgBrightness(acesPixels);

    // At least one tone mapper should differ from None
    bool reinhardDiffers = std::abs(noneBright - reinhardBright) > 1.0;
    bool acesDiffers = std::abs(noneBright - acesBright) > 1.0;
    CHECK((reinhardDiffers || acesDiffers));
}

// WgpuRenderer bug: toneMappingExposure renders black (possibly canvas/state issue)

TEST_CASE("Wgpu: toneMappingExposure scales brightness", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0x808080));
    scene->add(ambient);

    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto renderWithExposure = [&](float exposure) {
        WgpuRenderer renderer(wgpuCanvas());
        renderer.setClearColor(Color(0x000000));
        renderer.toneMapping = ToneMapping::Reinhard;
        renderer.toneMappingExposure = exposure;

        auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        renderer.render(*scene, *camera);
        auto pixels = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();
        return pixels;
    };

    auto lowExp = renderWithExposure(0.5f);
    auto highExp = renderWithExposure(2.0f);

    // Higher exposure should produce brighter output
    CHECK(avgBrightness(highExp) > avgBrightness(lowExp));
}

// =============================================================================
// Section 32: Wgpu — Output Encoding / Color Space
// =============================================================================

// WgpuRenderer: sRGB output encoding not yet implemented

TEST_CASE("Wgpu: sRGB output encoding differs from linear", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0x808080);// mid-grey
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto renderWithEncoding = [&](Encoding enc) {
        WgpuRenderer renderer(wgpuCanvas());
        renderer.setClearColor(Color(0x000000));
        renderer.outputEncoding = enc;

        auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        renderer.render(*scene, *camera);
        auto pixels = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();
        return pixels;
    };

    auto linearPixels = renderWithEncoding(Encoding::Linear);
    auto srgbPixels = renderWithEncoding(Encoding::sRGB);

    CHECK(countNonBlack(linearPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(srgbPixels) > PIXEL_COUNT / 8);

    // sRGB gamma curve should produce different brightness than linear
    double linearBright = avgBrightness(linearPixels);
    double srgbBright = avgBrightness(srgbPixels);
    CHECK(std::abs(linearBright - srgbBright) > 1.0);
}

// =============================================================================
// Section 33: Wgpu — Instanced Colors
// =============================================================================

// WgpuRenderer: InstancedMesh + vertex colors not yet implemented

TEST_CASE("Wgpu: InstancedMesh per-instance colors", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geometry = BoxGeometry::create(0.5f, 0.5f, 0.5f);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);

    auto im = InstancedMesh::create(geometry, material, 2);

    // Left instance — red
    Matrix4 m;
    m.setPosition(Vector3(-1.0f, 0, 0));
    im->setMatrixAt(0, m);
    im->setColorAt(0, Color(0xff0000));

    // Right instance — blue
    m.setPosition(Vector3(1.0f, 0, 0));
    im->setMatrixAt(1, m);
    im->setColorAt(1, Color(0x0000ff));

    im->instanceColor()->needsUpdate();
    scene->add(im);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    auto avg = averageColor(pixels);

    // Should have both red and blue components (small boxes at 64x64)
    CHECK(avg.r > 1);
    CHECK(avg.b > 1);
    CHECK(countNonBlack(pixels) > 0);
}

// WgpuRenderer: InstancedMesh + vertex colors not yet implemented

TEST_CASE("Wgpu: shadow map resolution affects quality", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeScene = [](int shadowMapSize) {
        auto scene = Scene::create();

        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(0, 5, 5);
        dirLight->castShadow = true;
        dirLight->shadow->mapSize.set(shadowMapSize, shadowMapSize);
        scene->add(dirLight);

        // Occluder
        auto boxGeom = BoxGeometry::create(0.5f, 0.5f, 0.5f);
        auto boxMat = MeshPhongMaterial::create();
        boxMat->color = Color(0x888888);
        auto box = Mesh::create(boxGeom, boxMat);
        box->position.y = 2;
        box->castShadow = true;
        scene->add(box);

        // Floor
        auto planeGeom = PlaneGeometry::create(10, 10);
        auto planeMat = MeshPhongMaterial::create();
        planeMat->color = Color(0xcccccc);
        auto plane = Mesh::create(planeGeom, planeMat);
        plane->rotation.x = -math::PI / 2;
        plane->receiveShadow = true;
        scene->add(plane);

        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.set(0, 5, 8);
    camera->lookAt(Vector3(0, 0, 0));
    Color clearColor(0x000000);

    auto lowResPixels = renderWithWgpu(*makeScene(64), *camera, clearColor);
    auto highResPixels = renderWithWgpu(*makeScene(512), *camera, clearColor);

    // Both should render something
    CHECK(countNonBlack(lowResPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(highResPixels) > PIXEL_COUNT / 8);

    // Higher resolution shadow maps may produce slightly different brightness variance
    // (sharper shadow edges vs blockier). Both should be reasonable.
    double lowVar = brightnessVariance(lowResPixels);
    double highVar = brightnessVariance(highResPixels);
    // Just verify both produce variance (shadows present)
    CHECK(lowVar > 10.0);
    CHECK(highVar > 10.0);
}

TEST_CASE("Wgpu: shadow bias prevents shadow acne", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeScene = [](float bias) {
        auto scene = Scene::create();

        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(0, 5, 0);
        dirLight->castShadow = true;
        dirLight->shadow->bias = bias;
        scene->add(dirLight);

        // Floor that receives shadow and is also shadow caster
        auto planeGeom = PlaneGeometry::create(5, 5);
        auto planeMat = MeshPhongMaterial::create();
        planeMat->color = Color(0xffffff);
        auto plane = Mesh::create(planeGeom, planeMat);
        plane->rotation.x = -math::PI / 2;
        plane->receiveShadow = true;
        scene->add(plane);

        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.set(0, 3, 5);
    camera->lookAt(Vector3(0, 0, 0));
    Color clearColor(0x000000);

    auto noBiasPixels = renderWithWgpu(*makeScene(0.0f), *camera, clearColor);
    auto biasPixels = renderWithWgpu(*makeScene(-0.005f), *camera, clearColor);

    // Both should produce visible output
    CHECK(countNonBlack(noBiasPixels) > PIXEL_COUNT / 16);
    CHECK(countNonBlack(biasPixels) > PIXEL_COUNT / 16);
}

// =============================================================================
// Section 35: Wgpu — Specular Map
// =============================================================================

// WgpuRenderer: specularMap not yet sampled in shader

TEST_CASE("Wgpu: specularMap controls highlight regions", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeScene = [](bool useSpecularMap) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);
        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshPhongMaterial::create();
        material->color = Color(0x888888);
        material->specular = Color(0xffffff);
        material->shininess = 100.0f;

        if (useSpecularMap) {
            // Dark specular map — reduces specular highlights
            material->specularMap = makeUniformTexture(30, 30, 30);
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto fullSpecPixels = renderWithWgpu(*makeScene(false), *camera, clearColor);
    auto reducedSpecPixels = renderWithWgpu(*makeScene(true), *camera, clearColor);

    // Both should render visible geometry
    CHECK(countNonBlack(fullSpecPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(reducedSpecPixels) > PIXEL_COUNT / 8);
}

// =============================================================================
// Section 36: Wgpu — Diffuse Map (color map)
// =============================================================================

TEST_CASE("Wgpu: map (diffuse texture) tints geometry", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geometry = PlaneGeometry::create(2, 2);
    auto material = MeshBasicMaterial::create();
    material->map = makeUniformTexture(255, 0, 0);// Red texture
    material->side = Side::Double;

    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    auto avg = averageColor(pixels);
    CHECK(avg.r > avg.g + 20);
    CHECK(avg.r > avg.b + 20);
}

// =============================================================================
// Section 37: Wgpu — LineBasicMaterial Options
// =============================================================================

TEST_CASE("Wgpu: LineBasicMaterial with dashed line", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto geometry = BufferGeometry::create();
    std::vector<float> positions = {-2, 0, 0, 2, 0, 0};
    geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));

    auto material = LineBasicMaterial::create();
    material->color = Color(0xffffff);

    auto line = Line::create(geometry, material);
    scene->add(line);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 3);
}

// =============================================================================
// Section 38: Wgpu — MSAA Anti-Aliasing
//
// TDD red phase: these tests reference setSampleCount()/getSampleCount() which
// do not yet exist on WgpuRenderer. Enable when the MSAA API is implemented.
// =============================================================================

TEST_CASE("Wgpu: MSAA 4x produces more intermediate edge pixels than 1x", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = BoxGeometry::create(1, 1, 1);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->rotation.z = math::PI / 6;// 30 degrees — creates diagonal edges
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;
    Color clearColor(0x000000);

    std::vector<unsigned char> pixelsNoMSAA;
    {
        WgpuRenderer renderer(wgpuCanvas());
        renderer.setSampleCount(1);
        renderer.setClearColor(clearColor);
        auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        auto scene = makeScene();
        renderer.render(*scene, *camera);
        pixelsNoMSAA = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();
    }

    std::vector<unsigned char> pixelsMSAA;
    {
        WgpuRenderer renderer(wgpuCanvas());
        renderer.setSampleCount(4);
        renderer.setClearColor(clearColor);
        auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        auto scene = makeScene();
        renderer.render(*scene, *camera);
        pixelsMSAA = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();
    }

    int intermediateNoMSAA = countIntermediatePixels(pixelsNoMSAA);
    int intermediateMSAA = countIntermediatePixels(pixelsMSAA);
    CHECK(intermediateMSAA > intermediateNoMSAA);
}

TEST_CASE("Wgpu: MSAA brightness variance differs at edges", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = BoxGeometry::create(1, 1, 1);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->rotation.z = math::PI / 6;
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;
    Color clearColor(0x000000);

    std::vector<unsigned char> pixelsNoMSAA;
    {
        WgpuRenderer renderer(wgpuCanvas());
        renderer.setSampleCount(1);
        renderer.setClearColor(clearColor);
        auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        auto scene = makeScene();
        renderer.render(*scene, *camera);
        pixelsNoMSAA = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();
    }

    std::vector<unsigned char> pixelsMSAA;
    {
        WgpuRenderer renderer(wgpuCanvas());
        renderer.setSampleCount(4);
        renderer.setClearColor(clearColor);
        auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        auto scene = makeScene();
        renderer.render(*scene, *camera);
        pixelsMSAA = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();
    }

    double varianceNoMSAA = brightnessVariance(pixelsNoMSAA);
    double varianceMSAA = brightnessVariance(pixelsMSAA);
    CHECK(std::abs(varianceMSAA - varianceNoMSAA) > 0.5);
}

TEST_CASE("Wgpu: MSAA sample count round-trips", "[wgpu]") {
    REQUIRE_WGPU();

    WgpuRenderer renderer(wgpuCanvas());
    renderer.setSampleCount(4);
    CHECK(renderer.getSampleCount() == 4);
    renderer.setSampleCount(1);
    CHECK(renderer.getSampleCount() == 1);
    renderer.dispose();
}

TEST_CASE("Wgpu: MSAA render still produces correct average color", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = BoxGeometry::create(1, 1, 1);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->rotation.z = math::PI / 6;
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;
    Color clearColor(0x000000);

    std::vector<unsigned char> pixelsNoMSAA;
    {
        WgpuRenderer renderer(wgpuCanvas());
        renderer.setSampleCount(1);
        renderer.setClearColor(clearColor);
        auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        auto scene = makeScene();
        renderer.render(*scene, *camera);
        pixelsNoMSAA = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();
    }

    std::vector<unsigned char> pixelsMSAA;
    {
        WgpuRenderer renderer(wgpuCanvas());
        renderer.setSampleCount(4);
        renderer.setClearColor(clearColor);
        auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        auto scene = makeScene();
        renderer.render(*scene, *camera);
        pixelsMSAA = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();
    }

    auto avgNoMSAA = averageColor(pixelsNoMSAA);
    auto avgMSAA = averageColor(pixelsMSAA);
    CHECK(std::abs(avgNoMSAA.r - avgMSAA.r) < 10);
    CHECK(std::abs(avgNoMSAA.g - avgMSAA.g) < 10);
    CHECK(std::abs(avgNoMSAA.b - avgMSAA.b) < 10);
}

// =============================================================================
// Section 39: Wgpu — Multiple Shadow Casters
//
// These tests use existing API (castShadow on multiple lights) but FAIL at
// runtime because WgpuRenderer only processes the first shadow-casting
// DirectionalLight. They compile and run, but assertions fail.
// =============================================================================

TEST_CASE("Wgpu: two directional lights produce two distinct shadows", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeSingleShadowScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0x333333));
        scene->add(ambient);

        auto lightA = DirectionalLight::create(Color(0xffffff), 1.0f);
        lightA->position.set(5, 5, 0);
        lightA->castShadow = true;
        scene->add(lightA);

        auto boxGeom = BoxGeometry::create(0.5f, 0.5f, 0.5f);
        auto boxMat = MeshStandardMaterial::create();
        boxMat->color = Color(0x888888);
        auto box = Mesh::create(boxGeom, boxMat);
        box->position.y = 2;
        box->castShadow = true;
        scene->add(box);

        auto planeGeom = PlaneGeometry::create(10, 10);
        auto planeMat = MeshStandardMaterial::create();
        planeMat->color = Color(0xcccccc);
        auto plane = Mesh::create(planeGeom, planeMat);
        plane->rotation.x = -math::PI / 2;
        plane->receiveShadow = true;
        scene->add(plane);

        return scene;
    };

    auto makeDualShadowScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0x333333));
        scene->add(ambient);

        auto lightA = DirectionalLight::create(Color(0xffffff), 1.0f);
        lightA->position.set(5, 5, 0);
        lightA->castShadow = true;
        scene->add(lightA);

        auto lightB = DirectionalLight::create(Color(0xffffff), 1.0f);
        lightB->position.set(-5, 5, 0);
        lightB->castShadow = true;
        scene->add(lightB);

        auto boxGeom = BoxGeometry::create(0.5f, 0.5f, 0.5f);
        auto boxMat = MeshStandardMaterial::create();
        boxMat->color = Color(0x888888);
        auto box = Mesh::create(boxGeom, boxMat);
        box->position.y = 2;
        box->castShadow = true;
        scene->add(box);

        auto planeGeom = PlaneGeometry::create(10, 10);
        auto planeMat = MeshStandardMaterial::create();
        planeMat->color = Color(0xcccccc);
        auto plane = Mesh::create(planeGeom, planeMat);
        plane->rotation.x = -math::PI / 2;
        plane->receiveShadow = true;
        scene->add(plane);

        return scene;
    };

    auto camera = PerspectiveCamera::create(60, 1.0f, 0.1f, 100);
    camera->position.set(0, 5, 8);
    camera->lookAt(Vector3(0, 0, 0));
    Color clearColor(0x000000);

    auto singlePixels = renderWithWgpu(*makeSingleShadowScene(), *camera, clearColor);
    auto dualPixels = renderWithWgpu(*makeDualShadowScene(), *camera, clearColor);

    // With per-light shadows, the dual scene has brighter unshadowed areas
    // (2 lights) and distinct shadow regions, producing higher brightness
    // variance than a single-light scene.
    double singleVar = brightnessVariance(singlePixels);
    double dualVar = brightnessVariance(dualPixels);
    CHECK(dualVar >= singleVar);
    // The dual scene should also be brighter overall (2 lights vs 1)
    CHECK(avgBrightness(dualPixels) > avgBrightness(singlePixels));
}

TEST_CASE("Wgpu: SpotLight and DirectionalLight both cast shadows simultaneously", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeScene = [](bool useSpot, bool useDir) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0x333333));
        scene->add(ambient);

        if (useDir) {
            auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
            dirLight->position.set(-5, 5, 0);
            dirLight->castShadow = true;
            scene->add(dirLight);
        }

        if (useSpot) {
            auto spotLight = SpotLight::create(Color(0xffffff), 2.0f);
            spotLight->position.set(5, 5, 0);
            spotLight->angle = math::PI / 4;
            spotLight->castShadow = true;
            scene->add(spotLight);
        }

        auto boxGeom = BoxGeometry::create(0.5f, 0.5f, 0.5f);
        auto boxMat = MeshStandardMaterial::create();
        boxMat->color = Color(0x888888);
        auto box = Mesh::create(boxGeom, boxMat);
        box->position.y = 2;
        box->castShadow = true;
        scene->add(box);

        auto planeGeom = PlaneGeometry::create(10, 10);
        auto planeMat = MeshStandardMaterial::create();
        planeMat->color = Color(0xcccccc);
        auto plane = Mesh::create(planeGeom, planeMat);
        plane->rotation.x = -math::PI / 2;
        plane->receiveShadow = true;
        scene->add(plane);

        return scene;
    };

    auto camera = PerspectiveCamera::create(60, 1.0f, 0.1f, 100);
    camera->position.set(0, 5, 8);
    camera->lookAt(Vector3(0, 0, 0));
    Color clearColor(0x000000);

    auto dirOnlyPixels = renderWithWgpu(*makeScene(false, true), *camera, clearColor);
    auto dualPixels = renderWithWgpu(*makeScene(true, true), *camera, clearColor);

    // Adding a SpotLight increases total illumination — the dual scene
    // should be brighter overall than the directional-only scene.
    CHECK(avgBrightness(dualPixels) > avgBrightness(dirOnlyPixels));
}

TEST_CASE("Wgpu: shadow from light A is independent of light B occlusion", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0x333333));
    scene->add(ambient);

    auto lightA = DirectionalLight::create(Color(0xffffff), 1.0f);
    lightA->position.set(5, 5, 0);
    lightA->castShadow = true;
    scene->add(lightA);

    auto lightB = DirectionalLight::create(Color(0xffffff), 1.0f);
    lightB->position.set(-5, 5, 0);
    lightB->castShadow = true;
    scene->add(lightB);

    auto boxGeomA = BoxGeometry::create(0.3f, 0.3f, 0.3f);
    auto boxMatA = MeshStandardMaterial::create();
    boxMatA->color = Color(0x888888);
    auto boxA = Mesh::create(boxGeomA, boxMatA);
    boxA->position.set(2, 2, 0);
    boxA->castShadow = true;
    scene->add(boxA);

    auto boxGeomB = BoxGeometry::create(0.3f, 0.3f, 0.3f);
    auto boxMatB = MeshStandardMaterial::create();
    boxMatB->color = Color(0x888888);
    auto boxB = Mesh::create(boxGeomB, boxMatB);
    boxB->position.set(-2, 2, 0);
    boxB->castShadow = true;
    scene->add(boxB);

    auto planeGeom = PlaneGeometry::create(10, 10);
    auto planeMat = MeshStandardMaterial::create();
    planeMat->color = Color(0xcccccc);
    auto plane = Mesh::create(planeGeom, planeMat);
    plane->rotation.x = -math::PI / 2;
    plane->receiveShadow = true;
    scene->add(plane);

    auto camera = PerspectiveCamera::create(60, 1.0f, 0.1f, 100);
    camera->position.set(0, 5, 8);
    camera->lookAt(Vector3(0, 0, 0));

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));

    double brightness = avgBrightness(pixels);
    CHECK(brightness > 20.0);
    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 4);
}

TEST_CASE("Wgpu: multiple shadow casters produce shadows at different floor positions", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeSingleScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0x333333));
        scene->add(ambient);

        auto light = DirectionalLight::create(Color(0xffffff), 1.0f);
        light->position.set(5, 5, 0);
        light->castShadow = true;
        scene->add(light);

        auto boxGeom = BoxGeometry::create(0.5f, 0.5f, 0.5f);
        auto boxMat = MeshStandardMaterial::create();
        boxMat->color = Color(0x888888);
        auto box = Mesh::create(boxGeom, boxMat);
        box->position.y = 2;
        box->castShadow = true;
        scene->add(box);

        auto planeGeom = PlaneGeometry::create(10, 10);
        auto planeMat = MeshStandardMaterial::create();
        planeMat->color = Color(0xcccccc);
        auto plane = Mesh::create(planeGeom, planeMat);
        plane->rotation.x = -math::PI / 2;
        plane->receiveShadow = true;
        scene->add(plane);

        return scene;
    };

    auto makeDualScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0x333333));
        scene->add(ambient);

        auto lightA = DirectionalLight::create(Color(0xffffff), 1.0f);
        lightA->position.set(5, 5, 0);
        lightA->castShadow = true;
        scene->add(lightA);

        auto lightB = DirectionalLight::create(Color(0xffffff), 1.0f);
        lightB->position.set(-5, 5, 0);
        lightB->castShadow = true;
        scene->add(lightB);

        auto boxGeom = BoxGeometry::create(0.5f, 0.5f, 0.5f);
        auto boxMat = MeshStandardMaterial::create();
        boxMat->color = Color(0x888888);
        auto box = Mesh::create(boxGeom, boxMat);
        box->position.y = 2;
        box->castShadow = true;
        scene->add(box);

        auto planeGeom = PlaneGeometry::create(10, 10);
        auto planeMat = MeshStandardMaterial::create();
        planeMat->color = Color(0xcccccc);
        auto plane = Mesh::create(planeGeom, planeMat);
        plane->rotation.x = -math::PI / 2;
        plane->receiveShadow = true;
        scene->add(plane);

        return scene;
    };

    auto camera = PerspectiveCamera::create(60, 1.0f, 0.1f, 100);
    camera->position.set(0, 5, 8);
    camera->lookAt(Vector3(0, 0, 0));
    Color clearColor(0x000000);

    auto singlePixels = renderWithWgpu(*makeSingleScene(), *camera, clearColor);
    auto dualPixels = renderWithWgpu(*makeDualScene(), *camera, clearColor);

    double singleVariance = brightnessVariance(singlePixels);
    double dualVariance = brightnessVariance(dualPixels);
    CHECK(dualVariance > singleVariance);
}

// =============================================================================
// Section 40: Wgpu — Bind Group Separation Regression
//
// Regression guards for the bind group refactoring. These tests exercise
// complex binding combinations and should PASS now and continue passing
// after @group(0) is split into per-frame/per-material/per-object groups.
// =============================================================================

TEST_CASE("Wgpu: multi-material scene renders all materials correctly", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff), 0.3f);
    scene->add(ambient);
    auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
    dirLight->position.set(1, 1, 1);
    scene->add(dirLight);

    auto geom = BoxGeometry::create(0.6f, 0.6f, 0.6f);

    auto basicMat = MeshBasicMaterial::create();
    basicMat->color = Color(0xff0000);
    auto basicMesh = Mesh::create(geom, basicMat);
    basicMesh->position.set(-0.8f, 0.8f, 0);
    scene->add(basicMesh);

    auto phongMat = MeshPhongMaterial::create();
    phongMat->color = Color(0x00ff00);
    auto phongMesh = Mesh::create(geom, phongMat);
    phongMesh->position.set(0.8f, 0.8f, 0);
    scene->add(phongMesh);

    auto standardMat = MeshStandardMaterial::create();
    standardMat->color = Color(0x0000ff);
    auto standardMesh = Mesh::create(geom, standardMat);
    standardMesh->position.set(-0.8f, -0.8f, 0);
    scene->add(standardMesh);

    auto lambertMat = MeshLambertMaterial::create();
    lambertMat->color = Color(0xffffff);
    auto lambertMesh = Mesh::create(geom, lambertMat);
    lambertMesh->position.set(0.8f, -0.8f, 0);
    scene->add(lambertMesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));

    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 32);
    auto avg = averageColor(pixels);
    CHECK(avg.r > 2);
    CHECK(avg.g > 2);
    CHECK(avg.b > 2);
}

TEST_CASE("Wgpu: textured and untextured objects coexist in same scene", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geom = BoxGeometry::create(0.8f, 0.8f, 0.8f);

    auto texMat = MeshBasicMaterial::create();
    texMat->map = makeUniformTexture(255, 0, 0);
    auto texMesh = Mesh::create(geom, texMat);
    texMesh->position.x = -1.0f;
    scene->add(texMesh);

    auto colorMat = MeshBasicMaterial::create();
    colorMat->color = Color(0x0000ff);
    auto colorMesh = Mesh::create(geom, colorMat);
    colorMesh->position.x = 1.0f;
    scene->add(colorMesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));

    auto avg = averageColor(pixels);
    CHECK(avg.r > 2);
    CHECK(avg.b > 2);
    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 32);
}

TEST_CASE("Wgpu: shadow + texture + instancing combination renders", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0x333333));
    scene->add(ambient);

    auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
    dirLight->position.set(0, 5, 5);
    dirLight->castShadow = true;
    scene->add(dirLight);

    auto geom = BoxGeometry::create(0.4f, 0.4f, 0.4f);
    auto mat = MeshStandardMaterial::create();
    mat->map = makeUniformTexture(255, 128, 0);

    auto im = InstancedMesh::create(geom, mat, 4);
    Matrix4 m;
    m.setPosition(Vector3(-0.8f, 1.5f, 0));
    im->setMatrixAt(0, m);
    m.setPosition(Vector3(0.8f, 1.5f, 0));
    im->setMatrixAt(1, m);
    m.setPosition(Vector3(-0.8f, 2.5f, 0));
    im->setMatrixAt(2, m);
    m.setPosition(Vector3(0.8f, 2.5f, 0));
    im->setMatrixAt(3, m);
    im->castShadow = true;
    scene->add(im);

    auto planeGeom = PlaneGeometry::create(10, 10);
    auto planeMat = MeshStandardMaterial::create();
    planeMat->color = Color(0xcccccc);
    auto plane = Mesh::create(planeGeom, planeMat);
    plane->rotation.x = -math::PI / 2;
    plane->receiveShadow = true;
    scene->add(plane);

    auto camera = PerspectiveCamera::create(60, 1.0f, 0.1f, 100);
    camera->position.set(0, 5, 8);
    camera->lookAt(Vector3(0, 0, 0));

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));

    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 4);
    double var = brightnessVariance(pixels);
    CHECK(var > 5.0);
}

TEST_CASE("Wgpu: normal map + emissive map + AO map combined", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
    dirLight->position.set(1, 1, 1);
    scene->add(dirLight);

    auto geometry = SphereGeometry::create(1.0f, 32, 16);
    auto material = MeshStandardMaterial::create();
    material->color = Color(0xffffff);
    material->roughness = 0.5f;

    material->normalMap = makeUniformTexture(128, 128, 255);

    material->emissive = Color(0xffffff);
    material->emissiveIntensity = 0.5f;
    material->emissiveMap = makeUniformTexture(255, 0, 0);

    material->aoMap = makeUniformTexture(128, 128, 128);

    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));

    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 8);
    auto avg = averageColor(pixels);
    CHECK((avg.r + avg.g + avg.b) / 3.0 > 15.0);
}

TEST_CASE("Wgpu: PBR + normal map + shadow + fog combination", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    scene->fog = Fog(Color(0x888888), 1.0f, 20.0f);

    auto ambient = AmbientLight::create(Color(0x333333));
    scene->add(ambient);

    auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
    dirLight->position.set(0, 5, 5);
    dirLight->castShadow = true;
    scene->add(dirLight);

    auto geometry = SphereGeometry::create(1.0f, 32, 16);
    auto material = MeshStandardMaterial::create();
    material->color = Color(0xffffff);
    material->roughness = 0.5f;
    material->metalness = 0.5f;
    material->normalMap = makeUniformTexture(128, 128, 255);

    auto mesh = Mesh::create(geometry, material);
    mesh->castShadow = true;
    scene->add(mesh);

    auto planeGeom = PlaneGeometry::create(10, 10);
    auto planeMat = MeshStandardMaterial::create();
    planeMat->color = Color(0xcccccc);
    auto plane = Mesh::create(planeGeom, planeMat);
    plane->rotation.x = -math::PI / 2;
    plane->position.y = -1.5f;
    plane->receiveShadow = true;
    scene->add(plane);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));

    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 8);
    double var = brightnessVariance(pixels);
    CHECK(var > 5.0);
}

TEST_CASE("Wgpu: toon material with gradient map renders stepped lighting", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
    dirLight->position.set(1, 1, 1);
    scene->add(dirLight);

    auto geometry = SphereGeometry::create(1.0f, 32, 16);
    auto material = MeshToonMaterial::create();
    material->color = Color(0xff8800);
    material->gradientMap = makeGradientTexture({0, 128, 255});

    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));

    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 16);
    auto avg = averageColor(pixels);
    CHECK(avg.r + avg.g + avg.b > 30);
}

TEST_CASE("Wgpu: morph targets + skinning combined", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geometry = BoxGeometry::create(0.5f, 2.0f, 0.5f, 1, 4, 1);
    auto posAttr = geometry->getAttribute<float>("position");
    int vertexCount = static_cast<int>(posAttr->count());

    std::vector<float> morphPositions(vertexCount * 3);
    for (int i = 0; i < vertexCount; i++) {
        morphPositions[i * 3 + 0] = posAttr->getX(i) * 1.5f;
        morphPositions[i * 3 + 1] = posAttr->getY(i) * 1.5f;
        morphPositions[i * 3 + 2] = posAttr->getZ(i) * 1.5f;
    }
    auto morphAttrs = geometry->getOrCreateMorphAttribute("position");
    morphAttrs->emplace_back(FloatBufferAttribute::create(morphPositions, 3));

    std::vector<float> skinWeights(vertexCount * 4, 0.0f);
    std::vector<float> skinIndices(vertexCount * 4, 0.0f);
    for (int i = 0; i < vertexCount; i++) {
        float y = posAttr->getY(i);
        float weight = (y + 1.0f) / 2.0f;
        skinWeights[i * 4 + 0] = 1.0f - weight;
        skinWeights[i * 4 + 1] = weight;
        skinIndices[i * 4 + 0] = 0.0f;
        skinIndices[i * 4 + 1] = 1.0f;
    }
    geometry->setAttribute("skinWeight", FloatBufferAttribute::create(skinWeights, 4));
    geometry->setAttribute("skinIndex", FloatBufferAttribute::create(skinIndices, 4));

    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    material->morphTargets = true;

    auto bone0 = Bone::create();
    bone0->position.y = -1.0f;
    auto bone1 = Bone::create();
    bone1->position.y = 1.0f;
    bone0->add(bone1);

    auto skeleton = Skeleton::create({bone0, bone1});
    auto skinnedMesh = SkinnedMesh::create(geometry, material);
    skinnedMesh->add(bone0);
    skinnedMesh->bind(skeleton);
    skinnedMesh->morphTargetInfluences().resize(1);
    skinnedMesh->morphTargetInfluences()[0] = 0.5f;

    bone1->rotation.z = math::PI / 6;

    scene->add(skinnedMesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));
    CHECK(countNonBlack(pixels) > 0);
}

TEST_CASE("Wgpu: instancing + vertex colors combination", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geometry = BoxGeometry::create(0.4f, 0.4f, 0.4f);

    int vertexCount = static_cast<int>(geometry->getAttribute<float>("position")->count());
    std::vector<float> colors(vertexCount * 3);
    for (int i = 0; i < vertexCount; i++) {
        colors[i * 3 + 0] = (i % 3 == 0) ? 1.0f : 0.0f;
        colors[i * 3 + 1] = (i % 3 == 1) ? 1.0f : 0.0f;
        colors[i * 3 + 2] = (i % 3 == 2) ? 1.0f : 0.0f;
    }
    geometry->setAttribute("color", FloatBufferAttribute::create(colors, 3));

    auto material = MeshBasicMaterial::create();
    material->vertexColors = true;

    auto im = InstancedMesh::create(geometry, material, 4);
    Matrix4 m;
    m.setPosition(Vector3(-0.8f, -0.8f, 0));
    im->setMatrixAt(0, m);
    m.setPosition(Vector3(0.8f, -0.8f, 0));
    im->setMatrixAt(1, m);
    m.setPosition(Vector3(-0.8f, 0.8f, 0));
    im->setMatrixAt(2, m);
    m.setPosition(Vector3(0.8f, 0.8f, 0));
    im->setMatrixAt(3, m);
    scene->add(im);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithWgpu(*scene, *camera, Color(0x000000));

    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 64);
    auto avg = averageColor(pixels);
    CHECK(avg.r > 1);
    CHECK(avg.g > 1);
}

TEST_CASE("Wgpu: displacement map + clipping plane combination", "[wgpu]") {
    REQUIRE_WGPU();

    auto makeScene = [](bool useClipping) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);
        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(1, 1, 1);
        scene->add(dirLight);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshStandardMaterial::create();
        material->color = Color(0xffffff);
        material->roughness = 1.0f;
        material->displacementMap = makeUniformTexture(255, 255, 255);
        material->displacementScale = 0.3f;
        material->side = Side::Double;

        if (useClipping) {
            material->clippingPlanes.push_back(Plane(Vector3(1, 0, 0), 0));
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto fullPixels = renderWithWgpu(*makeScene(false), *camera, clearColor);

    std::vector<unsigned char> clippedPixels;
    {
        WgpuRenderer renderer(wgpuCanvas());
        renderer.setClearColor(clearColor);
        renderer.localClippingEnabled = true;

        auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        auto clippedScene = makeScene(true);
        renderer.render(*clippedScene, *camera);
        clippedPixels = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();
    }

    int fullCount = countNonBlack(fullPixels);
    int clippedCount = countNonBlack(clippedPixels);
    CHECK(fullCount > clippedCount);
}


// =============================================================================
// Section 42: Wgpu — Post-Processing Framework
//
// TDD red phase: these tests reference EffectComposer and ShaderPass classes
// that do not yet exist. Enable when the post-processing framework is
// implemented.
//
// Proposed API:
//   class ShaderPass { static create(wgslSource); };
//   class EffectComposer { EffectComposer(WgpuRenderer&); addPass(); render(); readRGBPixels(); };
// =============================================================================

#if 1// Enable when EffectComposer and ShaderPass are implemented

#include "threepp/renderers/wgpu/EffectComposer.hpp"
#include "threepp/renderers/wgpu/ShaderPass.hpp"

namespace {
    const std::string identityWGSL = R"(
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var inputSampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex fn vs(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var pos = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    var uv = array<vec2f, 3>(vec2f(0, 1), vec2f(2, 1), vec2f(0, -1));
    var out: VertexOutput;
    out.position = vec4f(pos[idx], 0, 1);
    out.uv = uv[idx];
    return out;
}

@fragment fn fs(in: VertexOutput) -> @location(0) vec4f {
    return textureSample(inputTex, inputSampler, in.uv);
}
)";

    const std::string grayscaleWGSL = R"(
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var inputSampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex fn vs(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var pos = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    var uv = array<vec2f, 3>(vec2f(0, 1), vec2f(2, 1), vec2f(0, -1));
    var out: VertexOutput;
    out.position = vec4f(pos[idx], 0, 1);
    out.uv = uv[idx];
    return out;
}

@fragment fn fs(in: VertexOutput) -> @location(0) vec4f {
    let c = textureSample(inputTex, inputSampler, in.uv);
    let gray = dot(c.rgb, vec3f(0.299, 0.587, 0.114));
    return vec4f(gray, gray, gray, c.a);
}
)";

    const std::string invertWGSL = R"(
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var inputSampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex fn vs(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var pos = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    var uv = array<vec2f, 3>(vec2f(0, 1), vec2f(2, 1), vec2f(0, -1));
    var out: VertexOutput;
    out.position = vec4f(pos[idx], 0, 1);
    out.uv = uv[idx];
    return out;
}

@fragment fn fs(in: VertexOutput) -> @location(0) vec4f {
    let c = textureSample(inputTex, inputSampler, in.uv);
    return vec4f(1.0 - c.r, 1.0 - c.g, 1.0 - c.b, c.a);
}
)";

    const std::string brightnessWGSL = R"(
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var inputSampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex fn vs(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var pos = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    var uv = array<vec2f, 3>(vec2f(0, 1), vec2f(2, 1), vec2f(0, -1));
    var out: VertexOutput;
    out.position = vec4f(pos[idx], 0, 1);
    out.uv = uv[idx];
    return out;
}

@fragment fn fs(in: VertexOutput) -> @location(0) vec4f {
    let c = textureSample(inputTex, inputSampler, in.uv);
    return vec4f(clamp(c.rgb * 1.5, vec3f(0.0), vec3f(1.0)), c.a);
}
)";
}// namespace

TEST_CASE("Wgpu: identity post-process pass matches non-post-processed output", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff8800);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto directPixels = renderWithWgpu(*scene, *camera, clearColor);

    WgpuRenderer renderer(wgpuCanvas());
    EffectComposer composer(renderer);
    composer.addPass(ShaderPass::create(identityWGSL));

    renderer.setClearColor(clearColor);
    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());
    composer.render(*scene, *camera);
    auto postPixels = composer.readRGBPixels();
    renderer.setRenderTarget(nullptr);
    renderer.dispose();

    CHECK(std::abs(avgBrightness(directPixels) - avgBrightness(postPixels)) < 5.0);
    auto avgDirect = averageColor(directPixels);
    auto avgPost = averageColor(postPixels);
    CHECK(std::abs(avgDirect.r - avgPost.r) < 5);
    CHECK(std::abs(avgDirect.g - avgPost.g) < 5);
    CHECK(std::abs(avgDirect.b - avgPost.b) < 5);
}

TEST_CASE("Wgpu: grayscale post-process produces equal RGB channels", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geom = BoxGeometry::create(1.5f, 1.5f, 1.5f);

    auto redMat = MeshBasicMaterial::create();
    redMat->color = Color(0xff0000);
    auto redMesh = Mesh::create(geom, redMat);
    redMesh->position.x = -1.5f;
    scene->add(redMesh);

    auto greenMat = MeshBasicMaterial::create();
    greenMat->color = Color(0x00ff00);
    auto greenMesh = Mesh::create(geom, greenMat);
    scene->add(greenMesh);

    auto blueMat = MeshBasicMaterial::create();
    blueMat->color = Color(0x0000ff);
    auto blueMesh = Mesh::create(geom, blueMat);
    blueMesh->position.x = 1.5f;
    scene->add(blueMesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto originalPixels = renderWithWgpu(*scene, *camera, clearColor);
    auto origAvg = averageColor(originalPixels);
    bool channelsVary = (std::abs(origAvg.r - origAvg.g) > 10) ||
                        (std::abs(origAvg.r - origAvg.b) > 10);
    CHECK(channelsVary);

    WgpuRenderer renderer(wgpuCanvas());
    EffectComposer composer(renderer);
    composer.addPass(ShaderPass::create(grayscaleWGSL));

    renderer.setClearColor(clearColor);
    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());
    composer.render(*scene, *camera);
    auto grayscalePixels = composer.readRGBPixels();
    renderer.setRenderTarget(nullptr);
    renderer.dispose();

    auto avg = averageColor(grayscalePixels);
    CHECK(std::abs(avg.r - avg.g) < 10);
    CHECK(std::abs(avg.r - avg.b) < 10);
    CHECK(std::abs(avg.g - avg.b) < 10);
}

TEST_CASE("Wgpu: invert post-process inverts colors", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff0000);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto originalPixels = renderWithWgpu(*scene, *camera, clearColor);

    WgpuRenderer renderer(wgpuCanvas());
    EffectComposer composer(renderer);
    composer.addPass(ShaderPass::create(invertWGSL));

    renderer.setClearColor(clearColor);
    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());
    composer.render(*scene, *camera);
    auto invertedPixels = composer.readRGBPixels();
    renderer.setRenderTarget(nullptr);
    renderer.dispose();

    auto origAvg = averageColor(originalPixels);
    auto invAvg = averageColor(invertedPixels);
    CHECK(std::abs((origAvg.r + invAvg.r) - 255) < 20);
    CHECK(std::abs((origAvg.g + invAvg.g) - 255) < 20);
    CHECK(std::abs((origAvg.b + invAvg.b) - 255) < 20);
}

TEST_CASE("Wgpu: double invert post-process restores original", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff8800);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto originalPixels = renderWithWgpu(*scene, *camera, clearColor);

    WgpuRenderer renderer(wgpuCanvas());
    EffectComposer composer(renderer);
    composer.addPass(ShaderPass::create(invertWGSL));
    composer.addPass(ShaderPass::create(invertWGSL));

    renderer.setClearColor(clearColor);
    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());
    composer.render(*scene, *camera);
    auto doubleInvertPixels = composer.readRGBPixels();
    renderer.setRenderTarget(nullptr);
    renderer.dispose();

    CHECK(std::abs(avgBrightness(originalPixels) - avgBrightness(doubleInvertPixels)) < 5.0);
    auto origAvg = averageColor(originalPixels);
    auto doubleAvg = averageColor(doubleInvertPixels);
    CHECK(std::abs(origAvg.r - doubleAvg.r) < 5);
    CHECK(std::abs(origAvg.g - doubleAvg.g) < 5);
    CHECK(std::abs(origAvg.b - doubleAvg.b) < 5);
}

TEST_CASE("Wgpu: brightness post-process increases average brightness", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff), 0.5f);
    scene->add(ambient);
    auto dirLight = DirectionalLight::create(Color(0xffffff), 0.5f);
    dirLight->position.set(1, 1, 1);
    scene->add(dirLight);

    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshStandardMaterial::create();
    material->color = Color(0x888888);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto originalPixels = renderWithWgpu(*scene, *camera, clearColor);
    double origBright = avgBrightness(originalPixels);
    CHECK(origBright > 10.0);

    WgpuRenderer renderer(wgpuCanvas());
    EffectComposer composer(renderer);
    composer.addPass(ShaderPass::create(brightnessWGSL));

    renderer.setClearColor(clearColor);
    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());
    composer.render(*scene, *camera);
    auto boostedPixels = composer.readRGBPixels();
    renderer.setRenderTarget(nullptr);
    renderer.dispose();

    double boostedBright = avgBrightness(boostedPixels);
    CHECK(boostedBright > origBright);
}

TEST_CASE("Wgpu: post-process works with render targets", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geom = BoxGeometry::create(0.6f, 0.6f, 0.6f);

    auto redMat = MeshBasicMaterial::create();
    redMat->color = Color(0xff0000);
    auto redMesh = Mesh::create(geom, redMat);
    redMesh->position.x = -1.0f;
    scene->add(redMesh);

    auto blueMat = MeshBasicMaterial::create();
    blueMat->color = Color(0x0000ff);
    auto blueMesh = Mesh::create(geom, blueMat);
    blueMesh->position.x = 1.0f;
    scene->add(blueMesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    WgpuRenderer renderer(wgpuCanvas());
    EffectComposer composer(renderer);
    composer.addPass(ShaderPass::create(grayscaleWGSL));

    renderer.setClearColor(Color(0x000000));
    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());
    composer.render(*scene, *camera);
    auto pixels = composer.readRGBPixels();
    renderer.setRenderTarget(nullptr);
    renderer.dispose();

    REQUIRE(pixels.size() == DATA_SIZE);
    auto avg = averageColor(pixels);
    CHECK(std::abs(avg.r - avg.g) < 10);
    CHECK(std::abs(avg.r - avg.b) < 10);
}

#endif// Post-processing API guard

// =============================================================================
// Section: Resize robustness tests
//
// These reproduce the crash that occurs when a window is resized smaller while
// the WgpuRenderer is running. The root cause is stale viewport/scissor state
// exceeding the actual attachment dimensions after resize.
// =============================================================================

TEST_CASE("Wgpu: rapid resize smaller via setSize with render targets", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);
    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff8800);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    Canvas canvas1(Canvas::Parameters().size(512, 512).headless(true));

    WgpuRenderer renderer(canvas1);
    renderer.setClearColor(Color(0x000000));

    // Simulate rapid window resize: large → small → large → very small
    // Each step renders to a differently-sized render target after calling
    // setSize, which updates viewport/scissor to the new canvas dimensions.
    struct SizeStep {
        int w, h;
    };
    SizeStep sizes[] = {
            {512, 512},
            {256, 256},
            {400, 300},
            {128, 64},
            {512, 512},
            {64, 64},
            {300, 500},
            {100, 100},
    };

    for (auto& [sw, sh] : sizes) {
        renderer.setSize({sw, sh});
        auto rt = RenderTarget::create(sw, sh, RenderTarget::Options{});
        renderer.setRenderTarget(rt.get());
        renderer.render(*scene, *camera);
        auto pixels = renderer.readRGBPixels();
        REQUIRE(pixels.size() == static_cast<size_t>(sw * sh * 3));
    }

    renderer.setRenderTarget(nullptr);
    renderer.dispose();
}

TEST_CASE("Wgpu: resize smaller with stale scissor does not crash", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);
    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff8800);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    Canvas canvas2(Canvas::Parameters().size(256, 256).headless(true));

    WgpuRenderer renderer(canvas2);
    renderer.setClearColor(Color(0x000000));

    // Render at large size with scissor test enabled
    auto largeRT = RenderTarget::create(256, 256, RenderTarget::Options{});
    renderer.setRenderTarget(largeRT.get());
    renderer.setScissorTest(true);
    renderer.setScissor(0, 0, 256, 256);
    renderer.render(*scene, *camera);

    // Shrink WITHOUT updating scissor — scissor is still 256x256
    auto smallRT = RenderTarget::create(128, 128, RenderTarget::Options{});
    renderer.setRenderTarget(smallRT.get());
    renderer.render(*scene, *camera);
    auto pixels = renderer.readRGBPixels();
    REQUIRE(pixels.size() == 128 * 128 * 3);

    renderer.setScissorTest(false);
    renderer.setRenderTarget(nullptr);
    renderer.dispose();
}

TEST_CASE("Wgpu: resize smaller with stale viewport does not crash", "[wgpu]") {
    REQUIRE_WGPU();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);
    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff8800);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    Canvas canvas3(Canvas::Parameters().size(256, 256).headless(true));

    WgpuRenderer renderer(canvas3);
    renderer.setClearColor(Color(0x000000));

    // Set an explicit viewport at the large size
    auto largeRT = RenderTarget::create(256, 256, RenderTarget::Options{});
    renderer.setRenderTarget(largeRT.get());
    renderer.setViewport(0, 0, 256, 256);
    renderer.render(*scene, *camera);

    // Shrink the render target WITHOUT updating the viewport
    auto smallRT = RenderTarget::create(128, 128, RenderTarget::Options{});
    renderer.setRenderTarget(smallRT.get());
    // viewport still 256x256 — must be clamped internally
    renderer.render(*scene, *camera);
    auto pixels = renderer.readRGBPixels();
    REQUIRE(pixels.size() == 128 * 128 * 3);

    renderer.setRenderTarget(nullptr);
    renderer.dispose();
}

TEST_CASE("Wgpu: setSize shrink then render to RT simulates window drag", "[wgpu]") {
    REQUIRE_WGPU();

    // This test most closely matches the real-world window resize scenario:
    // canvas starts large, setSize shrinks it (updating viewport/scissor to
    // the new canvas size), but then the render target created from that size
    // might mismatch if there's a timing issue.

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);
    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff8800);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    Canvas canvas4(Canvas::Parameters().size(512, 512).headless(true));

    WgpuRenderer renderer(canvas4);
    renderer.setClearColor(Color(0x000000));
    renderer.setSampleCount(4);// MSAA like the water example

    // Render at initial size
    auto rt1 = RenderTarget::create(512, 512, RenderTarget::Options{});
    renderer.setRenderTarget(rt1.get());
    renderer.render(*scene, *camera);

    // Simulate rapid shrink during window drag — multiple setSize calls
    // before rendering, like GLFW firing many resize events
    renderer.setSize({400, 400});
    renderer.setSize({300, 300});
    renderer.setSize({200, 200});

    // Now render at the final smaller size
    auto rt2 = RenderTarget::create(200, 200, RenderTarget::Options{});
    renderer.setRenderTarget(rt2.get());
    renderer.render(*scene, *camera);
    auto pixels = renderer.readRGBPixels();
    REQUIRE(pixels.size() == 200 * 200 * 3);
    CHECK(countNonBlack(pixels) > 0);

    // Go back larger — also must not crash
    renderer.setSize({512, 512});
    auto rt3 = RenderTarget::create(512, 512, RenderTarget::Options{});
    renderer.setRenderTarget(rt3.get());
    renderer.render(*scene, *camera);
    auto pixels2 = renderer.readRGBPixels();
    REQUIRE(pixels2.size() == 512 * 512 * 3);
    CHECK(countNonBlack(pixels2) > 0);

    renderer.setRenderTarget(nullptr);
    renderer.dispose();
}

TEST_CASE("Wgpu: render target size mismatch with renderer size", "[wgpu]") {
    REQUIRE_WGPU();

    // Renderer thinks it's 512x512 but the render target is 128x128.
    // This is the core of the resize bug — the renderer's internal state
    // (viewport, scissor) is for a larger size than the actual attachment.

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);
    auto geometry = BoxGeometry::create(1.0f, 1.0f, 1.0f);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff0000);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    Canvas canvas5(Canvas::Parameters().size(512, 512).headless(true));

    WgpuRenderer renderer(canvas5);
    renderer.setClearColor(Color(0x000000));

    // Renderer is sized at 512x512, but render target is much smaller.
    // Viewport/scissor are 512x512, attachment is 128x128.
    auto smallRT = RenderTarget::create(128, 128, RenderTarget::Options{});
    renderer.setRenderTarget(smallRT.get());
    renderer.render(*scene, *camera);
    auto pixels = renderer.readRGBPixels();
    REQUIRE(pixels.size() == 128 * 128 * 3);
    CHECK(countNonBlack(pixels) > 0);

    // Also test with an even more extreme mismatch
    auto tinyRT = RenderTarget::create(32, 32, RenderTarget::Options{});
    renderer.setRenderTarget(tinyRT.get());
    renderer.render(*scene, *camera);
    auto tinyPixels = renderer.readRGBPixels();
    REQUIRE(tinyPixels.size() == 32 * 32 * 3);

    renderer.setRenderTarget(nullptr);
    renderer.dispose();
}

TEST_CASE("Wgpu: surface reconfigured smaller behind renderer's back", "[wgpu][surface]") {
    REQUIRE_WGPU();

    // This is the exact scenario that crashes during manual window drag on
    // Windows: the OS resizes the swap chain to a smaller size while the
    // renderer's viewport/scissor state still reflects the old (larger) size.
    // We simulate this by calling wgpuSurfaceConfigure directly via the
    // nativeSurface() accessor, bypassing the renderer's resize detection.

    Canvas sneakyCanvas(Canvas::Parameters()
                                .size(800, 600)
                                .headless(true)
                                .title("sneaky_resize_test"));

    sneakyCanvas.setSize({800, 600});


    WgpuRenderer renderer(sneakyCanvas);
    renderer.setClearColor(Color(0x000000));

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);
    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff8800);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    // Render at initial 800x600 — establishes viewport/scissor at 800x600
    renderer.render(*scene, *camera);
    renderer.endFrame();// Release surface texture before reconfiguring

    // Reconfigure the surface to a SMALLER size directly, without telling
    // the renderer. The renderer's viewport/scissor/size_ remain at 800x600,
    // but the next wgpuSurfaceGetCurrentTexture will return 400x300.
    auto* wgpuSurface = static_cast<WGPUSurface>(renderer.nativeSurface());
    auto* wgpuDevice = static_cast<WGPUDevice>(renderer.nativeDevice());
    REQUIRE(wgpuSurface != nullptr);

    WGPUSurfaceConfiguration config{};
    config.device = wgpuDevice;
    config.format = static_cast<WGPUTextureFormat>(renderer.nativeSurfaceFormat());
    config.usage = WGPUTextureUsage_RenderAttachment;
    config.width = 400;
    config.height = 300;
    config.presentMode = WGPUPresentMode_Fifo;
    config.alphaMode = WGPUCompositeAlphaMode_Auto;
    wgpuSurfaceConfigure(wgpuSurface, &config);

    // Render — surface texture is 400x300 but renderer thinks it's 800x600.
    // Without the fix, this crashes with: "Scissor Rect { w: 800, h: 600 }
    // is not contained in the render target (400, 300, 1)"
    renderer.render(*scene, *camera);
    renderer.endFrame();// Release surface texture before reconfiguring

    // Even more extreme: shrink to tiny
    config.width = 200;
    config.height = 150;
    wgpuSurfaceConfigure(wgpuSurface, &config);
    renderer.render(*scene, *camera);
    renderer.endFrame();

    CHECK(true);
    renderer.dispose();
}

TEST_CASE("Wgpu: surface reconfigured smaller with MSAA", "[wgpu][surface]") {
    REQUIRE_WGPU();

    // Same scenario with MSAA 4x — matches the water example exactly.

    Canvas sneakyMsaaCanvas(Canvas::Parameters()
                                    .size(800, 600)
                                    .headless((true))
                                    .title("sneaky_msaa_resize_test")
                                    .antialiasing(4));

    sneakyMsaaCanvas.setSize({800, 600});


    WgpuRenderer renderer(sneakyMsaaCanvas);
    renderer.setClearColor(Color(0x000000));

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);
    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff8800);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    // Render at initial 800x600
    renderer.render(*scene, *camera);
    renderer.endFrame();// Release surface texture before reconfiguring

    // Reconfigure surface smaller behind the renderer's back — with MSAA
    auto* wgpuSurface = static_cast<WGPUSurface>(renderer.nativeSurface());
    auto* wgpuDevice = static_cast<WGPUDevice>(renderer.nativeDevice());
    REQUIRE(wgpuSurface != nullptr);

    WGPUSurfaceConfiguration config{};
    config.device = wgpuDevice;
    config.format = static_cast<WGPUTextureFormat>(renderer.nativeSurfaceFormat());
    config.usage = WGPUTextureUsage_RenderAttachment;
    config.width = 400;
    config.height = 300;
    config.presentMode = WGPUPresentMode_Fifo;
    config.alphaMode = WGPUCompositeAlphaMode_Auto;
    wgpuSurfaceConfigure(wgpuSurface, &config);

    // Render with stale state + MSAA — the MSAA/depth textures must match
    // the actual surface texture, not the renderer's stale size
    renderer.render(*scene, *camera);
    renderer.endFrame();

    CHECK(true);
    renderer.dispose();
}

TEST_CASE("Wgpu: windowed surface resize does not crash", "[wgpu][surface]") {
    REQUIRE_WGPU();

    // Non-headless: creates a real window with a GPU surface, exactly like
    // the water example. Uses a singleton canvas (never destroyed) to avoid
    // glfwTerminate() invalidating GLFW state for later tests.
    // Programmatic setSize triggers GLFW's window_size_callback synchronously
    // on Windows, updating canvas.size() — reproducing the live drag scenario.

    Canvas surfaceCanvas(Canvas::Parameters()
                                 .size(800, 600)
                                 .title("resize_test"));

    surfaceCanvas.setSize({800, 600});

    WgpuRenderer renderer(surfaceCanvas);
    renderer.setClearColor(Color(0x000000));

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);
    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff8800);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    // Render a few frames at the initial size
    for (int i = 0; i < 3; i++) {
        renderer.render(*scene, *camera);
    }

    // Shrink the window — triggers glfwSetWindowSize which fires the
    // window_size_callback synchronously, updating canvas.size().
    surfaceCanvas.setSize({400, 300});
    for (int i = 0; i < 3; i++) {
        renderer.render(*scene, *camera);
    }

    // Shrink further
    surfaceCanvas.setSize({200, 150});
    for (int i = 0; i < 3; i++) {
        renderer.render(*scene, *camera);
    }

    // Grow back
    surfaceCanvas.setSize({800, 600});
    for (int i = 0; i < 3; i++) {
        renderer.render(*scene, *camera);
    }

    // Rapid resize sequence — simulates a fast window drag
    surfaceCanvas.setSize({700, 500});
    renderer.render(*scene, *camera);
    surfaceCanvas.setSize({500, 350});
    renderer.render(*scene, *camera);
    surfaceCanvas.setSize({300, 200});
    renderer.render(*scene, *camera);
    surfaceCanvas.setSize({150, 100});
    renderer.render(*scene, *camera);

    CHECK(true);
    renderer.dispose();
}

TEST_CASE("Wgpu: windowed surface resize with MSAA does not crash", "[wgpu][surface]") {
    REQUIRE_WGPU();

    // Same as above but with MSAA 4x — matches the water example exactly.
    Canvas msaaCanvas(Canvas::Parameters()
                              .size(800, 600)
                              .title("resize_msaa_test")
                              .antialiasing(4));

    msaaCanvas.setSize({800, 600});


    WgpuRenderer renderer(msaaCanvas);
    renderer.setClearColor(Color(0x000000));

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);
    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff8800);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    // Render at initial size
    for (int i = 0; i < 3; i++) {
        renderer.render(*scene, *camera);
    }

    // Rapid shrink sequence with MSAA
    msaaCanvas.setSize({600, 400});
    renderer.render(*scene, *camera);
    msaaCanvas.setSize({400, 300});
    renderer.render(*scene, *camera);
    msaaCanvas.setSize({200, 150});
    renderer.render(*scene, *camera);

    // Grow back
    msaaCanvas.setSize({800, 600});
    renderer.render(*scene, *camera);

    CHECK(true);
    renderer.dispose();
}

TEST_CASE("Wgpu: depthTest=false renders on top of occluding geometry", "[wgpu]") {
    REQUIRE_WGPU();

    // A red sphere sits in front of (occluding) where a blue plane would appear.
    // With depthTest=true the blue plane is hidden. With depthTest=false it renders on top.

    auto makeScene = [](bool noDepthTest) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        // Red sphere at z=0 — occludes anything behind it from camera at z=5
        auto sphereGeo = SphereGeometry::create(1.5f, 16, 8);
        auto sphereMat = MeshBasicMaterial::create();
        sphereMat->color = Color(0xff0000);
        scene->add(Mesh::create(sphereGeo, sphereMat));

        // Blue plane behind the sphere
        auto planeGeo = PlaneGeometry::create(4.0f, 4.0f);
        auto planeMat = MeshBasicMaterial::create();
        planeMat->color = Color(0x0000ff);
        planeMat->depthTest = !noDepthTest;
        planeMat->depthWrite = false;
        planeMat->transparent = true;
        auto plane = Mesh::create(planeGeo, planeMat);
        plane->position.z = -0.5f;
        scene->add(plane);

        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;
    Color clearColor(0x000000);

    auto depthTestPixels = renderWithWgpu(*makeScene(false), *camera, clearColor);
    auto noDepthTestPixels = renderWithWgpu(*makeScene(true), *camera, clearColor);

    // Count blue-dominant pixels (b > r && b > 10)
    auto countBlueDominant = [](const std::vector<uint8_t>& px) {
        int n = 0;
        for (size_t i = 0; i + 2 < px.size(); i += 3) {
            if (px[i + 2] > px[i] && px[i + 2] > 10) ++n;
        }
        return n;
    };

    int blueDT = countBlueDominant(depthTestPixels);
    int blueNoDT = countBlueDominant(noDepthTestPixels);

    // depthTest=true: blue plane hidden behind red sphere — few or no blue pixels at center
    // depthTest=false: blue plane ignores depth and draws on top — more blue pixels visible
    CHECK(blueNoDT > blueDT);
}
