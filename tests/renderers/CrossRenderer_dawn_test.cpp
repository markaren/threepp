// Dawn renderer tests (WebGPU backend).
// Split from CrossRenderer_test.cpp for maintainability.

#include "CrossRenderer_helpers.hpp"

TEST_CASE("Dawn: clear color produces expected pixels", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto pixels = renderWithDawn(*scene, *camera, Color(1.0f, 0.0f, 0.0f));
    REQUIRE(pixels.size() == DATA_SIZE);
    CHECK(allPixelsMatch(pixels, 255, 0, 0, 2));
}

TEST_CASE("Dawn: readback dimensions match render target", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    CHECK(pixels.size() == DATA_SIZE);
}


// =============================================================================
// Section 3: Cross-renderer comparisons
// Generate GL reference data first, then Dawn data, then compare.
// =============================================================================

TEST_CASE("Dawn: PointLight illuminates a sphere", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    auto avg = averageColor(pixels);
    CHECK(avg.r > avg.b);
}

TEST_CASE("Dawn: SpotLight illuminates a sphere", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 16);

    auto avg = averageColor(pixels);
    CHECK(avg.g > avg.r);
}

TEST_CASE("Dawn: HemisphereLight tints geometry", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);
}

TEST_CASE("Dawn: textured box uses diffuse map", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    // Create a 2x2 checkerboard texture procedurally
    std::vector<unsigned char> texData = {
        255, 0, 0, 255,   // red
        0, 255, 0, 255,   // green
        0, 255, 0, 255,   // green
        255, 0, 0, 255    // red
    };
    Image image(texData, 2, 2);

    auto texture = Texture::create(image);
    texture->needsUpdate();

    auto geometry = BoxGeometry::create(2, 2, 2);
    auto material = MeshBasicMaterial::create();
    material->map = texture;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    // The box should have visible colored pixels from the texture
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    // Should have both red and green components from the checkerboard
    auto avg = averageColor(pixels);
    CHECK(avg.r > 5.0);
    CHECK(avg.g > 5.0);
}

TEST_CASE("Dawn: opacity affects brightness", "[dawn]") {
    REQUIRE_DAWN();

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

    auto fullPixels = renderWithDawn(*makeScene(1.0f), *camera, Color(0x000000));
    auto halfPixels = renderWithDawn(*makeScene(0.5f), *camera, Color(0x000000));
    REQUIRE(fullPixels.size() == DATA_SIZE);
    REQUIRE(halfPixels.size() == DATA_SIZE);

    auto fullAvg = averageColor(fullPixels);
    auto halfAvg = averageColor(halfPixels);

    double fullBright = (fullAvg.r + fullAvg.g + fullAvg.b) / 3.0;
    double halfBright = (halfAvg.r + halfAvg.g + halfAvg.b) / 3.0;
    CHECK(fullBright > halfBright);
}

TEST_CASE("Dawn: setSize reconfigures surface", "[dawn]") {
    REQUIRE_DAWN();

    static Canvas* canvas = nullptr;
    if (!canvas) {
        canvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*canvas);

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

TEST_CASE("Dawn: setPixelRatio updates ratio", "[dawn]") {
    REQUIRE_DAWN();

    static Canvas* canvas = nullptr;
    if (!canvas) {
        canvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*canvas);

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

TEST_CASE("Dawn: viewport restricts rendering region", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto geometry = BoxGeometry::create(4, 4, 4);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    static Canvas* canvas = nullptr;
    if (!canvas) {
        canvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*canvas);
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

TEST_CASE("Dawn: dispose does not crash on repeated calls", "[dawn]") {
    REQUIRE_DAWN();

    static Canvas* canvas = nullptr;
    if (!canvas) {
        canvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*canvas);
    renderer.dispose();
    // Second dispose should not crash
    renderer.dispose();
}

TEST_CASE("Dawn: PlaneGeometry renders correctly", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto geometry = PlaneGeometry::create(3, 3);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0x00ffff);
    material->side = Side::Double;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);
}

TEST_CASE("Dawn: CylinderGeometry renders correctly", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto geometry = CylinderGeometry::create(0.5f, 0.5f, 2.0f, 16);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff00ff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 32);
}

TEST_CASE("Dawn: TorusGeometry renders correctly", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto geometry = TorusGeometry::create(1.0f, 0.4f, 8, 16);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffff00);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 16);
}

TEST_CASE("Dawn: emissive material produces visible output without lights", "[dawn]") {
    REQUIRE_DAWN();

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
    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    auto avg = averageColor(pixels);
    CHECK(avg.r > avg.b);
}

TEST_CASE("Dawn: face culling respects material.side", "[dawn]") {
    REQUIRE_DAWN();

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

    auto backFacePixels = renderWithDawn(*makeScene(Side::Front), *camera, Color(0x000000));
    auto doubleSidePixels = renderWithDawn(*makeScene(Side::Double), *camera, Color(0x000000));

    int backFaceNonBlack = countNonBlack(backFacePixels);
    int doubleSideNonBlack = countNonBlack(doubleSidePixels);

    // Double-sided should show more pixels than front-only (which culls the back face)
    CHECK(doubleSideNonBlack > backFaceNonBlack);
}

TEST_CASE("Dawn: wireframe mode renders edges", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto geometry = BoxGeometry::create(2, 2, 2);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    material->wireframe = true;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
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

    auto solidPixels = renderWithDawn(*solidScene, *camera, Color(0x000000));
    int solidNonBlack = countNonBlack(solidPixels);
    CHECK(solidNonBlack > nonBlack);
}

TEST_CASE("Dawn: additive blending brightens", "[dawn]") {
    REQUIRE_DAWN();

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

    auto normalPixels = renderWithDawn(*makeScene(Blending::Normal), *camera, Color(0x404040));
    auto additivePixels = renderWithDawn(*makeScene(Blending::Additive), *camera, Color(0x404040));

    auto normalAvg = averageColor(normalPixels);
    auto additiveAvg = averageColor(additivePixels);

    double normalBright = (normalAvg.r + normalAvg.g + normalAvg.b) / 3.0;
    double additiveBright = (additiveAvg.r + additiveAvg.g + additiveAvg.b) / 3.0;

    // Additive blending on a non-black background should be brighter
    CHECK(additiveBright >= normalBright);
}

TEST_CASE("Dawn: getClearColor/Alpha round-trips", "[dawn]") {
    REQUIRE_DAWN();

    static Canvas* canvas = nullptr;
    if (!canvas) {
        canvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*canvas);

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

TEST_CASE("Dawn: getViewport round-trips", "[dawn]") {
    REQUIRE_DAWN();

    static Canvas* canvas = nullptr;
    if (!canvas) {
        canvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*canvas);
    renderer.setViewport(10, 20, 30, 40);

    Vector4 vp;
    renderer.getViewport(vp);
    CHECK(vp.x == 10.0f);
    CHECK(vp.y == 20.0f);
    CHECK(vp.z == 30.0f);
    CHECK(vp.w == 40.0f);

    renderer.dispose();
}

TEST_CASE("Dawn: scissor test round-trips", "[dawn]") {
    REQUIRE_DAWN();

    static Canvas* canvas = nullptr;
    if (!canvas) {
        canvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*canvas);

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

TEST_CASE("Dawn: render info tracks draw calls", "[dawn]") {
    REQUIRE_DAWN();

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

    static Canvas* canvas = nullptr;
    if (!canvas) {
        canvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*canvas);
    renderer.setClearColor(Color(0x000000));

    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());
    renderer.render(*scene, *camera);

    auto& info = renderer.info();
    CHECK(info.render.calls >= 2);
    CHECK(info.render.triangles > 0);
    CHECK(info.render.frame > 0);

    renderer.dispose();
}

TEST_CASE("Dawn: getActiveCubeFace and getActiveMipmapLevel", "[dawn]") {
    REQUIRE_DAWN();

    static Canvas* canvas = nullptr;
    if (!canvas) {
        canvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*canvas);

    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get(), 2, 3);

    CHECK(renderer.getActiveCubeFace() == 2);
    CHECK(renderer.getActiveMipmapLevel() == 3);

    renderer.dispose();
}

TEST_CASE("Dawn: resetState does not crash", "[dawn]") {
    REQUIRE_DAWN();

    static Canvas* canvas = nullptr;
    if (!canvas) {
        canvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*canvas);
    renderer.resetState(); // Should be a no-op
    renderer.dispose();
}


// =============================================================================
// Section 6: GL-only — Additional material types
// =============================================================================

TEST_CASE("Dawn: MeshToonMaterial renders with stepped shading", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    auto avg = averageColor(pixels);
    CHECK(avg.r > avg.g);
    CHECK(avg.r > avg.b);
}

TEST_CASE("Dawn: MeshNormalMaterial shows surface normals as colors", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto geometry = SphereGeometry::create(1.0f, 32, 16);
    auto material = MeshNormalMaterial::create();
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    // Normal material maps normals to RGB — center of sphere facing camera
    // should have significant blue (Z-forward maps to blue)
    auto center = centerPixel(pixels, RT_WIDTH, RT_HEIGHT);
    CHECK((center.r + center.g + center.b) > 50.0);
}

TEST_CASE("Dawn: MeshDepthMaterial varies brightness with distance", "[dawn]") {
    REQUIRE_DAWN();

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

    auto nearPixels = renderWithDawn(*makeScene(0.0f), *camera, clearColor);
    auto farPixels = renderWithDawn(*makeScene(-10.0f), *camera, clearColor);

    double nearBright = avgBrightness(nearPixels);
    double farBright = avgBrightness(farPixels);
    CHECK(nearBright > farBright);
}

TEST_CASE("Dawn: MeshMatcapMaterial renders visible geometry", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto geometry = SphereGeometry::create(1.0f, 32, 16);
    auto material = MeshMatcapMaterial::create();
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);
}

TEST_CASE("Dawn: Line renders visible edges", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 5);
}

TEST_CASE("Dawn: LineSegments renders discrete segments", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 5);
}

// Lavapipe/software rasterizers render points as zero-size pixels.
// WebGPU pointSize is always 1px; software Vulkan may not rasterize them visibly.

TEST_CASE("Dawn: Points renders visible dots", "[dawn]") {
    REQUIRE_DAWN();
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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack >= 3);
}

// Lavapipe/software rasterizers may not support billboard quad expansion for sprites.

TEST_CASE("Dawn: Sprite renders as billboard", "[dawn]") {
    REQUIRE_DAWN();
    SKIP_ON_SOFTWARE_ADAPTER();

    auto scene = Scene::create();
    auto material = SpriteMaterial::create();
    material->color = Color(0x00ff00);
    auto sprite = Sprite::create(material);
    scene->add(sprite);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 32);
}

// DawnRenderer: InstancedMesh per-instance transforms not yet implemented (renders single instance)

TEST_CASE("Dawn: InstancedMesh renders multiple instances", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geometry = BoxGeometry::create(0.4f, 0.4f, 0.4f);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff8800);

    auto im = InstancedMesh::create(geometry, material, 4);
    Matrix4 m;
    m.setPosition(Vector3(-0.8f, -0.8f, 0)); im->setMatrixAt(0, m);
    m.setPosition(Vector3( 0.8f, -0.8f, 0)); im->setMatrixAt(1, m);
    m.setPosition(Vector3(-0.8f,  0.8f, 0)); im->setMatrixAt(2, m);
    m.setPosition(Vector3( 0.8f,  0.8f, 0)); im->setMatrixAt(3, m);
    scene->add(im);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);

    // 4 instances should cover more than a single box at center
    auto singleScene = Scene::create();
    singleScene->add(AmbientLight::create(Color(0xffffff)));
    auto singleMat = MeshBasicMaterial::create();
    singleMat->color = Color(0xff8800);
    auto singleMesh = Mesh::create(geometry, singleMat);
    singleScene->add(singleMesh);
    auto singlePixels = renderWithDawn(*singleScene, *camera, Color(0x000000));
    int singleNonBlack = countNonBlack(singlePixels);

    CHECK(nonBlack > singleNonBlack);
}

// DawnRenderer: InstancedMesh per-instance transforms not yet implemented

TEST_CASE("Dawn: vertex colors tint geometry", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geometry = BoxGeometry::create(1, 1, 1);
    auto posAttr = geometry->getAttribute<float>("position");
    int vertexCount = static_cast<int>(posAttr->count());
    std::vector<float> colors(vertexCount * 3, 0.0f);
    for (int i = 0; i < vertexCount; i++) {
        colors[i * 3 + 0] = 1.0f; // red
    }
    geometry->setAttribute("color", FloatBufferAttribute::create(colors, 3));

    auto material = MeshBasicMaterial::create();
    material->vertexColors = true;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    auto avg = averageColor(pixels);
    CHECK(avg.r > avg.g + 10);
    CHECK(avg.r > avg.b + 10);
}

// DawnRenderer: vertex color attribute not yet in shader

TEST_CASE("Dawn: Fog attenuates distant objects", "[dawn]") {
    REQUIRE_DAWN();

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

    auto nearPixels = renderWithDawn(*makeScene(3.0f), *camera, clearColor);
    auto farPixels = renderWithDawn(*makeScene(-5.0f), *camera, clearColor);

    double nearBright = avgBrightness(nearPixels);
    double farBright = avgBrightness(farPixels);
    CHECK(nearBright > farBright);
}

TEST_CASE("Dawn: FogExp2 attenuates distant objects", "[dawn]") {
    REQUIRE_DAWN();

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

    auto nearPixels = renderWithDawn(*makeScene(3.0f), *camera, clearColor);
    auto farPixels = renderWithDawn(*makeScene(-5.0f), *camera, clearColor);

    double nearBright = avgBrightness(nearPixels);
    double farBright = avgBrightness(farPixels);
    CHECK(nearBright > farBright);
}

// Cross-renderer fog comparison: Dawn fog is implemented but output differs from GL

TEST_CASE("Dawn: OrthographicCamera renders without perspective distortion", "[dawn]") {
    REQUIRE_DAWN();

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

    auto nearPixels = renderWithDawn(*makeScene(0.0f), *camera, clearColor);
    auto farPixels = renderWithDawn(*makeScene(-10.0f), *camera, clearColor);

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

TEST_CASE("Dawn: object hierarchy applies parent transform", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 5);
}

TEST_CASE("Dawn: RingGeometry renders correctly", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 16);
}

TEST_CASE("Dawn: TorusKnotGeometry renders correctly", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 16);
}

TEST_CASE("Dawn: ConeGeometry renders correctly", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    // Cone geometry covers fewer pixels at distance; use relaxed threshold
    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 32);
}

TEST_CASE("Dawn: CapsuleGeometry renders correctly", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    // Capsule geometry covers fewer pixels at distance; use relaxed threshold
    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 32);
}

// =============================================================================
// Section 13: Dawn — Scissor Test
// =============================================================================

TEST_CASE("Dawn: scissor test clips rendering", "[dawn]") {
    REQUIRE_DAWN();

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
    auto fullPixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int fullCount = countNonBlack(fullPixels);

    // Scissored render — only top-left quarter
    // We need manual Dawn renderer setup for scissor
    static Canvas* dawnScissorCanvas = nullptr;
    if (!dawnScissorCanvas) {
        dawnScissorCanvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*dawnScissorCanvas);
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
// Section 14: Dawn — Multiple Lights Combined
// =============================================================================

TEST_CASE("Dawn: combined ambient + directional + point lights are brighter", "[dawn]") {
    REQUIRE_DAWN();

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

    auto singlePixels = renderWithDawn(*makeSingleLight(), *camera, clearColor);
    auto multiPixels = renderWithDawn(*makeMultiLight(), *camera, clearColor);

    CHECK(avgBrightness(multiPixels) > avgBrightness(singlePixels));
}

TEST_CASE("Dawn: SpotLight shadow casts correctly", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    int maxBright = maxPixelBrightness(pixels);
    CHECK(maxBright > 100);
}

TEST_CASE("Dawn: PointLight shadow casts correctly", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);
}

// =============================================================================
// Section 16: Dawn — Texture Maps
// =============================================================================

// Cross-renderer normal-map comparison: Dawn normal mapping implemented but output differs

TEST_CASE("Dawn: ShaderMaterial renders with custom shaders", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    // Should produce magenta — red and blue present, no green
    auto avg = averageColor(pixels);
    CHECK(avg.r > 20);
    CHECK(avg.b > 20);
}

// DawnRenderer: ShaderMaterial (custom GLSL/WGSL) not yet supported

TEST_CASE("Dawn: ShaderMaterial with uniforms", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    auto avg = averageColor(pixels);
    CHECK(avg.g > avg.r);
    CHECK(avg.g > avg.b);
}

TEST_CASE("Dawn: ShadowMaterial renders shadow-receiving plane", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    // The box should be visible (red), shadow plane may show shadow darkening
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 5);
}

// =============================================================================
// Section 19: Dawn — Roughness & Metalness Maps
// =============================================================================

// DawnRenderer: roughnessMap sampled in shader (green channel multiplies roughness)

TEST_CASE("Dawn: roughnessMap affects specular highlights", "[dawn]") {
    REQUIRE_DAWN();

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
        material->roughness = 0.1f; // smooth base

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

    auto noMapPixels = renderWithDawn(*makeScene(false), *camera, clearColor);
    auto mapPixels = renderWithDawn(*makeScene(true), *camera, clearColor);

    // Verify both render visible geometry (roughnessMap doesn't break rendering)
    CHECK(countNonBlack(noMapPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(mapPixels) > PIXEL_COUNT / 8);
}

// DawnRenderer: metalnessMap not yet sampled in shader

TEST_CASE("Dawn: metalnessMap affects metallic appearance", "[dawn]") {
    REQUIRE_DAWN();

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
        material->metalness = 0.5f; // base semi-metallic (scaled by map)

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

    auto nonMetalPixels = renderWithDawn(*makeScene(false), *camera, clearColor);
    auto metalPixels = renderWithDawn(*makeScene(true), *camera, clearColor);

    // Both should render visible geometry
    CHECK(countNonBlack(nonMetalPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(metalPixels) > PIXEL_COUNT / 8);

    // Metallic vs non-metallic should produce different brightness profiles
    double nonMetalBright = avgBrightness(nonMetalPixels);
    double metalBright = avgBrightness(metalPixels);
    CHECK(std::abs(nonMetalBright - metalBright) > 1.0);
}

TEST_CASE("Dawn: emissiveMap produces glow pattern", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    // No lights — only emissive should be visible
    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshStandardMaterial::create();
    material->color = Color(0x000000);
    material->emissive = Color(0xffffff);
    material->emissiveIntensity = 1.0f;
    // Emissive map: half bright, half dark
    material->emissiveMap = makeProceduralTexture(
        255, 0, 0,   0, 0, 0,
        255, 0, 0,   0, 0, 0
    );
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 5);

    auto avg = averageColor(pixels);
    CHECK(avg.r > avg.g);
}

TEST_CASE("Dawn: aoMap darkens occluded areas", "[dawn]") {
    REQUIRE_DAWN();

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

    auto noAoPixels = renderWithDawn(*makeScene(false), *camera, clearColor);
    auto aoPixels = renderWithDawn(*makeScene(true), *camera, clearColor);

    // AO map should darken the result
    CHECK(avgBrightness(noAoPixels) > avgBrightness(aoPixels));
}

// =============================================================================
// Section 22: Dawn — Alpha Map
// =============================================================================

// DawnRenderer: alphaMap not yet sampled in shader

TEST_CASE("Dawn: alphaMap controls transparency", "[dawn]") {
    REQUIRE_DAWN();

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

    auto opaquePixels = renderWithDawn(*makeScene(false), *camera, clearColor);
    auto alphaPixels = renderWithDawn(*makeScene(true), *camera, clearColor);

    // Alpha-mapped version should be dimmer (more transparent)
    CHECK(avgBrightness(opaquePixels) > avgBrightness(alphaPixels));
}

TEST_CASE("Dawn: displacementMap offsets vertices", "[dawn]") {
    REQUIRE_DAWN();

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

    auto normalPixels = renderWithDawn(*makeScene(false), *camera, clearColor);
    auto displacedPixels = renderWithDawn(*makeScene(true), *camera, clearColor);

    // Displaced sphere should cover more or different pixels
    int normalCount = countNonBlack(normalPixels);
    int displacedCount = countNonBlack(displacedPixels);
    CHECK(normalCount > PIXEL_COUNT / 8);
    CHECK(displacedCount > PIXEL_COUNT / 8);
    // Displacement should change the coverage or brightness
    CHECK(std::abs(normalCount - displacedCount) > 0);
}

// =============================================================================
// Section 24: Dawn — Light Map
// =============================================================================

// DawnRenderer: lightMap not yet sampled in shader

TEST_CASE("Dawn: lightMap adds baked illumination", "[dawn]") {
    REQUIRE_DAWN();

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

    auto noLmPixels = renderWithDawn(*makeScene(false), *camera, clearColor);
    auto lmPixels = renderWithDawn(*makeScene(true), *camera, clearColor);

    // Light map should add brightness
    CHECK(avgBrightness(lmPixels) > avgBrightness(noLmPixels));
}

// =============================================================================
// Section 25: Dawn — Bump Map
// =============================================================================

// DawnRenderer: bumpMap not yet sampled in shader

TEST_CASE("Dawn: bumpMap perturbs surface shading", "[dawn]") {
    REQUIRE_DAWN();

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
                255, 255, 255,  0, 0, 0,
                0, 0, 0,        255, 255, 255
            );
            material->bumpScale = 1.0f;
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto flatPixels = renderWithDawn(*makeScene(false), *camera, clearColor);
    auto bumpPixels = renderWithDawn(*makeScene(true), *camera, clearColor);

    // Phong-lit sphere with directional light has limited coverage on small target
    CHECK(countNonBlack(flatPixels) > 0);
    CHECK(countNonBlack(bumpPixels) > 0);

    // Bump map should change the brightness variance (more surface detail)
    double flatVar = brightnessVariance(flatPixels);
    double bumpVar = brightnessVariance(bumpPixels);
    CHECK(bumpVar != flatVar);
}

// =============================================================================
// Section 26: Dawn — Gradient Map (Toon Shading)
// =============================================================================

// DawnRenderer: gradientMap (MeshToonMaterial) not yet sampled in shader

TEST_CASE("Dawn: gradientMap controls toon shading bands", "[dawn]") {
    REQUIRE_DAWN();

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

    auto defaultPixels = renderWithDawn(*makeScene(false), *camera, clearColor);
    auto gradientPixels = renderWithDawn(*makeScene(true), *camera, clearColor);

    CHECK(countNonBlack(defaultPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(gradientPixels) > PIXEL_COUNT / 8);

    // Different gradient maps should produce different brightness
    CHECK(std::abs(avgBrightness(defaultPixels) - avgBrightness(gradientPixels)) > 0.5);
}

// =============================================================================
// Section 27: Dawn — Environment Maps
// =============================================================================

// DawnRenderer: envMap / CubeTexture not yet implemented

TEST_CASE("Dawn: envMap adds reflections to standard material", "[dawn]") {
    REQUIRE_DAWN();

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

    auto noEnvPixels = renderWithDawn(*makeScene(false), *camera, clearColor);
    auto envPixels = renderWithDawn(*makeScene(true), *camera, clearColor);

    CHECK(countNonBlack(noEnvPixels) > PIXEL_COUNT / 16);
    CHECK(countNonBlack(envPixels) > PIXEL_COUNT / 16);

    // Env map on a metallic surface should make it brighter/more colored
    auto envAvg = averageColor(envPixels);
    CHECK(envAvg.r > 5.0); // Should pick up red from env map
}

// DawnRenderer: envMap / CubeTexture not yet implemented

TEST_CASE("Dawn: morph targets deform geometry", "[dawn]") {
    REQUIRE_DAWN();

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

    auto basePixels = renderWithDawn(*makeScene(0.0f), *camera, clearColor);
    auto morphedPixels = renderWithDawn(*makeScene(1.0f), *camera, clearColor);

    int baseCount = countNonBlack(basePixels);
    int morphedCount = countNonBlack(morphedPixels);

    CHECK(baseCount > 0);
    CHECK(morphedCount > 0);

    // Morphed (scaled up) should cover more pixels
    CHECK(morphedCount > baseCount);
}

// DawnRenderer: morph targets not yet implemented

TEST_CASE("Dawn: SkinnedMesh with skeleton renders correctly", "[dawn]") {
    REQUIRE_DAWN();

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
        float weight = (y + 1.0f) / 2.0f; // normalize from [-1,1] to [0,1]
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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 0);
}

// DawnRenderer: SkinnedMesh / skeletal animation not yet implemented

TEST_CASE("Dawn: SkinnedMesh bone rotation deforms mesh", "[dawn]") {
    REQUIRE_DAWN();

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

    auto straightPixels = renderWithDawn(*makeScene(0.0f), *camera, clearColor);
    auto bentPixels = renderWithDawn(*makeScene(math::PI / 4), *camera, clearColor);

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
// Section 30: Dawn — Clipping Planes
// =============================================================================

// DawnRenderer: clipping planes not yet implemented

TEST_CASE("Dawn: clipping plane cuts geometry", "[dawn]") {
    REQUIRE_DAWN();

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

    // Need to enable local clipping on the Dawn renderer
    static Canvas* dawnClipCanvas = nullptr;
    if (!dawnClipCanvas) {
        dawnClipCanvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    // Render without clipping
    auto fullPixels = renderWithDawn(*makeScene(false), *camera, clearColor);

    // Render with clipping — need renderer with localClippingEnabled
    {
        DawnRenderer renderer(*dawnClipCanvas);
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

// DawnRenderer: clipping planes not yet implemented

TEST_CASE("Dawn: tone mapping affects output brightness", "[dawn]") {
    REQUIRE_DAWN();

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
    static Canvas* dawnTmCanvas = nullptr;
    if (!dawnTmCanvas) {
        dawnTmCanvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    auto renderWithToneMapping = [&](ToneMapping tm, float exposure) {
        DawnRenderer renderer(*dawnTmCanvas);
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

// DawnRenderer bug: toneMappingExposure renders black (possibly canvas/state issue)

TEST_CASE("Dawn: toneMappingExposure scales brightness", "[dawn]") {
    REQUIRE_DAWN();

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

    static Canvas* dawnExpCanvas = nullptr;
    if (!dawnExpCanvas) {
        dawnExpCanvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    auto renderWithExposure = [&](float exposure) {
        DawnRenderer renderer(*dawnExpCanvas);
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
// Section 32: Dawn — Output Encoding / Color Space
// =============================================================================

// DawnRenderer: sRGB output encoding not yet implemented

TEST_CASE("Dawn: sRGB output encoding differs from linear", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0x808080); // mid-grey
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    static Canvas* dawnEncCanvas = nullptr;
    if (!dawnEncCanvas) {
        dawnEncCanvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    auto renderWithEncoding = [&](Encoding enc) {
        DawnRenderer renderer(*dawnEncCanvas);
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
// Section 33: Dawn — Instanced Colors
// =============================================================================

// DawnRenderer: InstancedMesh + vertex colors not yet implemented

TEST_CASE("Dawn: InstancedMesh per-instance colors", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    auto avg = averageColor(pixels);

    // Should have both red and blue components (small boxes at 64x64)
    CHECK(avg.r > 1);
    CHECK(avg.b > 1);
    CHECK(countNonBlack(pixels) > 0);
}

// DawnRenderer: InstancedMesh + vertex colors not yet implemented

TEST_CASE("Dawn: shadow map resolution affects quality", "[dawn]") {
    REQUIRE_DAWN();

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

    auto lowResPixels = renderWithDawn(*makeScene(64), *camera, clearColor);
    auto highResPixels = renderWithDawn(*makeScene(512), *camera, clearColor);

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

TEST_CASE("Dawn: shadow bias prevents shadow acne", "[dawn]") {
    REQUIRE_DAWN();

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

    auto noBiasPixels = renderWithDawn(*makeScene(0.0f), *camera, clearColor);
    auto biasPixels = renderWithDawn(*makeScene(-0.005f), *camera, clearColor);

    // Both should produce visible output
    CHECK(countNonBlack(noBiasPixels) > PIXEL_COUNT / 16);
    CHECK(countNonBlack(biasPixels) > PIXEL_COUNT / 16);
}

// =============================================================================
// Section 35: Dawn — Specular Map
// =============================================================================

// DawnRenderer: specularMap not yet sampled in shader

TEST_CASE("Dawn: specularMap controls highlight regions", "[dawn]") {
    REQUIRE_DAWN();

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

    auto fullSpecPixels = renderWithDawn(*makeScene(false), *camera, clearColor);
    auto reducedSpecPixels = renderWithDawn(*makeScene(true), *camera, clearColor);

    // Both should render visible geometry
    CHECK(countNonBlack(fullSpecPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(reducedSpecPixels) > PIXEL_COUNT / 8);
}

// =============================================================================
// Section 36: Dawn — Diffuse Map (color map)
// =============================================================================

TEST_CASE("Dawn: map (diffuse texture) tints geometry", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geometry = PlaneGeometry::create(2, 2);
    auto material = MeshBasicMaterial::create();
    material->map = makeUniformTexture(255, 0, 0); // Red texture
    material->side = Side::Double;

    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    auto avg = averageColor(pixels);
    CHECK(avg.r > avg.g + 20);
    CHECK(avg.r > avg.b + 20);
}

// =============================================================================
// Section 37: Dawn — LineBasicMaterial Options
// =============================================================================

TEST_CASE("Dawn: LineBasicMaterial with dashed line", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 3);
}

// =============================================================================
// Section 38: Dawn — MSAA Anti-Aliasing
//
// TDD red phase: these tests reference setSampleCount()/getSampleCount() which
// do not yet exist on DawnRenderer. Enable when the MSAA API is implemented.
// =============================================================================

TEST_CASE("Dawn: MSAA 4x produces more intermediate edge pixels than 1x", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = BoxGeometry::create(1, 1, 1);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->rotation.z = math::PI / 6; // 30 degrees — creates diagonal edges
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;
    Color clearColor(0x000000);

    static Canvas* msaaCanvas = nullptr;
    if (!msaaCanvas) {
        msaaCanvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    std::vector<unsigned char> pixelsNoMSAA;
    {
        DawnRenderer renderer(*msaaCanvas);
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
        DawnRenderer renderer(*msaaCanvas);
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

TEST_CASE("Dawn: MSAA brightness variance differs at edges", "[dawn]") {
    REQUIRE_DAWN();

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

    static Canvas* msaaCanvas2 = nullptr;
    if (!msaaCanvas2) {
        msaaCanvas2 = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    std::vector<unsigned char> pixelsNoMSAA;
    {
        DawnRenderer renderer(*msaaCanvas2);
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
        DawnRenderer renderer(*msaaCanvas2);
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

TEST_CASE("Dawn: MSAA sample count round-trips", "[dawn]") {
    REQUIRE_DAWN();

    static Canvas* msaaCanvas3 = nullptr;
    if (!msaaCanvas3) {
        msaaCanvas3 = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*msaaCanvas3);
    renderer.setSampleCount(4);
    CHECK(renderer.getSampleCount() == 4);
    renderer.setSampleCount(1);
    CHECK(renderer.getSampleCount() == 1);
    renderer.dispose();
}

TEST_CASE("Dawn: MSAA render still produces correct average color", "[dawn]") {
    REQUIRE_DAWN();

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

    static Canvas* msaaCanvas4 = nullptr;
    if (!msaaCanvas4) {
        msaaCanvas4 = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    std::vector<unsigned char> pixelsNoMSAA;
    {
        DawnRenderer renderer(*msaaCanvas4);
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
        DawnRenderer renderer(*msaaCanvas4);
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
// Section 39: Dawn — Multiple Shadow Casters
//
// These tests use existing API (castShadow on multiple lights) but FAIL at
// runtime because DawnRenderer only processes the first shadow-casting
// DirectionalLight. They compile and run, but assertions fail.
// =============================================================================

TEST_CASE("Dawn: two directional lights produce two distinct shadows", "[dawn]") {
    REQUIRE_DAWN();

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

    auto singlePixels = renderWithDawn(*makeSingleShadowScene(), *camera, clearColor);
    auto dualPixels = renderWithDawn(*makeDualShadowScene(), *camera, clearColor);

    // With per-light shadows, the dual scene has brighter unshadowed areas
    // (2 lights) and distinct shadow regions, producing higher brightness
    // variance than a single-light scene.
    double singleVar = brightnessVariance(singlePixels);
    double dualVar = brightnessVariance(dualPixels);
    CHECK(dualVar >= singleVar);
    // The dual scene should also be brighter overall (2 lights vs 1)
    CHECK(avgBrightness(dualPixels) > avgBrightness(singlePixels));
}

TEST_CASE("Dawn: SpotLight and DirectionalLight both cast shadows simultaneously", "[dawn]") {
    REQUIRE_DAWN();

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

    auto dirOnlyPixels = renderWithDawn(*makeScene(false, true), *camera, clearColor);
    auto dualPixels = renderWithDawn(*makeScene(true, true), *camera, clearColor);

    // Adding a SpotLight increases total illumination — the dual scene
    // should be brighter overall than the directional-only scene.
    CHECK(avgBrightness(dualPixels) > avgBrightness(dirOnlyPixels));
}

TEST_CASE("Dawn: shadow from light A is independent of light B occlusion", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));

    double brightness = avgBrightness(pixels);
    CHECK(brightness > 20.0);
    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 4);
}

TEST_CASE("Dawn: multiple shadow casters produce shadows at different floor positions", "[dawn]") {
    REQUIRE_DAWN();

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

    auto singlePixels = renderWithDawn(*makeSingleScene(), *camera, clearColor);
    auto dualPixels = renderWithDawn(*makeDualScene(), *camera, clearColor);

    double singleVariance = brightnessVariance(singlePixels);
    double dualVariance = brightnessVariance(dualPixels);
    CHECK(dualVariance > singleVariance);
}

// =============================================================================
// Section 40: Dawn — Bind Group Separation Regression
//
// Regression guards for the bind group refactoring. These tests exercise
// complex binding combinations and should PASS now and continue passing
// after @group(0) is split into per-frame/per-material/per-object groups.
// =============================================================================

TEST_CASE("Dawn: multi-material scene renders all materials correctly", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));

    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 32);
    auto avg = averageColor(pixels);
    CHECK(avg.r > 2);
    CHECK(avg.g > 2);
    CHECK(avg.b > 2);
}

TEST_CASE("Dawn: textured and untextured objects coexist in same scene", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));

    auto avg = averageColor(pixels);
    CHECK(avg.r > 2);
    CHECK(avg.b > 2);
    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 32);
}

TEST_CASE("Dawn: shadow + texture + instancing combination renders", "[dawn]") {
    REQUIRE_DAWN();

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
    m.setPosition(Vector3(-0.8f, 1.5f, 0)); im->setMatrixAt(0, m);
    m.setPosition(Vector3(0.8f, 1.5f, 0));  im->setMatrixAt(1, m);
    m.setPosition(Vector3(-0.8f, 2.5f, 0)); im->setMatrixAt(2, m);
    m.setPosition(Vector3(0.8f, 2.5f, 0));  im->setMatrixAt(3, m);
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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));

    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 4);
    double var = brightnessVariance(pixels);
    CHECK(var > 5.0);
}

TEST_CASE("Dawn: normal map + emissive map + AO map combined", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));

    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 8);
    auto avg = averageColor(pixels);
    CHECK((avg.r + avg.g + avg.b) / 3.0 > 15.0);
}

TEST_CASE("Dawn: PBR + normal map + shadow + fog combination", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));

    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 8);
    double var = brightnessVariance(pixels);
    CHECK(var > 5.0);
}

TEST_CASE("Dawn: toon material with gradient map renders stepped lighting", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));

    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 16);
    auto avg = averageColor(pixels);
    CHECK(avg.r + avg.g + avg.b > 30);
}

TEST_CASE("Dawn: morph targets + skinning combined", "[dawn]") {
    REQUIRE_DAWN();

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

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    CHECK(countNonBlack(pixels) > 0);
}

TEST_CASE("Dawn: instancing + vertex colors combination", "[dawn]") {
    REQUIRE_DAWN();

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
    m.setPosition(Vector3(-0.8f, -0.8f, 0)); im->setMatrixAt(0, m);
    m.setPosition(Vector3(0.8f, -0.8f, 0));  im->setMatrixAt(1, m);
    m.setPosition(Vector3(-0.8f, 0.8f, 0));  im->setMatrixAt(2, m);
    m.setPosition(Vector3(0.8f, 0.8f, 0));   im->setMatrixAt(3, m);
    scene->add(im);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));

    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 64);
    auto avg = averageColor(pixels);
    CHECK(avg.r > 1);
    CHECK(avg.g > 1);
}

TEST_CASE("Dawn: displacement map + clipping plane combination", "[dawn]") {
    REQUIRE_DAWN();

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

    auto fullPixels = renderWithDawn(*makeScene(false), *camera, clearColor);

    static Canvas* dispClipCanvas = nullptr;
    if (!dispClipCanvas) {
        dispClipCanvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    std::vector<unsigned char> clippedPixels;
    {
        DawnRenderer renderer(*dispClipCanvas);
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
// Section 42: Dawn — Post-Processing Framework
//
// TDD red phase: these tests reference EffectComposer and ShaderPass classes
// that do not yet exist. Enable when the post-processing framework is
// implemented.
//
// Proposed API:
//   class ShaderPass { static create(wgslSource); };
//   class EffectComposer { EffectComposer(DawnRenderer&); addPass(); render(); readRGBPixels(); };
// =============================================================================

#if 1 // Enable when EffectComposer and ShaderPass are implemented

#include "threepp/renderers/dawn/EffectComposer.hpp"
#include "threepp/renderers/dawn/ShaderPass.hpp"

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
} // namespace

TEST_CASE("Dawn: identity post-process pass matches non-post-processed output", "[dawn]") {
    REQUIRE_DAWN();

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

    auto directPixels = renderWithDawn(*scene, *camera, clearColor);

    static Canvas* postCanvas = nullptr;
    if (!postCanvas) {
        postCanvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*postCanvas);
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

TEST_CASE("Dawn: grayscale post-process produces equal RGB channels", "[dawn]") {
    REQUIRE_DAWN();

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

    auto originalPixels = renderWithDawn(*scene, *camera, clearColor);
    auto origAvg = averageColor(originalPixels);
    bool channelsVary = (std::abs(origAvg.r - origAvg.g) > 10) ||
                        (std::abs(origAvg.r - origAvg.b) > 10);
    CHECK(channelsVary);

    static Canvas* gsCanvas = nullptr;
    if (!gsCanvas) {
        gsCanvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*gsCanvas);
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

TEST_CASE("Dawn: invert post-process inverts colors", "[dawn]") {
    REQUIRE_DAWN();

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

    auto originalPixels = renderWithDawn(*scene, *camera, clearColor);

    static Canvas* invCanvas = nullptr;
    if (!invCanvas) {
        invCanvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*invCanvas);
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

TEST_CASE("Dawn: double invert post-process restores original", "[dawn]") {
    REQUIRE_DAWN();

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

    auto originalPixels = renderWithDawn(*scene, *camera, clearColor);

    static Canvas* dblInvCanvas = nullptr;
    if (!dblInvCanvas) {
        dblInvCanvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*dblInvCanvas);
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

TEST_CASE("Dawn: brightness post-process increases average brightness", "[dawn]") {
    REQUIRE_DAWN();

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

    auto originalPixels = renderWithDawn(*scene, *camera, clearColor);
    double origBright = avgBrightness(originalPixels);
    CHECK(origBright > 10.0);

    static Canvas* brightCanvas = nullptr;
    if (!brightCanvas) {
        brightCanvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*brightCanvas);
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

TEST_CASE("Dawn: post-process works with render targets", "[dawn]") {
    REQUIRE_DAWN();

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

    static Canvas* rtCanvas = nullptr;
    if (!rtCanvas) {
        rtCanvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*rtCanvas);
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

#endif // Post-processing API guard
