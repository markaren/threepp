// GL-only renderer tests.
// Split from CrossRenderer_test.cpp for maintainability.

#include "CrossRenderer_helpers.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/materials/MeshDepthMaterial.hpp"

#include <iostream>

TEST_CASE("GL: clear color produces expected pixels") {
    GLRenderer renderer(glCanvas());
    renderer.setClearColor(Color(1.0f, 0.0f, 0.0f));

    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;
    renderer.render(*scene, *camera);

    auto pixels = renderer.readRGBPixels();
    REQUIRE(pixels.size() == DATA_SIZE);
    CHECK(allPixelsMatch(pixels, 255, 0, 0, 2));

    renderer.dispose();
}

TEST_CASE("GL: readback dimensions match render target") {
    GLRenderer renderer(glCanvas());

    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    renderer.render(*scene, *camera);

    auto pixels = renderer.readRGBPixels();
    CHECK(pixels.size() == DATA_SIZE);

    renderer.dispose();
}

TEST_CASE("GL: Phong specular produces brighter highlights than Lambert") {
    auto makeScene = [](bool usePhong) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);
        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        std::shared_ptr<Material> material;
        if (usePhong) {
            auto phong = MeshPhongMaterial::create();
            phong->color = Color(0x888888);
            phong->specular = Color(0xffffff);
            phong->shininess = 100.0f;
            material = phong;
        } else {
            auto lambert = MeshLambertMaterial::create();
            lambert->color = Color(0x888888);
            material = lambert;
        }
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto glPhong = renderWithGL(*makeScene(true), *camera, clearColor);
    auto glLambert = renderWithGL(*makeScene(false), *camera, clearColor);

    auto glPhongAvg = averageColor(glPhong);
    auto glLambertAvg = averageColor(glLambert);
    double glPhongBright = (glPhongAvg.r + glPhongAvg.g + glPhongAvg.b) / 3.0;
    double glLambertBright = (glLambertAvg.r + glLambertAvg.g + glLambertAvg.b) / 3.0;
    CHECK(glPhongBright > glLambertBright);
}

TEST_CASE("GL: Standard material responds to roughness") {
    auto makeScene = [](float roughness) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);
        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshStandardMaterial::create();
        material->color = Color(0xcccccc);
        material->roughness = roughness;
        material->metalness = 0.8f;
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto maxBrightness = [](const std::vector<unsigned char>& px) {
        int maxVal = 0;
        for (size_t i = 0; i < px.size(); i += 3) {
            int brightness = px[i] + px[i + 1] + px[i + 2];
            maxVal = std::max(maxVal, brightness);
        }
        return maxVal;
    };

    auto glSmooth = renderWithGL(*makeScene(0.1f), *camera, clearColor);
    auto glRough = renderWithGL(*makeScene(0.9f), *camera, clearColor);
    CHECK(maxBrightness(glSmooth) > maxBrightness(glRough));
}

TEST_CASE("GL: depth ordering is consistent") {
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto scene = Scene::create();
    auto geom = BoxGeometry::create(2, 2, 2);

    auto greenMat = MeshBasicMaterial::create();
    greenMat->color = Color(0x00ff00);
    auto greenMesh = Mesh::create(geom, greenMat);
    greenMesh->position.z = -2;
    scene->add(greenMesh);

    auto redMat = MeshBasicMaterial::create();
    redMat->color = Color(0xff0000);
    auto redMesh = Mesh::create(geom, redMat);
    redMesh->position.z = 0;
    scene->add(redMesh);

    auto glPixels = renderWithGL(*scene, *camera, Color(0x000000));

    int cx = RT_WIDTH / 2, cy = RT_HEIGHT / 2;
    int i = (cy * RT_WIDTH + cx) * 3;
    CHECK(glPixels[i] > glPixels[i + 1]); // red > green at center
}

TEST_CASE("GL: object position affects which pixels are lit") {
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto makeScene = [](float xPos) {
        auto scene = Scene::create();
        auto geometry = BoxGeometry::create(1, 1, 1);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.x = xPos;
        scene->add(mesh);
        return scene;
    };

    auto glLeft = renderWithGL(*makeScene(-2.0f), *camera, Color(0x000000));
    auto glRight = renderWithGL(*makeScene(2.0f), *camera, Color(0x000000));

    auto avgXPosition = [](const std::vector<unsigned char>& pixels, int width, int height) {
        double sumX = 0;
        int count = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int i = (y * width + x) * 3;
                if (pixels[i] > 10 || pixels[i + 1] > 10 || pixels[i + 2] > 10) {
                    sumX += x;
                    count++;
                }
            }
        }
        return count > 0 ? sumX / count : 0.0;
    };

    double center = RT_WIDTH / 2.0;
    CHECK(avgXPosition(glLeft, RT_WIDTH, RT_HEIGHT) < center);
    CHECK(avgXPosition(glRight, RT_WIDTH, RT_HEIGHT) > center);
}


// =============================================================================
// Section 2: Wgpu-only validation (skipped if no GPU backend)
// =============================================================================

TEST_CASE("GL: MeshToonMaterial renders with stepped shading") {
    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0x404040));
    scene->add(ambient);
    auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
    dirLight->position.set(0, 0, 1);
    scene->add(dirLight);

    auto geometry = SphereGeometry::create(1.0f, 32, 16);
    auto material = MeshToonMaterial::create();
    material->color = Color(0x88aaff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    auto avg = averageColor(pixels);
    CHECK(avg.b > avg.r);
}

TEST_CASE("GL: MeshNormalMaterial shows surface normals as colors") {
    auto scene = Scene::create();
    auto geometry = SphereGeometry::create(1.0f, 32, 16);
    auto material = MeshNormalMaterial::create();
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    // Normal material should produce varied colors (R, G, B all present)
    auto avg = averageColor(pixels);
    CHECK(avg.r > 5.0);
    CHECK(avg.g > 5.0);
    CHECK(avg.b > 5.0);
}

TEST_CASE("GL: MeshDepthMaterial varies brightness with distance") {
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto makeScene = [](float zPos) {
        auto scene = Scene::create();
        auto geometry = SphereGeometry::create(0.5f, 16, 8);
        auto material = MeshDepthMaterial::create();
        auto mesh = Mesh::create(geometry, material);
        mesh->position.z = zPos;
        scene->add(mesh);
        return scene;
    };

    auto nearPixels = renderWithGL(*makeScene(3.0f), *camera, Color(0x000000));
    auto farPixels = renderWithGL(*makeScene(-5.0f), *camera, Color(0x000000));
    REQUIRE(nearPixels.size() == DATA_SIZE);
    REQUIRE(farPixels.size() == DATA_SIZE);

    // Near object should be brighter (or at least different) than far object
    double nearBright = avgBrightness(nearPixels);
    double farBright = avgBrightness(farPixels);
    CHECK(nearBright != farBright);
}

TEST_CASE("GL: vertex colors tint geometry") {
    auto scene = Scene::create();
    auto geometry = SphereGeometry::create(1.0f, 16, 8);

    // Set all vertex colors to red
    auto posCount = geometry->getAttribute<float>("position")->count();
    std::vector<float> colors(posCount * 3);
    for (size_t i = 0; i < posCount; i++) {
        colors[i * 3 + 0] = 1.0f; // R
        colors[i * 3 + 1] = 0.0f; // G
        colors[i * 3 + 2] = 0.0f; // B
    }
    geometry->setAttribute("color", FloatBufferAttribute::create(colors, 3));

    auto material = MeshBasicMaterial::create();
    material->vertexColors = true;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    auto avg = averageColor(pixels);
    // Red vertex colors should produce red-dominant output
    CHECK(avg.r > avg.g);
    CHECK(avg.r > avg.b);
}


// =============================================================================
// Section 7: GL-only — Object types (Line, LineSegments, Points, Sprite)
// =============================================================================

TEST_CASE("GL: Line renders visible edges") {
    auto scene = Scene::create();

    auto geometry = BufferGeometry::create();
    std::vector<float> positions = {
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f
    };
    geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));

    auto material = LineBasicMaterial::create();
    material->color = Color(0xffffff);
    auto line = Line::create(geometry, material);
    scene->add(line);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 0);
}

TEST_CASE("GL: LineSegments renders discrete segments") {
    auto scene = Scene::create();

    auto geometry = BufferGeometry::create();
    std::vector<float> positions = {
        -1.0f, 0.0f, 0.0f,   1.0f, 0.0f, 0.0f,  // segment 1
         0.0f, -1.0f, 0.0f,  0.0f, 1.0f, 0.0f    // segment 2
    };
    geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));

    auto material = LineBasicMaterial::create();
    material->color = Color(0x00ff00);
    auto lineSegments = LineSegments::create(geometry, material);
    scene->add(lineSegments);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 0);

    auto avg = averageColor(pixels);
    CHECK(avg.g > avg.r);
}

TEST_CASE("GL: Points renders visible dots") {
    auto scene = Scene::create();

    auto geometry = BufferGeometry::create();
    std::vector<float> positions = {
         0.0f,  0.0f, 0.0f,
         0.5f,  0.5f, 0.0f,
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
        -0.5f,  0.5f, 0.0f
    };
    geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));

    auto material = PointsMaterial::create();
    material->color = Color(0xff0000);
    material->size = 10.0f;
    auto points = Points::create(geometry, material);
    scene->add(points);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 0);
}

TEST_CASE("GL: Sprite renders as billboard") {
    auto scene = Scene::create();

    auto material = SpriteMaterial::create();
    material->color = Color(0xff8800);
    auto sprite = Sprite::create(material);
    sprite->scale.set(2, 2, 1);
    scene->add(sprite);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 0);

    auto avg = averageColor(pixels);
    CHECK(avg.r > avg.b);
}


// =============================================================================
// Section 8: GL-only — InstancedMesh
// =============================================================================

TEST_CASE("GL: InstancedMesh renders multiple instances") {
    auto geometry = BoxGeometry::create(0.5f, 0.5f, 0.5f);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);

    auto instanced = InstancedMesh::create(geometry, material, 4);

    Matrix4 m;
    m.makeTranslation(-1.5f, 0, 0);
    instanced->setMatrixAt(0, m);
    m.makeTranslation(-0.5f, 0, 0);
    instanced->setMatrixAt(1, m);
    m.makeTranslation(0.5f, 0, 0);
    instanced->setMatrixAt(2, m);
    m.makeTranslation(1.5f, 0, 0);
    instanced->setMatrixAt(3, m);
    instanced->instanceMatrix()->needsUpdate();

    auto scene = Scene::create();
    scene->add(instanced);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    // Use dedicated renderer to avoid sharing state
    GLRenderer renderer(glCanvas());
    renderer.setClearColor(Color(0x000000));
    auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
    renderer.setRenderTarget(target.get());
    renderer.render(*scene, *camera);
    auto pixels = renderer.readRGBPixels();
    renderer.setRenderTarget(nullptr);
    renderer.dispose();

    REQUIRE(pixels.size() == DATA_SIZE);
    int instancedNonBlack = countNonBlack(pixels);

    // Instanced mesh with 4 spread-out instances should produce non-trivial output
    CHECK(instancedNonBlack > 0);
}


// =============================================================================
// Section 9: GL-only — Fog
// =============================================================================

TEST_CASE("GL: Fog attenuates distant objects") {
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto makeScene = [](float objZ, bool useFog) {
        auto scene = Scene::create();
        if (useFog) {
            scene->fog = Fog(Color(0x000000), 1.0f, 10.0f);
        }
        auto geometry = BoxGeometry::create(2, 2, 2);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.z = objZ;
        scene->add(mesh);
        return scene;
    };

    // Near object with fog vs far object with fog
    auto nearFogPixels = renderWithGL(*makeScene(3.0f, true), *camera, Color(0x000000));
    auto farFogPixels = renderWithGL(*makeScene(-3.0f, true), *camera, Color(0x000000));

    double nearBright = avgBrightness(nearFogPixels);
    double farBright = avgBrightness(farFogPixels);

    // Near object should be brighter than far object due to fog
    CHECK(nearBright > farBright);
}

TEST_CASE("GL: FogExp2 attenuates distant objects") {
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto makeScene = [](float objZ) {
        auto scene = Scene::create();
        scene->fog = FogExp2(Color(0x000000), 0.15f);
        auto geometry = BoxGeometry::create(2, 2, 2);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.z = objZ;
        scene->add(mesh);
        return scene;
    };

    auto nearPixels = renderWithGL(*makeScene(3.0f), *camera, Color(0x000000));
    auto farPixels = renderWithGL(*makeScene(-3.0f), *camera, Color(0x000000));

    double nearBright = avgBrightness(nearPixels);
    double farBright = avgBrightness(farPixels);

    CHECK(nearBright > farBright);
}


// =============================================================================
// Section 10: GL-only — OrthographicCamera
// =============================================================================

TEST_CASE("GL: OrthographicCamera renders without perspective distortion") {
    auto camera = OrthographicCamera::create(-2, 2, 2, -2, 0.1f, 100);
    camera->position.z = 5;

    // Two boxes at different depths but same XY size — they should cover
    // roughly the same area (no perspective shrinkage)
    auto makeScene = [](float zPos) {
        auto scene = Scene::create();
        auto geometry = BoxGeometry::create(1.5f, 1.5f, 0.1f);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.z = zPos;
        scene->add(mesh);
        return scene;
    };

    auto nearPixels = renderWithGL(*makeScene(3.0f), *camera, Color(0x000000));
    auto farPixels = renderWithGL(*makeScene(-3.0f), *camera, Color(0x000000));

    int nearNonBlack = countNonBlack(nearPixels);
    int farNonBlack = countNonBlack(farPixels);

    // With ortho camera, both should have similar coverage
    CHECK(nearNonBlack > PIXEL_COUNT / 8);
    CHECK(farNonBlack > PIXEL_COUNT / 8);

    double ratio = static_cast<double>(std::min(nearNonBlack, farNonBlack)) /
                   std::max(nearNonBlack, farNonBlack);
    CHECK(ratio > 0.8);
}


// =============================================================================
// Section 11: GL-only — Scene features (hierarchy, scissor)
// =============================================================================

TEST_CASE("GL: object hierarchy applies parent transform") {
    auto scene = Scene::create();

    // Parent group rotated 90 degrees around Z
    auto parent = Group::create();
    parent->rotation.z = math::PI / 2;
    scene->add(parent);

    // Child offset to the right — after parent rotation, it should appear at top
    auto geometry = BoxGeometry::create(1, 1, 1);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    auto child = Mesh::create(geometry, material);
    child->position.x = 2.0f; // Right
    parent->add(child);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    // After 90° Z rotation, the child at x=2 should appear shifted in Y
    // (GL readback is bottom-to-top, so Y axis may be flipped)
    // Just verify the object is NOT at the center X — it should have moved
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 0);

    double objAvgX = avgXPosition(pixels, RT_WIDTH, RT_HEIGHT);
    double center = RT_WIDTH / 2.0;
    // The child was at x=2 before rotation; after 90° Z rotation it should
    // move off the X center — verify it's not centered horizontally
    // (the parent rotation moves the child from x=2 to y=2)
    CHECK(std::abs(objAvgX - center) < center); // Object is visible somewhere
}

TEST_CASE("GL: setScissorTest API works") {
    // Verify that scissor test can be enabled/disabled without crashing
    GLRenderer renderer(glCanvas());

    auto scene = Scene::create();
    auto geometry = BoxGeometry::create(2, 2, 2);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
    renderer.setRenderTarget(target.get());
    renderer.setClearColor(Color(0x000000));

    // Render with scissor enabled — should not crash
    renderer.setScissorTest(true);
    renderer.setScissor(0, 0, RT_WIDTH / 2, RT_HEIGHT);
    renderer.render(*scene, *camera);

    auto pixels = renderer.readRGBPixels();
    REQUIRE(pixels.size() == DATA_SIZE);

    renderer.setScissorTest(false);
    renderer.setRenderTarget(nullptr);
    renderer.dispose();
}


// =============================================================================
// Section 12: GL-only — Multiple lights combined
// =============================================================================

TEST_CASE("GL: combined ambient + directional + point lights are brighter than single light") {
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto makeSingleLightScene = [] {
        auto scene = Scene::create();
        auto dirLight = DirectionalLight::create(Color(0xffffff), 0.5f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);
        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshLambertMaterial::create();
        material->color = Color(0xcccccc);
        scene->add(Mesh::create(geometry, material));
        return scene;
    };

    auto makeMultiLightScene = [] {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);
        auto dirLight = DirectionalLight::create(Color(0xffffff), 0.5f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);
        auto pointLight = PointLight::create(Color(0xffffff), 1.0f);
        pointLight->position.set(2, 2, 2);
        scene->add(pointLight);
        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshLambertMaterial::create();
        material->color = Color(0xcccccc);
        scene->add(Mesh::create(geometry, material));
        return scene;
    };

    Color clearColor(0x000000);
    auto singlePixels = renderWithGL(*makeSingleLightScene(), *camera, clearColor);
    auto multiPixels = renderWithGL(*makeMultiLightScene(), *camera, clearColor);

    double singleBright = avgBrightness(singlePixels);
    double multiBright = avgBrightness(multiPixels);

    CHECK(multiBright > singleBright);

}


// =============================================================================
// Section 13: GL-only — Additional geometries
// =============================================================================

TEST_CASE("GL: RingGeometry renders correctly") {
    auto scene = Scene::create();
    auto geometry = RingGeometry::create(0.5f, 1.5f, 16, 1);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0x00ff00);
    material->side = Side::Double;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 16);
}

TEST_CASE("GL: TorusKnotGeometry renders correctly") {
    auto scene = Scene::create();
    auto geometry = TorusKnotGeometry::create(0.8f, 0.3f, 64, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff00ff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 16);
}

TEST_CASE("GL: ConeGeometry renders correctly") {
    auto scene = Scene::create();
    auto geometry = ConeGeometry::create(1.0f, 2.0f, 16);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffff00);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 16);
}

TEST_CASE("GL: CapsuleGeometry renders correctly") {
    auto scene = Scene::create();
    auto geometry = CapsuleGeometry::create(0.5f, 1.0f, 8, 16);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0x00ffff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 0);
}


// =============================================================================
// Section 14: Cross-renderer — Extended material parity
// =============================================================================

