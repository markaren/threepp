
#include "../../external/imgui/imgui.h"


#include <threepp/controls/OrbitControls.hpp>
#include <threepp/controls/TransformControls.hpp>
#include <threepp/extras/curves/CatmullRomCurve3.hpp>
#include <threepp/extras/imgui/ImguiContext.hpp>
#include <threepp/threepp.hpp>

#include <map>

using namespace threepp;

int main() {

    // Renderer & canvas
    Canvas canvas("Spline Editor", {{"aa", 4}});
    GLRenderer renderer(canvas.size());
    renderer.shadowMap().enabled = true;

    // Scene
    auto scene = Scene::create();
    scene->background = Color().setHex(0xf0f0f0);

    // Camera
    auto camera = PerspectiveCamera::create(70, canvas.aspect(), 1, 10000);
    camera->position.set(0, 250, 1000);
    scene->add(camera);

    // Lights
    scene->add(AmbientLight::create(0xf0f0f0, 0.5f));
    auto light = SpotLight::create(0xffffff, 1.5f);
    light->position.set(0, 1500, 200);
    light->angle = math::PI * 0.2f;
    light->decay = 0;
    light->castShadow = true;
    light->shadow->camera->nearPlane = 200;
    light->shadow->camera->farPlane = 2000;
    light->shadow->bias = -0.00022f;
    light->shadow->mapSize.x = 1024;
    light->shadow->mapSize.y = 1024;
    scene->add(light);

    // Ground
    auto planeGeo = PlaneGeometry::create(2000, 2000);
    planeGeo->rotateX(-math::PI / 2);
    auto planeMat = ShadowMaterial::create();
    planeMat->opacity = 0.2f;
    auto plane = Mesh::create(planeGeo, planeMat);
    plane->position.y = -200;
    plane->receiveShadow = true;
    scene->add(plane);

    // Grid helper
    auto grid = GridHelper::create(2000, 100);
    grid->position.y = -199;
    grid->material()->opacity = 0.25;
    grid->material()->transparent = true;
    scene->add(grid);

    // Spline points
    std::vector<Vector3> positions = {
            {289.768f, 452.515f, 56.1f},
            {-53.563f, 171.497f, -14.495f},
            {-91.401f, 176.431f, -6.958f},
            {-383.785f, 491.137f, 47.869f}};

    constexpr int ARC_SEGMENTS = 200;

    // Spline meshes
    std::map<std::string, std::shared_ptr<Line>> splines;

    auto createSplineLine = [&](const std::string& type, const Color& color) {
        auto curve = std::make_unique<CatmullRomCurve3>(positions);
        if (type == "uniform") curve->curveType = CatmullRomCurve3::CurveType::catmullrom;
        if (type == "centripetal") curve->curveType = CatmullRomCurve3::CurveType::centripetal;
        if (type == "chordal") curve->curveType = CatmullRomCurve3::CurveType::chordal;

        auto geom = BufferGeometry::create();
        geom->setAttribute("position", FloatBufferAttribute::create(std::vector<float>(ARC_SEGMENTS * 3), 3));
        auto line = Line::create(geom, LineBasicMaterial::create({{"color", color}}));
        line->castShadow = true;
        splines[type] = line;
        scene->add(line);
        return curve;
    };

    auto uniformCurve = createSplineLine("uniform", Color::red);
    auto centripetalCurve = createSplineLine("centripetal", Color::green);
    auto chordalCurve = createSplineLine("chordal", Color::blue);

    // Helper objects (control points)
    auto boxGeo = BoxGeometry::create(20, 20, 20);
    std::vector<Object3D*> splineHelpers;


    auto createSplineHelper = [&](const Vector3& pos) {
        auto mat = MeshLambertMaterial::create({{"color", Color().randomize()}});
        auto obj = Mesh::create(boxGeo, mat);
        obj->position.copy(pos);
        obj->castShadow = true;
        obj->receiveShadow = true;
        scene->add(obj);
        splineHelpers.emplace_back(obj.get());
        return obj;
    };

    for (auto& pos : positions) {
        createSplineHelper(pos);
    }

    // Orbit controls
    OrbitControls controls(*camera, canvas);
    controls.dampingFactor = 0.2f;

    auto updateSplines([&] {
        // Update spline meshes whenever a control point moves
        auto updateSplineMesh = [&](const CatmullRomCurve3& curve, const std::shared_ptr<Line>& line) {
            auto posAttr = line->geometry()->getAttribute<float>("position");
            for (int i = 0; i < ARC_SEGMENTS; i++) {
                float t = static_cast<float>(i) / static_cast<float>(ARC_SEGMENTS - 1);
                Vector3 p;
                curve.getPoint(t, p);
                posAttr->setXYZ(i, p.x, p.y, p.z);
            }
            posAttr->needsUpdate();
        };

        // Copy helper positions to curve
        for (int i = 0; i < splineHelpers.size(); i++) {
            positions[i] = splineHelpers[i]->position;
        }
        uniformCurve->points = positions;
        centripetalCurve->points = positions;
        chordalCurve->points = positions;

        updateSplineMesh(*uniformCurve, splines["uniform"]);
        updateSplineMesh(*centripetalCurve, splines["centripetal"]);
        updateSplineMesh(*chordalCurve, splines["chordal"]);
    });

    LambdaEventListener eventListener([&](const Event&) {
        updateSplines();
    });


    LambdaEventListener changeListener([&](const Event& event) {
        controls.enabled = !std::any_cast<bool>(event.target);
    });

    // Transform controls
    TransformControls transformControl(*camera, canvas);
    transformControl.addEventListener("change", eventListener);
    transformControl.addEventListener("dragging-changed", changeListener);
    scene->add(transformControl);


    Vector2 mouse{-Infinity<float>, -Infinity<float>};
    MouseDownListener mouseListener([&](int button, const Vector2& pos) {
        if (button == 0) {

            Raycaster raycaster;
            const auto size = canvas.size();
            mouse.x = (pos.x / static_cast<float>(size.width())) * 2 - 1;
            mouse.y = -(pos.y / static_cast<float>(size.height())) * 2 + 1;
            raycaster.setFromCamera(mouse, *camera);
            const auto intersects = raycaster.intersectObjects(splineHelpers, false);
            if (!intersects.empty()) {
                transformControl.attach(*intersects[0].object);
            }
        } else if (button == 1) {
            transformControl.detach();
        }
    });

    canvas.addMouseListener(mouseListener);


    ImguiFunctionalContext ui(canvas, [&] {
        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({0, 0}, 0);
        ImGui::Begin("Spline Editor");
        ImGui::Checkbox("uniform", &splines["uniform"]->visible);
        if (ImGui::SliderFloat("tension", &uniformCurve->tension, 0, 1)) {
            updateSplines();
        }
        ImGui::Checkbox("centripetal", &splines["centripetal"]->visible);
        ImGui::Checkbox("chordal", &splines["chordal"]->visible);
        if (ImGui::Button("addPoint")) {
            positions.emplace_back(math::randFloat() * 1000 - 500, math::randFloat() * 600, math::randFloat() * 800 - 400);
            createSplineHelper(positions.back());
            updateSplines();
        }
        if (ImGui::Button("removePoint")) {
            if (positions.size() > 2) {
                positions.pop_back();
                scene->remove(*splineHelpers.back());
                splineHelpers.pop_back();
                updateSplines();
            }
        }
        ImGui::End();
    });

    IOCapture capture{};
    capture.preventMouseEvent = [&] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&capture);

    canvas.onWindowResize([&](WindowSize size) {
        renderer.setSize(size);
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
    });

    updateSplines();
    canvas.animate([&] {
        controls.update();
        renderer.render(*scene, *camera);

        ui.render();
    });
}