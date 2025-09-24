
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/helpers/SkeletonHelper.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/threepp.hpp"

#include <cmath>

using namespace threepp;

namespace {

    struct Sizing {
        float segmentHeight;
        int segmentCount;
        float height;
        float halfHeight;
    };

    auto createGeometry(Sizing sizing) {

        auto geometry = CylinderGeometry::create(
                5,                      // radiusTop
                5,                      // radiusBottom
                sizing.height,          // height
                8,                      // radiusSegments
                sizing.segmentCount * 3,// heightSegments
                true                    // openEnded
        );

        auto position = geometry->getAttribute<float>("position");

        Vector3 vertex;

        std::vector<float> skinIndices;
        std::vector<float> skinWeights;

        for (unsigned i = 0; i < position->count(); i++) {

            position->setFromBufferAttribute(vertex, i);

            auto y = (vertex.y + sizing.halfHeight);

            auto skinIndex = std::floor(y / sizing.segmentHeight);
            auto skinWeight = float(fmod(y, sizing.segmentHeight) / sizing.segmentHeight);

            skinIndices.insert(skinIndices.end(), {skinIndex, skinIndex + 1, 0, 0});
            skinWeights.insert(skinWeights.end(), {1.f - skinWeight, skinWeight, 0, 0});
        }

        geometry->setAttribute("skinIndex", FloatBufferAttribute::create(skinIndices, 4));
        geometry->setAttribute("skinWeight", FloatBufferAttribute::create(skinWeights, 4));

        return geometry;
    }

    auto createBones(Sizing sizing) {

        std::vector<std::shared_ptr<Bone>> bones;

        auto prevBone = Bone::create();
        bones.emplace_back(prevBone);

        for (unsigned i = 0; i < sizing.segmentCount; i++) {

            auto bone = Bone::create();
            bone->position.y = static_cast<float>(sizing.segmentHeight);
            bones.emplace_back(bone);
            prevBone->add(bone);
            prevBone = bone;
        }

        return bones;
    }

    auto createMesh(const std::shared_ptr<BufferGeometry>& geometry, const std::vector<std::shared_ptr<Bone>>& bones) {

        auto material = MeshPhongMaterial::create({{"color", 0x156289},
                                                   {"emissive", 0x072534},
                                                   {"side", Side::Double},
                                                   {"flatShading", true}});

        auto mesh = SkinnedMesh::create(geometry, material);
        mesh->castShadow = true;
        const auto skeleton = Skeleton::create(bones);

        mesh->add(bones[0]);

        mesh->bind(skeleton);

        const auto skeletonHelper = SkeletonHelper::create(*mesh);
        skeletonHelper->material()->as<LineBasicMaterial>()->linewidth = 2;
        mesh->add(skeletonHelper);

        return mesh;
    }

    auto initBones() {
        constexpr float segmentHeight = 8;
        constexpr int segmentCount = 4;
        constexpr float height = segmentHeight * float(segmentCount);
        constexpr float halfHeight = height * 0.5f;

        const Sizing sizing{segmentHeight, segmentCount, height, halfHeight};

        const auto geometry = createGeometry(sizing);
        geometry->applyMatrix4(Matrix4().makeTranslation(0, halfHeight, 0));
        const auto bones = createBones(sizing);

        auto mesh = createMesh(geometry, bones);
        mesh->scale.multiplyScalar(1);

        return mesh;
    }

    auto initPlane() {
        constexpr int gridSize = 100;
        auto grid = GridHelper::create(gridSize, 10, Color::yellow);

        const auto geometry = PlaneGeometry::create(gridSize, gridSize);
        const auto material = ShadowMaterial::create();
        material->color = 0x000000;
        material->opacity = 0.2f;

        const auto plane = Mesh::create(geometry, material);
        plane->rotation.x = -math::PI / 2;
        plane->receiveShadow = true;
        grid->add(plane);

        return grid;
    }

}// namespace

int main() {

    Canvas canvas("Bones");
    GLRenderer renderer(canvas.size());
    renderer.shadowMap().enabled = true;

    Scene scene;
    scene.background = Color(0x444444);

    PerspectiveCamera camera(75, canvas.size().aspect(), 0.1, 200);
    camera.position.set(0, 50, 50);

    auto light1 = PointLight::create(0xffffff);
    light1->position.set(0, 200, 0);
    light1->castShadow = true;
    auto light2 = PointLight::create(0xffffff);
    light2->position.set(100, 200, 100);
    light2->castShadow = true;
    auto light3 = PointLight::create(0xffffff);
    light3->position.set(-100, -200, -100);
    light3->castShadow = true;

    scene.add(light1);
    scene.add(light2);
    scene.add(light3);

    auto plane = initPlane();
    scene.add(plane);

    auto bones = initBones();
    scene.add(bones);

    OrbitControls controls{camera, canvas};
    controls.enableZoom = false;

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    bool animate{false};
    ImguiFunctionalContext ui(canvas, [&] {
        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({0, 0}, 0);
        ImGui::Begin("Options");
        ImGui::Checkbox("animate", &animate);
        ImGui::End();
    });

    IOCapture capture{};
    capture.preventMouseEvent = [] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&capture);

    Clock clock;
    canvas.animate([&] {
        const auto time = clock.getElapsedTime();

        renderer.render(scene, camera);
        ui.render();

        if (animate) {
            for (unsigned i = 0; i < bones->skeleton->bones.size(); i++) {

                bones->skeleton->bones[i]->rotation.z = std::sin(time) * 2 / float(bones->skeleton->bones.size());
            }
        }
    });
}