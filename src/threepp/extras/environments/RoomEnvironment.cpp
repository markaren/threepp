
#include "threepp/extras/environments/RoomEnvironment.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/materials/MeshLambertMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"

#include <unordered_set>

using namespace threepp;

namespace {

    // An emissive-only material (three.js #31348): black diffuse, so the
    // panel adds exactly `intensity` of radiance and reflects nothing.
    std::shared_ptr<MeshLambertMaterial> createAreaLightMaterial(float intensity) {

        auto material = MeshLambertMaterial::create();
        material->color = Color(0x000000);
        material->emissive = Color(0xffffff);
        material->emissiveIntensity = intensity;
        return material;
    }

}// namespace

RoomEnvironment::RoomEnvironment() {

    name = "RoomEnvironment";
    position.y = -3.5f;

    auto geometry = BoxGeometry::create();
    geometry->deleteAttribute("uv");

    auto roomMaterial = MeshStandardMaterial::create();
    roomMaterial->side = Side::Back;
    auto boxMaterial = MeshStandardMaterial::create();

    // Physical units (three.js r155+, threepp's default useLegacyLights=false).
    auto mainLight = PointLight::create(0xffffff, 900.f, 28.f, 2.f);
    mainLight->position.set(0.418f, 16.199f, 0.300f);
    add(mainLight);

    auto room = Mesh::create(geometry, roomMaterial);
    room->position.set(-0.757f, 13.219f, 0.717f);
    room->scale.set(31.713f, 28.305f, 28.591f);
    add(room);

    auto boxes = InstancedMesh::create(geometry, boxMaterial, 6);
    Object3D transform;

    struct Box {
        float px, py, pz, ry, sx, sy, sz;
    };
    const Box layout[6] = {
            {-10.906f, 2.009f, 1.846f, -0.195f, 2.328f, 7.905f, 4.651f},
            {-5.607f, -0.754f, -0.758f, 0.994f, 1.970f, 1.534f, 3.955f},
            {6.167f, 0.857f, 7.803f, 0.561f, 3.927f, 6.285f, 3.687f},
            {-2.017f, 0.018f, 6.124f, 0.333f, 2.002f, 4.566f, 2.064f},
            {2.291f, -0.756f, -2.621f, -0.286f, 1.546f, 1.552f, 1.496f},
            {-2.193f, -0.369f, -5.547f, 0.516f, 3.875f, 3.487f, 2.986f},
    };
    for (size_t i = 0; i < 6; ++i) {
        const auto& b = layout[i];
        transform.position.set(b.px, b.py, b.pz);
        transform.rotation.set(0, b.ry, 0);
        transform.scale.set(b.sx, b.sy, b.sz);
        transform.updateMatrix();
        boxes->setMatrixAt(i, *transform.matrix);
    }
    add(boxes);

    struct Panel {
        float intensity, px, py, pz, sx, sy, sz;
    };
    const Panel panels[6] = {
            {50.f, -16.116f, 14.37f, 8.208f, 0.1f, 2.428f, 2.739f},  // -x right
            {50.f, -16.109f, 18.021f, -8.207f, 0.1f, 2.425f, 2.751f},// -x left
            {17.f, 14.904f, 12.198f, -1.832f, 0.15f, 4.265f, 6.331f},// +x
            {43.f, -0.462f, 8.89f, 14.520f, 4.38f, 5.441f, 0.088f},  // +z
            {20.f, 3.235f, 11.486f, -12.541f, 2.5f, 2.0f, 0.1f},     // -z
            {100.f, 0.0f, 20.0f, 0.0f, 1.0f, 0.1f, 1.0f},            // +y
    };
    for (const auto& p : panels) {
        auto light = Mesh::create(geometry, createAreaLightMaterial(p.intensity));
        light->position.set(p.px, p.py, p.pz);
        light->scale.set(p.sx, p.sy, p.sz);
        add(light);
    }
}

const std::string& RoomEnvironment::type() const {

    static const std::string typeName = "Scene";
    return typeName;
}

void RoomEnvironment::dispose() {

    std::unordered_set<BufferGeometry*> geometries;
    std::unordered_set<Material*> materials;

    traverse([&](Object3D& object) {
        if (auto* mesh = object.as<Mesh>()) {
            if (auto g = mesh->geometry()) geometries.insert(g.get());
            for (const auto& m : mesh->materials()) materials.insert(m.get());
        }
    });

    for (auto* g : geometries) g->dispose();
    for (auto* m : materials) m->dispose();
}

std::shared_ptr<RoomEnvironment> RoomEnvironment::create() {

    return std::make_shared<RoomEnvironment>();
}
