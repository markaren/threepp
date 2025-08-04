
#include "threepp/helpers/HemisphereLightHelper.hpp"

#include "threepp/geometries/OctahedronGeometry.hpp"
#include "threepp/lights/HemisphereLight.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"

using namespace threepp;

namespace {

    Vector3 _vector;
    Color _color1;
    Color _color2;

}// namespace

struct HemisphereLightHelper::Impl {

    bool disposed = false;
    HemisphereLight& light;
    HemisphereLightHelper& scope;
    std::shared_ptr<MeshBasicMaterial> material;

    Impl(HemisphereLightHelper& scope, HemisphereLight& light, float size)
        : light(light), scope(scope) {

        this->light.updateMatrixWorld();
        this->scope.matrix = light.matrixWorld;
        this->scope.matrixAutoUpdate = false;

        auto geometry = OctahedronGeometry::create(size);
        geometry->rotateY(math::PI * 0.5f);

        this->material = MeshBasicMaterial::create();
        this->material->wireframe = true;
        this->material->fog = false;
        this->material->toneMapped = false;

        if (!scope.color) this->material->vertexColors = true;

        auto position = geometry->getAttribute<float>("position");
        auto colors = std::vector<float>(position->count() * 3);

        geometry->setAttribute("color", FloatBufferAttribute::create(colors, 3));

        scope.add(Mesh::create(geometry, this->material));

        this->update();
    }

    void update() {

        auto& mesh = scope.children[0];

        if (this->scope.color) {

            this->material->color.copy(*this->scope.color);

        } else {

            const auto colors = mesh->geometry()->getAttribute<float>("color");

            _color1.copy(this->light.color);
            _color2.copy(this->light.groundColor);

            for (unsigned i = 0, l = colors->count(); i < l; i++) {

                const auto& color = (i < (l / 2)) ? _color1 : _color2;

                colors->setXYZ(i, color.r, color.g, color.b);
            }

            colors->needsUpdate();
        }

        mesh->lookAt(_vector.setFromMatrixPosition(*this->light.matrixWorld).negate());
    }

    void dispose() {
        if (!disposed) {
            disposed = true;
            scope.children[0]->geometry()->dispose();
            scope.children[0]->material()->dispose();
        }
    }

    ~Impl() {
        dispose();
    }
};

HemisphereLightHelper::HemisphereLightHelper(HemisphereLight& light, float size, const std::optional<Color>& color)
    : color(color), pimpl_(std::make_unique<Impl>(*this, light, size)) {}

void HemisphereLightHelper::update() {

    pimpl_->update();
}

void HemisphereLightHelper::dispose() {

    pimpl_->dispose();
}

std::shared_ptr<HemisphereLightHelper> HemisphereLightHelper::create(HemisphereLight& light, float size, const std::optional<Color>& color) {

    return std::shared_ptr<HemisphereLightHelper>(new HemisphereLightHelper(light, size, color));
}

HemisphereLightHelper::~HemisphereLightHelper() = default;
