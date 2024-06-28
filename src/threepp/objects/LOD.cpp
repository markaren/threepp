
#include <memory>

#include "threepp/objects/LOD.hpp"

#include "threepp/cameras/Camera.hpp"

using namespace threepp;

std::string LOD::type() const {

    return "LOD";
}

std::shared_ptr<LOD> LOD::create() {

    return std::make_shared<LOD>();
}

LOD& LOD::addLevel(Object3D& object, float distance) {

    distance = std::abs(distance);

    unsigned l;

    for (l = 0; l < levels.size(); l++) {

        if (distance < levels[l].distance) {

            break;
        }
    }

    levels.insert(levels.begin() + l, {distance, &object});

    this->add(object);

    return *this;
}

LOD& LOD::addLevel(const std::shared_ptr<Object3D>& object, float distance) {

    distance = std::abs(distance);

    unsigned l;

    for (l = 0; l < levels.size(); l++) {

        if (distance < levels[l].distance) {

            break;
        }
    }

    levels.insert(levels.begin() + l, {distance, object.get()});

    this->add(object);

    return *this;
}

size_t LOD::getCurrentLevel() const {

    return _currentLevel;
}

void LOD::update(Camera& camera) {

    static Vector3 _v1;
    static Vector3 _v2;

    if (levels.size() > 1) {

        _v1.setFromMatrixPosition(*camera.matrixWorld);
        _v2.setFromMatrixPosition(*this->matrixWorld);

        float distance = _v1.distanceTo(_v2) / camera.zoom;

        levels[0].object->visible = true;

        size_t i, l;

        for (i = 1, l = levels.size(); i < l; i++) {

            if (distance >= levels[i].distance) {

                levels[i - 1].object->visible = false;
                levels[i].object->visible = true;

            } else {

                break;
            }
        }

        this->_currentLevel = i - 1;

        for (; i < l; i++) {

            levels[i].object->visible = false;
        }
    }
}

void LOD::copy(const Object3D& source, bool /*recursive*/) {
    Object3D::copy(source, false);

    if (const auto l = source.as<LOD>()) {

        for (auto level : l->levels) {

            this->addLevel(level.object->clone(), level.distance);
        }

        this->autoUpdate = l->autoUpdate;
    }
}

std::shared_ptr<Object3D> LOD::createDefault() {

    return create();
}
