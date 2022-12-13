// https://github.com/mrdoob/three.js/blob/r129/src/helpers/DirectionalLightHelper.js

#ifndef THREEPP_DIRECTIONALLIGHTHELPER_HPP
#define THREEPP_DIRECTIONALLIGHTHELPER_HPP

#include <utility>

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/objects/Line.hpp"

namespace threepp {

    class DirectionalLightHelper : public Object3D {

    public:
        void update() {

            Vector3 _v1{};
            Vector3 _v2{};
            Vector3 _v3{};

            _v1.setFromMatrixPosition(*this->light->matrixWorld);
            _v2.setFromMatrixPosition(*this->light->target->matrixWorld);
            _v3.subVectors(_v2, _v1);

            this->lightPlane->lookAt(_v2);

            if (this->color) {

                this->lightPlane->material()->as<MaterialWithColor>()->color.copy(*this->color);
                this->targetLine->material()->as<MaterialWithColor>()->color.copy(*this->color);

            } else {

                this->lightPlane->material()->as<MaterialWithColor>()->color.copy(this->light->color);
                this->targetLine->material()->as<MaterialWithColor>()->color.copy(this->light->color);
            }

            this->targetLine->lookAt(_v2);
            this->targetLine->scale.z = _v3.length();

        }

        static std::shared_ptr<DirectionalLightHelper> create(const std::shared_ptr<DirectionalLight> &light, float size = 1, std::optional<unsigned int> color = std::nullopt) {

            return std::shared_ptr<DirectionalLightHelper>(new DirectionalLightHelper(light, size, color));
        }

    protected:
        float size;
        std::optional<Color> color;

        std::shared_ptr<DirectionalLight> light;
        std::shared_ptr<Line> lightPlane;
        std::shared_ptr<Line> targetLine;

        DirectionalLightHelper(std::shared_ptr<DirectionalLight> light, float size, std::optional<unsigned int> color)
            : light(std::move(light)), size(size), color(color) {

            this->light->updateMatrixWorld();

            this->matrix = this->light->matrixWorld;
            this->matrixAutoUpdate = false;

            auto geometry = BufferGeometry::create();
            geometry->setAttribute("position", FloatBufferAttribute::create(
                                                       {-size, size, 0,
                                                        size, size, 0,
                                                        size, -size, 0,
                                                        -size, -size, 0,
                                                        -size, size, 0},
                                                       3));

            auto material = LineBasicMaterial::create();
            material->fog = false;
            material->toneMapped = false;

            this->lightPlane = Line::create(geometry, material);
            this->add(this->lightPlane);

            geometry = BufferGeometry::create();
            geometry->setAttribute("position", FloatBufferAttribute::create({0, 0, 0, 0, 0, 1}, 3));

            this->targetLine = Line::create(geometry, material);
            this->add(this->targetLine);

            this->update();
        }
    };

}// namespace threepp

#endif//THREEPP_DIRECTIONALLIGHTHELPER_HPP
