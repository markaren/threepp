// https://github.com/mrdoob/three.js/blob/r129/src/helpers/SpotLightHelper.js

#ifndef THREEPP_SPOTLIGHTHELPER_HPP
#define THREEPP_SPOTLIGHTHELPER_HPP

#include <utility>

#include "threepp/lights/SpotLight.hpp"

namespace threepp {

    class SpotLightHelper : public Object3D {

    public:
        void update() {

            this->light->updateMatrixWorld();

            const auto coneLength = (this->light->distance > 0) ? this->light->distance : 1000;
            const auto coneWidth = coneLength * std::tan(this->light->angle);

            this->cone->scale.set(coneWidth, coneWidth, coneLength);

            Vector3 _vector{};
            _vector.setFromMatrixPosition(*this->light->target->matrixWorld);

            this->cone->lookAt(_vector);

            if (this->color) {

                this->material()->as<MaterialWithColor>()->color.copy(*this->color);

            } else {

                this->cone->material()->as<MaterialWithColor>()->color.copy(this->light->color);
            }
        }

        ~SpotLightHelper() override {
//            this->cone->geometry()->dispose();
//            this->cone->material()->dispose();
        }

        static std::shared_ptr<SpotLightHelper> create(const std::shared_ptr<SpotLight> &light, std::optional<unsigned int> color = std::nullopt) {

            return std::shared_ptr<SpotLightHelper>(new SpotLightHelper(light, color));
        }

    protected:
        std::shared_ptr<SpotLight> light;
        std::optional<Color> color;

        std::shared_ptr<LineSegments> cone;

        SpotLightHelper(std::shared_ptr<SpotLight> light, std::optional<Color> color)
            : light(std::move(light)), color(color) {

            this->light->updateMatrixWorld();

            this->matrix = this->light->matrixWorld;
            this->matrixAutoUpdate = false;

            auto geometry = BufferGeometry::create();

            std::vector<float> positions{
                    0, 0, 0, 0, 0, 1,
                    0, 0, 0, 1, 0, 1,
                    0, 0, 0, -1, 0, 1,
                    0, 0, 0, 0, 1, 1,
                    0, 0, 0, 0, -1, 1};

            for (int i = 0, j = 1, l = 32; i < l; i++, j++) {

                const auto p1 = (static_cast<float>(i) / static_cast<float>(l)) * math::PI * 2;
                const auto p2 = (static_cast<float>(j) / static_cast<float>(l)) * math::PI * 2;

                positions.insert(positions.end(),
                                 {std::cos(p1), std::sin(p1), 1,
                                  std::cos(p2), std::sin(p2), 1});
            }

            geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));

            auto material = LineBasicMaterial::create();
            material->fog = false;
            material->toneMapped = false;

            this->cone = LineSegments::create(geometry, material);
            this->add(this->cone);

            this->update();
        }
    };

}// namespace threepp

#endif//THREEPP_SPOTLIGHTHELPER_HPP
