// https://github.com/mrdoob/three.js/blob/r129/src/objects/Points.js

#ifndef THREEPP_POINTS_HPP
#define THREEPP_POINTS_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/PointsMaterial.hpp"

namespace threepp {

    class Points : public Object3D {

    public:
        BufferGeometry* geometry() override {

            return geometry_.get();
        }

        Material* material() override {

            return material_.get();
        }

        std::vector<Material*> materials() override {
            return {material_.get()};
        }

        static std::shared_ptr<Points> create(
                std::shared_ptr<BufferGeometry> geometry = BufferGeometry::create(),
                std::shared_ptr<Material> material = PointsMaterial::create()) {

            return std::shared_ptr<Points>(new Points(std::move(geometry), std::move(material)));
        }

    protected:
        Points(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material)
            : geometry_(std::move(geometry)), material_(std::move(material)) {
        }

    private:
        std::shared_ptr<BufferGeometry> geometry_;
        std::shared_ptr<Material> material_;
    };

}// namespace threepp

#endif//THREEPP_POINTS_HPP
