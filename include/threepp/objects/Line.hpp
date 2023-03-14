// https://github.com/mrdoob/three.js/blob/r129/src/objects/Line.js

#ifndef THREEPP_LINE_HPP
#define THREEPP_LINE_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace threepp {

    class Line: public Object3D {

    public:
        [[nodiscard]] std::string type() const override {

            return "Line";
        }

        BufferGeometry* geometry() override {

            return geometry_.get();
        }

        Material* material() override {

            return material_.get();
        }

        std::vector<Material*> materials() override {

            return {material_.get()};
        }

        virtual void computeLineDistances();

        void raycast(Raycaster& raycaster, std::vector<Intersection>& intersects) override;

        std::shared_ptr<Object3D> clone(bool recursive = true) override {
            auto clone = create(geometry_, material_);
            clone->copy(*this, recursive);

            return clone;
        }

        static std::shared_ptr<Line> create(const std::shared_ptr<BufferGeometry>& geometry = nullptr, const std::shared_ptr<Material>& material = nullptr) {

            return std::shared_ptr<Line>(new Line(geometry, (material)));
        }

    protected:
        std::shared_ptr<BufferGeometry> geometry_;
        std::shared_ptr<Material> material_;

        Line(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material);
    };

}// namespace threepp

#endif//THREEPP_LINE_HPP
