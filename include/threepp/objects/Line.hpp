// https://github.com/mrdoob/three.js/blob/r129/src/objects/Line.js

#ifndef THREEPP_LINE_HPP
#define THREEPP_LINE_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/objects/ObjectWithMaterials.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace threepp {

    class Line: public virtual Object3D, public ObjectWithMaterials {

    public:
        Line(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material);

        Line(Line&&) = delete;

        [[nodiscard]] std::string type() const override;

        BufferGeometry* geometry() override;

        void setGeometry(const std::shared_ptr<BufferGeometry>& geometry);

        virtual void computeLineDistances();

        void raycast(const Raycaster& raycaster, std::vector<Intersection>& intersects) override;

        std::shared_ptr<Object3D> clone(bool recursive = true) override;

        static std::shared_ptr<Line> create(const std::shared_ptr<BufferGeometry>& geometry = nullptr, const std::shared_ptr<Material>& material = nullptr);

    protected:
        std::shared_ptr<BufferGeometry> geometry_;
    };

}// namespace threepp

#endif//THREEPP_LINE_HPP
