// https://github.com/mrdoob/three.js/blob/r129/src/objects/Line.js

#ifndef THREEPP_LINE_HPP
#define THREEPP_LINE_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/objects/ObjectWithMaterials.hpp"

#include <memory>
#include <vector>

namespace threepp {

    class Line: public virtual Object3D, public ObjectWithMaterials {

    public:
        Line(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material);

        [[nodiscard]] std::string type() const override;

        std::shared_ptr<BufferGeometry> geometry() const override;

        void setGeometry(const std::shared_ptr<BufferGeometry>& geometry);

        virtual void computeLineDistances();

        void copy(const Object3D& source, bool recursive = true) override;

        void raycast(const Raycaster& raycaster, std::vector<Intersection>& intersects) override;

        static std::shared_ptr<Line> create(const std::shared_ptr<BufferGeometry>& geometry = nullptr, const std::shared_ptr<Material>& material = nullptr);

    protected:
        std::shared_ptr<BufferGeometry> geometry_;

        std::shared_ptr<Object3D> createDefault() override;
    };

}// namespace threepp

#endif//THREEPP_LINE_HPP
