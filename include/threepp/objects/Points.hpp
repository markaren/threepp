// https://github.com/mrdoob/three.js/blob/r129/src/objects/Points.js

#ifndef THREEPP_POINTS_HPP
#define THREEPP_POINTS_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/PointsMaterial.hpp"
#include "threepp/objects/ObjectWithMaterials.hpp"
#include "threepp/objects/ObjectWithMorphTargetInfluences.hpp"

namespace threepp {

    class Points: public virtual Object3D, public ObjectWithMorphTargetInfluences, public ObjectWithMaterials {

    public:
        Points(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material);

        [[nodiscard]] std::string type() const override;

        std::shared_ptr<BufferGeometry> geometry() const override;

        void setGeometry(const std::shared_ptr<BufferGeometry>& geometry);

        void raycast(const Raycaster& raycaster, std::vector<Intersection>& intersects) override;

        void copy(const Object3D& source, bool recursive = true) override;

        static std::shared_ptr<Points> create(
                std::shared_ptr<BufferGeometry> geometry = BufferGeometry::create(),
                std::shared_ptr<Material> material = PointsMaterial::create());

    protected:
        std::shared_ptr<BufferGeometry> geometry_;

        std::shared_ptr<Object3D> createDefault() override;
    };

}// namespace threepp

#endif//THREEPP_POINTS_HPP
