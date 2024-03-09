
#ifndef THREEPP_OBJECTWITHMATERIALS_HPP
#define THREEPP_OBJECTWITHMATERIALS_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"


#include <vector>

namespace threepp {

    class ObjectWithMaterials: public virtual Object3D {

    public:
        Material* material() override;

        void setMaterial(const std::shared_ptr<Material>& material);

        std::vector<Material*> materials();

        void setMaterials(const std::vector<std::shared_ptr<Material>>& materials);

        [[nodiscard]] size_t numMaterials() const;

    protected:
        std::vector<std::shared_ptr<Material>> materials_;

        explicit ObjectWithMaterials(std::vector<std::shared_ptr<Material>> materials);
    };

}// namespace threepp

#endif//THREEPP_OBJECTWITHMATERIALS_HPP
