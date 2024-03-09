
#include "threepp/objects/ObjectWithMaterials.hpp"

#include <algorithm>

using namespace threepp;


ObjectWithMaterials::ObjectWithMaterials(std::vector<std::shared_ptr<Material>> materials)
    : materials_(std::move(materials)) {}


Material* ObjectWithMaterials::material() {

    return materials_.front().get();
}

void ObjectWithMaterials::setMaterial(const std::shared_ptr<Material>& material) {

    setMaterials({material});
}

std::vector<Material*> ObjectWithMaterials::materials() {
    std::vector<Material*> res(materials_.size());
    std::transform(materials_.begin(), materials_.end(), res.begin(), [](auto& m) { return m.get(); });

    return res;
}

void ObjectWithMaterials::setMaterials(const std::vector<std::shared_ptr<Material>>& materials) {

    this->materials_ = materials;
}

size_t ObjectWithMaterials::numMaterials() const {

    return materials_.size();
}
