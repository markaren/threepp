
#include "threepp/objects/ObjectWithMaterials.hpp"

#include <algorithm>

using namespace threepp;


ObjectWithMaterials::ObjectWithMaterials(std::vector<std::shared_ptr<Material>> materials)
    : materials_(std::move(materials)) {}


std::shared_ptr<Material> ObjectWithMaterials::material() const {

    return materials_.front();
}

void ObjectWithMaterials::setMaterial(const std::shared_ptr<Material>& material) {

    setMaterials({material});
}

const std::vector<std::shared_ptr<Material>>& ObjectWithMaterials::materials() const {

    return materials_;
}

void ObjectWithMaterials::setMaterials(const std::vector<std::shared_ptr<Material>>& materials) {

    this->materials_ = materials;
}

size_t ObjectWithMaterials::numMaterials() const {

    return materials_.size();
}
