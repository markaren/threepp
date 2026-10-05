
#include "threepp/materials/RawShaderMaterial.hpp"

using namespace threepp;


RawShaderMaterial::RawShaderMaterial() = default;


const std::string& RawShaderMaterial::type() const {

    static const std::string typeName = "RawShaderMaterial";
    return typeName;
}

std::shared_ptr<RawShaderMaterial> RawShaderMaterial::create() {

    return std::shared_ptr<RawShaderMaterial>(new RawShaderMaterial());
}

std::shared_ptr<Material> RawShaderMaterial::createDefault() const {

    return std::shared_ptr<RawShaderMaterial>(new RawShaderMaterial());
}
