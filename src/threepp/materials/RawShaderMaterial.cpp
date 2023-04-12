
#include "threepp/materials/RawShaderMaterial.hpp"

using namespace threepp;


RawShaderMaterial::RawShaderMaterial() = default;


std::string RawShaderMaterial::type() const {

    return "RawShaderMaterial";
}

std::shared_ptr<RawShaderMaterial> RawShaderMaterial::create() {

    return std::shared_ptr<RawShaderMaterial>(new RawShaderMaterial());
}
