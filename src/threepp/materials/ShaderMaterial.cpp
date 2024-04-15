
#include "threepp/materials/ShaderMaterial.hpp"

#include "threepp/renderers/shaders/ShaderChunk.hpp"

using namespace threepp;

ShaderMaterial::ShaderMaterial()
    : MaterialWithLights(false),
      MaterialWithClipping(false),
      MaterialWithWireframe(false, 1),
      MaterialWithLineWidth(1),
      vertexShader(shaders::ShaderChunk::instance().default_vertex()),
      fragmentShader(shaders::ShaderChunk::instance().default_fragment()) {

    this->fog = false;
    this->lights = false;
    this->clipping = false;

    defaultAttributeValues["color"] = Color(1, 1, 1);
    defaultAttributeValues["uv"] = Vector2(0, 0);
    defaultAttributeValues["uv2"] = Vector2(0, 0);
}


std::string ShaderMaterial::type() const {

    return "ShaderMaterial";
}

std::shared_ptr<ShaderMaterial> ShaderMaterial::create() {

    return std::shared_ptr<ShaderMaterial>(new ShaderMaterial());
}

std::shared_ptr<Material> ShaderMaterial::createDefault() const {

    return std::shared_ptr<ShaderMaterial>(new ShaderMaterial());
}
