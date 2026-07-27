
#include "threepp/materials/ShaderMaterial.hpp"

#include "threepp/renderers/shaders/ShaderChunk.hpp"

using namespace threepp;

ShaderMaterial::ShaderMaterial()
    : MaterialWithClipping(false),
      MaterialWithLights(false),
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

bool ShaderMaterial::setValue(const std::string& key, const MaterialValue& value) {

    if (key == "vertexShader") {
        vertexShader = std::get<std::string>(value);
    } else if (key == "fragmentShader") {
        fragmentShader = std::get<std::string>(value);
    } else if (key == "wireframe") {
        wireframe = std::get<bool>(value);
    } else if (key == "wireframeLinewidth") {
        wireframeLinewidth = extractFloat(value);
    } else if (key == "linewidth") {
        linewidth = extractFloat(value);
    } else if (key == "lights") {
        lights = std::get<bool>(value);
    } else if (key == "clipping") {
        clipping = std::get<bool>(value);
    } else if (key == "envMap") {
        envMap = std::get<std::shared_ptr<Texture>>(value);
    } else {
        return false;
    }

    return true;
}
