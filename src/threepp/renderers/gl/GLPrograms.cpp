
#include "GLPrograms.hpp"

#include "GLProgram.hpp"

#include "threepp/materials/RawShaderMaterial.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/utils/StringUtils.hpp"

#include "threepp/renderers/shaders/ShaderLib.hpp"

using namespace threepp;
using namespace threepp::gl;

namespace {

    std::unordered_map<std::string, std::string> shaderIDs{
            {"MeshDepthMaterial", "depth"},
            {"MeshDistanceMaterial", "distanceRGBA"},
            {"MeshNormalMaterial", "normal"},
            {"MeshBasicMaterial", "basic"},
            {"MeshLambertMaterial", "lambert"},
            {"MeshPhongMaterial", "phong"},
            {"MeshToonMaterial", "toon"},
            {"MeshStandardMaterial", "physical"},
            {"MeshPhysicalMaterial", "physical"},
            {"MeshMatcapMaterial", "matcap"},
            {"LineBasicMaterial", "basic"},
            {"LineDashedMaterial", "dashed"},
            {"PointsMaterial", "points"},
            {"ShadowMaterial", "shadow"},
            {"SpriteMaterial", "sprite"}};

}// namespace


GLPrograms::GLPrograms(GLBindingStates &bindingStates, GLClipping &clipping)
    : logarithmicDepthBuffer(GLCapabilities::instance().logarithmicDepthBuffer),
      floatVertexTextures(GLCapabilities::instance().floatVertexTextures),
      maxVertexUniforms(GLCapabilities::instance().maxVertexUniforms),
      vertexTextures(GLCapabilities::instance().vertexTextures),
      bindingStates(bindingStates),
      clipping(clipping) {}


int GLPrograms::getTextureEncodingFromMap(std::optional<Texture> &map) const {

    return map ? map->encoding : LinearEncoding;
}

ProgramParameters GLPrograms::getParameters(Material *material, const GLLights::LightState &lights, int numShadows, Scene *scene, Object3D *object) {

    ProgramParameters p;

    p.shaderID = shaderIDs[material->type()];
    p.shaderName = material->type();

    p.isRawShaderMaterial = instanceof <RawShaderMaterial>(material);

    auto instancedMesh = dynamic_cast<InstancedMesh *>(object);
    p.instancing = instancedMesh != nullptr;
    p.instancingColor = instancedMesh != nullptr && instancedMesh->instanceColor != nullptr;

    p.supportsVertexTextures = vertexTextures;

    p.numDirLights = (int) lights.directional.size();
    p.numPointLights = (int) lights.point.size();
    p.numSpotLights = (int) lights.spot.size();
    p.numRectAreaLights = 0;
    p.numHemiLights = 0;

    p.numDirLightShadows = (int) lights.directionalShadowMap.size();
    p.numPointLightShadows = (int) lights.pointShadowMap.size();
    p.numSpotLightShadows = (int) lights.spotShadowMap.size();

    p.dithering = material->dithering;

    p.premultipliedAlpha = material->premultipliedAlpha;

    p.alphaTest = material->alphaTest;
    p.doubleSided = material->side == DoubleSide;
    p.flipSided = material->side == BackSide;

    p.index0AttributeName = std::nullopt;

    return p;
}

std::string GLPrograms::getProgramCacheKey(const GLRenderer &renderer, const ProgramParameters &parameters) {

    std::vector<std::string> array;

    if (parameters.shaderID) {

        array.emplace_back(*parameters.shaderID);

    } else {

        array.emplace_back(parameters.fragmentShader);
        array.emplace_back(parameters.vertexShader);
    }

    if (parameters.defines) {

        for (const auto &[name, value] : *parameters.defines) {

            array.emplace_back(name);
            array.emplace_back(value);
        }
    }

    if (!parameters.isRawShaderMaterial) {

        auto hash = utils::split(parameters.hash(), '\n');
        for (const auto &value : hash) {

            array.emplace_back(value);
        }

        array.emplace_back(std::to_string(renderer.outputEncoding));
        array.emplace_back(std::to_string(renderer.gammaFactor));
    }

    //    array.emplace_back(parameters.customProgramCacheKey);

    return utils::join(array, '\n');
}

std::unordered_map<std::string, Uniform> GLPrograms::getUniforms(Material *material) {

    std::unordered_map<std::string, Uniform> uniforms;

    if (shaderIDs.count(material->type())) {

        auto shaderID = shaderIDs.at(material->type());

        auto shader = shaders::ShaderLib::instance().get(shaderID);
        uniforms = shader.uniforms;

    } else {

        uniforms = material->uniforms;
    }

    return uniforms;
}

std::shared_ptr<GLProgram> GLPrograms::acquireProgram(const ProgramParameters &parameters, const std::string &cacheKey) {

    std::shared_ptr<GLProgram> program = nullptr;

    // Check if code has been already compiled
    for (auto &preexistingProgram : programs) {

        if (preexistingProgram->cacheKey == cacheKey) {

            program = preexistingProgram;
            ++program->usedTimes;

            break;
        }
    }

    if (!program) {

        program = GLProgram::create(cacheKey, parameters, bindingStates );
        programs.emplace_back(program);
    }

    return program;
}

void GLPrograms::releaseProgram(std::shared_ptr<GLProgram> &program) {

    if (--program->usedTimes == 0) {

        auto it = find(programs.begin(), programs.end(), program);
        auto i = it - programs.begin();

        // Remove from unordered set
        programs[i] = programs[programs.size() - 1];
        programs.pop_back();

        // Free WebGL resources
        program->destroy();
    }
}
