
#include "GLPrograms.hpp"

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


ProgramParameters GLPrograms::getParameters(
        const GLRenderer &renderer,
        Material *material,
        const GLLights::LightState &lights,
        int numShadows,
        const std::shared_ptr<Scene> &scene,
        const std::shared_ptr<Object3D> &object) {

    return ProgramParameters(renderer, lights, numShadows, object, scene, material, shaderIDs);
}

std::string GLPrograms::getProgramCacheKey(const GLRenderer &renderer, const ProgramParameters &parameters) {

    std::vector<std::string> array;

    if (parameters.shaderID) {

        array.emplace_back(*parameters.shaderID);

    } else {

        array.emplace_back(parameters.fragmentShader);
        array.emplace_back(parameters.vertexShader);
    }

    if (!parameters.defines.empty()) {

        for (const auto &[name, value] : parameters.defines) {

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

    return utils::join(array);
}

std::shared_ptr<UniformMap> GLPrograms::getUniforms(const std::shared_ptr<Material> &material) {

    std::shared_ptr<UniformMap> uniforms;

    if (shaderIDs.count(material->type())) {

        auto shaderID = shaderIDs.at(material->type());

        auto shader = shaders::ShaderLib::instance().get(shaderID);
        uniforms = std::make_shared<UniformMap>(shader.uniforms);

    } else {

        uniforms = material->uniforms;
    }

    return uniforms;
}

std::shared_ptr<GLProgram> GLPrograms::acquireProgram(const GLRenderer &renderer, const ProgramParameters &parameters, const std::string &cacheKey) {

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

        program = programs.emplace_back(GLProgram::create(renderer, cacheKey, parameters, bindingStates));
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
