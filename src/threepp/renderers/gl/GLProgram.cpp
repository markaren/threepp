
#include "GLProgram.hpp"
#include "GLCapabilities.hpp"
#include "GLPrograms.hpp"
#include "threepp/renderers/shaders/ShaderChunk.hpp"

#include "threepp/constants.hpp"
#include "threepp/utils/StringUtils.hpp"

#include <glad/glad.h>

#include <any>
#include <iostream>
#include <regex>
#include <vector>

using namespace threepp;
using namespace threepp::gl;

namespace {

    std::pair<std::string, std::string> getEncodingComponents(int encoding) {

        switch (encoding) {

            case LinearEncoding:
                return {"Linear", "( value )"};
            case sRGBEncoding:
                return {"sRGB", "( value )"};
            case RGBEEncoding:
                return {"RGBE", "( value )"};
            case RGBM7Encoding:
                return {"RGBM", "( value, 7.0 )"};
            case RGBM16Encoding:
                return {"RGBM", "( value, 16.0 )"};
            case RGBDEncoding:
                return {"RGBD", "( value, 256.0 )"};
            case GammaEncoding:
                return {"Gamma", "( value, float( GAMMA_FACTOR ) )"};
            case LogLuvEncoding:
                return {"LogLuv", "( value )"};
            default:
                std::cerr << "THREE.WebGLProgram: Unsupported encoding:" << encoding << std::endl;
                return {"Linear", "( value )"};
        }
    }

    std::string getTexelDecodingFunction(const std::string &functionName, int encoding) {

        const auto components = getEncodingComponents(encoding);
        return "vec4 " + functionName + "( vec4 value ) { return " + components.first + "ToLinear" + components.second + "; }";
    }

    std::string getTexelEncodingFunction(const std::string &functionName, int encoding) {

        const auto components = getEncodingComponents(encoding);
        return "vec4 " + functionName + "( vec4 value ) { return LinearTo" + components.first + components.second + "; }";
    }

    std::string getToneMappingFunction(const std::string &functionName, int toneMapping) {

        std::string toneMappingName;

        switch (toneMapping) {

            case LinearToneMapping:
                toneMappingName = "Linear";
                break;

            case ReinhardToneMapping:
                toneMappingName = "Reinhard";
                break;

            case CineonToneMapping:
                toneMappingName = "OptimizedCineon";
                break;

            case ACESFilmicToneMapping:
                toneMappingName = "ACESFilmic";
                break;

            case CustomToneMapping:
                toneMappingName = "Custom";
                break;

            default:
                std::cerr << "THREE.WebGLProgram: Unsupported toneMapping:" << toneMapping << std::endl;
                toneMappingName = "Linear";
        }

        return "vec3 " + functionName + "( vec3 color ) { return " + toneMappingName + "ToneMapping( color ); }";
    }

    std::string generateDefines(const std::unordered_map<std::string, std::string> &defines) {

        std::vector<std::string> chunks;

        for (const auto &[name, value] : defines) {

            if (value == "false") continue;

            std::string chunk("#define " + name + " " + value);
            chunks.emplace_back(chunk);
        }

        std::string result;
        utils::join(chunks, '\n', result);

        return result;
    }

    std::unordered_map<std::string, GLint> fetchAttributeLocations(GLuint program) {

        std::unordered_map<std::string, GLint> attributes;

        GLint n;
        glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &n);

        GLsizei length;
        GLint size;
        GLenum type;
        GLchar name[256];
        for (int i = 0; i < n; i++) {

            glGetActiveAttrib(program, i, 256, &length, &size, &type, name);

            attributes[name] = glGetAttribLocation(program, name);
        }

        return attributes;
    }

    bool filterEmptyLine(const std::string &str) {

        return str.empty();
    }

    std::string replaceLightNums(const std::string &str, const GLPrograms::Parameters &parameters) {

        std::string result = str;
        result = std::regex_replace(result, std::regex("NUM_DIR_LIGHTS"), std::to_string(parameters.numDirLights));
        result = std::regex_replace(result, std::regex("NUM_SPOT_LIGHTS"), std::to_string(parameters.numSpotLights));
        result = std::regex_replace(result, std::regex("NUM_RECT_AREA_LIGHTS"), std::to_string(0));
        result = std::regex_replace(result, std::regex("NUM_POINT_LIGHTS"), std::to_string(parameters.numPointLights));
        result = std::regex_replace(result, std::regex("NUM_HEMI_LIGHTS"), std::to_string(0));
        result = std::regex_replace(result, std::regex("NUM_DIR_LIGHT_SHADOWS"), std::to_string(parameters.numDirLightShadows));
        result = std::regex_replace(result, std::regex("NUM_SPOT_LIGHT_SHADOWS"), std::to_string(parameters.numSpotLightShadows));
        result = std::regex_replace(result, std::regex("NUM_POINT_LIGHT_SHADOWS"), std::to_string(parameters.numPointLightShadows));

        return result;
    }

    std::string replaceClippingPlaneNums(const std::string &str, const GLPrograms::Parameters &parameters) {

        std::string result;
        result = std::regex_replace(result, std::regex("NUM_CLIPPING_PLANES"), std::to_string(parameters.numClippingPlanes));
        result = std::regex_replace(result, std::regex("UNION_CLIPPING_PLANES"), std::to_string(parameters.numClippingPlanes - parameters.numClipIntersection));

        return result;
    }

    // Resolve Includes

    std::regex includePattern("^[ \\t]*#include +<([\\w\\d./]+)>", std::regex::extended);

    std::string resolveIncludes(const std::string &str) {

        std::string result;

        std::sregex_iterator rex_it(str.begin(), str.end(), includePattern);
        std::sregex_iterator end;
        size_t pos = 0;

        while (rex_it != end) {

            std::smatch match = *rex_it;
            result.append(str, pos, match.position(0) - pos);

            std::ssub_match sub = match[1];
            std::string r = shaders::ShaderChunk::instance().get(sub.str(), "ShaderChunk");

            if (r.empty()) {

                std::stringstream ss;
                ss << "unable to resolve #include <" << sub.str() << ">";
                throw std::logic_error(ss.str());
            }

            result.append(r);
            rex_it++;
        }

        if (pos == 0) {

            return str;
        } else {

            result.append(str, pos, str.length());
            return result;
        }
    }

    // Unroll loops

    std::regex unrollLoopPattern("#pragma unroll_loop_start\\s+for\\s*\\(\\s*int\\s+i\\s*=\\s*(\\d+)\\s*;\\s*i\\s*<\\s*(\\d+)\\s*;\\s*i\\s*\\+\\+\\s*\\)\\s*{([\\s\\S]+?)}\\s+#pragma unroll_loop_end");

    std::string unrollLoops(const std::string &glsl) {

        return "";
    }

    std::string generatePrecision() {

        return "precision highp float;\nprecision highp int;\n#define HIGH_PRECISION";
    }

    std::string generateShadowMapTypeDefine(const GLPrograms::Parameters &parameters) {

        std::string shadowMapTypeDefine = "SHADOWMAP_TYPE_BASIC";

        if (parameters.shadowMapType == PCFShadowMap) {

            shadowMapTypeDefine = "SHADOWMAP_TYPE_PCF";

        } else if (parameters.shadowMapType == PCFSoftShadowMap) {

            shadowMapTypeDefine = "SHADOWMAP_TYPE_PCF_SOFT";

        } else if (parameters.shadowMapType == VSMShadowMap) {

            shadowMapTypeDefine = "SHADOWMAP_TYPE_VSM";
        }

        return shadowMapTypeDefine;
    }

    std::string generateEnvMapTypeDefine(const GLPrograms::Parameters &parameters) {

        std::string envMapTypeDefine = "ENVMAP_TYPE_CUBE";

        if (parameters.envMap) {

            switch (parameters.envMapMode) {

                case CubeReflectionMapping:
                case CubeRefractionMapping:
                    envMapTypeDefine = "ENVMAP_TYPE_CUBE";
                    break;

                case CubeUVReflectionMapping:
                case CubeUVRefractionMapping:
                    envMapTypeDefine = "ENVMAP_TYPE_CUBE_UV";
                    break;
            }
        }

        return envMapTypeDefine;
    }


}// namespace