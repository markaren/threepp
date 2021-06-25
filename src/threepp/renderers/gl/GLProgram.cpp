
#include "GLProgram.hpp"
#include "GLCapabilities.hpp"
#include "GLPrograms.hpp"

#include "threepp/constants.hpp"

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

    std::string generateDefines(const std::unordered_map<std::string, std::any> &defines) {

        //            std::vector<std::string> chunks;
        //
        //            for ( const auto& [name, value] : defines ) {
        //
        //                if ( value == false ) continue;
        //
        //                chunks.emplace_back( "#define " + name + " " + value );
        //
        //            }
        //
        //            std::string result;
        //            join(chunks, '\n',result);
        //
        //            return result;

        //TODO

        return "";
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

    std::string replaceLightNums(const std::string &str, GLPrograms::Parameters &parameters) {

        std::string result = str;
        result = std::regex_replace(result, std::regex("NUM_DIR_LIGHTS"), std::to_string(parameters.numDirLights));
        result = std::regex_replace(result, std::regex("NUM_SPOT_LIGHTS"), std::to_string(parameters.numSpotLights));
        result = std::regex_replace(result, std::regex("NUM_RECT_AREA_LIGHTS"), std::to_string(parameters.numRectAreaLights));
        result = std::regex_replace(result, std::regex("NUM_POINT_LIGHTS"), std::to_string(parameters.numPointLights));
        result = std::regex_replace(result, std::regex("NUM_HEMI_LIGHTS"), std::to_string(parameters.numHemiLights));
        result = std::regex_replace(result, std::regex("NUM_DIR_LIGHT_SHADOWS"), std::to_string(parameters.numDirLightShadows));
        result = std::regex_replace(result, std::regex("NUM_SPOT_LIGHT_SHADOWS"), std::to_string(parameters.numSpotLightShadows));
        result = std::regex_replace(result, std::regex("NUM_POINT_LIGHT_SHADOWS"), std::to_string(parameters.numPointLightShadows));

        return result;
    }


}// namespace