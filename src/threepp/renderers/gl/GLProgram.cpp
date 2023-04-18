
#include "threepp/renderers/gl/GLProgram.hpp"

#include "threepp/renderers/gl/GLBindingStates.hpp"
#include "threepp/renderers/gl/GLPrograms.hpp"
#include "threepp/renderers/gl/GLUniforms.hpp"

#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/shaders/ShaderChunk.hpp"
#include "threepp/utils/RegexUtil.hpp"
#include "threepp/utils/StringUtils.hpp"

#include <cmath>
#include <iostream>
#include <list>
#include <vector>

#include <glad/glad.h>

using namespace threepp;
using namespace threepp::gl;

namespace {

    inline unsigned int createShader(int type, const char* str) {

        const auto shader = glCreateShader(type);

        glShaderSource(shader, 1, &str, nullptr);
        glCompileShader(shader);

        return shader;
    }

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

    std::string loopReplacer(const std::smatch& match) {

        static std::regex reg1(R"(\[\s*i\s*\])");
        static std::regex reg2("UNROLLED_LOOP_INDEX");

        auto start = std::stoi(match[1].str());
        auto end = std::stoi(match[2].str());

        std::stringstream ss;
        for (int i = start; i < end; ++i) {

            ss << std::regex_replace(match[3].str(), reg1, "[ " + std::to_string(i) + " ]");
            ss.str() = std::regex_replace(ss.str(), reg2, std::to_string(i));
        }

        return ss.str();
    }

    std::string getTexelDecodingFunction(const std::string& functionName, int encoding) {

        const auto components = getEncodingComponents(encoding);
        return "vec4 " + functionName + "( vec4 value ) { return " + components.first + "ToLinear" + components.second + "; }";
    }

    std::string getTexelEncodingFunction(const std::string& functionName, int encoding) {

        const auto components = getEncodingComponents(encoding);
        return "vec4 " + functionName + "( vec4 value ) { return LinearTo" + components.first + components.second + "; }";
    }

    std::string getToneMappingFunction(const std::string& functionName, int toneMapping) {

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

    std::string generateDefines(const std::unordered_map<std::string, std::string>& defines) {

        std::list<std::string> chunks;

        for (const auto& [name, value] : defines) {

            if (value == "false") continue;

            std::stringstream chunk;
            chunk << "#define " << name << " " << value;
            chunks.emplace_back(chunk.str());
        }

        return utils::join(chunks);
    }

    std::unordered_map<std::string, GLint> fetchAttributeLocations(GLuint program) {

        std::unordered_map<std::string, GLint> attributes;

        GLint n;
        glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &n);

        GLsizei length;
        GLint size;
        GLenum type;
        GLchar nameBuffer[256];
        for (int i = 0; i < n; ++i) {

            glGetActiveAttrib(program, i, 256, &length, &size, &type, nameBuffer);

            std::string name(nameBuffer, length);
            attributes[name] = glGetAttribLocation(program, name.c_str());
        }

        return attributes;
    }

    bool filterEmptyLine(const std::string& str) {

        return str.empty();
    }

    void replaceLightNums(std::string& str, const ProgramParameters* parameters) {

        std::string& result = str;
        utils::replaceAll(result, "NUM_DIR_LIGHTS", std::to_string(parameters->numDirLights));
        utils::replaceAll(result, "NUM_SPOT_LIGHTS", std::to_string(parameters->numSpotLights));
        utils::replaceAll(result, "NUM_RECT_AREA_LIGHTS", std::to_string(parameters->numRectAreaLights));
        utils::replaceAll(result, "NUM_POINT_LIGHTS", std::to_string(parameters->numPointLights));
        utils::replaceAll(result, "NUM_HEMI_LIGHTS", std::to_string(parameters->numHemiLights));
        utils::replaceAll(result, "NUM_DIR_LIGHT_SHADOWS", std::to_string(parameters->numDirLightShadows));
        utils::replaceAll(result, "NUM_SPOT_LIGHT_SHADOWS", std::to_string(parameters->numSpotLightShadows));
        utils::replaceAll(result, "NUM_POINT_LIGHT_SHADOWS", std::to_string(parameters->numPointLightShadows));
    }

    void replaceClippingPlaneNums(std::string& str, const ProgramParameters* parameters) {

        utils::replaceAll(str, "NUM_CLIPPING_PLANES", std::to_string(parameters->numClippingPlanes));
        utils::replaceAll(str, "UNION_CLIPPING_PLANES", std::to_string(parameters->numClippingPlanes - parameters->numClipIntersection));
    }

    // Resolve Includes

    std::string resolveIncludes(const std::string& str) {

        static const std::regex rex("#include +<([\\w\\d.]+)>");

        std::string result;

        std::sregex_iterator rex_it(str.begin(), str.end(), rex);
        std::sregex_iterator rex_end;
        size_t pos = 0;

        while (rex_it != rex_end) {
            std::smatch match = *rex_it;
            result.append(str, pos, match.position(0) - pos);
            pos = match.position(0) + match.length(0);

            const std::ssub_match& sub = match[1];
            const std::string& r = shaders::ShaderChunk::instance().get(sub.str());
            if (r.empty()) {
                std::stringstream ss;
                ss << "unable to resolve #include <" << sub.str() << ">";
                throw std::logic_error(ss.str());
            }
            result.append(r);
            ++rex_it;
        }

        if (pos == 0) return str;
        else {
            result.append(str, pos, str.length());
            return result;
        }
    }

    // Unroll loops

    std::string unrollLoops(const std::string& glsl) {

        static const std::regex rex(R"(#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*\{([\s\S]+?)\}\s+#pragma unroll_loop_end)");

        return regex_replace(glsl, rex, loopReplacer);
    }

    inline std::string generatePrecision() {

        return "precision highp float;\nprecision highp int;\n#define HIGH_PRECISION";
    }

    std::string generateShadowMapTypeDefine(const ProgramParameters* parameters) {

        std::string shadowMapTypeDefine = "SHADOWMAP_TYPE_BASIC";

        if (parameters->shadowMapType == PCFShadowMap) {

            shadowMapTypeDefine = "SHADOWMAP_TYPE_PCF";

        } else if (parameters->shadowMapType == PCFSoftShadowMap) {

            shadowMapTypeDefine = "SHADOWMAP_TYPE_PCF_SOFT";

        } else if (parameters->shadowMapType == VSMShadowMap) {

            shadowMapTypeDefine = "SHADOWMAP_TYPE_VSM";
        }

        return shadowMapTypeDefine;
    }

    std::string generateEnvMapTypeDefine(const ProgramParameters* parameters) {

        std::string envMapTypeDefine = "ENVMAP_TYPE_CUBE";

        if (parameters->envMap) {

            switch (parameters->envMapMode) {

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

    std::string generateEnvMapModeDefine(const ProgramParameters* parameters) {

        std::string envMapModeDefine = "ENVMAP_MODE_REFLECTION";

        if (parameters->envMap) {

            switch (parameters->envMapMode) {

                case CubeRefractionMapping:
                case CubeUVRefractionMapping:

                    envMapModeDefine = "ENVMAP_MODE_REFRACTION";
                    break;
            }
        }

        return envMapModeDefine;
    }

    std::string generateEnvMapBlendingDefine(const ProgramParameters* parameters) {

        std::string envMapBlendingDefine = "ENVMAP_BLENDING_NONE";

        if (parameters->envMap && parameters->combine.has_value()) {

            switch (*parameters->combine) {

                case MultiplyOperation:
                    envMapBlendingDefine = "ENVMAP_BLENDING_MULTIPLY";
                    break;

                case MixOperation:
                    envMapBlendingDefine = "ENVMAP_BLENDING_MIX";
                    break;

                case AddOperation:
                    envMapBlendingDefine = "ENVMAP_BLENDING_ADD";
                    break;
            }
        }

        return envMapBlendingDefine;
    }


}// namespace


GLProgram::GLProgram(const GLRenderer* renderer, std::string cacheKey, const ProgramParameters* parameters, GLBindingStates* bindingStates)
    : cacheKey(std::move(cacheKey)), bindingStates(bindingStates) {

    auto& defines = parameters->defines;

    auto vertexShader = parameters->vertexShader;
    auto fragmentShader = parameters->fragmentShader;

    auto shadowMapTypeDefine = generateShadowMapTypeDefine(parameters);
    auto envMapTypeDefine = generateEnvMapTypeDefine(parameters);
    auto envMapModeDefine = generateEnvMapModeDefine(parameters);
    auto envMapBlendingDefine = generateEnvMapBlendingDefine(parameters);

    auto gammaFactorDefine = (renderer->gammaFactor > 0) ? renderer->gammaFactor : 1.f;

    auto customDefines = generateDefines(defines);

    this->program = glCreateProgram();

    std::string prefixVertex, prefixFragment;

    if (parameters->isRawShaderMaterial) {

        {
            std::vector<std::string> v{customDefines};

            v.erase(
                    std::remove_if(
                            v.begin(),
                            v.end(),
                            [](const std::string& s) { return filterEmptyLine(s); }),
                    v.end());

            prefixVertex = utils::join(v);

            if (!prefixVertex.empty()) {

                prefixVertex += "\n";
            }
        }

        {
            std::vector<std::string> v{customDefines};

            v.erase(
                    std::remove_if(
                            v.begin(),
                            v.end(),
                            [](const std::string& s) { return filterEmptyLine(s); }),
                    v.end());

            prefixFragment = utils::join(v);

            if (!prefixFragment.empty()) {

                prefixFragment += "\n";
            }
        }
    } else {

        {
            std::vector<std::string> v{

                    generatePrecision(),

                    "#define SHADER_NAME " + parameters->shaderName,

                    customDefines,

                    parameters->instancing ? "#define USE_INSTANCING" : "",
                    parameters->instancingColor ? "#define USE_INSTANCING_COLOR" : "",

                    parameters->supportsVertexTextures ? "#define VERTEX_TEXTURES" : "",

                    "#define GAMMA_FACTOR " + std::to_string(gammaFactorDefine),

                    "#define MAX_BONES " + std::to_string(parameters->maxBones),
                    (parameters->useFog && parameters->fog) ? "#define USE_FOG" : "",
                    (parameters->useFog && parameters->fogExp2) ? "#define FOG_EXP2" : "",

                    parameters->map ? "#define USE_MAP" : "",
                    parameters->envMap ? "#define USE_ENVMAP" : "",
                    parameters->envMap ? "#define " + envMapModeDefine : "",
                    parameters->lightMap ? "#define USE_LIGHTMAP" : "",
                    parameters->aoMap ? "#define USE_AOMAP" : "",
                    parameters->emissiveMap ? "#define USE_EMISSIVEMAP" : "",
                    parameters->bumpMap ? "#define USE_BUMPMAP" : "",
                    parameters->normalMap ? "#define USE_NORMALMAP" : "",
                    (parameters->normalMap && parameters->objectSpaceNormalMap) ? "#define OBJECTSPACE_NORMALMAP" : "",
                    (parameters->normalMap && parameters->tangentSpaceNormalMap) ? "#define TANGENTSPACE_NORMALMAP" : "",

                    parameters->clearcoatMap ? "#define USE_CLEARCOATMAP" : "",
                    parameters->clearcoatRoughnessMap ? "#define USE_CLEARCOAT_ROUGHNESSMAP" : "",
                    parameters->clearcoatNormalMap ? "#define USE_CLEARCOAT_NORMALMAP" : "",
                    parameters->displacementMap && parameters->supportsVertexTextures ? "#define USE_DISPLACEMENTMAP" : "",
                    parameters->specularMap ? "#define USE_SPECULARMAP" : "",
                    parameters->roughnessMap ? "#define USE_ROUGHNESSMAP" : "",
                    parameters->metalnessMap ? "#define USE_METALNESSMAP" : "",
                    parameters->alphaMap ? "#define USE_ALPHAMAP" : "",
                    parameters->transmission ? "#define USE_TRANSMISSION" : "",
                    parameters->transmissionMap ? "#define USE_TRANSMISSIONMAP" : "",
                    parameters->thicknessMap ? "#define USE_THICKNESSMAP" : "",

                    parameters->vertexTangents ? "#define USE_TANGENT" : "",
                    parameters->vertexColors ? "#define USE_COLOR" : "",
                    parameters->vertexAlphas ? "#define USE_COLOR_ALPHA" : "",
                    parameters->vertexUvs ? "#define USE_UV" : "",
                    parameters->uvsVertexOnly ? "#define UVS_VERTEX_ONLY" : "",

                    parameters->flatShading ? "#define FLAT_SHADED" : "",

                    parameters->skinning ? "#define USE_SKINNING" : "",
                    parameters->useVertexTexture ? "#define BONE_TEXTURE" : "",

                    parameters->morphTargets ? "#define USE_MORPHTARGETS" : "",
                    parameters->morphNormals && !parameters->flatShading ? "#define USE_MORPHNORMALS" : "",
                    parameters->doubleSided ? "#define DOUBLE_SIDED" : "",
                    parameters->flipSided ? "#define FLIP_SIDED" : "",

                    parameters->shadowMapEnabled ? "#define USE_SHADOWMAP" : "",
                    parameters->shadowMapEnabled ? "#define " + shadowMapTypeDefine : "",

                    parameters->sizeAttenuation ? "#define USE_SIZEATTENUATION" : "",

                    parameters->logarithmicDepthBuffer ? "#define USE_LOGDEPTHBUF" : "",

                    "uniform mat4 modelMatrix;",
                    "uniform mat4 modelViewMatrix;",
                    "uniform mat4 projectionMatrix;",
                    "uniform mat4 viewMatrix;",
                    "uniform mat3 normalMatrix;",
                    "uniform vec3 cameraPosition;",
                    "uniform bool isOrthographic;",

                    "#ifdef USE_INSTANCING",

                    "	attribute mat4 instanceMatrix;",

                    "#endif",

                    "#ifdef USE_INSTANCING_COLOR",

                    "	attribute vec3 instanceColor;",

                    "#endif",

                    "attribute vec3 position;",
                    "attribute vec3 normal;",
                    "attribute vec2 uv;",

                    "#ifdef USE_TANGENT",

                    "	attribute vec4 tangent;",

                    "#endif",

                    "#if defined( USE_COLOR_ALPHA )",

                    "	attribute vec4 color;",

                    "#elif defined( USE_COLOR )",

                    "	attribute vec3 color;",

                    "#endif",

                    "\n"

            };

            v.erase(std::remove_if(v.begin(), v.end(), [](const std::string& s) {
                        return s.empty();
                    }),
                    v.end());

            prefixVertex = utils::join(v);
        }

        {
            std::vector<std::string> v{

                    generatePrecision(),

                    "#define SHADER_NAME " + parameters->shaderName,

                    customDefines,

                    (parameters->alphaTest != 0) ? "#define ALPHATEST " + std::to_string(parameters->alphaTest) + (!(std::isnan(std::fmod(parameters->alphaTest, 1.f))) ? "" : ".0") : "",// add ".0" if integer

                    "#define GAMMA_FACTOR " + std::to_string(gammaFactorDefine),

                    (parameters->useFog && parameters->fog) ? "#define USE_FOG" : "",
                    (parameters->useFog && parameters->fogExp2) ? "#define FOG_EXP2" : "",

                    parameters->map ? "#define USE_MAP" : "",
                    parameters->matcap ? "#define USE_MATCAP" : "",
                    parameters->envMap ? "#define USE_ENVMAP" : "",
                    parameters->envMap ? "#define " + envMapTypeDefine : "",
                    parameters->envMap ? "#define " + envMapModeDefine : "",
                    parameters->envMap ? "#define " + envMapBlendingDefine : "",
                    parameters->lightMap ? "#define USE_LIGHTMAP" : "",
                    parameters->aoMap ? "#define USE_AOMAP" : "",
                    parameters->emissiveMap ? "#define USE_EMISSIVEMAP" : "",
                    parameters->bumpMap ? "#define USE_BUMPMAP" : "",
                    parameters->normalMap ? "#define USE_NORMALMAP" : "",
                    (parameters->normalMap && parameters->objectSpaceNormalMap) ? "#define OBJECTSPACE_NORMALMAP" : "",
                    (parameters->normalMap && parameters->tangentSpaceNormalMap) ? "#define TANGENTSPACE_NORMALMAP" : "",
                    parameters->clearcoatMap ? "#define USE_CLEARCOATMAP" : "",
                    parameters->clearcoatRoughnessMap ? "#define USE_CLEARCOAT_ROUGHNESSMAP" : "",
                    parameters->clearcoatNormalMap ? "#define USE_CLEARCOAT_NORMALMAP" : "",
                    parameters->specularMap ? "#define USE_SPECULARMAP" : "",
                    parameters->roughnessMap ? "#define USE_ROUGHNESSMAP" : "",
                    parameters->metalnessMap ? "#define USE_METALNESSMAP" : "",
                    parameters->alphaMap ? "#define USE_ALPHAMAP" : "",

                    parameters->sheen ? "#define USE_SHEEN" : "",
                    parameters->transmission ? "#define USE_TRANSMISSION" : "",
                    parameters->transmissionMap ? "#define USE_TRANSMISSIONMAP" : "",
                    parameters->thicknessMap ? "#define USE_THICKNESSMAP" : "",

                    parameters->vertexTangents ? "#define USE_TANGENT" : "",
                    parameters->vertexColors || parameters->instancingColor ? "#define USE_COLOR" : "",
                    parameters->vertexAlphas ? "#define USE_COLOR_ALPHA" : "",
                    parameters->vertexUvs ? "#define USE_UV" : "",
                    parameters->uvsVertexOnly ? "#define UVS_VERTEX_ONLY" : "",

                    parameters->gradientMap ? "#define USE_GRADIENTMAP" : "",

                    parameters->flatShading ? "#define FLAT_SHADED" : "",

                    parameters->doubleSided ? "#define DOUBLE_SIDED" : "",
                    parameters->flipSided ? "#define FLIP_SIDED" : "",

                    parameters->shadowMapEnabled ? "#define USE_SHADOWMAP" : "",
                    parameters->shadowMapEnabled ? "#define " + shadowMapTypeDefine : "",

                    parameters->premultipliedAlpha ? "#define PREMULTIPLIED_ALPHA" : "",

                    parameters->physicallyCorrectLights ? "#define PHYSICALLY_CORRECT_LIGHTS" : "",

                    parameters->logarithmicDepthBuffer ? "#define USE_LOGDEPTHBUF" : "",

                    "uniform mat4 viewMatrix;",
                    "uniform vec3 cameraPosition;",
                    "uniform bool isOrthographic;",

                    (parameters->toneMapping != NoToneMapping) ? "#define TONE_MAPPING" : "",
                    (parameters->toneMapping != NoToneMapping) ? shaders::ShaderChunk::instance().tonemapping_pars_fragment() : "",// this code is required here because it is used by the toneMapping() function defined below
                    (parameters->toneMapping != NoToneMapping) ? getToneMappingFunction("toneMapping", parameters->toneMapping) : "",

                    parameters->dithering ? "#define DITHERING" : "",

                    shaders::ShaderChunk::instance().encodings_pars_fragment(),// this code is required here because it is used by the various encoding/decoding function defined below
                    parameters->map ? getTexelDecodingFunction("mapTexelToLinear", parameters->mapEncoding) : "",
                    parameters->matcap ? getTexelDecodingFunction("matcapTexelToLinear", parameters->matcapEncoding) : "",
                    parameters->envMap ? getTexelDecodingFunction("envMapTexelToLinear", parameters->envMapEncoding) : "",
                    parameters->emissiveMap ? getTexelDecodingFunction("emissiveMapTexelToLinear", parameters->emissiveMapEncoding) : "",
                    parameters->lightMap ? getTexelDecodingFunction("lightMapTexelToLinear", parameters->lightMapEncoding) : "",
                    getTexelEncodingFunction("linearToOutputTexel", parameters->outputEncoding),

                    parameters->depthPacking ? "#define DEPTH_PACKING " + std::to_string(parameters->depthPacking) : "",

                    "\n"

            };

            v.erase(
                    std::remove_if(
                            v.begin(),
                            v.end(),
                            [](const std::string& s) { return s.empty(); }),
                    v.end());

            prefixFragment = utils::join(v);
        }
    }

    vertexShader = resolveIncludes(vertexShader);
    replaceLightNums(vertexShader, parameters);
    replaceClippingPlaneNums(vertexShader, parameters);

    fragmentShader = resolveIncludes(fragmentShader);
    replaceLightNums(fragmentShader, parameters);
    replaceClippingPlaneNums(fragmentShader, parameters);

    vertexShader = unrollLoops(vertexShader);
    fragmentShader = unrollLoops(fragmentShader);

    if (!parameters->isRawShaderMaterial) {

        {
            std::vector<std::string> v{
                    "#version 330 core\n",
                    "#define attribute in",
                    "#define varying out",
                    "#define texture2D texture"

            };

            prefixVertex = utils::join(v) + "\n" + prefixVertex;
        }

        {
            std::vector<std::string> v{
                    "#version 330 core\n",
                    "#define varying in",
                    "out highp vec4 pc_fragColor;",
                    "#define gl_FragColor pc_fragColor",
                    //                    ( parameters->glslVersion == GLSL3 ) ? "" : "out highp vec4 pc_fragColor;",
                    //                    ( parameters->glslVersion == GLSL3 ) ? "" : "#define gl_FragColor pc_fragColor",
                    "#define gl_FragDepthEXT gl_FragDepth",
                    "#define texture2D texture",
                    "#define textureCube texture",
                    "#define texture2DProj textureProj",
                    "#define texture2DLodEXT textureLod",
                    "#define texture2DProjLodEXT textureProjLod",
                    "#define textureCubeLodEXT textureLod",
                    "#define texture2DGradEXT textureGrad",
                    "#define texture2DProjGradEXT textureProjGrad",
                    "#define textureCubeGradEXT textureGrad"

            };

            prefixFragment = utils::join(v) + "\n" + prefixFragment;
        }
    }

    std::string vertexGlsl = prefixVertex + vertexShader;
    std::string fragmentGlsl = prefixFragment + fragmentShader;

    const auto glVertexShader = createShader(GL_VERTEX_SHADER, vertexGlsl.c_str());
    const auto glFragmentShader = createShader(GL_FRAGMENT_SHADER, fragmentGlsl.c_str());

    glAttachShader(program, glVertexShader);
    glAttachShader(program, glFragmentShader);

    if (parameters->index0AttributeName) {

        glBindAttribLocation(program, 0, parameters->index0AttributeName.value().c_str());
    }

    glLinkProgram(program);

    if (renderer->checkShaderErrors) {

        int length;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);

        if (length != 0) {

            std::string msg;
            msg.resize(length);
            glGetProgramInfoLog(program, length, nullptr, &msg.front());

            std::cerr << "[Shader error] " << msg << std::endl;
        }
    }

    glDeleteShader(glVertexShader);
    glDeleteShader(glFragmentShader);
}

std::shared_ptr<GLUniforms> GLProgram::getUniforms() {

    if (!cachedUniforms) {
        cachedUniforms = std::make_shared<GLUniforms>(program);
    }

    return cachedUniforms;
}

std::unordered_map<std::string, int> GLProgram::getAttributes() {

    if (cachedAttributes.empty()) {

        cachedAttributes = fetchAttributeLocations(program);
    }

    return cachedAttributes;
}

void GLProgram::destroy() {

    bindingStates->releaseStatesOfProgram(*this);

    glDeleteProgram(program);
    this->program = -1;
}
