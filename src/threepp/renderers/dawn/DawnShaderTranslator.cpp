// GLSL -> SPIR-V translation pipeline for the Dawn renderer.
// Allows three.js ShaderMaterial GLSL to run on WebGPU via wgpu-native's
// native SPIR-V ingestion (WGPUSType_ShaderSourceSPIRV).

#include "DawnShaderTranslator.hpp"

#include "threepp/renderers/shaders/ShaderChunk.hpp"

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>

namespace threepp::dawn {

    namespace {

        // Three.js built-in transform uniforms → mapped to TransformUniforms UBO binding 0
        const std::set<std::string> kTransformUniforms = {
                "modelMatrix", "viewMatrix", "projectionMatrix",
                "modelViewMatrix", "normalMatrix", "cameraPosition"
        };

        bool isTransformUniform(const std::string& name) {
            return kTransformUniforms.count(name) > 0;
        }

        // glslang process-level init/shutdown (singleton)
        struct GlslangContext {
            GlslangContext()  { glslang::InitializeProcess(); }
            ~GlslangContext() { glslang::FinalizeProcess(); }
        };

        void ensureGlslangInit() {
            static GlslangContext ctx;
            (void)ctx;
        }

    }// namespace

    DawnShaderTranslator::DawnShaderTranslator() {
        ensureGlslangInit();
    }

    DawnShaderTranslator::~DawnShaderTranslator() = default;

    // ---------------------------------------------------------------------------
    // #include expansion
    // ---------------------------------------------------------------------------

    std::string DawnShaderTranslator::expandIncludes(const std::string& source) {
        static const std::regex rex(R"(#include +<([\w\d./]+)>)");

        std::string result;
        std::sregex_iterator it(source.begin(), source.end(), rex);
        std::sregex_iterator end;
        size_t pos = 0;

        while (it != end) {
            const std::smatch& match = *it;
            result.append(source, pos, match.position(0) - pos);
            pos = match.position(0) + match.length(0);

            const std::string chunkName = match[1].str();
            try {
                const std::string& chunk = shaders::ShaderChunk::instance().get(chunkName);
                result.append(chunk.empty() ? "" : expandIncludes(chunk));
            } catch (...) {
                // Unknown chunk — emit a comment so compilation can still proceed
                result += "/* [unresolved include: " + chunkName + "] */\n";
            }
            ++it;
        }

        if (pos == 0) return source;
        result.append(source, pos);
        return result;
    }

    // ---------------------------------------------------------------------------
    // Varying parser
    // ---------------------------------------------------------------------------

    std::vector<DawnShaderTranslator::VaryingInfo>
    DawnShaderTranslator::parseVaryings(const std::string& source) {
        // Match: [precision] TYPE name;
        static const std::regex rex(
                R"(\bvarying\s+(?:(?:lowp|mediump|highp)\s+)?(float|vec[234]|mat[234]|int|ivec[234])\s+(\w+)\s*;)");

        std::vector<VaryingInfo> result;
        std::set<std::string> seen;

        std::sregex_iterator it(source.begin(), source.end(), rex);
        std::sregex_iterator end;
        while (it != end) {
            std::string name = (*it)[2].str();
            if (seen.insert(name).second) {
                result.push_back({(*it)[1].str(), name});
            }
            ++it;
        }
        return result;
    }

    // ---------------------------------------------------------------------------
    // GLSL upgrader: three.js GLSL 100/ES → Vulkan GLSL 450
    // ---------------------------------------------------------------------------

    std::string DawnShaderTranslator::upgradeToVulkanGlsl(
            const std::string& expandedGlsl,
            bool isVertex,
            const std::vector<std::string>& varyingNames,
            const std::vector<int>& varyingLocations,
            bool hasCustomUniforms,
            const std::vector<std::string>& uniformNames,
            const std::vector<std::string>& textureNames) {

        std::ostringstream out;

        // --- Version header ---
        out << "#version 450\n\n";

        // --- Precision macros (GLSL 450 ignores them, but shaders reference them) ---
        out << "#define lowp\n";
        out << "#define mediump\n";
        out << "#define highp\n\n";

        // --- TransformUniforms UBO at binding 0 ---
        // Layout matches DawnRenderer's 256-byte TransformUniforms:
        //   model(64) + view(64) + proj(64) + normalCol0/1/2(3*16=48) + cameraPos(12) + _pad(4)
        out << "layout(std140, set=0, binding=0) uniform TransformUniforms {\n"
               "    mat4  modelMatrix;\n"
               "    mat4  viewMatrix;\n"
               "    mat4  projectionMatrix;\n"
               "    vec4  _normalCol0;\n"
               "    vec4  _normalCol1;\n"
               "    vec4  _normalCol2;\n"
               "    vec3  cameraPosition;\n"
               "    float _pad0;\n"
               "};\n\n";

        // normalMatrix derived from the three packed columns
        out << "#define normalMatrix mat3(_normalCol0.xyz, _normalCol1.xyz, _normalCol2.xyz)\n";
        // modelViewMatrix is not stored separately; derive it
        out << "#define modelViewMatrix (viewMatrix * modelMatrix)\n\n";

        // --- LightData UBO placeholder at binding 1 ---
        // We still declare the binding so the bind group slot is reserved,
        // but water's GLSL uses its own light helpers so we just need the slot.
        out << "layout(std140, set=0, binding=1) uniform LightPlaceholder {\n"
               "    vec4 _lightPad;\n"
               "};\n\n";

        // --- Custom uniforms UBO at binding 2 ---
        if (hasCustomUniforms) {
            // Extract user-declared non-transform, non-sampler uniforms from source
            static const std::regex uniformRex(
                    R"(\buniform\s+(?:(?:lowp|mediump|highp)\s+)?(float|vec[234]|mat[234]|int|ivec[234])\s+(\w+)\s*;)");

            std::set<std::string> emitted;
            std::vector<std::pair<std::string,std::string>> fields; // type, name

            std::sregex_iterator it(expandedGlsl.begin(), expandedGlsl.end(), uniformRex);
            std::sregex_iterator end;
            while (it != end) {
                std::string type = (*it)[1].str();
                std::string name = (*it)[2].str();
                bool isUserUniform = std::find(uniformNames.begin(), uniformNames.end(), name)
                                     != uniformNames.end();
                if (isUserUniform && !isTransformUniform(name) &&
                    emitted.insert(name).second) {
                    fields.push_back({type, name});
                }
                ++it;
            }

            if (!fields.empty()) {
                out << "layout(std140, set=0, binding=2) uniform CustomUniforms {\n";
                for (auto& [t, n] : fields) {
                    out << "    " << t << " " << n << ";\n";
                }
                out << "};\n\n";
            }
        }

        // --- Texture/sampler bindings at binding 3+ ---
        uint32_t nextBinding = hasCustomUniforms ? 3 : 2;
        for (auto& texName : textureNames) {
            // Check texture dimension
            bool isCube = expandedGlsl.find("samplerCube " + texName) != std::string::npos ||
                          expandedGlsl.find("samplerCube\t" + texName) != std::string::npos;
            if (isCube) {
                out << "layout(set=0, binding=" << nextBinding++ << ") uniform samplerCube " << texName << ";\n";
            } else {
                out << "layout(set=0, binding=" << nextBinding++ << ") uniform sampler2D " << texName << ";\n";
            }
        }
        if (!textureNames.empty()) out << "\n";

        // --- Vertex inputs (vertex stage only) ---
        if (isVertex) {
            out << "layout(location=0) in vec3 position;\n";
            out << "layout(location=1) in vec3 normal;\n";
            out << "layout(location=2) in vec2 uv;\n";
            out << "layout(location=3) in vec3 color;\n\n";
        }

        // --- Fragment color output (fragment stage only) ---
        if (!isVertex) {
            out << "layout(location=0) out vec4 fragColor;\n\n";
        }

        // --- Body: transform the original GLSL source ---
        std::string body = expandedGlsl;

        // 1. Strip #version and precision lines
        body = std::regex_replace(body, std::regex(R"(#version[^\n]*\n?)"), "");
        body = std::regex_replace(body,
                std::regex(R"(precision\s+(?:lowp|mediump|highp)\s+\w+\s*;[^\n]*)"), "");

        // 2. Strip attribute declarations (replaced by explicit layout inputs in header)
        body = std::regex_replace(body,
                std::regex(R"(\battribute\s+(?:(?:lowp|mediump|highp)\s+)?\w+\s+\w+\s*;[^\n]*)"), "");

        // 3. Strip all `uniform` declarations — they're now in UBOs or sampler bindings
        body = std::regex_replace(body,
                std::regex(R"(\buniform\s+(?:(?:lowp|mediump|highp)\s+)?\S+\s+\w+\s*;[^\n]*)"), "");

        // 4. Replace `varying TYPE name;` with layout(location=N) in/out
        {
            std::map<std::string,int> locMap;
            for (size_t i = 0; i < varyingNames.size(); ++i) {
                locMap[varyingNames[i]] = varyingLocations[i];
            }

            static const std::regex vRex(
                    R"(\bvarying\s+(?:(?:lowp|mediump|highp)\s+)?(float|vec[234]|mat[234]|int|ivec[234])\s+(\w+)\s*;)");

            std::string replaced;
            std::sregex_iterator it(body.begin(), body.end(), vRex);
            std::sregex_iterator end;
            size_t pos = 0;

            while (it != end) {
                const std::smatch& m = *it;
                replaced.append(body, pos, m.position(0) - pos);
                pos = m.position(0) + m.length(0);

                std::string type = m[1].str();
                std::string name = m[2].str();
                int loc = locMap.count(name) ? locMap[name] : 0;
                const char* dir = isVertex ? "out" : "in";
                replaced += "layout(location=" + std::to_string(loc) + ") " +
                            dir + " " + type + " " + name + ";";
                ++it;
            }

            if (pos > 0) {
                replaced.append(body, pos);
                body = std::move(replaced);
            }
        }

        // 5. texture2D/textureCube → texture
        body = std::regex_replace(body, std::regex(R"(\btexture2D\s*\()"), "texture(");
        body = std::regex_replace(body, std::regex(R"(\btextureCube\s*\()"), "texture(");

        // 6. gl_FragColor → fragColor
        if (!isVertex) {
            body = std::regex_replace(body, std::regex(R"(\bgl_FragColor\b)"), "fragColor");
        }

        out << body;
        return out.str();
    }

    // ---------------------------------------------------------------------------
    // SPIR-V compilation via glslang
    // ---------------------------------------------------------------------------

    std::optional<std::vector<uint32_t>> DawnShaderTranslator::compileToSpirv(
            const std::string& vulkanGlsl, bool isVertex, std::string& error) {

        EShLanguage stage = isVertex ? EShLangVertex : EShLangFragment;
        glslang::TShader shader(stage);

        const char* src = vulkanGlsl.c_str();
        const int srcLen = static_cast<int>(vulkanGlsl.size());
        shader.setStringsWithLengths(&src, &srcLen, 1);
        shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 100);
        shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
        shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);

        const TBuiltInResource* resources = GetDefaultResources();
        constexpr EShMessages msgs = EShMsgDefault;

        if (!shader.parse(resources, 450, false, msgs)) {
            error = "glslang parse:\n";
            error += shader.getInfoLog();
            const char* dbg = shader.getInfoDebugLog();
            if (dbg && dbg[0]) { error += "\n"; error += dbg; }
            return std::nullopt;
        }

        glslang::TProgram prog;
        prog.addShader(&shader);
        if (!prog.link(msgs)) {
            error = "glslang link:\n";
            error += prog.getInfoLog();
            return std::nullopt;
        }

        std::vector<uint32_t> spirv;
        glslang::SpvOptions opts;
        opts.validate = true;
        glslang::GlslangToSpv(*prog.getIntermediate(stage), spirv, &opts);

        if (spirv.empty()) {
            error = "glslang produced empty SPIR-V";
            return std::nullopt;
        }

        return spirv;
    }

    // ---------------------------------------------------------------------------
    // Public interface
    // ---------------------------------------------------------------------------

    TranslatedShader DawnShaderTranslator::translate(
            const std::string& vertexGlsl,
            const std::string& fragmentGlsl,
            const std::vector<std::string>& uniformNames,
            const std::vector<std::string>& textureNames) {

        // Cache lookup
        const size_t hash = std::hash<std::string>{}(vertexGlsl + "##SEP##" + fragmentGlsl);
        {
            auto it = cache_.find(hash);
            if (it != cache_.end()) return it->second;
        }

        TranslatedShader result;

        // Step 1: expand #include <chunk>
        std::string expandedVert = expandIncludes(vertexGlsl);
        std::string expandedFrag = expandIncludes(fragmentGlsl);

        // Step 2: collect varyings from both shaders; assign unified location indices
        auto vertVaryings = parseVaryings(expandedVert);
        auto fragVaryings = parseVaryings(expandedFrag);

        std::vector<std::string> varyingNames;
        std::vector<int> varyingLocations;
        std::set<std::string> seen;
        for (auto& v : vertVaryings) {
            if (seen.insert(v.name).second) {
                varyingLocations.push_back(static_cast<int>(varyingNames.size()));
                varyingNames.push_back(v.name);
            }
        }
        for (auto& v : fragVaryings) {
            if (seen.insert(v.name).second) {
                varyingLocations.push_back(static_cast<int>(varyingNames.size()));
                varyingNames.push_back(v.name);
            }
        }

        // Determine whether there are custom (non-transform, non-sampler) uniforms
        bool hasCustomUniforms = !uniformNames.empty();

        // Step 3: upgrade to Vulkan GLSL 450
        const std::string vulkanVert = upgradeToVulkanGlsl(
                expandedVert, true,
                varyingNames, varyingLocations,
                hasCustomUniforms, uniformNames, textureNames);

        const std::string vulkanFrag = upgradeToVulkanGlsl(
                expandedFrag, false,
                varyingNames, varyingLocations,
                hasCustomUniforms, uniformNames, textureNames);

        // Step 4: compile to SPIR-V
        std::string vertErr, fragErr;
        auto vertSpirv = compileToSpirv(vulkanVert, true, vertErr);
        if (!vertSpirv) {
            result.errorMessage = "Vertex: " + vertErr +
                    "\n--- Vertex GLSL ---\n" + vulkanVert;
            cache_[hash] = result;
            return result;
        }

        auto fragSpirv = compileToSpirv(vulkanFrag, false, fragErr);
        if (!fragSpirv) {
            result.errorMessage = "Fragment: " + fragErr +
                    "\n--- Fragment GLSL ---\n" + vulkanFrag;
            cache_[hash] = result;
            return result;
        }

        result.vertexSpirv   = std::move(*vertSpirv);
        result.fragmentSpirv = std::move(*fragSpirv);
        cache_[hash] = result;
        return result;
    }

}// namespace threepp::dawn
