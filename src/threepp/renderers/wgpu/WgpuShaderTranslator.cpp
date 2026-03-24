// GLSL -> SPIR-V translation pipeline for the Wgpu renderer.
// Allows three.js ShaderMaterial GLSL to run on WebGPU via wgpu-native's
// native SPIR-V ingestion (WGPUSType_ShaderSourceSPIRV).

#include "WgpuShaderTranslator.hpp"

#include "threepp/renderers/shaders/ShaderChunk.hpp"

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <SPIRV/GlslangToSpv.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>

namespace threepp::wgpu {

    namespace {

        // Three.js built-in transform uniforms -> mapped to TransformUniforms UBO binding 0
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

    WgpuShaderTranslator::WgpuShaderTranslator() {
        ensureGlslangInit();
    }

    WgpuShaderTranslator::~WgpuShaderTranslator() = default;

    // ---------------------------------------------------------------------------
    // #include expansion
    // ---------------------------------------------------------------------------

    std::string WgpuShaderTranslator::expandIncludes(const std::string& source) {
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
                // Unknown chunk -- emit a comment so compilation can still proceed
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

    std::vector<WgpuShaderTranslator::VaryingInfo>
    WgpuShaderTranslator::parseVaryings(const std::string& source) {
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
    // GLSL upgrader: three.js GLSL 100/ES -> Vulkan GLSL 450
    // ---------------------------------------------------------------------------

    std::string WgpuShaderTranslator::upgradeToVulkanGlsl(
            const std::string& expandedGlsl,
            bool isVertex,
            const std::vector<std::string>& varyingNames,
            const std::vector<int>& varyingLocations,
            const std::vector<std::pair<std::string,std::string>>& customFields,
            const std::vector<std::string>& textureNames) {

        std::ostringstream out;

        // --- Version header ---
        out << "#version 450\n\n";

        // --- Precision macros (GLSL 450 ignores them, but shaders reference them) ---
        out << "#define lowp\n";
        out << "#define mediump\n";
        out << "#define highp\n\n";

        // --- Encoding stubs: WebGPU/Vulkan output is linear; these are no-ops ---
        // Prevents "undefined function" errors from encodings_pars/encodings_fragment chunks
        out << "#define linearToOutputTexel(v) (v)\n";
        out << "#define LinearToLinear(v) (v)\n";
        out << "#define GammaToLinear(v) (v)\n";
        out << "#define LinearToGamma(v) (v)\n\n";

        // --- TransformUniforms UBO at binding 0 ---
        // Layout matches WgpuRenderer's 256-byte TransformUniforms:
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
        // Use the pre-built unified field list so both vertex and fragment stages
        // declare an IDENTICAL CustomUniforms struct (same fields, same order, same size).
        // WebGPU requires all pipeline stages to agree on the binding-2 struct layout.
        //
        // Padding strategy (std140):
        //   float/int:  use 3 SEPARATE float variables (not an array!) for padding.
        //               Arrays in std140 have stride rounded to 16 bytes (Rule 4), so
        //               `float _pad[3]` = 48 bytes. Individual floats = 4 bytes each.
        //   vec3/ivec3: add 1 individual float to fill the trailing 4 bytes of the
        //               16-byte vec3 slot (vec3 has 16-byte base alignment in std140).
        //   vec4/mat4:  no padding needed (already 16/64 bytes in std140).
        //
        // We use the original uniform name as the struct member so the shader body can
        // reference it directly, without #define macros that would conflict with function
        // parameter names from included chunks (e.g., `float G_GGX_Smith(float alpha, ...)`).
        if (!customFields.empty()) {
            out << "layout(std140, set=0, binding=2) uniform CustomUniforms {\n";
            for (auto& [t, n] : customFields) {
                out << "    " << t << " " << n << ";\n";
                if (t == "float" || t == "int") {
                    // 3 individual floats = 12 bytes; together with the field = 16 bytes.
                    out << "    float _p0" << n << "_, _p1" << n << "_, _p2" << n << "_;\n";
                } else if (t == "vec3" || t == "ivec3") {
                    // 1 float fills the remaining 4 bytes of the 16-byte vec3 slot.
                    out << "    float _p" << n << "_;\n";
                }
                // vec4, mat4: already sized correctly in std140.
            }
            out << "};\n\n";
        }

        // --- Texture/sampler bindings at binding 3+ ---
        // Naga's SPIR-V frontend rejects combined image-sampler OpLoad patterns.
        // Instead we declare separate texture2D + sampler at consecutive bindings
        // and reconstruct the combined sampler2D via a #define so that the original
        // texture(...) call syntax still compiles without modification.
        // Layout: binding N = texture, binding N+1 = sampler (matches WgpuPipelines BGL).
        uint32_t nextBinding = !customFields.empty() ? 3 : 2;
        for (auto& texName : textureNames) {
            bool isCube = expandedGlsl.find("samplerCube " + texName) != std::string::npos ||
                          expandedGlsl.find("samplerCube\t" + texName) != std::string::npos;
            const std::string texType = isCube ? "textureCube" : "texture2D";
            const std::string smpType = isCube ? "samplerCube"  : "sampler2D";
            // Declare separate texture and sampler
            out << "layout(set=0, binding=" << nextBinding
                << ") uniform " << texType << " _tex_" << texName << ";\n";
            out << "layout(set=0, binding=" << nextBinding + 1
                << ") uniform sampler _smp_" << texName << ";\n";
            // Reconstruct combined sampler so existing texture(sampler, ...) calls work
            out << "#define " << texName << " " << smpType << "(_tex_" << texName << ", _smp_" << texName << ")\n";
            nextBinding += 2;
        }
        if (!textureNames.empty()) out << "\n";

        // --- Vertex inputs (vertex stage only) ---
        if (isVertex) {
            out << "layout(location=0) in vec3 position;\n";
            out << "layout(location=1) in vec3 normal;\n";
            out << "layout(location=2) in vec2 uv;\n";
            // Skip the color vertex attribute if 'color' is already declared as a
            // custom uniform — injecting both would cause a redefinition error.
            bool colorIsUniform = std::ranges::any_of(customFields,
                    [](const auto& f) { return f.second == "color"; });
            if (!colorIsUniform) {
                out << "layout(location=3) in vec3 color;\n";
            }
            out << "\n";
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

        // 3. Strip all `uniform` declarations -- they're now in UBOs or sampler bindings
        // Handles simple: `uniform float alpha;`
        // And arrays:     `uniform vec3 lightProbe[ 9 ];` or `uniform vec3 v[ N ];`
        body = std::regex_replace(body,
                std::regex(R"(\buniform\s+(?:(?:lowp|mediump|highp)\s+)?\S+\s+\w+(?:\s*\[[^\]]*\])?\s*;[^\n]*)"), "");

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

        // 5. texture2D/textureCube/texture2DProj -> texture/textureProj
        body = std::regex_replace(body, std::regex(R"(\btexture2DProj\s*\()"), "textureProj(");
        body = std::regex_replace(body, std::regex(R"(\btexture2D\s*\()"), "texture(");
        body = std::regex_replace(body, std::regex(R"(\btextureCube\s*\()"), "texture(");

        // 6. gl_FragColor -> fragColor
        if (!isVertex) {
            body = std::regex_replace(body, std::regex(R"(\bgl_FragColor\b)"), "fragColor");
        }

        out << body;
        return out.str();
    }

    // ---------------------------------------------------------------------------
    // SPIR-V compilation via glslang
    // ---------------------------------------------------------------------------

    std::optional<std::vector<uint32_t>> WgpuShaderTranslator::compileToSpirv(
            const std::string& vulkanGlsl, bool isVertex, std::string& error) {

        EShLanguage stage = isVertex ? EShLangVertex : EShLangFragment;
        glslang::TShader shader(stage);

        const char* src = vulkanGlsl.c_str();
        const int srcLen = static_cast<int>(vulkanGlsl.size());
        shader.setStringsWithLengths(&src, &srcLen, 1);
        shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 450);
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
        opts.validate          = true;  // validate SPIR-V with glslang's built-in validator
        opts.generateDebugInfo = false;
        opts.stripDebugInfo    = true;  // remove OpLine/OpSource -- naga can trip on them
        opts.disableOptimizer  = true;  // glslang optimizer can transform ImplicitLod->ExplicitLod incorrectly
        glslang::GlslangToSpv(*prog.getIntermediate(stage), spirv, &opts);

        if (spirv.empty()) {
            error = "glslang produced empty SPIR-V";
            return std::nullopt;
        }

        // Workaround: glslang 15.x generates OpImageSampleExplicitLod (88) with wc=5,
        // missing the required image_operands word. This is invalid SPIR-V and naga
        // rejects it. Patch opcode 88->87 (OpImageSampleImplicitLod) when wc==5 --
        // the operand layout (result_type, result, sampled_image, coordinate) is
        // identical, so this is a safe in-place fix for fragment shaders.
        // SPIR-V opcodes: 86=OpSampledImage, 87=OpImageSampleImplicitLod, 88=OpImageSampleExplicitLod
        if (!isVertex) {
            // Skip 5-word SPIR-V header (magic, version, generator, bound, schema)
            for (size_t i = 5; i < spirv.size(); ) {
                const uint32_t wc     = spirv[i] >> 16;
                const uint32_t opcode = spirv[i] & 0xFFFF;
                if (wc == 0) break;
                if (opcode == 88 /*OpImageSampleExplicitLod*/ && wc == 5) {
                    spirv[i] = (wc << 16) | 87; // -> OpImageSampleImplicitLod
                }
                i += wc;
            }
        }

        return spirv;
    }

    // ---------------------------------------------------------------------------
    // Public interface
    // ---------------------------------------------------------------------------

    TranslatedShader WgpuShaderTranslator::translate(
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

        // Filter uniformNames to only those directly declared in the original (pre-expansion) GLSL.
        // Uniforms from included chunks (fog, lights, shadow) live inside #ifdef guards that
        // aren't defined here, so they'd never appear in the compiled shader. Excluding them
        // keeps the binding-2 UBO small and consistent with what the CPU packer writes.
        std::set<std::string> directlyDeclared;
        {
            static const std::regex dRex(
                    R"(\buniform\s+(?:(?:lowp|mediump|highp)\s+)?(?:float|vec[234]|mat[234]|int|ivec[234])\s+(\w+)\s*;)");
            auto scan = [&](const std::string& src) {
                std::sregex_iterator it(src.begin(), src.end(), dRex);
                std::sregex_iterator end;
                while (it != end) { directlyDeclared.insert((*it)[1].str()); ++it; }
            };
            scan(vertexGlsl);    // original, pre-expanded
            scan(fragmentGlsl);  // original, pre-expanded
        }
        std::vector<std::string> filteredUniformNames;
        for (auto& n : uniformNames) {
            if (directlyDeclared.count(n)) filteredUniformNames.push_back(n);
        }
        std::sort(filteredUniformNames.begin(), filteredUniformNames.end());
        result.customUniformNames = filteredUniformNames;

        // Build unified (type, name) field list by scanning BOTH expanded shaders.
        // Using a combined name->type map ensures both vertex and fragment stages receive
        // an identical CustomUniforms struct -- required by WebGPU pipeline validation.
        std::map<std::string,std::string> uniformTypeMap;
        {
            static const std::regex typeRex(
                    R"(\buniform\s+(?:(?:lowp|mediump|highp)\s+)?(float|vec[234]|mat[234]|int|ivec[234])\s+(\w+)\s*;)");
            auto scanTypes = [&](const std::string& src) {
                std::sregex_iterator it(src.begin(), src.end(), typeRex);
                std::sregex_iterator end;
                while (it != end) { uniformTypeMap[(*it)[2].str()] = (*it)[1].str(); ++it; }
            };
            scanTypes(expandedVert);
            scanTypes(expandedFrag);
        }
        std::vector<std::pair<std::string,std::string>> customFields; // (type, name), sorted
        for (auto& n : filteredUniformNames) {  // already sorted alphabetically
            auto it = uniformTypeMap.find(n);
            if (it != uniformTypeMap.end()) customFields.push_back({it->second, n});
        }

        // Compute UBO byte size: float/int/vec3 -> vec4 slot (16 bytes), mat4 -> 64 bytes
        uint32_t uboSize = 0;
        for (auto& [t, n] : customFields) {
            if (t == "mat4") uboSize += 64;
            else uboSize += 16; // float, int, vec3, vec4, ivec3 all fit in a 16-byte slot
        }
        result.customUniformSize = uboSize;

        // Step 3: upgrade to Vulkan GLSL 450
        const std::string vulkanVert = upgradeToVulkanGlsl(
                expandedVert, true,
                varyingNames, varyingLocations,
                customFields, textureNames);

        const std::string vulkanFrag = upgradeToVulkanGlsl(
                expandedFrag, false,
                varyingNames, varyingLocations,
                customFields, textureNames);

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

}// namespace threepp::wgpu
