// GLSL -> SPIR-V translation for the Dawn renderer.
// Allows ShaderMaterial with three.js-style GLSL to run on WebGPU by:
//   1. Expanding #include <chunk> directives via ShaderChunk
//   2. Upgrading three.js GLSL to Vulkan GLSL 450 (UBO layout, locations, etc.)
//   3. Compiling to SPIR-V via glslang
// The resulting SPIR-V is fed directly to wgpu-native (WGPUSType_ShaderSourceSPIRV).

#ifndef THREEPP_DAWNSHADERTRANSLATOR_HPP
#define THREEPP_DAWNSHADERTRANSLATOR_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace threepp::dawn {

    struct TranslatedShader {
        std::vector<uint32_t> vertexSpirv;
        std::vector<uint32_t> fragmentSpirv;
        std::string errorMessage;

        [[nodiscard]] bool success() const {
            return errorMessage.empty() &&
                   !vertexSpirv.empty() &&
                   !fragmentSpirv.empty();
        }
    };

    class DawnShaderTranslator {
    public:
        DawnShaderTranslator();
        ~DawnShaderTranslator();

        // Translate a pair of three.js GLSL shaders to SPIR-V.
        // uniformNames: non-texture uniform names from ShaderMaterial::uniforms
        // textureNames: texture uniform names (sorted), used to assign binding indices
        TranslatedShader translate(
                const std::string& vertexGlsl,
                const std::string& fragmentGlsl,
                const std::vector<std::string>& uniformNames,
                const std::vector<std::string>& textureNames);

    private:
        // Recursively expand #include <chunk> using ShaderChunk::instance()
        std::string expandIncludes(const std::string& source);

        // Parse `varying TYPE name;` declarations from a GLSL source string
        struct VaryingInfo { std::string type; std::string name; };
        std::vector<VaryingInfo> parseVaryings(const std::string& source);

        // Upgrade three.js GLSL to Vulkan GLSL 450:
        //   - Remove #version / precision lines
        //   - Inject UBO at binding 0 (TransformUniforms: model, view, proj, normalMatrix, cameraPos)
        //   - Inject custom UBO at binding 2 for user uniforms (if any)
        //   - Add layout(set=0, binding=N) to sampler2D uniforms
        //   - Replace `varying` with layout(location=N) in/out
        //   - Add vertex attribute inputs (vertex stage only)
        //   - Add fragColor output (fragment stage only)
        //   - Replace texture2D()/textureCube() with texture()
        //   - #define modelViewMatrix (viewMatrix * modelMatrix)
        std::string upgradeToVulkanGlsl(
                const std::string& expandedGlsl,
                bool isVertex,
                const std::vector<std::string>& varyingNames,
                const std::vector<int>& varyingLocations,
                bool hasCustomUniforms,
                const std::vector<std::string>& uniformNames,
                const std::vector<std::string>& textureNames);

        // Compile Vulkan GLSL 450 to SPIR-V via glslang
        std::optional<std::vector<uint32_t>> compileToSpirv(
                const std::string& vulkanGlsl, bool isVertex, std::string& error);

        // Cache keyed by hash(vertGlsl + fragGlsl)
        std::unordered_map<size_t, TranslatedShader> cache_;
    };

}// namespace threepp::dawn

#endif//THREEPP_DAWNSHADERTRANSLATOR_HPP
