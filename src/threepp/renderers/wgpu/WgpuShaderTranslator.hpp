// GLSL -> SPIR-V translation for the Wgpu renderer.
// Allows ShaderMaterial with three.js-style GLSL to run on WebGPU by:
//   1. Expanding #include <chunk> directives via ShaderChunk
//   2. Upgrading three.js GLSL to Vulkan GLSL 450 (UBO layout, locations, etc.)
//   3. Compiling to SPIR-V via glslang
// The resulting SPIR-V is fed directly to wgpu-native (WGPUSType_ShaderSourceSPIRV).

#ifndef THREEPP_WGPUSHADERTRANSLATOR_HPP
#define THREEPP_WGPUSHADERTRANSLATOR_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace threepp::wgpu {

    struct TranslatedShader {
        std::vector<uint32_t> vertexSpirv;
        std::vector<uint32_t> fragmentSpirv;
        std::string errorMessage;
        // Uniform names that were placed in the binding-2 UBO, sorted alphabetically.
        // Only includes uniforms directly declared in the original (pre-expanded) GLSL.
        // The CPU packer must use this list (same order) to match the UBO layout.
        std::vector<std::string> customUniformNames;
        // Byte size of the CustomUniforms UBO (std140 layout, with padding).
        // Zero if no custom uniforms exist.
        uint32_t customUniformSize = 0;

        [[nodiscard]] bool success() const {
            return errorMessage.empty() &&
                   !vertexSpirv.empty() &&
                   !fragmentSpirv.empty();
        }
    };

    class WgpuShaderTranslator {
    public:
        WgpuShaderTranslator();
        ~WgpuShaderTranslator();

        // Translate a pair of three.js GLSL shaders to SPIR-V.
        // uniformNames: non-texture uniform names from ShaderMaterial::uniforms
        // textureNames: texture uniform names (sorted), used to assign binding indices
        TranslatedShader translate(
                const std::string& vertexGlsl,
                const std::string& fragmentGlsl,
                const std::vector<std::string>& uniformNames,
                const std::vector<std::string>& textureNames,
                bool isInstanced = false);

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
        // customFields: unified (type, name) pairs sorted alphabetically --
        // same list is passed to both vertex and fragment so they share
        // an identical binding-2 struct layout.
        std::string upgradeToVulkanGlsl(
                const std::string& expandedGlsl,
                bool isVertex,
                const std::vector<std::string>& varyingNames,
                const std::vector<int>& varyingLocations,
                const std::vector<std::pair<std::string,std::string>>& customFields,
                const std::vector<std::string>& textureNames,
                bool isInstanced);

        // Compile Vulkan GLSL 450 to SPIR-V via glslang
        std::optional<std::vector<uint32_t>> compileToSpirv(
                const std::string& vulkanGlsl, bool isVertex, std::string& error);

        // Cache keyed by hash(vertGlsl + fragGlsl)
        std::unordered_map<size_t, TranslatedShader> cache_;
    };

}// namespace threepp::wgpu

#endif//THREEPP_WGPUSHADERTRANSLATOR_HPP
