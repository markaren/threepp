// https://github.com/mrdoob/three.js/blob/r129/src/materials/ShaderMaterial.js

#ifndef THREEPP_SHADERMATERIAL_HPP
#define THREEPP_SHADERMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

#include <unordered_map>

namespace threepp {

    class ShaderMaterial: public virtual Material,
                          public MaterialWithClipping,
                          public MaterialWithLights,
                          public MaterialWithWireframe,
                          public MaterialWithLineWidth,
                          public MaterialWithEnvMap,
                          public MaterialWithDefines {

    public:
        std::string vertexShader;
        std::string fragmentShader;
        UniformMap uniforms;

        /// Type-erased map for GPU-resident textures (e.g. WgpuTexture* for Wgpu).
        /// Keys are shader binding names. Values are backend-specific texture pointers.
        std::unordered_map<std::string, void*> customTextures;

        std::optional<std::string> index0AttributeName;
        bool uniformsNeedUpdate = false;

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<ShaderMaterial> create();

    protected:
        ShaderMaterial();

        std::shared_ptr<Material> createDefault() const override;
    };

}// namespace threepp

#endif//THREEPP_SHADERMATERIAL_HPP
