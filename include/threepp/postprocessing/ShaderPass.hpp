// https://github.com/mrdoob/three.js/blob/r129/examples/jsm/postprocessing/ShaderPass.js

#ifndef THREEPP_POSTPROCESSING_SHADERPASS_HPP
#define THREEPP_POSTPROCESSING_SHADERPASS_HPP

#include "threepp/core/Shader.hpp"
#include "threepp/postprocessing/Pass.hpp"

#include <string>

namespace threepp {

    class ShaderMaterial;

    // Runs one fragment shader over the whole frame.
    //
    // The pass binds the incoming image to the uniform named by `textureID`
    // ("tDiffuse" unless told otherwise) and draws the result into the write
    // buffer. Shaders written against the three.js addon shaders compile as-is:
    // the GL backend defines `varying`, `texture2D` and `gl_FragColor` for
    // GLSL 3.30 and resolves `#include <...>` chunks.
    class ShaderPass: public Pass {

    public:
        explicit ShaderPass(const Shader& shader, std::string textureID = "tDiffuse");

        explicit ShaderPass(std::shared_ptr<ShaderMaterial> material, std::string textureID = "tDiffuse");

        // The live uniform map — assign into it to drive the effect.
        [[nodiscard]] UniformMap& uniforms();

        [[nodiscard]] std::shared_ptr<ShaderMaterial> material() const;

        void render(GLRenderer& renderer,
                    RenderTarget* writeBuffer,
                    RenderTarget* readBuffer,
                    float deltaTime,
                    bool maskActive) override;

        ~ShaderPass() override;

    private:
        std::string textureID_;
        std::shared_ptr<ShaderMaterial> material_;
        std::unique_ptr<FullScreenQuad> fsQuad_;
    };

}// namespace threepp

#endif//THREEPP_POSTPROCESSING_SHADERPASS_HPP
