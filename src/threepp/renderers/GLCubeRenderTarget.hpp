
#ifndef THREEPP_GLCUBERENDERTARGET_HPP
#define THREEPP_GLCUBERENDERTARGET_HPP

#include "threepp/cameras/CubeCamera.hpp"
#include "threepp/core/Shader.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/GLRenderTarget.hpp"

namespace threepp {

    class GLCubeRenderTarget: public GLRenderTarget {

    public:
        explicit GLCubeRenderTarget(int size, const Options& options = {})
            : GLRenderTarget(size, size, options) {

            this->texture = CubeTexture::create();
            if (options.mapping) this->texture->mapping = *options.mapping;
            if (options.wrapS) this->texture->wrapS = *options.wrapS;
            if (options.wrapT) this->texture->wrapT = *options.wrapT;
            if (options.magFilter) this->texture->magFilter = *options.magFilter;
            if (options.format) this->texture->format = *options.format;
            if (options.type) this->texture->type = *options.type;
            if (options.anisotropy) this->texture->anisotropy = *options.anisotropy;
            if (options.encoding) this->texture->encoding = *options.encoding;

            this->texture->generateMipmaps = options.generateMipmaps;
            this->texture->minFilter = options.minFilter.value_or(Filter::Linear);
        }

        void fromEquirectangularTexture(GLRenderer& renderer, Texture& texture) {

            this->texture->type = texture.type;
            this->texture->format = Format::RGBA;// see #18859
            this->texture->encoding = texture.encoding;

            this->texture->generateMipmaps = texture.generateMipmaps;
            this->texture->minFilter = texture.minFilter;
            this->texture->magFilter = texture.magFilter;

            Shader shader{

                    UniformMap{{"tEquirect", Uniform()}},

                    R"(
                    varying vec3 vWorldDirection;

                    vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

                        return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

                    }

                    void main() {

                        vWorldDirection = transformDirection( position, modelMatrix );

                        #include <begin_vertex>
                        #include <project_vertex>

                    })",

                    R"(
                        uniform sampler2D tEquirect;

                        varying vec3 vWorldDirection;

                        #include <common>

                        void main() {

                            vec3 direction = normalize( vWorldDirection );

                            vec2 sampleUV = equirectUv( direction );

                            gl_FragColor = texture2D( tEquirect, sampleUV );

                        })"

            };

            const auto geometry = BoxGeometry::create(5, 5, 5);

            const auto material = ShaderMaterial::create();

            material->name = "CubemapFromEquirect";

            material->uniforms = shader.uniforms;
            material->vertexShader = shader.vertexShader;
            material->fragmentShader = shader.fragmentShader;
            material->side = Side::Back;
            material->blending = Blending::None;

            material->uniforms.at("tEquirect").setValue(&texture);

            auto mesh = Mesh(geometry, material);

            const auto currentMinFilter = texture.minFilter;

            auto camera = CubeCamera(1, 10, *this);
            camera.update(renderer, mesh);

            // Avoid blurred poles
            if (texture.minFilter == Filter::LinearMipmapLinear) texture.minFilter = Filter::Linear;

            texture.minFilter = currentMinFilter;

            mesh.geometry()->dispose();
            mesh.material()->dispose();
        }

        void clear(GLRenderer& renderer, bool color, bool depth, bool, bool stencil) {

            const auto& currentRenderTarget = renderer.getRenderTarget();

            for (int i = 0; i < 6; i++) {

                renderer.setRenderTarget(this, i);

                renderer.clear(color, depth, stencil);
            }

            renderer.setRenderTarget(currentRenderTarget);
        }
    };

}// namespace threepp

#endif//THREEPP_GLCUBERENDERTARGET_HPP
