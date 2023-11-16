
#include "GLCubeMaps.hpp"

#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/GLRenderTarget.hpp"

#include "threepp/renderers/GLCubeRenderTarget.hpp"
#include "threepp/textures/Texture.hpp"

using namespace threepp;
using namespace threepp::gl;

namespace {

    void mapTextureMapping(Texture& texture, Mapping mapping) {

        if (mapping == Mapping::EquirectangularReflection) {

            texture.mapping = Mapping::CubeReflection;

        } else if (mapping == Mapping::EquirectangularRefraction) {

            texture.mapping = Mapping::CubeRefraction;
        }
    }

}// namespace

GLCubeMaps::GLCubeMaps(GLRenderer& renderer)
    : renderer(renderer) {}

std::shared_ptr<Texture> GLCubeMaps::get(const std::shared_ptr<Texture>& texture) {

    if (texture) {

        const auto mapping = texture->mapping;

        if (mapping == Mapping::EquirectangularReflection || mapping == Mapping::EquirectangularRefraction) {

            if (cubemaps.count(texture)) {

                const auto cubemap = cubemaps.at(texture)->texture;
                mapTextureMapping(*cubemap, texture->mapping);
                return cubemap;

            } else {

                const auto& image = texture->image;

                if (image && image->height > 0) {


                    const auto& currentRenderTarget = renderer.getRenderTarget();

                    const auto renderTarget = std::make_shared<GLCubeRenderTarget>( image->height / 2 );
                    renderTarget->fromEquirectangularTexture( renderer, *texture );
                    cubemaps[texture] = renderTarget;


                }

            }

        }

    }

    return texture;
}

void GLCubeMaps::dispose() {

    cubemaps.clear();
}

