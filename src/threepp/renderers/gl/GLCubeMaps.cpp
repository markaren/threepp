
#include "GLCubeMaps.hpp"

#include "GLPMREM.hpp"

#include "threepp/renderers/GLCubeRenderTarget.hpp"

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

GLCubeMaps::~GLCubeMaps() = default;

Texture* GLCubeMaps::get(Texture* texture) {

    if (texture) {

        const auto mapping = texture->mapping;

        if (mapping == Mapping::EquirectangularReflection || mapping == Mapping::EquirectangularRefraction) {

            if (cubemaps.contains(texture)) {

                const auto cubemap = cubemaps.at(texture)->texture.get();
                mapTextureMapping(*cubemap, texture->mapping);
                return cubemap;

            } else {

                const auto& image = texture->image();

                if (image.height() > 0) {

                    const auto& currentRenderTarget = renderer.getRenderTarget();

                    auto renderTarget = std::make_unique<GLCubeRenderTarget>(image.height() / 2);
                    renderTarget->fromEquirectangularTexture(renderer, *texture);
                    cubemaps[texture] = std::move(renderTarget);

                    renderer.setRenderTarget(currentRenderTarget);

                    auto* cubemap = cubemaps[texture]->texture.get();
                    mapTextureMapping(*cubemap, texture->mapping);
                    return cubemap;

                } else {

                    return nullptr;
                }
            }
        }
    }

    return texture;
}

Texture* GLCubeMaps::getPMREM(Texture* texture) {

    if (!texture) return nullptr;

    const auto mapping = texture->mapping;
    const bool isEquirect = mapping == Mapping::EquirectangularReflection ||
                            mapping == Mapping::EquirectangularRefraction;
    if (!isEquirect) return texture;

    if (pmrems.contains(texture)) {
        return pmrems.at(texture)->texture.get();
    }

    const auto& image = texture->image();
    if (image.height() == 0) return nullptr;

    if (!pmremGenerator) {
        pmremGenerator = std::make_unique<GLPMREM>(renderer);
    }

    auto* currentRenderTarget = renderer.getRenderTarget();
    auto pmrem = pmremGenerator->fromEquirectangular(*texture);
    renderer.setRenderTarget(currentRenderTarget);

    auto* result = pmrem->texture.get();
    pmrems[texture] = std::move(pmrem);
    return result;
}

void GLCubeMaps::dispose() {

    cubemaps.clear();
    pmrems.clear();
    pmremGenerator.reset();
}
