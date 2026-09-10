
#include "GLCubeMaps.hpp"

#include "GLPMREM.hpp"

#include "threepp/renderers/GLCubeRenderTarget.hpp"
#include "threepp/textures/Texture.hpp"

#include <utility>
#include <vector>

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
        return pmrems.at(texture).target->texture.get();
    }

    const auto& image = texture->image();
    if (image.height() == 0) return nullptr;

    if (!pmremGenerator) {
        pmremGenerator = std::make_unique<GLPMREM>(renderer);
    }

    // ONE-SUN POLICY (shared with the Vulkan deferred path). Detect the HDRI's
    // sun disc and prefilter strips 1+ from a copy with the disc replaced by its
    // surround; GLRenderer re-injects the removed energy as a single analytic
    // directional light. Without this, every diffuse surface is lit by the baked
    // sun PLUS the scene's own DirectionalLight — the brighter, yellow-shifted,
    // flat-shadowed GL look. Float RGBA only: that is what the HDR loaders emit
    // and the only layout the detector reads (and all Vulkan accepts too).
    EnvSunExtract sun;
    std::shared_ptr<Texture> sunClamped;
    if (envSunExtraction && image.isFloat() && image.channels() == 4) {
        std::vector<float> clamped;
        if (extractEnvSun(image.width(), image.height(),
                          image.data<float>().data(), clamped, sun)) {
            std::vector<Image> images;
            images.emplace_back(std::move(clamped), image.width(), image.height(), 0u);
            sunClamped = Texture::create(std::move(images));
            sunClamped->name = texture->name + ".sunClamped";
            sunClamped->format = texture->format;
            sunClamped->type = texture->type;
            sunClamped->colorSpace = texture->colorSpace;
            sunClamped->mapping = texture->mapping;
            sunClamped->wrapS = texture->wrapS;
            sunClamped->wrapT = texture->wrapT;
            sunClamped->minFilter = texture->minFilter;
            sunClamped->magFilter = texture->magFilter;
            // The prefilter shader reads the source through textureLod, so the
            // clamped copy needs the same mip chain the original carries.
            sunClamped->generateMipmaps = texture->generateMipmaps;
            sunClamped->anisotropy = texture->anisotropy;
            sunClamped->needsUpdate();
        }
    }

    auto* currentRenderTarget = renderer.getRenderTarget();
    auto pmrem = pmremGenerator->fromEquirectangular(*texture, sunClamped.get());
    renderer.setRenderTarget(currentRenderTarget);

    auto* result = pmrem->texture.get();
    pmrems[texture] = PmremEntry{std::move(pmrem), sun};
    // The clamped copy is prefilter scratch — a second full-res RGBA32F upload
    // for the life of the env otherwise. Letting it die here disposes the GL
    // texture (Texture's destructor dispatches "dispose").
    return result;
}

const EnvSunExtract* GLCubeMaps::envSun(Texture* texture) const {

    if (!texture) return nullptr;
    const auto it = pmrems.find(texture);
    if (it == pmrems.end() || !it->second.sun.found) return nullptr;
    return &it->second.sun;
}

void GLCubeMaps::disposePMREMs() {

    pmrems.clear();
}

void GLCubeMaps::dispose() {

    cubemaps.clear();
    pmrems.clear();
    pmremGenerator.reset();
}
