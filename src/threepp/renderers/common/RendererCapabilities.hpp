// Backend-neutral renderer capabilities.
// Each backend populates this from its own capability queries.

#ifndef THREEPP_COMMON_RENDERERCAPABILITIES_HPP
#define THREEPP_COMMON_RENDERERCAPABILITIES_HPP

namespace threepp {

    struct RendererCapabilities {

        bool vertexTextures = true;
        bool floatVertexTextures = true;
        bool logarithmicDepthBuffer = false;

        int maxVertexUniforms = 0;
        int maxTextures = 0;
        int maxVertexTextures = 0;
        int maxTextureSize = 0;
        int maxCubemapSize = 0;
        int maxAttributes = 0;
        int maxSamples = 0;
    };

}// namespace threepp

#endif//THREEPP_COMMON_RENDERERCAPABILITIES_HPP
