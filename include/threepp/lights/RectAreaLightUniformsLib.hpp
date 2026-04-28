
#ifndef THREEPP_RECTAREALIGHTUNIFORMSLIB_HPP
#define THREEPP_RECTAREALIGHTUNIFORMSLIB_HPP

#include <memory>

namespace threepp {

    class Texture;

    /// Provides the two Linearly Transformed Cosines lookup textures used by
    /// the RectAreaLight BRDF approximation in the GL renderer. Port of
    /// three.js examples/jsm/lights/RectAreaLightUniformsLib.
    ///
    /// Call init() once before rendering a scene that contains RectAreaLights.
    /// The textures are built lazily the first time init() is invoked.
    class RectAreaLightUniformsLib {

    public:
        static RectAreaLightUniformsLib& instance();

        /// Build the LTC lookup textures if they haven't been built yet.
        void init();

        /// 64x64 RGBA float texture holding the inverse BRDF transform.
        [[nodiscard]] std::shared_ptr<Texture> ltc_1() const;

        /// 64x64 RGBA float texture holding the magnitude / Fresnel table.
        [[nodiscard]] std::shared_ptr<Texture> ltc_2() const;

    private:
        RectAreaLightUniformsLib() = default;

        std::shared_ptr<Texture> ltc1_;
        std::shared_ptr<Texture> ltc2_;
    };

}// namespace threepp

#endif//THREEPP_RECTAREALIGHTUNIFORMSLIB_HPP
