// uniform_environment.hpp — an environment of one colour from every direction, for scenes
// whose sky is a plain colour.
//
// In three.js (and threepp, on GL and Vulkan alike) scene.background only draws: it lights
// nothing and nothing reflects it. scene.environment is the light from everywhere, which
// matte surfaces are lit by and glossy ones reflect. A scene whose plain-colour sky should
// do both says so by setting both:
//
//   scene.background  = sky;
//   scene.environment = uniformEnvironment(sky);
//
// The texture is a small float equirect (VulkanRenderer takes only float HDR environments;
// GLRenderer prefilters it like any other).

#ifndef THREEPP_EXAMPLES_UNIFORM_ENVIRONMENT_HPP
#define THREEPP_EXAMPLES_UNIFORM_ENVIRONMENT_HPP

#include "threepp/math/Color.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <memory>
#include <vector>

namespace threepp {

    inline std::shared_ptr<DataTexture> uniformEnvironment(const Color& c, float intensity = 1.f) {
        constexpr int W = 8, H = 4;
        std::vector<float> px(W * H * 4);
        for (size_t i = 0; i < px.size(); i += 4) {
            px[i] = c.r * intensity;
            px[i + 1] = c.g * intensity;
            px[i + 2] = c.b * intensity;
            px[i + 3] = 1.f;
        }
        auto env = DataTexture::create(ImageData(std::move(px)), W, H);
        env->format = Format::RGBA;
        env->type = Type::Float;
        env->mapping = Mapping::EquirectangularReflection;
        env->needsUpdate();
        return env;
    }

}// namespace threepp

#endif//THREEPP_EXAMPLES_UNIFORM_ENVIRONMENT_HPP
