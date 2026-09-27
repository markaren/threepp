// https://github.com/mrdoob/three.js/blob/r129/src/scenes/Scene.js

#ifndef THREEPP_SCENE_HPP
#define THREEPP_SCENE_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Euler.hpp"

#include "threepp/scenes/Fog.hpp"
#include "threepp/scenes/FogExp2.hpp"

#include <memory>
#include <variant>

namespace threepp {

    class Texture;
    class CubeTexture;
    typedef std::variant<Fog, FogExp2> FogVariant;

    class Background {

    public:
        Background();
        Background(int color);
        Background(const Color& color);
        Background(const std::shared_ptr<Texture>& texture);
        Background(const std::shared_ptr<CubeTexture>& texture);

        [[nodiscard]] bool isColor() const;

        [[nodiscard]] bool isTexture() const;

        [[nodiscard]] Color& color();

        [[nodiscard]] std::shared_ptr<Texture> texture() const;

        [[nodiscard]] bool empty() const;

    private:
        bool hasValue_{false};
        std::optional<Color> color_;
        std::shared_ptr<Texture> texture_;
    };

    class Scene: public Object3D {

    public:
        Background background;
        std::shared_ptr<Texture> environment;
        std::optional<FogVariant> fog;

        // three.js r146-r163 background/environment controls (same names, same
        // defaults; the defaults reproduce the renderer's output from before
        // they existed). Honoured by GLRenderer only for now: the Vulkan
        // renderer ignores all five.
        //
        // Blur of a texture background, 0 (sharp) .. 1 (fully diffuse). Blurred
        // backgrounds are read from the environment's prefiltered PMREM, so they
        // need an equirectangular or cube texture background.
        float backgroundBlurriness = 0;
        // Linear multiplier on a texture background (not on a Color background).
        float backgroundIntensity = 1;
        // Rotation of a texture background (cube / equirect).
        Euler backgroundRotation;
        // Multiplier on the IBL of every material lit by `environment` (one that
        // has no envMap of its own). threepp multiplies it with the material's
        // envMapIntensity; see GLRenderer.
        float environmentIntensity = 1;
        // Rotation of `environment`, for every material lit by it.
        Euler environmentRotation;

        std::shared_ptr<Material> overrideMaterial;

        bool autoUpdate = true;

        [[nodiscard]] std::string type() const override;

        // Background, environment, fog and override material are shared with
        // the source, not cloned.
        void copy(const Object3D& source, bool recursive = true) override;

        static std::shared_ptr<Scene> create();

    protected:
        std::shared_ptr<Object3D> createDefault() override;
    };

}// namespace threepp

#endif//THREEPP_SCENE_HPP
