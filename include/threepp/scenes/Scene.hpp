// https://github.com/mrdoob/three.js/blob/r129/src/scenes/Scene.js

#ifndef THREEPP_SCENE_HPP
#define THREEPP_SCENE_HPP

#include "threepp/core/Object3D.hpp"

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

        std::shared_ptr<Material> overrideMaterial;

        bool autoUpdate = true;

        static std::shared_ptr<Scene> create();
    };

}// namespace threepp

#endif//THREEPP_SCENE_HPP
