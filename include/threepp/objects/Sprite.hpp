// https://github.com/mrdoob/three.js/blob/r129/src/objects/Sprite.js

#ifndef THREEPP_SPRITE_HPP
#define THREEPP_SPRITE_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/SpriteMaterial.hpp"

#include <functional>

namespace threepp {

    class Sprite: public Object3D {

    public:
        // Pivot within the sprite quad. 0.5,0.5 = centred; 0,0 = bottom-left
        // anchor; 1,1 = top-right anchor.
        Vector2 center{0.5f, 0.5f};

        // Screen-space rendering mode. When true, the renderer composites
        // this sprite over the 3D scene using an ortho camera derived from
        // the swapchain extent — no separate scene/camera setup. Disabled
        // by default so existing 3D sprites are unaffected.
        //
        // Layout when screenSpace == true:
        //   pixelPosition = screenAnchor * viewportSize + position.xy
        // i.e. `screenAnchor` is the viewport-relative anchor (0..1, default
        // 0,0 = top-left corner) and `position.xy` is the pixel offset from
        // that anchor. Negative offsets read as "from the opposite edge",
        // matching CSS convention.
        //
        // Resize is implicit: the renderer samples the viewport size each
        // frame, so anchored sprites move with the window.
        bool    screenSpace = false;
        Vector2 screenAnchor{0.f, 0.f};

        // Optional click handlers for screen-space sprites. Fired by
        // SpriteInteractor (see threepp/extras/SpriteInteractor.hpp) when
        // a mouse event hits the sprite's pixel-AABB. Ignored when
        // screenSpace == false.
        std::function<void(int button)> onMouseDown;
        std::function<void(int button)> onMouseUp;

        explicit Sprite(const std::shared_ptr<SpriteMaterial>& material);

        [[nodiscard]] std::string type() const override;

        void raycast(const Raycaster& raycaster, std::vector<Intersection>& intersects) override;

        std::shared_ptr<BufferGeometry> geometry() const override;

        std::shared_ptr<Material> material() const override;

        void setMaterial(const std::shared_ptr<SpriteMaterial>& material);

        void copy(const Object3D& source, bool recursive = true) override;

        static std::shared_ptr<Sprite> create(const std::shared_ptr<SpriteMaterial>& material = nullptr);

    protected:
        std::shared_ptr<Object3D> createDefault() override;

    private:
        std::shared_ptr<BufferGeometry> _geometry;
        std::shared_ptr<SpriteMaterial> _material;
    };

}// namespace threepp

#endif//THREEPP_SPRITE_HPP
