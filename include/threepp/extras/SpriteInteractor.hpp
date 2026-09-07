// Mouse-event dispatcher for screen-space sprites.
//
// Attaches itself to a Canvas (or any PeripheralsEventSource) on
// construction and walks the scene on each mouse event, hit-testing
// the configured screen-space sprites by their pixel-space AABB and
// firing the per-sprite onMouseDown/onMouseUp callbacks.
//
// Usage:
//   auto button = TextSprite::create(font, 30);
//   button->setText("Click me");
//   button->screenSpace  = true;
//   button->screenAnchor.set(0.5f, 0.5f);
//   button->position.set(0, 0, 0);
//   button->center.set(0.5f, 0.5f);
//   button->onMouseDown = [](int) { std::cout << "clicked\n"; };
//   scene->add(button);
//
//   SpriteInteractor interactor(canvas, *scene);   // lifetime owns the
//                                                  // event-listener hookup.
//
// Only screen-space Sprites are considered. 3D sprites (screenSpace=false)
// are ignored — pick those via Raycaster instead.
//
// Hit-test convention: the sprite occupies the pixel AABB
//   left   = pxCenter.x - center.x * scale.x
//   right  = left + scale.x
//   bottom = pxCenter.y - center.y * scale.y
//   top    = bottom + scale.y
// where pxCenter = screenAnchor·viewport + position.xy. Mouse Y is
// flipped from top-down (GLFW convention) to bottom-up so the test is
// in the same coord frame the renderer uses for screen-space sprites.
//
// Multiple sprites overlapping the cursor: the LAST one visited by
// scene.traverseVisible wins, matching "drawn last = on top" semantics.

#ifndef THREEPP_SPRITEINTERACTOR_HPP
#define THREEPP_SPRITEINTERACTOR_HPP

#include "threepp/canvas/Canvas.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/input/MouseListener.hpp"
#include "threepp/objects/Sprite.hpp"

namespace threepp {

    class SpriteInteractor: public MouseListener {

    public:
        SpriteInteractor(Canvas& canvas, Object3D& scene)
            : canvas_(canvas), scene_(scene) {
            canvas_.addMouseListener(*this);
        }

        ~SpriteInteractor() override {
            canvas_.removeMouseListener(*this);
        }

        SpriteInteractor(const SpriteInteractor&) = delete;
        SpriteInteractor& operator=(const SpriteInteractor&) = delete;

        void onMouseDown(int button, const Vector2& pos) override {
            dispatch(pos, button, /*down=*/true);
        }

        void onMouseUp(int button, const Vector2& pos) override {
            dispatch(pos, button, /*down=*/false);
        }

    private:
        Canvas&    canvas_;
        Object3D&  scene_;

        void dispatch(const Vector2& mp, int button, bool down) {
            const auto size = canvas_.size();
            const float w = static_cast<float>(size.width());
            const float h = static_cast<float>(size.height());
            if (w <= 0.f || h <= 0.f) return;

            // GLFW mouse origin is top-left; sprite pixel space (matching
            // the renderer's internal ortho cam) is bottom-up — flip Y.
            const float my = h - mp.y;

            Sprite* hit = nullptr;
            scene_.traverseVisible([&](Object3D& o) {
                auto* sp = o.as<Sprite>();
                if (!sp || !sp->screenSpace) return;
                if (!sp->onMouseDown && !sp->onMouseUp) return;

                const float pxX = sp->screenAnchor.x * w + static_cast<float>(sp->position.x);
                const float pxY = sp->screenAnchor.y * h + static_cast<float>(sp->position.y);
                const float sw = static_cast<float>(sp->scale.x);
                const float sh = static_cast<float>(sp->scale.y);
                const float left   = pxX - sp->center.x * sw;
                const float right  = left + sw;
                const float bottom = pxY - sp->center.y * sh;
                const float top    = bottom + sh;
                if (mp.x >= left && mp.x <= right && my >= bottom && my <= top) {
                    hit = sp;// last match wins (drawn last == visually on top)
                }
            });

            if (!hit) return;
            if (down && hit->onMouseDown) hit->onMouseDown(button);
            if (!down && hit->onMouseUp)   hit->onMouseUp(button);
        }
    };

}// namespace threepp

#endif//THREEPP_SPRITEINTERACTOR_HPP
