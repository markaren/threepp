// window_util.hpp — the windowing boilerplate every interactive demo repeats.
//
// Right now that is one thing: the resize handler. Resize the renderer, re-derive
// the camera aspect, rebuild the projection. Three lines — copy-pasted verbatim
// into 19 of the Vulkan demos alone, varying only on cosmetics:
//
//   • whether the aspect came from the event or from canvas.aspect(). These are
//     identical: Canvas stores its new size before running the listeners. The
//     event size is used here so the handler needs no reference to the canvas.
//   • whether the renderer and camera were held by value or by pointer. Both
//     spellings are accepted, so a demo does not have to reshape its locals.
//
// Header-only, and on every example target's include path (examples/libs), so
// any demo — GL or Vulkan — can include it with no build-system changes.
#pragma once

#include "threepp/canvas/Canvas.hpp"
#include "threepp/canvas/WindowSize.hpp"

#include <type_traits>

namespace demo {

    namespace detail {

        // Value, raw pointer or smart pointer -> reference to the object.
        template<class T>
        decltype(auto) target(T& v) {
            if constexpr (requires { *v; }) return *v;
            else return (v);
        }

    }// namespace detail

    // The body of the standard handler. Call this from a hand-written listener
    // when a demo has extra work to do on resize (re-arming a sensor, resetting
    // an accumulator); use bindResize below for the plain case.
    template<class Renderer, class Camera>
    void applyResize(Renderer& renderer, Camera& camera, threepp::WindowSize size) {
        using Cam = std::remove_reference_t<decltype(detail::target(camera))>;
        static_assert(requires(Cam& c) { c.aspect; c.updateProjectionMatrix(); },
                      "applyResize needs a camera with an `aspect` member and "
                      "updateProjectionMatrix() — i.e. a PerspectiveCamera. An "
                      "orthographic camera resizes by frustum bounds, not aspect, "
                      "so it keeps a hand-written listener.");
        detail::target(renderer).setSize(size);
        auto& cam = detail::target(camera);
        cam.aspect = size.aspect();
        cam.updateProjectionMatrix();
    }

    // `renderer` and `camera` are captured by reference, so both must outlive the
    // canvas — true when they are locals of main(), as in every demo.
    template<class Renderer, class Camera>
    void bindResize(threepp::Canvas& canvas, Renderer& renderer, Camera& camera) {
        canvas.onWindowResize([&renderer, &camera](threepp::WindowSize size) {
            applyResize(renderer, camera, size);
        });
    }

}// namespace demo
