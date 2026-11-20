#ifndef THREEPP_EDITOR_ORTHOCLIPFIT_HPP
#define THREEPP_EDITOR_ORTHOCLIPFIT_HPP

#include "threepp/math/Vector3.hpp"

namespace threepp {

    class Box3;
    class OrthographicCamera;

    namespace editor {

        // The far plane of an orthographic viewport camera, fitted to what it
        // has to show.
        //
        // A parallel projection spends its depth buffer evenly between near and
        // far: with ten kilometres between them a 24-bit buffer steps by 0.6 mm,
        // and a 32-bit float one does no better, since reverse Z leaves a
        // parallel camera's depths just under 1. Anything thinner than a step
        // lies in the same depth as what it rests on. The print on a circuit
        // board, a label, a decal: looked at straight down, all of it ties with
        // the surface below and which of the two is drawn is an accident of
        // order. A perspective camera never shows this, because its depth is
        // dense where it stands.
        //
        // So the range is kept as short as the scene allows, every frame, and
        // the step shrinks with it: a few metres of range step by a fraction of
        // a micrometre.
        struct OrthoClipFit {

            // Something flat the view only ever shows a part of, and which
            // should not by itself stretch the range to its full extent: the
            // editor's grid. A disc.
            struct Sheet {
                Vector3 centre;
                Vector3 normal{0, 1, 0};
                float radius = 0.f;
            };

            // The far plane for `camera` where it stands: behind all of
            // `bounds` (which may be empty) and behind as much of `sheet` as the
            // frustum can see, with room for what an editor draws around a
            // subject at a size set by the view. The near plane is left where
            // it is.
            //
            // The answer moves in steps (half octaves of the range), so the
            // projection holds still through small changes: a renderer that
            // carries history from frame to frame reads a new projection as a
            // new view.
            [[nodiscard]] static float farPlane(const OrthographicCamera& camera, const Box3& bounds,
                                                const Sheet* sheet = nullptr);
        };

    }// namespace editor

}// namespace threepp

#endif//THREEPP_EDITOR_ORTHOCLIPFIT_HPP
