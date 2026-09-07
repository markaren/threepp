
#ifndef THREEPP_EDITOR_SHADOWFIT_HPP
#define THREEPP_EDITOR_SHADOWFIT_HPP

namespace threepp {

    class Box3;
    class DirectionalLight;
    class Object3D;

    namespace editor {

        // A one-shot fit of a directional light's shadow camera to the scene.
        //
        // DirectionalLightShadow defaults to Ortho(-5,5,5,-5, 0.5, 500), which
        // three.js expects the author to replace. That default is why shadows
        // land in the first 2% of the depth range on a scene a few units
        // across, and why anything past 5 units of the origin gets none.
        //
        // Deliberately something the user asks for rather than something that
        // happens every frame. No fit is right for every scene — a 1000x1000
        // ground plane fits to an extent of 700 and spreads the map so thin
        // that anything standing on it casts a shadow two texels wide — so this
        // is a starting point the inspector's fields then own, not a policy.
        struct ShadowFit {

            // Fit one light to `bounds`. Returns whether anything moved.
            // A light with no shadow camera, or empty bounds, is left alone.
            static bool fit(DirectionalLight& light, const Box3& bounds);

            // Bounds of everything under `scene` that casts or receives a
            // shadow. Empty if nothing does.
            static Box3 shadowBounds(Object3D& scene);

            // Convenience: fit every shadow-casting directional light under
            // `scene` to shadowBounds(scene). Returns how many changed.
            static int fitAll(Object3D& scene);
        };

    }// namespace editor

}// namespace threepp

#endif//THREEPP_EDITOR_SHADOWFIT_HPP
