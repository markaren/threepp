// A planar mirror of the scene for a PBR material (MaterialWithPlanarReflection).
// The mirrored-camera and oblique-near-plane construction is objects/Reflector's
// (three.js r129 Reflector.js, Lengyel's oblique frustum); what differs is what
// it is for: Reflector is a mesh with its own shader, this renders into a
// target that a MeshPhysicalMaterial reads as its specular environment where
// the mirror holds something, keeping the environment map elsewhere.

#ifndef THREEPP_PLANARREFLECTION_HPP
#define THREEPP_PLANARREFLECTION_HPP

#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"

#include <memory>
#include <vector>

namespace threepp {

    class Camera;
    class Object3D;
    class Renderer;
    class Texture;
    struct MaterialWithPlanarReflection;

    class PlanarReflection {

    public:
        struct Options {

            // The target's size as a share of the renderer's frame. Half the
            // frame reads as sharp as the water's own sampling allows and costs
            // a quarter of a full-size pass.
            float resolutionScale = 0.5f;

            // The mirror's reach, metres: objects whose bounding sphere lies
            // wholly beyond it from the mirrored camera are left out of the
            // pass, and the camera's far plane closes there. The near hulls,
            // quays and trees cost a second pass; the town behind them stays
            // in the environment map. An object that straddles the limit (the
            // terrain) is drawn, and the oblique near plane skews the far
            // plane, so the cut through such an object is not a clean one.
            // 0: no limit.
            float farDistance = 400.f;

            // Pushes the oblique near plane off the mirror plane, as Reflector's
            // clipBias: a little hides seams where geometry meets the water.
            float clipBias = 0.f;

            // Draw scene.background into the mirror. Off, the target keeps
            // alpha 0 where nothing was drawn and the material's environment
            // map shows there, which is the point when the sky is already in it.
            bool background = false;
        };

        // Two overloads rather than `const Options& = Options()`: Clang needs
        // the nested struct's default member initializers to evaluate that
        // default argument while the enclosing class is still incomplete.
        PlanarReflection();
        explicit PlanarReflection(const Options& options);

        PlanarReflection(const PlanarReflection&) = delete;
        PlanarReflection& operator=(const PlanarReflection&) = delete;

        ~PlanarReflection();

        // The mirror plane: a point on it and its unit normal (world space).
        // Default: y = 0, normal +y, the usual water plane.
        void setPlane(const Vector3& point, const Vector3& normal);
        [[nodiscard]] const Vector3& planePoint() const;
        [[nodiscard]] const Vector3& planeNormal() const;

        // Objects set invisible for the mirrored render: the water mesh that
        // reads the mirror, anything else that should not stand in it.
        std::vector<Object3D*> hidden;

        Options options;

        // Render `scene` mirrored about the plane, as seen from `camera`, into
        // the target. Returns false (and renders nothing) when the camera is
        // behind the plane. Restores the renderer's render target, clear colour
        // and shadow-map auto-update, and the hidden objects' visibility.
        bool render(Renderer& renderer, Object3D& scene, Camera& camera);

        // The mirror: scene-linear RGBA, alpha 0 where nothing was drawn.
        [[nodiscard]] std::shared_ptr<Texture> texture() const;

        // World position -> the mirror's texture coordinates (bias * projection
        // * view of the mirrored camera), as of the last render.
        [[nodiscard]] const Matrix4& textureMatrix() const;

        // How many objects the far limit left out of the last render.
        [[nodiscard]] size_t culledByFar() const;

        // Hand the material the texture and the matrix (call after render()).
        // The first time, and whenever the texture is taken away again, the
        // material needs material.needsUpdate(): the GL program is keyed on
        // the texture being set, as it is for any other map.
        void apply(MaterialWithPlanarReflection& material) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_PLANARREFLECTION_HPP
