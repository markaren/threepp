// https://github.com/mrdoob/three.js/blob/r129/src/lights/RectAreaLight.js

#ifndef THREEPP_RECTAREALIGHT_HPP
#define THREEPP_RECTAREALIGHT_HPP

#include "threepp/lights/Light.hpp"

#include <memory>

namespace threepp {

    class Mesh;

    /// Rectangular area light. Emits uniformly from one face of a rectangle.
    /// Local axes: the rectangle lies in the XY plane; emissive face faces -Z.
    /// Integration model: auto-creates a child Mesh (PlaneGeometry +
    /// MeshStandardMaterial with emissive=color, emissiveIntensity=intensity).
    /// The path tracer picks it up through the regular emissive-triangle path
    /// (emissive-tri CDF + NEE). Mutating color/intensity after construction
    /// requires calling WgpuPathTracer::markDirty() to rebuild the emissive CDF.
    class RectAreaLight: public Light {

    public:
        const float width;
        const float height;

        [[nodiscard]] std::string type() const override;

        /// Total luminous power emitted by the rectangle (Lambertian):
        ///   power = intensity * PI * width * height
        [[nodiscard]] float getPower() const;

        /// Inverse of getPower(). Sets intensity from a power target.
        void setPower(float power);

        /// Child mesh representing the emissive quad. Exposed so users can
        /// configure visibility, layers, or material overrides. Mutations
        /// to the material require WgpuPathTracer::markDirty().
        [[nodiscard]] const std::shared_ptr<Mesh>& mesh() const;

        void updateMatrixWorld(bool force = false) override;

        void copy(const Object3D& source, bool recursive) override;

        static std::shared_ptr<RectAreaLight> create(
                const Color& color = 0xffffff,
                std::optional<float> intensity = std::nullopt,
                float width = 1.0f,
                float height = 1.0f);

    protected:
        RectAreaLight(const Color& color, std::optional<float> intensity, float width, float height);

    private:
        std::shared_ptr<Mesh> mesh_;
    };

}// namespace threepp

#endif//THREEPP_RECTAREALIGHT_HPP
