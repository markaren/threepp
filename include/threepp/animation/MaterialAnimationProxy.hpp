
#ifndef THREEPP_MATERIALANIMATIONPROXY_HPP
#define THREEPP_MATERIALANIMATIONPROXY_HPP

#include "threepp/core/Object3D.hpp"

#include <memory>

namespace threepp {

    class Material;

    // Invisible scene-graph node that carries a shared Material pointer so
    // PropertyBinding can route animation tracks (e.g. glTF KHR_animation_pointer)
    // at material fields — color, opacity, emissive, metalness, roughness, etc.
    class MaterialAnimationProxy: public Object3D {

    public:
        std::shared_ptr<Material> targetMaterial;

        static std::shared_ptr<MaterialAnimationProxy> create(std::shared_ptr<Material> material) {
            auto p = std::shared_ptr<MaterialAnimationProxy>(new MaterialAnimationProxy());
            p->targetMaterial = std::move(material);
            p->visible = false;
            return p;
        }

    private:
        MaterialAnimationProxy() = default;
    };

}// namespace threepp

#endif//THREEPP_MATERIALANIMATIONPROXY_HPP
