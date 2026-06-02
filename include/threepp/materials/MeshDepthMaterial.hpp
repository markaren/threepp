// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshDepthMaterial.js

#ifndef THREEPP_MESHDEPTHMATERIAL_HPP
#define THREEPP_MESHDEPTHMATERIAL_HPP

#include "Material.hpp"
#include "interfaces.hpp"

#include <optional>
#include <utility>

namespace threepp {

    class MeshDepthMaterial
        : public virtual Material,
          public MaterialWithMap,
          public MaterialWithAlphaMap,
          public MaterialWithDisplacementMap,
          public MaterialWithWireframe,
          public MaterialWithDepthPacking {

    public:
        // Typed, order-free, compiler-checked alternative to the stringly-typed
        // create({{"key", value}}) map. A typo or wrong type is a COMPILE error (not a runtime
        // "Unused material values" line or std::bad_variant_access), every field autocompletes,
        // and -- unlike C++20 designated initializers -- the fluent setters chain in ANY order,
        // matching three.js's order-free object literal.
        //
        // Only the fields you set are applied; the rest keep their constructor defaults.
        class Params : public MaterialParams<Params> {
        public:
#define TPP_PARAM(type, field)  \
    Params& field(type v) {     \
        field##_ = std::move(v);\
        return *this;           \
    }
            TPP_PARAM(std::shared_ptr<Texture>, map)
            TPP_PARAM(std::shared_ptr<Texture>, alphaMap)
            TPP_PARAM(std::shared_ptr<Texture>, displacementMap)
            TPP_PARAM(float, displacementScale)
            TPP_PARAM(float, displacementBias)
            TPP_PARAM(bool, wireframe)
            TPP_PARAM(float, wireframeLinewidth)
            TPP_PARAM(DepthPacking, depthPacking)
#undef TPP_PARAM

        private:
            friend class MeshDepthMaterial;

            std::shared_ptr<Texture> map_;
            std::shared_ptr<Texture> alphaMap_;
            std::shared_ptr<Texture> displacementMap_;
            std::optional<float> displacementScale_;
            std::optional<float> displacementBias_;
            std::optional<bool> wireframe_;
            std::optional<float> wireframeLinewidth_;
            std::optional<DepthPacking> depthPacking_;
        };

        [[nodiscard]] std::string type() const override {

            return "MeshDepthMaterial";
        }

        void copyInto(Material& material) const override {

            Material::copyInto(material);

            auto m = material.as<MeshDepthMaterial>();

            m->depthPacking = depthPacking;

            m->map = map;

            m->alphaMap = alphaMap;

            m->displacementMap = displacementMap;
            m->displacementScale = displacementScale;
            m->displacementBias = displacementBias;

            m->wireframe = false;
            m->wireframeLinewidth = 1;

            m->fog = false;
        }

        static std::shared_ptr<MeshDepthMaterial> create() {

            return std::shared_ptr<MeshDepthMaterial>(new MeshDepthMaterial());
        }

        static std::shared_ptr<MeshDepthMaterial> create(const Params& p) {

            auto m = std::shared_ptr<MeshDepthMaterial>(new MeshDepthMaterial());
            p.applyBaseTo(*m);
#define TPP_SET(field) if (p.field##_) m->field = *p.field##_;
#define TPP_TEX(field) if (p.field##_) m->field = p.field##_;
            TPP_TEX(map)
            TPP_TEX(alphaMap)
            TPP_TEX(displacementMap)
            TPP_SET(displacementScale)
            TPP_SET(displacementBias)
            TPP_SET(wireframe)
            TPP_SET(wireframeLinewidth)
            TPP_SET(depthPacking)
#undef TPP_SET
#undef TPP_TEX
            return m;
        }

    protected:
        MeshDepthMaterial()
            : MaterialWithDepthPacking(DepthPacking::Basic),
              MaterialWithDisplacementMap(1, 0),
              MaterialWithWireframe(false, 1) {

            this->fog = false;
        }

        std::shared_ptr<Material> createDefault() const override {

            return std::shared_ptr<MeshDepthMaterial>(new MeshDepthMaterial());
        }
    };


}// namespace threepp

#endif//THREEPP_MESHDEPTHMATERIAL_HPP
