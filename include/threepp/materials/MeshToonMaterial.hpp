// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshToonMaterial.js

#ifndef THREEPP_MESHTOONMATERIAL_HPP
#define THREEPP_MESHTOONMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

#include "threepp/materials/MaterialParams.hpp"

#include "threepp/math/Color.hpp"
#include "threepp/math/Vector2.hpp"

#include <optional>
#include <utility>

namespace threepp {

    class MeshToonMaterial: public virtual Material,
                            public MaterialWithColor,
                            public MaterialWithMap,
                            public MaterialWithAlphaMap,
                            public MaterialWithGradientMap,
                            public MaterialWithBumpMap,
                            public MaterialWithNormalMap,
                            public MaterialWithLightMap,
                            public MaterialWithAoMap,
                            public MaterialWithDisplacementMap,
                            public MaterialWithEmissive,
                            public MaterialWithWireframe,
                            public MaterialWithDefines {

    public:
        // Typed, order-free, compiler-checked alternative to the stringly-typed
        // create({{"key", value}}) map. A typo or wrong type is a COMPILE error (not a runtime
        // "Unused material values" line or std::bad_variant_access), every field autocompletes,
        // and -- unlike C++20 designated initializers -- the fluent setters chain in ANY order,
        // matching three.js's order-free object literal:
        //
        //     auto m = MeshToonMaterial::create(
        //         MeshToonMaterial::Params{}
        //             .wireframe(true)   // order-free
        //             .color(0x00ff00));
        //
        // Only the fields you set are applied; the rest keep their constructor defaults.
        class Params : public MaterialParams<Params> {
        public:
#define TPP_PARAM(type, field)  \
    Params& field(type v) {     \
        field##_ = std::move(v);\
        return *this;           \
    }
            TPP_PARAM(Color, color)
            TPP_PARAM(std::shared_ptr<Texture>, map)
            TPP_PARAM(std::shared_ptr<Texture>, alphaMap)
            TPP_PARAM(std::shared_ptr<Texture>, gradientMap)
            TPP_PARAM(std::shared_ptr<Texture>, bumpMap)
            TPP_PARAM(float, bumpScale)
            TPP_PARAM(std::shared_ptr<Texture>, normalMap)
            TPP_PARAM(NormalMapType, normalMapType)
            TPP_PARAM(Vector2, normalScale)
            TPP_PARAM(std::shared_ptr<Texture>, lightMap)
            TPP_PARAM(float, lightMapIntensity)
            TPP_PARAM(std::shared_ptr<Texture>, aoMap)
            TPP_PARAM(float, aoMapIntensity)
            TPP_PARAM(std::shared_ptr<Texture>, displacementMap)
            TPP_PARAM(float, displacementScale)
            TPP_PARAM(float, displacementBias)
            TPP_PARAM(Color, emissive)
            TPP_PARAM(float, emissiveIntensity)
            TPP_PARAM(std::shared_ptr<Texture>, emissiveMap)
            TPP_PARAM(bool, wireframe)
            TPP_PARAM(float, wireframeLinewidth)
#undef TPP_PARAM

        private:
            friend class MeshToonMaterial;

            std::optional<Color> color_;
            std::shared_ptr<Texture> map_;
            std::shared_ptr<Texture> alphaMap_;
            std::shared_ptr<Texture> gradientMap_;
            std::shared_ptr<Texture> bumpMap_;
            std::optional<float> bumpScale_;
            std::shared_ptr<Texture> normalMap_;
            std::optional<NormalMapType> normalMapType_;
            std::optional<Vector2> normalScale_;
            std::shared_ptr<Texture> lightMap_;
            std::optional<float> lightMapIntensity_;
            std::shared_ptr<Texture> aoMap_;
            std::optional<float> aoMapIntensity_;
            std::shared_ptr<Texture> displacementMap_;
            std::optional<float> displacementScale_;
            std::optional<float> displacementBias_;
            std::optional<Color> emissive_;
            std::optional<float> emissiveIntensity_;
            std::shared_ptr<Texture> emissiveMap_;
            std::optional<bool> wireframe_;
            std::optional<float> wireframeLinewidth_;
        };

        [[nodiscard]] std::string type() const override {

            return "MeshToonMaterial";
        }

        void copyInto(Material& material) const override {

            Material::copyInto(material);

            auto m = material.as<MeshToonMaterial>();

            m->color.copy(color);

            m->map = map;
            m->gradientMap = gradientMap;

            m->lightMap = lightMap;
            m->lightMapIntensity = lightMapIntensity;

            m->aoMap = aoMap;
            m->aoMapIntensity = aoMapIntensity;

            m->emissive.copy(emissive);
            m->emissiveMap = emissiveMap;
            m->emissiveIntensity = emissiveIntensity;

            m->bumpMap = bumpMap;
            m->bumpScale = bumpScale;

            m->normalMap = normalMap;
            m->normalMapType = normalMapType;
            m->normalScale.copy(normalScale);

            m->displacementMap = displacementMap;
            m->displacementScale = displacementScale;
            m->displacementBias = displacementBias;

            m->alphaMap = alphaMap;

            m->wireframe = wireframe;
            m->wireframeLinewidth = wireframeLinewidth;
        }

        static std::shared_ptr<MeshToonMaterial> create() {

            return std::shared_ptr<MeshToonMaterial>(new MeshToonMaterial());
        }

        static std::shared_ptr<MeshToonMaterial> create(const Params& p) {

            auto m = std::shared_ptr<MeshToonMaterial>(new MeshToonMaterial());
            p.applyBaseTo(*m);
#define TPP_SET(field) if (p.field##_) m->field = *p.field##_;
#define TPP_TEX(field) if (p.field##_) m->field = p.field##_;
            TPP_SET(color)
            TPP_TEX(map)
            TPP_TEX(alphaMap)
            TPP_TEX(gradientMap)
            TPP_TEX(bumpMap)
            TPP_SET(bumpScale)
            TPP_TEX(normalMap)
            TPP_SET(normalMapType)
            TPP_SET(normalScale)
            TPP_TEX(lightMap)
            TPP_SET(lightMapIntensity)
            TPP_TEX(aoMap)
            TPP_SET(aoMapIntensity)
            TPP_TEX(displacementMap)
            TPP_SET(displacementScale)
            TPP_SET(displacementBias)
            TPP_SET(emissive)
            TPP_SET(emissiveIntensity)
            TPP_TEX(emissiveMap)
            TPP_SET(wireframe)
            TPP_SET(wireframeLinewidth)
#undef TPP_SET
#undef TPP_TEX
            return m;
        }

    protected:
        MeshToonMaterial()
            : MaterialWithColor(0xffffff),
              MaterialWithAoMap(1),
              MaterialWithBumpMap(1),
              MaterialWithLightMap(1),
              MaterialWithDisplacementMap(1, 0),
              MaterialWithEmissive(0x000000, 1),
              MaterialWithWireframe(false, 1),
              MaterialWithNormalMap(NormalMapType::TangentSpace, {1, 1}) {

            this->defines["TOON"] = "";
        }

        std::shared_ptr<Material> createDefault() const override {

            return std::shared_ptr<MeshToonMaterial>(new MeshToonMaterial());
        }
    };

}// namespace threepp

#endif//THREEPP_MESHTOONMATERIAL_HPP
