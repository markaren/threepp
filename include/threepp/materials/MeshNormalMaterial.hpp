// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshNormalMaterial.js

#ifndef THREEPP_MESHNORMALMATERIAL_HPP
#define THREEPP_MESHNORMALMATERIAL_HPP

#include "Material.hpp"
#include "interfaces.hpp"

#include <optional>
#include <utility>

namespace threepp {

    class MeshNormalMaterial: public virtual Material,
                              public MaterialWithBumpMap,
                              public MaterialWithNormalMap,
                              public MaterialWithDisplacementMap,
                              public MaterialWithWireframe,
                              public MaterialWithFlatShading {

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
            TPP_PARAM(bool, wireframe)
            TPP_PARAM(float, wireframeLinewidth)
            TPP_PARAM(bool, flatShading)
            TPP_PARAM(std::shared_ptr<Texture>, normalMap)
            TPP_PARAM(NormalMapType, normalMapType)
            TPP_PARAM(std::shared_ptr<Texture>, displacementMap)
            TPP_PARAM(float, displacementBias)
            TPP_PARAM(float, displacementScale)
#undef TPP_PARAM

        private:
            friend class MeshNormalMaterial;

            std::optional<bool> wireframe_;
            std::optional<float> wireframeLinewidth_;
            std::optional<bool> flatShading_;
            std::shared_ptr<Texture> normalMap_;
            std::optional<NormalMapType> normalMapType_;
            std::shared_ptr<Texture> displacementMap_;
            std::optional<float> displacementBias_;
            std::optional<float> displacementScale_;
        };

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<MeshNormalMaterial> create(const std::unordered_map<std::string, MaterialValue>& values = {});

        static std::shared_ptr<MeshNormalMaterial> create(const Params& params);

    protected:
        MeshNormalMaterial();

        std::shared_ptr<Material> createDefault() const override;

        void copyInto(Material& material) const override;

        bool setValue(const std::string& key, const MaterialValue& value) override;
    };

}// namespace threepp

#endif//THREEPP_MESHNORMALMATERIAL_HPP
