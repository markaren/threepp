// https://github.com/mrdoob/three.js/blob/r129/src/materials/LineDashedMaterial.js

#ifndef THREEPP_LINEDASHEDMATERIAL_HPP
#define THREEPP_LINEDASHEDMATERIAL_HPP

#include "threepp/materials/LineBasicMaterial.hpp"

namespace threepp {

    class LineDashedMaterial: public LineBasicMaterial {

    public:
        float dashSize = 3;
        float gapSize = 1;
        float scale = 1;

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<LineDashedMaterial> create(const std::unordered_map<std::string, MaterialValue>& values = {});

    protected:
        LineDashedMaterial();

        std::shared_ptr<Material> createDefault() const override;

        void copyInto(Material& material) const override;

        bool setValue(const std::string& key, const MaterialValue& value) override;
    };

}// namespace threepp

#endif//THREEPP_LINEDASHEDMATERIAL_HPP
