// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshBasicMaterial.js

#ifndef THREEPP_MESHBASICMATERIAL_HPP
#define THREEPP_MESHBASICMATERIAL_HPP

#include "threepp/materials/Material.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/textures/Texture.hpp"

#include <optional>

namespace threepp {

    class MeshBasicMaterial : public Material {

    public:
        Color color = Color(0xffffff);

        std::optional<Texture> map = std::nullopt;

        std::optional<Texture> lightMap = std::nullopt;
        float lightMapIntensity = 1.0;

        std::optional<Texture> aoMap = std::nullopt;
        float aoMapIntensity = 1.0;

        std::optional<Texture> specularMap = std::nullopt;

        std::optional<Texture> alphaMap = std::nullopt;

        std::optional<Texture> envMap = std::nullopt;
        int combine = MultiplyOperation;
        float reflectivity = 1;
        float refractionRatio = 0.98f;

        std::string wireframeLinecap = "round";
        std::string wireframeLinejoin = "round";

        bool wireframe = false;
        float wireframeLinewidth = 1;


    protected:

        MeshBasicMaterial() = default;

        static std::shared_ptr<MeshBasicMaterial> create() {
            return std::shared_ptr<MeshBasicMaterial>(new MeshBasicMaterial());
        }
    };

}// namespace threepp

#endif//THREEPP_MESHBASICMATERIAL_HPP
