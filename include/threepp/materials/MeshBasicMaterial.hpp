// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshBasicMaterial.js

#ifndef THREEPP_MESHBASICMATERIAL_HPP
#define THREEPP_MESHBASICMATERIAL_HPP

#include "threepp/materials/Material.hpp"
#include "threepp/materials/interfaces.hpp"

#include "threepp/math/Color.hpp"
#include "threepp/textures/Texture.hpp"

#include <optional>

namespace threepp {

    class MeshBasicMaterial
        : public virtual Material,
          public MaterialWithColor,
          public MaterialWithMap,
          public MaterialWithLightMap,
          public MaterialWithAoMap,
          public MaterialWithSpecularMap,
          public MaterialWithAlphaMap,
          public MaterialWithEnvMap,
          public MaterialWithReflectivity,
          public MaterialWithWireframe,
          public MaterialWithCombine {

    public:

        [[nodiscard]] std::string type() const override {

            return "MeshBasicMaterial";
        }

        [[nodiscard]] std::shared_ptr<MeshBasicMaterial> clone() const {

            auto m =  std::shared_ptr<MeshBasicMaterial>(new MeshBasicMaterial());
            copyInto(m.get());
            
            m->color.copy( color );
            
            m->map = map;

            m->lightMap = lightMap;
            m->lightMapIntensity = lightMapIntensity;

            m->aoMap = aoMap;
            m->aoMapIntensity = aoMapIntensity;

            m->specularMap = specularMap;

            m->alphaMap = alphaMap;

            m->envMap = envMap;
            m->combine = combine;
            m->reflectivity = reflectivity;
            m->refractionRatio = refractionRatio;

            m->wireframe = wireframe;
            m->wireframeLinewidth = wireframeLinewidth;

            return m;
        }

        static std::shared_ptr<MeshBasicMaterial> create() {

            return std::shared_ptr<MeshBasicMaterial>(new MeshBasicMaterial());
        }

    protected:
        MeshBasicMaterial()
            : MaterialWithColor(0xffffff),
              MaterialWithAoMap(1),
              MaterialWithLightMap(1),
              MaterialWithCombine(MultiplyOperation),
              MaterialWithReflectivity(1, 0.98f),
              MaterialWithWireframe(false, 1) {}
    };

}// namespace threepp

#endif//THREEPP_MESHBASICMATERIAL_HPP
