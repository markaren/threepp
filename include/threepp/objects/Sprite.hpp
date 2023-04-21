// https://github.com/mrdoob/three.js/blob/r129/src/objects/Sprite.js

#ifndef THREEPP_SPRITE_HPP
#define THREEPP_SPRITE_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/SpriteMaterial.hpp"

namespace threepp {

    class Sprite: public Object3D {

    public:
        Vector2 center{0.5f, 0.5f};

        std::shared_ptr<SpriteMaterial> material;

        [[nodiscard]] std::string type() const override;

        void raycast(Raycaster& raycaster, std::vector<Intersection>& intersects) override;

        BufferGeometry* geometry() override;

        static std::shared_ptr<Sprite> create(const std::shared_ptr<SpriteMaterial>& material = SpriteMaterial::create());

    private:
        std::shared_ptr<BufferGeometry> _geometry;

        explicit Sprite(const std::shared_ptr<SpriteMaterial>& material);
    };

}// namespace threepp

#endif//THREEPP_SPRITE_HPP
