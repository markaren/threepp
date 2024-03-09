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

        explicit Sprite(const std::shared_ptr<SpriteMaterial>& material);

        [[nodiscard]] std::string type() const override;

        void raycast(const Raycaster& raycaster, std::vector<Intersection>& intersects) override;

        BufferGeometry* geometry() override;

        Material* material() override;

        void setMaterial(const std::shared_ptr<SpriteMaterial>& material);

        static std::shared_ptr<Sprite> create(const std::shared_ptr<SpriteMaterial>& material = nullptr);

    private:
        std::shared_ptr<BufferGeometry> _geometry;
        std::shared_ptr<SpriteMaterial> _material;
    };

}// namespace threepp

#endif//THREEPP_SPRITE_HPP
