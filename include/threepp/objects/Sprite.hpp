// https://github.com/mrdoob/three.js/blob/r129/src/objects/Sprite.js

#ifndef THREEPP_SPRITE_HPP
#define THREEPP_SPRITE_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/InterleavedBuffer.hpp"

namespace threepp {

    class Sprite : public Object3D {

    public:
        void raycast(Raycaster &raycaster, std::vector<Intersection> &intersects) override;

        static std::shared_ptr<Sprite> create(const std::shared_ptr<SpriteMaterial> &material = std::make_shared<SpriteMaterial>()) {

            return std::shared_ptr<Sprite>(new Sprite(material));
        }

    private:

        Vector2 center{0.5f, 0.5f};

        std::shared_ptr<BufferGeometry> _geometry;
        std::shared_ptr<SpriteMaterial> material;

        explicit Sprite(const std::shared_ptr<SpriteMaterial> &material);

        friend class GLRenderer;
    };

}// namespace threepp

#endif//THREEPP_SPRITE_HPP
