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

        Vector2 center{0.5f, 0.5f};

        std::shared_ptr<BufferGeometry> _geometry;
        std::shared_ptr<SpriteMaterial> material;

        void raycast(Raycaster &raycaster, std::vector<Intersection> &intersects) override;

        BufferGeometry *geometry() override {
            return _geometry.get();
        }

        static std::shared_ptr<Sprite> create(const std::shared_ptr<SpriteMaterial> &material = SpriteMaterial::create()) {

            return std::shared_ptr<Sprite>(new Sprite(material));
        }

    private:

        explicit Sprite(const std::shared_ptr<SpriteMaterial> &material);

        friend class GLRenderer;
    };

}// namespace threepp

#endif//THREEPP_SPRITE_HPP
