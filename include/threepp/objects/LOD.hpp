// https://github.com/mrdoob/three.js/blob/r129/src/objects/LOD.js

#ifndef THREEPP_LOD_HPP
#define THREEPP_LOD_HPP

#include "threepp/core/Object3D.hpp"

#include <utility>

namespace threepp {

    class Camera;

    struct Level {

        float distance;
        Object3D* object;

        Level(float distance, Object3D* object)
            : distance(distance), object(object) {}
    };

    class LOD: public Object3D {

    public:
        bool autoUpdate = true;

        [[nodiscard]] std::string type() const override;

        LOD& addLevel(const std::shared_ptr<Object3D>& object, float distance = 0);

        [[nodiscard]] size_t getCurrentLevel() const;

        void update(Camera* camera);

        static std::shared_ptr<LOD> create();

    protected:
        LOD() = default;

    private:
        size_t _currentLevel = 0;
        std::vector<Level> levels;
    };

}// namespace threepp

#endif//THREEPP_LOD_HPP
