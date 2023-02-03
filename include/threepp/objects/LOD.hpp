// https://github.com/mrdoob/three.js/blob/r129/src/objects/LOD.js

#ifndef THREEPP_LOD_HPP
#define THREEPP_LOD_HPP

#include <utility>

#include "threepp/core/Object3D.hpp"

namespace threepp {

    struct Level {

        float distance;
        Object3D* object;

        Level(float distance, Object3D* object)
            : distance(distance), object(object) {}
    };

    class LOD : public Object3D {

    public:
        bool autoUpdate = true;

        LOD &addLevel(const std::shared_ptr<Object3D> &object, float distance = 0) {

            distance = std::abs(distance);

            int l;

            for (l = 0; l < levels.size(); l++) {

                if (distance < levels[l].distance) {

                    break;
                }
            }

            levels.insert(levels.begin() + l, {distance, object.get()});

            this->add(object);

            return *this;
        }


        [[nodiscard]] size_t getCurrentLevel() const {

            return _currentLevel;
        }

        void update(Camera *camera) {

            if (levels.size() > 1) {

                _v1.setFromMatrixPosition(*camera->matrixWorld);
                _v2.setFromMatrixPosition(*this->matrixWorld);

                float distance = _v1.distanceTo(_v2) / camera->zoom;

                levels[0].object->visible = true;

                size_t i, l;

                for (i = 1, l = levels.size(); i < l; i++) {

                    if (distance >= levels[i].distance) {

                        levels[i - 1].object->visible = false;
                        levels[i].object->visible = true;

                    } else {

                        break;
                    }
                }

                this->_currentLevel = i - 1;

                for (; i < l; i++) {

                    levels[i].object->visible = false;
                }
            }
        }

        static std::shared_ptr<LOD> create() {

            return std::shared_ptr<LOD>(new LOD());
        }

    protected:
        LOD() = default;

    private:
        size_t _currentLevel = 0;
        std::vector<Level> levels;

        inline static Vector3 _v1;
        inline static Vector3 _v2;
    };

}// namespace threepp

#endif//THREEPP_LOD_HPP
