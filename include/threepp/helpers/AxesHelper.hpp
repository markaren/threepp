
#ifndef THREEPP_AXESHELPER_HPP
#define THREEPP_AXESHELPER_HPP

#include "threepp/objects/LineSegments.hpp"

namespace threepp {

    class AxesHelper: public LineSegments {

    public:

        static std::shared_ptr<AxesHelper> create(float size) {

            return std::shared_ptr<AxesHelper>(new AxesHelper(size));
        }
    protected:
        float size;

    public:
        AxesHelper(float size) : LineSegments(BufferGeometry::create(), LineBasicMaterial::create()), size(size) {

            std::vector<float> vertices {
                    0, 0, 0, size, 0, 0,
                    0, 0, 0, 0, size, 0,
                    0, 0, 0, 0, 0, size
            };

            std::vector<float> colors {
                    1, 0, 0, 1, 0.6f, 0,
                    0, 1, 0, 0.6f, 1, 0,
                    0, 0, 1, 0, 0.6f, 1
            };

            geometry_->setAttribute("position", FloatBufferAttribute::create(vertices, 3));
            geometry_->setAttribute("color", FloatBufferAttribute::create(colors, 3));

            material_->vertexColors = true;

        }
    };

}

#endif//THREEPP_AXESHELPER_HPP
