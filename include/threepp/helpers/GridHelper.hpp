// https://github.com/mrdoob/three.js/blob/r129/src/helpers/GridHelper.js

#ifndef THREEPP_GRIDHELPER_HPP
#define THREEPP_GRIDHELPER_HPP

#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/objects/LineSegments.hpp"

namespace threepp {

    class GridHelper : public LineSegments {

    public:
        static std::shared_ptr<GridHelper> create(int size = 10, int divisions = 10, Color color1 = 0x444444, Color color2 = 0x888888) {

            return std::shared_ptr<GridHelper>(new GridHelper(size, divisions, color1, color2));
        }

    protected:
        GridHelper(int size, int divisions, Color color1, Color color2)
            : LineSegments(nullptr, nullptr) {

            Color c1{color1};
            Color c2{color2};

            const auto center = divisions / 2;
            const auto step = static_cast<float>(size) / divisions;
            const auto halfSize = static_cast<float>(size) / 2;

            std::vector<float> vertices;
            std::vector<float> colors((divisions + 1) * 12);

            int j = 0;
            float k = -halfSize;
            for (int i = 0; i <= divisions; i++) {

                vertices.insert(vertices.end(), {-halfSize, 0, k, halfSize, 0, k});
                vertices.insert(vertices.end(), {k, 0, -halfSize, k, 0, halfSize});

                Color &color = (i == center ? c1 : c2);

                color.toArray(colors, j);
                j += 3;
                color.toArray(colors, j);
                j += 3;
                color.toArray(colors, j);
                j += 3;
                color.toArray(colors, j);
                j += 3;

                k += step;
            }

            auto geometry = BufferGeometry::create();
            geometry->setAttribute("position", FloatBufferAttribute::create(vertices, 3));
            geometry->setAttribute("color", FloatBufferAttribute::create(colors, 3));

            auto material = LineBasicMaterial::create();
            material->vertexColors = true;
            material->toneMapped = false;

            material_ = material;
            geometry_ = geometry;
        }
    };

}// namespace threepp

#endif//THREEPP_GRIDHELPER_HPP
