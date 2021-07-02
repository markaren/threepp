// https://github.com/mrdoob/three.js/blob/r129/src/helpers/ArrowHelper.js

#ifndef THREEPP_ARROWHELPER_HPP
#define THREEPP_ARROWHELPER_HPP

#include "threepp/objects/Line.hpp"
#include "threepp/objects/Mesh.hpp"

namespace threepp {

    class ArrowHelper : public Object3D {

    public:
        void setDirection(const Vector3 &dir);

        void setLength(float length, std::optional<float> headLength = std::nullopt, std::optional<float> headWidth = std::nullopt);

        void setColor(int color);

        static std::shared_ptr<ArrowHelper> create(Vector3 dir = Vector3(0, 0, 1),
                                                   Vector3 origin = Vector3(0, 0, 0),
                                                   float length = 1,
                                                   int color = 0xffff00,
                                                   std::optional<float> headLength = std::nullopt,
                                                   std::optional<float> headWidth = std::nullopt) {

            return std::shared_ptr<ArrowHelper>(new ArrowHelper(dir, origin, length, color, headLength, headWidth));
        }

    protected:
        ArrowHelper(
                Vector3 dir,
                Vector3 origin,
                float length,
                int color,
                std::optional<float> headLength,
                std::optional<float> headWidth);

    private:
        std::shared_ptr<Line> line;
        std::shared_ptr<Mesh> cone;
    };


}// namespace threepp

#endif//THREEPP_ARROWHELPER_HPP
