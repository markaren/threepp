// https://github.com/mrdoob/three.js/blob/dev/src/geometries/BoxGeometry.js

#ifndef THREEPP_BOX_GEOMETRY_HPP
#define THREEPP_BOX_GEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"


namespace threepp {

    class BoxGeometry: public BufferGeometry {

    public:

        struct Params {
            float width;
            float height;
            float depth;

            unsigned int widthSegments;
            unsigned int heightSegments;
            unsigned int depthSegments;

            explicit Params(float width = 1, float height = 1, float depth = 1, unsigned int widthSegments = 1, unsigned int heightSegments = 1, unsigned int depthSegments = 1)
                : width(width), height(height), depth(depth), widthSegments(widthSegments), heightSegments(heightSegments), depthSegments(depthSegments) {}
        };

        const float width;
        const float height;
        const float depth;

        [[nodiscard]] std::string type() const override {

            return "BoxGeometry";
        }

        static std::shared_ptr<BoxGeometry> create(const Params& params) {
            return std::shared_ptr<BoxGeometry>(new BoxGeometry(params));
        }

        static std::shared_ptr<BoxGeometry> create(float width = 1, float height = 1, float depth = 1, unsigned int widthSegments = 1, unsigned int heightSegments = 1, unsigned int depthSegments = 1) {
            return create(Params(width, height, depth, widthSegments, heightSegments, depthSegments));
        }

    protected:
        explicit BoxGeometry(const Params& params);
    };

}// namespace threepp

#endif//THREEPP_BOX_GEOMETRY_HPP
