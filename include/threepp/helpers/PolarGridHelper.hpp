// https://github.com/mrdoob/three.js/blob/r151/src/helpers/PolarGridHelper.js

#ifndef THREEPP_POLARGRIDHELPER_HPP
#define THREEPP_POLARGRIDHELPER_HPP

#include "threepp/objects/LineSegments.hpp"

namespace threepp {

    class PolarGridHelper: public LineSegments {

    public:
        struct Options {
            float radius;
            unsigned int sectors;
            unsigned int rings;
            unsigned int divisions;
            Color color1;
            Color color2;

            explicit Options(float radius = 10,
                             unsigned int sectors = 16,
                             unsigned int rings = 8,
                             unsigned int divisions = 64,
                             const Color& color1 = 0x444444,
                             const Color& color2 = 0x888888);
        };

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<PolarGridHelper> create(PolarGridHelper::Options options);

        static std::shared_ptr<PolarGridHelper> create(
                float radius = 10,
                unsigned int sectors = 16,
                unsigned int rings = 8,
                unsigned int divisions = 64,
                const Color& color1 = 0x444444,
                const Color& color2 = 0x888888);

    private:
        PolarGridHelper(float radius, unsigned int sectors, unsigned int rings, unsigned int divisions, const Color& color1, const Color& color2);
    };

}// namespace threepp

#endif//THREEPP_POLARGRIDHELPER_HPP
