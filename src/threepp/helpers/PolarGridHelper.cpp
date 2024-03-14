
#include "threepp/helpers/PolarGridHelper.hpp"

#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/math/MathUtils.hpp"

#include <cmath>

using namespace threepp;

PolarGridHelper::PolarGridHelper(float radius, unsigned int sectors, unsigned int rings, unsigned int divisions, const Color& color1, const Color& color2)
    : LineSegments(BufferGeometry::create(), LineBasicMaterial::create()) {

    std::vector<float> vertices;
    std::vector<float> colors;

    // create the sectors

    if (sectors > 1) {

        for (unsigned i = 0; i < sectors; i++) {

            const auto v = (static_cast<float>(i) / static_cast<float>(sectors)) * (math::TWO_PI);

            const auto x = std::sin(v) * radius;
            const auto z = std::cos(v) * radius;

            vertices.insert(vertices.end(), {0, 0, 0});
            vertices.insert(vertices.end(), {x, 0, z});

            const auto color = (i & 1) ? color1 : color2;

            colors.insert(colors.end(), {color.r, color.g, color.b});
            colors.insert(colors.end(), {color.r, color.g, color.b});
        }

        // create the rings

        for (unsigned i = 0; i < rings; i++) {

            const auto color = (i & 1) ? color1 : color2;

            const auto r = radius - (radius / static_cast<float>(rings) * i);

            for (unsigned j = 0; j < divisions; j++) {

                // first vertex

                auto v = (static_cast<float>(j) / static_cast<float>(divisions)) * (math::TWO_PI);

                auto x = std::sin(v) * r;
                auto z = std::cos(v) * r;

                vertices.insert(vertices.end(), {x, 0, z});
                colors.insert(colors.end(), {color.r, color.g, color.b});

                // second vertex

                v = (static_cast<float>(j + 1) / static_cast<float>(divisions)) * (math::TWO_PI);

                x = std::sin(v) * r;
                z = std::cos(v) * r;

                vertices.insert(vertices.end(), {x, 0, z});
                colors.insert(colors.end(), {color.r, color.g, color.b});
            }
        }
    }

    geometry_->setAttribute("position", FloatBufferAttribute::create(vertices, 3));
    geometry_->setAttribute("color", FloatBufferAttribute::create(colors, 3));

    material()->setValues({{"vertexColors", true}, {"toneMapped", false}});
}


std::string PolarGridHelper::type() const {

    return "PolarGridHelper";
}

std::shared_ptr<PolarGridHelper> PolarGridHelper::create(float radius, unsigned int sectors, unsigned int rings, unsigned int divisions, const Color& color1, const Color& color2) {

    return std::shared_ptr<PolarGridHelper>(new PolarGridHelper(radius, sectors, rings, divisions, color1, color2));
}

std::shared_ptr<PolarGridHelper> PolarGridHelper::create(PolarGridHelper::Options options) {

    return create(options.radius, options.sectors, options.rings, options.divisions, options.color1, options.color2);
}

PolarGridHelper::Options::Options(float radius, unsigned int sectors, unsigned int rings, unsigned int divisions, const Color& color1, const Color& color2)
    : radius(radius), sectors(sectors), rings(rings), divisions(divisions), color1(color1), color2(color2) {}
