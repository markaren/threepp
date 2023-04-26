
#include "threepp/loaders/SVGLoader.hpp"
#include "threepp/extras/core/ShapePath.hpp"
#include "threepp/geometries/ShapeGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"

#include <iostream>

#define NANOSVG_ALL_COLOR_KEYWORDS
#define NANOSVG_IMPLEMENTATION
#include "nanosvg.h"

using namespace threepp;

namespace {

    std::shared_ptr<Group> loadSVG(NSVGimage* image) {

        auto group = Group::create();
        for (auto shape = image->shapes; shape != nullptr; shape = shape->next) {

            ShapePath s;
            for (auto path = shape->paths; path != nullptr; path = path->next) {

                for (unsigned i = 0; i < path->npts - 1; i += 3) {
                    float* p = &path->pts[i * 2];
                    s.moveTo(p[0], p[1]);
                    s.bezierCurveTo(p[2], p[3], p[4], p[5], p[6], p[7]);
                }
            }
            auto material = MeshBasicMaterial::create(
                    {{"color", shape->fill.color},
                     {"opacity", shape->opacity},
                     {"transparent", shape->opacity != 1},
                     {"side", DoubleSide},
                     {"depthWrite", false}});

            auto geometry = ShapeGeometry::create(s.toShapes());
            auto mesh = Mesh::create(geometry, material);
            group->add(mesh);
        }
        return group;
    }

}// namespace

std::shared_ptr<Group> SVGLoader::load(const std::filesystem::path& filePath) {

    struct NSVGimage* image;
    image = nsvgParseFromFile(filePath.string().c_str(), defaultUnit.c_str(), defaultDPI);
    //    std::vector<ShapePath> result;
    //    getPaths(image, result);
    //    nsvgDelete(image);

    auto svg = loadSVG(image);
    nsvgDelete(image);
    return svg;
}

std::shared_ptr<Group> SVGLoader::parse(std::string text) {

    struct NSVGimage* image;
    image = nsvgParse(const_cast<char*>(text.c_str()), defaultUnit.c_str(), defaultDPI);

    //    std::vector<ShapePath> result;
    //    getPaths(image, result);
    //    nsvgDelete(image);

    //    return result;
    auto svg = loadSVG(image);
    nsvgDelete(image);
    return svg;
}
