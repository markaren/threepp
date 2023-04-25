
#include "threepp/loaders/SVGLoader.hpp"
#include "threepp/extras/core/ShapePath.hpp"
#include "threepp/geometries/ShapeGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/Mesh.hpp"

#define NANOSVG_IMPLEMENTATION
#include "nanosvg.h"

using namespace threepp;

namespace {

    std::shared_ptr<Group> loadSVG(NSVGimage* image) {

        auto group = Group::create();
        for (auto shape = image->shapes; shape != nullptr; shape = shape->next) {

            auto material = MeshBasicMaterial::create(
                    {{"color", 0xff0000},
                     {"side", DoubleSide},
                     {"depthWrite", false}});

            std::vector<ShapePath> result;
            for (auto path = shape->paths; path != nullptr; path = path->next) {

                for (unsigned i = 0; i < path->npts - 1; i += 3) {
                    float* p = &path->pts[i * 2];
                    ShapePath& s = result.emplace_back();
                    s.moveTo(p[0], p[1]);
                    s.bezierCurveTo(p[2], p[3], p[4], p[5], p[6], p[7]);
                }

            }
            auto geometry = ShapeGeometry::create(result.back().toShapes());
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
