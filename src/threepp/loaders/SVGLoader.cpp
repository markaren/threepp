
#include "threepp/loaders/SVGLoader.hpp"

#define NANOSVG_IMPLEMENTATION
#include "nanosvg.h"

using namespace threepp;

namespace {

    void getPaths(NSVGimage* image, std::vector<ShapePath>& result) {

        for (auto shape = image->shapes; shape != nullptr; shape = shape->next) {

            for (auto path = shape->paths; path != nullptr; path = path->next) {
                for (unsigned i = 0; i < path->npts-1; i += 3) {
                    float* p = &path->pts[i*2];
                    auto& s = result.emplace_back();
                    s.splineThru({{p[0],p[1]}, {p[2],p[3]}, {p[4],p[5]}, {p[6],p[7]}});

                }
            }
        }
    }

}

std::vector<ShapePath> SVGLoader::load(const std::filesystem::path& filePath) {

    struct NSVGimage* image;
    image = nsvgParseFromFile(filePath.string().c_str(), defaultUnit.c_str(), defaultDPI);
    std::vector<ShapePath> result;
    getPaths(image, result);
    nsvgDelete(image);

    return result;
}

std::vector<ShapePath> SVGLoader::parse(std::string text) {

    struct NSVGimage* image;
    image = nsvgParse(const_cast<char*>(text.c_str()), defaultUnit.c_str(), defaultDPI);

    std::vector<ShapePath> result;
    getPaths(image, result);
    nsvgDelete(image);

    return result;

}

