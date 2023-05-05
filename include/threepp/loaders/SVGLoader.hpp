
#ifndef THREEPP_SVGLOADER_HPP
#define THREEPP_SVGLOADER_HPP

#include <filesystem>
#include <memory>
#include <string>

#include "threepp/extras/core/ShapePath.hpp"

namespace threepp {

    class BufferGeometry;

    class SVGLoader {

    public:
        struct Style {
            std::optional<Color> stroke;
            float fillOpacity{1};
            float strokeOpacity{1};
            float strokeWidth{1};
            float strokeMiterLimit{4};

            std::string strokeLineJoin;
            std::string strokeLineCap;
        };

        struct Shape {
            Style style;
            std::string id;
            ShapePath path;
        };


        float defaultDPI = 90;
        // Accepted units: 'mm', 'cm', 'in', 'pt', 'pc', 'px'
        std::string defaultUnit = "px";

        std::vector<SVGLoader::Shape> load(const std::filesystem::path& path);

        std::vector<SVGLoader::Shape> parse(std::string text);

        static std::unique_ptr<threepp::Shape> createShapes(const ShapePath& shapePath);

        static std::shared_ptr<BufferGeometry> pointsToStroke(const std::vector<Vector2>& points, const SVGLoader::Style& style, unsigned int arcDivisions = 12, float minDistance = 0.001f);
    };

}// namespace threepp

#endif//THREEPP_SVGLOADER_HPP
