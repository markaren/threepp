
#ifndef THREEPP_SVGLOADER_HPP
#define THREEPP_SVGLOADER_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/core/ShapePath.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace threepp {

    class SVGLoader {

    public:
        struct Style {

            std::optional<std::string> fill;
            float fillOpacity;
            float strokeOpacity;
            float strokeWidth;
            std::string strokeLineJoin;
            std::string strokeLineCap;
            float strokeMiterLimit;
            std::string fillRule;
            bool visibility = true;
            float opacity;
            std::optional<std::string> stroke;
            std::string id;
        };

        struct SVGData {

            Style style;
            ShapePath path;
        };

        float defaultDPI = 90;
        // Accepted units: 'mm', 'cm', 'in', 'pt', 'pc', 'px'
        std::string defaultUnit = "px";

        SVGLoader();

        std::vector<SVGData> load(const std::filesystem::path& path);

        std::vector<SVGData> parse(const std::string& text);

        static std::vector<Shape> createShapes(const SVGData& data);

        static std::shared_ptr<BufferGeometry> pointsToStroke(
                const std::vector<Vector2>& points,
                const SVGLoader::Style& style,
                unsigned int arcDivisions = 12,
                float minDistance = 0.001f);

        ~SVGLoader();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp


#endif//THREEPP_SVGLOADER_HPP
