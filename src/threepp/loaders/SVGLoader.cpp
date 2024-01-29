
#include "threepp/loaders/SVGLoader.hpp"

#include "threepp/extras/ShapeUtils.hpp"
#include "threepp/loaders/svg/SVGFunctions.hpp"

#include "pugixml.hpp"

#include <algorithm>

using namespace threepp;

namespace {

    std::vector<std::string> units{"mm", "cm", "in", "pt", "pc", "px"};

    std::unordered_map<std::string, std::unordered_map<std::string, float>> unitConversion{
            {"mm", {{"mm", 1.f}, {"cm", 0.1f}, {"in", 1.f / 25.4f}, {"pt", 72.f / 25.4f}, {"pc", 6.f / 25.4f}, {"px", -1.f}}},
            {"cm", {{"mm", 10.f}, {"cm", 1.f}, {"in", 1.f / 2.54f}, {"pt", 72.f / 2.54f}, {"pc", 6.f / 2.54f}, {"px", -1.f}}},
            {"in", {{"mm", 25.4f}, {"cm", 2.54f}, {"in", 1.f}, {"pt", 72.f}, {"pc", 6.f}, {"px", -1.f}}},
            {"pt", {{"mm", 25.4f / 72.f}, {"cm", 2.54f / 72.f}, {"in", 1.f / 72.f}, {"pt", 1.f}, {"pc", 6.f / 72.f}, {"px", -1.f}}},
            {"pc", {{"mm", 25.4f / 6.f}, {"cm", 2.54f / 6.f}, {"in", 1.f / 6.f}, {"pt", 72.f / 6.f}, {"pc", 1.f}, {"px", -1.f}}},
            {"px", {{"px", 1.f}}}};

}// namespace

ShapePath parsePathNode(const pugi::xml_node& node);

unsigned int pointsToStrokeWithBuffers(
        std::vector<Vector2> points,
        const SVGLoader::Style& style,
        unsigned int arcDivisions,
        float minDistance,
        std::vector<float>& vertices,
        std::vector<float>& normals,
        std::vector<float>& uvs,
        unsigned int vertexOffset = 0);

std::shared_ptr<BufferGeometry> pointsToStroke(
        const std::vector<Vector2>& points,
        const SVGLoader::Style& style,
        unsigned int arcDivisions,
        float minDistance);

struct SVGLoader::Impl {

    SVGLoader* scope;

    std::vector<SVGLoader::SVGData> paths;
    std::unordered_map<std::string, std::string> styleSheets;

    std::vector<Matrix3> transformStack;
    Matrix3 currentTransform;

    explicit Impl(SVGLoader& scope): scope(&scope) {}

    std::vector<SVGLoader::SVGData> load(const pugi::xml_node& node);

    [[nodiscard]] float clamp(float v) const {

        return std::max(0.f, std::min(1.f, parseFloatWithUnits(v)));
    }

    [[nodiscard]] float positive(float v) const {

        return std::max(0.f, parseFloatWithUnits(v));
    }

    [[nodiscard]] float parseFloatWithUnits(float value, const std::string& theUnit = "px") const;

    [[nodiscard]] float parseFloatWithUnits(const std::string& str) const;

    void parseNode(const pugi::xml_node& node, Style style);

    std::optional<Matrix3> getNodeTransform(const pugi::xml_node& node);

    [[nodiscard]] Matrix3 parseNodeTransform(const pugi::xml_node& node) const;

    /*
        * According to https://www.w3.org/TR/SVG/shapes.html#RectElementRXAttribute
        * rounded corner should be rendered to elliptical arc, but bezier curve does the job well enough
        */
    [[nodiscard]] ShapePath parseRectNode(const pugi::xml_node& node) const;

    [[nodiscard]] ShapePath parsePolygonNode(const pugi::xml_node& node) const;

    [[nodiscard]] ShapePath parsePolylineNode(const pugi::xml_node& node) const;

    [[nodiscard]] ShapePath parseCircleNode(const pugi::xml_node& node) const;

    [[nodiscard]] ShapePath parseEllipseNode(const pugi::xml_node& node) const;

    [[nodiscard]] ShapePath parseLineNode(const pugi::xml_node& node) const;

    [[nodiscard]] Style parseStyle(const pugi::xml_node& node, Style style) const;
};

std::vector<SVGLoader::SVGData> SVGLoader::load(const std::filesystem::path& filePath) {

    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(filePath.string().c_str());
    if (!result) {
        throw std::runtime_error("Unable to parse modelDescription.xml");
    }

    return pimpl_->load(doc);
}

std::vector<SVGLoader::SVGData> SVGLoader::parse(const std::string& text) {

    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_string(text.c_str());
    if (!result) {
        throw std::runtime_error("Unable to parse modelDescription.xml");
    }

    return pimpl_->load(doc);
}

std::vector<Shape> SVGLoader::createShapes(const SVGData& data) {

    const auto& shapePath = data.path;

    const auto BIGNUMBER = 999999999.f;

    auto scanlineMinX = BIGNUMBER;
    auto scanlineMaxX = -BIGNUMBER;

    std::vector<svg::SimplePath> simplePaths;
    const auto& subPaths = shapePath.subPaths;
    std::transform(subPaths.begin(), subPaths.end(), std::back_inserter(simplePaths), [&](const auto& p) {
        const auto& points = p->getPoints();
        float maxY = -BIGNUMBER;
        float minY = BIGNUMBER;
        float maxX = -BIGNUMBER;
        float minX = BIGNUMBER;

        //points.forEach(p => p.y *= -1);

        for (unsigned i = 0; i < points.size(); i++) {

            const auto& p = points[i];

            if (p.y > maxY) {

                maxY = p.y;
            }

            if (p.y < minY) {

                minY = p.y;
            }

            if (p.x > maxX) {

                maxX = p.x;
            }

            if (p.x < minX) {

                minX = p.x;
            }
        }

        //
        if (scanlineMaxX <= maxX) {

            scanlineMaxX = maxX + 1;
        }

        if (scanlineMinX >= minX) {

            scanlineMinX = minX - 1;
        }

        return svg::SimplePath{p->curves, points, shapeutils::isClockWise(points), -1, Box2({minX, minY}, {maxX, maxY})};
    });

    // simplePaths = simplePaths.filter( sp => sp.points.length > 1 );
    for (auto it = simplePaths.begin(); it != simplePaths.end();) {
        if (it->points.size() > 1) {
            ++it;
        } else {
            it = simplePaths.erase(it);
        }
    }

    for (int identifier = 0; identifier < simplePaths.size(); identifier++) {

        simplePaths[identifier].identifier = identifier;
    }

    std::vector<std::optional<svg::AHole>> isAHole;
    std::transform(simplePaths.begin(), simplePaths.end(), std::back_inserter(isAHole), [&](const auto& p) {
        return isHoleTo(p, simplePaths, scanlineMinX, scanlineMaxX, data.style.fillRule);
    });

    std::vector<Shape> shapesToReturn;
    for (const auto& p : simplePaths) {

        const auto amIAHole = isAHole[p.identifier];

        if (amIAHole && !amIAHole->isHole) {

            Shape& shape = shapesToReturn.emplace_back();
            shape.curves = p.curves;

            for (const auto& h : isAHole) {

                if (h->isHole && h->_for == p.identifier) {

                    const auto& hole = simplePaths[h->identifier];
                    auto path = std::make_shared<Path>();
                    path->curves = hole.curves;
                    shape.holes.emplace_back(path);
                }
            }
        }
    }

    return shapesToReturn;
}

SVGLoader::SVGLoader()
    : pimpl_(std::make_unique<Impl>(*this)) {}

std::shared_ptr<BufferGeometry> SVGLoader::pointsToStroke(const std::vector<Vector2>& points, const SVGLoader::Style& style, unsigned int arcDivisions, float minDistance) {
    return ::pointsToStroke(points, style, arcDivisions, minDistance);
}

SVGLoader::~SVGLoader() = default;


unsigned int pointsToStrokeWithBuffers(
        std::vector<Vector2> points,
        const SVGLoader::Style& style,
        unsigned int arcDivisions,
        float minDistance,
        std::vector<float>& vertices,
        std::vector<float>& normals,
        std::vector<float>& uvs,
        unsigned int vertexOffset) {

    // This function can be called to update existing arrays or buffers.
    // Accepts same parameters as pointsToStroke, plus the buffers and optional offset.
    // Param vertexOffset: Offset vertices to start writing in the buffers (3 elements/vertex for vertices and normals, and 2 elements/vertex for uvs)
    // Returns number of written vertices / normals / uvs pairs
    // if 'vertices' parameter is undefined no triangles will be generated, but the returned vertices count will still be valid (useful to preallocate the buffers)
    // 'normals' and 'uvs' buffers are optional

    Vector2 tempV2_1;
    Vector2 tempV2_2;
    Vector2 tempV2_3;
    Vector2 tempV2_4;
    Vector2 tempV2_5;
    Vector2 tempV2_6;
    Vector2 tempV2_7;
    Vector2 lastPointL;
    Vector2 lastPointR;
    Vector2 point0L;
    Vector2 point0R;
    Vector2 currentPointL;
    Vector2 currentPointR;
    Vector2 nextPointL;
    Vector2 nextPointR;
    Vector2 innerPoint;
    Vector2 outerPoint;

    svg::removeDuplicatedPoints(points, minDistance);

    auto numPoints = points.size();

    if (numPoints < 2) return 0;

    bool isClosed = points[0].equals(points[numPoints - 1]);

    Vector2 currentPoint;
    Vector2 previousPoint = points[0];
    std::optional<Vector2> nextPoint;

    float strokeWidth2 = style.strokeWidth / 2;

    float deltaU = 1.f / static_cast<float>(numPoints - 1);
    float u0 = 0, u1;

    bool innerSideModified{};
    bool joinIsOnLeftSide{};
    bool isMiter{};
    bool initialJoinIsOnLeftSide = false;

    size_t numVertices = 0;
    auto currentCoordinate = vertexOffset * 3;
    auto currentCoordinateUV = vertexOffset * 2;

    // Get initial left and right stroke points
    svg::getNormal(points[0], points[1], tempV2_1).multiplyScalar(strokeWidth2);
    lastPointL.copy(points[0]).sub(tempV2_1);
    lastPointR.copy(points[0]).add(tempV2_1);
    point0L.copy(lastPointL);
    point0R.copy(lastPointR);

    auto addVertex = [&](const Vector2& position, float u, float v) {
        vertices.resize(currentCoordinate + 3);
        vertices[currentCoordinate] = position.x;
        vertices[currentCoordinate + 1] = position.y;
        vertices[currentCoordinate + 2] = 0;

        normals.resize(currentCoordinate + 3);
        normals[currentCoordinate] = 0;
        normals[currentCoordinate + 1] = 0;
        normals[currentCoordinate + 2] = 1;

        currentCoordinate += 3;

        uvs.resize(currentCoordinateUV + 2);
        uvs[currentCoordinateUV] = u;
        uvs[currentCoordinateUV + 1] = v;

        currentCoordinateUV += 2;

        numVertices += 3;
    };

    auto makeSegmentTriangles = [&]() {
        addVertex(lastPointR, u0, 1);
        addVertex(lastPointL, u0, 0);
        addVertex(currentPointL, u1, 0);

        addVertex(lastPointR, u0, 1);
        addVertex(currentPointL, u1, 1);
        addVertex(currentPointR, u1, 0);
    };

    auto makeSegmentWithBevelJoin = [&](bool joinIsOnLeftSide, bool innerSideModified, float u) {
        if (innerSideModified) {

            // Optimized segment + bevel triangles

            if (joinIsOnLeftSide) {

                // Path segments triangles

                addVertex(lastPointR, u0, 1);
                addVertex(lastPointL, u0, 0);
                addVertex(currentPointL, u1, 0);

                addVertex(lastPointR, u0, 1);
                addVertex(currentPointL, u1, 0);
                addVertex(innerPoint, u1, 1);

                // Bevel join triangle

                addVertex(currentPointL, u, 0);
                addVertex(nextPointL, u, 0);
                addVertex(innerPoint, u, 0.5f);

            } else {

                // Path segments triangles

                addVertex(lastPointR, u0, 1);
                addVertex(lastPointL, u0, 0);
                addVertex(currentPointR, u1, 1);

                addVertex(lastPointL, u0, 0);
                addVertex(innerPoint, u1, 0);
                addVertex(currentPointR, u1, 1);

                // Bevel join triangle

                addVertex(currentPointR, u, 1);
                addVertex(nextPointR, u, 0);
                addVertex(innerPoint, u, 0.5f);
            }

        } else {

            // Bevel join triangle. The segment triangles are done in the main loop

            if (joinIsOnLeftSide) {

                addVertex(currentPointL, u, 0);
                addVertex(nextPointL, u, 0);
                addVertex(currentPoint, u, 0.5f);

            } else {

                addVertex(currentPointR, u, 1);
                addVertex(nextPointR, u, 0);
                addVertex(currentPoint, u, 0.5f);
            }
        }
    };

    auto createSegmentTrianglesWithMiddleSection = [&](bool joinIsOnLeftSide, bool innerSideModified) {
        if (innerSideModified) {

            if (joinIsOnLeftSide) {

                addVertex(lastPointR, u0, 1);
                addVertex(lastPointL, u0, 0);
                addVertex(currentPointL, u1, 0);

                addVertex(lastPointR, u0, 1);
                addVertex(currentPointL, u1, 0);
                addVertex(innerPoint, u1, 1);

                addVertex(currentPointL, u0, 0);
                addVertex(currentPoint, u1, 0.5f);
                addVertex(innerPoint, u1, 1);

                addVertex(currentPoint, u1, 0.5f);
                addVertex(nextPointL, u0, 0);
                addVertex(innerPoint, u1, 1);

            } else {

                addVertex(lastPointR, u0, 1);
                addVertex(lastPointL, u0, 0);
                addVertex(currentPointR, u1, 1);

                addVertex(lastPointL, u0, 0);
                addVertex(innerPoint, u1, 0);
                addVertex(currentPointR, u1, 1);

                addVertex(currentPointR, u0, 1);
                addVertex(innerPoint, u1, 0);
                addVertex(currentPoint, u1, 0.5f);

                addVertex(currentPoint, u1, 0.5f);
                addVertex(innerPoint, u1, 0);
                addVertex(nextPointR, u0, 1);
            }
        }
    };


    auto makeCircularSector = [&](const Vector2& center, const Vector2& p1, const Vector2& p2, float u, float v) {
        // param p1, p2: Points in the circle arc.
        // p1 and p2 are in clockwise direction.

        tempV2_1.copy(p1).sub(center).normalize();
        tempV2_2.copy(p2).sub(center).normalize();

        float angle = math::PI;
        const auto dot = tempV2_1.dot(tempV2_2);
        if (std::abs(dot) < 1) angle = std::abs(std::acos(dot));

        angle /= static_cast<float>(arcDivisions);

        tempV2_3.copy(p1);

        for (unsigned i = 0, il = arcDivisions - 1; i < il; i++) {

            tempV2_4.copy(tempV2_3).rotateAround(center, angle);

            addVertex(tempV2_3, u, v);
            addVertex(tempV2_4, u, v);
            addVertex(center, u, 0.5f);

            tempV2_3.copy(tempV2_4);
        }

        addVertex(tempV2_4, u, v);
        addVertex(p2, u, v);
        addVertex(center, u, 0.5f);
    };


    auto addCapGeometry = [&](const Vector2& center, const Vector2& p1, const Vector2& p2, bool joinIsOnLeftSide, bool start, float u) {
        // param center: End point of the path
        // param p1, p2: Left and right cap points

        if (style.strokeLineCap == "round") {

            if (start) {

                makeCircularSector(center, p2, p1, u, 0.5f);

            } else {

                makeCircularSector(center, p1, p2, u, 0.5f);
            }

        } else if (style.strokeLineCap == "square") {
            if (start) {

                tempV2_1.subVectors(p1, center);
                tempV2_2.set(tempV2_1.y, -tempV2_1.x);

                tempV2_3.addVectors(tempV2_1, tempV2_2).add(center);
                tempV2_4.subVectors(tempV2_2, tempV2_1).add(center);

                // Modify already existing vertices
                if (joinIsOnLeftSide) {

                    tempV2_3.toArray(vertices, 1 * 3);
                    tempV2_4.toArray(vertices, 0 * 3);
                    tempV2_4.toArray(vertices, 3 * 3);

                } else {

                    tempV2_3.toArray(vertices, 1 * 3);
                    tempV2_3.toArray(vertices, 3 * 3);
                    tempV2_4.toArray(vertices, 0 * 3);
                }

            } else {

                tempV2_1.subVectors(p2, center);
                tempV2_2.set(tempV2_1.y, -tempV2_1.x);

                tempV2_3.addVectors(tempV2_1, tempV2_2).add(center);
                tempV2_4.subVectors(tempV2_2, tempV2_1).add(center);

                const auto vl = vertices.size();

                // Modify already existing vertices
                if (joinIsOnLeftSide) {

                    tempV2_3.toArray(vertices, vl - 1 * 3);
                    tempV2_4.toArray(vertices, vl - 2 * 3);
                    tempV2_4.toArray(vertices, vl - 4 * 3);

                } else {

                    tempV2_3.toArray(vertices, vl - 2 * 3);
                    tempV2_4.toArray(vertices, vl - 1 * 3);
                    tempV2_4.toArray(vertices, vl - 4 * 3);
                }
            }
        }
    };

    for (unsigned iPoint = 1; iPoint < numPoints; iPoint++) {

        currentPoint = points[iPoint];

        // Get next point
        if (iPoint == numPoints - 1) {

            if (isClosed) {

                // Skip duplicated initial point
                nextPoint = points[1];

            } else
                nextPoint = std::nullopt;

        } else {

            nextPoint = points[iPoint + 1];
        }

        // Normal of previous segment in tempV2_1
        auto& normal1 = tempV2_1;
        svg::getNormal(previousPoint, currentPoint, normal1);

        tempV2_3.copy(normal1).multiplyScalar(strokeWidth2);
        currentPointL.copy(currentPoint).sub(tempV2_3);
        currentPointR.copy(currentPoint).add(tempV2_3);

        u1 = u0 + deltaU;

        innerSideModified = false;

        if (nextPoint) {

            // Normal of next segment in tempV2_2
            svg::getNormal(currentPoint, *nextPoint, tempV2_2);

            tempV2_3.copy(tempV2_2).multiplyScalar(strokeWidth2);
            nextPointL.copy(currentPoint).sub(tempV2_3);
            nextPointR.copy(currentPoint).add(tempV2_3);

            joinIsOnLeftSide = true;
            tempV2_3.subVectors(*nextPoint, previousPoint);
            if (normal1.dot(tempV2_3) < 0) {

                joinIsOnLeftSide = false;
            }

            if (iPoint == 1) initialJoinIsOnLeftSide = joinIsOnLeftSide;

            tempV2_3.subVectors(*nextPoint, currentPoint);
            tempV2_3.normalize();
            const auto dot = std::abs(normal1.dot(tempV2_3));

            // If path is straight, don't create join
            if (dot > std::numeric_limits<float>::epsilon()) {

                // Compute inner and outer segment intersections
                const auto miterSide = strokeWidth2 / dot;
                tempV2_3.multiplyScalar(-miterSide);
                tempV2_4.subVectors(currentPoint, previousPoint);
                tempV2_5.copy(tempV2_4).setLength(miterSide).add(tempV2_3);
                innerPoint.copy(tempV2_5).negate();
                const auto miterLength2 = tempV2_5.length();
                const auto segmentLengthPrev = tempV2_4.length();
                tempV2_4.divideScalar(segmentLengthPrev);
                tempV2_6.subVectors(*nextPoint, currentPoint);
                const auto segmentLengthNext = tempV2_6.length();
                tempV2_6.divideScalar(segmentLengthNext);
                // Check that previous and next segments doesn't overlap with the innerPoint of intersection
                if (tempV2_4.dot(innerPoint) < segmentLengthPrev && tempV2_6.dot(innerPoint) < segmentLengthNext) {

                    innerSideModified = true;
                }

                outerPoint.copy(tempV2_5).add(currentPoint);
                innerPoint.add(currentPoint);

                isMiter = false;

                if (innerSideModified) {

                    if (joinIsOnLeftSide) {

                        nextPointR.copy(innerPoint);
                        currentPointR.copy(innerPoint);

                    } else {

                        nextPointL.copy(innerPoint);
                        currentPointL.copy(innerPoint);
                    }

                } else {

                    // The segment triangles are generated here if there was overlapping

                    makeSegmentTriangles();
                }

                if (style.strokeLineJoin == "bevel") {

                    makeSegmentWithBevelJoin(joinIsOnLeftSide, innerSideModified, u1);

                } else if (style.strokeLineJoin == "round") {

                    // Segment triangles

                    createSegmentTrianglesWithMiddleSection(joinIsOnLeftSide, innerSideModified);

                    // Join triangles

                    if (joinIsOnLeftSide) {

                        makeCircularSector(currentPoint, currentPointL, nextPointL, u1, 0);

                    } else {

                        makeCircularSector(currentPoint, nextPointR, currentPointR, u1, 1);
                    }
                } else {

                    const auto miterFraction = (strokeWidth2 * style.strokeMiterLimit) / miterLength2;

                    if (miterFraction < 1) {

                        // The join miter length exceeds the miter limit

                        if (style.strokeLineJoin != "miter-clip") {

                            makeSegmentWithBevelJoin(joinIsOnLeftSide, innerSideModified, u1);
                            break;

                        } else {

                            // Segment triangles

                            createSegmentTrianglesWithMiddleSection(joinIsOnLeftSide, innerSideModified);

                            // Miter-clip join triangles

                            if (joinIsOnLeftSide) {

                                tempV2_6.subVectors(outerPoint, currentPointL).multiplyScalar(miterFraction).add(currentPointL);
                                tempV2_7.subVectors(outerPoint, nextPointL).multiplyScalar(miterFraction).add(nextPointL);

                                addVertex(currentPointL, u1, 0);
                                addVertex(tempV2_6, u1, 0);
                                addVertex(currentPoint, u1, 0.5f);

                                addVertex(currentPoint, u1, 0.5f);
                                addVertex(tempV2_6, u1, 0);
                                addVertex(tempV2_7, u1, 0);

                                addVertex(currentPoint, u1, 0.5f);
                                addVertex(tempV2_7, u1, 0);
                                addVertex(nextPointL, u1, 0);

                            } else {

                                tempV2_6.subVectors(outerPoint, currentPointR).multiplyScalar(miterFraction).add(currentPointR);
                                tempV2_7.subVectors(outerPoint, nextPointR).multiplyScalar(miterFraction).add(nextPointR);

                                addVertex(currentPointR, u1, 1);
                                addVertex(tempV2_6, u1, 1);
                                addVertex(currentPoint, u1, 0.5f);

                                addVertex(currentPoint, u1, 0.5f);
                                addVertex(tempV2_6, u1, 1);
                                addVertex(tempV2_7, u1, 1);

                                addVertex(currentPoint, u1, 0.5f);
                                addVertex(tempV2_7, u1, 1);
                                addVertex(nextPointR, u1, 1);
                            }
                        }

                    } else {

                        // Miter join segment triangles

                        if (innerSideModified) {

                            // Optimized segment + join triangles

                            if (joinIsOnLeftSide) {

                                addVertex(lastPointR, u0, 1);
                                addVertex(lastPointL, u0, 0);
                                addVertex(outerPoint, u1, 0);

                                addVertex(lastPointR, u0, 1);
                                addVertex(outerPoint, u1, 0);
                                addVertex(innerPoint, u1, 1);

                            } else {

                                addVertex(lastPointR, u0, 1);
                                addVertex(lastPointL, u0, 0);
                                addVertex(outerPoint, u1, 1);

                                addVertex(lastPointL, u0, 0);
                                addVertex(innerPoint, u1, 0);
                                addVertex(outerPoint, u1, 1);
                            }


                            if (joinIsOnLeftSide) {

                                nextPointL.copy(outerPoint);

                            } else {

                                nextPointR.copy(outerPoint);
                            }


                        } else {

                            // Add extra miter join triangles

                            if (joinIsOnLeftSide) {

                                addVertex(currentPointL, u1, 0);
                                addVertex(outerPoint, u1, 0);
                                addVertex(currentPoint, u1, 0.5f);

                                addVertex(currentPoint, u1, 0.5f);
                                addVertex(outerPoint, u1, 0);
                                addVertex(nextPointL, u1, 0);

                            } else {

                                addVertex(currentPointR, u1, 1);
                                addVertex(outerPoint, u1, 1);
                                addVertex(currentPoint, u1, 0.5f);

                                addVertex(currentPoint, u1, 0.5f);
                                addVertex(outerPoint, u1, 1);
                                addVertex(nextPointR, u1, 1);
                            }
                        }

                        isMiter = true;
                    }
                }

            } else {

                // The segment triangles are generated here when two consecutive points are collinear

                makeSegmentTriangles();
            }

        } else {

            // The segment triangles are generated here if it is the ending segment

            makeSegmentTriangles();
        }

        if (!isClosed && iPoint == numPoints - 1) {

            // Start line endcap
            addCapGeometry(points[0], point0L, point0R, joinIsOnLeftSide, true, u0);
        }

        // Increment loop variables

        u0 = u1;

        previousPoint = currentPoint;

        lastPointL.copy(nextPointL);
        lastPointR.copy(nextPointR);
    }

    if (!isClosed) {

        // Ending line endcap
        addCapGeometry(currentPoint, currentPointL, currentPointR, joinIsOnLeftSide, false, u1);

    } else if (innerSideModified) {

        // Modify path first segment vertices to adjust to the segments inner and outer intersections

        auto& lastOuter = outerPoint;
        auto& lastInner = innerPoint;

        if (initialJoinIsOnLeftSide != joinIsOnLeftSide) {

            lastOuter = innerPoint;
            lastInner = outerPoint;
        }

        if (joinIsOnLeftSide) {

            if (isMiter || initialJoinIsOnLeftSide) {

                lastInner.toArray(vertices, 0 * 3);
                lastInner.toArray(vertices, 3 * 3);

                if (isMiter) {

                    lastOuter.toArray(vertices, 1 * 3);
                }
            }

        } else {

            if (isMiter || !initialJoinIsOnLeftSide) {

                lastInner.toArray(vertices, 1 * 3);
                lastInner.toArray(vertices, 3 * 3);

                if (isMiter) {

                    lastOuter.toArray(vertices, 0 * 3);
                }
            }
        }
    }

    return numVertices;
}

std::vector<SVGLoader::SVGData> SVGLoader::Impl::load(const pugi::xml_node& node) {

    paths.clear();
    styleSheets.clear();
    transformStack.clear();

    parseNode(node.child("svg"), {"#000", 1, 1, 1, "miter", "butt", 4});

    return paths;
}

void SVGLoader::Impl::parseNode(const pugi::xml_node& node, SVGLoader::Style style) {

    if (node.type() != pugi::xml_node_type::node_element) return;

    const auto transform = getNodeTransform(node);

    bool traverseChildNodes = true;

    std::optional<ShapePath> path;

    std::string nodeName{node.name()};
    if (nodeName == "svg") {

    } else if (nodeName == "style") {

        style = parseStyle(node, style);

    } else if (nodeName == "g") {

        style = parseStyle(node, style);

    } else if (nodeName == "path") {

        style = parseStyle(node, style);
        if (node.attribute("d")) {
            path = parsePathNode(node);
        }

    } else if (nodeName == "rect") {

        style = parseStyle(node, style);
        path = parseRectNode(node);

    } else if (nodeName == "polygon") {

        style = parseStyle(node, style);
        path = parsePolygonNode(node);

    } else if (nodeName == "polyline") {

        style = parseStyle(node, style);
        path = parsePolylineNode(node);

    } else if (nodeName == "circle") {

        style = parseStyle(node, style);
        path = parseCircleNode(node);

    } else if (nodeName == "ellipse") {

        style = parseStyle(node, style);
        path = parseEllipseNode(node);

    } else if (nodeName == "line") {

        style = parseStyle(node, style);
        path = parseLineNode(node);

    } else if (nodeName == "defs") {

        traverseChildNodes = false;

    } else if (nodeName == "use") {

        style = parseStyle(node, style);
    }

    if (path) {

        if (style.fill && !style.fill->empty() && *style.fill != "none") {

            path->color.setStyle(*style.fill);
        }

        svg::transformPath(*path, currentTransform);

        paths.emplace_back(SVGLoader::SVGData{style, *path});
    }

    if (traverseChildNodes) {

        for (const auto& n : node.children()) {

            parseNode(n, style);
        }
    }

    if (transform) {

        transformStack.pop_back();

        if (!transformStack.empty()) {

            currentTransform.copy(transformStack[transformStack.size() - 1]);

        } else {

            currentTransform.identity();
        }
    }
}

float SVGLoader::Impl::parseFloatWithUnits(float value, const std::string& theUnit) const {

    float scale;

    if (theUnit == "px" && scope->defaultUnit != "px") {

        // Conversion scale from  pixels to inches, then to default units

        scale = unitConversion.at("in").at(scope->defaultUnit) / scope->defaultDPI;

    } else {

        scale = unitConversion.at(theUnit).at(scope->defaultUnit);

        if (scale < 0) {

            // Conversion scale to pixels

            scale = unitConversion.at(theUnit).at("in") * scope->defaultDPI;
        }
    }

    return scale * value;
}

float SVGLoader::Impl::parseFloatWithUnits(const std::string& str) const {

    std::string theUnit = "px";

    float value;

    if (!utils::isNumber(str)) {

        for (unsigned i = 0, n = units.size(); i < n; i++) {

            const auto u = units[i];

            if (utils::endsWith(str, u)) {

                theUnit = u;
                value = std::stof(str.substr(0, str.size() - u.size()));
                break;
            }
        }

    } else {

        value = std::stof(str);
    }

    return parseFloatWithUnits(value, theUnit);
}

std::optional<Matrix3> SVGLoader::Impl::getNodeTransform(const pugi::xml_node& node) {

    if (!(node.attribute("transform") || (std::string(node.name()) == "use" && (node.attribute("x") || node.attribute("y"))))) {

        return std::nullopt;
    }

    auto transform = parseNodeTransform(node);

    if (!transformStack.empty()) {

        transform.premultiply(transformStack[transformStack.size() - 1]);
    }

    currentTransform.copy(transform);
    transformStack.emplace_back(transform);

    return transform;
}

Matrix3 SVGLoader::Impl::parseNodeTransform(const pugi::xml_node& node) const {

    Matrix3 transform;
    Matrix3 currentTransform;
    Matrix3 tempTransform0;
    Matrix3 tempTransform1;
    Matrix3 tempTransform2;
    Matrix3 tempTransform3;

    if (std::string(node.name()) == "use" && (node.attribute("x") || node.attribute("y"))) {

        const auto tx = parseFloatWithUnits(node.attribute("x").value());
        const auto ty = parseFloatWithUnits(node.attribute("y").value());

        transform.translate(tx, ty);
    }

    if (node.attribute("transform")) {

        auto transformsTexts = utils::split(node.attribute("transform").value(), ')');

        for (int tIndex = transformsTexts.size() - 1; tIndex >= 0; tIndex--) {

            const auto transformText = utils::trim(transformsTexts[tIndex]);

            if (transformText.empty()) continue;

            const auto openParPos = transformText.find('(');
            const auto closeParPos = transformText.size();

            if (openParPos != std::string::npos && openParPos > 0 && openParPos < closeParPos) {

                const auto transformType = transformText.substr(0, openParPos);

                const auto array = svg::parseFloats(transformText.substr(openParPos + 1, closeParPos - openParPos - 1));

                currentTransform.identity();

                if (transformType == "translate") {

                    if (!array.empty()) {

                        const auto tx = array[0];
                        auto ty = tx;

                        if (array.size() >= 2) {

                            ty = array[1];
                        }

                        currentTransform.translate(tx, ty);
                    }
                } else if (transformType == "rotate") {

                    if (!array.empty()) {

                        float angle = 0;
                        float cx = 0;
                        float cy = 0;

                        // Angle
                        angle = -array[0] * math::PI / 180;

                        if (array.size() >= 3) {

                            // Center x, y
                            cx = array[1];
                            cy = array[2];
                        }

                        // Rotate around center (cx, cy)
                        tempTransform1.identity().translate(-cx, -cy);
                        tempTransform2.identity().rotate(angle);
                        tempTransform3.multiplyMatrices(tempTransform2, tempTransform1);
                        tempTransform1.identity().translate(cx, cy);
                        currentTransform.multiplyMatrices(tempTransform1, tempTransform3);
                    }

                } else if (transformType == "scale") {

                    if (!array.empty()) {

                        const auto scaleX = array[0];
                        auto scaleY = scaleX;

                        if (array.size() >= 2) {

                            scaleY = array[1];
                        }

                        currentTransform.scale(scaleX, scaleY);
                    }
                } else if (transformType == "skewX") {

                    if (array.size() == 1) {

                        currentTransform.set(
                                1, std::tan(array[0] * math::PI / 180), 0,
                                0, 1, 0,
                                0, 0, 1);
                    }

                } else if (transformType == "skewY") {

                    if (array.size() == 1) {

                        currentTransform.set(
                                1, 0, 0,
                                std::tan(array[0] * math::PI / 180), 1, 0,
                                0, 0, 1);
                    }

                } else if (transformType == "matrix") {

                    if (array.size() == 6) {

                        currentTransform.set(
                                array[0], array[2], array[4],
                                array[1], array[3], array[5],
                                0, 0, 1);
                    }
                }
            }

            transform.premultiply(currentTransform);
        }
    }

    return transform;
}

ShapePath SVGLoader::Impl::parseRectNode(const pugi::xml_node& node) const {

    const auto x = parseFloatWithUnits(node.attribute("x").as_string("0"));
    const auto y = parseFloatWithUnits(node.attribute("y").as_string("0"));
    const auto rx = parseFloatWithUnits(node.attribute("rx").as_string("0"));
    const auto ry = parseFloatWithUnits(node.attribute("ry").as_string("0"));
    const auto w = parseFloatWithUnits(node.attribute("width").as_string("0"));
    const auto h = parseFloatWithUnits(node.attribute("height").as_string("0"));

    ShapePath path;
    path.moveTo(x + 2 * rx, y);
    path.lineTo(x + w - 2 * rx, y);
    if (rx != 0 || ry != 0) path.bezierCurveTo(x + w, y, x + w, y, x + w, y + 2 * ry);
    path.lineTo(x + w, y + h - 2 * ry);
    if (rx != 0 || ry != 0) path.bezierCurveTo(x + w, y + h, x + w, y + h, x + w - 2 * rx, y + h);
    path.lineTo(x + 2 * rx, y + h);

    if (rx != 0 || ry != 0) {

        path.bezierCurveTo(x, y + h, x, y + h, x, y + h - 2 * ry);
    }

    path.lineTo(x, y + 2 * ry);

    if (rx != 0 || ry != 0) {

        path.bezierCurveTo(x, y, x, y, x + 2 * rx, y);
    }

    return path;
}

ShapePath SVGLoader::Impl::parsePolygonNode(const pugi::xml_node& node) const {

    const static std::regex regex{R"((-?[+\d\.?]+)[,|\s](-?[\d\.?]+))"};

    ShapePath path;

    std::string points = node.attribute("points").value();

    int index = 0;
    for (auto it = std::sregex_iterator(points.begin(), points.end(), regex); it != std::sregex_iterator(); ++it) {

        const std::smatch& m = *it;

        std::string a = m[1].str();
        std::string b = m[2].str();

        const auto x = parseFloatWithUnits(a);
        const auto y = parseFloatWithUnits(b);

        if (index == 0) {

            path.moveTo(x, y);

        } else {

            path.lineTo(x, y);
        }

        ++index;
    }

    path.currentPath->autoClose = true;

    return path;
}

ShapePath SVGLoader::Impl::parsePolylineNode(const pugi::xml_node& node) const {

    const static std::regex regex{R"((-?[\d\.?]+)[,|\s](-?[\d\.?]+))"};

    ShapePath path;

    std::string points = node.attribute("points").value();

    int index = 0;
    for (auto it = std::sregex_iterator(points.begin(), points.end(), regex); it != std::sregex_iterator(); ++it) {

        const std::smatch& m = *it;

        std::string a = m[1].str();
        std::string b = m[2].str();

        const auto x = parseFloatWithUnits(a);
        const auto y = parseFloatWithUnits(b);

        if (index == 0) {

            path.moveTo(x, y);

        } else {

            path.lineTo(x, y);
        }

        ++index;
    }

    path.currentPath->autoClose = false;

    return path;
}

ShapePath SVGLoader::Impl::parseCircleNode(const pugi::xml_node& node) const {

    const auto x = parseFloatWithUnits(node.attribute("cx").as_string("0"));
    const auto y = parseFloatWithUnits(node.attribute("cy").as_string("0"));
    const auto r = parseFloatWithUnits(node.attribute("x").as_string("0"));

    auto subpath = std::make_shared<Path>();
    subpath->absarc(x, y, r, 0, math::PI * 2);

    ShapePath path;
    path.subPaths.emplace_back(subpath);

    return path;
}

ShapePath SVGLoader::Impl::parseEllipseNode(const pugi::xml_node& node) const {

    const auto x = parseFloatWithUnits(node.attribute("cx").as_string("0"));
    const auto y = parseFloatWithUnits(node.attribute("cy").as_string("0"));
    const auto rx = parseFloatWithUnits(node.attribute("rx").as_string("0"));
    const auto ry = parseFloatWithUnits(node.attribute("ry").as_string("0"));

    auto subpath = std::make_shared<Path>();
    subpath->absellipse(x, y, rx, ry, 0, math::PI * 2);

    ShapePath path;
    path.subPaths.emplace_back(subpath);

    return path;
}

ShapePath SVGLoader::Impl::parseLineNode(const pugi::xml_node& node) const {

    const auto x1 = parseFloatWithUnits(node.attribute("x1").as_string("0"));
    const auto y1 = parseFloatWithUnits(node.attribute("y1").as_string("0"));
    const auto x2 = parseFloatWithUnits(node.attribute("x2").as_string("0"));
    const auto y2 = parseFloatWithUnits(node.attribute("y2").as_string("0"));

    ShapePath path;
    path.moveTo(x1, y1);
    path.lineTo(x2, y2);
    path.currentPath->autoClose = false;

    return path;
}

SVGLoader::Style SVGLoader::Impl::parseStyle(const pugi::xml_node& node, SVGLoader::Style style) const {

    std::unordered_map<std::string, std::string> stylesheetStyles;

    if (node.attribute("class")) {

        //                static std::regex r("\\s");
        //
        //                auto classSelectors = regexSplit(node.attribute("class").value(), r);
        //                for (auto& str : classSelectors) {
        //                    utils::trimInplace(str);
        //                }
    }

    if (node.attribute("id")) {
        style.id = node.attribute("id").value();
        //                    stylesheetStyles = Object.assign( stylesheetStyles, stylesheets[ '#' + node.getAttribute( 'id' ) ] );
    }

    if (node.attribute("style")) {
        auto a = node.attribute("style");
        auto components = utils::split(a.as_string(), ';');
        for (const auto& c : components) {
            auto value = utils::split(utils::trim(c), ':');
            if (value.size() == 2) {

                auto key = utils::trim(value[0]);
                auto strValue = utils::trim(value[1]);

                if (key == "fill") {
                    style.fill = strValue;
                } else if (key == "fill-rule") {
                    style.fillRule = strValue;
                } else if (key == "fill-opacity") {
                    style.fillOpacity = clamp(std::stof(strValue));
                } else if (key == "stroke") {
                    style.stroke = strValue;
                } else if (key == "stroke-opacity") {
                    style.strokeOpacity = clamp(std::stof(strValue));
                } else if (key == "stroke-linejoin") {
                    style.strokeLineJoin = strValue;
                } else if (key == "stroke-linecap") {
                    style.strokeLineCap = strValue;
                } else if (key == "stroke-width") {
                    style.strokeWidth = positive(std::stof(strValue));
                } else if (key == "stroke-miter-limit") {
                    style.strokeMiterLimit = positive(std::stof(strValue));
                } else if (key == "visibility") {
                    style.visibility = strValue == "true";
                }
            }
        }
    }

    if (node.attribute("fill")) {
        style.fill = node.attribute("fill").as_string();
    }
    if (node.attribute("fill-rule")) {
        style.fillRule = node.attribute("fill-rule").as_string();
    }
    if (node.attribute("stroke")) {
        style.stroke = node.attribute("stroke").as_string();
    }
    if (node.attribute("stroke-line-join")) {
        style.strokeLineJoin = node.attribute("stroke-line-join").as_string();
    }
    if (node.attribute("stroke-line-cap")) {
        style.strokeLineCap = node.attribute("stroke-line-cap").as_string();
    }

    if (node.attribute("opacity")) {
        style.opacity = clamp(node.attribute("opacity").as_float());
    }
    if (node.attribute("fill-opacity")) {
        style.fillOpacity = clamp(node.attribute("fill-opacity").as_float());
    }
    if (node.attribute("stroke-opacity")) {
        style.strokeOpacity = clamp(node.attribute("stroke-opacity").as_float());
    }
    if (node.attribute("stroke-width")) {
        style.strokeWidth = positive(node.attribute("stroke-width").as_float());
    }
    if (node.attribute("stroke-miter-limit")) {
        style.strokeMiterLimit = positive(node.attribute("stroke-miter-limit").as_float());
    }

    if (node.attribute("visibility")) {
        style.visibility = node.attribute("visibility").as_bool();
    }

    return style;
}

ShapePath parsePathNode(const pugi::xml_node& node) {

    ShapePath path;

    Vector2 point;
    Vector2 control;

    Vector2 firstPoint;
    bool isFirstPoint = true;
    bool doSetFirstPoint = false;

    const std::string d = node.attribute("d").value();

    const static std::regex r{"[a-df-z][^a-df-z]*", std::regex::icase};

    for (std::sregex_iterator i = std::sregex_iterator(d.begin(), d.end(), r);
         i != std::sregex_iterator(); ++i) {

        const std::smatch& m = *i;

        std::string command = m.str();

        const auto type = command[0];
        const auto data = utils::trim(command.substr(1));

        if (isFirstPoint) {

            doSetFirstPoint = true;
            isFirstPoint = false;
        }

        std::vector<float> numbers;

        switch (type) {

            case 'M':
                numbers = svg::parseFloats(data);
                for (unsigned j = 0, jl = numbers.size(); j < jl; j += 2) {

                    point.x = numbers[j + 0];
                    point.y = numbers[j + 1];
                    control.x = point.x;
                    control.y = point.y;

                    if (j == 0) {

                        path.moveTo(point.x, point.y);

                    } else {

                        path.lineTo(point.x, point.y);
                    }

                    if (j == 0 && doSetFirstPoint) firstPoint.copy(point);
                }

                break;

            case 'H':
                numbers = svg::parseFloats(data);

                for (unsigned j = 0, jl = numbers.size(); j < jl; j++) {

                    point.x = numbers[j];
                    control.x = point.x;
                    control.y = point.y;
                    path.lineTo(point.x, point.y);

                    if (j == 0 && doSetFirstPoint) firstPoint.copy(point);
                }

                break;

            case 'V':
                numbers = svg::parseFloats(data);

                for (unsigned j = 0, jl = numbers.size(); j < jl; j++) {

                    point.y = numbers[j];
                    control.x = point.x;
                    control.y = point.y;
                    path.lineTo(point.x, point.y);

                    if (j == 0 && doSetFirstPoint) firstPoint.copy(point);
                }

                break;

            case 'L':
                numbers = svg::parseFloats(data);

                for (unsigned j = 0, jl = numbers.size(); j < jl; j += 2) {

                    point.x = numbers[j + 0];
                    point.y = numbers[j + 1];
                    control.x = point.x;
                    control.y = point.y;
                    path.lineTo(point.x, point.y);

                    if (j == 0 && doSetFirstPoint) firstPoint.copy(point);
                }

                break;

            case 'C':
                numbers = svg::parseFloats(data);

                for (unsigned j = 0, jl = numbers.size(); j < jl; j += 6) {

                    path.bezierCurveTo(
                            numbers[j + 0],
                            numbers[j + 1],
                            numbers[j + 2],
                            numbers[j + 3],
                            numbers[j + 4],
                            numbers[j + 5]);
                    control.x = numbers[j + 2];
                    control.y = numbers[j + 3];
                    point.x = numbers[j + 4];
                    point.y = numbers[j + 5];

                    if (j == 0 && doSetFirstPoint) firstPoint.copy(point);
                }

                break;

            case 'S':
                numbers = svg::parseFloats(data);

                for (unsigned j = 0, jl = numbers.size(); j < jl; j += 4) {

                    path.bezierCurveTo(
                            svg::getReflection(point.x, control.x),
                            svg::getReflection(point.y, control.y),
                            numbers[j + 0],
                            numbers[j + 1],
                            numbers[j + 2],
                            numbers[j + 3]);
                    control.x = numbers[j + 0];
                    control.y = numbers[j + 1];
                    point.x = numbers[j + 2];
                    point.y = numbers[j + 3];

                    if (j == 0 && doSetFirstPoint) firstPoint.copy(point);
                }

                break;

            case 'Q':
                numbers = svg::parseFloats(data);

                for (unsigned j = 0, jl = numbers.size(); j < jl; j += 4) {

                    path.quadraticCurveTo(
                            numbers[j + 0],
                            numbers[j + 1],
                            numbers[j + 2],
                            numbers[j + 3]);
                    control.x = numbers[j + 0];
                    control.y = numbers[j + 1];
                    point.x = numbers[j + 2];
                    point.y = numbers[j + 3];

                    if (j == 0 && doSetFirstPoint) firstPoint.copy(point);
                }

                break;

            case 'T':
                numbers = svg::parseFloats(data);

                for (unsigned j = 0, jl = numbers.size(); j < jl; j += 2) {

                    const auto rx = svg::getReflection(point.x, control.x);
                    const auto ry = svg::getReflection(point.y, control.y);
                    path.quadraticCurveTo(
                            rx,
                            ry,
                            numbers[j + 0],
                            numbers[j + 1]);
                    control.x = rx;
                    control.y = ry;
                    point.x = numbers[j + 0];
                    point.y = numbers[j + 1];

                    if (j == 0 && doSetFirstPoint) firstPoint.copy(point);
                }

                break;

            case 'A':
                numbers = svg::parseFloats(data, {3, 4}, 7);

                for (unsigned j = 0, jl = numbers.size(); j < jl; j += 7) {

                    // skip command if start point == end point
                    if (numbers[j + 5] == point.x && numbers[j + 6] == point.y) continue;

                    const auto start = point.clone();
                    point.x = numbers[j + 5];
                    point.y = numbers[j + 6];
                    control.x = point.x;
                    control.y = point.y;
                    svg::parseArcCommand(
                            path, numbers[j], numbers[j + 1], numbers[j + 2], numbers[j + 3], numbers[j + 4], start, point);

                    if (j == 0 && doSetFirstPoint) firstPoint.copy(point);
                }

                break;

            case 'm':
                numbers = svg::parseFloats(data);

                for (unsigned j = 0, jl = numbers.size(); j < jl; j += 2) {

                    point.x += numbers[j + 0];
                    point.y += numbers[j + 1];
                    control.x = point.x;
                    control.y = point.y;

                    if (j == 0) {

                        path.moveTo(point.x, point.y);

                    } else {

                        path.lineTo(point.x, point.y);
                    }

                    if (j == 0 && doSetFirstPoint) firstPoint.copy(point);
                }

                break;

            case 'h':
                numbers = svg::parseFloats(data);

                for (unsigned j = 0, jl = numbers.size(); j < jl; j++) {

                    point.x += numbers[j];
                    control.x = point.x;
                    control.y = point.y;
                    path.lineTo(point.x, point.y);

                    if (j == 0 && doSetFirstPoint) firstPoint.copy(point);
                }

                break;

            case 'v':
                numbers = svg::parseFloats(data);

                for (unsigned j = 0, jl = numbers.size(); j < jl; j++) {

                    point.y += numbers[j];
                    control.x = point.x;
                    control.y = point.y;
                    path.lineTo(point.x, point.y);

                    if (j == 0 && doSetFirstPoint) firstPoint.copy(point);
                }

                break;

            case 'l':
                numbers = svg::parseFloats(data);

                for (unsigned j = 0, jl = numbers.size(); j < jl; j += 2) {

                    point.x += numbers[j + 0];
                    point.y += numbers[j + 1];
                    control.x = point.x;
                    control.y = point.y;
                    path.lineTo(point.x, point.y);

                    if (j == 0 && doSetFirstPoint) firstPoint.copy(point);
                }

                break;

            case 'c':
                numbers = svg::parseFloats(data);

                for (unsigned j = 0, jl = numbers.size(); j < jl; j += 6) {

                    path.bezierCurveTo(
                            point.x + numbers[j + 0],
                            point.y + numbers[j + 1],
                            point.x + numbers[j + 2],
                            point.y + numbers[j + 3],
                            point.x + numbers[j + 4],
                            point.y + numbers[j + 5]);
                    control.x = point.x + numbers[j + 2];
                    control.y = point.y + numbers[j + 3];
                    point.x += numbers[j + 4];
                    point.y += numbers[j + 5];

                    if (j == 0 && doSetFirstPoint) firstPoint.copy(point);
                }

                break;

            case 's':
                numbers = svg::parseFloats(data);

                for (unsigned j = 0, jl = numbers.size(); j < jl; j += 4) {

                    path.bezierCurveTo(
                            svg::getReflection(point.x, control.x),
                            svg::getReflection(point.y, control.y),
                            point.x + numbers[j + 0],
                            point.y + numbers[j + 1],
                            point.x + numbers[j + 2],
                            point.y + numbers[j + 3]);
                    control.x = point.x + numbers[j + 0];
                    control.y = point.y + numbers[j + 1];
                    point.x += numbers[j + 2];
                    point.y += numbers[j + 3];

                    if (j == 0 && doSetFirstPoint) firstPoint.copy(point);
                }

                break;

            case 'q':
                numbers = svg::parseFloats(data);

                for (unsigned j = 0, jl = numbers.size(); j < jl; j += 4) {

                    path.quadraticCurveTo(
                            point.x + numbers[j + 0],
                            point.y + numbers[j + 1],
                            point.x + numbers[j + 2],
                            point.y + numbers[j + 3]);
                    control.x = point.x + numbers[j + 0];
                    control.y = point.y + numbers[j + 1];
                    point.x += numbers[j + 2];
                    point.y += numbers[j + 3];

                    if (j == 0 && doSetFirstPoint) firstPoint.copy(point);
                }

                break;

            case 't':
                numbers = svg::parseFloats(data);

                for (unsigned j = 0, jl = numbers.size(); j < jl; j += 2) {

                    const auto rx = svg::getReflection(point.x, control.x);
                    const auto ry = svg::getReflection(point.y, control.y);
                    path.quadraticCurveTo(
                            rx,
                            ry,
                            point.x + numbers[j + 0],
                            point.y + numbers[j + 1]);
                    control.x = rx;
                    control.y = ry;
                    point.x = point.x + numbers[j + 0];
                    point.y = point.y + numbers[j + 1];

                    if (j == 0 && doSetFirstPoint) firstPoint.copy(point);
                }

                break;

            case 'a':
                numbers = svg::parseFloats(data, {3, 4}, 7);

                for (unsigned j = 0, jl = numbers.size(); j < jl; j += 7) {

                    // skip command if no displacement
                    if (numbers[j + 5] == 0 && numbers[j + 6] == 0) continue;

                    const auto start = point.clone();
                    point.x += numbers[j + 5];
                    point.y += numbers[j + 6];
                    control.x = point.x;
                    control.y = point.y;
                    svg::parseArcCommand(
                            path, numbers[j], numbers[j + 1], numbers[j + 2], numbers[j + 3], numbers[j + 4], start, point);

                    if (j == 0 && doSetFirstPoint) firstPoint.copy(point);
                }

                break;

            case 'Z':
            case 'z':
                path.currentPath->autoClose = true;

                if (!path.currentPath->curves.empty()) {

                    // Reset point to beginning of Path
                    point.copy(firstPoint);
                    path.currentPath->currentPoint.copy(point);
                    isFirstPoint = true;
                }

                break;

            default:
                std::cerr << command << std::endl;
        }

        doSetFirstPoint = false;
    }

    return path;
}

std::shared_ptr<BufferGeometry> pointsToStroke(const std::vector<Vector2>& points, const SVGLoader::Style& style, unsigned int arcDivisions, float minDistance) {

    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> uvs;

    if (pointsToStrokeWithBuffers(points, style, arcDivisions, minDistance, vertices, normals, uvs) == 0) {

        return nullptr;
    }

    auto geometry = BufferGeometry::create();
    geometry->setAttribute("position", FloatBufferAttribute::create(vertices, 3));
    geometry->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
    geometry->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));

    return geometry;
}
