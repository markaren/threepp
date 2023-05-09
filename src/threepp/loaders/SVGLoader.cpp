
#include "threepp/loaders/SVGLoader.hpp"
#include "threepp/extras/ShapeUtils.hpp"
#include "threepp/extras/curves/CubicBezierCurve.hpp"
#include "threepp/extras/curves/EllipseCurve.hpp"
#include "threepp/extras/curves/LineCurve.hpp"
#include "threepp/extras/curves/QuadraticBezierCurve.hpp"
#include "threepp/geometries/ShapeGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/math/Box2.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"

#include "threepp/utils/RegexUtil.hpp"
#include "threepp/utils/StringUtils.hpp"

#include "threepp/loaders/svg/SVGFunctions.hpp"

#include "pugixml.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace threepp;
using namespace threepp::svg;

namespace {

    std::vector<std::string> units{"mm", "cm", "in", "pt", "pc", "px"};

    std::unordered_map<std::string, std::unordered_map<std::string, float>> unitConversion{
            {"mm", {{"mm", 1.f}, {"cm", 0.1f}, {"in", 1.f / 25.4f}, {"pt", 72.f / 25.4f}, {"pc", 6.f / 25.4f}, {"px", -1.f}}},
            {"cm", {{"mm", 10.f}, {"cm", 1.f}, {"in", 1.f / 2.54f}, {"pt", 72.f / 2.54f}, {"pc", 6.f / 2.54f}, {"px", -1.f}}},
            {"in", {{"mm", 25.4f}, {"cm", 2.54f}, {"in", 1.f}, {"pt", 72.f}, {"pc", 6.f}, {"px", -1.f}}},
            {"pt", {{"mm", 25.4f / 72.f}, {"cm", 2.54f / 72.f}, {"in", 1.f / 72.f}, {"pt", 1.f}, {"pc", 6.f / 72.f}, {"px", -1.f}}},
            {"pc", {{"mm", 25.4f / 6.f}, {"cm", 2.54f / 6.f}, {"in", 1.f / 6.f}, {"pt", 72.f / 6.f}, {"pc", 1.f}, {"px", -1.f}}},
            {"px", {{"px", 1.f}}}};

    unsigned int pointsToStrokeWithBuffers(
            std::vector<Vector2> points,
            const SVGLoader::Style& style,
            unsigned int arcDivisions,
            float minDistance,
            std::vector<float>& vertices,
            std::vector<float>& normals,
            std::vector<float>& uvs,
            unsigned int vertexOffset = 0) {

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

        removeDuplicatedPoints(points, minDistance);

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
        getNormal(points[0], points[1], tempV2_1).multiplyScalar(strokeWidth2);
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
            getNormal(previousPoint, currentPoint, normal1);

            tempV2_3.copy(normal1).multiplyScalar(strokeWidth2);
            currentPointL.copy(currentPoint).sub(tempV2_3);
            currentPointR.copy(currentPoint).add(tempV2_3);

            u1 = u0 + deltaU;

            innerSideModified = false;

            if (nextPoint) {

                // Normal of next segment in tempV2_2
                getNormal(currentPoint, *nextPoint, tempV2_2);

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


}// namespace

struct SVGLoader::Impl {

    std::vector<SVGLoader::SVGData> paths;
    std::unordered_map<std::string, std::string> styleSheets;

    std::vector<Matrix3> transformStack;

    Matrix3 tempTransform0;
    Matrix3 tempTransform1;
    Matrix3 tempTransform2;
    Matrix3 tempTransform3;
    Vector2 tempV2;
    Vector3 tempV3;

    Matrix3 currentTransform;

    explicit Impl(SVGLoader& scope)
        : scope(scope) {}

    std::vector<SVGLoader::SVGData> load(const pugi::xml_node& node) {

        paths.clear();
        styleSheets.clear();
        transformStack.clear();

        parseNode(node.child("svg"), {"#000", 1, 1, 1, "miter", "butt", 4});

        return paths;
    }

    float clamp(float v) {

        return std::max(0.f, std::min(1.f, parseFloatWithUnits(v)));
    }

    float positive(float v) {

        return std::max(0.f, parseFloatWithUnits(v));
    }

    float parseFloatWithUnits(float value, const std::string& theUnit = "px") {

        float scale;

        if (theUnit == "px" && scope.defaultUnit != "px") {

            // Conversion scale from  pixels to inches, then to default units

            scale = unitConversion["in"][scope.defaultUnit] / scope.defaultDPI;

        } else {

            scale = unitConversion[theUnit][scope.defaultUnit];

            if (scale < 0) {

                // Conversion scale to pixels

                scale = unitConversion[theUnit]["in"] * scope.defaultDPI;
            }
        }

        return scale * value;
    }

    float parseFloatWithUnits(const std::string& str) {

        std::string theUnit = "px";

        float value;

        if (!utils::isNumber(str)) {

            for (unsigned i = 0, n = units.size(); i < n; i++) {

                const auto u = units[i];

                if (utils::endsWith(str, u)) {

                    theUnit = u;
                    value = std::stof(str.substr(0, str.size() - u.size()));
                    break;
                } else {

                    throw std::runtime_error("THREE.SVGLoader: unexpected codepath in parseFloatWithUnits");
                }
            }

        } else {

            value = std::stof(str);
        }

        return parseFloatWithUnits(value, theUnit);
    }


    Matrix3 parseNodeTransform(const pugi::xml_node& node) {

        Matrix3 transform;
        Matrix3& currentTransform = tempTransform0;

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

                    const auto array = parseFloats(transformText.substr(openParPos + 1, closeParPos - openParPos - 1));

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
    };

    std::optional<Matrix3> getNodeTransform(const pugi::xml_node& node) {

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

    /*
    * According to https://www.w3.org/TR/SVG/shapes.html#RectElementRXAttribute
    * rounded corner should be rendered to elliptical arc, but bezier curve does the job well enough
    */
    ShapePath parseRectNode(const pugi::xml_node& node) {

        const auto x = parseFloatWithUnits(node.attribute("x").as_float(0));
        const auto y = parseFloatWithUnits(node.attribute("y").as_float(0));
        const auto rx = parseFloatWithUnits(node.attribute("rx").as_float(0));
        const auto ry = parseFloatWithUnits(node.attribute("ry").as_float(0));
        const auto w = parseFloatWithUnits(node.attribute("width").value());
        const auto h = parseFloatWithUnits(node.attribute("height").value());

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

    ShapePath parsePolygonNode(const pugi::xml_node& node) {

        //                int index = 0;

        //                auto iterator = ( match, a, b ) {
        //
        //                    const x = parseFloatWithUnits( a );
        //                    const y = parseFloatWithUnits( b );
        //
        //                    if ( index === 0 ) {
        //
        //                        path.moveTo( x, y );
        //
        //                    } else {
        //
        //                        path.lineTo( x, y );
        //
        //                    }
        //
        //                    index ++;
        //
        //                }

        const static std::regex regex{R"((-?[+\d\.?]+)[,|\s](-?[\d\.?]+))"};

        ShapePath path;

        std::string points = node.attribute("points").value();

        int index = 0;
        for (auto it = std::sregex_iterator(points.begin(), points.end(), regex); it != std::sregex_iterator(); ++it) {
            std::smatch m = *it;

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

    ShapePath parsePolylineNode(const pugi::xml_node& node) {

        const std::regex regex{R"((-?[\d\.?]+)[,|\s](-?[\d\.?]+))"};

        ShapePath path;

        std::string points = node.attribute("points").value();

        int index = 0;
        for (auto it = std::sregex_iterator(points.begin(), points.end(), regex); it != std::sregex_iterator(); ++it) {
            std::smatch m = *it;

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

    ShapePath parseCircleNode(const pugi::xml_node& node) {

        const auto x = parseFloatWithUnits(node.attribute("cx").as_float(0));
        const auto y = parseFloatWithUnits(node.attribute("cy").as_float(0));
        const auto r = parseFloatWithUnits(node.attribute("x").as_float(0));

        auto subpath = std::make_shared<Path>();
        subpath->absarc(x, y, r, 0, math::PI * 2);

        ShapePath path;
        path.subPaths.emplace_back(subpath);

        return path;
    }

    ShapePath parseEllipseNode(const pugi::xml_node& node) {

        const auto x = parseFloatWithUnits(node.attribute("cx").as_float(0));
        const auto y = parseFloatWithUnits(node.attribute("cy").as_float(0));
        const auto rx = parseFloatWithUnits(node.attribute("rx").as_float(0));
        const auto ry = parseFloatWithUnits(node.attribute("ry").as_float(0));

        auto subpath = std::make_shared<Path>();
        subpath->absellipse(x, y, rx, ry, 0, math::PI * 2);

        ShapePath path;
        path.subPaths.emplace_back(subpath);

        return path;
    }

    ShapePath parseLineNode(const pugi::xml_node& node) {

        const auto x1 = parseFloatWithUnits(node.attribute("x1").as_float(0));
        const auto y1 = parseFloatWithUnits(node.attribute("y1").as_float(0));
        const auto x2 = parseFloatWithUnits(node.attribute("x2").as_float(0));
        const auto y2 = parseFloatWithUnits(node.attribute("y2").as_float(0));

        ShapePath path;
        path.moveTo(x1, y1);
        path.lineTo(x2, y2);
        path.currentPath->autoClose = false;

        return path;
    }

    Style parseStyle(const pugi::xml_node& node, Style style) {

        std::unordered_map<std::string, std::string> stylesheetStyles;

        if (node.attribute("class")) {

            static std::regex r("\\s");

            auto classSelectors = regexSplit(node.attribute("class").value(), r);
            for (auto& str : classSelectors) {
                utils::trimInplace(str);
            }
        }

        //        if ( node.attribute( "id" ) ) {
        //
        //            stylesheetStyles = Object.assign( stylesheetStyles, stylesheets[ '#' + node.getAttribute( 'id' ) ] );
        //
        //        }

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
                    } else if (key == "stroke") {
                        style.stroke = strValue;
                    } else if (key == "strokeLineJoin") {
                        style.strokeLineCap = strValue;
                    } else if (key == "strokeLineCap") {
                        style.strokeLineCap = strValue;
                    } else if (key == "stroke-width") {
                        style.strokeWidth = std::stof(strValue);
                    } else if (key == "strokeMiterLimit") {
                        style.strokeMiterLimit = std::stof(strValue);
                    }
                }
            }
        }

        if (node.attribute("fill")) {
            style.fill = node.attribute("fill").as_string();
        }
        if (node.attribute("stroke")) {
            style.stroke = node.attribute("stroke").as_string();
        }
        if (node.attribute("strokeLineJoin")) {
            style.strokeLineJoin = node.attribute("strokeLineJoin").as_string();
        }
        if (node.attribute("strokeLineCap")) {
            style.strokeLineCap = node.attribute("strokeLineCap").as_string();
        }

        if (node.attribute("fillOpacity")) {
            style.fillOpacity = clamp(node.attribute("fillOpacity").as_float());
        }
        if (node.attribute("opacity")) {
            style.opacity = clamp(node.attribute("opacity").as_float());
        }
        if (node.attribute("strokeWidth")) {
            style.strokeWidth = positive(node.attribute("strokeWidth").as_float());
        }
        if (node.attribute("strokeMiterLimit")) {
            style.strokeMiterLimit = positive(node.attribute("strokeMiterLimit").as_float());
        }

        if (node.attribute("visibility")) {
            style.fillOpacity = node.attribute("visibility").as_float();
        }

        return style;
    }

    void transformPath(ShapePath& path, const Matrix3& m) {

        auto transfVec2 = [&](Vector2& v2) {
            tempV3.set(v2.x, v2.y, 1).applyMatrix3(m);

            v2.set(tempV3.x, tempV3.y);
        };

        const auto isRotated = isTransformRotated(m);

        const auto& subPaths = path.subPaths;

        for (unsigned i = 0, n = subPaths.size(); i < n; i++) {

            const auto& subPath = subPaths[i];
            const auto& curves = subPath->curves;

            for (unsigned j = 0; j < curves.size(); j++) {

                const auto& curve = curves[j];

                if (auto lineCurve = std::dynamic_pointer_cast<LineCurve>(curve)) {

                    transfVec2(lineCurve->v1);
                    transfVec2(lineCurve->v2);

                } else if (auto cubicBezierCurve = std::dynamic_pointer_cast<CubicBezierCurve>(curve)) {

                    transfVec2(cubicBezierCurve->v0);
                    transfVec2(cubicBezierCurve->v1);
                    transfVec2(cubicBezierCurve->v2);
                    transfVec2(cubicBezierCurve->v3);

                } else if (auto quadraticBezierCurve = std::dynamic_pointer_cast<QuadraticBezierCurve>(curve)) {

                    transfVec2(quadraticBezierCurve->v0);
                    transfVec2(quadraticBezierCurve->v1);
                    transfVec2(quadraticBezierCurve->v2);

                } else if (auto ellipseCurve = std::dynamic_pointer_cast<EllipseCurve>(curve)) {

                    if (isRotated) {

                        std::cerr << "SVGLoader: Elliptic arc or ellipse rotation or skewing is not implemented." << std::endl;
                    }

                    tempV2.set(ellipseCurve->aX, ellipseCurve->aY);
                    transfVec2(tempV2);
                    ellipseCurve->aX = tempV2.x;
                    ellipseCurve->aY = tempV2.y;

                    ellipseCurve->xRadius *= getTransformScaleX(m);
                    ellipseCurve->yRadius *= getTransformScaleY(m);
                }
            }
        }
    }

    void parseNode(pugi::xml_node node, Style style) {

        if (node.type() != pugi::xml_node_type::node_element) return;

        const auto transform = getNodeTransform(node);

        bool traverseChildNodes = true;

        bool set = false;
        std::optional<ShapePath> path;

        std::string nodeName{node.name()};
        if (nodeName == "svg") {

        } else if (nodeName == "style") {

        } else if (nodeName == "g") {

            style = parseStyle(node, style);

        } else if (nodeName == "path") {

            style = parseStyle(node, style);
            if (node.attribute("d")) {
                path = parsePathNode(node);
                set = true;
            }

        } else if (nodeName == "rect") {

            style = parseStyle(node, style);

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

            transformPath(*path, currentTransform);

            paths.emplace_back(SVGLoader::SVGData{style, *path});
        }

        if (traverseChildNodes) {

            for (const auto& n : node.children()) {

                parseNode(n, style);
            }
        }
    }


private:
    SVGLoader& scope;
};

threepp::SVGLoader::SVGLoader()
    : pimpl_(std::make_unique<Impl>(*this)) {}

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

    const auto BIGNUMBER = 999999999;

    auto scanlineMinX = BIGNUMBER;
    auto scanlineMaxX = -BIGNUMBER;

    std::vector<SimplePath> simplePaths;
    const auto& subPaths = shapePath.subPaths;
    std::transform(subPaths.begin(), subPaths.end(), std::back_inserter(simplePaths), [&](const auto& p) {
        const auto& points = p->getPoints();
        auto maxY = -BIGNUMBER;
        auto minY = BIGNUMBER;
        auto maxX = -BIGNUMBER;
        auto minX = BIGNUMBER;

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

        return SimplePath{p->curves, points, shapeutils::isClockWise(points), -1, Box2({minX, minY}, {maxX, maxY})};
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

    std::vector<AHole> isAHole;
    std::transform(simplePaths.begin(), simplePaths.end(), std::back_inserter(isAHole), [&](const auto& p) {
        return isHoleTo(p, simplePaths, scanlineMinX, scanlineMaxX, data.style.fillRule);
    });

    std::vector<Shape> shapesToReturn;
    for (const auto& p : simplePaths) {

        const auto amIAHole = isAHole[p.identifier];

        if (!amIAHole.isHole) {

            Shape shape;
            shape.curves = p.curves;

            for (const auto& h : isAHole) {

                if (h.isHole && h._for == p.identifier) {

                    const auto& hole = simplePaths[h.identifier];
                    auto path = std::make_shared<Path>();
                    path->curves = hole.curves;
                    shape.holes.emplace_back(path);
                }
            }
            shapesToReturn.emplace_back(shape);
        }
    }

    return shapesToReturn;
}

std::shared_ptr<BufferGeometry> SVGLoader::pointsToStroke(const std::vector<Vector2>& points, const SVGLoader::Style& style, unsigned int arcDivisions, float minDistance) {

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

threepp::SVGLoader::~SVGLoader() = default;
