
#include "threepp/loaders/SVGLoader.hpp"
#include "threepp/extras/core/ShapePath.hpp"
#include "threepp/geometries/ShapeGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/math/Box2.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"

#include <iostream>

#define NANOSVG_ALL_COLOR_KEYWORDS
#define NANOSVG_IMPLEMENTATION
#include "nanosvg.h"

using namespace threepp;

namespace {

    Color getColor(NSVGpaint paint) {

        Color c = paint.color;
        return c.setRGB(c.b, c.g, c.r);
    }

    enum class IntersectionLocationType {
        ORIGIN,
        DESTINATION,
        BETWEEN,
        LEFT,
        RIGHT,
        BEHIND,
        BEYOND
    };

    struct ClassifyResult {
        IntersectionLocationType loc = IntersectionLocationType::ORIGIN;
        float t = 0;
    } classifyResult;

    struct EdgeIntersection {
        float x;
        float y;
        float t;
    };

    struct Intersection {
        std::string identifier;
        bool isCW;
        Vector2 point;
    };

    std::string getStrokeLineCap(char c) {
        switch (c) {
            case NSVG_CAP_BUTT:
                return "butt";
            case NSVG_CAP_ROUND:
                return "round";
            case NSVG_CAP_SQUARE:
                return "square";
            default:
                return "butt";
        }
    }


    std::string getStrokeLineJoin(char c) {
        switch (c) {
            case NSVG_JOIN_BEVEL:
                return "bevel";
            case NSVG_JOIN_MITER:
                return "miter";
            case NSVG_JOIN_ROUND:
                return "round";
            default:
                return "miter";
        }
    }

    std::vector<SVGLoader::Shape> loadSVG(NSVGimage* image) {

        std::vector<SVGLoader::Shape> shapes;
        for (auto shape = image->shapes; shape != nullptr; shape = shape->next) {

            if (!(shape->flags & NSVG_FLAGS_VISIBLE)) continue;
            if (shape->fill.type == NSVG_PAINT_NONE) continue;

            auto& data = shapes.emplace_back();

            data.id = shape->id;
            data.style.fillOpacity = shape->opacity;
            data.style.strokeOpacity = shape->opacity;
            data.style.strokeWidth = shape->strokeWidth;
            data.style.strokeLineCap = getStrokeLineCap(shape->strokeLineCap);
            data.style.strokeLineJoin = getStrokeLineJoin(shape->strokeLineJoin);
            data.style.strokeMiterLimit = shape->miterLimit;

            if (shape->stroke.type != NSVG_PAINT_NONE) {
                data.style.stroke = getColor(shape->stroke);
            }

            ShapePath& s = data.path;
            for (auto path = shape->paths; path != nullptr; path = path->next) {

                s.color = getColor(shape->fill);

                for (unsigned i = 0; i < path->npts - 1; i += 3) {
                    float* p = &path->pts[i * 2];

                    if (i == 0) {
                        s.moveTo(p[0], p[1]);
                    }

                    s.bezierCurveTo(p[2], p[3], p[4], p[5], p[6], p[7]);
                }
            }
        }

        return shapes;
    }

    void classifyPoint(const Vector2& p, const Vector2& edgeStart, const Vector2& edgeEnd) {

        const auto ax = edgeEnd.x - edgeStart.x;
        const auto ay = edgeEnd.y - edgeStart.y;
        const auto bx = p.x - edgeStart.x;
        const auto by = p.y - edgeStart.y;
        const auto sa = ax * by - bx * ay;

        if ((p.x == edgeStart.x) && (p.y == edgeStart.y)) {

            classifyResult.loc = IntersectionLocationType::ORIGIN;
            classifyResult.t = 0;
            return;
        }

        if ((p.x == edgeEnd.x) && (p.y == edgeEnd.y)) {

            classifyResult.loc = IntersectionLocationType::DESTINATION;
            classifyResult.t = 1;
            return;
        }

        if (sa < -std::numeric_limits<float>::epsilon()) {

            classifyResult.loc = IntersectionLocationType::LEFT;
            return;
        }

        if (sa > std::numeric_limits<float>::epsilon()) {

            classifyResult.loc = IntersectionLocationType::RIGHT;
            return;
        }

        if (((ax * bx) < 0) || ((ay * by) < 0)) {

            classifyResult.loc = IntersectionLocationType::BEHIND;
            return;
        }

        if ((std::sqrt(ax * ax + ay * ay)) < (std::sqrt(bx * bx + by * by))) {

            classifyResult.loc = IntersectionLocationType::BEYOND;
            return;
        }

        float t;

        if (ax != 0) {

            t = bx / ax;

        } else {

            t = by / ay;
        }

        classifyResult.loc = IntersectionLocationType::BETWEEN;
        classifyResult.t = t;
    }

    std::optional<EdgeIntersection> findEdgeIntersection(const Vector2& a0, const Vector2& a1, const Vector2& b0, const Vector2& b1) {
        const auto x1 = a0.x;
        const auto x2 = a1.x;
        const auto x3 = b0.x;
        const auto x4 = b1.x;
        const auto y1 = a0.y;
        const auto y2 = a1.y;
        const auto y3 = b0.y;
        const auto y4 = b1.y;
        const auto nom1 = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3);
        const auto nom2 = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3);
        const auto denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1);
        const auto t1 = nom1 / denom;
        const auto t2 = nom2 / denom;

        if (((denom == 0) && (nom1 != 0)) || (t1 <= 0) || (t1 >= 1) || (t2 < 0) || (t2 > 1)) {

            //1. lines are parallel or edges don't intersect

            return std::nullopt;

        } else if ((nom1 == 0) && (denom == 0)) {

            //2. lines are colinear

            //check if endpoints of edge2 (b0-b1) lies on edge1 (a0-a1)
            for (unsigned i = 0; i < 2; i++) {

                classifyPoint(i == 0 ? b0 : b1, a0, a1);
                //find position of this endpoints relatively to edge1
                if (classifyResult.loc == IntersectionLocationType::ORIGIN) {

                    const auto& point = (i == 0 ? b0 : b1);
                    return EdgeIntersection{point.x, point.y, classifyResult.t};

                } else if (classifyResult.loc == IntersectionLocationType::BETWEEN) {

                    const auto x = +((x1 + classifyResult.t * (x2 - x1)));
                    const auto y = +((y1 + classifyResult.t * (y2 - y1)));
                    return EdgeIntersection{
                            x,
                            y,
                            classifyResult.t,
                    };
                }
            }

            return std::nullopt;

        } else {

            //3. edges intersect

            for (unsigned i = 0; i < 2; i++) {

                classifyPoint(i == 0 ? b0 : b1, a0, a1);

                if (classifyResult.loc == IntersectionLocationType::ORIGIN) {

                    const auto& point = (i == 0 ? b0 : b1);
                    return EdgeIntersection{point.x, point.y, classifyResult.t};
                }
            }

            const auto x = +((x1 + t1 * (x2 - x1)));
            const auto y = +((y1 + t1 * (y2 - y1)));
            return EdgeIntersection{x, y, t1};
        }
    }


    std::vector<Vector2> getIntersections(const std::vector<Vector2>& path1, const std::vector<Vector2>& path2) {

        std::vector<EdgeIntersection> intersectionsRaw;
        std::vector<Vector2> intersections;

        for (unsigned index = 1; index < path1.size(); index++) {

            const auto& path1EdgeStart = path1[index - 1];
            const auto& path1EdgeEnd = path1[index];

            for (unsigned index2 = 1; index2 < path2.size(); index2++) {

                const auto& path2EdgeStart = path2[index2 - 1];
                const auto& path2EdgeEnd = path2[index2];

                const auto intersection = findEdgeIntersection(path1EdgeStart, path1EdgeEnd, path2EdgeStart, path2EdgeEnd);

                if (intersection && std::find_if(intersectionsRaw.begin(), intersectionsRaw.end(), [&](auto& i) {
                                        return i.t <= intersection->t + std::numeric_limits<float>::epsilon() &&
                                               i.t >= intersection->t - std::numeric_limits<float>::epsilon();
                                    }) != intersectionsRaw.end()) {

                    intersectionsRaw.emplace_back(*intersection);
                    intersections.emplace_back(intersection->x, intersection->y);
                }
            }
        }

        return intersections;
    }

    //    std::vector<Intersection> getScanlineIntersections(const std::vector<Vector2>& scanline, const Box2& boundingBox, paths) {
    //
    //        Vector2 center;
    //        boundingBox.getCenter(center);
    //
    //        std::vector<Intersection> allIntersections;
    //
    //        for (const auto& path : paths) {
    //
    //            // check if the center of the bounding box is in the bounding box of the paths.
    //            // this is a pruning method to limit the search of intersections in paths that can't envelop of the current path.
    //            // if a path envelops another path. The center of that oter path, has to be inside the bounding box of the enveloping path.
    //            if (path.boundingBox.containsPoint(center)) {
    //
    //                const auto intersections = getIntersections(scanline, path.points);
    //
    //                for (const auto& p : intersections) {
    //
    //                    allIntersections.emplace_back({path.identifier, path.isCW, p});
    //                }
    //            }
    //        }
    //
    //        std::sort(allIntersections.begin(), allIntersections.end(), [](const auto& i1, const auto& i2) {
    //            return i1.point.x - i2.point.x;
    //        });
    //
    //        return allIntersections;
    //    }

    void removeDuplicatedPoints(std::vector<Vector2>& points, float minDistance) {

        // Creates a new array if necessary with duplicated points removed.
        // This does not remove duplicated initial and ending points of a closed path.

        bool dupPoints = false;
        for (unsigned i = 1, n = points.size() - 1; i < n; i++) {

            if (points[i].distanceTo(points[i + 1]) < minDistance) {

                dupPoints = true;
                break;
            }
        }

        if (!dupPoints) return;

        std::vector<Vector2> newPoints;
        newPoints.emplace_back(points[0]);

        for (unsigned i = 1, n = points.size() - 1; i < n; i++) {

            if (points[i].distanceTo(points[i + 1]) >= minDistance) {

                newPoints.emplace_back(points[i]);
            }
        }

        newPoints.emplace_back(points[points.size() - 1]);

        points = newPoints;
    }

    Vector2& getNormal(const Vector2& p1, const Vector2& p2, Vector2& result) {

        result.subVectors(p2, p1);
        return result.set(-result.y, result.x).normalize();
    }

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

std::vector<SVGLoader::Shape> SVGLoader::load(const std::filesystem::path& filePath) {

    struct NSVGimage* image;
    image = nsvgParseFromFile(filePath.string().c_str(), defaultUnit.c_str(), defaultDPI);

    auto data = loadSVG(image);
    nsvgDelete(image);
    return data;
}

std::vector<SVGLoader::Shape> SVGLoader::parse(std::string text) {

    struct NSVGimage* image;
    image = nsvgParse(const_cast<char*>(text.c_str()), defaultUnit.c_str(), defaultDPI);

    auto data = loadSVG(image);
    nsvgDelete(image);
    return data;
}

std::unique_ptr<threepp::Shape> SVGLoader::createShapes(const ShapePath& shapePath) {

    // Param shapePath: a shapepath as returned by the parse function of this class
    // Returns Shape object

    const auto BIGNUMBER = 999999999;


    return nullptr;
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
