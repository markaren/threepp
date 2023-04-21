
#include "threepp/extras/core/ShapePath.hpp"
#include "threepp/extras/ShapeUtils.hpp"

#include <array>
#include <limits>
#include <memory>

using namespace threepp;

namespace {

    struct NewShape {

        std::shared_ptr<Shape> s;
        std::vector<Vector2> p;
    };

    struct NewShapeHoles {

        std::shared_ptr<Path> h;
        Vector2 p;
    };

    std::vector<std::shared_ptr<Shape>> toShapesNoHoles(const std::vector<std::shared_ptr<Path>>& inSubpaths) {

        std::vector<std::shared_ptr<Shape>> shapes;

        for (const auto& tmpPath : inSubpaths) {

            auto tmpShape = std::make_unique<Shape>();
            tmpShape->curves = tmpPath->curves;

            shapes.emplace_back(std::move(tmpShape));
        }

        return shapes;
    }

    bool isPointInsidePolygon(const Vector2& inPt, const std::vector<Vector2>& inPolygon) {

        const auto polyLen = inPolygon.size();

        // inPt on polygon contour => immediate success    or
        // toggling of inside/outside at every single! intersection point of an edge
        //  with the horizontal line through inPt, left of inPt
        //  not counting lowerY endpoints of edges and whole edges on that line
        bool inside = false;
        for (unsigned p = polyLen - 1, q = 0; q < polyLen; p = q++) {

            auto edgeLowPt = inPolygon[p];
            auto edgeHighPt = inPolygon[q];

            auto edgeDx = edgeHighPt.x - edgeLowPt.x;
            auto edgeDy = edgeHighPt.y - edgeLowPt.y;

            if (std::abs(edgeDy) > std::numeric_limits<float>::epsilon()) {

                // not parallel
                if (edgeDy < 0) {

                    edgeLowPt = inPolygon[q];
                    edgeDx = -edgeDx;
                    edgeHighPt = inPolygon[p];
                    edgeDy = -edgeDy;
                }

                if ((inPt.y < edgeLowPt.y) || (inPt.y > edgeHighPt.y)) continue;

                if (inPt.y == edgeLowPt.y) {

                    if (inPt.x == edgeLowPt.x) return true;// inPt is on contour ?
                                                           // continue;				// no intersection or edgeLowPt => doesn't count !!!

                } else {

                    const auto perpEdge = edgeDy * (inPt.x - edgeLowPt.x) - edgeDx * (inPt.y - edgeLowPt.y);
                    if (perpEdge == 0) return true;// inPt is on contour ?
                    if (perpEdge < 0) continue;
                    inside = !inside;// true intersection left of inPt
                }

            } else {

                // parallel or collinear
                if (inPt.y != edgeLowPt.y) continue;// parallel
                // edge lies on the same horizontal line as inPt
                if (((edgeHighPt.x <= inPt.x) && (inPt.x <= edgeLowPt.x)) ||
                    ((edgeLowPt.x <= inPt.x) && (inPt.x <= edgeHighPt.x))) return true;// inPt: Point on contour !
                                                                                       // continue;
            }
        }

        return inside;
    }

}// namespace


ShapePath& ShapePath::moveTo(float x, float y) {

    this->currentPath = std::make_shared<Path>();
    this->currentPath = this->subPaths.emplace_back(this->currentPath);
    this->currentPath->moveTo(x, y);

    return *this;
}

ShapePath& ShapePath::lineTo(float x, float y) {

    this->currentPath->lineTo(x, y);

    return *this;
}

ShapePath& ShapePath::quadraticCurveTo(float aCPx, float aCPy, float aX, float aY) {

    this->currentPath->quadraticCurveTo(aCPx, aCPy, aX, aY);

    return *this;
}

ShapePath& ShapePath::bezierCurveTo(float aCP1x, float aCP1y, float aCP2x, float aCP2y, float aX, float aY) {

    this->currentPath->bezierCurveTo(aCP1x, aCP1y, aCP2x, aCP2y, aX, aY);

    return *this;
}

ShapePath& ShapePath::splineThru(const std::vector<Vector2>& pts) {

    this->currentPath->splineThru(pts);

    return *this;
}

std::vector<std::shared_ptr<Shape>> ShapePath::toShapes(bool isCCW, bool noHoles) const {

    if (subPaths.empty()) return {};

    if (noHoles) return toShapesNoHoles(subPaths);


    bool solid;
    std::shared_ptr<Path> tmpPath;
    std::shared_ptr<Shape> tmpShape;
    std::vector<std::shared_ptr<Shape>> shapes;

    if (subPaths.size() == 1) {

        tmpPath = subPaths[0];
        tmpShape = std::make_unique<Shape>();
        tmpShape->curves = tmpPath->curves;
        shapes.emplace_back(std::move(tmpShape));
        return shapes;
    }

    bool holesFirst = !shapeutils::isClockWise(subPaths[0]->getPoints());
    holesFirst = isCCW ? !holesFirst : holesFirst;

    std::vector<std::vector<NewShapeHoles>> betterShapeHoles;
    std::vector<std::optional<NewShape>> newShapes;
    std::vector<std::vector<NewShapeHoles>> newShapeHoles;
    unsigned int mainIdx = 0;
    std::vector<Vector2> tmpPoints;

    newShapes.emplace_back();
    newShapeHoles.emplace_back();

    for (const auto& subPath : subPaths) {

        tmpPath = subPath;
        tmpPoints = tmpPath->getPoints();
        solid = shapeutils::isClockWise(tmpPoints);
        solid = isCCW ? !solid : solid;

        if (solid) {

            if ((!holesFirst) && (newShapes[mainIdx])) mainIdx++;

            newShapes.resize(mainIdx + 1);
            newShapes[mainIdx] = NewShape{std::make_shared<Shape>(), tmpPoints};
            newShapes[mainIdx]->s->curves = tmpPath->curves;

            if (holesFirst) mainIdx++;
            newShapeHoles.resize(mainIdx + 1);
            newShapeHoles[mainIdx] = {};

        } else {

            newShapeHoles[mainIdx].emplace_back(NewShapeHoles{tmpPath, tmpPoints[0]});
        }
    }

    // only Holes? -> probably all Shapes with wrong orientation
    if (!newShapes[0]) return toShapesNoHoles(subPaths);


    if (newShapes.size() > 1) {

        bool ambiguous = false;
        std::vector<std::array<unsigned int, 3>> toChange;

        for (unsigned sIdx = 0, sLen = newShapes.size(); sIdx < sLen; sIdx++) {

            betterShapeHoles.resize(sIdx + 1);
            betterShapeHoles[sIdx] = {};
        }

        for (unsigned sIdx = 0, sLen = newShapes.size(); sIdx < sLen; sIdx++) {

            const auto& sho = newShapeHoles[sIdx];

            for (unsigned hIdx = 0; hIdx < sho.size(); hIdx++) {

                const auto& ho = sho[hIdx];
                bool hole_unassigned = true;

                for (unsigned s2Idx = 0; s2Idx < newShapes.size(); s2Idx++) {

                    if (isPointInsidePolygon(ho.p, newShapes[s2Idx]->p)) {

                        if (sIdx != s2Idx) toChange.emplace_back(std::array<unsigned int, 3>{sIdx, s2Idx, hIdx});
                        if (hole_unassigned) {

                            hole_unassigned = false;
                            betterShapeHoles[s2Idx].emplace_back(ho);

                        } else {

                            ambiguous = true;
                        }
                    }
                }

                if (hole_unassigned) {

                    betterShapeHoles[sIdx].emplace_back(ho);
                }
            }
        }

        if (!toChange.empty()) {

            if (!ambiguous) newShapeHoles = betterShapeHoles;
        }
    }

    std::vector<NewShapeHoles> tmpHoles;

    for (unsigned i = 0, il = newShapes.size(); i < il; i++) {

        tmpShape = newShapes[i]->s;
        shapes.emplace_back(tmpShape);
        tmpHoles = newShapeHoles[i];

        for (NewShapeHoles& tmpHole : tmpHoles) {

            tmpShape->holes.emplace_back(tmpHole.h);
        }
    }

    return shapes;
}
