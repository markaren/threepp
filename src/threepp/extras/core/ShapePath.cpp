
#include "threepp/extras/core/ShapePath.hpp"

using namespace threepp;

namespace {

    std::vector<std::unique_ptr<Shape>> toShapesNoHoles(const std::vector<std::unique_ptr<Path>>& inSubpaths) {

        std::vector<std::unique_ptr<Shape>> shapes;

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

    this->currentPath = new Path();
    this->subPaths.emplace_back(this->currentPath);
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

std::vector<std::unique_ptr<Shape>> ShapePath::toShapes(bool isCCW, bool noHoles) {

    return {}; // TODO

//    if (subPaths.empty()) return {};
//
//    if (noHoles) return toShapesNoHoles(subPaths);
//
//
//    bool solid;
//    Path* tmpPath;
//    std::unique_ptr<Shape> tmpShape;
//    std::vector<std::unique_ptr<Shape>> shapes;
//
//    if (subPaths.size() == 1) {
//
//        tmpPath = subPaths[0].get();
//        tmpShape = std::make_unique<Shape>();
//        tmpShape->curves = tmpPath->curves;
//        shapes.emplace_back(std::move(tmpShape));
//        return shapes;
//    }
//
//    bool holesFirst = !isClockWise(subPaths[0]->getPoints());
//    holesFirst = isCCW ? !holesFirst : holesFirst;
//
//    // console.log("Holes first", holesFirst);
//
//    const betterShapeHoles = [];
//    const newShapes = [];
//    let newShapeHoles = [];
//    unsigned int mainIdx = 0;
//    std::vector<Vector2> tmpPoints;
//
//    newShapes[mainIdx] = undefined;
//    newShapeHoles[mainIdx] = [];
//
//    for (unsigned i = 0, l = subPaths.size(); i < l; i++) {
//
//        tmpPath = subPaths[i].get();
//        tmpPoints = tmpPath->getPoints();
//        solid = isClockWise(tmpPoints);
//        solid = isCCW ? !solid : solid;
//
//        if (solid) {
//
//            if ((!holesFirst) && (newShapes[mainIdx])) mainIdx++;
//
//            newShapes[mainIdx] = {new Shape(), tmpPoints};
//            newShapes[mainIdx].s.curves = tmpPath.curves;
//
//            if (holesFirst) mainIdx++;
//            newShapeHoles[mainIdx] = [];
//
//            //console.log('cw', i);
//
//        } else {
//
//            newShapeHoles[mainIdx].push({h: tmpPath, p: tmpPoints[0]});
//
//            //console.log('ccw', i);
//        }
//    }
//
//    // only Holes? -> probably all Shapes with wrong orientation
//    if (!newShapes[0]) return toShapesNoHoles(subPaths);
//
//
//    if (newShapes.length > 1) {
//
//        bool ambiguous = false;
//        const toChange = [];
//
//        for (unsigned sIdx = 0, sLen = newShapes.length; sIdx < sLen; sIdx++) {
//
//            betterShapeHoles[sIdx] = [];
//        }
//
//        for (let sIdx = 0, sLen = newShapes.length; sIdx < sLen; sIdx++) {
//
//            const sho = newShapeHoles[sIdx];
//
//            for (let hIdx = 0; hIdx < sho.length; hIdx++) {
//
//                const ho = sho[hIdx];
//                let hole_unassigned = true;
//
//                for (let s2Idx = 0; s2Idx < newShapes.length; s2Idx++) {
//
//                    if (isPointInsidePolygon(ho.p, newShapes[s2Idx].p)) {
//
//                        if (sIdx != = s2Idx) toChange.push({froms: sIdx, tos: s2Idx, hole: hIdx});
//                        if (hole_unassigned) {
//
//                            hole_unassigned = false;
//                            betterShapeHoles[s2Idx].push(ho);
//
//                        } else {
//
//                            ambiguous = true;
//                        }
//                    }
//                }
//
//                if (hole_unassigned) {
//
//                    betterShapeHoles[sIdx].push(ho);
//                }
//            }
//        }
//        // console.log("ambiguous: ", ambiguous);
//
//        if (toChange.length > 0) {
//
//            // console.log("to change: ", toChange);
//            if (!ambiguous) newShapeHoles = betterShapeHoles;
//        }
//    }
//
//    let tmpHoles;
//
//    for (unsigned i = 0, il = newShapes.size(); i < il; i++) {
//
//        tmpShape = newShapes[i].s;
//        shapes.push(tmpShape);
//        tmpHoles = newShapeHoles[i];
//
//        for (let j = 0, jl = tmpHoles.length; j < jl; j++) {
//
//            tmpShape.holes.push(tmpHoles[j].h);
//        }
//    }
//
//    //console.log("shape", shapes);
//
//    return shapes;
}
