
#ifndef THREEPP_SVGFUNCTIONS_HPP
#define THREEPP_SVGFUNCTIONS_HPP

#include "threepp/extras/core/ShapePath.hpp"

#include "threepp/extras/curves/CubicBezierCurve.hpp"
#include "threepp/extras/curves/EllipseCurve.hpp"
#include "threepp/extras/curves/LineCurve.hpp"
#include "threepp/extras/curves/QuadraticBezierCurve.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix3.hpp"

#include "SVGTypes.hpp"

#include <cmath>
#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <algorithm>


namespace threepp::svg {

    struct RE {

        inline static std::regex SEPARATOR{R"([ \t\r\n\,.\-+])"};
        inline static std::regex WHITESPACE{R"([ \t\r\n])"};
        inline static std::regex DIGIT{"[\\d]"};
        inline static std::regex SIGN{"[+-]"};
        inline static std::regex POINT{"\\."};
        inline static std::regex COMMA{"\\,"};
        inline static std::regex EXP{"e", std::regex::icase};
        inline static std::regex FLAGS{"[01]"};
    };

    // http://www.w3.org/TR/SVG11/implnote.html#PathElementImplementationNotes
    inline float getReflection(float a, float b) {

        return a - (b - a);
    }

    inline std::vector<float> parseFloats(const std::string& input, std::vector<int> flags = {}, int stride = 0) {

        // States
        const int SEP = 0;
        const int INT = 1;
        const int FLOAT = 2;
        const int EXP = 3;

        int state = SEP;
        bool seenComma = true;
        std::string number, exponent;
        std::vector<float> result;

        auto throwSyntaxError = [](const std::string& current, unsigned int i) {
            std::string error{"Unexpected character '" + current + "' at index " + std::to_string(i) + "."};
            throw std::runtime_error(error);
        };

        auto newNumber = [&] {
            if (!number.empty()) {
                if (exponent.empty()) {
                    result.emplace_back(std::stof(number));
                } else {
                    result.emplace_back(std::stof(number) * std::pow(10.f, std::stof(exponent)));
                }
            }

            number = "";
            exponent = "";
        };

        std::string current;
        const auto length = input.size();

        for (unsigned i = 0; i < length; i++) {

            current = input[i];

            // check for flags
            if (!flags.empty() && std::ranges::find(flags, result.size() % stride) != flags.end() && std::regex_match(current, RE::FLAGS)) {

                state = INT;
                number = current;
                newNumber();
                continue;
            }

            // parse until next number
            if (state == SEP) {

                // eat whitespace
                if (std::regex_match(current, RE::WHITESPACE)) {

                    continue;
                }

                // start new number
                if (std::regex_match(current, RE::DIGIT) || std::regex_match(current, RE::SIGN)) {

                    state = INT;
                    number = current;
                    continue;
                }

                if (std::regex_match(current, RE::POINT)) {

                    state = FLOAT;
                    number = current;
                    continue;
                }

                // throw on double commas (e.g. "1, , 2")
                if (std::regex_match(current, RE::COMMA)) {

                    if (seenComma) {

                        throwSyntaxError(current, i);
                    }

                    seenComma = true;
                }
            }

            // parse integer part
            if (state == INT) {

                if (std::regex_match(current, RE::DIGIT)) {

                    number += current;
                    continue;
                }

                if (std::regex_match(current, RE::POINT)) {

                    number += current;
                    state = FLOAT;
                    continue;
                }

                if (std::regex_match(current, RE::EXP)) {

                    state = EXP;
                    continue;
                }

                // throw on double signs ("-+1"), but not on sign as separator ("-1-2")
                if (std::regex_match(current, RE::SIGN) && number.size() == 1 && std::regex_match(std::string{number[0]}, RE::SIGN)) {

                    throwSyntaxError(current, i);
                }
            }

            // parse decimal part
            if (state == FLOAT) {

                if (std::regex_match(current, RE::DIGIT)) {

                    number += current;
                    continue;
                }

                if (std::regex_match(current, RE::EXP)) {

                    state = EXP;
                    continue;
                }

                // throw on double decimal points (e.g. "1..2")
                if (std::regex_match(current, RE::POINT) && number[number.size() - 1] == '.') {

                    throwSyntaxError(current, i);
                }
            }

            // parse exponent part
            if (state == EXP) {

                if (std::regex_match(current, RE::DIGIT)) {

                    exponent += current;
                    continue;
                }

                if (std::regex_match(current, RE::SIGN)) {

                    if (exponent.empty()) {

                        exponent += current;
                        continue;
                    }

                    if (exponent.size() == 1 && std::regex_match(exponent, RE::SIGN)) {

                        throwSyntaxError(current, i);
                    }
                }
            }

            // end of number
            if (std::regex_match(current, RE::WHITESPACE)) {

                newNumber();
                state = SEP;
                seenComma = false;

            } else if (std::regex_match(current, RE::COMMA)) {

                newNumber();
                state = SEP;
                seenComma = true;

            } else if (std::regex_match(current, RE::SIGN)) {

                newNumber();
                state = INT;
                number = current;

            } else if (std::regex_match(current, RE::POINT)) {

                newNumber();
                state = FLOAT;
                number = current;

            } else {

                throwSyntaxError(current, i);
            }
        }

        // add the last number found (if any)
        newNumber();

        return result;
    }

    inline float svgAngle(float ux, float uy, float vx, float vy) {

        const auto dot = ux * vx + uy * vy;
        const auto len = std::sqrt(ux * ux + uy * uy) * std::sqrt(vx * vx + vy * vy);
        float ang = std::acos(std::max(-1.f, std::min(1.f, dot / len)));// floating point precision, slightly over values appear
        if ((ux * vy - uy * vx) < 0) ang = -ang;
        return ang;
    }

    /**
     * https://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes
     * https://mortoray.com/2017/02/16/rendering-an-svg-elliptical-arc-as-bezier-curves/ Appendix: Endpoint to center arc conversion
     * From
     * rx ry x-axis-rotation large-arc-flag sweep-flag x y
     * To
     * aX, aY, xRadius, yRadius, aStartAngle, aEndAngle, aClockwise, aRotation
     */
    inline void parseArcCommand(ShapePath& path, float rx, float ry, float x_axis_rotation, bool large_arc_flag, bool sweep_flag, const Vector2& start, const Vector2& end) {

        if (rx == 0 || ry == 0) {

            // draw a line if either of the radii == 0
            path.lineTo(end.x, end.y);
            return;
        }

        x_axis_rotation = x_axis_rotation * math::PI / 180.f;

        // Ensure radii are positive
        rx = std::abs(rx);
        ry = std::abs(ry);

        // Compute (x1', y1')
        const auto dx2 = (start.x - end.x) / 2.0f;
        const auto dy2 = (start.y - end.y) / 2.0f;
        const auto x1p = std::cos(x_axis_rotation) * dx2 + std::sin(x_axis_rotation) * dy2;
        const auto y1p = -std::sin(x_axis_rotation) * dx2 + std::cos(x_axis_rotation) * dy2;

        // Compute (cx', cy')
        auto rxs = rx * rx;
        auto rys = ry * ry;
        const auto x1ps = x1p * x1p;
        const auto y1ps = y1p * y1p;

        // Ensure radii are large enough
        const auto cr = x1ps / rxs + y1ps / rys;

        if (cr > 1) {

            // scale up rx,ry equally so cr == 1
            const auto s = std::sqrt(cr);
            rx = s * rx;
            ry = s * ry;
            rxs = rx * rx;
            rys = ry * ry;
        }

        const auto dq = (rxs * y1ps + rys * x1ps);
        const auto pq = (rxs * rys - dq) / dq;
        float q = std::sqrt(std::max(0.f, pq));
        if (large_arc_flag == sweep_flag) q = -q;
        const auto cxp = q * rx * y1p / ry;
        const auto cyp = -q * ry * x1p / rx;

        // Step 3: Compute (cx, cy) from (cx', cy')
        const auto cx = std::cos(x_axis_rotation) * cxp - std::sin(x_axis_rotation) * cyp + (start.x + end.x) / 2;
        const auto cy = std::sin(x_axis_rotation) * cxp + std::cos(x_axis_rotation) * cyp + (start.y + end.y) / 2;

        // Step 4: Compute θ1 and Δθ
        const auto theta = svgAngle(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry);
        const auto delta = std::fmod(svgAngle((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry), (math::PI * 2));

        path.currentPath->absellipse(cx, cy, rx, ry, theta, theta + delta, sweep_flag == 0, x_axis_rotation);
    }

    inline void classifyPoint(const Vector2& p, const Vector2& edgeStart, const Vector2& edgeEnd) {

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

    inline std::optional<EdgeIntersection> findEdgeIntersection(const Vector2& a0, const Vector2& a1, const Vector2& b0, const Vector2& b1) {
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


    inline std::vector<Vector2> getIntersections(const std::vector<Vector2>& path1, const std::vector<Vector2>& path2) {

        std::vector<EdgeIntersection> intersectionsRaw;
        std::vector<Vector2> intersections;

        for (unsigned index = 1; index < path1.size(); index++) {

            const auto& path1EdgeStart = path1[index - 1];
            const auto& path1EdgeEnd = path1[index];

            for (unsigned index2 = 1; index2 < path2.size(); index2++) {

                const auto& path2EdgeStart = path2[index2 - 1];
                const auto& path2EdgeEnd = path2[index2];

                const auto intersection = findEdgeIntersection(path1EdgeStart, path1EdgeEnd, path2EdgeStart, path2EdgeEnd);

                if (intersection && std::ranges::find_if(intersectionsRaw, [&](auto& i) {
                                        return i.t <= intersection->t + std::numeric_limits<float>::epsilon() &&
                                               i.t >= intersection->t - std::numeric_limits<float>::epsilon();
                                    }) == intersectionsRaw.end()) {

                    intersectionsRaw.emplace_back(*intersection);
                    intersections.emplace_back(intersection->x, intersection->y);
                }
            }
        }

        return intersections;
    }

    inline std::vector<svg::Intersection> getScanlineIntersections(const std::vector<Vector2>& scanline, const Box2& boundingBox, const std::vector<SimplePath>& paths) {

        Vector2 center;
        boundingBox.getCenter(center);

        std::vector<svg::Intersection> allIntersections;

        for (const auto& path : paths) {

            // check if the center of the bounding box is in the bounding box of the paths.
            // this is a pruning method to limit the search of intersections in paths that can't envelop of the current path.
            // if a path envelops another path. The center of that oter path, has to be inside the bounding box of the enveloping path.
            if (path.boundingBox.containsPoint(center)) {

                const auto intersections = getIntersections(scanline, path.points);

                for (const auto& p : intersections) {

                    allIntersections.emplace_back(svg::Intersection{path.identifier, p});
                }
            }
        }

        std::ranges::sort(allIntersections, [](const auto& i1, const auto& i2) {
            return i1.point.x < i2.point.x;
        });

        return allIntersections;
    }

    inline std::optional<AHole> isHoleTo(const SimplePath& simplePath, const std::vector<SimplePath>& allPaths, float scanlineMinX, float scanlineMaxX, std::string fillRule) {

        if (fillRule.empty()) {

            fillRule = "nonzero";
        }

        Vector2 centerBoundingBox;
        simplePath.boundingBox.getCenter(centerBoundingBox);

        const std::vector scanline{Vector2(scanlineMinX, centerBoundingBox.y), Vector2(scanlineMaxX, centerBoundingBox.y)};

        auto scanlineIntersections = getScanlineIntersections(scanline, simplePath.boundingBox, allPaths);

        std::ranges::sort(scanlineIntersections, [](const auto& i1, const auto& i2) {
            return i1.point.x < i2.point.x;
        });

        std::vector<svg::Intersection> baseIntersections;
        std::vector<svg::Intersection> otherIntersections;

        for (const auto& i : scanlineIntersections) {
            if (i.identifier == simplePath.identifier) {

                baseIntersections.emplace_back(i);

            } else {

                otherIntersections.emplace_back(i);
            }
        }

        const auto firstXOfPath = baseIntersections[0].point.x;

        // build up the path hierarchy
        std::vector<int> stack;
        unsigned i = 0;

        while (i < otherIntersections.size() && otherIntersections[i].point.x < firstXOfPath) {

            if (!stack.empty() && stack[stack.size() - 1] == otherIntersections[i].identifier) {

                stack.pop_back();

            } else {

                stack.emplace_back(otherIntersections[i].identifier);
            }

            i++;
        }

        stack.emplace_back(simplePath.identifier);

        if (fillRule == "evenodd") {

            const auto isHole = stack.size() % 2 == 0;
            std::optional<int> isHoleFor;
            if (isHole) {
                isHoleFor = stack[stack.size() - 2];
            }

            return AHole{simplePath.identifier, isHole, isHoleFor};

        } else if (fillRule == "nonzero") {

            // check if path is a hole by counting the amount of paths with alternating rotations it has to cross.
            bool isHole = true;
            std::optional<int> isHoleFor;
            std::optional<bool> lastCWValue;

            for (int identifier : stack) {

                if (isHole) {

                    lastCWValue = allPaths[identifier].isCW;
                    isHole = false;
                    isHoleFor = identifier;

                } else if (lastCWValue != allPaths[identifier].isCW) {

                    lastCWValue = allPaths[identifier].isCW;
                    isHole = true;
                }
            }

            return AHole{simplePath.identifier, isHole, isHoleFor};

        } else {

            std::cerr << "fill-rule: '" << fillRule << "' is currently not implemented." << std::endl;
            return std::nullopt;
        }
    }

    inline void removeDuplicatedPoints(std::vector<Vector2>& points, float minDistance) {

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

    inline Vector2& getNormal(const Vector2& p1, const Vector2& p2, Vector2& result) {

        result.subVectors(p2, p1);
        return result.set(-result.y, result.x).normalize();
    }

    inline bool isTransformRotated(const Matrix3& m) {

        return m.elements[1] != 0 || m.elements[3] != 0;
    }

    inline bool isTransformFlipped(const Matrix3& m) {

        const auto& te = m.elements;
        return te[0] * te[4] - te[1] * te[3] < 0;
    }

    inline float getTransformScaleX(const Matrix3& m) {

        const auto& te = m.elements;
        return std::sqrt(te[0] * te[0] + te[1] * te[1]);
    }

    inline float getTransformScaleY(const Matrix3& m) {

        const auto& te = m.elements;
        return std::sqrt(te[3] * te[3] + te[4] * te[4]);
    }

    inline bool isTransformSkewed(const Matrix3& m) {

        const auto& te = m.elements;
        const auto basisDot = te[0] * te[3] + te[1] * te[4];

        // Shortcut for trivial rotations and transformations
        if (basisDot == 0) return false;

        const auto sx = getTransformScaleX(m);
        const auto sy = getTransformScaleY(m);

        return std::abs(basisDot / (sx * sy)) > std::numeric_limits<float>::epsilon();
    }

    // Calculates the eigensystem of a real symmetric 2x2 matrix
    //    [ A  B ]
    //    [ B  C ]
    // in the form
    //    [ A  B ]  =  [ cs  -sn ] [ rt1   0  ] [  cs  sn ]
    //    [ B  C ]     [ sn   cs ] [  0   rt2 ] [ -sn  cs ]
    // where rt1 >= rt2.
    //
    // Adapted from: https://www.mpi-hd.mpg.de/personalhomes/globes/3x3/index.html
    // -> Algorithms for real symmetric matrices -> Analytical (2x2 symmetric)
    inline EigenDecomposition eigenDecomposition(float A, float B, float C) {

        float rt1{}, rt2{}, cs{}, sn{}, t{};
        const auto sm = A + C;
        const auto df = A - C;
        const auto rt = std::sqrt(df * df + 4 * B * B);

        if (sm > 0) {

            rt1 = 0.5f * (sm + rt);
            t = 1.f / rt1;
            rt2 = A * t * C - B * t * B;

        } else if (sm < 0) {

            rt2 = 0.5f * (sm - rt);

        } else {

            // This case needs to be treated separately to avoid div by 0

            rt1 = 0.5f * rt;
            rt2 = -0.5f * rt;
        }

        // Calculate eigenvectors

        if (df > 0) {

            cs = df + rt;

        } else {

            cs = df - rt;
        }

        if (std::abs(cs) > 2 * std::abs(B)) {

            t = -2 * B / cs;
            sn = 1.f / std::sqrt(1 + t * t);
            cs = t * sn;

        } else if (std::abs(B) == 0) {

            cs = 1;
            sn = 0;

        } else {

            t = -0.5f * cs / B;
            cs = 1.f / std::sqrt(1 + t * t);
            sn = t * cs;
        }

        if (df > 0) {

            t = cs;
            cs = -sn;
            sn = t;
        }

        return {rt1, rt2, cs, sn};
    }

    inline void transfEllipseGeneric(EllipseCurve& curve, const Matrix3& m) {

        Matrix3 tempTransform0;
        Matrix3 tempTransform1;
        Matrix3 tempTransform2;
        // Matrix3 tempTransform3;

        // For math description see:
        // https://math.stackexchange.com/questions/4544164

        const auto a = curve.xRadius;
        const auto b = curve.yRadius;

        const auto cosTheta = std::cos(curve.aRotation);
        const auto sinTheta = std::sin(curve.aRotation);

        auto v1 = Vector3(a * cosTheta, a * sinTheta, 0);
        auto v2 = Vector3(-b * sinTheta, b * cosTheta, 0);

        const auto f1 = v1.applyMatrix3(m);
        const auto f2 = v2.applyMatrix3(m);

        const auto mF = tempTransform0.set(
                f1.x, f2.x, 0,
                f1.y, f2.y, 0,
                0, 0, 1);

        const auto mFInv = tempTransform1.copy(mF).invert();
        auto mFInvT = tempTransform2.copy(mFInv).transpose();
        const auto mQ = mFInvT.multiply(mFInv);
        const auto mQe = mQ.elements;

        const auto ed = eigenDecomposition(mQe[0], mQe[1], mQe[4]);
        const auto rt1sqrt = std::sqrt(ed.rt1);
        const auto rt2sqrt = std::sqrt(ed.rt2);

        curve.xRadius = 1.f / rt1sqrt;
        curve.yRadius = 1.f / rt2sqrt;
        curve.aRotation = std::atan2(ed.sn, ed.cs);

        const auto isFullEllipse =
                std::fmod((curve.aEndAngle - curve.aStartAngle), (2 * math::PI)) < std::numeric_limits<float>::epsilon();

        // Do not touch angles of a full ellipse because after transformation they
        // would converge to a sinle value effectively removing the whole curve

        if (!isFullEllipse) {

            auto mDsqrt = tempTransform1.set(
                    rt1sqrt, 0, 0,
                    0, rt2sqrt, 0,
                    0, 0, 1);

            const auto mRT = tempTransform2.set(
                    ed.cs, ed.sn, 0,
                    -ed.sn, ed.cs, 0,
                    0, 0, 1);

            const auto mDRF = mDsqrt.multiply(mRT).multiply(mF);

            auto transformAngle = [&](auto phi) {
                Vector3 v(std::cos(phi), std::sin(phi), 0);
                v.applyMatrix3(mDRF);

                return std::atan2(v.y, v.x);
            };

            curve.aStartAngle = transformAngle(curve.aStartAngle);
            curve.aEndAngle = transformAngle(curve.aEndAngle);

            if (isTransformFlipped(m)) {

                curve.aClockwise = !curve.aClockwise;
            }
        }
    }

    inline void transfEllipseNoSkew(EllipseCurve& curve, const Matrix3& m) {

        // Faster shortcut if no skew is applied
        // (e.g, a euclidean transform of a group containing the ellipse)

        const auto sx = getTransformScaleX(m);
        const auto sy = getTransformScaleY(m);

        curve.xRadius *= sx;
        curve.yRadius *= sy;

        // Extract rotation angle from the matrix of form:
        //
        //  | cosθ sx   -sinθ sy |
        //  | sinθ sx    cosθ sy |
        //
        // Remembering that tanθ = sinθ / cosθ; and that
        // `sx`, `sy`, or both might be zero.
        const auto theta =
                sx > std::numeric_limits<float>::epsilon()
                        ? std::atan2(m.elements[1], m.elements[0])
                        : std::atan2(-m.elements[3], m.elements[4]);

        curve.aRotation += theta;

        if (isTransformFlipped(m)) {

            curve.aStartAngle *= -1;
            curve.aEndAngle *= -1;
            curve.aClockwise = !curve.aClockwise;
        }
    }

    inline void transformPath(ShapePath& path, const Matrix3& m) {

        Vector2 tempV2;
        Vector3 tempV3;

        auto transfVec2 = [&](Vector2& v2) {
            tempV3.set(v2.x, v2.y, 1).applyMatrix3(m);

            v2.set(tempV3.x, tempV3.y);
        };

        // const auto isRotated = isTransformRotated(m);

        const auto& subPaths = path.subPaths;

        for (const auto& subPath : subPaths) {

            const auto& curves = subPath->curves;

            for (const auto& curve : curves) {

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

                    // Transform ellipse center point

                    tempV2.set(ellipseCurve->aX, ellipseCurve->aY);
                    transfVec2(tempV2);
                    ellipseCurve->aX = tempV2.x;
                    ellipseCurve->aY = tempV2.y;

                    // Transform ellipse shape parameters

                    if (isTransformSkewed(m)) {

                        transfEllipseGeneric(*ellipseCurve, m);

                    } else {

                        transfEllipseNoSkew(*ellipseCurve, m);
                    }
                }
            }
        }
    }

}// namespace threepp::svg

#endif//THREEPP_SVGFUNCTIONS_HPP
