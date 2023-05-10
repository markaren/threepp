
#ifndef THREEPP_SVGFUNCTIONS_HPP
#define THREEPP_SVGFUNCTIONS_HPP

#include "threepp/extras/core/ShapePath.hpp"

#include "threepp/math/MathUtils.hpp"
#include "threepp/utils/StringUtils.hpp"

#include "SVGTypes.hpp"

#include <cmath>
#include <iostream>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "pugixml.hpp"

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

    float getReflection(float a, float b) {

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
            if (!flags.empty() && std::find(flags.begin(), flags.end(), result.size() % stride) != flags.end() && std::regex_match(current, RE::FLAGS)) {

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

    float svgAngle(float ux, float uy, float vx, float vy) {

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
    void parseArcCommand(ShapePath& path, float rx, float ry, float x_axis_rotation, bool large_arc_flag, bool sweep_flag, const Vector2& start, const Vector2& end) {

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


    ShapePath parsePathNode(const pugi::xml_node& node) {

        ShapePath path;

        Vector2 point;
        Vector2 control;

        Vector2 firstPoint;
        bool isFirstPoint = true;
        bool doSetFirstPoint = false;

        const std::string d = node.attribute("d").value();

        std::regex r{"[a-df-z][^a-df-z]*", std::regex::icase};

        for (std::sregex_iterator i = std::sregex_iterator(d.begin(), d.end(), r);
             i != std::sregex_iterator();
             ++i) {
            std::smatch m = *i;

            std::string command = m.str();

            const auto type = command[0];
            const auto data = utils::trim(command.substr(1));

            if (isFirstPoint == true) {

                doSetFirstPoint = true;
                isFirstPoint = false;
            }

            std::vector<float> numbers;

            switch (type) {

                case 'M':
                    numbers = parseFloats(data);
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

                        if (j == 0 && doSetFirstPoint == true) firstPoint.copy(point);
                    }

                    break;

                case 'H':
                    numbers = parseFloats(data);

                    for (unsigned j = 0, jl = numbers.size(); j < jl; j++) {

                        point.x = numbers[j];
                        control.x = point.x;
                        control.y = point.y;
                        path.lineTo(point.x, point.y);

                        if (j == 0 && doSetFirstPoint == true) firstPoint.copy(point);
                    }

                    break;

                case 'V':
                    numbers = parseFloats(data);

                    for (unsigned j = 0, jl = numbers.size(); j < jl; j++) {

                        point.y = numbers[j];
                        control.x = point.x;
                        control.y = point.y;
                        path.lineTo(point.x, point.y);

                        if (j == 0 && doSetFirstPoint == true) firstPoint.copy(point);
                    }

                    break;

                case 'L':
                    numbers = parseFloats(data);

                    for (unsigned j = 0, jl = numbers.size(); j < jl; j += 2) {

                        point.x = numbers[j + 0];
                        point.y = numbers[j + 1];
                        control.x = point.x;
                        control.y = point.y;
                        path.lineTo(point.x, point.y);

                        if (j == 0 && doSetFirstPoint == true) firstPoint.copy(point);
                    }

                    break;

                case 'C':
                    numbers = parseFloats(data);

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

                        if (j == 0 && doSetFirstPoint == true) firstPoint.copy(point);
                    }

                    break;

                case 'S':
                    numbers = parseFloats(data);

                    for (unsigned j = 0, jl = numbers.size(); j < jl; j += 4) {

                        path.bezierCurveTo(
                                getReflection(point.x, control.x),
                                getReflection(point.y, control.y),
                                numbers[j + 0],
                                numbers[j + 1],
                                numbers[j + 2],
                                numbers[j + 3]);
                        control.x = numbers[j + 0];
                        control.y = numbers[j + 1];
                        point.x = numbers[j + 2];
                        point.y = numbers[j + 3];

                        if (j == 0 && doSetFirstPoint == true) firstPoint.copy(point);
                    }

                    break;

                case 'Q':
                    numbers = parseFloats(data);

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

                        if (j == 0 && doSetFirstPoint == true) firstPoint.copy(point);
                    }

                    break;

                case 'T':
                    numbers = parseFloats(data);

                    for (unsigned j = 0, jl = numbers.size(); j < jl; j += 2) {

                        const auto rx = getReflection(point.x, control.x);
                        const auto ry = getReflection(point.y, control.y);
                        path.quadraticCurveTo(
                                rx,
                                ry,
                                numbers[j + 0],
                                numbers[j + 1]);
                        control.x = rx;
                        control.y = ry;
                        point.x = numbers[j + 0];
                        point.y = numbers[j + 1];

                        if (j == 0 && doSetFirstPoint == true) firstPoint.copy(point);
                    }

                    break;

                case 'A':
                    numbers = parseFloats(data, {3, 4}, 7);

                    for (unsigned j = 0, jl = numbers.size(); j < jl; j += 7) {

                        // skip command if start point == end point
                        if (numbers[j + 5] == point.x && numbers[j + 6] == point.y) continue;

                        const auto start = point.clone();
                        point.x = numbers[j + 5];
                        point.y = numbers[j + 6];
                        control.x = point.x;
                        control.y = point.y;
                        parseArcCommand(
                                path, numbers[j], numbers[j + 1], numbers[j + 2], numbers[j + 3], numbers[j + 4], start, point);

                        if (j == 0 && doSetFirstPoint == true) firstPoint.copy(point);
                    }

                    break;

                case 'm':
                    numbers = parseFloats(data);

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

                        if (j == 0 && doSetFirstPoint == true) firstPoint.copy(point);
                    }

                    break;

                case 'h':
                    numbers = parseFloats(data);

                    for (unsigned j = 0, jl = numbers.size(); j < jl; j++) {

                        point.x += numbers[j];
                        control.x = point.x;
                        control.y = point.y;
                        path.lineTo(point.x, point.y);

                        if (j == 0 && doSetFirstPoint == true) firstPoint.copy(point);
                    }

                    break;

                case 'v':
                    numbers = parseFloats(data);

                    for (unsigned j = 0, jl = numbers.size(); j < jl; j++) {

                        point.y += numbers[j];
                        control.x = point.x;
                        control.y = point.y;
                        path.lineTo(point.x, point.y);

                        if (j == 0 && doSetFirstPoint == true) firstPoint.copy(point);
                    }

                    break;

                case 'l':
                    numbers = parseFloats(data);

                    for (unsigned j = 0, jl = numbers.size(); j < jl; j += 2) {

                        point.x += numbers[j + 0];
                        point.y += numbers[j + 1];
                        control.x = point.x;
                        control.y = point.y;
                        path.lineTo(point.x, point.y);

                        if (j == 0 && doSetFirstPoint == true) firstPoint.copy(point);
                    }

                    break;

                case 'c':
                    numbers = parseFloats(data);

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

                        if (j == 0 && doSetFirstPoint == true) firstPoint.copy(point);
                    }

                    break;

                case 's':
                    numbers = parseFloats(data);

                    for (unsigned j = 0, jl = numbers.size(); j < jl; j += 4) {

                        path.bezierCurveTo(
                                getReflection(point.x, control.x),
                                getReflection(point.y, control.y),
                                point.x + numbers[j + 0],
                                point.y + numbers[j + 1],
                                point.x + numbers[j + 2],
                                point.y + numbers[j + 3]);
                        control.x = point.x + numbers[j + 0];
                        control.y = point.y + numbers[j + 1];
                        point.x += numbers[j + 2];
                        point.y += numbers[j + 3];

                        if (j == 0 && doSetFirstPoint == true) firstPoint.copy(point);
                    }

                    break;

                case 'q':
                    numbers = parseFloats(data);

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

                        if (j == 0 && doSetFirstPoint == true) firstPoint.copy(point);
                    }

                    break;

                case 't':
                    numbers = parseFloats(data);

                    for (unsigned j = 0, jl = numbers.size(); j < jl; j += 2) {

                        const auto rx = getReflection(point.x, control.x);
                        const auto ry = getReflection(point.y, control.y);
                        path.quadraticCurveTo(
                                rx,
                                ry,
                                point.x + numbers[j + 0],
                                point.y + numbers[j + 1]);
                        control.x = rx;
                        control.y = ry;
                        point.x = point.x + numbers[j + 0];
                        point.y = point.y + numbers[j + 1];

                        if (j == 0 && doSetFirstPoint == true) firstPoint.copy(point);
                    }

                    break;

                case 'a':
                    numbers = parseFloats(data, {3, 4}, 7);

                    for (unsigned j = 0, jl = numbers.size(); j < jl; j += 7) {

                        // skip command if no displacement
                        if (numbers[j + 5] == 0 && numbers[j + 6] == 0) continue;

                        const auto start = point.clone();
                        point.x += numbers[j + 5];
                        point.y += numbers[j + 6];
                        control.x = point.x;
                        control.y = point.y;
                        parseArcCommand(
                                path, numbers[j], numbers[j + 1], numbers[j + 2], numbers[j + 3], numbers[j + 4], start, point);

                        if (j == 0 && doSetFirstPoint == true) firstPoint.copy(point);
                    }

                    break;

                case 'Z':
                case 'z':
                    path.currentPath->autoClose = true;

                    if (path.currentPath->curves.size() > 0) {

                        // Reset point to beginning of Path
                        point.copy(firstPoint);
                        path.currentPath->currentPoint.copy(point);
                        isFirstPoint = true;
                    }

                    break;

                default:
                    std::cerr << command << std::endl;
            }

            // console.log( type, parseFloats( data ), parseFloats( data ).length  )

            doSetFirstPoint = false;
        }

        return path;
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
                    return EdgeIntersection{x,y,classifyResult.t,
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
                                    }) == intersectionsRaw.end()) {

                    intersectionsRaw.emplace_back(*intersection);
                    intersections.emplace_back(intersection->x, intersection->y);
                }
            }
        }

        return intersections;
    }

    std::vector<svg::Intersection> getScanlineIntersections(const std::vector<Vector2>& scanline, const Box2& boundingBox, const std::vector<SimplePath>& paths) {

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

        std::sort(allIntersections.begin(), allIntersections.end(), [](const auto& i1, const auto& i2) {
            return i1.point.x < i2.point.x;
        });

        return allIntersections;
    }

    std::optional<AHole> isHoleTo(const SimplePath& simplePath, const std::vector<SimplePath>& allPaths, float scanlineMinX, float scanlineMaxX, std::string fillRule) {

        if (fillRule.empty()) {

            fillRule = "nonzero";
        }

        Vector2 centerBoundingBox;
        simplePath.boundingBox.getCenter(centerBoundingBox);

        const std::vector<Vector2> scanline{Vector2(scanlineMinX, centerBoundingBox.y), Vector2(scanlineMaxX, centerBoundingBox.y)};

        auto scanlineIntersections = getScanlineIntersections(scanline, simplePath.boundingBox, allPaths);

        std::sort(scanlineIntersections.begin(), scanlineIntersections.end(), [](const auto& i1, const auto& i2) {
            return i1.point.x > i2.point.x;
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

        if (fillRule == "evenodd" && stack.size() > 1) {

            const auto isHole = stack.size() % 2 == 0;
            const auto isHoleFor = stack[stack.size() - 2];

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

    bool isTransformRotated(const Matrix3& m) {

        return m.elements[1] != 0 || m.elements[3] != 0;
    }

    float getTransformScaleX(const Matrix3& m) {

        const auto& te = m.elements;
        return std::sqrt(te[0] * te[0] + te[1] * te[1]);
    }

    float getTransformScaleY(const Matrix3& m) {

        const auto& te = m.elements;
        return std::sqrt(te[3] * te[3] + te[4] * te[4]);
    }

}// namespace threepp::svg

#endif//THREEPP_SVGFUNCTIONS_HPP
