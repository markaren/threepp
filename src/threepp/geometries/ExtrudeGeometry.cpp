
#include "threepp/geometries/ExtrudeGeometry.hpp"

#include "threepp/extras/ShapeUtils.hpp"
#include "threepp/extras/core/Shape.hpp"
#include "threepp/math/MathUtils.hpp"

#include <cmath>
#include <functional>

using namespace threepp;

namespace {

    std::vector<Vector2> generateTopUV(const std::vector<float>& vertices, unsigned int indexA, unsigned int indexB, unsigned int indexC) {

        const auto a_x = vertices[indexA * 3];
        const auto a_y = vertices[indexA * 3 + 1];
        const auto b_x = vertices[indexB * 3];
        const auto b_y = vertices[indexB * 3 + 1];
        const auto c_x = vertices[indexC * 3];
        const auto c_y = vertices[indexC * 3 + 1];

        return {
                Vector2(a_x, a_y),
                Vector2(b_x, b_y),
                Vector2(c_x, c_y)};
    }

    std::vector<Vector2> generateSideWallUV(const std::vector<float>& vertices, unsigned int indexA, unsigned int indexB, unsigned int indexC, unsigned int indexD) {

        const auto a_x = vertices[indexA * 3];
        const auto a_y = vertices[indexA * 3 + 1];
        const auto a_z = vertices[indexA * 3 + 2];
        const auto b_x = vertices[indexB * 3];
        const auto b_y = vertices[indexB * 3 + 1];
        const auto b_z = vertices[indexB * 3 + 2];
        const auto c_x = vertices[indexC * 3];
        const auto c_y = vertices[indexC * 3 + 1];
        const auto c_z = vertices[indexC * 3 + 2];
        const auto d_x = vertices[indexD * 3];
        const auto d_y = vertices[indexD * 3 + 1];
        const auto d_z = vertices[indexD * 3 + 2];

        if (std::abs(a_y - b_y) < std::abs(a_x - b_x)) {

            return {
                    Vector2(a_x, 1 - a_z),
                    Vector2(b_x, 1 - b_z),
                    Vector2(c_x, 1 - c_z),
                    Vector2(d_x, 1 - d_z)};

        } else {

            return {
                    Vector2(a_y, 1 - a_z),
                    Vector2(b_y, 1 - b_z),
                    Vector2(c_y, 1 - c_z),
                    Vector2(d_y, 1 - d_z)};
        }
    }

    /////  Internal functions

    Vector2 scalePt2(Vector2 pt, const Vector2& vec, float size) {

        return pt.addScaledVector(vec, size);
    }

    Vector2 getBevelVec(const Vector2& inPt, const Vector2& inPrev, const Vector2& inNext) {

        // computes for inPt the corresponding point inPt' on a new contour
        //   shifted by 1 unit (length of normalized vector) to the left
        // if we walk along contour clockwise, this new contour is outside the old one
        //
        // inPt' is the intersection of the two lines parallel to the two
        //  adjacent edges of inPt at a distance of 1 unit on the left side.

        float v_trans_x, v_trans_y, shrink_by;// resulting translation vector for inPt

        // good reading for geometry algorithms (here: line-line intersection)
        // http://geomalgorithms.com/a05-_intersect-1.html

        const auto v_prev_x = inPt.x - inPrev.x,
                   v_prev_y = inPt.y - inPrev.y;
        const auto v_next_x = inNext.x - inPt.x,
                   v_next_y = inNext.y - inPt.y;

        const auto v_prev_lensq = (v_prev_x * v_prev_x + v_prev_y * v_prev_y);

        // check for collinear edges
        const auto collinear0 = (v_prev_x * v_next_y - v_prev_y * v_next_x);

        if (std::abs(collinear0) > std::numeric_limits<float>::epsilon()) {

            // not collinear

            // length of vectors for normalizing

            const auto v_prev_len = std::sqrt(v_prev_lensq);
            const auto v_next_len = std::sqrt(v_next_x * v_next_x + v_next_y * v_next_y);

            // shift adjacent points by unit vectors to the left

            const auto ptPrevShift_x = (inPrev.x - v_prev_y / v_prev_len);
            const auto ptPrevShift_y = (inPrev.y + v_prev_x / v_prev_len);

            const auto ptNextShift_x = (inNext.x - v_next_y / v_next_len);
            const auto ptNextShift_y = (inNext.y + v_next_x / v_next_len);

            // scaling factor for v_prev to intersection point

            const auto sf = ((ptNextShift_x - ptPrevShift_x) * v_next_y -
                             (ptNextShift_y - ptPrevShift_y) * v_next_x) /
                            (v_prev_x * v_next_y - v_prev_y * v_next_x);

            // vector from inPt to intersection point

            v_trans_x = (ptPrevShift_x + v_prev_x * sf - inPt.x);
            v_trans_y = (ptPrevShift_y + v_prev_y * sf - inPt.y);

            // Don't normalize!, otherwise sharp corners become ugly
            //  but prevent crazy spikes
            const auto v_trans_lensq = (v_trans_x * v_trans_x + v_trans_y * v_trans_y);
            if (v_trans_lensq <= 2) {

                return {v_trans_x, v_trans_y};

            } else {

                shrink_by = std::sqrt(v_trans_lensq / 2);
            }

        } else {

            // handle special case of collinear edges

            bool direction_eq = false;// assumes: opposite

            if (v_prev_x > std::numeric_limits<float>::epsilon()) {

                if (v_next_x > std::numeric_limits<float>::epsilon()) {

                    direction_eq = true;
                }

            } else {

                if (v_prev_x < -std::numeric_limits<float>::epsilon()) {

                    if (v_next_x < -std::numeric_limits<float>::epsilon()) {

                        direction_eq = true;
                    }

                } else {

                    if (math::sgn(v_prev_y) == math::sgn(v_next_y)) {

                        direction_eq = true;
                    }
                }
            }

            if (direction_eq) {

                // console.log("Warning: lines are a straight sequence");
                v_trans_x = -v_prev_y;
                v_trans_y = v_prev_x;
                shrink_by = std::sqrt(v_prev_lensq);

            } else {

                // console.log("Warning: lines are a straight spike");
                v_trans_x = v_prev_x;
                v_trans_y = v_prev_y;
                shrink_by = std::sqrt(v_prev_lensq / 2);
            }
        }

        return {v_trans_x / shrink_by, v_trans_y / shrink_by};
    }

    void v(std::vector<float>& placeholder, float x, float y, float z) {

        placeholder.emplace_back(x);
        placeholder.emplace_back(y);
        placeholder.emplace_back(z);
    }

    void addVertex(std::vector<float>& verticesArray, const std::vector<float>& placeholder, unsigned int index) {

        verticesArray.emplace_back(placeholder[index * 3 + 0]);
        verticesArray.emplace_back(placeholder[index * 3 + 1]);
        verticesArray.emplace_back(placeholder[index * 3 + 2]);
    }

    void addUV(std::vector<float>& uvArray, const Vector2& vector2) {

        uvArray.emplace_back(vector2.x);
        uvArray.emplace_back(vector2.y);
    }


    void f3(std::vector<float>& uvArray, std::vector<float>& verticesArray, const std::vector<float>& placeholder, unsigned int a, unsigned int b, unsigned int c) {

        addVertex(verticesArray, placeholder, a);
        addVertex(verticesArray, placeholder, b);
        addVertex(verticesArray, placeholder, c);

        const auto nextIndex = verticesArray.size() / 3;
        const auto uvs = generateTopUV(verticesArray, nextIndex - 3, nextIndex - 2, nextIndex - 1);

        addUV(uvArray, uvs[0]);
        addUV(uvArray, uvs[1]);
        addUV(uvArray, uvs[2]);
    }


    void f4(std::vector<float>& uvArray, std::vector<float>& verticesArray, const std::vector<float>& placeholder, unsigned int a, unsigned int b, unsigned int c, unsigned int d) {

        addVertex(verticesArray, placeholder, a);
        addVertex(verticesArray, placeholder, b);
        addVertex(verticesArray, placeholder, d);

        addVertex(verticesArray, placeholder, b);
        addVertex(verticesArray, placeholder, c);
        addVertex(verticesArray, placeholder, d);


        const auto nextIndex = verticesArray.size() / 3;
        const auto uvs = generateSideWallUV(verticesArray, nextIndex - 6, nextIndex - 3, nextIndex - 2, nextIndex - 1);

        addUV(uvArray, uvs[0]);
        addUV(uvArray, uvs[1]);
        addUV(uvArray, uvs[3]);

        addUV(uvArray, uvs[1]);
        addUV(uvArray, uvs[2]);
        addUV(uvArray, uvs[3]);
    }

}// namespace


ExtrudeGeometry::ExtrudeGeometry(const std::vector<const Shape*>& shapes, const ExtrudeGeometry::Options& options) {

    std::vector<float> verticesArray;
    std::vector<float> uvArray;

    std::function<void(const Shape&)> addShape = [&](const Shape& shape) {
        std::vector<float> placeholder;

        // options

        const auto curveSegments = options.curveSegments;
        const auto steps = options.steps;
        const auto depth = options.depth;

        auto bevelEnabled = options.bevelEnabled;
        auto bevelThickness = options.bevelThickness;
        auto bevelSize = options.bevelSize;
        auto bevelOffset = options.bevelOffset;
        auto bevelSegments = options.bevelSegments;

        auto extrudePath = options.extrudePath;

        std::vector<Vector3> extrudePts;
        bool extrudeByPath = false;
        Curve3::FrenetFrames splineTube;
        Vector3 binormal, normal, position2;

        if (extrudePath) {

            extrudePts = extrudePath->getSpacedPoints(steps);

            extrudeByPath = true;
            bevelEnabled = false;// bevels not supported for path extrusion

            // SETUP TNB variables

            // TODO1 - have a .isClosed in spline?

            splineTube = extrudePath->computeFrenetFrames(steps, false);
        }

        // Safeguards if bevels are not enabled

        if (!bevelEnabled) {

            bevelSegments = 0;
            bevelThickness = 0;
            bevelSize = 0;
            bevelOffset = 0;
        }

        // Variables initialization

        ShapePoints shapePoints = shape.extractPoints(curveSegments);

        auto& vertices = shapePoints.shape;
        auto& holes = shapePoints.holes;

        const auto reverse = !shapeutils::isClockWise(vertices);

        if (reverse) {

            std::reverse(vertices.begin(), vertices.end());

            // Maybe we should also check if holes are in the opposite direction, just to be safe ...

            for (auto& ahole : holes) {

                if (shapeutils::isClockWise(ahole)) {

                    std::reverse(ahole.begin(), ahole.end());
                }
            }
        }

        const auto faces = shapeutils::triangulateShape(vertices, holes);

        /* Vertices */

        auto contour = vertices;// vertices has all points but contour has only points of circumference

        for (const auto& ahole : holes) {

            vertices.insert(vertices.end(), ahole.begin(), ahole.end());
        }

        const auto vlen = vertices.size(), flen = faces.size();

        // Find directions for point movement

        std::vector<Vector2> contourMovements;

        for (unsigned i = 0, il = contour.size(), j = il - 1, k = i + 1; i < il; i++, j++, k++) {

            if (j == il) j = 0;
            if (k == il) k = 0;

            //  (j)---(i)---(k)
            contourMovements.emplace_back(getBevelVec(contour[i], contour[j], contour[k]));
        }

        std::vector<std::vector<Vector2>> holesMovements;
        std::vector<Vector2> oneHoleMovements;
        std::vector<Vector2> verticesMovements = contourMovements;

        for (const auto& ahole : holes) {

            oneHoleMovements.clear();

            for (unsigned i = 0, il = ahole.size(), j = il - 1, k = i + 1; i < il; i++, j++, k++) {

                if (j == il) j = 0;
                if (k == il) k = 0;

                //  (j)---(i)---(k)
                oneHoleMovements.emplace_back(getBevelVec(ahole[i], ahole[j], ahole[k]));
            }

            holesMovements.emplace_back(oneHoleMovements);
            verticesMovements.insert(verticesMovements.end(), oneHoleMovements.begin(), oneHoleMovements.end());
        }

        // Loop bevelSegments, 1 for the front, 1 for the back

        for (unsigned b = 0; b < bevelSegments; b++) {

            const auto t = static_cast<float>(b) / static_cast<float>(bevelSegments);
            const auto z = bevelThickness * std::cos(t * math::PI / 2);
            const auto bs = bevelSize * std::sin(t * math::PI / 2) + bevelOffset;

            // contract shape

            for (unsigned i = 0, il = contour.size(); i < il; i++) {

                const auto vert = scalePt2(contour[i], contourMovements[i], bs);

                v(placeholder, vert.x, vert.y, -z);
            }

            // expand holes

            for (unsigned h = 0, hl = holes.size(); h < hl; h++) {

                const auto& ahole = holes[h];
                oneHoleMovements = holesMovements[h];

                for (unsigned i = 0, il = ahole.size(); i < il; i++) {

                    const auto vert = scalePt2(ahole[i], oneHoleMovements[i], bs);

                    v(placeholder, vert.x, vert.y, -z);
                }
            }
        }

        const auto bs = bevelSize + bevelOffset;

        // Back facing vertices

        for (unsigned i = 0; i < vlen; i++) {

            const auto vert = bevelEnabled ? scalePt2(vertices[i], verticesMovements[i], bs) : vertices[i];

            if (!extrudeByPath) {

                v(placeholder, vert.x, vert.y, 0);

            } else {

                // v( vert.x, vert.y + extrudePts[ 0 ].y, extrudePts[ 0 ].x );

                normal.copy(splineTube.normals[0]).multiplyScalar(vert.x);
                binormal.copy(splineTube.binormals[0]).multiplyScalar(vert.y);

                position2.copy(extrudePts[0]).add(normal).add(binormal);

                v(placeholder, position2.x, position2.y, position2.z);
            }
        }

        // Add stepped vertices...
        // Including front facing vertices

        for (unsigned s = 1; s <= steps; s++) {

            for (unsigned i = 0; i < vlen; i++) {

                const auto vert = bevelEnabled ? scalePt2(vertices[i], verticesMovements[i], bs) : vertices[i];

                if (!extrudeByPath) {

                    v(placeholder, vert.x, vert.y, depth / static_cast<float>(steps * s));

                } else {

                    // v( vert.x, vert.y + extrudePts[ s - 1 ].y, extrudePts[ s - 1 ].x );

                    normal.copy(splineTube.normals[s]).multiplyScalar(vert.x);
                    binormal.copy(splineTube.binormals[s]).multiplyScalar(vert.y);

                    position2.copy(extrudePts[s]).add(normal).add(binormal);

                    v(placeholder, position2.x, position2.y, position2.z);
                }
            }
        }

        // Add bevel segments planes

        //for ( b = 1; b <= bevelSegments; b ++ ) {
        for (int b = bevelSegments - 1; b >= 0; b--) {

            const auto t = static_cast<float>(b) / static_cast<float>(bevelSegments);
            const auto z = bevelThickness * std::cos(t * math::PI / 2);
            const auto bs = bevelSize * std::sin(t * math::PI / 2) + bevelOffset;

            // contract shape

            for (unsigned i = 0, il = contour.size(); i < il; i++) {

                const auto vert = scalePt2(contour[i], contourMovements[i], bs);
                v(placeholder, vert.x, vert.y, depth + z);
            }

            // expand holes

            for (unsigned h = 0, hl = holes.size(); h < hl; h++) {

                const auto& ahole = holes[h];
                oneHoleMovements = holesMovements[h];

                for (unsigned i = 0, il = ahole.size(); i < il; i++) {

                    const auto vert = scalePt2(ahole[i], oneHoleMovements[i], bs);

                    if (!extrudeByPath) {

                        v(placeholder, vert.x, vert.y, depth + z);

                    } else {

                        v(placeholder, vert.x, vert.y + extrudePts[steps - 1].y, extrudePts[steps - 1].x + z);
                    }
                }
            }
        }


        /* Faces */

        std::function<void()> buildLidFaces = [&] {
            int start = static_cast<int>(verticesArray.size() / 3);

            if (bevelEnabled) {

                unsigned int layer = 0;// steps + 1
                unsigned int offset = vlen * layer;

                // Bottom faces

                for (unsigned i = 0; i < flen; i++) {

                    const auto& face = faces[i];
                    f3(uvArray, verticesArray, placeholder, face[2] + offset, face[1] + offset, face[0] + offset);
                }

                layer = steps + bevelSegments * 2;
                offset = vlen * layer;

                // Top faces

                for (unsigned i = 0; i < flen; i++) {

                    const auto& face = faces[i];
                    f3(uvArray, verticesArray, placeholder, face[0] + offset, face[1] + offset, face[2] + offset);
                }

            } else {

                // Bottom faces

                for (unsigned i = 0; i < flen; i++) {

                    const auto& face = faces[i];
                    f3(uvArray, verticesArray, placeholder, face[2], face[1], face[0]);
                }

                // Top faces

                for (unsigned i = 0; i < flen; i++) {

                    const auto& face = faces[i];
                    f3(uvArray, verticesArray, placeholder, face[0] + vlen * steps, face[1] + vlen * steps, face[2] + vlen * steps);
                }
            }

            addGroup(start, static_cast<int>(verticesArray.size() / 3) - start, 0);
        };


        // Top and bottom faces

        buildLidFaces();

        // Sides faces

        std::function<void(const std::vector<Vector2>&, unsigned int)> sidewalls = [&](const std::vector<Vector2>& contour, unsigned int layeroffset) {
            int i = static_cast<int>(contour.size());

            while (--i >= 0) {

                const auto j = i;
                auto k = i - 1;
                if (k < 0) k = static_cast<int>(contour.size()) - 1;

                for (unsigned s = 0, sl = (steps + bevelSegments * 2); s < sl; s++) {

                    const auto slen1 = vlen * s;
                    const auto slen2 = vlen * (s + 1);

                    const auto a = layeroffset + j + slen1,
                               b = layeroffset + k + slen1,
                               c = layeroffset + k + slen2,
                               d = layeroffset + j + slen2;

                    f4(uvArray, verticesArray, placeholder, a, b, c, d);
                }
            }
        };

        // Create faces for the z-sides of the shape

        std::function<void()> buildSideFaces = [&] {
            int start = static_cast<int>(verticesArray.size() / 3);
            unsigned int layeroffset = 0;
            sidewalls(contour, layeroffset);
            layeroffset += contour.size();

            for (auto& ahole : holes) {

                sidewalls(ahole, layeroffset);

                //, true
                layeroffset += ahole.size();
            }


            addGroup(start, static_cast<int>(verticesArray.size() / 3) - start, 1);
        };

        buildSideFaces();
    };


    for (auto shape : shapes) {

        addShape(*shape);
    }

    // build geometry

    this->setAttribute("position", FloatBufferAttribute::create(verticesArray, 3));
    this->setAttribute("uv", FloatBufferAttribute::create(uvArray, 2));

    this->computeVertexNormals();
}

std::shared_ptr<ExtrudeGeometry> ExtrudeGeometry::create(const Shape& shape, const ExtrudeGeometry::Options& options) {

    return std::shared_ptr<ExtrudeGeometry>(new ExtrudeGeometry({&shape}, options));
}

std::shared_ptr<ExtrudeGeometry> ExtrudeGeometry::create(const std::vector<std::shared_ptr<Shape>>& shape, const ExtrudeGeometry::Options& options) {

    std::vector<const Shape*> ptrs(shape.size());
    std::transform(shape.begin(), shape.end(), ptrs.begin(), [&](auto& s) { return s.get(); });

    return std::shared_ptr<ExtrudeGeometry>(new ExtrudeGeometry(ptrs, options));
}

std::string ExtrudeGeometry::type() const {

    return "ExtrudeGeometry";
}

ExtrudeGeometry::Options::Options()
    : curveSegments(12), steps(1), depth(1),
      bevelEnabled(true), bevelThickness(0.2f), bevelSize(bevelThickness - 0.1f),
      bevelOffset(0), bevelSegments(3), extrudePath(nullptr) {}
