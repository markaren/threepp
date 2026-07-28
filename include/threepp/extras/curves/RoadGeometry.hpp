// A road surface: the authored curve, offset both sides and TRIMMED.
//
// The road follows the spline the user drew, sample for sample — no fit, no
// consolidation, nothing that changes the shape they authored. Both edges are
// the exact offset polylines of the sampled centreline at ±`width`/2, level
// side to side with Y carried, in the curve's OWN space.
//
// An offset polyline develops a SWALLOWTAIL inside a bend tighter than the
// offset: the edge runs past the centre of curvature, turns back, and crosses
// itself. That loop is not part of the swept region — it is the artefact every
// road mesher has to answer for, and the answer here is to CUT IT OUT. Each
// edge is scanned for local self-intersection, the loop between the two
// crossing segments is removed, and the edge continues from the crossing point
// itself. So the outer edge of a bend is smooth the whole way round, and the
// inner edge is smooth except for one crisp corner exactly where the swept
// region really has one. No fans, no pinning, no narrowing.
//
// Trimming breaks the ring correspondence a ribbon relies on — one edge loses
// vertices the other keeps — so the surface is stitched between the two edge
// polylines directly: a two-chain walk that advances whichever edge is behind
// in arc length. Around a trim that becomes a fan from the corner vertex out
// to the smooth edge opposite it, which is what the region looks like there.
//
// Vertices are laid out as the whole left edge and then the whole right edge.
// UVs run u = arc length / uvLength along the road (a trim vertex interpolates
// the parameter it cuts at) and v = 0 on the left edge, 1 on the right.
//
// Only LOCAL self-intersection is trimmed. A road whose two far-apart lobes
// overlap is out of scope: the pieces are correct, their overlap is not
// resolved, and the same was true of the ribbon.

#ifndef THREEPP_ROADGEOMETRY_HPP
#define THREEPP_ROADGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/core/Curve.hpp"

#include <memory>

namespace threepp {

    class RoadGeometry: public BufferGeometry {

    public:
        struct Params {
            float width;
            // Samples along the curve. The road follows every one of them.
            unsigned int divisions;
            // Metres of road per U tile.
            float uvLength;
            bool closed;

            explicit Params(float width = 1,
                            unsigned int divisions = 64,
                            float uvLength = 1,
                            bool closed = false);
        };

        const float width;

        [[nodiscard]] std::string type() const override;

        // `path` is only read.
        static std::shared_ptr<RoadGeometry> create(const Curve3& path, const Params& params);

        static std::shared_ptr<RoadGeometry> create(const Curve3& path,
                                                    float width = 1,
                                                    unsigned int divisions = 64,
                                                    float uvLength = 1,
                                                    bool closed = false);

        // The closed solid a static collider is cooked from: `surface`, a copy
        // of it pushed `thickness` straight down with its winding reversed, and
        // a wall along every boundary edge. Every edge of the result is shared
        // by exactly two triangles — a surface on its own has no inside, and a
        // body moving fast enough passes through one between substeps whatever
        // the solver does about it.
        //
        // Vertices are welded by position first, so a closed road's seam
        // duplicate does not read as a boundary and grow a wall across the
        // road. Static colliders only: the result is a triangle mesh.
        [[nodiscard]] static std::shared_ptr<BufferGeometry> solid(const BufferGeometry& surface,
                                                                   float thickness);

    private:
        RoadGeometry(const Curve3& path, const Params& params);
    };

}// namespace threepp

#endif//THREEPP_ROADGEOMETRY_HPP
