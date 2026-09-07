// Height brushes for a terrain mesh — the kernels, with no UI in them.
//
// Everything here works on the (dim×dim) height lattice that TerrainConfig
// defines: vertex index iz*dim+ix, only Y ever written, X and Z the immutable
// grid. That makes the brushes headless-testable and keeps the viewport code in
// EditorApp down to "where did the pointer land, and which brush is armed".
//
// Three decisions worth knowing before reading:
//
//   Ray hits are a HEIGHTFIELD MARCH, not a triangle raycast. A 512² terrain is
//   half a million triangles and the pointer moves every frame; marching the
//   ray in the mesh's local space at half-cell steps and bisecting the first
//   sign change is O(steps), and it cannot snag on some other object that
//   happens to be under the cursor.
//
//   Normals mid-stroke are CENTRAL DIFFERENCES over the touched rect, not
//   computeVertexNormals over the whole mesh. On a lattice the analytic normal
//   is exact, and rebuilding 500k normals per mouse-move is the difference
//   between a brush and a slideshow.
//
//   One stroke is ONE undo entry. The caller snapshots the whole Y column on
//   press (a megabyte at 512², and bounded), and on release asks diff() for the
//   tight rect that actually moved. Undo restores heights byte-identically —
//   it replays no arithmetic.

#ifndef THREEPP_EDITOR_TERRAINSCULPT_HPP
#define THREEPP_EDITOR_TERRAINSCULPT_HPP

#include "threepp/math/Vector3.hpp"

#include <vector>

namespace threepp {

    class BufferGeometry;

}// namespace threepp

namespace threepp::editor {

    struct TerrainBrush {

        enum class Kind {
            Raise,  // ± strength, Shift (invert) digs
            Smooth, // toward the neighbourhood average
            Flatten // toward the height under the press
        };

        Kind kind = Kind::Raise;
        // World units. Sized against the default terrain rather than the
        // lattice, so changing the resolution does not change the brush.
        float radius = 12.f;
        // Metres per second at the centre of the brush for Raise; a rate in
        // 0..1-ish terms for the two blend brushes.
        float strength = 8.f;
        bool invert = false;

        static const char* label(Kind kind);
        static constexpr int kindCount = 3;
    };

    // Index ↔ world mapping for the height lattice, read OFF the geometry
    // rather than recomputed from worldSize. PlaneGeometry's vertex order and
    // the rotateX that lays it flat between them decide which way iz runs;
    // deriving x0/step from the actual attribute means this code does not have
    // to agree with that derivation, only observe it.
    struct TerrainLattice {

        int dim = 0;
        float x0 = 0.f, z0 = 0.f;// world position of lattice cell (0,0)
        float stepX = 1.f, stepZ = 1.f;// signed: either may run negative

        [[nodiscard]] bool valid() const { return dim >= 2; }
        [[nodiscard]] float worldX(float ix) const { return x0 + ix * stepX; }
        [[nodiscard]] float worldZ(float iz) const { return z0 + iz * stepZ; }
        // Fractional lattice coordinates for a world point (may fall outside).
        [[nodiscard]] float latticeX(float wx) const { return (wx - x0) / stepX; }
        [[nodiscard]] float latticeZ(float wz) const { return (wz - z0) / stepZ; }
        // The smaller of the two cell sizes, in world units — the march step
        // and the "one ring" the refresh grows by are both quoted in this.
        [[nodiscard]] float cellSize() const;

        // nullopt-ish: returns an invalid lattice when the geometry is not a
        // square grid of `dim`.
        [[nodiscard]] static TerrainLattice of(const BufferGeometry& geometry, int dim);
    };

    class TerrainSculpt {

    public:
        // Inclusive lattice-index rectangle. Empty until something is added.
        struct Rect {
            int x0 = 0, z0 = 0, x1 = -1, z1 = -1;

            [[nodiscard]] bool empty() const { return x1 < x0 || z1 < z0; }
            void add(int x, int z);
            // Grow by `ring` cells, clamped to [0, dim-1].
            void grow(int ring, int dim);
        };

        // One undo entry's worth of a stroke: the rect that moved and both
        // sides of it. Heights only — the geometry is rebuilt from them.
        struct Patch {
            int x0 = 0, z0 = 0, w = 0, h = 0;
            std::vector<float> before;
            std::vector<float> after;

            [[nodiscard]] bool empty() const { return w <= 0 || h <= 0; }
        };

        // Bilinear height at a world XZ. False when the point is off the patch.
        [[nodiscard]] static bool sample(const std::vector<float>& heights,
                                         const TerrainLattice& lattice,
                                         float wx, float wz, float& out);

        // First surface crossing along a ray given in the mesh's LOCAL space.
        // Marches at half a cell and bisects the sign change; `maxDistance` is
        // in the same units. False when the ray misses the patch entirely.
        [[nodiscard]] static bool raycast(const std::vector<float>& heights,
                                          const TerrainLattice& lattice,
                                          const Vector3& origin, const Vector3& direction,
                                          float maxDistance, Vector3& hit);

        // One brush tick at world (cx, cz). `dt` is seconds; `flattenTarget` is
        // the height captured when the stroke started (ignored by the others).
        // Returns the lattice rect it touched.
        static Rect apply(std::vector<float>& heights, const TerrainLattice& lattice,
                          const TerrainBrush& brush, float cx, float cz,
                          float dt, float flattenTarget);

        // Write `heights` into the geometry's Y over `rect` and recompute the
        // normals there by central differences. Bounds are NOT refreshed — that
        // is a stroke-end job (see the header note).
        static void refresh(BufferGeometry& geometry, const std::vector<float>& heights,
                            const TerrainLattice& lattice, const Rect& rect);

        // The tight rect in which `before` and `after` differ, with both sides
        // copied out of it. Empty patch when nothing moved.
        [[nodiscard]] static Patch diff(const std::vector<float>& before,
                                        const std::vector<float>& after, int dim);

        // Put one side of a patch back. `useBefore` picks which.
        static void applyPatch(std::vector<float>& heights, int dim,
                               const Patch& patch, bool useBefore);
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_TERRAINSCULPT_HPP
