// What the editor's curve overlays (SplineOverlay.cpp, ConveyorOverlay.cpp)
// share: the byte hash their rebuild keys are made of, and the in-place line
// write.

#ifndef THREEPP_EDITOR_OVERLAYLINES_HPP
#define THREEPP_EDITOR_OVERLAYLINES_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Line.hpp"

#include <cstddef>
#include <vector>

namespace threepp::editor::overlay {

    // Mixes `size` bytes into `seed`.
    inline void hashBytes(std::size_t& seed, const void* data, std::size_t size) {

        const auto* bytes = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            seed ^= static_cast<std::size_t>(bytes[i]) + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
        }
    }

    // Writes the points into the line IN PLACE, the way
    // examples/extras/curves/spline_editor.cpp always has: the renderer caches
    // GPU buffers by attribute identity, so replacing the attribute on every
    // rebuild (what setFromPoints does) can hand it a recycled pointer with the
    // same fresh version count — which reads as already uploaded, and the drawn
    // curve freezes at its first shape until something destroys the whole line.
    // The attribute is only replaced when the points OUTGROW it, and then the
    // whole geometry is swapped and the old one disposed, so the renderer
    // provably lets go of the stale buffers. The tail beyond the count still
    // holds whatever a longer line left there; the draw range hides it.
    //
    // Takes Line by reference — LineSegments IS one, so the conveyor's aids
    // buffer fits through the same door (a template here tripped clang's
    // dependent-name rules on the CI runners; MSVC was merely permissive).
    inline void writeLinePoints(Line& line, int& capacity, const std::vector<Vector3>& points) {

        const auto count = static_cast<int>(points.size());
        if (count > capacity) {
            const auto old = line.geometry();
            auto geometry = BufferGeometry::create();
            geometry->setAttribute("position", FloatBufferAttribute::create(
                                                       std::vector<float>(points.size() * 3), 3));
            line.setGeometry(geometry);
            if (old) old->dispose();
            capacity = count;
        }
        auto* position = line.geometry()->getAttribute<float>("position");
        for (int i = 0; i < count; ++i) {
            position->setXYZ(i, points[i].x, points[i].y, points[i].z);
        }
        position->needsUpdate();
        line.geometry()->drawRange = {0, count};
    }

}// namespace threepp::editor::overlay

#endif//THREEPP_EDITOR_OVERLAYLINES_HPP
