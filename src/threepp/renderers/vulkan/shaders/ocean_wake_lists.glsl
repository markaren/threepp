// The wake field's SOURCE LISTS: which of a step's sources each patch reads,
// as wake_field.comp (SETTLE) and wake_ripple.comp (TURN) loop over them.
// vulkan::WakeFieldPipeline::record writes them in the frame's command stream
// with the table, after the room the table buffer keeps for its sources
// (WakeFieldPipeline's kTableBytes), and mirrors the header as
// WakeSourceListsGpu.
//
// A list holds a patch's sources by their index in the table, in the table's
// order, and leaves out only the sources the loop would pass over at every
// texel of that patch: for SETTLE one that only pushes (the ripples' forcing)
// or whose reach (d^2 <= 24 r^2) does not meet the patch, for TURN one that
// does not push on that patch's waves. So the loop takes the few sources that
// can touch the patch, in the same order, and gives what a loop over every
// source of the step gives.

#ifndef THREEPP_OCEAN_WAKE_LISTS_GLSL
#define THREEPP_OCEAN_WAKE_LISTS_GLSL

#include "ocean_wake.glsl"

// = DisplacedMesh::kMaxWakeSources
#define OCEAN_WAKE_MAX_SOURCES 256
// Where the lists start in the table buffer: the table's 88-byte header and
// its patches (76 bytes each), then the room for its sources (72 bytes each).
#define OCEAN_WAKE_LISTS_OFFSET (88 + 76 * OCEAN_WAKE_MAX_PATCHES + 72 * OCEAN_WAKE_MAX_SOURCES)
// first[] and count[] hold the field's lists (SETTLE) at a patch's index and
// the ripples' (TURN) after them.
#define OCEAN_WAKE_LIST_FIELD  0
#define OCEAN_WAKE_LIST_RIPPLE OCEAN_WAKE_MAX_PATCHES

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer WakeSourceLists {
    uint first[2 * OCEAN_WAKE_MAX_PATCHES];// a list's first entry in index[]
    uint count[2 * OCEAN_WAKE_MAX_PATCHES];
    uint index[];                          // sources, by their index in the table
};

WakeSourceLists oceanWakeLists(uint64_t tableAddr) {
    return WakeSourceLists(tableAddr + uint64_t(OCEAN_WAKE_LISTS_OFFSET));
}

#endif// THREEPP_OCEAN_WAKE_LISTS_GLSL
