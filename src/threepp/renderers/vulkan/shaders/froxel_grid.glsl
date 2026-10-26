// The froxel grid: kFroxelX x kFroxelY screen tiles by kFroxelZ view-distance
// slices, spaced exponentially from kFroxelZMin to kFroxelZMax. froxel_inject
// and froxel_integrate walk it to build the LUT. The tile and slice counts are
// vulkan_shared.h's kFroxelGrid*, which the host also sizes the 3D images and
// the dispatches from.
//
// The LUT's readers, the deferred shade and particle_light.comp, look it up by
// view distance on the ladder alone and define FROXEL_GRID_LADDER_ONLY before
// the include: unused there, the counts would still take SPIR-V ids.

#ifndef FROXEL_GRID_GLSL
#define FROXEL_GRID_GLSL

#include "vulkan_shared.h"// kFroxelGridX/Y/Z

const float kFroxelZMin = 0.25, kFroxelZMax = 512.0;

#ifndef FROXEL_GRID_LADDER_ONLY
const uint  kFroxelX = kFroxelGridX, kFroxelY = kFroxelGridY, kFroxelZ = kFroxelGridZ;
float froxelDist(float s) {// continuous slice coord [0,kFroxelZ] → view distance
    return kFroxelZMin * pow(kFroxelZMax / kFroxelZMin, s / float(kFroxelZ));
}
#endif

#endif// FROXEL_GRID_GLSL
