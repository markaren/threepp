// The wake field's TILES: the 8 x 8 texel tiles of a producer's patch that
// its step writes (wake_tiles.comp lists them, wake_field.comp steps them).
// vulkan::WakeFieldPipeline::State::tiles holds, a word a tile of each
// producer's patch, two arrays of flags (last step's and this step's) and the
// step's list, and a word an 8 x 8 cell of its ripples' steepest slope
// (wake_ripple.comp, SLOPES).

#ifndef THREEPP_OCEAN_WAKE_TILES_GLSL
#define THREEPP_OCEAN_WAKE_TILES_GLSL

// A tile's flags.
#define WAKE_TILE_DIRTY 1u// the step wrote the tile: its images may hold water
#define WAKE_TILE_NZ    2u// the water it holds after the step is not all still (SETTLE)
// A list entry's: the tile takes zeros. The last step wrote it, and this
// step leaves no water in it.
#define WAKE_TILE_CLEAR 0x80000000u

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer WakeTileWords {
    uint w[];
};

#endif// THREEPP_OCEAN_WAKE_TILES_GLSL
