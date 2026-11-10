# OceanFoamTile.inc

The detail tile crest foam is drawn with (`DisplacedMesh::Params::crestFoam`,
deferred shade binding 92): 512 x 512 bytes, one a texel, packed into
little-endian 64-bit words. `VulkanCoreTextures.cpp` embeds it.

Source photograph: "Ocean foam drone view" by Tommy Lisbin, Huntington Beach,
California, 2017. Published on Unsplash before 5 June 2017 under the Creative
Commons CC0 1.0 Universal Public Domain Dedication; taken from Wikimedia
Commons, `File:Ocean_foam_drone_view_(Unsplash).jpg`
(https://commons.wikimedia.org/wiki/File:Ocean_foam_drone_view_(Unsplash).jpg).

What was done to it (`make_ocean_foam_tile.py`, which needs numpy, scipy and
Pillow and takes the photograph and an output folder):

1. the foam over open water is cropped out (1700 x 1700 px of the 4000 x 3000 frame);
2. linear luminance, minus its own blur over many foam cells, divided by the
   local contrast: foam density rather than brightness;
3. resampled and cross-faded over a 30 % overlap in x and y so the tile wraps,
   the fade normalised so the seam keeps its contrast;
4. equalised by rank, so the fraction of the tile above any level is that level.
