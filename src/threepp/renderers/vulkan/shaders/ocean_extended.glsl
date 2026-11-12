// the extended ocean's master switch in the shaders (THREEPP_OCEAN_EXTENDED, OceanExtended.hpp).
//
// kOceanExt: 1 = the extended ocean clouds, ocean/water, foam and fog (the default; the host passes it);
//            0 = upstream's own code paths, which every branch on it keeps verbatim. The default here is
//            0 so a pipeline the host builds without the entry is upstream's, and with the switch off the
//            host passes no entry at all: the pipeline's specialization is upstream's.
// kWaterGbufNormal: 1 = a water pixel carries its shading normal's x/z in the G-buffer's motion.w / uv.w
//            (signed half floats; the precise underwater normal), THREEPP_OCEAN_WATER_GBUFFER_NORMAL;
//            0 = upstream's packing (motion.w = 0, uv.w = the aoMap occlusion) and upstream's AO read.
//            The host sets it only with kOceanExt.
#ifndef THREEPP_OCEAN_EXTENDED_GLSL
#define THREEPP_OCEAN_EXTENDED_GLSL
layout(constant_id = 21) const uint kOceanExtended   = 0u;
layout(constant_id = 22) const uint kWaterGbufNormal = 0u;
const bool kOceanExt    = kOceanExtended != 0u;
const bool kWaterGbufN  = kWaterGbufNormal != 0u;
#endif// THREEPP_OCEAN_EXTENDED_GLSL
