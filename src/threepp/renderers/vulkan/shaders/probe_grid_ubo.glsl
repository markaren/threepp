// The probe grid's layout UBO (ProbeGI::GridUboData), shared by every shader
// that samples or updates the probes: deferred_shade.comp and
// particle_light.comp (binding 37 of the deferred set) and probe_update.comp
// (binding 0 of its own set). The includer sets PROBE_GRID_UBO_BINDING before
// including; the block is otherwise identical, which is the point: the host
// writes one struct and the layout drift is caught by one static_assert.
//
// Two layouts live in it (ProbeGI.hpp, "scene fit" and "cascades"):
//   cascadeCount == 1  one grid fitted to the scene AABB, as the renderer
//                      always had. origin/spacing/dims describe it and the
//                      cascade-0 entries repeat them.
//   cascadeCount  > 1  camera-centred, toroidally addressed cascades, finest
//                      first, each dims probes per axis at cSpacing[c]. The
//                      probe at window coordinate g (0..dims-1) is world cell
//                      w = cBase[c] + g, sits at cOrigin[c] + g * cSpacing[c],
//                      and is STORED at slot (w mod dims) of the cascade's
//                      range, which starts cBase[c].w probes into the buffers.
//                      Moving the window changes cBase; the slots it keeps go
//                      on holding the same world cells, and only the entering
//                      ones are reset (probe_update.comp's scroll mode, which
//                      reads cPrevBase).
#ifndef PROBE_GRID_UBO_BINDING
#define PROBE_GRID_UBO_BINDING 37
#endif
const int kProbeCascadesMax = 4;// must match ProbeGI::kCascadesMax
layout(set = 0, binding = PROBE_GRID_UBO_BINDING, scalar) uniform ProbeGridUbo {
    vec3  origin;  float enabled;      // cascade 0 (enabled gates all SAMPLING)
    vec3  spacing; float _pad0;
    ivec3 dims;    int   cascadeCount; // probes per axis, same for every cascade
    vec4  cOrigin[kProbeCascadesMax];  // xyz = world position of window coordinate (0,0,0)
    vec4  cSpacing[kProbeCascadesMax]; // xyz = probe spacing, w = probeMaxDist (1.5 * |spacing|)
    ivec4 cBase[kProbeCascadesMax];    // xyz = world cell of window coordinate (0,0,0), w = first probe index
    ivec4 cPrevBase[kProbeCascadesMax];// xyz = cBase before this frame's scroll (== cBase when none)
} probeGrid;
