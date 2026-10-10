// The PCG hash and the LCG-stepped uniform the deferred shade and the probe
// update draw their samples from. `rnd` advances `seed` as an LCG and returns
// the hashed output; the two are one function pair here so the shade's and
// the probes' streams stay the same stream. (froxel_inject.comp keeps its own
// variant on purpose: its rnd re-seeds with the hash, a different sequence.)

#ifndef PCG_RNG_GLSL
#define PCG_RNG_GLSL

uint pcgHash(uint v) {
    v = v * 747796405u + 2891336453u;
    const uint s = ((v >> ((v >> 28) + 4u)) ^ v) * 277803737u;
    return (s >> 22) ^ s;
}
float rnd(inout uint seed) {
    seed = seed * 747796405u + 2891336453u;
    uint r = ((seed >> ((seed >> 28) + 4u)) ^ seed) * 277803737u;
    r = (r >> 22) ^ r;
    return float(r) * (1.0 / 4294967296.0);
}

#endif// PCG_RNG_GLSL
