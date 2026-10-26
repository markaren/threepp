// The lowbias32 integer hash the ParticleField device emitter
// (particle_emit.comp) and its billboards (particlefield_billboard.vert) draw
// every per-slot random stream from; each mixes slot, stream and seed into it
// in its own rnd01. particle_emit.comp says why it is a stateless hash.

#ifndef PARTICLE_HASH_GLSL
#define PARTICLE_HASH_GLSL

uint hashU(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

#endif// PARTICLE_HASH_GLSL
