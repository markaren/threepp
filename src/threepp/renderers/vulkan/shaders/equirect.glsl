// The environment equirect's texel -> direction mapping, for the passes that
// write it: sky_preetham.comp bakes mip 0, prefilter_env.comp the GGX mips.
// The includer defines PI and TWO_PI.

#ifndef EQUIRECT_GLSL
#define EQUIRECT_GLSL

// Equirect → world direction. Matches deferred_shade's sampleEnvLod UV convention
// (u = 0.5 + atan2(z,x)/2π, v = 0.5 + asin(y)/π) so a sky pixel at uv on
// mip 0 maps to the same direction we'd sample at on the prefiltered mips.
vec3 dirFromUV(vec2 uv) {
    const float phi   = (uv.x - 0.5) * TWO_PI;// longitude (about y-axis)
    const float theta = (uv.y - 0.5) * PI;    // latitude (asin(y))
    const float cosT  = cos(theta);
    return vec3(cosT * cos(phi), sin(theta), cosT * sin(phi));
}

#endif// EQUIRECT_GLSL
