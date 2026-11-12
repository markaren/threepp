// Shared by the sensor chain's kernels (sensor_rgb.comp, sensor_evs.comp):
// the view's picture through a lens remap, and a counter-based RNG.

// One output pixel's lens tap: the base-resolution source pixel of the
// bilinear footprint's top-left texel (-1: outside the render) and the
// footprint's fractions. Built on the host from the lens model.
struct Tap {
    int   i00;
    float fx;
    float fy;
    float pad;
};

// The view's picture as the swapchain format holds it: tone-mapped, sRGB
// encoded, 0..1 per channel. The view is rendered `ss` samples a side per
// base pixel; a base pixel is the mean of its ss x ss samples.
vec3 basePixelEncoded(ivec2 b, uint ss) {
    vec3 sum = vec3(0.0);
    for (uint j = 0u; j < ss; ++j)
        for (uint i = 0u; i < ss; ++i)
            sum += imageLoad(srcImg, b * int(ss) + ivec2(int(i), int(j))).rgb;
    return sum / float(ss * ss);
}

float srgbDecode(float c) {
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

float srgbEncode(float l) {
    l = clamp(l, 0.0, 1.0);
    return l <= 0.0031308 ? 12.92 * l : 1.055 * pow(l, 1.0 / 2.4) - 0.055;
}

vec3 basePixelLinear(ivec2 b, uint ss) {
    vec3 sum = vec3(0.0);
    for (uint j = 0u; j < ss; ++j)
        for (uint i = 0u; i < ss; ++i) {
            const vec3 c = imageLoad(srcImg, b * int(ss) + ivec2(int(i), int(j))).rgb;
            sum += vec3(srgbDecode(c.r), srgbDecode(c.g), srgbDecode(c.b));
        }
    return sum / float(ss * ss);
}

// PCG hash (Jarzynski & Olano 2020): a stream per pixel and frame.
uint pcgHash(uint v) {
    const uint state = v * 747796405u + 2891336453u;
    const uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rnd(inout uint s) {
    s = pcgHash(s);
    return (float(s >> 8u) + 0.5) * (1.0 / 16777216.0);// (0, 1)
}

float rndNormal(inout uint s) {
    const float u1 = rnd(s), u2 = rnd(s);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

// Poisson count of mean lambda: Knuth's product for small means, a rounded
// normal past 30 (a hot pixel at its top rate over a long tick).
int rndPoisson(inout uint s, float lambda) {
    if (lambda <= 0.0) return 0;
    if (lambda > 30.0) return max(0, int(floor(lambda + sqrt(lambda) * rndNormal(s) + 0.5)));
    const float L = exp(-lambda);
    int   k = 0;
    float p = rnd(s);
    while (p > L && k < 200) {
        ++k;
        p *= rnd(s);
    }
    return k;
}
