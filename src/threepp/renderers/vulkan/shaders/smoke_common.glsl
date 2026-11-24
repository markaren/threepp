// Shared by every smoke_*.comp: the grid smoke solver of a ParticleField's GridSmoke
// (include/threepp/objects/ParticleField.hpp, src/.../SmokeGrid.cpp).
//
// ONE descriptor set layout for the whole solver, so a single set per
// (velocity, scalar, pressure) ping-pong parity serves every pass and the host
// never rewrites a set. A pass names only the bindings it reads.
//
//   uVel  / oVel    velocity, rgba16f, m/s                  (xyz)
//   uSca  / oSca    scalars,  rg16f:  x = density, y = temperature
//   uP    / oP      pressure, r32f
//   aux / uAux      scratch,  rgba16f: vorticity (curl, |curl|), the
//                   MacCormack predictor, then the divergence
//   Params          sources + colliders + clock (vkCmdUpdateBuffer each frame)
//   oDensity        the field's r32ui sigma_t volume (the final pass only)
//
// The grid is `res` cells per axis over the world box [box, box + res / invCell];
// cell i has its centre at box + (i + 0.5) / invCell. Cells are NOT assumed to
// be cubes: every derivative takes the per-axis spacing from invCell.

layout(local_size_x = 8, local_size_y = 4, local_size_z = 4) in;

layout(set = 0, binding = 0) uniform sampler3D uVel;
layout(set = 0, binding = 1, rgba16f) uniform writeonly image3D oVel;
layout(set = 0, binding = 2) uniform sampler3D uSca;
layout(set = 0, binding = 3, rg16f) uniform writeonly image3D oSca;
layout(set = 0, binding = 4) uniform sampler3D uP;
layout(set = 0, binding = 5, r32f) uniform writeonly image3D oP;
layout(set = 0, binding = 6, rgba16f) uniform image3D aux;
layout(set = 0, binding = 7) uniform sampler3D uAux;

struct Source {
    vec4 posRadius;// xyz world, w radius (m)
    vec4 rates;    // x density / s, y temperature / s, z velocity blend / s
    vec4 vel;      // xyz emission velocity (m/s), w noise amplitude (0..1)
};
struct Collider {
    vec4 a;  // xyz capsule end A (world), w radius
    vec4 b;  // xyz capsule end B (world)
    vec4 vel;// xyz surface velocity (m/s)
};
// The colliders as the solver sees them, rasterised once a frame by smoke_solid.comp: xyz = the solid's velocity,
// w = 1 where a collider covers the cell's centre. Every pass reads this instead of walking the collider list for
// each of its six neighbours.
layout(set = 0, binding = 9, rgba16f) uniform image3D solidMask;

struct Box {
    vec4 lo;// xyz minimum corner (world)
    vec4 hi;// xyz maximum corner
    vec4 vel;
};
layout(set = 0, binding = 8, std430) readonly buffer Params {
    vec4     head;// x sources, y capsule colliders, z time (s), w boxes
    Source   src[16];   // kSmokeMaxSources in SmokeGrid.hpp
    Collider col[64];   // kSmokeMaxColliders
    Box      box[128];  // kSmokeMaxBoxes
};

layout(push_constant) uniform Pc {
    ivec4 res;    // xyz cells per axis
    vec4  invCell;// xyz 1 / cell size (1/m), w dt (s)
    vec4  box;    // xyz world minimum, w solid floor (0 / 1)
    vec4  phys0;  // x buoyancy, y weight, z -, w density dissipation (1/s)
    vec4  phys1;  // x cooling (1/s), y vorticity confinement, z velocity damping (1/s), w sigma_t in Q20.12 per unit density
    vec4  wind;   // xyz m/s, w speed cap (m/s)
} pc;

bool inGrid(ivec3 c) { return all(greaterThanEqual(c, ivec3(0))) && all(lessThan(c, pc.res.xyz)); }

vec3 worldOfCell(ivec3 c) { return pc.box.xyz + (vec3(c) + 0.5) / pc.invCell.xyz; }

// The cell-space position of a world point, in the units a sampler3D wants.
vec3 uvwOfWorld(vec3 w) { return (w - pc.box.xyz) * pc.invCell.xyz / vec3(pc.res.xyz); }

// Signed distance to the nearest collider (negative inside) and that collider's surface velocity.
float colliderSdf(vec3 w, out vec3 vel) {
    float best = 1e9;
    vel        = vec3(0.0);
    const int n = int(head.y);
    for (int i = 0; i < n; ++i) {
        const vec3  a  = col[i].a.xyz;
        const vec3  ab = col[i].b.xyz - a;
        const float h  = clamp(dot(w - a, ab) / max(dot(ab, ab), 1e-8), 0.0, 1.0);
        const float d  = length(w - (a + ab * h)) - col[i].a.w;
        if (d < best) {
            best = d;
            vel  = col[i].vel.xyz;
        }
    }
    const int nb = int(head.w);
    for (int i = 0; i < nb; ++i) {
        const vec3  c = 0.5 * (box[i].lo.xyz + box[i].hi.xyz);
        const vec3  h = 0.5 * (box[i].hi.xyz - box[i].lo.xyz);
        const vec3  q = abs(w - c) - h;
        const float d = length(max(q, vec3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
        if (d < best) {
            best = d;
            vel  = box[i].vel.xyz;
        }
    }
    return best;
}

// Is this cell solid (a collider covers its centre, or it is below a solid floor)? `vel` is what the solid does there.
bool solidCell(ivec3 c, out vec3 vel) {
    vel = vec3(0.0);
    if (c.y < 0 && pc.box.w > 0.5) return true;
    if (!inGrid(c) || head.y + head.w < 0.5) return false;
    const vec4 m = imageLoad(solidMask, c);
    vel          = m.xyz;
    return m.w > 0.5;
}

// Velocity of a cell, as a neighbour sees it: a solid's own velocity, the edge value beyond an open face.
vec3 velAt(ivec3 c) {
    vec3 sv;
    if (solidCell(c, sv)) return sv;
    return texelFetch(uVel, clamp(c, ivec3(0), pc.res.xyz - 1), 0).xyz;
}

// Pressure of a neighbour as the cell at `c0` sees it: a solid wall is a zero-gradient wall (its own
// pressure), the open boundary is p = 0.
float pressureAt(ivec3 c, float pCentre) {
    vec3 sv;
    if (solidCell(c, sv)) return pCentre;
    if (!inGrid(c)) return 0.0;
    return texelFetch(uP, c, 0).x;
}

float hash31(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.x + p.y) * p.z);
}

// Smooth value noise in [0, 1].
float vnoise(vec3 p) {
    const vec3 i = floor(p);
    vec3       f = fract(p);
    f            = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash31(i), hash31(i + vec3(1, 0, 0)), f.x),
                   mix(hash31(i + vec3(0, 1, 0)), hash31(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(hash31(i + vec3(0, 0, 1)), hash31(i + vec3(1, 0, 1)), f.x),
                   mix(hash31(i + vec3(0, 1, 1)), hash31(i + vec3(1, 1, 1)), f.x), f.y),
               f.z);
}
