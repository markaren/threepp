// Shared by every smoke_*.comp: the grid smoke solver of a ParticleField's GridSmoke
// (include/threepp/objects/ParticleField.hpp, src/.../SmokeGrid.cpp).
//
// ONE descriptor set layout for the whole solver, so a single set per
// (velocity, scalar, pressure) ping-pong parity serves every pass and the host
// never rewrites a set. A pass names only the bindings it reads.
//
//   uVel  / oVel    velocity, rgba16f, m/s, ON THE FACES (below), (res+1)^3;
//                   w = the gas the step's burn released in that cell (1/s),
//                   from smoke_forces.comp to the projection
//   uSca  / oSca    scalars,  rgba16f: x = density (the soot), y = temperature,
//                   z = fuel (unburnt, invisible), w = the oxygen used
//                   (0 = fresh air, 1 = none left)
//   uP    / oP      pressure, r32f
//   aux / uAux      scratch,  rgba16f: vorticity (curl, |curl|), the
//                   MacCormack predictor, then the divergence
//   Params          sources + colliders + the fire + clock (vkCmdUpdateBuffer each frame)
//   oDensity        the field's r32ui sigma_t volume (the final pass only)
//   oKelvin         the field's temperature volume for the glow (the final pass only)
//
// Outside the box all four scalars read 0: fresh air, with no fuel and no smoke
// in it, is what comes in through an open face.
//
// The grid is `res` cells per axis over the world box [box, box + res / invCell];
// cell i has its centre at box + (i + 0.5) / invCell. Cells are NOT assumed to
// be cubes: every derivative takes the per-axis spacing from invCell. "Cell
// coordinates" put cell i's centre at i + 0.5.
//
// THE VELOCITY IS STAGGERED (a MAC grid): each component lives on the faces it
// crosses. Texel (i, j, k) of the velocity image holds .x = u on the x face at
// the min side of cell (i, j, k), at cell coordinates (i, j + 0.5, k + 0.5); .y =
// v on its min y face; .z = w on its min z face. The texels at i, j or k = res
// hold the faces of the box's max sides; a component whose face does not exist
// (u at j = res, say) is held at 0. With the velocity there, the divergence a
// cell sees and the pressure gradient a face feels are the same differences, so
// the projection leaves no divergence the pressure solve did not; and a wall is
// a face the air cannot cross, not a cell to clamp beside.
//
// The velocity is stored RELATIVE TO THE WIND. Outside the box the air moves
// with the wind, which the border sampler reads as 0; inside, the wind is air
// like any other, so walls and colliders turn it.

layout(local_size_x = 8, local_size_y = 4, local_size_z = 4) in;

layout(set = 0, binding = 0) uniform sampler3D uVel;
layout(set = 0, binding = 1, rgba16f) uniform writeonly image3D oVel;
layout(set = 0, binding = 2) uniform sampler3D uSca;
layout(set = 0, binding = 3, rgba16f) uniform writeonly image3D oSca;
layout(set = 0, binding = 4) uniform sampler3D uP;
layout(set = 0, binding = 5, r32f) uniform writeonly image3D oP;
layout(set = 0, binding = 6, rgba16f) uniform image3D aux;
layout(set = 0, binding = 7) uniform sampler3D uAux;

struct Source {
    vec4 posRadius;// xyz world, w radius (m)
    vec4 rates;    // x density / s, y temperature / s, z velocity blend / s, w fuel / s
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
    // The fire (ParticleField::GridSmoke). burn0: x ignition temperature, y burn rate (1/s), z heat release
    // (temperature per unit of fuel burned), w soot yield (density per unit of fuel burned). burn1: x oxygen per fuel
    // (the share of a cell's oxygen one unit of fuel uses), y expansion (gas released per unit of fuel burned, as a
    // share of the cell), z kelvin per temperature unit, w the ambient (K).
    vec4     burn0;
    vec4     burn1;
    Source   src[16];   // kSmokeMaxSources in SmokeGrid.hpp
    Collider col[64];   // kSmokeMaxColliders
    Box      box[128];  // kSmokeMaxBoxes
};

layout(push_constant) uniform Pc {
    ivec4 res;    // xyz cells per axis
    vec4  invCell;// xyz 1 / cell size (1/m), w dt (s)
    vec4  box;    // xyz world minimum, w solid floor (0 / 1)
    vec4  phys0;  // x buoyancy, y weight, z boundary fade, w density dissipation (1/s)
    vec4  phys1;  // x cooling (1/s), y vorticity confinement, z velocity damping (1/s), w sigma_t in Q20.12 per unit density
    vec4  wind;   // xyz m/s, w speed cap (m/s)
    vec4  turb;   // x the wind's eddies' speed (m/s, rms per axis; 0 = smooth air), y their size (m)
} pc;

bool inGrid(ivec3 c) { return all(greaterThanEqual(c, ivec3(0))) && all(lessThan(c, pc.res.xyz)); }

// A texel of the velocity image (the faces): 0..res on every axis.
bool inFaces(ivec3 c) { return all(greaterThanEqual(c, ivec3(0))) && all(lessThanEqual(c, pc.res.xyz)); }

// Does texel c hold a face on `axis` (0..res along it, 0..res-1 across it)?
bool faceExists(int axis, ivec3 c) {
    ivec3 lim = pc.res.xyz;
    lim[axis] += 1;
    return all(greaterThanEqual(c, ivec3(0))) && all(lessThan(c, lim));
}

ivec3 unitAxis(int axis) {
    ivec3 e = ivec3(0);
    e[axis] = 1;
    return e;
}

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

// Is face `axis` of texel c a solid's (a solid cell on either side of it, the solid floor included)? `vel` is then
// that solid's velocity along the axis, relative to the wind: the air there moves with the solid, no faster and no
// slower, so no air goes into it. A face between a solid and the open air beyond the box is the solid's too.
bool solidFace(int axis, ivec3 c, out float vel) {
    vec3       sa, sb;
    const bool a = solidCell(c - unitAxis(axis), sa);
    const bool b = solidCell(c, sb);
    vel          = 0.0;
    if (!a && !b) return false;
    const vec3 s = (a && b) ? 0.5 * (sa + sb) : (a ? sa : sb);
    vel          = s[axis] - pc.wind[axis];
    return true;
}

// A face's velocity (relative to the wind), a solid's face moving with its solid. `c` must hold the face.
float faceVel(int axis, ivec3 c) {
    float sv;
    if (solidFace(axis, c, sv)) return sv;
    return texelFetch(uVel, c, 0)[axis];
}

// The face's centre, in world space.
vec3 worldOfFace(int axis, ivec3 c) {
    vec3 p = vec3(c) + 0.5;
    p[axis] -= 0.5;
    return pc.box.xyz + p / pc.invCell.xyz;
}

// The air's velocity (relative to the wind) at a cell centre: the mean of its two faces on each axis. A solid cell
// moves with its solid; beyond an open face of the box, the edge cell's.
vec3 cellVel(ivec3 c) {
    vec3 sv;
    if (solidCell(c, sv)) return sv - pc.wind.xyz;
    c            = clamp(c, ivec3(0), pc.res.xyz - 1);
    const vec3 v = texelFetch(uVel, c, 0).xyz;
    return 0.5 * vec3(v.x + texelFetch(uVel, c + ivec3(1, 0, 0), 0).x,
                      v.y + texelFetch(uVel, c + ivec3(0, 1, 0), 0).y,
                      v.z + texelFetch(uVel, c + ivec3(0, 0, 1), 0).z);
}

// One velocity component (relative to the wind) anywhere, `p` in cell coordinates: trilinear over that component's
// own faces. Beyond the box the border reads 0: the air there is the wind.
float sampleFace(int axis, vec3 p) {
    p[axis] += 0.5;
    return textureLod(uVel, p / vec3(pc.res.xyz + 1), 0.0)[axis];
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

// ── the wind's eddies ───────────────────────────────────────────────────────
// One component of the curl of a vector potential made of three smooth noises (offset copies of vnoise), by central
// differences: a velocity pattern with no divergence of its own. `p` is in units of the eddy size.
float curlNoiseAxis(int axis, vec3 p) {
    const float e   = 0.15;
    const vec3  o1  = vec3(31.4, 47.2, 11.9), o2 = vec3(-17.8, 23.1, 59.3);
    const vec3  ex  = vec3(e, 0.0, 0.0), ey = vec3(0.0, e, 0.0), ez = vec3(0.0, 0.0, e);
    if (axis == 0) return (vnoise(p + ey + o2) - vnoise(p - ey + o2) - vnoise(p + ez + o1) + vnoise(p - ez + o1)) / (2.0 * e);
    if (axis == 1) return (vnoise(p + ez) - vnoise(p - ez) - vnoise(p + ex + o2) + vnoise(p - ex + o2)) / (2.0 * e);
    return (vnoise(p + ex + o1) - vnoise(p - ex + o1) - vnoise(p + ey) + vnoise(p - ey)) / (2.0 * e);
}

// The eddies' velocity along `axis` at world point w (m/s, relative to the wind): two octaves, the large eddies
// carrying most of the energy, frozen into the air and carried by the wind. The constant makes each component's rms
// pc.turb.x (measured over the noise in numpy: the two octaves' sum has an rms of 0.523 before it).
float gustAxis(int axis, vec3 w) {
    const vec3 q = (w - pc.wind.xyz * head.z) / pc.turb.y;
    return pc.turb.x * (1.0 / 0.523) * (curlNoiseAxis(axis, q) + 0.25 * 2.3 * curlNoiseAxis(axis, q * 2.3 + 7.7));
}
