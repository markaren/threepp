#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_GOOGLE_include_directive : enable

// Path-traced LIDAR — closest-hit shader.
//
// Evaluates the LIDAR equation in back-scatter form using a proper
// Cook-Torrance microfacet BRDF evaluated at the back-scatter geometry
// (sensor = transmitter = receiver, so L = V):
//
//     I = P_tx · f_back · cos θ · η(r) / r²
//
//     f_back = albedo·(1-metal)/π  +  F·D(N·N) / (4·cos θ)
//
// where
//   P_tx     = `laserPower` push constant (transmitter power, arbitrary unit)
//   D        = GGX normal distribution evaluated at H=N (the back-scatter
//              direction puts the half-vector parallel to the surface normal
//              when the beam strikes head-on, and ~off-axis as θ grows)
//   F        = Schlick Fresnel at cos θ (metals use albedo as F0)
//   cos θ    = beam · surface-normal (also N·H here, since H ≡ surface
//              tangent of the back-scatter geometry)
//   η(r)     = exp(-2σ_ext r)  Beer-Lambert round-trip atmospheric extinction
//   r        = slant range from sensor to hit point
//
// Then divided by `invReferenceIntensity` so a perpendicular 1.0-albedo
// surface at the reference range reads as 1.0 in the output.
//
// Why this matters for chrome:
//   roughness 0.05 → α² ≈ 6e-6 → D peak ≈ 5e4 at θ=0 and crashes to
//   ~0 by θ ≈ 5°. Real chrome saturates a LIDAR detector at perpendicular
//   incidence and returns essentially nothing elsewhere — that's the
//   "specular flash + huge dropout" signature automotive LIDAR engineers
//   spend years compensating for. Matte concrete (roughness 0.95) has a
//   flat D term, contributes almost entirely through the diffuse lobe,
//   and returns a smooth moderate-intensity surface — the bright stable
//   floor every LIDAR loves.
//
// Transmission (glass) reduces back-scatter proportionally — the fraction
// that transmits through escapes to the next surface. Multi-return handling
// for translucent layers is a future extension; this version reports the
// strongest single return.

#include "vulkan_shared.h"
#include "lidar_shared.h"

// Geometry descriptors mirror VulkanRenderer::Impl::GeometryDesc and the
// chit's own declaration in closest_hit.rchit — single source of truth would
// be nice but the chit includes it inline, so we mirror.
struct GeometryDesc {
    uint64_t vertexAddress;
    uint64_t normalAddress;
    uint64_t indexAddress;
    uint64_t uvAddress;
    uint64_t foamAddress;
    uint64_t prevVertexAddress;
    uint     indexed;
    uint     _pad;
};

layout(buffer_reference, scalar) readonly buffer VertexBuf { float p[]; };
layout(buffer_reference, scalar) readonly buffer IndexBuf  { uint  i[]; };

layout(set = 0, binding = 1, scalar) readonly buffer GeomDescBuf {
    GeometryDesc geoms[];
};
layout(set = 0, binding = 2, scalar) readonly buffer MatDescBuf {
    MaterialDesc mats[];
};

layout(push_constant) uniform Pc {
    uint  numBeams;
    float maxRange;
    float laserPower;
    float invReferenceIntensity;
    float atmosphericExtinction;
    float detectorThreshold;
    uint  rngSeed;
    uint  _pad;
} pc;

struct Payload {
    vec3  hitPos;
    vec3  hitNormal;
    float distance;
    float intensity;
    int   instanceId;
};
layout(location = 0) rayPayloadInEXT Payload pl;

// Required by spec for any rchit invoked from a triangle hit group, even
// when we derive the normal from face geometry and never read the
// barycentrics directly.
hitAttributeEXT vec2 bary;

void main() {
    const uint instId = uint(gl_InstanceCustomIndexEXT);
    const GeometryDesc geom = geoms[instId];
    const MaterialDesc mat  = mats[instId];

    const uint primId = uint(gl_PrimitiveID);

    // Resolve the three vertex indices of the hit triangle.
    uint i0, i1, i2;
    if (geom.indexed != 0u) {
        IndexBuf ib = IndexBuf(geom.indexAddress);
        i0 = ib.i[primId * 3 + 0];
        i1 = ib.i[primId * 3 + 1];
        i2 = ib.i[primId * 3 + 2];
    } else {
        i0 = primId * 3 + 0;
        i1 = primId * 3 + 1;
        i2 = primId * 3 + 2;
    }

    // Fetch object-space positions and derive the face normal. A LIDAR
    // bounces off the actual surface plane — we deliberately do NOT use
    // smooth-shading per-vertex normals here; using face normals matches
    // real-sensor behaviour and avoids the "fake interpolated detail"
    // that smooth normals would inject into the intensity term.
    VertexBuf vb = VertexBuf(geom.vertexAddress);
    const vec3 p0 = vec3(vb.p[i0 * 3 + 0], vb.p[i0 * 3 + 1], vb.p[i0 * 3 + 2]);
    const vec3 p1 = vec3(vb.p[i1 * 3 + 0], vb.p[i1 * 3 + 1], vb.p[i1 * 3 + 2]);
    const vec3 p2 = vec3(vb.p[i2 * 3 + 0], vb.p[i2 * 3 + 1], vb.p[i2 * 3 + 2]);

    vec3 nObj = cross(p1 - p0, p2 - p0);
    // Degenerate triangles (collinear verts) produce a zero cross. Bail
    // with a no-hit signal — the rgen treats negative distance / negative
    // instanceId as a miss.
    const float nLen2 = dot(nObj, nObj);
    if (nLen2 < 1e-20) {
        pl.instanceId = -1;
        pl.distance   = 0.0;
        pl.intensity  = 0.0;
        return;
    }
    nObj *= inversesqrt(nLen2);

    // Object→world transform for the normal. For uniform-scaled meshes the
    // upper-3x3 of gl_ObjectToWorldEXT is correct; non-uniform scale would
    // need the inverse-transpose. Most scene meshes use uniform scale, and
    // the resulting renormalisation below masks small inaccuracies.
    const mat3 objToWorld = mat3(gl_ObjectToWorldEXT);
    vec3 nWorld = normalize(objToWorld * nObj);

    // Two-sided: flip the normal so it points toward the incoming ray.
    // sideMode 1 (Back-only) or 2 (Double) lets the sensor see back-faces
    // (e.g. the inside of a thin wall) and still get a valid intensity.
    const vec3 dWorld = gl_WorldRayDirectionEXT;
    float cosTheta = -dot(dWorld, nWorld);
    if (cosTheta < 0.0) {
        nWorld   = -nWorld;
        cosTheta = -cosTheta;
    }
    cosTheta = clamp(cosTheta, 0.0, 1.0);

    const float r        = gl_HitTEXT;
    const vec3  hitWorld = gl_WorldRayOriginEXT + r * gl_WorldRayDirectionEXT;

    // ── Back-scatter BRDF (Cook-Torrance @ L = V) ─────────────────────
    // Diffuse lobe: Lambert, view-independent. Albedo damped by metalness
    // (metals route their energy through the specular F·D term) AND by
    // transmission — diffuse return comes from sub-surface scattering,
    // which doesn't happen on clear glass / water, so transmissive
    // materials lose the diffuse path entirely.
    const vec3 diffuse = mat.albedo
                       * (1.0 - mat.metalness)
                       * (1.0 - mat.transmission)
                       * (1.0 / 3.14159265);

    // Specular lobe via GGX D evaluated at the back-scatter half-vector,
    // which equals the surface tangent of the incident beam (so N·H = cos θ).
    // α is clamped above zero so perfect mirrors don't explode to inf, but
    // the floor is *much* higher for transmissive surfaces. Visual rendering
    // intentionally sets water-like materials to very low roughness (e.g.
    // ocean uses 0.04) to keep the specular highlight crisp; LIDAR back-
    // scatter at that roughness would only fire on near-perfect perpendicular
    // hits and miss every wave surface that isn't aimed exactly at the sensor.
    // Real water has sub-mesh capillary-chop micro-roughness (~0.15-0.2 eff)
    // that the visual material doesn't model, so we approximate it here
    // with a heavier α floor for transmissive surfaces.
    const float baseAlpha = mat.roughness * mat.roughness;
    const float floorAlpha = (mat.transmission > 0.5) ? 0.04 : 2.0e-4;
    const float alpha    = max(baseAlpha, floorAlpha);
    const float a2       = alpha * alpha;
    const float NoH      = cosTheta;
    const float denomTr  = NoH * NoH * (a2 - 1.0) + 1.0;
    const float D        = a2 / (3.14159265 * denomTr * denomTr);

    // Schlick Fresnel; F0 = 0.04 for dielectrics, albedo for metals.
    const vec3  F0       = mix(vec3(0.04), mat.albedo, mat.metalness);
    const float fOneM   = pow(1.0 - cosTheta, 5.0);
    const vec3  F        = F0 + (vec3(1.0) - F0) * fOneM;

    // Microfacet specular at back-scatter: D · F / (4 · NoV · NoL) collapses
    // to D · F / (4 · cos² θ) because NoV = NoL = cos θ in this geometry.
    // We omit Smith G for simplicity — it's near 1 across the dominant
    // perpendicular hits and the visualisation tolerates the small bias.
    //
    // IMPORTANT: the specular term is NOT damped by (1 - transmission).
    // Fresnel reflection IS the fraction that doesn't transmit, by
    // definition — so for water at near-perpendicular incidence we still
    // return ~2-4% to the detector, and that fraction grows toward 100%
    // at grazing incidence (Schlick's pow(1-cosθ, 5) term). Damping
    // specular by (1 - T) on top of Fresnel would double-discount and
    // make every water surface invisible to the LIDAR.
    const vec3  specular = (D * F) / max(4.0 * cosTheta * cosTheta, 1e-6);

    const vec3  fBack    = diffuse + specular;

    // Near-IR LIDAR detectors don't resolve colour — use Rec. 709 luminance.
    const float rho      = 0.2126 * fBack.r + 0.7152 * fBack.g + 0.0722 * fBack.b;

    // Beer-Lambert round-trip atmospheric extinction.
    const float atm = exp(-2.0 * max(0.0, pc.atmosphericExtinction) * r);

    // LIDAR equation, normalised by the reference intensity. The cos θ
    // here is the projected-area term (irradiance on the surface); the
    // BRDF f_back already encodes the directional reflectance.
    float I = pc.laserPower * rho * cosTheta * atm / max(r * r, 1e-6);
    I *= pc.invReferenceIntensity;
    I = clamp(I, 0.0, 1.0);

    pl.hitPos     = hitWorld;
    pl.hitNormal  = nWorld;
    pl.distance   = r;
    pl.intensity  = I;
    pl.instanceId = int(instId);
}
