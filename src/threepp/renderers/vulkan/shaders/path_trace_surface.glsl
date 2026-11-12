// The path tracer's surface at a ray hit: what gbuffer.frag makes of a raster
// fragment, rebuilt from the triangle the ray found (path_trace.comp).
//
//   - the geometric normal, the interpolated vertex normal and the uv;
//   - the tangent along u from the triangle's positions and uvs, the
//     per-triangle form of gbuffer.frag's screen-derivative frame (dp/du,
//     Gram-Schmidt against the normal, B = N x T), so a normal map bends the
//     traced normal as it bends the raster's;
//   - the per-vertex colour (GeometryDesc.colorAddress) on the albedo;
//   - the world-anchored detail layer (MaterialWithDetailMap) and the terrain
//     bands (MaterialWithTerrainMaps): the same triplanar projections,
//     stochastic tiling and height blend as gbuffer.frag, read at the finest
//     mip. The raster fades both layers out where a repeat shrinks under a
//     pixel, which is a filter for one sample a pixel; a trace that takes
//     hundreds of samples a pixel integrates the texture itself, so it keeps
//     them everywhere (the mean of a 0.5-neutral layer is neutral anyway).
//
// Needs from the includer: geometry_fetch.glsl and its geoms[], the bindless
// albedoMaps[], kMaxMaterialTextures and GL_EXT_nonuniform_qualifier.

#ifndef PATH_TRACE_SURFACE_GLSL
#define PATH_TRACE_SURFACE_GLSL

layout(buffer_reference, scalar) readonly buffer PtColorBuf { float c[]; };

struct PtHit {
    vec3 Ng;  // geometric normal, world, unit (the triangle's winding side)
    vec3 Ns;  // interpolated vertex normal, world, unit
    vec3 dpdu;// world tangent along u (not normalised); zero without uvs
    vec2 uv;
    vec3 vcol;// per-vertex colour, (1,1,1) without one
};

PtHit ptFetchHit(int instIdx, int primId, vec2 bary, mat4x3 objToWorld, mat4x3 worldToObj) {
    PtHit h;
    const GeometryDesc g   = geoms[instIdx];
    const uvec3        idx = gfetchTri(g, primId);
    const float        w   = 1.0 - bary.x - bary.y;

    PosBuf      pb = PosBuf(g.vertexAddress);
    const vec3  p0 = vec3(pb.p[idx.x * 3u], pb.p[idx.x * 3u + 1u], pb.p[idx.x * 3u + 2u]);
    const vec3  p1 = vec3(pb.p[idx.y * 3u], pb.p[idx.y * 3u + 1u], pb.p[idx.y * 3u + 2u]);
    const vec3  p2 = vec3(pb.p[idx.z * 3u], pb.p[idx.z * 3u + 1u], pb.p[idx.z * 3u + 2u]);
    const mat3  nrm = transpose(mat3(worldToObj));// object -> world for normals
    const mat3  lin = mat3(objToWorld);           // object -> world for vectors
    const vec3  e1 = p1 - p0;
    const vec3  e2 = p2 - p0;
    vec3        ng = nrm * cross(e1, e2);
    const float ngl = length(ng);
    h.Ng = ngl > 0.0 ? ng / ngl : vec3(0.0, 1.0, 0.0);

    const vec3 n0 = gfetchNormal(g, idx.x);
    const vec3 n1 = gfetchNormal(g, idx.y);
    const vec3 n2 = gfetchNormal(g, idx.z);
    const vec3 ns = nrm * (w * n0 + bary.x * n1 + bary.y * n2);
    const float nsl = length(ns);
    h.Ns = nsl > 1e-12 ? ns / nsl : h.Ng;

    h.uv   = vec2(0.0);
    h.dpdu = vec3(0.0);
    if (g.uvAddress != 0ul) {
        const vec2 u0 = gfetchUv(g, idx.x);
        const vec2 u1 = gfetchUv(g, idx.y);
        const vec2 u2 = gfetchUv(g, idx.z);
        h.uv = w * u0 + bary.x * u1 + bary.y * u2;
        const vec2  d1  = u1 - u0;
        const vec2  d2  = u2 - u0;
        const float det = d1.x * d2.y - d2.x * d1.y;
        if (abs(det) > 1e-20) h.dpdu = lin * ((e1 * d2.y - e2 * d1.y) / det);
    }

    h.vcol = vec3(1.0);
    if ((g.flags & kGeomFlagVertexColor) != 0u && g.colorAddress != 0ul) {
        if ((g.flags & kGeomFlagPackedColor) != 0u) {
            PackedAttrBuf cb = PackedAttrBuf(g.colorAddress);
            h.vcol = w * unpackUnorm4x8(cb.w[idx.x]).rgb + bary.x * unpackUnorm4x8(cb.w[idx.y]).rgb +
                     bary.y * unpackUnorm4x8(cb.w[idx.z]).rgb;
        } else {
            PtColorBuf cb = PtColorBuf(g.colorAddress);
            h.vcol = w * vec3(cb.c[idx.x * 3u], cb.c[idx.x * 3u + 1u], cb.c[idx.x * 3u + 2u]) +
                     bary.x * vec3(cb.c[idx.y * 3u], cb.c[idx.y * 3u + 1u], cb.c[idx.y * 3u + 2u]) +
                     bary.y * vec3(cb.c[idx.z * 3u], cb.c[idx.z * 3u + 1u], cb.c[idx.z * 3u + 2u]);
        }
    }
    return h;
}

vec4 ptTex(int texIndex, vec2 uv) {
    const int i = clamp(texIndex, 0, int(kMaxMaterialTextures) - 1);
    return textureLod(albedoMaps[nonuniformEXT(i)], uv, 0.0);
}

// gbuffer.frag's tangent-space normal map: T from dp/du, B = N x T, the map's
// xy scaled by normalScale and z rebuilt.
vec3 ptNormalMap(const MaterialDesc m, vec3 N, vec3 dpdu, vec2 uv) {
    if (m.normalTexIndex < 0) return N;
    vec3        T  = dpdu - dot(dpdu, N) * N;
    const float tl = length(T);
    if (tl <= 1e-12) return N;
    T /= tl;
    const vec3 B  = cross(N, T);
    vec3       ns = ptTex(m.normalTexIndex, (m.uvTransformNormal * vec3(uv, 1.0)).xy).rgb * 2.0 - 1.0;
    ns.xy *= m.normalScale;
    ns.z = sqrt(max(0.0, 1.0 - dot(ns.xy, ns.xy)));
    return normalize(T * ns.x + B * ns.y + N * ns.z);
}

// ── Stochastic tiling (Deliot & Heitz 2019), as gbuffer.frag runs it ────────
struct PtLattice {
    vec2 v1, v2, v3;
    vec3 w;
};

PtLattice ptLattice(vec2 duv) {
    PtLattice  lt;
    const mat2 kSkew  = mat2(1.0, 0.0, -0.57735027, 1.15470054);
    const vec2 skewed = kSkew * (duv * 1.154700538);
    const vec2 baseId = floor(skewed);
    vec3       bary   = vec3(fract(skewed), 0.0);
    bary.z            = 1.0 - bary.x - bary.y;
    if (bary.z > 0.0) {
        lt.w  = vec3(bary.z, bary.y, bary.x);
        lt.v1 = baseId; lt.v2 = baseId + vec2(0, 1); lt.v3 = baseId + vec2(1, 0);
    } else {
        lt.w  = vec3(-bary.z, 1.0 - bary.y, 1.0 - bary.x);
        lt.v1 = baseId + vec2(1, 1); lt.v2 = baseId + vec2(1, 0); lt.v3 = baseId + vec2(0, 1);
    }
    return lt;
}

// gbuffer.frag's detailHashExact: the exact integer hash of a lattice id.
vec2 ptLatticeHash(vec2 p) {
    const uvec2 q = uvec2(ivec2(p));
    uint        h = q.x * 0x8da6b343u + q.y * 0xd8163841u + 0x9e3779b9u;
    h ^= h >> 16; h *= 0x7feb352du;
    h ^= h >> 15; h *= 0x846ca68bu;
    h ^= h >> 16;
    return vec2(float(h & 0xffffu), float(h >> 16)) * (1.0 / 65536.0);
}

vec4 ptStochastic(int idx, vec2 duv, PtLattice lt) {
    const vec4 d1 = ptTex(idx, duv + ptLatticeHash(lt.v1));
    const vec4 d2 = ptTex(idx, duv + ptLatticeHash(lt.v2));
    const vec4 d3 = ptTex(idx, duv + ptLatticeHash(lt.v3));
    return vec4(0.5) + ((d1 - 0.5) * lt.w.x + (d2 - 0.5) * lt.w.y + (d3 - 0.5) * lt.w.z) *
                               inversesqrt(max(dot(lt.w, lt.w), 1e-6));
}

// One projection of a 0.5-neutral layer: the albedo overlay (x2 around the
// neutral, by strength), the bent normal, the roughness multiplier and the
// height in the albedo's alpha (terrain bands bake it there).
struct PtLayer {
    vec3  overlay;
    vec3  normal;
    float roughMul;
    float height;
};

PtLayer ptLayerProject(vec2 duv, vec3 axisU, vec3 axisV, vec3 N, int albIdx, int nrmIdx,
                       float strength, float nScale, float rStrength, bool stochastic) {
    PtLayer c;
    c.overlay  = vec3(1.0);
    c.normal   = N;
    c.roughMul = 1.0;
    c.height   = 0.5;
    PtLattice lt;
    if (stochastic) lt = ptLattice(duv);
    if (albIdx >= 0) {
        const vec4 det = stochastic ? ptStochastic(albIdx, duv, lt) : ptTex(albIdx, duv);
        c.overlay = mix(vec3(1.0), det.rgb * 2.0, strength);
        c.height  = det.a;
    }
    if (nrmIdx >= 0) {
        const vec4  dn = stochastic ? ptStochastic(nrmIdx, duv, lt) : ptTex(nrmIdx, duv);
        vec3        T  = axisU - N * dot(N, axisU);
        vec3        B  = axisV - N * dot(N, axisV);
        const float tl = length(T), bl = length(B);
        if (tl > 1e-4 && bl > 1e-4) {
            T /= tl;
            B /= bl;
            const vec2  nxy = (dn.xy - 0.5) * 2.0 * nScale;
            const float nz  = sqrt(max(1e-4, 1.0 - dot(nxy, nxy)));
            c.normal = normalize(T * nxy.x + B * nxy.y + N * nz);
        }
        c.roughMul = mix(1.0, 2.0 * dn.a, rStrength);
    }
    return c;
}

// The triplanar weights gbuffer.frag takes from the geometric normal: |Ng|^7,
// minor axes under 8 % of the dominant one dropped, normalised.
vec3 ptTriplanarWeights(vec3 Ngeo) {
    vec3 aw = pow(abs(Ngeo), vec3(7.0));
    aw = max(aw - 0.08 * max(aw.x, max(aw.y, aw.z)), vec3(0.0));
    return aw / max(aw.x + aw.y + aw.z, 1e-6);
}

// All three projections of one layer at `rep` repeats a metre, blended by aw;
// the dominant one stochastic, the minors one plain tap each.
PtLayer ptLayerTriplanar(vec3 P, float rep, vec3 aw, vec3 N, int albIdx, int nrmIdx,
                         float strength, float nScale, float rStrength) {
    const float maxw = max(aw.x, max(aw.y, aw.z));
    vec3  ov = vec3(0.0), nn = vec3(0.0);
    float rm = 0.0, hh = 0.0, pw = 0.0;
    if (aw.y > 0.001) {
        const PtLayer c = ptLayerProject(P.xz * rep, vec3(1, 0, 0), vec3(0, 0, 1), N, albIdx, nrmIdx,
                                         strength, nScale, rStrength, aw.y >= maxw - 1e-6);
        ov += aw.y * c.overlay; nn += aw.y * c.normal; rm += aw.y * c.roughMul; hh += aw.y * c.height; pw += aw.y;
    }
    if (aw.z > 0.001) {
        const PtLayer c = ptLayerProject(P.xy * rep, vec3(1, 0, 0), vec3(0, 1, 0), N, albIdx, nrmIdx,
                                         strength, nScale, rStrength, aw.z >= maxw - 1e-6);
        ov += aw.z * c.overlay; nn += aw.z * c.normal; rm += aw.z * c.roughMul; hh += aw.z * c.height; pw += aw.z;
    }
    if (aw.x > 0.001) {
        const PtLayer c = ptLayerProject(P.zy * rep, vec3(0, 0, 1), vec3(0, 1, 0), N, albIdx, nrmIdx,
                                         strength, nScale, rStrength, aw.x >= maxw - 1e-6);
        ov += aw.x * c.overlay; nn += aw.x * c.normal; rm += aw.x * c.roughMul; hh += aw.x * c.height; pw += aw.x;
    }
    PtLayer r;
    if (pw <= 1e-4) {
        r.overlay = vec3(1.0); r.normal = N; r.roughMul = 1.0; r.height = 0.5;
        return r;
    }
    const float ipw = 1.0 / pw;
    r.overlay  = ov * ipw;
    r.normal   = normalize(nn);
    r.roughMul = rm * ipw;
    r.height   = hh * ipw;
    return r;
}

// The shaded surface after every layer gbuffer.frag applies: albedo (vertex
// colour included), roughness, metalness and the shading normal. `N` comes in
// as the side-corrected vertex normal and `Ngeo` is that normal before any
// map (the projections' orientation).
struct PtSurface {
    vec3  albedo;
    float roughness;// < 0: unlit
    float metalness;
    vec3  N;
};

PtSurface ptSurface(const MaterialDesc m, const PtHit h, vec3 P, vec3 N, vec3 Ngeo) {
    PtSurface s;
    N = ptNormalMap(m, N, h.dpdu, h.uv);

    // Terrain world-space normal map: replaces the normal outright.
    if (m.terrainNormalTexIndex >= 0) {
        const vec3  wTap = ptTex(m.terrainNormalTexIndex, h.uv).rgb * 2.0 - 1.0;
        const float wLen = length(wTap);
        if (wLen > 1e-3) N = wTap / wLen;
    }

    vec3 albedoSample = vec3(1.0);
    if (m.albedoTexIndex >= 0) albedoSample = ptTex(m.albedoTexIndex, (m.uvTransform * vec3(h.uv, 1.0)).xy).rgb;

    // Detail layer (MaterialWithDetailMap).
    float detailRoughMul = 1.0;
    if (m.detailTexIndex >= 0 || m.detailNormalTexIndex >= 0) {
        const PtLayer c = ptLayerTriplanar(P, m.detailRepeat, ptTriplanarWeights(Ngeo), N,
                                           m.detailTexIndex, m.detailNormalTexIndex,
                                           m.detailStrength, m.detailNormalScale, m.detailRoughStrength);
        if (m.detailTexIndex >= 0) albedoSample *= c.overlay;
        if (m.detailNormalTexIndex >= 0) {
            N = c.normal;
            detailRoughMul = c.roughMul;
        }
    }

    // Terrain bands (MaterialWithTerrainMaps): up to four band sets picked by
    // the weight map, height-blended, over the macro colour.
    float terrainRough = -1.0;
    if (m.terrainWeightTexIndex >= 0) {
        const vec4  w4       = ptTex(m.terrainWeightTexIndex, h.uv);
        const float coverage = min(w4.x + w4.y + w4.z + w4.w, 1.0);
        if (coverage > 0.01) {
            const vec3 aw = ptTriplanarWeights(Ngeo);
            vec3  bandOv[4];
            vec3  bandN[4];
            float bandRM[4];
            float bandH[4];
            float bandW[4];
            float wSum = 0.0;
            for (int b = 0; b < 4; ++b) {
                bandW[b] = 0.0;
                const float wb = w4[b];
                if (wb < 0.03) continue;
                const int aIdx = m.terrainBandAlbedoTex[b];
                const int nIdx = m.terrainBandNormalTex[b];
                if (aIdx < 0 && nIdx < 0) continue;
                const PtLayer c = ptLayerTriplanar(P, m.terrainBandRepeat[b], aw, N, aIdx, nIdx,
                                                   m.terrainBandStrength, m.terrainNormalScale,
                                                   m.terrainRoughStrength);
                bandOv[b] = c.overlay;
                bandN[b]  = c.normal;
                bandRM[b] = c.roughMul;
                bandH[b]  = c.height;
                bandW[b]  = wb;
                wSum += wb;
            }
            if (wSum > 1e-3) {
                float sw[4];
                float sSum = 0.0;
                for (int b = 0; b < 4; ++b) {
                    sw[b] = bandW[b] > 0.0 ? bandW[b] * exp(m.terrainHeightBlend * (bandH[b] - 0.5)) : 0.0;
                    sSum += sw[b];
                }
                const float inv = 1.0 / max(sSum, 1e-5);
                vec3  ov = vec3(0.0), nn = vec3(0.0);
                float rBase = 0.0;
                for (int b = 0; b < 4; ++b) {
                    if (sw[b] <= 0.0) continue;
                    const float wgt = sw[b] * inv;
                    ov += wgt * bandOv[b];
                    nn += wgt * bandN[b];
                    rBase += wgt * m.terrainBandRough[b] * bandRM[b];
                }
                albedoSample *= mix(vec3(1.0), ov, coverage);
                N = normalize(mix(N, normalize(nn), coverage));
                terrainRough = mix(m.roughness, rBase, coverage);
            }
        }
        // The far tint, by the hit's distance from the view's eye whatever
        // bounce found it: the plants it stands in for are placed round the
        // eye, so a mirror shows the same ground the camera does.
        if (m.terrainFarInvRange > 0.0) {
            albedoSample *= terrainFarTint(m.terrainBandFarTint, m.terrainFarStart, m.terrainFarInvRange,
                                           w4, distance(P, cam.viewInverse[3].xyz));
        }
    }

    s.albedo = m.albedo * albedoSample * h.vcol;
    s.N      = N;

    float rough = m.roughness;
    float metal = m.metalness;
    if (rough >= 0.0) {
        const vec2 uvRM = (m.uvTransformRoughMetal * vec3(h.uv, 1.0)).xy;
        if (m.roughnessTexIndex >= 0) rough *= ptTex(m.roughnessTexIndex, uvRM).g;
        if (m.metalnessTexIndex >= 0) metal *= ptTex(m.metalnessTexIndex, uvRM).b;
        rough *= detailRoughMul;
        if (terrainRough >= 0.0) rough = terrainRough;
        rough = clamp(rough, 0.04, 1.0);
        metal = clamp(metal, 0.0, 1.0);
    }
    s.roughness = rough;
    s.metalness = metal;
    return s;
}

#endif// PATH_TRACE_SURFACE_GLSL
