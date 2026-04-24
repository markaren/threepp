#include "threepp/renderers/wgpu/pathtracer/WgpuPathTracerGeometry.hpp"
#include "WgpuPathTracerAtlas.hpp"

#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/textures/Texture.hpp"

#include <algorithm>
#include <cstring>
#include <utility>

namespace threepp::wgpu_pt {

int buildGeometryBuffers(
        const std::vector<RtMeshEntry>& entries,
        const std::unordered_map<Texture*, int>& texSlotMap,
        std::vector<float>& triBuffer,
        std::vector<float>& matBuffer,
        std::vector<float>& rawObjTriBuf,
        std::vector<float>& matrixBuf,
        int maxTris, int maxMats, int maxMeshes,
        int triOffset, int matOffset, int meshOffset,
        std::vector<std::pair<int, int>>* entryTriRanges) {
    // Full build: clear everything.  Append mode (offsets > 0): preserve existing data.
    if (triOffset == 0 && matOffset == 0 && meshOffset == 0) {
        std::ranges::fill(triBuffer, 0.f);
        std::ranges::fill(matBuffer, 0.f);
        std::ranges::fill(rawObjTriBuf, 0.f);
        std::ranges::fill(matrixBuf, 0.f);
    }

    int triCount = triOffset;
    int matCount = matOffset;
    int meshCount = meshOffset;

    if (entryTriRanges) {
        entryTriRanges->assign(entries.size(), {triCount, 0});
    }

    // Deduplicate by Material* so meshes sharing the same material get the same matIdx,
    // enabling the denoiser to share samples across them.
    std::unordered_map<Material*, int> meshToMatIdx;

    auto setTexel = [&](std::vector<float>& buf, int width, int col, int row,
                        float x, float y, float z, float w) {
        int idx;
        if (width > TEX_PAGE_WIDTH) {
            const int page = col / TEX_PAGE_WIDTH;
            const int pcol = col % TEX_PAGE_WIDTH;
            idx = ((page * TRI_TEX_HEIGHT + row) * TEX_PAGE_WIDTH + pcol) * 4;
        } else {
            idx = (row * width + col) * 4;
        }
        buf[idx + 0] = x;
        buf[idx + 1] = y;
        buf[idx + 2] = z;
        buf[idx + 3] = w;
    };
    auto setObj = [&](int ti, int field, float x, float y, float z, float w) {
        float* p = rawObjTriBuf.data() + ti * 32 + field * 4;
        p[0] = x; p[1] = y; p[2] = z; p[3] = w;
    };

    for (std::size_t eIdx = 0; eIdx < entries.size(); ++eIdx) {
        auto& entry = entries[eIdx];
        const int entryTriStart = triCount;
        if (entryTriRanges) {
            (*entryTriRanges)[eIdx] = {entryTriStart, 0};
        }
        if (triCount >= maxTris || meshCount >= maxMeshes) break;

        int matIdx;
        auto matIt = meshToMatIdx.find(entry.mesh->material().get());
        if (matIt != meshToMatIdx.end()) {
            matIdx = matIt->second;
        } else {
            if (matCount >= maxMats) continue;
            matIdx = matCount++;
            meshToMatIdx[entry.mesh->material().get()] = matIdx;

            auto em = extractMaterial(entry.mesh->material().get());
            setTexel(matBuffer, maxMats, matIdx, 0,
                     em.albedo.r, em.albedo.g, em.albedo.b, em.shininess);

            float texSlot = -1.f;
            float normalSlot = -1.f;
            if (auto* mwm = dynamic_cast<MaterialWithMap*>(entry.mesh->material().get())) {
                if (mwm->map) {
                    auto it = texSlotMap.find(mwm->map.get());
                    if (it != texSlotMap.end()) {
                        texSlot = encodeSlotWrap(it->second, mwm->map.get());
                    }
                }
            }
            if (auto* mnm = dynamic_cast<MaterialWithNormalMap*>(entry.mesh->material().get())) {
                if (mnm->normalMap) {
                    auto it = texSlotMap.find(mnm->normalMap.get());
                    if (it != texSlotMap.end()) {
                        normalSlot = encodeSlotWrap(it->second, mnm->normalMap.get());
                    }
                }
            }
            float roughSlot = -1.f;
            if (auto* mwr = dynamic_cast<MaterialWithRoughness*>(entry.mesh->material().get())) {
                if (mwr->roughnessMap) {
                    auto it = texSlotMap.find(mwr->roughnessMap.get());
                    if (it != texSlotMap.end()) {
                        roughSlot = encodeSlotWrap(it->second, mwr->roughnessMap.get());
                    }
                }
            }
            setTexel(matBuffer, maxMats, matIdx, 1, texSlot, em.metalness, normalSlot, roughSlot);
            setTexel(matBuffer, maxMats, matIdx, 2,
                    em.emissive.r, em.emissive.g, em.emissive.b, em.transmission);
            // sideFlag: 0 = Side::Front (cull back faces)
            //           1 = Side::Double (no culling) — also forced for glass
            //           2 = Side::Back (cull front faces, flip shading normal)
            // Glass must always be double-sided for refraction to work, so
            // transmission > 0 overrides the material's nominal side.
            float sideFlag;
            if (em.transmission > 0.f) {
                sideFlag = 1.f;
            } else {
                switch (entry.mesh->material()->side) {
                    case Side::Double: sideFlag = 1.f; break;
                    case Side::Back:   sideFlag = 2.f; break;
                    case Side::Front:
                    default:           sideFlag = 0.f; break;
                }
            }
            const float opacity = std::clamp(entry.mesh->material()->opacity, 0.f, 1.f);
            // Encode blend mode: negative opacity signals stochastic alpha (BLEND mode).
            // We must keep BLEND mode any time `transparent=true` without alphaTest,
            // even if scalar opacity==1.0 — because the baseColor texture may still
            // carry a non-trivial alpha channel (logos, decals, cutouts drawn with
            // soft edges). The GPU path has per-texel early-outs that silently
            // promote alpha≥0.99 texels to opaque (zero variance) while preserving
            // the transparent regions, so the logo shape survives without the
            // stochastic-noise cost on solid paint.
            const float opacityEnc = (entry.mesh->material()->transparent && em.alphaTest <= 0.f)
                                     ? -opacity : opacity;
            setTexel(matBuffer, maxMats, matIdx, 3, em.ior, em.alphaTest, sideFlag, opacityEnc);
            setTexel(matBuffer, maxMats, matIdx, 4,
                    em.attenuationColor.r, em.attenuationColor.g, em.attenuationColor.b, em.attenuationDistance);
            float emissiveSlot = -1.f;
            if (auto* mwe = dynamic_cast<MaterialWithEmissive*>(entry.mesh->material().get())) {
                if (mwe->emissiveMap) {
                    auto it = texSlotMap.find(mwe->emissiveMap.get());
                    if (it != texSlotMap.end()) {
                        emissiveSlot = encodeSlotWrap(it->second, mwe->emissiveMap.get());
                    }
                }
            }
            float aoSlot = -1.f;
            if (auto* mwa = dynamic_cast<MaterialWithAoMap*>(entry.mesh->material().get())) {
                if (mwa->aoMap) {
                    auto it = texSlotMap.find(mwa->aoMap.get());
                    if (it != texSlotMap.end()) {
                        aoSlot = encodeSlotWrap(it->second, mwa->aoMap.get());
                    }
                }
            }
            setTexel(matBuffer, maxMats, matIdx, 5, em.clearcoat, em.clearcoatRoughness, emissiveSlot, aoSlot);

            // Per-channel UV transforms (rows 6-15, 2 rows per channel)
            // Layout per channel: row N = (a, b, tx, c), row N+1 = (d, ty, texCoord, 0)
            // where u' = a*u + b*v + tx,  v' = c*u + d*v + ty
            auto writeUvTransform = [&](int row, const Texture* tex, float extraW = 0.f) {
                if (tex) {
                    const_cast<Texture*>(tex)->updateMatrix();
                    const auto& e = tex->matrix.elements;
                    // Column-major: e[0]=a, e[3]=b, e[6]=tx, e[1]=c, e[4]=d, e[7]=ty
                    setTexel(matBuffer, maxMats, matIdx, row,
                             e[0], e[3], e[6], e[1]);
                    setTexel(matBuffer, maxMats, matIdx, row + 1,
                             e[4], e[7], static_cast<float>(tex->texCoord), extraW);
                } else {
                    // Identity transform, UV0
                    setTexel(matBuffer, maxMats, matIdx, row, 1.f, 0.f, 0.f, 0.f);
                    setTexel(matBuffer, maxMats, matIdx, row + 1, 1.f, 0.f, 0.f, extraW);
                }
            };

            auto* mat = entry.mesh->material().get();
            auto* mwm = dynamic_cast<MaterialWithMap*>(mat);
            auto* mnm = dynamic_cast<MaterialWithNormalMap*>(mat);
            auto* mwr = dynamic_cast<MaterialWithRoughness*>(mat);
            auto* mwe = dynamic_cast<MaterialWithEmissive*>(mat);
            auto* mwa = dynamic_cast<MaterialWithAoMap*>(mat);

            const Texture* uvTextures[5] = {
                mwm ? mwm->map.get() : nullptr,
                mwr ? mwr->roughnessMap.get() : nullptr,
                mnm ? mnm->normalMap.get() : nullptr,
                mwe ? mwe->emissiveMap.get() : nullptr,
                mwa ? mwa->aoMap.get() : nullptr
            };
            const float normalScaleY = mnm ? mnm->normalScale.y : 1.0f;
            writeUvTransform(6,  uvTextures[0]);                    // baseColor
            writeUvTransform(8,  uvTextures[1]);                    // metalRough
            writeUvTransform(10, uvTextures[2], normalScaleY);      // normal (W = normalScale.y)
            writeUvTransform(12, uvTextures[3]);                    // emissive
            writeUvTransform(14, uvTextures[4]);                    // occlusion

            // Check if any channel uses non-identity UV transform or UV1
            float hasCustomUV = 0.f;
            for (const auto* tex : uvTextures) {
                if (tex) {
                    const_cast<Texture*>(tex)->updateMatrix();
                    const auto& e = tex->matrix.elements;
                    // Identity check: a=1, b=0, tx=0, c=0, d=1, ty=0
                    if (e[0] != 1.f || e[3] != 0.f || e[6] != 0.f ||
                        e[1] != 0.f || e[4] != 1.f || e[7] != 0.f ||
                        tex->texCoord != 0) {
                        hasCustomUV = 1.f;
                        break;
                    }
                }
            }

            // Row 16: sheen (r, g, b, roughness)
            setTexel(matBuffer, maxMats, matIdx, 16,
                    em.sheenColor.r, em.sheenColor.g, em.sheenColor.b, em.sheenRoughness);
            // Row 17: PBR specular (r, g, b, intensity)
            setTexel(matBuffer, maxMats, matIdx, 17,
                    em.specularColor.r, em.specularColor.g, em.specularColor.b, em.specularIntensity);
            // Row 18: dispersion + thickness + hasCustomUV + hasAdvancedPBR flags
            const bool hasAdvanced = (em.sheenColor.r != 0.f || em.sheenColor.g != 0.f || em.sheenColor.b != 0.f ||
                                      em.specularIntensity != 1.f ||
                                      em.specularColor.r != 1.f || em.specularColor.g != 1.f || em.specularColor.b != 1.f ||
                                      em.dispersion != 0.f || em.thickness != 0.f);
            setTexel(matBuffer, maxMats, matIdx, 18, em.dispersion, em.thickness, hasCustomUV, hasAdvanced ? 1.f : 0.f);
        }

        const int meshIdx = meshCount++;

        // Per-entry world matrix
        const auto& world = entry.worldMatrix;
        Matrix4 normalMat(world);
        normalMat.invert().transpose();

        if (meshIdx < maxMeshes) {
            float* mp = matrixBuf.data() + meshIdx * 32;
            std::memcpy(mp, world.elements.data(), 16 * sizeof(float));
            std::memcpy(mp + 16, normalMat.elements.data(), 16 * sizeof(float));
        }

        auto* geo = entry.mesh->geometry().get();
        auto* pos = geo->getAttribute<float>("position");
        if (!pos) continue;
        auto* nrm = geo->getAttribute<float>("normal");
        auto* uvs = geo->getAttribute<float>("uv");
        auto* idx = geo->getIndex();

        auto vert = [&](int i) {
            Vector3 v(pos->getX(i), pos->getY(i), pos->getZ(i));
            v.applyMatrix4(world);
            return v;
        };
        auto norm = [&](int i) -> Vector3 {
            if (!nrm) return {0.f, 1.f, 0.f};
            Vector3 n(nrm->getX(i), nrm->getY(i), nrm->getZ(i));
            n.transformDirection(normalMat);
            return n;
        };
        auto objVert = [&](int i) -> Vector3 {
            return {pos->getX(i), pos->getY(i), pos->getZ(i)};
        };
        auto objNorm = [&](int i) -> Vector3 {
            if (!nrm) return {0.f, 1.f, 0.f};
            return {nrm->getX(i), nrm->getY(i), nrm->getZ(i)};
        };
        auto uv = [&](int i) -> std::pair<float, float> {
            if (!uvs) return {0.f, 0.f};
            return {uvs->getX(i), uvs->getY(i)};
        };
        auto vi = [&](int tri, int corner) -> int {
            return idx ? static_cast<int>(idx->getX(tri * 3 + corner)) : tri * 3 + corner;
        };

        const int nTris = idx ? static_cast<int>(idx->count()) / 3
                              : static_cast<int>(pos->count()) / 3;
        for (int i = 0; i < nTris && triCount < maxTris; ++i) {
            const int i0 = vi(i, 0), i1 = vi(i, 1), i2 = vi(i, 2);
            const Vector3 v0 = vert(i0), v1 = vert(i1), v2 = vert(i2);
            const Vector3 n0 = norm(i0), n1 = norm(i1), n2 = norm(i2);
            const auto [u0, v0uv] = uv(i0);
            const auto [u1, v1uv] = uv(i1);
            const auto [u2, v2uv] = uv(i2);

            // Use paged layout (matches triGet/pagedIdx/GPU triCoord)
            auto setTri = [&](int row, float x, float y, float z, float w) {
                int idx = pagedIdx(triCount, row);
                triBuffer[idx + 0] = x; triBuffer[idx + 1] = y;
                triBuffer[idx + 2] = z; triBuffer[idx + 3] = w;
            };
            setTri(0, v0.x, v0.y, v0.z, static_cast<float>(matIdx));
            setTri(1, v1.x, v1.y, v1.z, 0.f);
            setTri(2, v2.x, v2.y, v2.z, 0.f);
            setTri(3, n0.x, n0.y, n0.z, 0.f);
            setTri(4, n1.x, n1.y, n1.z, 0.f);
            setTri(5, n2.x, n2.y, n2.z, 0.f);
            setTri(6, u0, v0uv, u1, v1uv);
            setTri(7, u2, v2uv, 0.f, 0.f);

            const Vector3 ov0 = objVert(i0), ov1 = objVert(i1), ov2 = objVert(i2);
            const Vector3 on0 = objNorm(i0), on1 = objNorm(i1), on2 = objNorm(i2);
            setObj(triCount, 0, ov0.x, ov0.y, ov0.z, static_cast<float>(matIdx));
            setObj(triCount, 1, ov1.x, ov1.y, ov1.z, static_cast<float>(meshIdx));
            setObj(triCount, 2, ov2.x, ov2.y, ov2.z, 0.f);
            setObj(triCount, 3, on0.x, on0.y, on0.z, 0.f);
            setObj(triCount, 4, on1.x, on1.y, on1.z, 0.f);
            setObj(triCount, 5, on2.x, on2.y, on2.z, 0.f);
            setObj(triCount, 6, u0, v0uv, u1, v1uv);
            setObj(triCount, 7, u2, v2uv, 0.f, 0.f);

            ++triCount;
        }

        if (entryTriRanges) {
            (*entryTriRanges)[eIdx] = {entryTriStart, triCount - entryTriStart};
        }
    }
    return triCount;
}

}// namespace threepp::wgpu_pt
