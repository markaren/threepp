#include "threepp/renderers/wgpu/pathtracer/WgpuPathTracerGeometry.hpp"
#include "WgpuPathTracerAtlas.hpp"

#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/textures/Texture.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

namespace threepp::wgpu_pt {

namespace {

void setMatTexel(std::vector<float>& buf, int width, int col, int row,
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
}

}// namespace

void writeMaterialRows(
        std::vector<float>& matBuffer,
        int maxMats,
        int matIdx,
        Material* mat,
        const std::unordered_map<Texture*, int>& texSlotMap) {

    auto em = extractMaterial(mat);
    setMatTexel(matBuffer, maxMats, matIdx, 0,
                em.albedo.r, em.albedo.g, em.albedo.b, em.shininess);

    float texSlot = -1.f;
    float normalSlot = -1.f;
    if (auto* mwm = dynamic_cast<MaterialWithMap*>(mat)) {
        if (mwm->map) {
            auto it = texSlotMap.find(mwm->map.get());
            if (it != texSlotMap.end()) {
                texSlot = encodeSlotWrap(it->second, mwm->map.get());
            }
        }
    }
    if (auto* mnm = dynamic_cast<MaterialWithNormalMap*>(mat)) {
        if (mnm->normalMap) {
            auto it = texSlotMap.find(mnm->normalMap.get());
            if (it != texSlotMap.end()) {
                normalSlot = encodeSlotWrap(it->second, mnm->normalMap.get());
            }
        }
    }
    float roughSlot = -1.f;
    if (auto* mwr = dynamic_cast<MaterialWithRoughness*>(mat)) {
        if (mwr->roughnessMap) {
            auto it = texSlotMap.find(mwr->roughnessMap.get());
            if (it != texSlotMap.end()) {
                roughSlot = encodeSlotWrap(it->second, mwr->roughnessMap.get());
            }
        }
    }
    setMatTexel(matBuffer, maxMats, matIdx, 1, texSlot, em.metalness, normalSlot, roughSlot);
    setMatTexel(matBuffer, maxMats, matIdx, 2,
                em.emissive.r, em.emissive.g, em.emissive.b, em.transmission);
    float sideFlag;
    if (em.transmission > 0.f) {
        sideFlag = 1.f;
    } else {
        switch (mat->side) {
            case Side::Double: sideFlag = 1.f; break;
            case Side::Back:   sideFlag = 2.f; break;
            case Side::Front:
            default:           sideFlag = 0.f; break;
        }
    }
    const float opacity = std::clamp(mat->opacity, 0.f, 1.f);
    const float opacityEnc = (mat->transparent && em.alphaTest <= 0.f) ? -opacity : opacity;
    setMatTexel(matBuffer, maxMats, matIdx, 3, em.ior, em.alphaTest, sideFlag, opacityEnc);
    setMatTexel(matBuffer, maxMats, matIdx, 4,
                em.attenuationColor.r, em.attenuationColor.g, em.attenuationColor.b, em.attenuationDistance);
    float emissiveSlot = -1.f;
    if (auto* mwe = dynamic_cast<MaterialWithEmissive*>(mat)) {
        if (mwe->emissiveMap) {
            auto it = texSlotMap.find(mwe->emissiveMap.get());
            if (it != texSlotMap.end()) {
                emissiveSlot = encodeSlotWrap(it->second, mwe->emissiveMap.get());
            }
        }
    }
    float aoSlot = -1.f;
    if (auto* mwa = dynamic_cast<MaterialWithAoMap*>(mat)) {
        if (mwa->aoMap) {
            auto it = texSlotMap.find(mwa->aoMap.get());
            if (it != texSlotMap.end()) {
                aoSlot = encodeSlotWrap(it->second, mwa->aoMap.get());
            }
        }
    }
    setMatTexel(matBuffer, maxMats, matIdx, 5, em.clearcoat, em.clearcoatRoughness, emissiveSlot, aoSlot);

    auto writeUvTransform = [&](int row, const Texture* tex, float extraW = 0.f) {
        if (tex) {
            const_cast<Texture*>(tex)->updateMatrix();
            const auto& e = tex->matrix.elements;
            setMatTexel(matBuffer, maxMats, matIdx, row,
                        e[0], e[3], e[6], e[1]);
            setMatTexel(matBuffer, maxMats, matIdx, row + 1,
                        e[4], e[7], static_cast<float>(tex->texCoord), extraW);
        } else {
            setMatTexel(matBuffer, maxMats, matIdx, row, 1.f, 0.f, 0.f, 0.f);
            setMatTexel(matBuffer, maxMats, matIdx, row + 1, 1.f, 0.f, 0.f, extraW);
        }
    };

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
    writeUvTransform(6,  uvTextures[0]);
    writeUvTransform(8,  uvTextures[1]);
    writeUvTransform(10, uvTextures[2], normalScaleY);
    writeUvTransform(12, uvTextures[3]);
    writeUvTransform(14, uvTextures[4]);

    float hasCustomUV = 0.f;
    for (const auto* tex : uvTextures) {
        if (tex) {
            const_cast<Texture*>(tex)->updateMatrix();
            const auto& e = tex->matrix.elements;
            if (e[0] != 1.f || e[3] != 0.f || e[6] != 0.f ||
                e[1] != 0.f || e[4] != 1.f || e[7] != 0.f ||
                tex->texCoord != 0) {
                hasCustomUV = 1.f;
                break;
            }
        }
    }

    setMatTexel(matBuffer, maxMats, matIdx, 16,
                em.sheenColor.r, em.sheenColor.g, em.sheenColor.b, em.sheenRoughness);
    setMatTexel(matBuffer, maxMats, matIdx, 17,
                em.specularColor.r, em.specularColor.g, em.specularColor.b, em.specularIntensity);
    const bool hasAdvanced = (em.sheenColor.r != 0.f || em.sheenColor.g != 0.f || em.sheenColor.b != 0.f ||
                              em.specularIntensity != 1.f ||
                              em.specularColor.r != 1.f || em.specularColor.g != 1.f || em.specularColor.b != 1.f ||
                              em.dispersion != 0.f || em.thickness != 0.f);
    setMatTexel(matBuffer, maxMats, matIdx, 18, em.dispersion, em.thickness, hasCustomUV, hasAdvanced ? 1.f : 0.f);
}

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
            writeMaterialRows(matBuffer, maxMats, matIdx,
                              entry.mesh->material().get(), texSlotMap);
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

int repackEntryObjTri(
        const RtMeshEntry& entry,
        int matIdx,
        int meshIdx,
        float* dstObjTri,
        int maxTris) {
    auto* geo = entry.mesh->geometry().get();
    auto* pos = geo->getAttribute<float>("position");
    if (!pos || maxTris <= 0) return 0;
    auto* nrm = geo->getAttribute<float>("normal");
    auto* uvs = geo->getAttribute<float>("uv");
    auto* idx = geo->getIndex();

    auto vi = [&](int tri, int corner) -> int {
        return idx ? static_cast<int>(idx->getX(tri * 3 + corner)) : tri * 3 + corner;
    };
    auto uv = [&](int i) -> std::pair<float, float> {
        if (!uvs) return {0.f, 0.f};
        return {uvs->getX(i), uvs->getY(i)};
    };

    // SkinnedMesh: pre-compute CPU-skinned positions + normals once per vertex.
    // Output is mesh-local space (bindMatrixInverse applied at the end), so the
    // VT pass's `mesh.matrixWorld` multiply lifts it correctly to world space.
    auto* skinnedMesh = dynamic_cast<SkinnedMesh*>(entry.mesh);
    const bool doSkin = skinnedMesh != nullptr
                        && skinnedMesh->skeleton != nullptr
                        && !skinnedMesh->skeleton->bones.empty();
    std::vector<Vector3> skinnedPos, skinnedNrm;
    if (doSkin) {
        auto* skinIdxAttr = geo->getAttribute<float>("skinIndex");
        auto* skinWAttr   = geo->getAttribute<float>("skinWeight");
        if (!skinIdxAttr || !skinWAttr) {
            // Missing skin attributes — fall back to static positions.
            // (shouldn't happen for properly built SkinnedMesh)
        } else {
            const int vtxCount = static_cast<int>(pos->count());
            skinnedPos.assign(vtxCount, Vector3{});
            if (nrm) skinnedNrm.assign(vtxCount, Vector3{});

            const auto& skel = *skinnedMesh->skeleton;
            const Matrix4& bindMat    = skinnedMesh->bindMatrix;
            const Matrix4& bindMatInv = skinnedMesh->bindMatrixInverse;
            const int boneCount = static_cast<int>(skel.bones.size());

            // Pre-compute per-bone skinning matrices: matrixWorld * boneInverse
            std::vector<Matrix4> boneMats(boneCount);
            for (int b = 0; b < boneCount; ++b) {
                if (skel.bones[b]) {
                    boneMats[b].multiplyMatrices(*skel.bones[b]->matrixWorld, skel.boneInverses[b]);
                }
            }

            Vector4 sIdx, sW;
            for (int i = 0; i < vtxCount; ++i) {
                skinIdxAttr->setFromBufferAttribute(sIdx, i);
                skinWAttr->setFromBufferAttribute(sW, i);

                // bindMatrix * position (point, w=1)
                Vector3 bindPos(pos->getX(i), pos->getY(i), pos->getZ(i));
                bindPos.applyMatrix4(bindMat);

                // bindMatrix * normal (direction, use upper 3x3 — no translation, no normalize yet)
                Vector3 bindNrm;
                if (nrm) {
                    bindNrm.set(nrm->getX(i), nrm->getY(i), nrm->getZ(i));
                    const auto& e = bindMat.elements;
                    Vector3 t;
                    t.x = e[0] * bindNrm.x + e[4] * bindNrm.y + e[8]  * bindNrm.z;
                    t.y = e[1] * bindNrm.x + e[5] * bindNrm.y + e[9]  * bindNrm.z;
                    t.z = e[2] * bindNrm.x + e[6] * bindNrm.y + e[10] * bindNrm.z;
                    bindNrm = t;
                }

                Vector3 accPos(0.f, 0.f, 0.f);
                Vector3 accNrm(0.f, 0.f, 0.f);
                for (unsigned b = 0; b < 4; ++b) {
                    const float w = sW[b];
                    if (w == 0.f) continue;
                    const int bIdx = static_cast<int>(sIdx[b]);
                    if (bIdx < 0 || bIdx >= boneCount) continue;
                    const Matrix4& m = boneMats[bIdx];

                    // m * bindPos
                    Vector3 bp = bindPos;
                    bp.applyMatrix4(m);
                    accPos.addScaledVector(bp, w);

                    if (nrm) {
                        // m (3x3 only) * bindNrm
                        const auto& e = m.elements;
                        Vector3 bn;
                        bn.x = e[0] * bindNrm.x + e[4] * bindNrm.y + e[8]  * bindNrm.z;
                        bn.y = e[1] * bindNrm.x + e[5] * bindNrm.y + e[9]  * bindNrm.z;
                        bn.z = e[2] * bindNrm.x + e[6] * bindNrm.y + e[10] * bindNrm.z;
                        accNrm.addScaledVector(bn, w);
                    }
                }

                // bindMatrixInverse applied to both (the shader does this via
                // bindMatrixInverse * skinnedPos and bindMatrixInverse * (skinnedNormal,0))
                accPos.applyMatrix4(bindMatInv);
                skinnedPos[i] = accPos;

                if (nrm) {
                    const auto& e = bindMatInv.elements;
                    Vector3 t;
                    t.x = e[0] * accNrm.x + e[4] * accNrm.y + e[8]  * accNrm.z;
                    t.y = e[1] * accNrm.x + e[5] * accNrm.y + e[9]  * accNrm.z;
                    t.z = e[2] * accNrm.x + e[6] * accNrm.y + e[10] * accNrm.z;
                    const float len = std::sqrt(t.x * t.x + t.y * t.y + t.z * t.z);
                    if (len > 0.f) {
                        const float inv = 1.f / len;
                        t.x *= inv; t.y *= inv; t.z *= inv;
                    }
                    skinnedNrm[i] = t;
                }
            }
        }
    }

    auto getPos = [&](int i, float& x, float& y, float& z) {
        if (!skinnedPos.empty()) {
            x = skinnedPos[i].x; y = skinnedPos[i].y; z = skinnedPos[i].z;
        } else {
            x = pos->getX(i); y = pos->getY(i); z = pos->getZ(i);
        }
    };
    auto getNrm = [&](int i, float& x, float& y, float& z) {
        if (!skinnedNrm.empty()) {
            x = skinnedNrm[i].x; y = skinnedNrm[i].y; z = skinnedNrm[i].z;
        } else if (nrm) {
            x = nrm->getX(i); y = nrm->getY(i); z = nrm->getZ(i);
        } else {
            x = 0.f; y = 1.f; z = 0.f;
        }
    };

    const int nTris = idx ? static_cast<int>(idx->count()) / 3
                          : static_cast<int>(pos->count()) / 3;
    const int writeCount = std::min(nTris, maxTris);
    for (int ti = 0; ti < writeCount; ++ti) {
        const int i0 = vi(ti, 0), i1 = vi(ti, 1), i2 = vi(ti, 2);
        float x0, y0, z0, x1, y1, z1, x2, y2, z2;
        getPos(i0, x0, y0, z0);
        getPos(i1, x1, y1, z1);
        getPos(i2, x2, y2, z2);
        float nx0, ny0, nz0, nx1, ny1, nz1, nx2, ny2, nz2;
        getNrm(i0, nx0, ny0, nz0);
        getNrm(i1, nx1, ny1, nz1);
        getNrm(i2, nx2, ny2, nz2);
        const auto [u0, v0uv] = uv(i0);
        const auto [u1, v1uv] = uv(i1);
        const auto [u2, v2uv] = uv(i2);

        float* p = dstObjTri + static_cast<size_t>(ti) * 32;
        // Field 0: v0, matIdx
        p[0]  = x0; p[1]  = y0; p[2]  = z0; p[3]  = static_cast<float>(matIdx);
        // Field 1: v1, meshIdx
        p[4]  = x1; p[5]  = y1; p[6]  = z1; p[7]  = static_cast<float>(meshIdx);
        // Field 2: v2
        p[8]  = x2; p[9]  = y2; p[10] = z2; p[11] = 0.f;
        // Field 3..5: object-space normals
        p[12] = nx0; p[13] = ny0; p[14] = nz0; p[15] = 0.f;
        p[16] = nx1; p[17] = ny1; p[18] = nz1; p[19] = 0.f;
        p[20] = nx2; p[21] = ny2; p[22] = nz2; p[23] = 0.f;
        // Field 6..7: UVs
        p[24] = u0;  p[25] = v0uv; p[26] = u1;  p[27] = v1uv;
        p[28] = u2;  p[29] = v2uv; p[30] = 0.f; p[31] = 0.f;
    }
    return writeCount;
}

}// namespace threepp::wgpu_pt
