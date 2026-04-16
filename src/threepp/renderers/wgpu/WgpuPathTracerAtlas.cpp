#include "WgpuPathTracerAtlas.hpp"
#include "WgpuPathTracerBCn.hpp"
#include "WgpuPathTracerTypes.hpp"

#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/textures/Texture.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <unordered_set>

namespace threepp::wgpu_pt {

std::vector<RtMeshEntry> expandMeshEntries(const std::vector<Mesh*>& meshes) {
    std::vector<RtMeshEntry> entries;
    for (auto* mesh : meshes) {
        auto* inst = dynamic_cast<InstancedMesh*>(mesh);
        if (inst && inst->count() > 0) {
            for (size_t j = 0; j < inst->count(); ++j) {
                Matrix4 instMat;
                inst->getMatrixAt(j, instMat);
                Matrix4 world;
                world.multiplyMatrices(*mesh->matrixWorld, instMat);
                entries.push_back({mesh, world});
            }
        } else {
            entries.push_back({mesh, *mesh->matrixWorld});
        }
    }
    return entries;
}

std::tuple<std::vector<unsigned char>, int, int, int> buildAtlas(
        const std::vector<Mesh*>& meshes,
        std::unordered_map<Texture*, int>& texSlotMap,
        int TILE_SIZE) {
    const int ATLAS_COLS = ATLAS_WIDTH / TILE_SIZE;
    // First pass: count unique textures to determine atlas size.
    int slotCount = 0;
    std::unordered_set<Texture*> seen;
    for (auto* mesh : meshes) {
        if (slotCount >= MAX_TEX_SLOTS) break;
        auto countTex = [&](Texture* tex) {
            if (tex && slotCount < MAX_TEX_SLOTS && !seen.count(tex)) {
                auto& img = tex->image();
                if (img.width > 0 && img.height > 0) {
                    seen.insert(tex);
                    ++slotCount;
                }
            }
        };
        if (auto* mwm = dynamic_cast<MaterialWithMap*>(mesh->material().get()))
            countTex(mwm->map.get());
        if (auto* mnm = dynamic_cast<MaterialWithNormalMap*>(mesh->material().get()))
            countTex(mnm->normalMap.get());
        if (auto* mwr = dynamic_cast<MaterialWithRoughness*>(mesh->material().get()))
            countTex(mwr->roughnessMap.get());
        if (auto* mwe = dynamic_cast<MaterialWithEmissive*>(mesh->material().get()))
            countTex(mwe->emissiveMap.get());
        if (auto* mwa = dynamic_cast<MaterialWithAoMap*>(mesh->material().get()))
            countTex(mwa->aoMap.get());
    }

    // Layout: 8×8 grid of 1024px tiles per layer (8192×8192), multiple layers for >64 textures
    const int slotsPerLayer = ATLAS_COLS * ATLAS_COLS; // 64
    const int numLayers = std::max(1, (slotCount + slotsPerLayer - 1) / slotsPerLayer);
    const int atlasW = ATLAS_COLS * TILE_SIZE;
    const int atlasH = ATLAS_COLS * TILE_SIZE;
    const size_t layerBytes = static_cast<size_t>(atlasW) * atlasH * 4;
    std::vector<unsigned char> atlas(layerBytes * numLayers, 255);

    auto addTexture = [&](Texture* tex, int& slot) {
        if (!tex || slot >= MAX_TEX_SLOTS) return;
        if (texSlotMap.count(tex)) return;
        auto& img = tex->image();
        if (img.width == 0 || img.height == 0) return;
        const int srcW = static_cast<int>(img.width);
        const int srcH = static_cast<int>(img.height);

        // Decompress BCn/DXT blocks to plain RGBA8 for the atlas blit.
        std::vector<uint8_t> decompressed;
        if (img.compressedFormat.has_value()) {
            const auto& compressed = img.data<unsigned char>();
            decompressed = bcnDecompress(compressed.data(), srcW, srcH, *img.compressedFormat);
            if (decompressed.empty()) {
                std::cerr << "[Atlas] Unsupported compressed format 0x"
                          << std::hex << *img.compressedFormat << std::dec
                          << " — texture skipped\n";
                return;
            }
        }

        const uint8_t* srcPtr = img.compressedFormat.has_value()
            ? decompressed.data()
            : img.data<unsigned char>().data();
        const int ch = img.compressedFormat.has_value()
            ? 4
            : static_cast<int>(img.data<unsigned char>().size()) / (srcW * srcH);

        const int layer = slot / slotsPerLayer;
        const int localSlot = slot % slotsPerLayer;
        const int col = localSlot % ATLAS_COLS;
        const int row = localSlot / ATLAS_COLS;
        const int destX = col * TILE_SIZE;
        const int destY = row * TILE_SIZE;
        unsigned char* layerBase = atlas.data() + layer * layerBytes;

        if (srcW == TILE_SIZE && srcH == TILE_SIZE && ch == 4) {
            for (int ty = 0; ty < TILE_SIZE; ++ty) {
                const int di = ((destY + ty) * atlasW + destX) * 4;
                const int si = ty * srcW * 4;
                std::memcpy(layerBase + di, srcPtr + si, TILE_SIZE * 4);
            }
        } else {
            std::vector<int> xmap(TILE_SIZE);
            for (int tx = 0; tx < TILE_SIZE; ++tx)
                xmap[tx] = tx * srcW / TILE_SIZE;
            for (int ty = 0; ty < TILE_SIZE; ++ty) {
                const int sy = ty * srcH / TILE_SIZE;
                const int srcRowOff = sy * srcW;
                unsigned char* dst = layerBase + ((destY + ty) * atlasW + destX) * 4;
                if (ch == 4) {
                    for (int tx = 0; tx < TILE_SIZE; ++tx) {
                        std::memcpy(dst + tx * 4, srcPtr + (srcRowOff + xmap[tx]) * 4, 4);
                    }
                } else {
                    for (int tx = 0; tx < TILE_SIZE; ++tx) {
                        const int si = (srcRowOff + xmap[tx]) * ch;
                        const int di = tx * 4;
                        dst[di + 0] = srcPtr[si + 0];
                        dst[di + 1] = srcPtr[si + 1];
                        dst[di + 2] = srcPtr[si + 2];
                        dst[di + 3] = ch >= 4 ? srcPtr[si + 3] : 255u;
                    }
                }
            }
        }
        texSlotMap[tex] = slot++;
    };

    int slot = 0;
    for (auto& mesh : meshes) {
        if (slot >= MAX_TEX_SLOTS) break;
        auto* mwm = dynamic_cast<MaterialWithMap*>(mesh->material().get());
        if (mwm && mwm->map) addTexture(mwm->map.get(), slot);
        auto* mnm = dynamic_cast<MaterialWithNormalMap*>(mesh->material().get());
        if (mnm && mnm->normalMap) addTexture(mnm->normalMap.get(), slot);
        auto* mwr = dynamic_cast<MaterialWithRoughness*>(mesh->material().get());
        if (mwr && mwr->roughnessMap) addTexture(mwr->roughnessMap.get(), slot);
        auto* mwe = dynamic_cast<MaterialWithEmissive*>(mesh->material().get());
        if (mwe && mwe->emissiveMap) addTexture(mwe->emissiveMap.get(), slot);
        auto* mwa = dynamic_cast<MaterialWithAoMap*>(mesh->material().get());
        if (mwa && mwa->aoMap) addTexture(mwa->aoMap.get(), slot);
    }
    std::cerr << "[PathTracer] Atlas: " << slot << " unique textures in "
              << numLayers << " layer(s), " << ATLAS_COLS << "x" << ATLAS_COLS
              << " grid (" << atlasW << "x" << atlasH << " px/layer)" << std::endl;
    return {std::move(atlas), numLayers, ATLAS_COLS, TILE_SIZE};
}

float encodeSlotWrap(int slot, const Texture* tex) {
    int ws = 0, wt = 0;
    if (tex) {
        if (tex->wrapS == TextureWrapping::ClampToEdge) ws = 1;
        else if (tex->wrapS == TextureWrapping::MirroredRepeat) ws = 2;
        if (tex->wrapT == TextureWrapping::ClampToEdge) wt = 1;
        else if (tex->wrapT == TextureWrapping::MirroredRepeat) wt = 2;
    }
    return static_cast<float>(slot * 16 + ws * 4 + wt);
}

ExtractedMaterial extractMaterial(const Material* mat) {
    ExtractedMaterial m;
    if (!mat) return m;
    if (auto* c = dynamic_cast<const MaterialWithColor*>(mat))
        m.albedo = c->color;
    // MeshBasicMaterial → unlit: shininess = -1 signals no lighting
    if (dynamic_cast<const MeshBasicMaterial*>(mat)) {
        m.shininess = -1.f;
        return m;
    }
    if (auto* r = dynamic_cast<const MaterialWithRoughness*>(mat)) {
        const float rough = std::max(0.f, std::min(1.f, r->roughness));
        m.shininess = std::max(1e-4f, rough * rough);
    } else if (auto* s = dynamic_cast<const MaterialWithSpecular*>(mat)) {
        const float n = std::max(1.f, s->shininess);
        m.shininess = std::max(0.04f, std::sqrt(2.f / (n + 2.f)));
    }
    if (auto* mm = dynamic_cast<const MaterialWithMetalness*>(mat))
        m.metalness = std::max(0.f, std::min(1.f, mm->metalness));
    if (auto* e = dynamic_cast<const MaterialWithEmissive*>(mat))
        m.emissive = Color(e->emissive.r * e->emissiveIntensity,
                         e->emissive.g * e->emissiveIntensity,
                         e->emissive.b * e->emissiveIntensity);
    if (auto* t = dynamic_cast<const MaterialWithTransmission*>(mat)) {
        m.transmission = std::clamp(t->transmission, 0.f, 1.f);
        m.ior = std::max(1.f, t->ior);
        m.dispersion = std::max(0.f, t->dispersion);
    }
    if (auto* a = dynamic_cast<const MaterialWithAttenuation*>(mat)) {
        m.attenuationColor = a->attenuationColor;
        m.attenuationDistance = std::max(0.f, a->attenuationDistance);
    }
    if (auto* th = dynamic_cast<const MaterialWithThickness*>(mat)) {
        m.thickness = std::max(0.f, th->thickness);
    }
    if (auto* cc = dynamic_cast<const MaterialWithClearcoat*>(mat)) {
        m.clearcoat = std::clamp(cc->clearcoat, 0.f, 1.f);
        const float ccr = std::clamp(cc->clearcoatRoughness, 0.f, 1.f);
        m.clearcoatRoughness = std::max(1e-4f, ccr * ccr);
    }
    if (auto* sh = dynamic_cast<const MaterialWithSheen*>(mat)) {
        m.sheenColor = sh->sheenColor;
        m.sheenRoughness = std::clamp(sh->sheenRoughness, 0.f, 1.f);
    }
    if (auto* sp = dynamic_cast<const MaterialWithPbrSpecular*>(mat)) {
        m.specularIntensity = std::max(0.f, sp->specularIntensity);
        m.specularColor = sp->specularColor;
    }
    m.alphaTest = mat->alphaTest;
    return m;
}

}// namespace threepp::wgpu_pt
