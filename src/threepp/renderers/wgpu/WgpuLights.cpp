
#include "WgpuLights.hpp"

#include "WgpuShaders.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/lights/lights.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

using namespace threepp;
using namespace threepp::wgpu;

WgpuLights::WgpuLights(WgpuState& state)
    : state_(state) {

    WGPUBufferDescriptor d{};
    d.label = WGPUStringView{"light_buf", WGPU_STRLEN} ;
    d.size = state_.lightLimits.lightUniformSize();
    d.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    lightBuffer_ = wgpuDeviceCreateBuffer(state_.device, &d);
}

void WgpuLights::update(Object3D& scene) {
    auto& lim = state_.lightLimits;
    size_t bufSize = lim.lightUniformSize();

    // Reuse scratch buffer to avoid per-frame heap allocation.
    scratch_.assign(bufSize / sizeof(float), 0.0f);
    std::vector<float>& data = scratch_;
    auto* u32 = reinterpret_cast<uint32_t*>(data.data());

    uint32_t nDir = 0, nPt = 0, nSp = 0, nHm = 0, nRa = 0;
    float ambR = 0, ambG = 0, ambB = 0;

    struct DirEntry { Vector3 dir; Color col; bool shadow; };
    struct PtEntry  { Vector3 pos; Color col; float dist; float decay; bool shadow; };
    struct SpEntry  { Vector3 pos; Vector3 dir; Color col; float dist; float decay; float coneCos; float penumbraCos; bool shadow; };
    struct HmEntry  { Vector3 dir; Color sky; Color gnd; };
    struct RaEntry  { Vector3 pos; Color col; Vector3 halfWidth; Vector3 halfHeight; };
    std::vector<DirEntry> dirs;
    std::vector<PtEntry>  pts;
    std::vector<SpEntry>  sps;
    std::vector<HmEntry>  hms;
    std::vector<RaEntry>  ras;

    std::function<void(Object3D&)> collect = [&](Object3D& obj) {
        if (auto al = obj.as<AmbientLight>()) {
            ambR += al->color.r * al->intensity;
            ambG += al->color.g * al->intensity;
            ambB += al->color.b * al->intensity;
        } else if (auto dl = obj.as<DirectionalLight>()) {
            if (dirs.size() < static_cast<size_t>(lim.maxDirLights)) {
                Vector3 lightPos, targetPos;
                lightPos.setFromMatrixPosition(*dl->matrixWorld);
                targetPos.setFromMatrixPosition(*dl->target().matrixWorld);
                Vector3 direction = lightPos.clone().sub(targetPos).normalize();
                dirs.push_back({direction, Color(dl->color).multiplyScalar(dl->intensity), dl->castShadow});
            }
        } else if (auto pl = obj.as<PointLight>()) {
            if (pts.size() < static_cast<size_t>(lim.maxPointLights)) {
                Vector3 pos;
                pos.setFromMatrixPosition(*pl->matrixWorld);
                pts.push_back({pos, Color(pl->color).multiplyScalar(pl->intensity), pl->distance, pl->decay, pl->castShadow});
            }
        } else if (auto sl = obj.as<SpotLight>()) {
            if (sps.size() < static_cast<size_t>(lim.maxSpotLights)) {
                Vector3 pos, targetPos;
                pos.setFromMatrixPosition(*sl->matrixWorld);
                targetPos.setFromMatrixPosition(*sl->target().matrixWorld);
                Vector3 direction = pos.clone().sub(targetPos).normalize();
                float coneCos = std::cos(sl->angle);
                // penumbraCos must be strictly > coneCos for smoothstep to be well-defined in WGSL.
                // When penumbra == 0 they'd be equal, causing undefined behavior (Dawn returns 0).
                float penumbraCos = std::cos(sl->angle * (1.0f - sl->penumbra));
                if (penumbraCos <= coneCos) penumbraCos = coneCos + 1e-4f;
                sps.push_back({pos, direction, Color(sl->color).multiplyScalar(sl->intensity),
                               sl->distance, sl->decay,
                               coneCos, penumbraCos, sl->castShadow});
            }
        } else if (auto hl = obj.as<HemisphereLight>()) {
            if (hms.size() < static_cast<size_t>(lim.maxHemiLights)) {
                Vector3 dir;
                dir.setFromMatrixPosition(*hl->matrixWorld).normalize();
                hms.push_back({dir, Color(hl->color).multiplyScalar(hl->intensity),
                               Color(hl->groundColor).multiplyScalar(hl->intensity)});
            }
        } else if (auto ral = obj.as<RectAreaLight>()) {
            if (ras.size() < static_cast<size_t>(lim.maxRectAreaLights)) {
                // Extract world-space position and rotate local half-axes
                // (±X for width, ±Y for height; emissive face -Z) into world.
                Vector3 pos;
                pos.setFromMatrixPosition(*ral->matrixWorld);
                Matrix4 rotation;
                rotation.extractRotation(*ral->matrixWorld);
                Vector3 halfWidth(ral->width * 0.5f, 0.f, 0.f);
                Vector3 halfHeight(0.f, ral->height * 0.5f, 0.f);
                halfWidth.applyMatrix4(rotation);
                halfHeight.applyMatrix4(rotation);
                ras.push_back({pos, Color(ral->color).multiplyScalar(ral->intensity),
                               halfWidth, halfHeight});
            }
        }
        for (auto& child : obj.children) collect(*child);
    };
    collect(scene);

    // Sort shadow-casting lights first within each type so that shadow map
    // indices align with light buffer indices (shadow index i == light index i).
    std::stable_partition(dirs.begin(), dirs.end(), [](const DirEntry& e) { return e.shadow; });
    std::stable_partition(pts.begin(), pts.end(), [](const PtEntry& e) { return e.shadow; });
    std::stable_partition(sps.begin(), sps.end(), [](const SpEntry& e) { return e.shadow; });

    nDir = static_cast<uint32_t>(dirs.size());
    nPt = static_cast<uint32_t>(pts.size());
    nSp = static_cast<uint32_t>(sps.size());
    nHm = static_cast<uint32_t>(hms.size());
    nRa = static_cast<uint32_t>(ras.size());
    hasRectAreaLights_ = nRa > 0;
    u32[0] = nDir; u32[1] = nPt; u32[2] = nSp; u32[3] = nHm;
    data[4] = ambR; data[5] = ambG; data[6] = ambB;
    u32[7] = nRa;

    // Floats per light type in the buffer layout:
    // DirLight: 8 floats, PointLight: 12 floats, SpotLight: 16 floats, HemiLight: 12 floats
    size_t off = 8;
    for (uint32_t i = 0; i < nDir; i++) {
        data[off+0] = dirs[i].dir.x; data[off+1] = dirs[i].dir.y; data[off+2] = dirs[i].dir.z; data[off+3] = 0;
        data[off+4] = dirs[i].col.r; data[off+5] = dirs[i].col.g; data[off+6] = dirs[i].col.b; data[off+7] = 0;
        off += 8;
    }

    off = 8 + lim.maxDirLights * 8;
    for (uint32_t i = 0; i < nPt; i++) {
        data[off+0] = pts[i].pos.x; data[off+1] = pts[i].pos.y; data[off+2] = pts[i].pos.z; data[off+3] = 0;
        data[off+4] = pts[i].col.r; data[off+5] = pts[i].col.g; data[off+6] = pts[i].col.b; data[off+7] = pts[i].dist;
        data[off+8] = pts[i].decay; data[off+9] = 0; data[off+10] = 0; data[off+11] = 0;
        off += 12;
    }

    off = 8 + lim.maxDirLights * 8 + lim.maxPointLights * 12;
    for (uint32_t i = 0; i < nSp; i++) {
        data[off+0] = sps[i].pos.x; data[off+1] = sps[i].pos.y; data[off+2] = sps[i].pos.z; data[off+3] = 0;
        data[off+4] = sps[i].dir.x; data[off+5] = sps[i].dir.y; data[off+6] = sps[i].dir.z; data[off+7] = 0;
        data[off+8] = sps[i].col.r; data[off+9] = sps[i].col.g; data[off+10] = sps[i].col.b; data[off+11] = sps[i].dist;
        data[off+12] = sps[i].decay; data[off+13] = sps[i].coneCos; data[off+14] = sps[i].penumbraCos; data[off+15] = 0;
        off += 16;
    }

    off = 8 + lim.maxDirLights * 8 + lim.maxPointLights * 12 + lim.maxSpotLights * 16;
    for (uint32_t i = 0; i < nHm; i++) {
        data[off+0] = hms[i].dir.x; data[off+1] = hms[i].dir.y; data[off+2] = hms[i].dir.z; data[off+3] = 0;
        data[off+4] = hms[i].sky.r; data[off+5] = hms[i].sky.g; data[off+6] = hms[i].sky.b; data[off+7] = 0;
        data[off+8] = hms[i].gnd.r; data[off+9] = hms[i].gnd.g; data[off+10] = hms[i].gnd.b; data[off+11] = 0;
        off += 12;
    }

    off = 8 + lim.maxDirLights * 8 + lim.maxPointLights * 12 + lim.maxSpotLights * 16 + lim.maxHemiLights * 12;
    for (uint32_t i = 0; i < nRa; i++) {
        data[off+0] = ras[i].pos.x; data[off+1] = ras[i].pos.y; data[off+2] = ras[i].pos.z; data[off+3] = 0;
        data[off+4] = ras[i].col.r; data[off+5] = ras[i].col.g; data[off+6] = ras[i].col.b; data[off+7] = 0;
        data[off+8] = ras[i].halfWidth.x; data[off+9] = ras[i].halfWidth.y; data[off+10] = ras[i].halfWidth.z; data[off+11] = 0;
        data[off+12] = ras[i].halfHeight.x; data[off+13] = ras[i].halfHeight.y; data[off+14] = ras[i].halfHeight.z; data[off+15] = 0;
        off += 16;
    }

    // Only upload if the data actually changed (lights are usually static).
    if (data != uploaded_) {
        wgpuQueueWriteBuffer(state_.queue, lightBuffer_, 0, data.data(), bufSize);
        uploaded_ = data;
    }
}

size_t WgpuLights::lightUniformSize() const {
    return state_.lightLimits.lightUniformSize();
}

void WgpuLights::recreateBuffer() {
    if (lightBuffer_) {
        wgpuBufferRelease(lightBuffer_);
        lightBuffer_ = nullptr;
    }
    uploaded_.clear(); // force re-upload after buffer recreation
    WGPUBufferDescriptor d{};
    d.label = WGPUStringView{"light_buf", WGPU_STRLEN} ;
    d.size = state_.lightLimits.lightUniformSize();
    d.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    lightBuffer_ = wgpuDeviceCreateBuffer(state_.device, &d);
}

void WgpuLights::dispose() {
    if (lightBuffer_) {
        wgpuBufferRelease(lightBuffer_);
        lightBuffer_ = nullptr;
    }
}
