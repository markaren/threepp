
#include "DawnLights.hpp"

#include "DawnShaders.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/lights/lights.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

using namespace threepp;
using namespace threepp::dawn;

DawnLights::DawnLights(DawnState& state)
    : state_(state) {

    WGPUBufferDescriptor d{};
    d.label = {.data = "light_buf", .length = 9};
    d.size = LIGHT_UNIFORM_SIZE;
    d.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    lightBuffer_ = wgpuDeviceCreateBuffer(state_.device, &d);
}

void DawnLights::update(Object3D& scene) {
    std::vector<float> data(LIGHT_UNIFORM_SIZE / sizeof(float), 0.0f);
    auto* u32 = reinterpret_cast<uint32_t*>(data.data());

    uint32_t nDir = 0, nPt = 0, nSp = 0, nHm = 0;
    float ambR = 0, ambG = 0, ambB = 0;

    struct DirEntry { Vector3 dir; Color col; bool shadow; };
    struct PtEntry  { Vector3 pos; Color col; float dist; float decay; };
    struct SpEntry  { Vector3 pos; Vector3 dir; Color col; float dist; float decay; float coneCos; float penumbraCos; bool shadow; };
    struct HmEntry  { Vector3 dir; Color sky; Color gnd; };
    std::vector<DirEntry> dirs;
    std::vector<PtEntry>  pts;
    std::vector<SpEntry>  sps;
    std::vector<HmEntry>  hms;

    std::function<void(Object3D&)> collect = [&](Object3D& obj) {
        if (auto al = obj.as<AmbientLight>()) {
            ambR += al->color.r * al->intensity;
            ambG += al->color.g * al->intensity;
            ambB += al->color.b * al->intensity;
        } else if (auto dl = obj.as<DirectionalLight>()) {
            if (dirs.size() < static_cast<size_t>(MAX_DIR_LIGHTS)) {
                Vector3 lightPos, targetPos;
                lightPos.setFromMatrixPosition(*dl->matrixWorld);
                targetPos.setFromMatrixPosition(*dl->target().matrixWorld);
                Vector3 direction = lightPos.clone().sub(targetPos).normalize();
                dirs.push_back({direction, Color(dl->color).multiplyScalar(dl->intensity), dl->castShadow});
            }
        } else if (auto pl = obj.as<PointLight>()) {
            if (pts.size() < static_cast<size_t>(MAX_POINT_LIGHTS)) {
                Vector3 pos;
                pos.setFromMatrixPosition(*pl->matrixWorld);
                pts.push_back({pos, Color(pl->color).multiplyScalar(pl->intensity), pl->distance, pl->decay});
            }
        } else if (auto sl = obj.as<SpotLight>()) {
            if (sps.size() < static_cast<size_t>(MAX_SPOT_LIGHTS)) {
                Vector3 pos, targetPos;
                pos.setFromMatrixPosition(*sl->matrixWorld);
                targetPos.setFromMatrixPosition(*sl->target().matrixWorld);
                Vector3 direction = pos.clone().sub(targetPos).normalize();
                sps.push_back({pos, direction, Color(sl->color).multiplyScalar(sl->intensity),
                               sl->distance, sl->decay,
                               std::cos(sl->angle), std::cos(sl->angle * (1.0f - sl->penumbra)), sl->castShadow});
            }
        } else if (auto hl = obj.as<HemisphereLight>()) {
            if (hms.size() < static_cast<size_t>(MAX_HEMI_LIGHTS)) {
                Vector3 dir;
                dir.setFromMatrixPosition(*hl->matrixWorld).normalize();
                hms.push_back({dir, Color(hl->color).multiplyScalar(hl->intensity),
                               Color(hl->groundColor).multiplyScalar(hl->intensity)});
            }
        }
        for (auto& child : obj.children) collect(*child);
    };
    collect(scene);

    // Sort shadow-casting lights first within each type so that shadow map
    // indices align with light buffer indices (shadow index i == light index i).
    std::stable_partition(dirs.begin(), dirs.end(), [](const DirEntry& e) { return e.shadow; });
    std::stable_partition(sps.begin(), sps.end(), [](const SpEntry& e) { return e.shadow; });

    nDir = static_cast<uint32_t>(dirs.size());
    nPt = static_cast<uint32_t>(pts.size());
    nSp = static_cast<uint32_t>(sps.size());
    nHm = static_cast<uint32_t>(hms.size());
    u32[0] = nDir; u32[1] = nPt; u32[2] = nSp; u32[3] = nHm;
    data[4] = ambR; data[5] = ambG; data[6] = ambB; data[7] = 0;

    size_t off = 8;
    for (uint32_t i = 0; i < nDir; i++) {
        data[off+0] = dirs[i].dir.x; data[off+1] = dirs[i].dir.y; data[off+2] = dirs[i].dir.z; data[off+3] = 0;
        data[off+4] = dirs[i].col.r; data[off+5] = dirs[i].col.g; data[off+6] = dirs[i].col.b; data[off+7] = 0;
        off += 8;
    }

    off = 8 + MAX_DIR_LIGHTS * 8;
    for (uint32_t i = 0; i < nPt; i++) {
        data[off+0] = pts[i].pos.x; data[off+1] = pts[i].pos.y; data[off+2] = pts[i].pos.z; data[off+3] = 0;
        data[off+4] = pts[i].col.r; data[off+5] = pts[i].col.g; data[off+6] = pts[i].col.b; data[off+7] = pts[i].dist;
        data[off+8] = pts[i].decay; data[off+9] = 0; data[off+10] = 0; data[off+11] = 0;
        off += 12;
    }

    off = 8 + MAX_DIR_LIGHTS * 8 + MAX_POINT_LIGHTS * 12;
    for (uint32_t i = 0; i < nSp; i++) {
        data[off+0] = sps[i].pos.x; data[off+1] = sps[i].pos.y; data[off+2] = sps[i].pos.z; data[off+3] = 0;
        data[off+4] = sps[i].dir.x; data[off+5] = sps[i].dir.y; data[off+6] = sps[i].dir.z; data[off+7] = 0;
        data[off+8] = sps[i].col.r; data[off+9] = sps[i].col.g; data[off+10] = sps[i].col.b; data[off+11] = sps[i].dist;
        data[off+12] = sps[i].decay; data[off+13] = sps[i].coneCos; data[off+14] = sps[i].penumbraCos; data[off+15] = 0;
        off += 16;
    }

    off = 8 + MAX_DIR_LIGHTS * 8 + MAX_POINT_LIGHTS * 12 + MAX_SPOT_LIGHTS * 16;
    for (uint32_t i = 0; i < nHm; i++) {
        data[off+0] = hms[i].dir.x; data[off+1] = hms[i].dir.y; data[off+2] = hms[i].dir.z; data[off+3] = 0;
        data[off+4] = hms[i].sky.r; data[off+5] = hms[i].sky.g; data[off+6] = hms[i].sky.b; data[off+7] = 0;
        data[off+8] = hms[i].gnd.r; data[off+9] = hms[i].gnd.g; data[off+10] = hms[i].gnd.b; data[off+11] = 0;
        off += 12;
    }

    wgpuQueueWriteBuffer(state_.queue, lightBuffer_, 0, data.data(), LIGHT_UNIFORM_SIZE);
}

void DawnLights::dispose() {
    if (lightBuffer_) {
        wgpuBufferRelease(lightBuffer_);
        lightBuffer_ = nullptr;
    }
}
