
#include "DawnGeometries.hpp"

#include "threepp/core/BufferGeometry.hpp"

#include <set>

using namespace threepp;
using namespace threepp::dawn;

DawnGeometries::DawnGeometries(DawnState& state)
    : state_(state) {}

std::vector<float> DawnGeometries::buildInterleavedVertexData(BufferGeometry* geometry, uint32_t count) {
    auto posAttr = geometry->getAttribute<float>("position");
    auto& posArr = posAttr->array();
    int posItemSize = static_cast<int>(posAttr->itemSize());

    const float* normalData = nullptr;
    const float* uvData = nullptr;
    const float* colorData = nullptr;
    int normalItemSize = 3, uvItemSize = 2, colorItemSize = 3;

    if (geometry->hasAttribute("normal")) {
        auto nAttr = geometry->getAttribute<float>("normal");
        normalData = nAttr->array().data();
        normalItemSize = static_cast<int>(nAttr->itemSize());
    }
    if (geometry->hasAttribute("uv")) {
        auto uvAttr = geometry->getAttribute<float>("uv");
        uvData = uvAttr->array().data();
        uvItemSize = static_cast<int>(uvAttr->itemSize());
    }
    if (geometry->hasAttribute("color")) {
        auto cAttr = geometry->getAttribute<float>("color");
        colorData = cAttr->array().data();
        colorItemSize = static_cast<int>(cAttr->itemSize());
    }

    // 11 floats per vertex: pos(3) + normal(3) + uv(2) + color(3)
    std::vector<float> interleaved(count * 11);
    for (uint32_t i = 0; i < count; i++) {
        size_t base = i * 11;
        interleaved[base + 0] = posArr[i * posItemSize + 0];
        interleaved[base + 1] = posArr[i * posItemSize + 1];
        interleaved[base + 2] = (posItemSize > 2) ? posArr[i * posItemSize + 2] : 0.f;

        if (normalData) {
            interleaved[base + 3] = normalData[i * normalItemSize + 0];
            interleaved[base + 4] = normalData[i * normalItemSize + 1];
            interleaved[base + 5] = (normalItemSize > 2) ? normalData[i * normalItemSize + 2] : 0.f;
        } else {
            interleaved[base + 3] = 0.f;
            interleaved[base + 4] = 0.f;
            interleaved[base + 5] = 1.f;
        }

        if (uvData) {
            interleaved[base + 6] = uvData[i * uvItemSize + 0];
            interleaved[base + 7] = (uvItemSize > 1) ? uvData[i * uvItemSize + 1] : 0.f;
        } else {
            interleaved[base + 6] = 0.f;
            interleaved[base + 7] = 0.f;
        }

        if (colorData) {
            interleaved[base + 8] = colorData[i * colorItemSize + 0];
            interleaved[base + 9] = (colorItemSize > 1) ? colorData[i * colorItemSize + 1] : 0.f;
            interleaved[base + 10] = (colorItemSize > 2) ? colorData[i * colorItemSize + 2] : 0.f;
        } else {
            interleaved[base + 8] = 1.f;
            interleaved[base + 9] = 1.f;
            interleaved[base + 10] = 1.f;
        }
    }
    return interleaved;
}

void DawnGeometries::storeAttributeVersions(BufferGeometry* geometry, GeometryBuffers& gb) {
    if (geometry->hasAttribute("position"))
        gb.positionVersion = geometry->getAttribute<float>("position")->version;
    if (geometry->hasAttribute("normal"))
        gb.normalVersion = geometry->getAttribute<float>("normal")->version;
    if (geometry->hasAttribute("uv"))
        gb.uvVersion = geometry->getAttribute<float>("uv")->version;
    if (geometry->hasAttribute("color"))
        gb.colorVersion = geometry->getAttribute<float>("color")->version;
    if (geometry->getIndex())
        gb.indexVersion = geometry->getIndex()->version;
}

bool DawnGeometries::geometryNeedsUpdate(BufferGeometry* geometry, const GeometryBuffers& gb) {
    if (geometry->hasAttribute("position") &&
        geometry->getAttribute<float>("position")->version > gb.positionVersion)
        return true;
    if (geometry->hasAttribute("normal") &&
        geometry->getAttribute<float>("normal")->version > gb.normalVersion)
        return true;
    if (geometry->hasAttribute("uv") &&
        geometry->getAttribute<float>("uv")->version > gb.uvVersion)
        return true;
    if (geometry->hasAttribute("color") &&
        geometry->getAttribute<float>("color")->version > gb.colorVersion)
        return true;
    return false;
}

bool DawnGeometries::indexNeedsUpdate(BufferGeometry* geometry, const GeometryBuffers& gb) {
    return geometry->getIndex() && geometry->getIndex()->version > gb.indexVersion;
}

GeometryBuffers& DawnGeometries::getOrCreateGeometryBuffers(BufferGeometry* geometry) {
    auto id = geometry->id;
    auto it = geometryCache_.find(id);

    if (it != geometryCache_.end()) {
        auto& gb = it->second;

        // Check if vertex attributes have been updated
        if (geometry->hasAttribute("position") && geometryNeedsUpdate(geometry, gb)) {
            auto posAttr = geometry->getAttribute<float>("position");
            uint32_t newCount = static_cast<uint32_t>(posAttr->count());

            if (newCount != gb.vertexCount) {
                if (gb.vertexBuffer) wgpuBufferRelease(gb.vertexBuffer);
                auto interleaved = buildInterleavedVertexData(geometry, newCount);
                auto byteSize = interleaved.size() * sizeof(float);
                WGPUBufferDescriptor vbDesc{};
                vbDesc.label = {.data = "vertex_buf", .length = 10};
                vbDesc.size = byteSize;
                vbDesc.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
                gb.vertexBuffer = wgpuDeviceCreateBuffer(state_.device, &vbDesc);
                wgpuQueueWriteBuffer(state_.queue, gb.vertexBuffer, 0, interleaved.data(), byteSize);
                gb.vertexCount = newCount;
            } else {
                auto interleaved = buildInterleavedVertexData(geometry, newCount);
                wgpuQueueWriteBuffer(state_.queue, gb.vertexBuffer, 0,
                                     interleaved.data(), interleaved.size() * sizeof(float));
            }
            storeAttributeVersions(geometry, gb);
        }

        // Check if index buffer has been updated
        if (indexNeedsUpdate(geometry, gb)) {
            auto indexAttr = geometry->getIndex();
            auto& arr = indexAttr->array();
            std::vector<uint32_t> indices(arr.size());
            for (size_t i = 0; i < arr.size(); ++i) {
                indices[i] = static_cast<uint32_t>(arr[i]);
            }
            auto byteSize = indices.size() * sizeof(uint32_t);

            if (static_cast<uint32_t>(indices.size()) != gb.indexCount) {
                if (gb.indexBuffer) wgpuBufferRelease(gb.indexBuffer);
                WGPUBufferDescriptor ibDesc{};
                ibDesc.label = {.data = "index_buf", .length = 9};
                ibDesc.size = byteSize;
                ibDesc.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
                gb.indexBuffer = wgpuDeviceCreateBuffer(state_.device, &ibDesc);
                gb.indexCount = static_cast<uint32_t>(indices.size());
            }
            wgpuQueueWriteBuffer(state_.queue, gb.indexBuffer, 0, indices.data(), byteSize);
            gb.indexVersion = indexAttr->version;
        }

        return gb;
    }

    // First-time creation
    GeometryBuffers gb{};

    if (geometry->hasAttribute("position")) {
        auto posAttr = geometry->getAttribute<float>("position");
        uint32_t count = static_cast<uint32_t>(posAttr->count());
        gb.vertexCount = count;

        auto interleaved = buildInterleavedVertexData(geometry, count);
        auto byteSize = interleaved.size() * sizeof(float);
        WGPUBufferDescriptor vbDesc{};
        vbDesc.label = {.data = "vertex_buf", .length = 10};
        vbDesc.size = byteSize;
        vbDesc.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
        gb.vertexBuffer = wgpuDeviceCreateBuffer(state_.device, &vbDesc);
        wgpuQueueWriteBuffer(state_.queue, gb.vertexBuffer, 0, interleaved.data(), byteSize);
    }

    if (geometry->getIndex()) {
        auto indexAttr = geometry->getIndex();
        auto& arr = indexAttr->array();
        std::vector<uint32_t> indices(arr.size());
        for (size_t i = 0; i < arr.size(); ++i) {
            indices[i] = static_cast<uint32_t>(arr[i]);
        }
        auto byteSize = indices.size() * sizeof(uint32_t);
        WGPUBufferDescriptor ibDesc{};
        ibDesc.label = {.data = "index_buf", .length = 9};
        ibDesc.size = byteSize;
        ibDesc.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
        gb.indexBuffer = wgpuDeviceCreateBuffer(state_.device, &ibDesc);
        wgpuQueueWriteBuffer(state_.queue, gb.indexBuffer, 0, indices.data(), byteSize);
        gb.indexCount = static_cast<uint32_t>(indices.size());
    }

    storeAttributeVersions(geometry, gb);
    geometryCache_[id] = gb;
    return geometryCache_[id];
}

WireframeBuffers& DawnGeometries::getOrCreateWireframeBuffers(BufferGeometry* geometry) {
    auto id = geometry->id;
    auto it = wireframeCache_.find(id);
    if (it != wireframeCache_.end()) {
        return it->second;
    }

    WireframeBuffers wb{};

    if (geometry->getIndex()) {
        auto& arr = geometry->getIndex()->array();
        std::set<uint64_t> edgeSet;
        std::vector<uint32_t> edges;

        for (size_t i = 0; i + 2 < arr.size(); i += 3) {
            uint32_t a = static_cast<uint32_t>(arr[i]);
            uint32_t b = static_cast<uint32_t>(arr[i + 1]);
            uint32_t c = static_cast<uint32_t>(arr[i + 2]);

            auto addEdge = [&](uint32_t p, uint32_t q) {
                uint64_t key = (static_cast<uint64_t>(std::min(p, q)) << 32) | std::max(p, q);
                if (edgeSet.insert(key).second) {
                    edges.push_back(p);
                    edges.push_back(q);
                }
            };
            addEdge(a, b);
            addEdge(b, c);
            addEdge(c, a);
        }

        if (!edges.empty()) {
            auto byteSize = edges.size() * sizeof(uint32_t);
            WGPUBufferDescriptor bd{};
            bd.label = {.data = "wire_idx", .length = 8};
            bd.size = byteSize;
            bd.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
            wb.indexBuffer = wgpuDeviceCreateBuffer(state_.device, &bd);
            wgpuQueueWriteBuffer(state_.queue, wb.indexBuffer, 0, edges.data(), byteSize);
            wb.indexCount = static_cast<uint32_t>(edges.size());
        }
    }

    wireframeCache_[id] = wb;
    return wireframeCache_[id];
}

void DawnGeometries::dispose() {
    for (auto& [id, gb] : geometryCache_) {
        if (gb.vertexBuffer) wgpuBufferRelease(gb.vertexBuffer);
        if (gb.indexBuffer) wgpuBufferRelease(gb.indexBuffer);
    }
    geometryCache_.clear();

    for (auto& [id, wb] : wireframeCache_) {
        if (wb.indexBuffer) wgpuBufferRelease(wb.indexBuffer);
    }
    wireframeCache_.clear();
}
