
#include "threepp/objects/ParticleField.hpp"

#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

using namespace threepp;

namespace {

    // The placeholder a ParticleField carries as its Mesh geometry: three
    // COINCIDENT vertices, i.e. exactly zero area.
    //
    // It is not a representation and it is not a proxy — it exists so that a
    // ParticleField is a structurally valid Mesh everywhere a Mesh is assumed.
    // GLRenderer dereferences object->geometry() unconditionally (projectObject
    // → Frustum::intersectsObject → geometry->computeBoundingSphere), so a null
    // geometry is a crash on the GL backend, which this type must survive
    // without rendering anything. Zero area is what makes "survive" and "render
    // nothing" the same statement: a zero-area triangle covers no sample on any
    // rasteriser, and a zero-area primitive in a BLAS is measure-zero to every
    // ray. The Vulkan backend consequently gets its one harmless MeshEntry /
    // GeometryDesc / MaterialDesc / TLAS-instance slot for free, with no
    // special-casing beyond the isParticleField flag.
    //
    // Per field rather than shared: a shared function-local static would tie a
    // BufferGeometry's lifetime to static destruction order, which runs after
    // the renderer has torn the device down.
    std::shared_ptr<BufferGeometry> makePlaceholderGeometry() {

        auto geometry = BufferGeometry::create();
        const std::vector<float> zeros(9, 0.f);
        geometry->setAttribute("position", FloatBufferAttribute::create(zeros, 3));
        geometry->setAttribute("normal", FloatBufferAttribute::create(zeros, 3));
        return geometry;
    }

}// namespace

ParticleField::ParticleField(const Config& config)
    : Mesh(makePlaceholderGeometry(), MeshBasicMaterial::create()),
      config_(config) {

    host_.resize(config_.capacity);
}

std::shared_ptr<ParticleField> ParticleField::create(const Config& config) {

    if (config.capacity == 0) {
        throw std::invalid_argument(
                "ParticleField::create: capacity must be > 0 — a field is created once at "
                "its final capacity and is never resized (see the churn contract in "
                "ParticleField.hpp)");
    }
    if (config.ownership != Ownership::HostRing) {
        throw std::invalid_argument(
                "ParticleField::create: only Ownership::HostRing is implemented in this "
                "phase; Ownership::Interop (CUDA zero-copy) and Ownership::Renderer "
                "(compute-written) are declared but not yet wired");
    }
    return std::make_shared<ParticleField>(config);
}

void ParticleField::submit(const void* pxVec4Array, std::uint32_t n) {

    if (n > config_.capacity) n = config_.capacity;
    if (n > 0) {
        if (!pxVec4Array) {
            throw std::invalid_argument("ParticleField::submit: null source with n > 0");
        }
        std::memcpy(host_.data(), pxVec4Array, std::size_t(n) * sizeof(ParticlePos));
    }
    liveCount_ = n;
    ++dataSerial_;
}

void ParticleField::setMeshRepr(std::shared_ptr<BufferGeometry> proxy,
                                std::shared_ptr<Material> material) {

    if (!proxy || !material) {
        throw std::invalid_argument(
                "ParticleField::setMeshRepr: the mesh representation needs both a proxy "
                "geometry and a material");
    }
    meshRepr_.geometry = std::move(proxy);
    meshRepr_.material = material;
    meshRepr_.enabled  = true;
    // The G-buffer shades a particle through the field ENTRY's MaterialDesc,
    // which the backend derives from Object3D::material() exactly as it does
    // for any other mesh. The Mesh geometry stays the zero-area placeholder —
    // only the material has to follow the representation.
    setMaterial(material);
}

void ParticleField::setOrientations(const float* quatXyzw, std::uint32_t n) {

    if (!config_.orientations) {
        throw std::invalid_argument(
                "ParticleField::setOrientations: Config::orientations was not set, so no "
                "orientation buffer exists (it is fixed at create, like capacity)");
    }
    if (n > config_.capacity) n = config_.capacity;
    if (n > 0 && !quatXyzw) {
        throw std::invalid_argument("ParticleField::setOrientations: null source with n > 0");
    }
    ori_.assign(std::size_t(config_.capacity) * 4u, 0);
    for (std::uint32_t i = 0; i < n; ++i) {
        for (int c = 0; c < 4; ++c) {
            const float v = std::max(-1.f, std::min(1.f, quatXyzw[std::size_t(i) * 4u + c]));
            // Round-to-nearest over the 32767 half-range: the same convention
            // unpackSnorm2x16 inverts, so a decoded quaternion is within
            // 1/32767 of the authored one per component. That quantisation is
            // the documented reason a ParticleField capture is not expected to
            // be bit-identical to the InstancedMesh it replaces.
            ori_[std::size_t(i) * 4u + c] =
                    static_cast<std::int16_t>(std::lround(v * 32767.f));
        }
    }
    // Slots past n keep the all-zero quaternion, which decodes to a zero matrix
    // — the same "no pixels" collapse a dead slot gets, and unreachable anyway
    // because instanceCount never exceeds the live count.
    ++oriSerial_;
}

void ParticleField::setLiveCount(std::uint32_t n) {

    if (n > config_.capacity) n = config_.capacity;
    if (n == liveCount_) return;
    liveCount_ = n;
    ++dataSerial_;
}
