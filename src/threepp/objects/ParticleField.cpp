
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

    if (config_.ownership == Ownership::Interop) {
        // No host staging either, and for the same reason as Renderer below:
        // the positions are written on the device by a foreign API and the host
        // never sees them. The one path that needs a staging block is the
        // no-external-memory fallback, and setHostFallback() allocates it there
        // — so a working Interop field costs the host nothing at all.
        //
        // liveCount stays 0 rather than becoming capacity: unlike the stateless
        // emitter, a sim HAS a real live count, only the application knows it,
        // and it publishes it with setLiveCount() every frame.
    } else if (config_.ownership == Ownership::Renderer) {
        // No host staging at all: the positions are written by the device and
        // the host never sees them. Allocating capacity * 16 B of unreachable
        // staging for a 1M-particle weather field would be 16 MB of memory
        // whose only purpose is to be ignored.
        //
        // liveCount = capacity, ONCE. A stateless emitter has no compaction:
        // every slot is dispatched every frame and the ones outside their
        // lifetime window write w < 0, which is the predicate every consumer
        // already tests. See setLiveCount in the header.
        liveCount_ = config_.capacity;
    } else {
        host_.resize(config_.capacity);
    }
}

std::shared_ptr<ParticleField> ParticleField::create(const Config& config) {

    if (config.capacity == 0) {
        throw std::invalid_argument(
                "ParticleField::create: capacity must be > 0 — a field is created once at "
                "its final capacity and is never resized (see the churn contract in "
                "ParticleField.hpp)");
    }
    return std::make_shared<ParticleField>(config);
}

void ParticleField::submit(const void* pxVec4Array, std::uint32_t n, float dtSec) {

    // The mode split, enforced rather than documented. A Renderer field's
    // positions are device-local — there is no host buffer to memcpy into and
    // no mapping to write through — so accepting this call would leave the
    // caller believing it had supplied positions that the emitter overwrites
    // every frame anyway.
    if (config_.ownership == Ownership::Renderer) {
        throw std::invalid_argument(
                "ParticleField::submit: this field is Ownership::Renderer — its positions "
                "are written on the device by the emitter. Use setEmitter()/"
                "setEmitterTime(), or create the field with Ownership::HostRing");
    }
    // Same rule, one exception. An Interop field's positions are written device
    // to device by the foreign API that imported the renderer's exported
    // buffer, so host bytes handed in here would be written to a block nothing
    // reads — except on a device that cannot export at all, where the renderer
    // has said so and switched this field to the host ring (see hostFallback).
    if (config_.ownership == Ownership::Interop && !hostFallback_) {
        throw std::invalid_argument(
                "ParticleField::submit: this field is Ownership::Interop — its positions "
                "are written device-to-device by the foreign API that imported the "
                "renderer's exported buffer (VulkanRenderer::enableParticleFieldInterop). "
                "Publish the count with setLiveCount(); submit() is legal here only after "
                "the renderer reports hostFallback()");
    }
    if (n > config_.capacity) n = config_.capacity;
    if (n > 0) {
        if (!pxVec4Array) {
            throw std::invalid_argument("ParticleField::submit: null source with n > 0");
        }
        std::memcpy(host_.data(), pxVec4Array, std::size_t(n) * sizeof(ParticlePos));
    }
    liveCount_ = n;
    // Only a POSITIVE step is a step. 0 keeps the previous answer rather than
    // dividing the stretch by zero, and a caller that never passes one keeps
    // the 1/60 default it has always had.
    if (dtSec > 0.f) hostDt_ = dtSec;
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

void ParticleField::setDensityRepr(const Vector3& center, const Vector3& halfExtent,
                                   float sigmaPerParticle, std::uint32_t resolution) {

    if (halfExtent.x <= 0.f || halfExtent.y <= 0.f || halfExtent.z <= 0.f) {
        throw std::invalid_argument(
                "ParticleField::setDensityRepr: halfExtent must be positive on every axis "
                "— it is the world box the density volume covers");
    }
    if (!(sigmaPerParticle > 0.f)) {
        throw std::invalid_argument(
                "ParticleField::setDensityRepr: sigmaPerParticle must be > 0 (it is the "
                "extinction, in 1/m, one particle contributes to its voxel)");
    }
    densityRepr_.center           = center;
    densityRepr_.halfExtent       = halfExtent;
    densityRepr_.sigmaPerParticle = sigmaPerParticle;
    // Clamped rather than rejected: the bound is the image the backend
    // allocates (256^3 of r32ui is 64 MB), not a semantic limit.
    densityRepr_.resolution = std::max(8u, std::min(256u, resolution));
    densityRepr_.enabled    = true;
}

void ParticleField::setBillboardRepr(const Color& colorHot, const Color& colorCool,
                                     float intensity, float sizeScale) {

    if (!(intensity >= 0.f)) {
        throw std::invalid_argument(
                "ParticleField::setBillboardRepr: intensity must be >= 0 (it is the HDR "
                "radiance scale on both colours)");
    }
    if (!(sizeScale > 0.f)) {
        throw std::invalid_argument(
                "ParticleField::setBillboardRepr: sizeScale must be > 0 (it multiplies the "
                "particle's own world radius)");
    }
    billboardRepr_.colorHot  = colorHot;
    billboardRepr_.colorCool = colorCool;
    billboardRepr_.intensity = intensity;
    billboardRepr_.sizeScale = sizeScale;
    billboardRepr_.enabled   = true;
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

void ParticleField::setAttributes(const float* rgba, std::uint32_t n) {

    if (!config_.attributes) {
        throw std::invalid_argument(
                "ParticleField::setAttributes: Config::attributes was not set, so no "
                "attribute buffer exists (it is fixed at create, like capacity)");
    }
    // The mode split, exactly as submit() enforces it for positions: an interop
    // field's attributes are written device-to-device by the foreign API that
    // imported the renderer's exported buffer, so host bytes here would land in
    // a block nothing reads.
    if (config_.ownership == Ownership::Interop && !hostFallback_) {
        throw std::invalid_argument(
                "ParticleField::setAttributes: this field is Ownership::Interop — its "
                "attributes are written device-to-device through the second handle "
                "VulkanRenderer::enableParticleFieldInterop returns. setAttributes() is "
                "legal here only after the renderer reports hostFallback()");
    }
    if (n > config_.capacity) n = config_.capacity;
    if (n > 0 && !rgba) {
        throw std::invalid_argument("ParticleField::setAttributes: null source with n > 0");
    }
    // Sized to CAPACITY and zero-filled past n: the backend uploads the whole
    // buffer once, and a slot the host never wrote must read as black rather
    // than as whatever the allocator handed us.
    attr_.assign(std::size_t(config_.capacity) * 4u, 0.f);
    if (n > 0) {
        std::memcpy(attr_.data(), rgba, std::size_t(n) * 4u * sizeof(float));
    }
    ++attrSerial_;
}

void ParticleField::setEmitter(const EmitterParams& params) {

    if (config_.ownership != Ownership::Renderer) {
        throw std::invalid_argument(
                "ParticleField::setEmitter: the device emitter only exists for "
                "Ownership::Renderer fields; a HostRing field's positions come from "
                "submit()");
    }
    emitter_ = params;
    // Floors, applied here rather than in the shader for the same reason
    // DensityRepr::tempFalloff is clamped host-side: a division by a period of
    // zero would poison every slot in the field, and the host pays for it once
    // instead of once per particle per frame.
    emitter_.lifetime       = std::max(emitter_.lifetime, 1e-3f);
    emitter_.lifetimeJitter = std::max(0.f, std::min(1.f, emitter_.lifetimeJitter));
    emitter_.dutyCycle      = std::max(1e-3f, std::min(1.f, emitter_.dutyCycle));
    emitter_.driftGrowth    = std::max(0.f, std::min(1.f, emitter_.driftGrowth));
    emitter_.sizeJitter     = std::max(0.f, std::min(1.f, emitter_.sizeJitter));
    emitter_.size           = std::max(emitter_.size, 0.f);
}

void ParticleField::setEmitterTime(float timeSec, float dtSec) {

    if (config_.ownership != Ownership::Renderer) {
        throw std::invalid_argument(
                "ParticleField::setEmitterTime: only an Ownership::Renderer field has an "
                "emitter clock to advance");
    }
    emitTime_ = timeSec;
    // A negative dt would put prevPositions in the FUTURE and reverse every
    // motion vector — the exact defect the plan says numbers will not catch.
    emitDt_ = std::max(dtSec, 0.f);
}

void ParticleField::setFollowCenter(const Vector3& worldCenter) {

    if (config_.ownership != Ownership::Renderer) {
        throw std::invalid_argument(
                "ParticleField::setFollowCenter: only an Ownership::Renderer field has a "
                "device emitter to re-centre");
    }
    // floor(), not round(): floor is monotone and has no tie, so a camera
    // creeping across a lattice line crosses it ONCE. round()'s tie at .5 puts
    // the boundary where float noise can flip it back and forth between two
    // frames, which is a whole box of snow teleporting twice for no motion.
    const float g = emitter_.followSnap;
    if (g > 1e-4f) {
        followCenter_.set(std::floor(worldCenter.x / g) * g,
                          worldCenter.y,// unsnapped: the wrap is lateral only
                          std::floor(worldCenter.z / g) * g);
    } else {
        followCenter_.copy(worldCenter);
    }
}

void ParticleField::setLiveCount(std::uint32_t n) {

    if (n > config_.capacity) n = config_.capacity;
    if (n == liveCount_) return;
    liveCount_ = n;
    ++dataSerial_;
}
