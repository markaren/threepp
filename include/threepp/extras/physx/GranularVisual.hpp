// Drawing a pile of PBD grains: two implementations of one interface.
//
// PbdParticles is not an Object3D — the solver hands back a block of positions
// and rendering them is the application's job — so this is the other half of
// the granular story, promoted out of examples/projects/Physics/
// granular_conveyor.cpp (which still carries its own copies; converting it is
// optional cleanup). Both implementations do the same thing, "point N proxy
// meshes at N particle positions", and differ only in what that costs:
//
//   InstancedGrainVisual  one InstancedMesh, N instance matrices written per
//                         frame. The only path OpenGL has, and a first-class
//                         one there — the demo measures 500k moving grains at
//                         26 fps whole-app.
//   FieldGrainVisual      one threepp::ParticleField over the host ring.
//                         VULKAN ONLY (the type draws nothing elsewhere by
//                         decision). The per-frame CPU cost is ONE memcpy of
//                         the position block, and the renderer carries one
//                         entry whatever the grain count.
//
// Both are created ONCE at the group's full capacity and never re-created:
// capacity is immutable for a ParticleField (its churn contract), and an
// InstancedMesh that changes count() invalidates the Vulkan renderer's
// per-instance expansion — an entry-list rebuild, a device wait, and a cleared
// TAA history. Which is why the instanced count only ever steps in blocks.
//
// Orientations are baked once, from a seed, because PBD particles carry none:
// without them a pile reads as a lattice of identically-facing rocks.
//
// The positions handed in are the solver's, i.e. WORLD space. Neither visual
// transforms them, so the node returned by object() belongs wherever its own
// world matrix is identity — the scene root, in practice.

#ifndef THREEPP_PHYSX_GRANULARVISUAL_HPP
#define THREEPP_PHYSX_GRANULARVISUAL_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Rng.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/ParticleField.hpp"

#include <PxPhysicsAPI.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace threepp {

    class IGrainVisual {

    public:
        virtual ~IGrainVisual() = default;

        // Per frame, after PbdParticles::pull(): the group's first `n` live
        // positions (PxVec4, w = inverse mass — not a size).
        virtual void update(const ::physx::PxVec4* positions, unsigned n) = 0;

        // The node to put in the scene. Held rather than borrowed: a session
        // adds it, removes it, and lets go.
        [[nodiscard]] virtual std::shared_ptr<Object3D> object() const = 0;
    };

    namespace detail {

        // Uniform random orientation (Shoemake), one per grain. Shared by both
        // implementations so switching visual does not switch the pile.
        inline std::vector<Quaternion> grainOrientations(unsigned capacity, unsigned seed) {

            std::vector<Quaternion> out(capacity);
            math::Rng rng{seed};
            for (unsigned i = 0; i < capacity; ++i) {
                const float u1 = rng.nextFloat(), u2 = rng.nextFloat(), u3 = rng.nextFloat();
                const float s1 = std::sqrt(1.f - u1), s2 = std::sqrt(u1);
                out[i].set(s1 * std::sin(math::TWO_PI * u2), s1 * std::cos(math::TWO_PI * u2),
                           s2 * std::sin(math::TWO_PI * u3), s2 * std::cos(math::TWO_PI * u3));
            }
            return out;
        }

    }// namespace detail

    class InstancedGrainVisual: public IGrainVisual {

    public:
        InstancedGrainVisual(const std::shared_ptr<BufferGeometry>& geometry,
                             const std::shared_ptr<Material>& material, unsigned capacity,
                             unsigned seed)
            : rot_(std::size_t(capacity) * 9), capacity_(capacity) {

            mesh_ = InstancedMesh::create(geometry, material, capacity);
            mesh_->setCount(0);
            mesh_->frustumCulled = false;// the pile spans whatever it spans

            const auto quats = detail::grainOrientations(capacity, seed);
            auto& e = mesh_->instanceMatrix()->array();
            std::memset(e.data(), 0, e.size() * sizeof(float));
            Matrix4 m;
            for (unsigned i = 0; i < capacity_; ++i) {
                m.makeRotationFromQuaternion(quats[i]);
                for (int c = 0; c < 3; ++c)
                    for (int r = 0; r < 3; ++r)
                        rot_[std::size_t(i) * 9 + c * 3 + r] = m.elements[c * 4 + r];
                e[std::size_t(i) * 16 + 15] = 1.f;
            }
        }

        void update(const ::physx::PxVec4* positions, unsigned n) override {

            n = std::min(n, capacity_);
            auto& e = mesh_->instanceMatrix()->array();
            float* base = e.data();
            // Newly claimed slots get their (fixed) rotation written in; from
            // then on only the translation moves. Slots past `n` keep a zero
            // 3x3, which collapses the instance to a point — no pixels — so the
            // up-to-kStep-1 spares inside the current count step are invisible.
            for (unsigned i = claimed_; i < n; ++i) {
                float* b = base + std::size_t(i) * 16;
                const float* r = rot_.data() + std::size_t(i) * 9;
                b[0] = r[0]; b[1] = r[1]; b[2] = r[2];
                b[4] = r[3]; b[5] = r[4]; b[6] = r[5];
                b[8] = r[6]; b[9] = r[7]; b[10] = r[8];
            }
            claimed_ = std::max(claimed_, n);

            for (unsigned i = 0; i < n; ++i) {
                float* b = base + std::size_t(i) * 16;
                b[12] = positions[i].x;
                b[13] = positions[i].y;
                b[14] = positions[i].z;
            }
            mesh_->instanceMatrix()->needsUpdate();

            const unsigned want = std::min(capacity_, ((n + kStep - 1) / kStep) * kStep);
            if (want != mesh_->count()) mesh_->setCount(want);
        }

        [[nodiscard]] std::shared_ptr<Object3D> object() const override { return mesh_; }

    private:
        // Count granularity. Coarse on purpose: see the file header.
        static constexpr unsigned kStep = 4096;

        std::shared_ptr<InstancedMesh> mesh_;
        std::vector<float> rot_;// 3x3 per instance, written once, read on claim
        unsigned capacity_ = 0;
        unsigned claimed_ = 0;
    };

    class FieldGrainVisual: public IGrainVisual {

    public:
        FieldGrainVisual(const std::shared_ptr<BufferGeometry>& geometry,
                         const std::shared_ptr<Material>& material, unsigned capacity,
                         unsigned seed, float radius) {

            ParticleField::Config cfg;
            cfg.capacity = capacity;
            // The host ring, not Interop: the zero-copy device-to-device feed
            // needs THREEPP_PHYSX_CUDA_VK_INTEROP, which the apps do not define
            // (it is the stretch goal in plans/particle-authoring.md §8). What
            // arrives here is a pull()ed host mirror, so submit() is the API.
            cfg.ownership = ParticleField::Ownership::HostRing;
            // PhysX writes inverse mass into w, which says nothing about size,
            // so the proxy (authored at `radius`) draws at scale 1 and a
            // negative w is the dead-slot predicate.
            cfg.wSemantic = ParticleField::WSemantic::InvMass;
            cfg.uniformRadius = radius;
            cfg.orientations = true;
            field_ = ParticleField::create(cfg);
            field_->setMeshRepr(geometry, material);
            // The field's own geometry is the zero-area placeholder, so its
            // bounding sphere says nothing about where the grains are.
            field_->frustumCulled = false;

            // Byte-identical orientations to the instanced path's, from the
            // same seed, before the snorm16x4 quantisation the buffer applies.
            const auto quats = detail::grainOrientations(capacity, seed);
            std::vector<float> xyzw(std::size_t(capacity) * 4u);
            for (unsigned i = 0; i < capacity; ++i) {
                xyzw[std::size_t(i) * 4u + 0u] = quats[i].x;
                xyzw[std::size_t(i) * 4u + 1u] = quats[i].y;
                xyzw[std::size_t(i) * 4u + 2u] = quats[i].z;
                xyzw[std::size_t(i) * 4u + 3u] = quats[i].w;
            }
            field_->setOrientations(xyzw.data(), capacity);
        }

        void update(const ::physx::PxVec4* positions, unsigned n) override {

            // The entire per-frame CPU cost: no loop, no count step, no
            // entry-list consequence. PxVec4 IS ParticlePos, so no repack.
            field_->submit(positions, n);
        }

        [[nodiscard]] std::shared_ptr<Object3D> object() const override { return field_; }

    private:
        std::shared_ptr<ParticleField> field_;
    };

}// namespace threepp

#endif// THREEPP_PHYSX_GRANULARVISUAL_HPP
