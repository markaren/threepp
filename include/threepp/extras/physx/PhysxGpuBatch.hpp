
#ifndef THREEPP_PHYSX_GPU_BATCH_HPP
#define THREEPP_PHYSX_GPU_BATCH_HPP

#include "threepp/extras/physx/PhysxWorld.hpp"

#include <PxPhysicsAPI.h>
#include <PxDirectGPUAPI.h>

#include <stdexcept>
#include <vector>

namespace threepp {

    // Batched GPU-resident state I/O for a set of reduced-coordinate articulations
    // sharing one direct-GPU PhysX scene. This is the hot-loop primitive for GPU
    // vectorized RL: one call reads ALL robots' joint/root state into a CUDA buffer,
    // or writes ALL drive targets from a CUDA buffer, with no CPU readback. Hand a
    // torch CUDA tensor's data_ptr() straight in (read/write) for zero-copy interop,
    // or use the *_host() helpers (which stage through host memory) for debugging.
    //
    // Requires PhysxWorld{Settings::enableDirectGpu=true}. Construct AFTER the
    // articulations have been finalize()'d into the scene; the constructor runs one
    // warmup simulate() so the GPU pipeline assigns each articulation its GPU index.
    //
    // Data layout (per articulation block, row-major [n, block]):
    //   JOINT_POSITION / JOINT_VELOCITY / JOINT_TARGET_POSITION : maxDofs floats
    //   ROOT_POSE  : 7 floats  [qx,qy,qz,qw, px,py,pz]  (PxTransform = quat then pos)
    //   ROOT_LINVEL / ROOT_ANGVEL : 3 floats
    class PhysxGpuBatch {
    public:
        using Read = ::physx::PxArticulationGPUAPIReadType;
        using Write = ::physx::PxArticulationGPUAPIWriteType;

        PhysxGpuBatch(PhysxWorld& world,
                      std::vector<::physx::PxArticulationReducedCoordinate*> arts)
            : world_(world), arts_(std::move(arts)) {
            using namespace ::physx;
            if (!world_.directGpuEnabled())
                throw std::runtime_error("PhysxGpuBatch: world must be created with enableDirectGpu");
            cuda_ = world_.cudaContextManager();
            if (!cuda_) throw std::runtime_error("PhysxGpuBatch: world has no CUDA context");
            n_ = static_cast<PxU32>(arts_.size());
            if (n_ == 0) throw std::runtime_error("PhysxGpuBatch: no articulations");

            // Warm up: the GPU sim assigns articulation GPU indices on the first
            // simulate(); they are not valid before that.
            world_.simulateRaw(world_.settings().fixedTimestep);

            auto& dg = world_.scene().getDirectGPUAPI();
            maxDofs_ = dg.getArticulationGPUAPIMaxCounts().maxDofs;

            std::vector<PxArticulationGPUIndex> idx(n_);
            for (PxU32 i = 0; i < n_; ++i) idx[i] = arts_[i]->getGPUIndex();

            PxScopedCudaLock lock(*cuda_);
            auto* ctx = cuda_->getCudaContext();
            ctx->memAlloc(&dIndices_, n_ * sizeof(PxArticulationGPUIndex));
            ctx->memcpyHtoD(dIndices_, idx.data(), n_ * sizeof(PxArticulationGPUIndex));
            ctx->eventCreate(&finishEvent_, 0 /*CU_EVENT_DEFAULT*/);
        }

        ~PhysxGpuBatch() {
            if (!cuda_) return;
            ::physx::PxScopedCudaLock lock(*cuda_);
            auto* ctx = cuda_->getCudaContext();
            if (finishEvent_) ctx->eventDestroy(finishEvent_);
            if (dIndices_) ctx->memFree(dIndices_);
        }

        PhysxGpuBatch(const PhysxGpuBatch&) = delete;
        PhysxGpuBatch& operator=(const PhysxGpuBatch&) = delete;

        ::physx::PxU32 count() const { return n_; }
        ::physx::PxU32 maxDofs() const { return maxDofs_; }

        // Number of floats per articulation block for a given read/write data type.
        static ::physx::PxU32 blockFloatsRead(Read::Enum t, ::physx::PxU32 maxDofs) {
            switch (t) {
                case Read::eJOINT_POSITION:
                case Read::eJOINT_VELOCITY: return maxDofs;
                case Read::eROOT_GLOBAL_POSE: return 7;
                case Read::eROOT_LINEAR_VELOCITY:
                case Read::eROOT_ANGULAR_VELOCITY: return 3;
                default: return maxDofs;
            }
        }
        static ::physx::PxU32 blockFloatsWrite(Write::Enum t, ::physx::PxU32 maxDofs) {
            switch (t) {
                case Write::eJOINT_POSITION:
                case Write::eJOINT_VELOCITY:
                case Write::eJOINT_TARGET_POSITION:
                case Write::eJOINT_TARGET_VELOCITY: return maxDofs;
                case Write::eROOT_GLOBAL_POSE: return 7;
                case Write::eROOT_LINEAR_VELOCITY:
                case Write::eROOT_ANGULAR_VELOCITY: return 3;
                default: return maxDofs;
            }
        }

        // Advance all articulations one substep on the GPU (no binding sync).
        void step(float dt) { world_.simulateRaw(dt); }

        // --- device-pointer (zero-copy) path -------------------------------------
        // `dst`/`src` are CUDA device pointers sized for n_ blocks (row-major). The
        // call blocks (event-synchronized) so the data is ready when it returns.
        void read(CUdeviceptr dst, Read::Enum type) {
            using namespace ::physx;
            PxScopedCudaLock lock(*cuda_);
            auto& dg = world_.scene().getDirectGPUAPI();
            if (!dg.getArticulationData(reinterpret_cast<void*>(dst), reinterpret_cast<const ::physx::PxArticulationGPUIndex*>(dIndices_), type, n_, nullptr, finishEvent_))
                throw std::runtime_error("PhysxGpuBatch::read: getArticulationData failed");
            cuda_->getCudaContext()->eventSynchronize(finishEvent_);
        }
        void write(CUdeviceptr src, Write::Enum type) {
            using namespace ::physx;
            PxScopedCudaLock lock(*cuda_);
            auto& dg = world_.scene().getDirectGPUAPI();
            if (!dg.setArticulationData(reinterpret_cast<const void*>(src), reinterpret_cast<const ::physx::PxArticulationGPUIndex*>(dIndices_), type, n_, nullptr, finishEvent_))
                throw std::runtime_error("PhysxGpuBatch::write: setArticulationData failed");
            cuda_->getCudaContext()->eventSynchronize(finishEvent_);
        }

        // Subset write: only the `nb` articulations whose GPU indices are in the
        // device buffer `subIndices`; `src` holds nb blocks. Used to reset just the
        // done envs each step without disturbing the others.
        void writeSubset(CUdeviceptr src, CUdeviceptr subIndices,
                         Write::Enum type, ::physx::PxU32 nb) {
            using namespace ::physx;
            if (nb == 0) return;
            PxScopedCudaLock lock(*cuda_);
            auto& dg = world_.scene().getDirectGPUAPI();
            if (!dg.setArticulationData(reinterpret_cast<const void*>(src),
                                        reinterpret_cast<const PxArticulationGPUIndex*>(subIndices),
                                        type, nb, nullptr, finishEvent_))
                throw std::runtime_error("PhysxGpuBatch::writeSubset: setArticulationData failed");
            cuda_->getCudaContext()->eventSynchronize(finishEvent_);
        }

        // The K articulation GPU indices (host copy) — upload to a torch cuda tensor
        // once to build subset-index buffers for resets.
        std::vector<::physx::PxU32> gpuIndicesHost() const {
            std::vector<::physx::PxU32> idx(n_);
            for (::physx::PxU32 i = 0; i < n_; ++i) idx[i] = arts_[i]->getGPUIndex();
            return idx;
        }

        // --- host-staged path (debugging / validation; no torch needed) ----------
        std::vector<float> readHost(Read::Enum type) {
            using namespace ::physx;
            const PxU32 block = blockFloatsRead(type, maxDofs_);
            const size_t bytes = static_cast<size_t>(n_) * block * sizeof(float);
            std::vector<float> host(static_cast<size_t>(n_) * block);
            PxScopedCudaLock lock(*cuda_);
            auto* ctx = cuda_->getCudaContext();
            CUdeviceptr scratch = 0;
            ctx->memAlloc(&scratch, bytes);
            auto& dg = world_.scene().getDirectGPUAPI();
            if (!dg.getArticulationData(reinterpret_cast<void*>(scratch), reinterpret_cast<const ::physx::PxArticulationGPUIndex*>(dIndices_), type, n_, nullptr, finishEvent_)) {
                ctx->memFree(scratch);
                throw std::runtime_error("PhysxGpuBatch::readHost: getArticulationData failed");
            }
            ctx->eventSynchronize(finishEvent_);
            ctx->memcpyDtoH(host.data(), scratch, bytes);
            ctx->memFree(scratch);
            return host;
        }
        void writeHost(const std::vector<float>& host, Write::Enum type) {
            using namespace ::physx;
            const PxU32 block = blockFloatsWrite(type, maxDofs_);
            const size_t bytes = static_cast<size_t>(n_) * block * sizeof(float);
            if (host.size() != static_cast<size_t>(n_) * block)
                throw std::runtime_error("PhysxGpuBatch::writeHost: wrong buffer size");
            PxScopedCudaLock lock(*cuda_);
            auto* ctx = cuda_->getCudaContext();
            CUdeviceptr scratch = 0;
            ctx->memAlloc(&scratch, bytes);
            ctx->memcpyHtoD(scratch, host.data(), bytes);
            auto& dg = world_.scene().getDirectGPUAPI();
            if (!dg.setArticulationData(reinterpret_cast<const void*>(scratch), reinterpret_cast<const ::physx::PxArticulationGPUIndex*>(dIndices_), type, n_, nullptr, finishEvent_)) {
                ctx->memFree(scratch);
                throw std::runtime_error("PhysxGpuBatch::writeHost: setArticulationData failed");
            }
            ctx->eventSynchronize(finishEvent_);
            ctx->memFree(scratch);
        }

    private:
        PhysxWorld& world_;
        std::vector<::physx::PxArticulationReducedCoordinate*> arts_;
        ::physx::PxCudaContextManager* cuda_ = nullptr;
        CUdeviceptr dIndices_ = 0;
        CUevent finishEvent_ = nullptr;
        ::physx::PxU32 n_ = 0;
        ::physx::PxU32 maxDofs_ = 0;
    };

}// namespace threepp

#endif//THREEPP_PHYSX_GPU_BATCH_HPP
