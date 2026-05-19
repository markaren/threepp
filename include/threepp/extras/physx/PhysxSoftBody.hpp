
#ifndef THREEPP_PHYSX_SOFT_BODY_HPP
#define THREEPP_PHYSX_SOFT_BODY_HPP

// SoftBody (deformable volume) integration for PhysxWorld. Requires
// PhysxWorld::Settings::enableGpuDynamics = true (PhysX runs soft bodies on
// CUDA). Each frame, PhysxWorld pulls deformed tet positions GPU->CPU and
// writes them into the bound BufferGeometry's position attribute.
//
// Two skinning modes:
//   - direct: visual geometry vertex count matches the collision tet mesh
//     (the common case when we cook from the visual geometry itself).
//   - barycentric: visual geometry vertex count differs (caller-supplied
//     higher-detail visual). Built once at addSoftBody time on a thread pool.

#include "threepp/extras/physx/PhysxWorld.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/objects/Mesh.hpp"

#include <PxPhysicsAPI.h>
#include <cudamanager/PxCudaContext.h>
#include <extensions/PxCudaHelpersExt.h>
#include <extensions/PxDeformableVolumeExt.h>
#include <extensions/PxRemeshingExt.h>
#include <extensions/PxTetMakerExt.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace threepp {

    namespace tet_util {

        // Signed 6x volume of a tetrahedron. Sign is positive for CCW-from-outside.
        [[nodiscard]] inline float tetVolume6(const ::physx::PxVec3& a,
                                              const ::physx::PxVec3& b,
                                              const ::physx::PxVec3& c,
                                              const ::physx::PxVec3& d) {
            return (b - a).dot((c - a).cross(d - a));
        }

        // Closest point on triangle (a,b,c) to p; returns the point and barycentric (u,v,w).
        // Standard Voronoi-region test from Real-Time Collision Detection §5.1.5.
        inline ::physx::PxVec3 closestPointTriangle(
                const ::physx::PxVec3& p,
                const ::physx::PxVec3& a,
                const ::physx::PxVec3& b,
                const ::physx::PxVec3& c,
                ::physx::PxVec3& bary) {
            using namespace ::physx;
            PxVec3 ab = b - a, ac = c - a, ap = p - a;
            float d1 = ab.dot(ap), d2 = ac.dot(ap);
            if (d1 <= 0 && d2 <= 0) {
                bary = PxVec3(1, 0, 0);
                return a;
            }
            PxVec3 bp = p - b;
            float d3 = ab.dot(bp), d4 = ac.dot(bp);
            if (d3 >= 0 && d4 <= d3) {
                bary = PxVec3(0, 1, 0);
                return b;
            }
            float vc = d1 * d4 - d3 * d2;
            if (vc <= 0 && d1 >= 0 && d3 <= 0) {
                float v = d1 / (d1 - d3);
                bary = PxVec3(1 - v, v, 0);
                return a + ab * v;
            }
            PxVec3 cp = p - c;
            float d5 = ab.dot(cp), d6 = ac.dot(cp);
            if (d6 >= 0 && d5 <= d6) {
                bary = PxVec3(0, 0, 1);
                return c;
            }
            float vb = d5 * d2 - d1 * d6;
            if (vb <= 0 && d2 >= 0 && d6 <= 0) {
                float w = d2 / (d2 - d6);
                bary = PxVec3(1 - w, 0, w);
                return a + ac * w;
            }
            float va = d3 * d6 - d5 * d4;
            if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
                float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                bary = PxVec3(0, 1 - w, w);
                return b + (c - b) * w;
            }
            float denom = 1.0f / (va + vb + vc);
            float v = vb * denom, w = vc * denom;
            bary = PxVec3(1 - v - w, v, w);
            return a + ab * v + ac * w;
        }

        struct ClosestTetResult {
            ::physx::PxVec3 point;
            float w0, w1, w2, w3;
        };

        // Closest point on the surface of a tetrahedron to v. Used as a fallback when
        // v lies outside every tetrahedron (vertices on the visual surface can sit
        // just outside the simulation hull after voxelisation).
        inline ClosestTetResult closestPointTetrahedron(
                const ::physx::PxVec3& v,
                const ::physx::PxVec3& p0,
                const ::physx::PxVec3& p1,
                const ::physx::PxVec3& p2,
                const ::physx::PxVec3& p3) {
            using namespace ::physx;
            float bestDist = FLT_MAX;
            ClosestTetResult best{};
            auto testFace = [&](const PxVec3& a, const PxVec3& b, const PxVec3& c, int missing) {
                PxVec3 bary;
                const PxVec3 q = closestPointTriangle(v, a, b, c, bary);
                const float d2 = (q - v).magnitudeSquared();
                if (d2 >= bestDist) return;
                bestDist = d2;
                best.point = q;
                float w[4] = {0, 0, 0, 0};
                if (missing == 0) { w[1] = bary.x; w[2] = bary.y; w[3] = bary.z; }
                if (missing == 1) { w[0] = bary.x; w[2] = bary.y; w[3] = bary.z; }
                if (missing == 2) { w[0] = bary.x; w[1] = bary.y; w[3] = bary.z; }
                if (missing == 3) { w[0] = bary.x; w[1] = bary.y; w[2] = bary.z; }
                float sum = w[0] + w[1] + w[2] + w[3];
                if (sum < 1e-6f) { w[0] = w[1] = w[2] = w[3] = 0.25f; sum = 1; }
                best.w0 = w[0] / sum; best.w1 = w[1] / sum;
                best.w2 = w[2] / sum; best.w3 = w[3] / sum;
            };
            testFace(p1, p2, p3, 0);
            testFace(p0, p2, p3, 1);
            testFace(p0, p1, p3, 2);
            testFace(p0, p1, p2, 3);
            return best;
        }

        // Extract the boundary triangles of a tet mesh (each interior face is shared
        // by exactly 2 tets, so faces with count==1 are the outer surface). Winding
        // is fixed so the normal points away from the opposite vertex.
        inline void extractBoundaryTriangles(::physx::PxTetrahedronMesh* tetMesh,
                                             std::vector<unsigned int>& outIndices) {
            using namespace ::physx;
            outIndices.clear();
            if (!tetMesh) return;
            const PxU32 nbTets = tetMesh->getNbTetrahedrons();
            const auto* tets = static_cast<const PxU32*>(tetMesh->getTetrahedrons());
            if (!tets || nbTets == 0) return;
            const auto* verts = tetMesh->getVertices();
            if (!verts) return;

            struct Key {
                std::array<PxU32, 3> s;
                bool operator==(Key const& o) const { return s == o.s; }
            };
            struct KeyHash {
                size_t operator()(Key const& k) const noexcept {
                    return static_cast<size_t>(k.s[0]) * 73856093u ^
                           static_cast<size_t>(k.s[1]) * 19349663u ^
                           static_cast<size_t>(k.s[2]) * 83492791u;
                }
            };
            struct Entry {
                int count = 0;
                std::array<PxU32, 3> tri{};
                PxU32 opposite = 0;
            };

            std::unordered_map<Key, Entry, KeyHash> counts;
            counts.reserve(nbTets * 4);
            auto pushFace = [&](PxU32 a, PxU32 b, PxU32 c, PxU32 opposite) {
                Key k{};
                k.s = {a, b, c};
                std::sort(k.s.begin(), k.s.end());
                auto& e = counts[k];
                if (e.count == 0) {
                    e.tri = {a, b, c};
                    e.opposite = opposite;
                }
                e.count += 1;
            };
            for (PxU32 ti = 0; ti < nbTets; ++ti) {
                const PxU32 i0 = tets[ti * 4 + 0];
                const PxU32 i1 = tets[ti * 4 + 1];
                const PxU32 i2 = tets[ti * 4 + 2];
                const PxU32 i3 = tets[ti * 4 + 3];
                pushFace(i0, i1, i2, i3);
                pushFace(i0, i1, i3, i2);
                pushFace(i0, i2, i3, i1);
                pushFace(i1, i2, i3, i0);
            }
            outIndices.reserve(counts.size() * 3);
            for (const auto& kv : counts) {
                const Entry& e = kv.second;
                if (e.count != 1) continue;
                auto tri = e.tri;
                const PxVec3& p0 = verts[tri[0]];
                const PxVec3& p1 = verts[tri[1]];
                const PxVec3& p2 = verts[tri[2]];
                const PxVec3 faceCenter = (p0 + p1 + p2) / 3.0f;
                const PxVec3 normal = (p1 - p0).cross(p2 - p0);
                const PxVec3& opV = verts[e.opposite];
                if (normal.dot(opV - faceCenter) > 0) {
                    std::swap(tri[1], tri[2]);
                }
                outIndices.push_back(tri[0]);
                outIndices.push_back(tri[1]);
                outIndices.push_back(tri[2]);
            }
        }

    }// namespace tet_util


    // Cook a PxDeformableVolumeMesh from a triangle surface (positions + indices).
    // - voxelResolution = number of voxels along the longest AABB axis when building
    //   the simulation mesh. Higher = finer simulation, more solver work.
    // - The collision mesh is conforming (matches the surface); the simulation mesh
    //   is voxelised (uniform, well-conditioned for the solver).
    inline ::physx::PxDeformableVolumeMesh* cookDeformableVolumeMesh(
            ::physx::PxPhysics& physics,
            const ::physx::PxCookingParams& params,
            const ::physx::PxArray<::physx::PxVec3>& triVerts,
            const ::physx::PxArray<::physx::PxU32>& triIndices,
            unsigned int voxelResolution = 10) {
        using namespace ::physx;
        PxSimpleTriangleMesh surfaceMesh;
        surfaceMesh.points.count = triVerts.size();
        surfaceMesh.points.data = triVerts.begin();
        surfaceMesh.triangles.count = triIndices.size() / 3;
        surfaceMesh.triangles.data = triIndices.begin();

        PxArray<PxVec3> collVerts = triVerts;
        PxArray<PxU32> collIndices = triIndices;
        PxArray<PxVec3> simVerts = triVerts;
        PxArray<PxU32> simIndices = triIndices;
        PxTetMaker::createConformingTetrahedronMesh(surfaceMesh, collVerts, collIndices);
        PxTetrahedronMeshDesc collDesc(collVerts, collIndices);

        PxArray<PxI32> vertexToTet;
        vertexToTet.resize(collDesc.points.count);
        PxTetMaker::createVoxelTetrahedronMesh(collDesc, voxelResolution, simVerts, simIndices, vertexToTet.begin());
        PxTetrahedronMeshDesc simDesc(simVerts, simIndices);
        PxDeformableVolumeSimulationDataDesc simDataDesc(vertexToTet);

        return PxCreateDeformableVolumeMesh(params, simDesc, collDesc, simDataDesc,
                                            physics.getPhysicsInsertionCallback());
    }


    // Owns a PxDeformableVolume and the GPU/CPU bridge for it. Built by PhysxWorld;
    // users hold a non-owning pointer. PhysxWorld::removeSoftBody() to destroy.
    class SoftBody {

    public:
        using TetBind = SoftBodyTetBind;

        // Constructor — internal use; prefer PhysxWorld::addSoftBody.
        // When cachedBindings is non-null, the expensive per-vertex binding
        // computation is skipped and the cached values are used directly.
        SoftBody(::physx::PxDeformableVolume* volume,
                 ::physx::PxCudaContextManager* cuda,
                 const std::shared_ptr<BufferGeometry>& visualGeometry,
                 const std::vector<TetBind>* cachedBindings = nullptr)
            : volume_(volume), cuda_(cuda), visualGeometry_(visualGeometry) {

            using namespace ::physx;
            auto* tetMesh = volume_->getCollisionMesh();
            nbCollVerts_ = tetMesh->getNbVertices();
            positionsInvMass_ = PX_EXT_PINNED_MEMORY_ALLOC(PxVec4, *cuda_, nbCollVerts_);

            pullDeformedPositionsSync();

            auto vPosAttr = visualGeometry_->getAttribute<float>("position");
            if (!vPosAttr) {
                throw std::runtime_error("SoftBody: visual geometry has no position attribute");
            }
            const size_t visualVerts = vPosAttr->count();
            useDirectMapping_ = (visualVerts == static_cast<size_t>(nbCollVerts_));

            if (!useDirectMapping_) {
                if (cachedBindings) {
                    bindings_ = *cachedBindings;
                } else {
                    buildBindings();
                }
            }
            applyDeformedPositions();
            visualGeometry_->computeVertexNormals();
        }

        SoftBody(const SoftBody&) = delete;
        SoftBody& operator=(const SoftBody&) = delete;

        ~SoftBody() {
            if (positionsInvMass_) {
                PX_EXT_PINNED_MEMORY_FREE(*cuda_, positionsInvMass_);
                positionsInvMass_ = nullptr;
            }
            if (volume_) {
                volume_->release();
                volume_ = nullptr;
            }
        }

        // Issue async DtoH copy on the supplied stream. Caller must already hold the
        // CUDA lock and must synchronise the stream before applyDeformedPositions().
        // This is the path used by PhysxWorld when batching many soft bodies per frame.
        void pullDeformedPositionsAsync(CUstream stream) const {
            using namespace ::physx;
            cuda_->getCudaContext()->memcpyDtoHAsync(
                    positionsInvMass_,
                    reinterpret_cast<CUdeviceptr>(volume_->getPositionInvMassBufferD()),
                    nbCollVerts_ * sizeof(PxVec4),
                    stream);
        }

        // Synchronous variant for one-shot use (initial binding, manual sync).
        void pullDeformedPositionsSync() const {
            using namespace ::physx;
            PxScopedCudaLock _lock(*cuda_);
            cuda_->getCudaContext()->memcpyDtoH(
                    positionsInvMass_,
                    reinterpret_cast<CUdeviceptr>(volume_->getPositionInvMassBufferD()),
                    nbCollVerts_ * sizeof(PxVec4));
        }

        // Write the current pinned positions into the visual geometry's position
        // attribute. Fast path: direct copy when vertex counts match. Slow path:
        // barycentric skinning via precomputed bindings.
        //
        // Writes go straight through the attribute's backing `std::vector<float>`
        // via `.data()` — `setXYZ` is virtual and bounds-checks per call, which
        // dominated this loop on hot soft-body scenes (~3-5× speedup measured).
        // The single `needsUpdate()` at the end bumps the version once, so the
        // renderer's geomVersion dirty-detection still fires correctly.
        void applyDeformedPositions() const {
            auto vPosAttr = visualGeometry_->getAttribute<float>("position");
            if (!vPosAttr) return;
            float* dst = vPosAttr->array().data();
            if (useDirectMapping_) {
                for (::physx::PxU32 i = 0; i < nbCollVerts_; ++i) {
                    const auto& p = positionsInvMass_[i];
                    dst[i * 3 + 0] = p.x;
                    dst[i * 3 + 1] = p.y;
                    dst[i * 3 + 2] = p.z;
                }
            } else {
                const auto& binds = bindings_;
                const auto* src = positionsInvMass_;
                for (size_t i = 0; i < binds.size(); ++i) {
                    const auto& b = binds[i];
                    dst[i * 3 + 0] = b.w0 * src[b.i0].x + b.w1 * src[b.i1].x +
                                     b.w2 * src[b.i2].x + b.w3 * src[b.i3].x;
                    dst[i * 3 + 1] = b.w0 * src[b.i0].y + b.w1 * src[b.i1].y +
                                     b.w2 * src[b.i2].y + b.w3 * src[b.i3].y;
                    dst[i * 3 + 2] = b.w0 * src[b.i0].z + b.w1 * src[b.i1].z +
                                     b.w2 * src[b.i2].z + b.w3 * src[b.i3].z;
                }
            }
            vPosAttr->needsUpdate();
        }

        [[nodiscard]] ::physx::PxDeformableVolume* actor() const { return volume_; }
        [[nodiscard]] const std::shared_ptr<BufferGeometry>& visualGeometry() const { return visualGeometry_; }

        // Non-owning back-pointer to the visual Mesh, set by PhysxWorld::addSoftBody(Mesh&).
        // Null when the soft body was created from a bare BufferGeometry overload.
        // PhysxWorld::removeSoftBody() uses it to detach the Mesh from its parent so
        // a single call cleans up both physics and scene graph.
        [[nodiscard]] Mesh* mesh() const { return mesh_; }

        // Compute normals each frame after the deformation is applied; default on.
        // Disable when the caller updates normals separately (or doesn't need them).
        void setRecomputeNormals(bool enabled) { recomputeNormals_ = enabled; }
        [[nodiscard]] bool recomputeNormals() const { return recomputeNormals_; }

    private:
        ::physx::PxDeformableVolume* volume_;
        ::physx::PxCudaContextManager* cuda_;
        ::physx::PxVec4* positionsInvMass_ = nullptr;
        ::physx::PxU32 nbCollVerts_ = 0;
        std::shared_ptr<BufferGeometry> visualGeometry_;
        Mesh* mesh_ = nullptr;
        bool useDirectMapping_ = true;
        std::vector<TetBind> bindings_;
        bool recomputeNormals_ = true;
        friend class PhysxWorld;

        // Build per-vertex bindings from visual mesh into the collision tet mesh.
        // Walks every tet's AABB (with a small pad) and tests barycentric containment.
        // Outside vertices fall back to closest point on tet surface. Threaded.
        void buildBindings() {
            using namespace ::physx;
            auto vPosAttr = visualGeometry_->getAttribute<float>("position");
            if (!vPosAttr) return;
            auto* tetMesh = volume_->getCollisionMesh();
            // Tet positions come from positionsInvMass_ (synced just above), NOT
            // tetMesh->getVertices(). getVertices() is the cooked rest pose in the
            // mesh's local space; when the volume is reused from PhysxWorld's cook
            // cache that local space differs from the world-space visual geometry,
            // so every barycentric containment test would miss. positionsInvMass_
            // is the live collision pose in the same world space as the visual
            // geometry — and the same array applyDeformedPositions() skins against.
            const PxVec4* verts = positionsInvMass_;
            const auto* tets = static_cast<const PxU32*>(tetMesh->getTetrahedrons());
            const PxU32 nbTets = tetMesh->getNbTetrahedrons();
            const size_t vVerts = vPosAttr->count();
            bindings_.resize(vVerts);

            struct TetData {
                PxU32 i0, i1, i2, i3;
                PxVec3 p0, p1, p2, p3;
                PxVec3 aabbMin, aabbMax;
            };
            std::vector<TetData> tetsData;
            tetsData.reserve(nbTets);
            const float aabbPad = 2e-2f;
            for (PxU32 ti = 0; ti < nbTets; ++ti) {
                PxU32 i0 = tets[ti * 4 + 0];
                PxU32 i1 = tets[ti * 4 + 1];
                PxU32 i2 = tets[ti * 4 + 2];
                PxU32 i3 = tets[ti * 4 + 3];
                PxVec3 p0 = verts[i0].getXYZ(), p1 = verts[i1].getXYZ(),
                       p2 = verts[i2].getXYZ(), p3 = verts[i3].getXYZ();
                PxVec3 mn(std::min({p0.x, p1.x, p2.x, p3.x}),
                          std::min({p0.y, p1.y, p2.y, p3.y}),
                          std::min({p0.z, p1.z, p2.z, p3.z}));
                PxVec3 mx(std::max({p0.x, p1.x, p2.x, p3.x}),
                          std::max({p0.y, p1.y, p2.y, p3.y}),
                          std::max({p0.z, p1.z, p2.z, p3.z}));
                mn -= PxVec3(aabbPad);
                mx += PxVec3(aabbPad);
                tetsData.push_back({i0, i1, i2, i3, p0, p1, p2, p3, mn, mx});
            }

            const auto& vpos = vPosAttr->array();
            const float eps = 5e-3f;
            unsigned threads = std::thread::hardware_concurrency();
            if (threads == 0) threads = 1;
            const size_t chunk = (vVerts + threads - 1) / threads;
            std::vector<std::thread> ths;
            ths.reserve(threads);
            for (unsigned t = 0; t < threads; ++t) {
                size_t start = t * chunk;
                size_t end = std::min(start + chunk, vVerts);
                if (start >= end) break;
                ths.emplace_back([&, start, end]() {
                    for (size_t vi = start; vi < end; ++vi) {
                        PxVec3 v(vpos[vi * 3 + 0], vpos[vi * 3 + 1], vpos[vi * 3 + 2]);
                        float bestDist = FLT_MAX;
                        TetBind best{tetsData[0].i0, tetsData[0].i1, tetsData[0].i2, tetsData[0].i3,
                                     0.25f, 0.25f, 0.25f, 0.25f};
                        bool found = false;
                        for (const TetData& T : tetsData) {
                            if (v.x < T.aabbMin.x || v.x > T.aabbMax.x ||
                                v.y < T.aabbMin.y || v.y > T.aabbMax.y ||
                                v.z < T.aabbMin.z || v.z > T.aabbMax.z) continue;
                            float V = tet_util::tetVolume6(T.p0, T.p1, T.p2, T.p3);
                            if (std::fabs(V) < 1e-8f) continue;
                            float w0 = tet_util::tetVolume6(v, T.p1, T.p2, T.p3) / V;
                            float w1 = tet_util::tetVolume6(T.p0, v, T.p2, T.p3) / V;
                            float w2 = tet_util::tetVolume6(T.p0, T.p1, v, T.p3) / V;
                            float w3 = tet_util::tetVolume6(T.p0, T.p1, T.p2, v) / V;
                            if (w0 >= -eps && w1 >= -eps && w2 >= -eps && w3 >= -eps) {
                                float sum = w0 + w1 + w2 + w3;
                                if (std::fabs(sum) < 1e-6f) sum = 1.0f;
                                bindings_[vi] = {T.i0, T.i1, T.i2, T.i3,
                                                 w0 / sum, w1 / sum, w2 / sum, w3 / sum};
                                found = true;
                                break;
                            }
                            auto r = tet_util::closestPointTetrahedron(v, T.p0, T.p1, T.p2, T.p3);
                            const float d2 = (r.point - v).magnitudeSquared();
                            if (d2 < bestDist) {
                                bestDist = d2;
                                best = {T.i0, T.i1, T.i2, T.i3, r.w0, r.w1, r.w2, r.w3};
                            }
                        }
                        if (!found) bindings_[vi] = best;
                    }
                });
            }
            for (auto& th : ths) {
                if (th.joinable()) th.join();
            }
        }
    };


    // Inline implementations of the PhysxWorld soft-body API, deferred to here so
    // they can see the full SoftBody definition.

    inline ::physx::PxDeformableVolumeMaterial* PhysxWorld::createSoftBodyMaterial(
            float youngsModulus, float poissonsRatio, float dynamicFriction) {
        if (!cuda_) throw std::runtime_error("PhysxWorld::createSoftBodyMaterial: enableGpuDynamics is false");
        return physics_->createDeformableVolumeMaterial(youngsModulus, poissonsRatio, dynamicFriction);
    }

    inline SoftBody* PhysxWorld::addSoftBody(
            const std::shared_ptr<BufferGeometry>& visualGeometry,
            ::physx::PxDeformableVolumeMaterial* material,
            int voxelResolution,
            unsigned solverIterations,
            bool selfCollision) {
        using namespace ::physx;
        if (!cuda_) throw std::runtime_error("PhysxWorld::addSoftBody: enableGpuDynamics is false");
        if (!visualGeometry) throw std::runtime_error("PhysxWorld::addSoftBody: visualGeometry is null");
        if (!material) material = defaultSoftBodyMaterial();

        // 1. Extract positions/indices into PxArrays (world-space, since the visual
        //    geometry is what we render). The simulation then runs in world space.
        PxArray<PxVec3> triVerts;
        PxArray<PxU32> triIndices;
        {
            const auto* posAttr = visualGeometry->getAttribute<float>("position");
            if (!posAttr) throw std::runtime_error("PhysxWorld::addSoftBody: geometry has no position attribute");
            const auto count = static_cast<PxU32>(posAttr->count());
            triVerts.resize(count);
            for (PxU32 i = 0; i < count; ++i) {
                triVerts[i] = PxVec3(posAttr->getX(i), posAttr->getY(i), posAttr->getZ(i));
            }
            const auto* idx = visualGeometry->getIndex();
            if (idx) {
                const auto n = static_cast<PxU32>(idx->count());
                triIndices.resize(n);
                for (PxU32 i = 0; i < n; ++i) triIndices[i] = idx->getX(i);
            } else {
                triIndices.reserve(count);
                for (PxU32 i = 0; i < count; ++i) triIndices.pushBack(i);
            }
        }

        // 2. Remesh + voxelise so the simulation mesh is well-conditioned. The
        //    voxelResolution counts cells along the longest AABB axis; 10 is a
        //    reasonable interactive default. Skip remeshing when resolution <= 0.
        if (voxelResolution > 0) {
            PxRemeshingExt::limitMaxEdgeLength(triIndices, triVerts, 1.0f);
            PxTetMaker::remeshTriangleMesh(triVerts, triIndices, static_cast<PxU32>(voxelResolution),
                                           triVerts, triIndices);
        }

        // 3. Cook deformable volume mesh.
        PxCookingParams params(physics_->getTolerancesScale());
        params.meshWeldTolerance = 0.001f;
        params.meshPreprocessParams = PxMeshPreprocessingFlags(PxMeshPreprocessingFlag::eWELD_VERTICES);
        params.buildTriangleAdjacencies = false;
        params.buildGPUData = true;
        const int res = (voxelResolution > 0) ? voxelResolution : 6;
        PxDeformableVolumeMesh* mesh = cookDeformableVolumeMesh(*physics_, params, triVerts, triIndices,
                                                                static_cast<unsigned>(res));
        if (!mesh) throw std::runtime_error("PhysxWorld::addSoftBody: failed to cook deformable volume mesh");

        // 4. Instantiate the deformable volume + shape + simulation mesh.
        PxDeformableVolume* volume = physics_->createDeformableVolume(*cuda_);
        const PxShapeFlags shapeFlags = PxShapeFlag::eVISUALIZATION | PxShapeFlag::eSIMULATION_SHAPE;
        PxTetrahedronMeshGeometry tetGeom(mesh->getCollisionMesh());
        PxShape* shape = physics_->createShape(tetGeom, &material, 1, true, shapeFlags);
        volume->attachShape(*shape);
        volume->attachSimulationMesh(*mesh->getSimulationMesh(), *mesh->getDeformableVolumeAuxData());
        shape->release();

        // 5. Place into world space, set mass from density, push to GPU.
        PxVec4 *simPos, *simVel, *collPos, *restPos;
        PxDeformableVolumeExt::allocateAndInitializeHostMirror(
                *volume, cuda_, simPos, simVel, collPos, restPos);
        constexpr PxReal maxInvMassRatio = 50.f;
        constexpr PxReal density = 1.f;
        constexpr PxReal scale = 1.f;
        PxDeformableVolumeExt::transform(*volume, PxTransform(PxIdentity), scale,
                                         simPos, simVel, collPos, restPos);
        PxDeformableVolumeExt::updateMass(*volume, density, maxInvMassRatio, simPos);
        PxDeformableVolumeExt::copyToDevice(*volume, PxDeformableVolumeDataFlag::eALL,
                                            simPos, simVel, collPos, restPos);
        PX_EXT_PINNED_MEMORY_FREE(*cuda_, simPos);
        PX_EXT_PINNED_MEMORY_FREE(*cuda_, simVel);
        PX_EXT_PINNED_MEMORY_FREE(*cuda_, collPos);
        PX_EXT_PINNED_MEMORY_FREE(*cuda_, restPos);

        volume->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, !selfCollision);
        volume->setSolverIterationCounts(solverIterations);

        scene_->addActor(*volume);

        auto sb = std::make_unique<SoftBody>(volume, cuda_, visualGeometry);
        SoftBody* raw = sb.get();
        softBodies_.push_back(std::move(sb));
        return raw;
    }

    inline SoftBody* PhysxWorld::addSoftBody(
            Mesh& mesh,
            ::physx::PxDeformableVolumeMaterial* material,
            int voxelResolution,
            unsigned solverIterations,
            bool selfCollision,
            const std::string& cacheKey) {
        using namespace ::physx;
        auto geom = mesh.geometry();
        if (!geom) throw std::runtime_error("PhysxWorld::addSoftBody(Mesh): no geometry");
        mesh.updateMatrixWorld();
        auto* posAttr = geom->getAttribute<float>("position");
        if (!posAttr) throw std::runtime_error("PhysxWorld::addSoftBody(Mesh): geometry has no position attribute");

        // Bake mesh.matrixWorld into a visual geometry clone; reset mesh transform
        // to identity. PhysX writes world-space positions back each frame.
        auto bakedGeom = BufferGeometry::create();
        std::vector<float> bakedPos(posAttr->array());
        Matrix4 mw = *mesh.matrixWorld;
        for (size_t i = 0; i < bakedPos.size() / 3; ++i) {
            Vector3 v(bakedPos[i * 3], bakedPos[i * 3 + 1], bakedPos[i * 3 + 2]);
            v.applyMatrix4(mw);
            bakedPos[i * 3] = v.x;
            bakedPos[i * 3 + 1] = v.y;
            bakedPos[i * 3 + 2] = v.z;
        }
        bakedGeom->setAttribute("position", FloatBufferAttribute::create(bakedPos, 3));
        for (const auto& [name, attr] : geom->getAttributes()) {
            if (name != "position") {
                bakedGeom->setAttribute(name, attr);
            }
        }
        if (auto* idx = geom->getIndex()) {
            bakedGeom->setIndex(std::vector<unsigned int>(idx->array().begin(), idx->array().end()));
        }
        if (!geom->groups.empty()) {
            bakedGeom->groups = geom->groups;
        }
        mesh.setGeometry(bakedGeom);
        mesh.position.set(0, 0, 0);
        mesh.quaternion.set(0, 0, 0, 1);
        mesh.scale.set(1, 1, 1);

        if (cacheKey.empty()) {
            SoftBody* raw = addSoftBody(bakedGeom, material, voxelResolution, solverIterations, selfCollision);
            raw->mesh_ = &mesh;
            return raw;
        }

        // --- Cached path: cook from local geometry, apply world transform ---
        if (!cuda_) throw std::runtime_error("PhysxWorld::addSoftBody: enableGpuDynamics is false");
        if (!material) material = defaultSoftBodyMaterial();

        Vector3 wPos, wScl;
        Quaternion wRot;
        mw.decompose(wPos, wRot, wScl);
        PxTransform spawnTf(toPxVec3(wPos), toPxQuat(wRot));

        std::string fullKey = cacheKey + "_v" + std::to_string(voxelResolution);
        CookCacheEntry* entry = nullptr;
        PxDeformableVolumeMesh* cookedMesh;

        auto it = cookCache_.find(fullKey);
        if (it != cookCache_.end()) {
            entry = &it->second;
            cookedMesh = entry->mesh;
        } else {
            // Cook from LOCAL geometry (before world baking). This is the expensive
            // part that the cache eliminates on subsequent spawns.
            PxArray<PxVec3> triVerts;
            PxArray<PxU32> triIndices;
            const auto count = static_cast<PxU32>(posAttr->count());
            triVerts.resize(count);
            for (PxU32 i = 0; i < count; ++i) {
                triVerts[i] = PxVec3(posAttr->getX(i), posAttr->getY(i), posAttr->getZ(i));
            }
            const auto* idx = geom->getIndex();
            if (idx) {
                const auto n = static_cast<PxU32>(idx->count());
                triIndices.resize(n);
                for (PxU32 i = 0; i < n; ++i) triIndices[i] = idx->getX(i);
            } else {
                triIndices.reserve(count);
                for (PxU32 i = 0; i < count; ++i) triIndices.pushBack(i);
            }

            if (voxelResolution > 0) {
                PxRemeshingExt::limitMaxEdgeLength(triIndices, triVerts, 1.0f);
                PxTetMaker::remeshTriangleMesh(triVerts, triIndices,
                                               static_cast<PxU32>(voxelResolution), triVerts, triIndices);
            }

            PxCookingParams params(physics_->getTolerancesScale());
            params.meshWeldTolerance = 0.001f;
            params.meshPreprocessParams = PxMeshPreprocessingFlags(PxMeshPreprocessingFlag::eWELD_VERTICES);
            params.buildTriangleAdjacencies = false;
            params.buildGPUData = true;
            const int res = (voxelResolution > 0) ? voxelResolution : 6;
            cookedMesh = cookDeformableVolumeMesh(*physics_, params, triVerts, triIndices,
                                                  static_cast<unsigned>(res));
            if (!cookedMesh) throw std::runtime_error("PhysxWorld::addSoftBody: failed to cook deformable volume mesh");

            cookCache_[fullKey] = {cookedMesh, {}, false};
            entry = &cookCache_[fullKey];
        }

        // Instantiate a new deformable volume from the (possibly cached) mesh.
        PxDeformableVolume* volume = physics_->createDeformableVolume(*cuda_);
        const PxShapeFlags shapeFlags = PxShapeFlag::eVISUALIZATION | PxShapeFlag::eSIMULATION_SHAPE;
        PxTetrahedronMeshGeometry tetGeom(cookedMesh->getCollisionMesh());
        PxShape* shape = physics_->createShape(tetGeom, &material, 1, true, shapeFlags);
        volume->attachShape(*shape);
        volume->attachSimulationMesh(*cookedMesh->getSimulationMesh(), *cookedMesh->getDeformableVolumeAuxData());
        shape->release();

        PxVec4 *simPos, *simVel, *collPos, *restPos;
        PxDeformableVolumeExt::allocateAndInitializeHostMirror(
                *volume, cuda_, simPos, simVel, collPos, restPos);
        constexpr PxReal maxInvMassRatio = 50.f;
        constexpr PxReal density = 1.f;
        PxDeformableVolumeExt::transform(*volume, spawnTf, 1.f, simPos, simVel, collPos, restPos);
        PxDeformableVolumeExt::updateMass(*volume, density, maxInvMassRatio, simPos);
        PxDeformableVolumeExt::copyToDevice(*volume, PxDeformableVolumeDataFlag::eALL,
                                            simPos, simVel, collPos, restPos);
        PX_EXT_PINNED_MEMORY_FREE(*cuda_, simPos);
        PX_EXT_PINNED_MEMORY_FREE(*cuda_, simVel);
        PX_EXT_PINNED_MEMORY_FREE(*cuda_, collPos);
        PX_EXT_PINNED_MEMORY_FREE(*cuda_, restPos);

        volume->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, !selfCollision);
        volume->setSolverIterationCounts(solverIterations);
        scene_->addActor(*volume);

        // Build or reuse bindings. Bindings are barycentric weights + tet indices,
        // invariant under rigid transforms, so the set built for the first spawn of
        // this cooked mesh is valid for every later spawn regardless of placement.
        const std::vector<SoftBodyTetBind>* cachedBindings = nullptr;
        if (entry->hasBindings) {
            cachedBindings = &entry->bindings;
        }

        auto sb = std::make_unique<SoftBody>(volume, cuda_, bakedGeom, cachedBindings);
        if (!entry->hasBindings && !sb->useDirectMapping_) {
            entry->bindings = sb->bindings_;
            entry->hasBindings = true;
        }

        SoftBody* raw = sb.get();
        raw->mesh_ = &mesh;
        softBodies_.push_back(std::move(sb));
        return raw;
    }

    inline void PhysxWorld::removeSoftBody(SoftBody* softBody) {
        if (!softBody) return;
        if (auto* m = softBody->mesh(); m && m->parent) {
            m->removeFromParent();
        }
        if (auto* a = softBody->actor()) {
            scene_->removeActor(*a);
        }
        softBodies_.erase(
                std::remove_if(softBodies_.begin(), softBodies_.end(),
                               [&](const std::unique_ptr<SoftBody>& s) { return s.get() == softBody; }),
                softBodies_.end());
    }

    inline ::physx::PxDeformableVolumeMaterial* PhysxWorld::defaultSoftBodyMaterial() {
        if (!defaultSoftBodyMat_) {
            defaultSoftBodyMat_ = createSoftBodyMaterial(1e6f, 0.45f, 0.5f);
        }
        return defaultSoftBodyMat_;
    }

    inline void PhysxWorld::syncSoftBodies() {
        if (softBodies_.empty()) return;
        using namespace ::physx;
        {
            // Batch GPU->CPU copies on a dedicated stream, then sync once.
            PxScopedCudaLock _lock(*cuda_);
            for (auto& sb : softBodies_) {
                sb->pullDeformedPositionsAsync(cudaCopyStream_);
            }
            cuda_->getCudaContext()->streamSynchronize(cudaCopyStream_);
        }
        for (auto& sb : softBodies_) {
            sb->applyDeformedPositions();
            // Recompute bounds so frustum culling tracks the deformed body — the
            // initial bounds only cover the rest pose. Both sphere AND box are
            // needed: GL/WGPU renderers cull on boundingSphere, Vulkan culls on
            // boundingBox. Without the box refresh, a settled body that has
            // fallen below its spawn-y stays culled out by Vulkan's raster
            // gbuf pass once the camera approaches it from below.
            sb->visualGeometry()->computeBoundingSphere();
            sb->visualGeometry()->computeBoundingBox();
            if (sb->recomputeNormals()) sb->visualGeometry()->computeVertexNormals();
        }
    }

}// namespace threepp

#endif//THREEPP_PHYSX_SOFT_BODY_HPP
