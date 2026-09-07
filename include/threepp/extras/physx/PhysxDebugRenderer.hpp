
#ifndef THREEPP_PHYSX_DEBUG_RENDERER_HPP
#define THREEPP_PHYSX_DEBUG_RENDERER_HPP

#include "threepp/extras/physx/PhysxWorld.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/PointsMaterial.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Points.hpp"

#include <PxPhysicsAPI.h>

#include <algorithm>

namespace threepp {

    // Visualises PhysX debug primitives (collision shapes, joints, contact
    // points, ...) by reading PxScene::getRenderBuffer() each frame and writing
    // into vertex-coloured LineSegments / Points / Mesh children.
    //
    // Construct, add to the scene, and call update() each frame after world.step().
    // Toggle .visible to show/hide. Enable individual primitive groups via
    // setVisualizationParameter() on the underlying PxScene — enableDefaults()
    // turns on a sensible set (collision shapes + actor axes + joint frames).
    class PhysxDebugRenderer: public Object3D {

    public:
        explicit PhysxDebugRenderer(PhysxWorld& world)
            : world_(&world) {

            auto lineMat = LineBasicMaterial::create();
            lineMat->vertexColors = true;
            debugLines_ = LineSegments::create(BufferGeometry::create(), lineMat);

            auto pointMat = PointsMaterial::create();
            pointMat->vertexColors = true;
            pointMat->size = 5.f;
            pointMat->sizeAttenuation = false;
            debugPoints_ = Points::create(BufferGeometry::create(), pointMat);

            auto triMat = MeshBasicMaterial::create();
            triMat->vertexColors = true;
            triMat->side = Side::Double;
            triMat->transparent = true;
            triMat->opacity = 0.5f;
            debugTriangles_ = Mesh::create(BufferGeometry::create(), triMat);

            add(debugLines_);
            add(debugPoints_);
            add(debugTriangles_);

            visible = false;
        }

        // Turn on the most useful PxScene visualization channels. Without this
        // (or equivalent calls direct on the PxScene) the render buffer stays empty.
        void enableDefaults(float scale = 1.f) {
            using namespace ::physx;
            auto& s = world_->scene();
            s.setVisualizationParameter(PxVisualizationParameter::eSCALE, scale);
            s.setVisualizationParameter(PxVisualizationParameter::eCOLLISION_SHAPES, 1.f);
            s.setVisualizationParameter(PxVisualizationParameter::eACTOR_AXES, 1.f);
            s.setVisualizationParameter(PxVisualizationParameter::eJOINT_LIMITS, 1.f);
            s.setVisualizationParameter(PxVisualizationParameter::eJOINT_LOCAL_FRAMES, 1.f);
        }

        void update() {
            if (!parent || !visible) {
                debugLines_->visible = false;
                debugPoints_->visible = false;
                debugTriangles_->visible = false;
                return;
            }

            const auto& buffer = world_->scene().getRenderBuffer();

            uploadLines(buffer);
            uploadTriangles(buffer);
            uploadPoints(buffer);
        }

    private:
        PhysxWorld* world_;
        std::shared_ptr<LineSegments> debugLines_;
        std::shared_ptr<Points> debugPoints_;
        std::shared_ptr<Mesh> debugTriangles_;

        static void writeOrReplace(const std::shared_ptr<Object3D>& host,
                                   BufferGeometry& geom,
                                   const std::vector<float>& positions,
                                   const std::vector<float>& colors,
                                   int floatsPerVertex = 3) {
            auto* pos = geom.getAttribute<float>("position");
            auto* col = geom.getAttribute<float>("color");
            if (pos && col && pos->array().size() >= positions.size()) {
                std::ranges::copy(positions, pos->array().begin());
                std::ranges::copy(colors, col->array().begin());
                pos->needsUpdate();
                col->needsUpdate();
                geom.setDrawRange(0, static_cast<int>(positions.size() / floatsPerVertex));
            } else {
                auto fresh = BufferGeometry::create();
                fresh->setAttribute("position", FloatBufferAttribute::create(positions, 3));
                fresh->setAttribute("color", FloatBufferAttribute::create(colors, 3));
                if (auto m = std::dynamic_pointer_cast<Mesh>(host)) m->setGeometry(fresh);
                else if (auto l = std::dynamic_pointer_cast<Line>(host)) l->setGeometry(fresh);
                else if (auto p = std::dynamic_pointer_cast<Points>(host)) p->setGeometry(fresh);
            }
        }

        void uploadLines(const ::physx::PxRenderBuffer& buffer) {
            const auto n = buffer.getNbLines();
            std::vector<float> verts;
            std::vector<float> cols;
            verts.reserve(n * 6);
            cols.reserve(n * 6);
            for (::physx::PxU32 i = 0; i < n; ++i) {
                const auto& l = buffer.getLines()[i];
                verts.insert(verts.end(), {l.pos0.x, l.pos0.y, l.pos0.z, l.pos1.x, l.pos1.y, l.pos1.z});
                const Color c0(l.color0);
                const Color c1(l.color1);
                cols.insert(cols.end(), {c0.r, c0.g, c0.b, c1.r, c1.g, c1.b});
            }
            debugLines_->visible = !verts.empty();
            if (!verts.empty()) writeOrReplace(debugLines_, *debugLines_->geometry(), verts, cols);
        }

        void uploadTriangles(const ::physx::PxRenderBuffer& buffer) {
            const auto n = buffer.getNbTriangles();
            std::vector<float> verts;
            std::vector<float> cols;
            verts.reserve(n * 9);
            cols.reserve(n * 9);
            for (::physx::PxU32 i = 0; i < n; ++i) {
                const auto& t = buffer.getTriangles()[i];
                verts.insert(verts.end(), {
                                                  t.pos0.x, t.pos0.y, t.pos0.z,
                                                  t.pos1.x, t.pos1.y, t.pos1.z,
                                                  t.pos2.x, t.pos2.y, t.pos2.z});
                const Color c0(t.color0);
                const Color c1(t.color1);
                const Color c2(t.color2);
                cols.insert(cols.end(), {c0.r, c0.g, c0.b,
                                         c1.r, c1.g, c1.b,
                                         c2.r, c2.g, c2.b});
            }
            debugTriangles_->visible = !verts.empty();
            if (!verts.empty()) writeOrReplace(debugTriangles_, *debugTriangles_->geometry(), verts, cols);
        }

        void uploadPoints(const ::physx::PxRenderBuffer& buffer) {
            const auto n = buffer.getNbPoints();
            std::vector<float> verts;
            std::vector<float> cols;
            verts.reserve(n * 3);
            cols.reserve(n * 3);
            for (::physx::PxU32 i = 0; i < n; ++i) {
                const auto& p = buffer.getPoints()[i];
                verts.insert(verts.end(), {p.pos.x, p.pos.y, p.pos.z});
                const Color c(p.color);
                cols.insert(cols.end(), {c.r, c.g, c.b});
            }
            debugPoints_->visible = !verts.empty();
            if (!verts.empty()) writeOrReplace(debugPoints_, *debugPoints_->geometry(), verts, cols);
        }
    };

}// namespace threepp

#endif//THREEPP_PHYSX_DEBUG_RENDERER_HPP
