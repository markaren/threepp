
#ifndef THREEPP_AGXPHYSICS_HPP
#define THREEPP_AGXPHYSICS_HPP

#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/threepp.hpp"
#include "threepp/utils/BufferGeometryUtils.hpp"

#include <agx/Hinge.h>
#include <agx/RigidBody.h>
#include <agxCable/Cable.h>
#include <agxCollide/Geometry.h>
#include <agxCollide/ShapePrimitives.h>
#include <agxCollide/Trimesh.h>
#include <agxIO/ReaderWriter.h>
#include <agxSDK/Assembly.h>
#include <agxSDK/Simulation.h>
#include <agxWire/Wire.h>

namespace threepp {

    template<typename Constraint>
    agx::ref_ptr<Constraint> createConstraint(const agx::Vec3& pos, const agx::Vec3& axis, agx::RigidBody* rb1, agx::RigidBody* rb2 = nullptr) {

        auto f1 = new agx::Frame();
        auto f2 = new agx::Frame();
        agx::Constraint::calculateFramesFromBody(pos, axis, rb1, f1, rb2, f2);
        return new Constraint(rb1, f1, rb2, f2);
    }


    template<typename Constraint>
    agx::ref_ptr<Constraint> createConstraint(const agx::Vec3& axis, agx::RigidBody* rb1, agx::RigidBody* rb2 = nullptr) {

        return createConstraint<Constraint>(agx::Vec3(), axis, rb1, rb2);
    }

    namespace detail {

        std::shared_ptr<Material> defaultWireMaterial() {
            auto m = MeshBasicMaterial::create();
            m->color = Color::whitesmoke;
            return m;
        }

        std::shared_ptr<Material> defaultGeometryMaterial() {
            auto m = MeshPhongMaterial::create();
            m->color = Color::gray;
            return m;
        }

    }// namespace detail

    class Wire: public Mesh {

    public:
        explicit Wire(const std::shared_ptr<Material>& material): Mesh(BufferGeometry::create(), material) {}

        void update(agxWire::Wire* w) {

            std::vector<Vector3> points;
            auto it = w->getRenderBeginIterator();
            const auto end = w->getRenderEndIterator();
            while (it != end) {
                auto node = *it;
                auto& pos = node->getWorldPosition();
                points.emplace_back(
                        static_cast<float>(pos.x()),
                        static_cast<float>(pos.y()),
                        static_cast<float>(pos.z()));
                it++;
            }

            auto path = std::make_shared<CatmullRomCurve3>(points);
            auto tube = TubeGeometry::create(path, 64, w->getRadius(), 32);

            setGeometry(tube);
        }
    };

    class AgxVisualisation: public Object3D {

    public:
        bool showConstraints = false;

        void makeVisual(agxSDK::Assembly* assembly, const std::shared_ptr<Material>& material = nullptr) {

            for (const auto& g : assembly->getGeometries()) {
                makeVisual(g, material);
            }

            for (const auto& rb : assembly->getRigidBodies()) {
                makeVisual(rb, material);
            }

            for (const auto& c : assembly->getConstraints()) {
                makeVisual(c);
            }

            for (const auto& a : assembly->getAssemblies()) {
                makeVisual(a, material);
            }

        }

        void makeVisual(agx::Constraint* c) {

            if (c->is<agx::Hinge>()) {
                auto attachment = c->getAttachment(agx::UInt(0));
                auto pos = attachment->get(agx::Attachment::ANCHOR_POS);
                auto frame = attachment->getFrame();
                auto rot = frame->getLocalRotate();
                auto arrow = ArrowHelper::create();

                arrow->setLength(1, 0.2f, 0.2f);
                arrow->visible = showConstraints;
                arrow->setColor(Color::orange);
                arrow->position.set(pos.x(), pos.y(), pos.z());
                arrow->quaternion.set(rot.x(), rot.y(), rot.z(), rot.w());
                arrow->rotateX(math::PI/2);
                constraints_[c] = arrow;
                Object3D::add(arrow);
            }
        }

        void makeVisual(agxWire::Wire* w, const std::shared_ptr<Material>& material = nullptr) {

            auto wire = std::make_shared<Wire>(material ? material : detail::defaultWireMaterial());
            wires_[w] = wire;
            Object3D::add(wire);
        }

        void makeVisual(agxCollide::Geometry* geometry, const std::shared_ptr<Material> material = nullptr) {
            auto shape = geometry->getShape();
            auto mesh = getMeshFromShape(shape, material ? material : detail::defaultGeometryMaterial());
            mesh->matrixAutoUpdate = false;
            updateVisualisationObject(*mesh, geometry->getTransform());
            geometries_[geometry] = mesh;
            Object3D::add(mesh);
        }

        void makeVisual(agx::RigidBody* rb, const std::shared_ptr<Material> material = nullptr) {
            auto group = Group::create();
            auto geometries = rb->getGeometries();
            for (const auto& geometry : geometries) {
                const auto shape = geometry->getShape();
                auto mesh = getMeshFromShape(shape, material ? material : detail::defaultGeometryMaterial());
                updateVisualisationObject(*mesh, geometry->getLocalTransform());
                mesh->matrixAutoUpdate = false;
                group->add(mesh);
            }
            bodies_[rb] = group;
            Object3D::add(group);
        }

        void updateVisuals() {
            for (auto& [g, o] : bodies_) {
                updateVisualisationObject(*o, g->getTransform());
            }
            for (auto& [c, o] : constraints_) {
                o->visible = showConstraints;
            }
            for (auto& [w, o] : wires_) {
                o->update(w);
            }
        }

        static std::shared_ptr<AgxVisualisation> create(agxSDK::Simulation& sim) {

            return std::shared_ptr<AgxVisualisation>(new AgxVisualisation(sim));
        }

    private:
        agxSDK::Simulation& sim_;
        std::unordered_map<agx::RigidBody*, std::shared_ptr<Object3D>> bodies_;
        std::unordered_map<agxCollide::Geometry*, std::shared_ptr<Object3D>> geometries_;
        std::unordered_map<agx::Constraint*, std::shared_ptr<Object3D>> constraints_;
        std::unordered_map<agxWire::Wire*, std::shared_ptr<Wire>> wires_;

        explicit AgxVisualisation(agxSDK::Simulation& sim): sim_(sim) {}

        static void updateVisualisationObject(Object3D& o, const agx::AffineMatrix4x4& transform) {
            auto pos = transform.getTranslate();
            auto rot = transform.getRotate();
            o.position.set(pos.x(), pos.y(), pos.z());
            o.quaternion.set(rot.x(), rot.y(), rot.z(), rot.w());
            o.updateMatrix();
        }

        static std::shared_ptr<Mesh> getMeshFromShape(agxCollide::Shape* shape, const std::shared_ptr<Material>& material = MeshBasicMaterial::create()) {
            if (auto box = shape->asSafe<agxCollide::Box>()) {
                auto extents = box->getHalfExtents() * 2;
                return Mesh::create(BoxGeometry::create(extents.x(), extents.y(), extents.z()), material);
            } else if (auto sphere = shape->asSafe<agxCollide::Sphere>()) {
                return Mesh::create(SphereGeometry::create(sphere->getRadius()), material);
            } else if (auto cylinder = shape->asSafe<agxCollide::Cylinder>()) {
                float radius = cylinder->getRadius();
                float height = cylinder->getHeight();
                return Mesh::create(CylinderGeometry::create(radius, radius, height), material);
            } else if (auto capsule = shape->asSafe<agxCollide::Capsule>()) {
                float radius = capsule->getRadius();
                float height = capsule->getHeight();
                return Mesh::create(CapsuleGeometry::create(radius, height), material);
            } else {
                return nullptr;
            }
        }
    };

}// namespace threepp

#endif//THREEPP_AGXPHYSICS_HPP
