
#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/extras/imgui/imgui_context.hpp"
#include "threepp/threepp.hpp"

#include <agx/Hinge.h>
#include <agx/RigidBody.h>
#include <agxCollide/Geometry.h>
#include <agxCollide/ShapePrimitives.h>
#include <agxCollide/Trimesh.h>
#include <agxIO/ReaderWriter.h>
#include <agxSDK/Simulation.h>
#include <agxWire/Wire.h>

using namespace threepp;

namespace {

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

    class Wire: public Mesh {

    public:
        Wire(const std::shared_ptr<Material>& material): Mesh(BufferGeometry::create(), material) {}

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
        bool showContacts = false;

        void add(agxCollide::Geometry* geometry, const std::shared_ptr<Material>& material = MeshBasicMaterial::create()) {
            auto shape = geometry->getShape();
            auto mesh = getMeshFromShape(shape, material);
            mesh->matrixAutoUpdate = false;
            updateVisualisationObject(*mesh, geometry->getTransform());
            geometries_[geometry] = mesh;
            sim->add(geometry);
            Object3D::add(mesh);
        }

        void remove(agxCollide::Geometry* geometry) {

            geometries_.at(geometry)->removeFromParent();
            geometries_.erase(geometry);
            sim->remove(geometry);
        }

        void add(agx::RigidBody* rb, const std::shared_ptr<Material>& material = MeshBasicMaterial::create()) {
            auto group = Group::create();
            auto geometries = rb->getGeometries();
            for (auto& geometry : geometries) {
                auto shape = geometry->getShape();
                auto mesh = getMeshFromShape(shape, material);
                mesh->matrixAutoUpdate = false;
                group->add(mesh);
            }
            bodies_[rb] = group;
            sim->add(rb);
            Object3D::add(group);
        }

        void remove(agx::RigidBody* rb) {

            bodies_.at(rb)->removeFromParent();
            bodies_.erase(rb);
            sim->remove(rb);
        }

        void add(agx::Constraint* c) {

            if (c->is<agx::Hinge>()) {
                auto pos = c->getAttachment(agx::UInt(0))->get(agx::Attachment::Transformed::ANCHOR_POS);
                auto arrow = ArrowHelper::create();
                arrow->setLength(1, 0.2f, 0.2f);
                arrow->visible = showConstraints;
                arrow->setColor(Color::orange);
                arrow->position.set(pos.x(), pos.y(), pos.z());

                //            updateVisualisationObject(*arrow, frame->getMatrix());

                constraints_[c] = arrow;
                Object3D::add(arrow);
            }

            sim->add(c);
        }

        void remove(agx::Constraint* c) {

            constraints_.at(c)->removeFromParent();
            constraints_.erase(c);
            sim->remove(c);
        }

        void add(agxWire::Wire* w, const std::shared_ptr<Material>& material = MeshBasicMaterial::create()) {

            auto wire = std::make_shared<Wire>(material);
            wires_[w] = wire;
            Object3D::add(wire);
            sim->add(w);
        }

        void remove(agxWire::Wire* w) {

            wires_.at(w)->removeFromParent();
            wires_.erase(w);
            sim->remove(w);
        }

        void step(float dt) {
            t += dt;
            while ((sim->getTimeStamp() + sim->getTimeStep()) < t) {
                sim->stepForward();
            }
            updateVisuals();
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

        void saveScene(const std::string& name) {

            agxIO::writeFile(name + ".agx", sim);
        }

        static std::shared_ptr<AgxVisualisation> create() {

            return std::shared_ptr<AgxVisualisation>(new AgxVisualisation());
        }

    private:
        float t{};
        agx::AutoInit init;
        agxSDK::SimulationRef sim = new agxSDK::Simulation();
        std::unordered_map<agx::RigidBody*, std::shared_ptr<Object3D>> bodies_;
        std::unordered_map<agxCollide::Geometry*, std::shared_ptr<Object3D>> geometries_;
        std::unordered_map<agx::Constraint*, std::shared_ptr<Object3D>> constraints_;
        std::unordered_map<agxWire::Wire*, std::shared_ptr<Wire>> wires_;

        AgxVisualisation() {
            sim->setUniformGravity({0, -9.81, 0});
        }

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

}// namespace

int main() {

    Canvas canvas("agx_test", {{"antialiasing", 4}});
    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 100);
    camera->position.set(5, 5, 5);

    OrbitControls controls{camera, canvas};

    auto light = HemisphereLight::create();
    scene->add(light);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    auto agxVisualisation = AgxVisualisation::create();
    scene->add(agxVisualisation);

    agxCollide::GeometryRef sphereGeometry = new agxCollide::Geometry(new agxCollide::Sphere(0.1));
    sphereGeometry->setPosition({2, 0, 0});

    agx::RigidBodyRef boxBody = new agx::RigidBody(new agxCollide::Geometry(new agxCollide::Box(agx::Vec3{1, 1, 1})));
    boxBody->getMassProperties()->setMass(1);
    boxBody->setPosition({0, 2, 0});

    agx::RigidBodyRef sphereBody = new agx::RigidBody(new agxCollide::Geometry(new agxCollide::Sphere(0.5)));
    boxBody->setPosition({0, 4, 0});

    agxVisualisation->add(sphereGeometry, MeshPhongMaterial::create({{"color", Color::green}}));
    agxVisualisation->add(boxBody, MeshPhongMaterial::create({{"color", Color::gray}}));
    agxVisualisation->add(sphereBody, MeshPhongMaterial::create({{"color", Color::green}}));

    auto hinge = createConstraint<agx::Hinge>({0, 1, 0}, {0, 0, 1}, boxBody);
    hinge->getMotor1D()->setSpeed(1);
    hinge->getLock1D()->setEnable(false);
    hinge->getMotor1D()->setEnable(true);
    agxVisualisation->add(hinge);

    auto wire = new agxWire::Wire(0.01, 3);
    wire->add(new agxWire::BodyFixedNode(boxBody, {0, -1, 0}));
    wire->add(new agxWire::BodyFixedNode(sphereBody, {0, 0.5, 0}));

    agxVisualisation->add(wire);

    agxVisualisation->saveScene("vis");

    imgui_functional_context ui(canvas.window_ptr(), [&] {
        ImGui::SetNextWindowPos({}, 0, {});
        ImGui::SetNextWindowSize({230, 0}, 0);
        ImGui::Begin("agx_test");
        ImGui::Checkbox("showConstraints", &agxVisualisation->showConstraints);
        controls.enabled = !ImGui::IsWindowHovered();
        ImGui::End();
    });

    canvas.animate([&](float t) {
        agxVisualisation->step(t);

        renderer.render(scene, camera);
        ui.render();
    });
}
