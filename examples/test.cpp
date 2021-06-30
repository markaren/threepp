
#include <iostream>
#include <thread>
#include <threepp/core/Clock.hpp>
#include <threepp/lights/AmbientLight.hpp>
#include <threepp/math/Box2.hpp>
#include <threepp/math/Color.hpp>
#include <threepp/math/Euler.hpp>
#include <threepp/math/Line3.hpp>
#include <threepp/math/Matrix4.hpp>
#include <threepp/math/Quaternion.hpp>
#include <threepp/math/Triangle.hpp>
#include <threepp/math/Vector2.hpp>
#include <threepp/math/Vector3.hpp>
#include <threepp/math/Vector4.hpp>

#include <threepp/core/Uniform.hpp>

#include <threepp/scenes/Fog.hpp>

#include <threepp/cameras/PerspectiveCamera.hpp>
#include <threepp/cameras/OrthographicCamera.hpp>

#include <threepp/core/EventDispatcher.hpp>

#include <threepp/cameras/PerspectiveCamera.hpp>

#include <threepp/core/BufferAttribute.hpp>
#include <threepp/core/BufferGeometry.hpp>
#include <threepp/core/Layers.hpp>

#include "threepp/math/Sphere.hpp"

#include "threepp/materials/Material.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/materials/RawShaderMaterial.hpp"

#include "threepp/scenes/Scene.hpp"

#include <threepp/core/Object3D.hpp>
#include <threepp/geometries/BoxGeometry.hpp>
#include <threepp/geometries/SphereGeometry.hpp>
#include <threepp/objects/Line.hpp>
#include <threepp/objects/Mesh.hpp>
#include <vector>

#include "threepp/Canvas.hpp"

#include "threepp/renderers/gl/GLBindingStates.hpp"
#include "threepp/renderers/gl/GLBufferRenderer.hpp"
#include "threepp/renderers/gl/GLCapabilities.hpp"
#include "threepp/renderers/gl/GLClipping.hpp"
#include "threepp/renderers/gl/GLInfo.hpp"
#include "threepp/renderers/gl/GLMaterials.hpp"

#include "threepp/core/Uniform.hpp"

#include "threepp/renderers/shaders/UniformsLib.hpp"

#include "threepp/lights/LightShadow.hpp"

#include "threepp/utils/InstanceOf.hpp"

#include "threepp/renderers/shaders/ShaderChunk.hpp"
#include "threepp/renderers/shaders/ShaderLib.hpp"

using namespace threepp;

namespace {

    void test(Mesh *m) {
        std::cout << "mesh" << std::endl;
    }

    void test(Object3D *o) {
        std::cout << "o" << std::endl;
    }

    struct MyEventListener : EventListener {

        void onEvent(Event &e) override {
            std::cout << "Event type:" << e.type << std::endl;
        }
    };

}// namespace

int main() {

    Vector3 v;
    v[1] = 2;

    std::cout << "x=" << v << std::endl;

    std::vector<float> a{1, 2, 3};

    v.fromArray(a);

    std::cout << v << std::endl;

    std::vector<float> arr(3);

    v.toArray(arr);

    std::cout << arr[0] << std::endl;

    Matrix4 m4;
    m4.set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    m4[0] = 3;
    std::cout << m4[0] << std::endl;

    auto copy = m4;
    copy[0] = -99;

    std::cout << m4[0] << ":" << copy[0] << std::endl;

    Box2 b2;
    std::cout << b2 << std::endl;
    std::cout << "empty box: " << (b2.isEmpty() ? "true" : "false") << std::endl;

    threepp::Clock c;
    c.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    std::cout << "Elapsed=" << c.getElapsedTime() << std::endl;

    auto sphere = SphereGeometry::create();
    sphere->computeBoundingSphere();

    std::cout << "expected radius 1, got=" << sphere->boundingSphere->radius << std::endl;

    EventDispatcher evt;

    MyEventListener l;

    LambdaEventListener l1([](Event &e) {
        std::cout << "Event type:" << e.type << std::endl;
    });

    evt.addEventListener("per", &l);
    evt.addEventListener("truls", &l1);

    evt.dispatchEvent("per");
    evt.dispatchEvent("per");

    evt.removeEventListener("per", &l);

    evt.dispatchEvent("per");
    evt.dispatchEvent("truls");

    std::cout << "has per evt:" << evt.hasEventListener("per", &l) << std::endl;
    std::cout << "has truls evt:" << evt.hasEventListener("truls", &l1) << std::endl;


    Uniform uniform(m4);
    m4[0] = 98;
    auto m = uniform.value<Matrix4>();
    std::cout << m4[0] << ":" << m[0] << std::endl;

    Fog f(Color::aliceblue);

    std::cout << AmbientLight::create(0xffffff)->type() << std::endl;

    std::vector<float> vec{1, 2};
    std::vector<float> vec2{-1, -1};

    auto b = TypedBufferAttribute<float>::create(vec, 2);
    b->copyArray(vec2);


    std::cout << b->getX(0) << std::endl;

    auto box = BoxGeometry::create();
    auto attr = box->getAttribute<float>("position");
    std::cout << attr->getX(0) << std::endl;
    attr->setX(0, 1);
    std::cout << attr->getX(0) << std::endl;
    std::cout << box->getAttribute<float>("position")->getX(0) << std::endl;

    auto obj = Object3D::create();

    auto o = Object3D::create();
    o->name = "parent";
    auto p = PerspectiveCamera::create(60);

    o->add(p);

    p->far = 10;

    auto pp = std::dynamic_pointer_cast<PerspectiveCamera>(o->children[0]);

    std::cout << o->children[0]->type() << std::endl;

    std::cout << "Expecting " << p->far << ", got " << pp->far << std::endl;
    std::cout << pp->parent->name << std::endl;

    o->remove(pp);

    std::cout << "Expected 0, got " << o->children.size() << std::endl;

    auto boxGeometry = BoxGeometry::create();
    auto material = MeshBasicMaterial::create();
    auto mesh = Mesh::create(boxGeometry, material);

    Material *baseMaterial = material.get();

    std::cout << "RefractionRatio " << dynamic_cast<MaterialWithReflectivity *>(baseMaterial)->refractionRatio << std::endl;

    o->add(mesh);

    {
        auto objectWithGeometry = std::reinterpret_pointer_cast<Mesh>(o->children[0]);
        auto hasGeometry = objectWithGeometry->geometry() != nullptr;
        std::cout << "successful  " << (hasGeometry ? "true" : "false") << std::endl;
        auto g = objectWithGeometry->geometry();
    }

    std::vector<float> vf(3);
    auto fba = FloatBufferAttribute::create(vf, 3);

    std::cout << "instanceof: " << instanceof <FloatBufferAttribute>(fba.get()) << std::endl;

    std::vector<Uniform> uv;
    uv.emplace_back(Matrix4());

    auto &ulib = shaders::UniformsLib::instance();
    std::cout << ulib.common["diffuse"].value<Color>().r << std::endl;

    Vector3 v1;
    Vector3 v2;

    std::cout << "v1==v2: " << ((v1 == v2) ? "true" : "false") << std::endl;

//        Canvas canvas(Canvas::Parameters().title(""));
//        canvas.animate([](float dt) {
//            std::cout << gl::GLCapabilities::instance() << std::endl;
//        });

    o->clear();

    Uniform u;

    std::cout << "Color r=" << u.value<float>() << std::endl;
    u.value<float>() = 0.5f;
    std::cout << "Color r=" << u.value<float>() << std::endl;

    Vector3 myVec(1.f, 1.f, 1.f);
    Uniform u1(myVec);
    std::cout << u1.value<Vector3>() << std::endl;
    u1.value<Vector3>().y = -1;
    std::cout << u1.value<Vector3>() << std::endl;

    std::cout << shaders::ShaderChunk::instance().alphamap_fragment() << std::endl;


    return 0;
}
