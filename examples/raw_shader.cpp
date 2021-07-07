
#include "threepp/threepp.hpp"

#include <string>

namespace {

    std::string vertexSource() {

        return "uniform mat4 modelViewMatrix; // optional\n"
               "\t\t\tuniform mat4 projectionMatrix; // optional\n"
               "\t\t\tattribute vec3 position;\n"
               "\t\t\tattribute vec4 color;\n"
               "\t\t\tvarying vec3 vPosition;\n"
               "\t\t\tvarying vec4 vColor;\n"
               "\t\t\tvoid main()\t{\n"
               "\t\t\t\tvPosition = position;\n"
               "\t\t\t\tvColor = color;\n"
               "\t\t\t\tgl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n"
               "\t\t\t}";
    }

    std::string fragmentSource() {

        return "uniform float time;\n"
               "\t\t\tvarying vec3 vPosition;\n"
               "\t\t\tvarying vec4 vColor;\n"
               "\t\t\tvoid main()\t{\n"
               "\t\t\t\tvec4 color = vec4( vColor );\n"
               "\t\t\t\tcolor.r += sin( vPosition.x * 10.0 + time ) * 0.5;\n"
               "\t\t\t\tgl_FragColor = color;\n"
               "\t\t\t}";
    }

    float myRandom() {
        return ((float) (rand() % 100)) / 100;
    }

}// namespace

using namespace threepp;

int main() {

    Canvas canvas(Canvas::Parameters()
                          .title("Raw Shader demo")
                          .antialising(8));

    auto renderer = GLRenderer(canvas);
    renderer.checkShaderErrors = true;
    auto scene = Scene::create();

    auto camera = PerspectiveCamera::create(60, canvas.getAspect(), 1, 10);
    camera->position.z = 2;

    int triangles = 500;

    auto geometry = BufferGeometry::create();

    std::vector<float> positions;
    std::vector<float> colors;


    for (int i = 0; i < triangles; i++) {
        positions.emplace_back(myRandom() - .5f);
        positions.emplace_back(myRandom() - .5f);
        positions.emplace_back(myRandom() - .5f);

        colors.emplace_back(myRandom());
        colors.emplace_back(myRandom());
        colors.emplace_back(myRandom());
    }

    geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));
    geometry->setAttribute("color", FloatBufferAttribute::create(colors, 3));

    auto material = RawShaderMaterial::create();
    material->uniforms->operator[]("time") = Uniform();
    material->vertexShader = vertexSource();
    material->fragmentShader = fragmentSource();
    material->side = DoubleSide;
    material->transparent = true;

    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    canvas.onWindowResize([&](WindowSize size){
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    float value = 0.f;
    mesh->rotation.order(Euler::RotationOrders::YZX);
    canvas.animate([&](float dt) {
        value += 0.005f ;
        mesh->rotation.y(value);
        material->uniforms->operator[]("time").setValue(value * 10);

        renderer.render(scene, camera);
    });

}