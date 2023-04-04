
#include "threepp/materials/RawShaderMaterial.hpp"
#include "threepp/threepp.hpp"

#include <string>

namespace {

    std::string vertexSource() {

        return R"(
               #version 330 core
               #define attribute in
               #define varying out
               uniform mat4 modelViewMatrix; // optional
               uniform mat4 projectionMatrix; // optional
               attribute vec3 position;
               attribute vec4 color;
               varying vec3 vPosition;
               varying vec4 vColor;
               void main() {
                   vPosition = position;
                   vColor = color;
                   gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
               })";
    }

    std::string fragmentSource() {

        return R"(
                #version 330 core
                #define varying in
                out highp vec4 pc_fragColor;
                #define gl_FragColor pc_fragColor
                uniform float time;
                varying vec3 vPosition;
                varying vec4 vColor;
                void main() {
                   vec4 color = vec4( vColor );
                   color.r += sin( vPosition.x * 10.0 + time ) * 0.5;
                   gl_FragColor = color;
                })";
    }

}// namespace

using namespace threepp;

int main() {

    Canvas canvas("Raw Shader demo", {{"antialiasing", 4}});

    GLRenderer renderer(canvas);
    renderer.checkShaderErrors = true;
    auto scene = Scene::create();

    auto camera = PerspectiveCamera::create(60, canvas.getAspect(), 1, 10);
    camera->position.z = 2;

    int triangles = 1000;
    std::vector<float> positions;
    positions.reserve(triangles*3);
    std::vector<float> colors;
    colors.reserve(triangles*4);

    for (int i = 0; i < triangles; i++) {
        positions.emplace_back(math::random() - .5f);
        positions.emplace_back(math::random() - .5f);
        positions.emplace_back(math::random() - .5f);

        colors.emplace_back(math::random());
        colors.emplace_back(math::random());
        colors.emplace_back(math::random());
        colors.emplace_back(math::random());
    }

    auto geometry = BufferGeometry::create();
    geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));
    geometry->setAttribute("color", FloatBufferAttribute::create(colors, 4));

    auto material = RawShaderMaterial::create();
    (*material->uniforms)["time"] = Uniform();
    material->vertexShader = vertexSource();
    material->fragmentShader = fragmentSource();
    material->side = DoubleSide;
    material->transparent = true;

    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float t, float dt) {
        mesh->rotation.y = t * 0.5f;
        material->uniforms->at("time").setValue(t * 5);

        renderer.render(scene, camera);
    });
}
