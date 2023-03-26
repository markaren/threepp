
#include "threepp/materials/RawShaderMaterial.hpp"
#include "threepp/threepp.hpp"

#include <string>

namespace {

    auto vertexSource() {

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
               void main(){
                   vPosition = position;
                   vColor = color;
                   gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
               })";
    }

    std::string fragmentSource() {

        return "#version 330 core\n\n"
               "#define varying in\n"
               "out highp vec4 pc_fragColor;\n"
               "#define gl_FragColor pc_fragColor\n"
               "uniform float time;\n"
               "varying vec3 vPosition;\n"
               "varying vec4 vColor;\n\n"
               "void main()\t{\n"
               "\tvec4 color = vec4( vColor );\n"
               "\tcolor.r += sin( vPosition.x * 10.0 + time ) * 0.5;\n"
               "\tgl_FragColor = color;\n"
               "}";
    }

}// namespace

using namespace threepp;

int main() {

    Canvas canvas(Canvas::Parameters()
                          .title("Raw Shader demo")
                          .antialiasing(8));

    GLRenderer renderer(canvas);
    renderer.checkShaderErrors = true;
    auto scene = Scene::create();

    auto camera = PerspectiveCamera::create(60, canvas.getAspect(), 1, 10);
    camera->position.z = 2;

    int triangles = 1000;

    auto geometry = BufferGeometry::create();

    std::vector<float> positions;
    std::vector<float> colors;

    for (int i = 0; i < triangles; i++) {
        positions.emplace_back(math::random() - .5f);
        positions.emplace_back(math::random() - .5f);
        positions.emplace_back(math::random() - .5f);

        colors.emplace_back(math::random());
        colors.emplace_back(math::random());
        colors.emplace_back(math::random());
        colors.emplace_back(math::random());
    }

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
