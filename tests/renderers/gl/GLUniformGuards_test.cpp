// Guards on the uniform upload path, each a crash before it was added:
//
// - a sampler uniform whose value is null binds the renderer's empty texture
//   (three.js: `v || emptyTexture`) instead of dereferencing null, the shape
//   of a post pass whose input is assigned after its first render;
// - an array uniform given fewer elements than the shader declares loads a
//   partial array instead of reading past the vector;
// - a uniform declared with no value yet is skipped by the upload instead of
//   reading an empty optional, and is loaded once it has one.
//
// Each case draws a full-frame plane through a ShaderMaterial and reads the
// centre pixel, so the assertion is on what the shader actually saw. One
// renderer per case, reused across its renders: the second render of a case
// is the one that exercises the update path.

#include "gl_test_helpers.hpp"

#include "threepp/materials/ShaderMaterial.hpp"

namespace {

    // Clip-space pass-through: a 2x2 plane covers the frame whatever the camera.
    const char* fullFrameVertex = R"(
        void main() {
            gl_Position = vec4(position.xy, 0.0, 1.0);
        }
    )";

    struct Frame {

        std::shared_ptr<ShaderMaterial> material;
        std::shared_ptr<Scene> scene;
        std::shared_ptr<PerspectiveCamera> camera;
        GLRenderer renderer;

        explicit Frame(const std::string& fragment)
            : material(ShaderMaterial::create()),
              scene(Scene::create()),
              camera(PerspectiveCamera::create(50, 1.f, 0.1f, 10.f)),
              renderer(glCanvas()) {

            material->vertexShader = fullFrameVertex;
            material->fragmentShader = fragment;

            auto mesh = Mesh::create(PlaneGeometry::create(2.f, 2.f), material);
            mesh->frustumCulled = false;
            scene->add(mesh);

            camera->position.z = 1;
            renderer.setClearColor(Color(0x000000));
        }

        AvgColor centre() {
            renderer.render(*scene, *camera);
            return centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);
        }
    };

}// namespace

TEST_CASE("a null sampler2D uniform reads black, and the texture assigned later is read") {

    Frame f(R"(
        uniform sampler2D tDiffuse;
        void main() {
            gl_FragColor = vec4(texture2D(tDiffuse, vec2(0.5)).rgb + vec3(0.0, 0.0, 1.0), 1.0);
        }
    )");
    f.material->uniforms["tDiffuse"] = Uniform(static_cast<Texture*>(nullptr));

    const auto empty = f.centre();
    INFO("null texture: " << empty.r << " " << empty.g << " " << empty.b);
    CHECK(empty.r < 8);
    CHECK(empty.g < 8);
    CHECK(empty.b > 240);

    auto white = makeUniformTexture(255, 255, 255);
    f.material->uniforms["tDiffuse"].setValue(white.get());
    f.material->uniformsNeedUpdate = true;

    const auto assigned = f.centre();
    INFO("white texture: " << assigned.r << " " << assigned.g << " " << assigned.b);
    CHECK(assigned.r > 240);
    CHECK(assigned.g > 240);
}

TEST_CASE("an array uniform shorter than its declaration loads a partial array") {

    Frame f(R"(
        uniform vec3 pts[4];
        void main() {
            gl_FragColor = vec4(pts[0] + pts[1], 1.0);
        }
    )");
    f.material->uniforms["pts"] = Uniform(std::vector<Vector3>{Vector3(1, 0, 0), Vector3(0, 1, 0)});

    const auto c = f.centre();
    INFO(c.r << " " << c.g << " " << c.b);
    CHECK(c.r > 240);
    CHECK(c.g > 240);
    CHECK(c.b < 8);
}

TEST_CASE("a uniform declared without a value is skipped, then loaded once it has one") {

    Frame f(R"(
        uniform float later;
        void main() {
            gl_FragColor = vec4(later, 1.0, 0.0, 1.0);
        }
    )");
    f.material->uniforms["later"] = Uniform();

    // A linked program's floats start at zero, and nothing was loaded.
    const auto before = f.centre();
    INFO("no value: " << before.r << " " << before.g << " " << before.b);
    CHECK(before.r < 8);
    CHECK(before.g > 240);

    f.material->uniforms["later"].setValue(1.f);
    f.material->uniformsNeedUpdate = true;

    const auto after = f.centre();
    INFO("value set: " << after.r << " " << after.g << " " << after.b);
    CHECK(after.r > 240);
    CHECK(after.g > 240);
}

TEST_CASE("a struct uniform missing a member loads the members it has") {

    Frame f(R"(
        struct Light { vec3 color; float intensity; };
        uniform Light light;
        void main() {
            gl_FragColor = vec4(light.color, 1.0) + vec4(0.0, 0.0, light.intensity, 0.0);
        }
    )");
    // `intensity` is left out; a linked program's floats start at zero.
    f.material->uniforms["light"] = Uniform(std::unordered_map<std::string, NestedUniformValue>{{"color", Color(1.f, 0.f, 0.f)}});

    const auto c = f.centre();
    INFO(c.r << " " << c.g << " " << c.b);
    CHECK(c.r > 240);
    CHECK(c.g < 8);
    CHECK(c.b < 8);
}
