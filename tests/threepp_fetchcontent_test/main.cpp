#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    std::shared_ptr<Mesh> createBox(const Vector3 &pos, const Color &color) {
        const auto geometry = BoxGeometry::create();
        const auto material = MeshBasicMaterial::create();
        material->color.copy(color);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.copy(pos);

        return mesh;
    }

    std::unique_ptr<HUD> createHUD(WindowSize size) {
        auto hud = std::make_unique<HUD>(size);
        FontLoader fontLoader;
        const auto font = fontLoader.defaultFont();
        TextGeometry::Options opts(font, 20, 5);
        const auto hudText2 = Text2D::create(opts, "Hello World!");
        hudText2->setColor(Color::gray);
        hud->add(hudText2, HUD::Options()
                                  .setNormalizedPosition({1, 1})
                                  .setHorizontalAlignment(HUD::HorizontalAlignment::RIGHT)
                                  .setVerticalAlignment(HUD::VerticalAlignment::TOP));

        return hud;
    }

}// namespace

int main() {

    Canvas canvas("threepp demo", {{"aa", 4}});
    GLRenderer renderer(canvas.size());
    renderer.autoClear = false;// hud

    auto camera = PerspectiveCamera::create(50, canvas.aspect());
    camera->position.z = 5;

    OrbitControls controls{*camera, canvas};

    auto scene = Scene::create();
    scene->background = Color::aliceblue;

    auto group = Group::create();
    group->add(createBox({-1, 0, 0}, Color::green));
    group->add(createBox({1, 0, 0}, Color::blue));
    scene->add(group);

    const auto hud = createHUD(canvas.size());

    canvas.onWindowResize([&](const WindowSize &size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);

        hud->setSize(size);
    });

    // Load font file into memory
    std::vector<unsigned char> fontBuffer;
    {
        std::ifstream file("C:/Windows/Fonts/arial.ttf", std::ios::binary);
        fontBuffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
    }

    stbtt_fontinfo font;
    stbtt_InitFont(&font, fontBuffer.data(), stbtt_GetFontOffsetForIndex(fontBuffer.data(), 0));

    // Set up text and size
    std::string text = "Hello, stb!";
    float fontSize = 32.0f;
    int width = 256, height = 64;
    std::vector<unsigned char> pixels(width * height, 0);

    // Calculate scale
    float scale = stbtt_ScaleForPixelHeight(&font, fontSize);

    // Draw text
    int x = 0, y = int(fontSize);
    for (char c : text) {
        int ax, lsb;
        stbtt_GetCodepointHMetrics(&font, c, &ax, &lsb);

        int c_x1, c_y1, c_x2, c_y2;
        stbtt_GetCodepointBitmapBox(&font, c, scale, scale, &c_x1, &c_y1, &c_x2, &c_y2);

        int byteOffset = x + lsb * scale + (y + c_y1) * width;
        stbtt_MakeCodepointBitmap(&font, pixels.data() + byteOffset, c_x2 - c_x1, c_y2 - c_y1, width, scale, scale, c);

        x += ax * scale;
    }

    // Convert grayscale to RGBA
    std::vector<unsigned char> rgba(width * height * 4, 0);
    for (int i = 0; i < width * height; ++i) {
        rgba[i * 4 + 0] = 255;
        rgba[i * 4 + 1] = 255;
        rgba[i * 4 + 2] = 255;
        rgba[i * 4 + 3] = pixels[i];
    }

    Clock clock;
    float rotationSpeed = 1;
    canvas.animate([&] {
        const auto dt = clock.getDelta();
        group->rotation.y += rotationSpeed * dt;

        renderer.clear();//autoClear is false
        renderer.render(*scene, *camera);
        hud->apply(renderer);
    });
}
