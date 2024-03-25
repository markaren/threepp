
#ifndef THREEPP_DEBUGMAPSPROVIDER_HPP
#define THREEPP_DEBUGMAPSPROVIDER_HPP

#include "geo/providers/MapProvider.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/loaders/FontLoader.hpp"
#include "threepp/objects/Text.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/scenes/Scene.hpp"

namespace threepp {

    class DebugMapProvider: public MapProvider {

    public:
        explicit DebugMapProvider(GLRenderer* renderer)
            : renderer(renderer), hud(WindowSize(resolution, resolution)) {

            auto opts = TextGeometry::Options(loader.defaultFont(), 25);
            text1 = Text2D::create(opts);
            text1->setColor(Color::white);

            text2 = Text2D::create(opts);
            text2->setColor(Color::white);

            hud.add(text1, HUD::Options()
                                   .setNormalizedPosition({0.5, 0.6})
                                   .setHorizontalAlignment(HUD::HorizontalAlignment::CENTER)
                                   .setVerticalAlignment(HUD::VerticalAlignment::CENTER));

            hud.add(text2, HUD::Options()
                                   .setNormalizedPosition({0.5, 0.4})
                                   .setHorizontalAlignment(HUD::HorizontalAlignment::CENTER)
                                   .setVerticalAlignment(HUD::VerticalAlignment::CENTER));

            camera.position.z = 1;
        }

        Image fetchTile(int zoom, int x, int y) override {

            Color oldColor;
            renderer->getClearColor(oldColor);
            renderer->setClearColor(Color(Color::green).lerpHSL(Color::red, static_cast<float>(zoom - minZoom) / static_cast<float>(maxZoom - minZoom)));

            auto oldSize = renderer->getSize();
            renderer->setSize({resolution, resolution});

            std::vector<unsigned char> data(resolution * resolution * 4);

            text1->setText("(" + std::to_string(zoom) + ")");
            hud.needsUpdate(*text1);

            text2->setText("(" + std::to_string(x) + ", " + std::to_string(y) + ")");
            hud.needsUpdate(*text2);

            hud.apply(*renderer);
            renderer->readPixels({}, renderer->getSize(), Format::RGBA, data.data());
            renderer->setSize(oldSize);
            renderer->setClearColor(oldColor);
            renderer->clear();

            return Image(data, resolution, resolution);
        }

    private:
        int resolution = 256;

        OrthographicCamera camera;
        GLRenderer* renderer;
        HUD hud;
        FontLoader loader;
        std::shared_ptr<Text2D> text1, text2;
    };

}// namespace threepp

#endif//THREEPP_DEBUGMAPSPROVIDER_HPP
