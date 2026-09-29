// MinimapView.hpp: the same game again, flat, one pixel per cell, in a corner of the window.

#ifndef SNAKE_MINIMAPVIEW_HPP
#define SNAKE_MINIMAPVIEW_HPP

#include "model/Game.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/canvas/WindowSize.hpp"
#include "threepp/renderers/Renderer.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <memory>

namespace snake {

    class MinimapView {
    public:
        explicit MinimapView(const Game& game);

        void update(const Game& game);

        // draws into the top-right corner, on top of whatever is there
        void render(threepp::Renderer& renderer, threepp::WindowSize window);

    private:
        threepp::Scene scene_;
        threepp::OrthographicCamera camera_;
        std::shared_ptr<threepp::DataTexture> texture_;
        int width_;
        int height_;
    };

}// namespace snake

#endif//SNAKE_MINIMAPVIEW_HPP
