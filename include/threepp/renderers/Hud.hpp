
#ifndef THREEPP_HUD_HPP
#define THREEPP_HUD_HPP

#include <memory>
#include <string>

namespace threepp {

    class GLRenderer;

    class HUD {

    public:
        HUD(GLRenderer& renderer);

        void addText(const std::string& str);

        ~HUD();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_HUD_HPP
