#ifndef THREEPP_SHADERPASS_HPP
#define THREEPP_SHADERPASS_HPP

#include <memory>
#include <string>

namespace threepp {

    class EffectComposer;

    /// A single full-screen shader pass for post-processing.
    /// Stores WGSL source; GPU pipeline is lazily created by EffectComposer.
    class ShaderPass {

    public:
        static std::shared_ptr<ShaderPass> create(const std::string& wgslSource);

        ~ShaderPass();

        ShaderPass(const ShaderPass&) = delete;
        ShaderPass& operator=(const ShaderPass&) = delete;

    private:
        explicit ShaderPass(const std::string& wgslSource);

        struct Impl;
        std::unique_ptr<Impl> pimpl_;

        friend class EffectComposer;
    };

}// namespace threepp

#endif//THREEPP_SHADERPASS_HPP
