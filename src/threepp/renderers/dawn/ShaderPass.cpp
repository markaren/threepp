
#include "ShaderPassImpl.hpp"

using namespace threepp;

ShaderPass::ShaderPass(const std::string& wgslSource)
    : pimpl_(std::make_unique<Impl>()) {
    pimpl_->wgslSource = wgslSource;
}

ShaderPass::~ShaderPass() = default;

std::shared_ptr<ShaderPass> ShaderPass::create(const std::string& wgslSource) {
    return std::shared_ptr<ShaderPass>(new ShaderPass(wgslSource));
}
