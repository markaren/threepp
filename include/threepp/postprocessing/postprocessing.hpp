// Convenience umbrella for the OpenGL post-processing chain.
//
// Not pulled in by threepp.hpp: like the three.js addons these classes sit
// beside the renderer rather than inside it, and only the GL backend has them
// (the Vulkan renderer is swapchain-only and owns its own post chain).

#ifndef THREEPP_POSTPROCESSING_HPP
#define THREEPP_POSTPROCESSING_HPP

#include "threepp/postprocessing/BokehPass.hpp"
#include "threepp/postprocessing/ClearPass.hpp"
#include "threepp/postprocessing/EffectComposer.hpp"
#include "threepp/postprocessing/MaskPass.hpp"
#include "threepp/postprocessing/Pass.hpp"
#include "threepp/postprocessing/RenderPass.hpp"
#include "threepp/postprocessing/SavePass.hpp"
#include "threepp/postprocessing/ShaderPass.hpp"
#include "threepp/postprocessing/TexturePass.hpp"
#include "threepp/postprocessing/UnrealBloomPass.hpp"

#include "threepp/postprocessing/shaders/BokehShader.hpp"
#include "threepp/postprocessing/shaders/CopyShader.hpp"
#include "threepp/postprocessing/shaders/LuminosityHighPassShader.hpp"

#endif//THREEPP_POSTPROCESSING_HPP
