// A 3D Gaussian Splat cloud, rendered by the GL backend.
//
// Provenance: clean-room. The EWA splatting math is from Zwicker et al. 2001 as
// specialised for 3DGS in arXiv 2308.04079; the SH basis is threepp/splats/SplatSH.hpp.
// No third-party splatting code was consulted.
//
// HOW IT RIDES THE EXISTING RENDERER
// ----------------------------------
// It is an InstancedMesh over a single unit quad, so the GL backend already
// knows how to draw it: is<Mesh>() puts it in the render list, as<InstancedMesh>()
// turns the draw into glDrawElementsInstanced, and nothing in the renderer had
// to change.
//
// Per-splat data does NOT ride in vertex attributes. It lives in three RGBA32F
// DataTextures fetched with texelFetch by instance index — the same trick
// Skeleton uses for boneTexture, and the reason a million splats need no
// per-instance attribute plumbing:
//
//   splatMeanTex   1 texel  / splat   (mean.xyz, opacity)
//   splatCovTex    2 texels / splat   (Sxx,Sxy,Sxz,Syy), (Syz,Szz,-,-)
//   splatShTex     N texels / splat   one texel per SH coefficient, rgb used
//
// splatShTex spends its alpha channel to keep a coefficient from straddling a
// texel boundary; at degree 3 that is 16 texels per splat instead of 12. The
// shader stays a single texelFetch per coefficient, which is worth 25% of the
// SH memory at prototype scale.
//
// THE ONLY PER-FRAME TRAFFIC is the draw order. Splats must be blended
// back-to-front, so update() counting-sorts the splat indices by view depth and
// writes the sorted index into instanceColor.x (floats are exact up to 2^24, so
// the index survives the round trip). The shader reads that one float and
// fetches everything else by it. instanceMatrix is deliberately unused and stays
// identity — the renderer uploads it once regardless, 64 bytes per splat of
// dead VRAM, which is the price of not editing the renderer.
//
// CALL update(camera) BEFORE Renderer::render(). If you don't, the object's own
// onBeforeRender hook does it, but the renderer has already uploaded
// instanceColor for this frame by then, so the sort lands one frame late — fine
// in an animation loop, wrong for a single-frame capture or a pixel test.
//
// GL 3.3 core only for now: the material is a RawShaderMaterial carrying its own
// "#version 330 core", which is not what a WebGL2 / Emscripten build needs.

#ifndef THREEPP_SPLATCLOUD_HPP
#define THREEPP_SPLATCLOUD_HPP

#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/splats/SplatData.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace threepp {

    class Camera;
    class DataTexture;
    class RawShaderMaterial;

    class SplatCloud: public InstancedMesh {

    public:
        explicit SplatCloud(SplatData data);

        [[nodiscard]] static std::shared_ptr<SplatCloud> create(SplatData data);

        [[nodiscard]] const SplatData& data() const { return data_; }

        [[nodiscard]] size_t splatCount() const { return data_.count(); }

        // Sorts back-to-front for this camera and refreshes the per-frame
        // uniforms. Cheap to call redundantly: it early-outs when neither the
        // camera nor the cloud has moved since the last sort.
        //
        // One order at a time. Drawing the same cloud from two cameras in one
        // frame re-sorts for each, but there is a single instanceColor buffer,
        // so both views end up drawn in whichever order was uploaded last. A
        // split-screen or multi-view setup needs one SplatCloud per view until
        // the ordering moves onto the GPU.
        void update(Camera& camera);

        // Pixel size of the framebuffer being drawn into. Set automatically
        // from the renderer each frame; override it when rendering into a
        // render target whose size differs from the renderer's own.
        void setViewportSize(int width, int height);

        // Draw non-finite fragments as magenta instead of discarding them.
        // Off by default. A splat cloud that renders "fine" while quietly
        // producing NaNs looks identical to one that doesn't, which is exactly
        // the bug this exists to make visible.
        //
        // Covers the colour path, where a corrupt SH coefficient would
        // otherwise be scrubbed into a plausible-looking colour by the clamp.
        // A non-finite *geometry* input never reaches a fragment at all: the
        // vertex stage collapses the quad off-screen, in debug mode or out of
        // it, because there is no meaningful position to paint magenta at.
        void setDebugNonFinite(bool flag);

        [[nodiscard]] std::string type() const override { return "SplatCloud"; }

        // The GLSL, exposed so tests can assert the shader and the C++ SH table
        // still share their constants.
        [[nodiscard]] static const std::string& vertexShaderSource();
        [[nodiscard]] static const std::string& fragmentShaderSource();

    private:
        SplatData data_;

        std::shared_ptr<DataTexture> meanTexture_;
        std::shared_ptr<DataTexture> covTexture_;
        std::shared_ptr<DataTexture> shTexture_;
        std::shared_ptr<RawShaderMaterial> splatMaterial_;

        // Counting-sort scratch, allocated once. 16-bit key, so 65536 buckets.
        std::vector<float> depths_;
        std::vector<std::uint16_t> keys_;
        std::vector<std::uint32_t> histogram_;

        // The modelView the last sort used, so a redundant update() — the
        // common case, since onBeforeRender calls it too — costs 16 compares.
        std::array<float, 16> lastSortMatrix_{};
        bool sorted_{false};

        void buildTextures();
        void sortByDepth(Camera& camera);
    };

}// namespace threepp

#endif//THREEPP_SPLATCLOUD_HPP
