// A 3D Gaussian Splat cloud, rendered by the GL backend.
//
// Provenance: clean-room. The EWA splatting math is from Zwicker et al. 2001 as
// specialised for 3DGS in arXiv 2308.04079; the SH basis is threepp/splats/SplatSH.hpp.
// No third-party splatting code was consulted.
//
// HOW IT RIDES THE EXISTING RENDERER
// ----------------------------------
// It is a Mesh whose geometry is an InstancedBufferGeometry over a single unit
// quad: is<Mesh>() puts it in the render list, the geometry's instanceCount
// turns the draw into glDrawElementsInstanced, and the one piece of per-instance
// data — the sorted splat index — rides as an InstancedBufferAttribute the
// shader declares by name.
//
// It WAS an InstancedMesh, which needed no renderer change at all, and that is
// why it started there. What it cost was the 16-float instanceMatrix that class
// allocates per instance and this one never writes anything but identity into:
// 64 bytes a splat of host memory AND 64 of VRAM (GLObjects uploads it for every
// InstancedMesh in the render list, whether or not the program declares the
// attribute — and this program does not), 768 MB of pure identity across both
// at 6M splats. Deriving from Mesh instead recovers all of it, and the sorted
// index shrank from a vec3 instanceColor to a single float on the way, since
// only .x was ever read.
//
// Per-splat data does NOT ride in vertex attributes. It lives in three
// DataTextures fetched with texelFetch by instance index — the same trick
// Skeleton uses for boneTexture, and the reason a million splats need no
// per-instance attribute plumbing:
//
//   splatMeanTex   RGBA32F  1 texel  / splat   (mean.xyz, opacity)
//   splatCovTex    RGBA32F  2 texels / splat   (Sxx,Sxy,Sxz,Syy), (Syz,Szz,-,-)
//   splatShTex     RGBA16F  N texels / splat   one texel per SH coefficient, rgb used
//
// splatShTex is HALF and the other two are not. SH coefficients are small,
// smooth, and get summed against basis functions, so half's 11-bit significand
// is far inside the error budget of an 8-bit colour — and SH is where the
// memory is: at degree 3 it is 16 of the 19 texels a splat occupies. The mean
// and covariance stay fp32 because the shader inverts the projected
// covariance, and a near-singular conic is how a splat renderer turns a
// rounding error into a screen-wide smear.
//
// splatShTex spends its alpha channel to keep a coefficient from straddling a
// texel boundary; at degree 3 that is 16 texels per splat instead of 12. The
// shader stays a single texelFetch per coefficient, which is worth 25% of the
// SH memory at prototype scale.
//
// THE ONLY PER-FRAME TRAFFIC is the draw order. Splats must be blended
// back-to-front, so update() counting-sorts the splat indices by view depth and
// writes the sorted index into the `splatIndex` attribute (a float, exact up to
// 2^24, so the index survives the round trip). The 16-bit key is quantised over
// a PERCENTILE interval of the view depths rather than their full extent — a
// scan's strays are a thousand units outside a twenty-unit subject, and
// spreading 65536 buckets over that leaves the subject with a few hundred.
// The shader reads that one float and fetches everything else by it.
//
// CALL update(camera) BEFORE Renderer::render(). If you don't, the object's own
// onBeforeRender hook does it, but the renderer has already uploaded
// splatIndex for this frame by then, so the sort lands one frame late — fine
// in an animation loop, wrong for a single-frame capture or a pixel test.
//
// GL 3.3 core only for now: the material is a RawShaderMaterial carrying its own
// "#version 330 core", which is not what a WebGL2 / Emscripten build needs.

#ifndef THREEPP_SPLATCLOUD_HPP
#define THREEPP_SPLATCLOUD_HPP

#include "threepp/core/InstancedBufferGeometry.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/splats/SplatData.hpp"
#include "threepp/splats/SplatLod.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace threepp {

    class Camera;
    class DataTexture;
    class RawShaderMaterial;

    class SplatCloud: public Mesh {

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
        // frame re-sorts for each, but there is a single splatIndex buffer,
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

        // Readable so a backend that does not consume the GL material's
        // uniforms (the Vulkan compute rasterizer) can honour the same switch.
        [[nodiscard]] bool debugNonFinite() const { return debugNonFinite_; }

        // ── Point rendering ─────────────────────────────────────────────────
        // 0 (the default) draws every splat as its Gaussian. 1 draws it as an
        // opaque disc of pointSize() pixels centred on its mean — the point
        // cloud view — composited in the same depth order, so the nearest
        // point wins where two overlap. Values between blend the two: the
        // projected covariance is lerped toward the disc's and the opacity
        // toward 1, which makes a 0 -> 1 sweep a continuous dissolve from
        // surface to dots.
        //
        // Both backends read it, and everything else about the pass is
        // unchanged: the sort, the depth test against scene geometry, the
        // overlay depth stamp, the depth AOV. A point cloud is therefore
        // occluded by a mesh and occludes a gizmo exactly as the Gaussians
        // are. At mix 0 the shaders take their pre-existing path and the
        // frame is bit-identical to one rendered before this existed.
        void setPointMix(float mix);
        [[nodiscard]] float pointMix() const { return pointMix_; }

        // Disc diameter in pixels at mix 1 — PointsMaterial::size's
        // convention. Floored at 1. Default 2.
        void setPointSize(float pixels);
        [[nodiscard]] float pointSize() const { return pointSize_; }

        // The standard deviation, in pixels, of the isotropic footprint the
        // mix lerps toward. The disc edge sits at 3 sigma and carries a
        // one-pixel feather centred on the requested radius, so
        // sigma = (size / 2 + 0.5) / 3. Both backends build the disc from
        // this one number; the Vulkan collector reads it from here.
        [[nodiscard]] float pointSigmaPixels() const;

        // Whether the GL-side data textures exist yet — the assertable form of
        // "a Vulkan-only cloud never pays the GL copy" (see ensureGlResources).
        [[nodiscard]] bool glResourcesBuilt() const { return glResourcesBuilt_; }

        // Host memory this cloud holds, in bytes: the splat data, the sorted index
        // and sort scratch, and — only once a GL frame has built them — the data
        // textures. GPU-side residency is the backend's own and not counted here.
        //
        // Exists because something has to be able to ask. A scan is three orders of
        // magnitude heavier than any mesh in the same scene, so anything that holds
        // clouds and has a budget (the editor's undo history is the first) has to
        // weigh them rather than count them.
        //
        // Measured per splat at SH degree 3, which is 423 B in total:
        //
        //   236  splat data (see SplatData::byteSize)
        //    11  sorted index and counting-sort scratch
        //   176  the data textures, and only after a GL frame
        //
        // So 1.4 GiB for a 6M-splat scan on Vulkan, 2.4 GiB once GL has drawn it.
        // It was 606 B a splat, measured the same way. 112 of the difference is
        // the rotation, which used to be a threepp::Quaternion carrying change-
        // notification plumbing; the other 72 is the identity instanceMatrix plus
        // the two unread components of a vec3 index, both of which left with
        // InstancedMesh.
        [[nodiscard]] std::size_t cpuBytes() const;

        // ── Partial submission: which splats this frame actually draws ────────
        // A list of (offset, count) into this cloud's own splats. Empty — the
        // default — draws all of them.
        //
        // This is the mechanism per-chunk LOD is built on, and the reason it is
        // a range list rather than anything else is measured: a second
        // SplatCloud costs ~1.3 ms because the Vulkan pass runs end to end per
        // cloud, and re-packing a merged buffer per selection change is a
        // re-upload of up to 1.2 GB. Keep every chunk at every detail level in
        // ONE cloud, uploaded once, and let the frame pick ranges. A chunk
        // outside the frustum is a chunk left out of the list, which is all
        // chunk culling ever needs to be.
        //
        // Order matters and is preserved: the ranges are drawn in the order
        // given, so a list covering the whole cloud in ascending order is
        // exactly equivalent to an empty one. Offsets past the end are dropped
        // and counts are clamped by the backend; at most 64 ranges are honoured.
        //
        // VULKAN ONLY today — the GL path draws every splat regardless. Setting
        // this changes nothing there rather than silently disagreeing between
        // backends, which is why the example prints which backend is running.
        void setSubmitRanges(std::vector<std::pair<uint32_t, uint32_t>> ranges) {

            submitRanges_ = std::move(ranges);
        }
        [[nodiscard]] const std::vector<std::pair<uint32_t, uint32_t>>& submitRanges() const {

            return submitRanges_;
        }

        // The multi-level table splats::selectLod drives the ranges from, when
        // this cloud was loaded with several detail levels resident (see
        // splats::loadSogWithLod). ON the cloud rather than in a side map: the
        // table is cloud data, it travels with the object across scenes and
        // threads (imports build it on a worker), and nothing has to remember
        // to clean up an entry when the cloud dies — today's residency-cache
        // lesson, applied at the layer above.
        void setLodTable(splats::LodTable table) { lodTable_ = std::move(table); }
        [[nodiscard]] splats::LodTable& lodTable() { return lodTable_; }
        [[nodiscard]] const splats::LodTable& lodTable() const { return lodTable_; }

        // Ray against the cloud's own 3-sigma bounding sphere, and nothing
        // finer. What it replaces is the reason it exists: the inherited raycast
        // tests the geometry, which is one unit quad at the origin — so every
        // ray either hits those two triangles or misses the whole scan. That is
        // both wrong and useless, and it is what stands between a splat cloud
        // and being clickable in an editor viewport. (Under InstancedMesh it was
        // wrong AND slow: the same two triangles, once per instance.)
        //
        // One intersection at most, no instanceId: which splat was hit is a
        // later refinement (nearest mean along the ray), and picking the whole
        // cloud is what selection actually wants.
        void raycast(const Raycaster& raycaster, std::vector<Intersection>& intersects) override;

        [[nodiscard]] std::string type() const override { return "SplatCloud"; }

        // The GLSL, exposed so tests can assert the shader and the C++ SH table
        // still share their constants.
        [[nodiscard]] static const std::string& vertexShaderSource();
        [[nodiscard]] static const std::string& fragmentShaderSource();

    private:
        // The geometry this class always owns, and the one per-instance
        // attribute on it. Accessors rather than cached pointers: nothing here
        // is hot enough to care, and a cached pointer would outlive a geometry
        // someone swapped out.
        [[nodiscard]] InstancedBufferGeometry* splatGeometry() const {

            return static_cast<InstancedBufferGeometry*>(geometry_.get());
        }

        [[nodiscard]] InstancedBufferAttribute* splatIndexAttribute() const {

            return static_cast<InstancedBufferAttribute*>(geometry_->getAttribute("splatIndex"));
        }

        SplatData data_;

        std::shared_ptr<DataTexture> meanTexture_;
        std::shared_ptr<DataTexture> covTexture_;
        std::shared_ptr<DataTexture> shTexture_;
        std::shared_ptr<RawShaderMaterial> splatMaterial_;

        // Counting-sort scratch, allocated once. 16-bit key, so 65536 buckets.
        // sample_ is the fixed-stride subset of depths_ the key range is taken
        // from; see the clamp constants in SplatCloud.cpp.
        std::vector<float> depths_;
        std::vector<float> sample_;
        std::vector<std::uint16_t> keys_;
        std::vector<std::uint32_t> histogram_;

        // The modelView the last sort used, so a redundant update() — the
        // common case, since onBeforeRender calls it too — costs 16 compares.
        std::array<float, 16> lastSortMatrix_{};
        bool sorted_{false};
        bool debugNonFinite_{false};
        bool glResourcesBuilt_{false};
        float pointMix_{0.f};
        float pointSize_{2.f};
        std::vector<std::pair<uint32_t, uint32_t>> submitRanges_;
        splats::LodTable lodTable_;

        // The data textures, built on first GL use (update(), which the object's
        // own onBeforeRender calls) and never on a Vulkan backend. ~1 GB at 6M
        // splats, and the editor's undo history retains it per held copy, which
        // is why lazy matters twice.
        //
        // The sorted-index attribute is NOT here: an attribute created during
        // onBeforeRender arrives after the renderer has already decided what to
        // upload, so it is allocated in the constructor. The constructor says why
        // at length; it cost an editor crash to learn.
        void ensureGlResources();
        void buildTextures();
        void sortByDepth(Camera& camera);
    };

}// namespace threepp

#endif//THREEPP_SPLATCLOUD_HPP
