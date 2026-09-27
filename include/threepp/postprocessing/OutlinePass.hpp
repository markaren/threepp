// https://github.com/mrdoob/three.js/blob/r186/examples/jsm/postprocessing/OutlinePass.js

#ifndef THREEPP_POSTPROCESSING_OUTLINEPASS_HPP
#define THREEPP_POSTPROCESSING_OUTLINEPASS_HPP

#include "threepp/math/Color.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/postprocessing/Pass.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace threepp {

    class Camera;
    class MeshDepthMaterial;
    class Object3D;
    class RenderTarget;
    class Scene;
    class ShaderMaterial;
    class Texture;

    // Draws a glowing outline around the selected objects - the "this is the
    // one you clicked" highlight of an editor.
    //
    // How it works, as in three.js: the scene without the selection is
    // rendered into a depth target, then only the selection is rendered into a
    // mask, each fragment comparing itself against that depth to learn whether
    // something is in front of it. An edge detect on the (downsampled) mask
    // finds the silhouette and picks a colour per pixel - visibleEdgeColor
    // where the selection is in plain view, hiddenEdgeColor where it is behind
    // something - and two blurs widen and soften it before it is added onto
    // the image.
    //
    // The pass renders the scene twice more per frame. It hides objects by
    // flipping Object3D::visible and swaps Scene::overrideMaterial and
    // Scene::background while it does, and puts every one of them back before
    // it returns. Only meshes and sprites take part; points and lines are
    // hidden during its renders so they neither outline nor occlude.
    //
    // Skinned and instanced meshes outline in their posed / instanced shape:
    // the mask shader runs the skinning and instancing chunks, like three.js.
    // Morph targets do not: threepp enables them per material, and the pass's
    // own material has no morph flag, so a morphed mesh outlines its base
    // shape.
    //
    // The composer's targets are linear, so the edge colours are linear-light
    // values added to the image; a pure white edge at edgeStrength 3 is 3.0
    // before tone mapping.
    class OutlinePass: public Pass {

    public:
        // Objects to outline. A group outlines every mesh under it. Not owned:
        // an object must outlive its place in this list.
        std::vector<Object3D*> selectedObjects;

        // Edge colour where the selection is in plain view.
        Color visibleEdgeColor{1.f, 1.f, 1.f};

        // Edge colour where something in front hides the selection.
        Color hiddenEdgeColor{0.1f, 0.04f, 0.02f};

        // Adds a wider, softer copy of the edge (the quarter-resolution blur).
        // 0 is a crisp line.
        float edgeGlow = 0.f;

        // Fill the selection with patternTexture (tiled 6x across the screen).
        bool usePatternTexture = false;
        std::shared_ptr<Texture> patternTexture;

        // Width of the edge blur, in downsampled pixels. Up to 4.
        float edgeThickness = 1.f;

        // Multiplies the edge before it is added.
        float edgeStrength = 3.f;

        // The edge is found and blurred at 1/downSampleRatio resolution. Takes
        // effect on the next setSize.
        float downSampleRatio = 2.f;

        // Seconds-ish period of a brightness pulse; 0 disables it. Same time
        // base as three.js (cos(ms * 0.01 / pulsePeriod)).
        float pulsePeriod = 0.f;

        Vector2 resolution;

        OutlinePass(const Vector2& resolution,
                    Scene& scene,
                    Camera& camera,
                    std::vector<Object3D*> selectedObjects = {});

        void setSize(unsigned int width, unsigned int height) override;

        void render(GLRenderer& renderer,
                    RenderTarget* writeBuffer,
                    RenderTarget* readBuffer,
                    float deltaTime,
                    bool maskActive) override;

        ~OutlinePass() override;

    private:
        Scene* renderScene_;
        Camera* renderCamera_;

        std::unordered_map<Object3D*, bool> visibilityCache_;
        std::unordered_set<Object3D*> selectionCache_;

        std::unique_ptr<RenderTarget> renderTargetMaskBuffer_;
        std::unique_ptr<RenderTarget> renderTargetDepthBuffer_;
        std::unique_ptr<RenderTarget> renderTargetMaskDownSampleBuffer_;
        std::unique_ptr<RenderTarget> renderTargetBlurBuffer1_;
        std::unique_ptr<RenderTarget> renderTargetBlurBuffer2_;
        std::unique_ptr<RenderTarget> renderTargetEdgeBuffer1_;
        std::unique_ptr<RenderTarget> renderTargetEdgeBuffer2_;

        std::shared_ptr<MeshDepthMaterial> depthMaterial_;
        std::shared_ptr<ShaderMaterial> prepareMaskMaterial_;
        std::shared_ptr<ShaderMaterial> edgeDetectionMaterial_;
        std::shared_ptr<ShaderMaterial> separableBlurMaterial1_;
        std::shared_ptr<ShaderMaterial> separableBlurMaterial2_;
        std::shared_ptr<ShaderMaterial> overlayMaterial_;
        std::shared_ptr<ShaderMaterial> materialCopy_;

        std::shared_ptr<Texture> placeholderTexture_;

        std::unique_ptr<FullScreenQuad> fsQuad_;

        Matrix4 textureMatrix_;
        bool perspective_;

        void updateSelectionCache();
        void changeVisibilityOfSelectedObjects(bool visible);
        void changeVisibilityOfNonSelectedObjects(bool visible);
        void updateTextureMatrix();
    };

}// namespace threepp

#endif//THREEPP_POSTPROCESSING_OUTLINEPASS_HPP
