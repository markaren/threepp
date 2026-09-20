
#ifndef THREEPP_GLCUBEMAPS_HPP
#define THREEPP_GLCUBEMAPS_HPP

#include "threepp/renderers/GLCubeRenderTarget.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/renderers/common/EnvSunExtract.hpp"
#include "threepp/textures/CubeTexture.hpp"

#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace threepp {

    namespace gl {

        class GLPMREM;

        class GLCubeMaps {

        public:
            explicit GLCubeMaps(GLRenderer& renderer);
            ~GLCubeMaps();

            // Returns the converted CubeTexture for equirectangular inputs,
            // the original texture for anything else, or nullptr if not ready.
            // Mirrors three.js WebGLCubeMaps.get().
            Texture* get(Texture* texture);

            // Returns a PMREM (CubeUV-packed 2D) texture for equirectangular
            // inputs used as IBL / envMap. Falls back to the original texture
            // for non-equirect inputs (pre-baked cube PMREMs etc.).
            Texture* getPMREM(Texture* texture);

            // The sun measured out of `texture` when its PMREM was built, or
            // nullptr when the PMREM is not built yet, extraction was off, or no
            // disc was found. GLRenderer re-injects it as an analytic light.
            [[nodiscard]] const EnvSunExtract* envSun(Texture* texture) const;

            // False keeps the raw sun in every strip (EnvSunPolicy::Off).
            // GLRenderer drops the cached PMREMs when this flips, so the next
            // frame rebuilds them from the right source.
            bool envSunExtraction = true;

            // Erases only the PMREM atlases; the background cubemaps stay.
            void disposePMREMs();

            void dispose();

        private:
            // The atlas plus what the detector found in the source it was built
            // from — one lookup, so a frame never re-runs the CPU detector.
            struct PmremEntry {
                std::unique_ptr<RenderTarget> target;
                EnvSunExtract sun;
            };

            // Both maps are keyed on a RAW Texture*, so an entry has to go when
            // its source texture does. Otherwise the cube render target and the
            // PMREM atlas built from it are leaked, and — the part that is not
            // merely wasteful — a later Texture allocated at the same address
            // finds the dead entry and silently renders the PREVIOUS environment.
            // r129 keys on a WeakMap and still bothers to listen for dispose
            // (WebGLCubeMaps.js:51, 73); with raw pointers the listener is the
            // only thing standing between this and a use-after-free.
            struct SourceTextureEventListener: EventListener {

                explicit SourceTextureEventListener(GLCubeMaps* scope): scope_(scope) {}

                void onEvent(Event& event) override;

            private:
                GLCubeMaps* scope_;
            };

            // Subscribes once per source texture, however many of the two caches
            // it ends up in.
            void watch(Texture* texture);
            void unwatchAll();
            void forget(Texture* texture);

            GLRenderer& renderer;
            std::unordered_map<Texture*, std::unique_ptr<GLCubeRenderTarget>> cubemaps;
            std::unordered_map<Texture*, PmremEntry> pmrems;
            std::unordered_set<Texture*> watched_;
            SourceTextureEventListener onSourceDispose_;
            std::unique_ptr<GLPMREM> pmremGenerator;
        };

    }// namespace gl

}// namespace threepp

#endif//THREEPP_GLCUBEMAPS_HPP
