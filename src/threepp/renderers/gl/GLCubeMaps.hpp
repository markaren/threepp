
#ifndef THREEPP_GLCUBEMAPS_HPP
#define THREEPP_GLCUBEMAPS_HPP

#include "threepp/renderers/GLCubeRenderTarget.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/renderers/common/EnvSunExtract.hpp"
#include "threepp/textures/CubeTexture.hpp"

#include <memory>
#include <unordered_map>

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

            GLRenderer& renderer;
            std::unordered_map<Texture*, std::unique_ptr<GLCubeRenderTarget>> cubemaps;
            std::unordered_map<Texture*, PmremEntry> pmrems;
            std::unique_ptr<GLPMREM> pmremGenerator;
        };

    }// namespace gl

}// namespace threepp

#endif//THREEPP_GLCUBEMAPS_HPP
