// https://github.com/mrdoob/three.js/blob/r186/src/extras/PMREMGenerator.js (fromScene)

#ifndef THREEPP_PMREMGENERATOR_HPP
#define THREEPP_PMREMGENERATOR_HPP

#include "threepp/math/Vector3.hpp"

#include <memory>

namespace threepp {

    class GLRenderer;
    class Scene;
    class Texture;

    // three.js r186 #30477 fromScene() options.
    struct PMREMFromSceneOptions {
        // Edge of each face of the cube the scene is rendered into. The
        // environment it produces is 4*size x 2*size.
        int size = 256;
        // Where the cube camera sits.
        Vector3 position;
    };

    // Turns a Scene into an environment map, for image-based lighting without
    // an HDR file:
    //
    //   PMREMGenerator pmrem(renderer);
    //   RoomEnvironment room;
    //   scene->environment = pmrem.fromScene(room, 0.04f);
    //
    // GLRenderer only. Unlike three.js, fromScene returns the environment as an
    // equirectangular HalfFloat texture rather than a finished PMREM render
    // target: the renderer prefilters it into its own PMREM (GLPMREM's equirect
    // strip atlas) the first time a material uses it, exactly like an HDR file
    // loaded with RGBELoader, and it works as scene.background too. The texture
    // lives on the GPU only and belongs to the renderer that made it; it stays
    // valid for as long as anything holds the returned pointer, independent of
    // this generator.
    class PMREMGenerator {

    public:
        using Options = PMREMFromSceneOptions;

        explicit PMREMGenerator(GLRenderer& renderer);

        PMREMGenerator(const PMREMGenerator&) = delete;
        PMREMGenerator& operator=(const PMREMGenerator&) = delete;

        ~PMREMGenerator();

        // Renders `scene` into a cube from options.position (near/far planes as
        // given, tone mapping off) and returns it as an equirectangular
        // environment. `sigma` is a Gaussian blur radius in radians, applied
        // before prefiltering (three.js uses 0.04 for RoomEnvironment).
        std::shared_ptr<Texture> fromScene(Scene& scene, float sigma = 0, float near = 0.1f, float far = 100,
                                           const Options& options = Options());

    private:
        GLRenderer& renderer_;

        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

}// namespace threepp

#endif//THREEPP_PMREMGENERATOR_HPP
