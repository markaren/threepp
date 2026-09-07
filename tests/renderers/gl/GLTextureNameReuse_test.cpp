// Regression tests for the GLState bind-cache name-reuse hazard.
//
// GLState::bindTexture caches {type, name} per texture unit and skips the
// glBindTexture when they match. glDeleteTextures frees the name for reuse,
// and drivers may hand freed names back out of glGenTextures — so a texture
// created after a dispose can inherit the deleted texture's name. If the
// cache entry is not purged on delete (GLState::purgeTexture), the new
// texture's first bind is skipped and the unit samples texture 0 (black).
//
// three.js never needs this: a WebGLTexture is an object identity that is
// never reused after deletion, so its otherwise-identical cache cannot
// false-hit. GLuint names are recycled, so ours can.
//
// Name reuse is driver-dependent (Mesa hands back the lowest free name,
// NVIDIA defers), so the first test does not rely on it: it inspects the
// bind cache directly and asserts the dispose path evicted the entry. The
// second, behavioral test renders through a dispose/recreate cycle — it only
// exercises the skipped-bind path on drivers that do recycle promptly (the
// Mesa/llvmpipe CI runners), but it is cheap and asserts end-to-end pixels.

#include "gl_test_helpers.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <memory>
#include <vector>

namespace {

    std::shared_ptr<DataTexture> makeSolidTexture(unsigned char r, unsigned char g, unsigned char b) {
        constexpr unsigned int size = 4;
        std::vector<unsigned char> data(size * size * 4);
        for (unsigned int i = 0; i < size * size; i++) {
            data[i * 4 + 0] = r;
            data[i * 4 + 1] = g;
            data[i * 4 + 2] = b;
            data[i * 4 + 3] = 255;
        }
        return DataTexture::create(data, size, size);
    }

    // Entries recording a real texture name. Binds of the internal empty
    // textures pass nullopt for the name, so only user textures count here.
    int namedCacheEntries(gl::GLState& state) {
        int count = 0;
        for (const auto& [slot, bound] : state.currentBoundTextures) {
            if (bound.texture) count++;
        }
        return count;
    }

}// namespace

TEST_CASE("GL: disposing a texture evicts its name from the bind cache") {
    GLRenderer renderer(glCanvas());
    renderer.setClearColor(Color(0.0f, 0.0f, 1.0f));

    auto scene = Scene::create();
    auto camera = OrthographicCamera::create(-1, 1, 1, -1, 0.1f, 10);
    camera->position.z = 1;

    auto tex = makeSolidTexture(255, 0, 0);
    auto material = MeshBasicMaterial::create();
    material->map = tex;
    auto mesh = Mesh::create(PlaneGeometry::create(2, 2), material);
    scene->add(mesh);

    renderer.render(*scene, *camera);
    auto pixels = renderer.readRGBPixels();
    REQUIRE(pixels.size() == DATA_SIZE);
    CHECK(allPixelsMatch(pixels, 255, 0, 0, 2));

    // The one real texture in the scene was bound, so exactly its name is in
    // the cache. Dispose must purge it — a stale entry here is the recycled-
    // name trap armed.
    REQUIRE(namedCacheEntries(renderer.state()) >= 1);

    scene->remove(*mesh);
    tex->dispose();

    CHECK(namedCacheEntries(renderer.state()) == 0);

    renderer.dispose();
}

TEST_CASE("GL: texture created after a dispose is bound, not skipped by the bind cache") {
    GLRenderer renderer(glCanvas());
    renderer.setClearColor(Color(0.0f, 0.0f, 1.0f));

    auto scene = Scene::create();
    auto camera = OrthographicCamera::create(-1, 1, 1, -1, 0.1f, 10);
    camera->position.z = 1;

    auto geometry = PlaneGeometry::create(2, 2);

    auto red = makeSolidTexture(255, 0, 0);
    auto matA = MeshBasicMaterial::create();
    matA->map = red;
    auto meshA = Mesh::create(geometry, matA);
    scene->add(meshA);

    renderer.render(*scene, *camera);
    auto pixels = renderer.readRGBPixels();
    REQUIRE(pixels.size() == DATA_SIZE);
    CHECK(allPixelsMatch(pixels, 255, 0, 0, 2));

    // Free the GL name; on drivers that recycle promptly the next texture
    // inherits it, and only a purged cache lets its first bind through.
    scene->remove(*meshA);
    red->dispose();

    auto green = makeSolidTexture(0, 255, 0);
    auto matB = MeshBasicMaterial::create();
    matB->map = green;
    auto meshB = Mesh::create(geometry, matB);
    scene->add(meshB);

    renderer.render(*scene, *camera);
    pixels = renderer.readRGBPixels();
    REQUIRE(pixels.size() == DATA_SIZE);
    CHECK(allPixelsMatch(pixels, 0, 255, 0, 2));

    renderer.dispose();
}
