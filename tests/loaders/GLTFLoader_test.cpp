#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/core/AttributeView.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace threepp;
namespace fs = std::filesystem;

namespace {

    // --- little-endian binary appenders -------------------------------------

    template<class T>
    void append(std::vector<uint8_t>& b, T v) {
        auto* p = reinterpret_cast<const uint8_t*>(&v);
        b.insert(b.end(), p, p + sizeof(T));
    }

    // A growable BIN blob that keeps every block 4-byte aligned (glTF requires
    // bufferView offsets aligned to the component size; 4 covers everything here).
    struct Bin {
        std::vector<uint8_t> data;
        template<class T>
        size_t put(std::initializer_list<T> vs) {
            while (data.size() % 4) data.push_back(0);
            size_t off = data.size();
            for (T v : vs) append(data, v);
            return off;
        }
        size_t putBytes(const std::vector<uint8_t>& bytes) {
            while (data.size() % 4) data.push_back(0);
            size_t off = data.size();
            data.insert(data.end(), bytes.begin(), bytes.end());
            return off;
        }
    };

    // Assemble a GLB from a JSON string + a BIN buffer (buffer 0).
    std::vector<uint8_t> makeGlb(std::string json, const std::vector<uint8_t>& bin) {
        auto pad4 = [](size_t n) { return (4 - (n % 4)) % 4; };
        json.append(pad4(json.size()), ' ');
        std::vector<uint8_t> b = bin;
        b.insert(b.end(), pad4(b.size()), 0);

        std::vector<uint8_t> out;
        auto push32 = [&](uint32_t v) { append(out, v); };
        const uint32_t jsonLen = static_cast<uint32_t>(json.size());
        const uint32_t binLen = static_cast<uint32_t>(b.size());
        const uint32_t total = 12 + 8 + jsonLen + (binLen ? 8 + binLen : 0);

        push32(0x46546C67);// "glTF"
        push32(2);
        push32(total);
        push32(jsonLen);
        push32(0x4E4F534A);// "JSON"
        out.insert(out.end(), json.begin(), json.end());
        if (binLen) {
            push32(binLen);
            push32(0x004E4942);// "BIN\0"
            out.insert(out.end(), b.begin(), b.end());
        }
        return out;
    }

    int g_counter = 0;
    fs::path writeTempGlb(const std::vector<uint8_t>& bytes) {
        auto path = fs::temp_directory_path() /
                    ("threepp_gltf_test_" + std::to_string(g_counter++) + ".glb");
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
        out.close();
        return path;
    }

    // Find the first Mesh in an object tree (depth-first).
    Mesh* firstMesh(Object3D* o) {
        if (auto* m = o->as<Mesh>()) return m;
        for (auto* c : o->children)
            if (auto* m = firstMesh(c)) return m;
        return nullptr;
    }

    void collectMeshes(Object3D* o, std::vector<Mesh*>& out) {
        if (auto* m = o->as<Mesh>()) out.push_back(m);
        for (auto* c : o->children) collectMeshes(c, out);
    }

    // A minimal valid 2x2 RGB PNG (used for the sampler-filter test).
    const std::vector<uint8_t> kPng2x2 = {
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x08, 0x02, 0x00, 0x00, 0x00, 0xfd, 0xd4, 0x9a,
            0x73, 0x00, 0x00, 0x00, 0x14, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0xf8, 0xcf, 0xc0, 0xc0,
            0x00, 0xc2, 0x0c, 0xff, 0xff, 0xff, 0x67, 0x00, 0x00, 0x1e, 0xef, 0x04, 0xfc, 0x73, 0x1c, 0x53,
            0xcc, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};

}// namespace

TEST_CASE("GLTFLoader decodes attribute and index values") {
    Bin bin;
    // 3 float positions (a triangle) + 3 ushort indices
    size_t posOff = bin.put<float>({0.f, 0.f, 0.f, 2.f, 0.f, 0.f, 0.f, 3.f, 0.f});
    size_t idxOff = bin.put<uint16_t>({0, 1, 2});

    std::string json = R"({
      "asset":{"version":"2.0"},
      "buffers":[{"byteLength":)" + std::to_string(bin.data.size()) + R"(}],
      "bufferViews":[
        {"buffer":0,"byteOffset":)" + std::to_string(posOff) + R"(,"byteLength":36},
        {"buffer":0,"byteOffset":)" + std::to_string(idxOff) + R"(,"byteLength":6}],
      "accessors":[
        {"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},
        {"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}],
      "meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],
      "nodes":[{"mesh":0}],
      "scenes":[{"nodes":[0]}]
    })";

    auto path = writeTempGlb(makeGlb(json, bin.data));
    GLTFLoader loader;
    auto res = loader.load(path);
    fs::remove(path);

    REQUIRE(res);
    auto* mesh = firstMesh(res->scene.get());
    REQUIRE(mesh);
    auto geom = mesh->geometry();
    auto* pos = geom->getAttribute<float>("position");
    REQUIRE(pos);
    REQUIRE(pos->count() == 3);
    const auto& a = pos->array();
    CHECK(a[3] == 2.f);// vertex 1 x
    CHECK(a[7] == 3.f);// vertex 2 y
    REQUIRE(geom->hasIndex());
    const auto& idx = geom->getIndex()->array();
    REQUIRE(idx.size() == 3);
    CHECK(idx[0] == 0u);
    CHECK(idx[1] == 1u);
    CHECK(idx[2] == 2u);
}

TEST_CASE("GLTFLoader honours the normalized flag") {
    Bin bin;
    // POSITION: UNSIGNED_SHORT, normalized=false -> raw integer values
    size_t posOff = bin.put<uint16_t>({0, 0, 0, 100, 0, 0, 0, 65535, 0});
    // NORMAL: signed BYTE (5120), normalized=true -> v/127 clamped to [-1,1]
    size_t nrmOff = bin.put<int8_t>({0, 0, 127, 0, 0, -127, 127, 0, 0});

    std::string json = R"({
      "asset":{"version":"2.0"},
      "buffers":[{"byteLength":)" + std::to_string(bin.data.size()) + R"(}],
      "bufferViews":[
        {"buffer":0,"byteOffset":)" + std::to_string(posOff) + R"(,"byteLength":18},
        {"buffer":0,"byteOffset":)" + std::to_string(nrmOff) + R"(,"byteLength":9}],
      "accessors":[
        {"bufferView":0,"componentType":5123,"normalized":false,"count":3,"type":"VEC3"},
        {"bufferView":1,"componentType":5120,"normalized":true,"count":3,"type":"VEC3"}],
      "meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1}}]}],
      "nodes":[{"mesh":0}],
      "scenes":[{"nodes":[0]}]
    })";

    auto path = writeTempGlb(makeGlb(json, bin.data));
    GLTFLoader loader;
    auto res = loader.load(path);
    fs::remove(path);

    REQUIRE(res);
    auto* mesh = firstMesh(res->scene.get());
    REQUIRE(mesh);
    auto geom = mesh->geometry();

    const auto& pos = geom->getAttribute<float>("position")->array();
    // Non-normalized USHORT must be verbatim, NOT divided by 65535.
    CHECK(pos[3] == 100.f);  // vertex 1 x
    CHECK(pos[7] == 65535.f);// vertex 2 y

    const auto& nrm = geom->getAttribute<float>("normal")->array();
    // Signed BYTE normalized: 127->1, -127->-1, 0->0.
    CHECK(nrm[2] == 1.f);  // vertex 0 z
    CHECK(nrm[5] == -1.f); // vertex 1 z
    CHECK(nrm[6] == 1.f);  // vertex 2 x
    CHECK(nrm[0] == 0.f);  // vertex 0 x
}

TEST_CASE("GLTFLoader applies a sparse accessor overlay") {
    Bin bin;
    // Base POSITION: 3 float verts, all at y=0.
    size_t baseOff = bin.put<float>({0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 2.f, 0.f, 0.f});
    // Sparse: replace index 1 with (1, 9, 0).
    size_t sIdxOff = bin.put<uint16_t>({1});
    size_t sValOff = bin.put<float>({1.f, 9.f, 0.f});

    std::string json = R"({
      "asset":{"version":"2.0"},
      "buffers":[{"byteLength":)" + std::to_string(bin.data.size()) + R"(}],
      "bufferViews":[
        {"buffer":0,"byteOffset":)" + std::to_string(baseOff) + R"(,"byteLength":36},
        {"buffer":0,"byteOffset":)" + std::to_string(sIdxOff) + R"(,"byteLength":2},
        {"buffer":0,"byteOffset":)" + std::to_string(sValOff) + R"(,"byteLength":12}],
      "accessors":[{
        "bufferView":0,"componentType":5126,"count":3,"type":"VEC3",
        "sparse":{"count":1,
          "indices":{"bufferView":1,"byteOffset":0,"componentType":5123},
          "values":{"bufferView":2,"byteOffset":0}}}],
      "meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],
      "nodes":[{"mesh":0}],
      "scenes":[{"nodes":[0]}]
    })";

    auto path = writeTempGlb(makeGlb(json, bin.data));
    GLTFLoader loader;
    auto res = loader.load(path);
    fs::remove(path);

    REQUIRE(res);
    auto* mesh = firstMesh(res->scene.get());
    REQUIRE(mesh);
    const auto& pos = mesh->geometry()->getAttribute<float>("position")->array();
    // Vertex 0 and 2 untouched, vertex 1 overlaid.
    CHECK(pos[0] == 0.f);
    CHECK(pos[3] == 1.f);// vertex 1 x (overlaid)
    CHECK(pos[4] == 9.f);// vertex 1 y (overlaid)
    CHECK(pos[6] == 2.f);// vertex 2 x (base)
}

TEST_CASE("GLTFLoader rejects a truncated GLB cleanly") {
    Bin bin;
    bin.put<float>({0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f});
    std::string json = R"({"asset":{"version":"2.0"},
      "buffers":[{"byteLength":36}],
      "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36}],
      "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}],
      "meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],
      "nodes":[{"mesh":0}],"scenes":[{"nodes":[0]}]})";
    auto glb = makeGlb(json, bin.data);

    SECTION("cut in half") {
        std::vector<uint8_t> truncated(glb.begin(), glb.begin() + glb.size() / 2);
        auto path = writeTempGlb(truncated);
        GLTFLoader loader;
        auto res = loader.load(path);// must not crash
        fs::remove(path);
        CHECK_FALSE(res.has_value());
    }

    SECTION("chunk length overruns file") {
        // Corrupt the JSON chunk length (bytes 12..15) to claim a huge size.
        auto corrupt = glb;
        uint32_t huge = 0x7FFFFFFF;
        std::memcpy(corrupt.data() + 12, &huge, 4);
        auto path = writeTempGlb(corrupt);
        GLTFLoader loader;
        auto res = loader.load(path);
        fs::remove(path);
        CHECK_FALSE(res.has_value());
    }
}

TEST_CASE("GLTFLoader builds independent graphs for scenes sharing a node") {
    Bin bin;
    size_t posOff = bin.put<float>({0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f});
    size_t idxOff = bin.put<uint16_t>({0, 1, 2});

    std::string json = R"({
      "asset":{"version":"2.0"},
      "buffers":[{"byteLength":)" + std::to_string(bin.data.size()) + R"(}],
      "bufferViews":[
        {"buffer":0,"byteOffset":)" + std::to_string(posOff) + R"(,"byteLength":36},
        {"buffer":0,"byteOffset":)" + std::to_string(idxOff) + R"(,"byteLength":6}],
      "accessors":[
        {"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},
        {"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}],
      "meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],
      "nodes":[{"name":"Shared","mesh":0},{"name":"OnlyB","mesh":0}],
      "scenes":[{"nodes":[0]},{"nodes":[0,1]}]
    })";

    auto path = writeTempGlb(makeGlb(json, bin.data));
    GLTFLoader loader;
    auto res = loader.load(path);
    fs::remove(path);

    REQUIRE(res);
    REQUIRE(res->scenes.size() == 2);

    std::vector<Mesh*> a, b;
    collectMeshes(res->scenes[0].get(), a);
    collectMeshes(res->scenes[1].get(), b);

    // Scene A has the shared node; scene B has the shared node + its own.
    REQUIRE(a.size() == 1);
    REQUIRE(b.size() == 2);
    // The shared node's mesh object is distinct per scene (not reparented).
    for (auto* ma : a)
        for (auto* mb : b)
            CHECK(ma != mb);
    // ...but they share the cached geometry.
    CHECK(a[0]->geometry() == b[0]->geometry());
}

TEST_CASE("GLTFLoader shares geometry across nodes referencing one mesh") {
    Bin bin;
    size_t posOff = bin.put<float>({0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f});
    size_t idxOff = bin.put<uint16_t>({0, 1, 2});

    std::string json = R"({
      "asset":{"version":"2.0"},
      "buffers":[{"byteLength":)" + std::to_string(bin.data.size()) + R"(}],
      "bufferViews":[
        {"buffer":0,"byteOffset":)" + std::to_string(posOff) + R"(,"byteLength":36},
        {"buffer":0,"byteOffset":)" + std::to_string(idxOff) + R"(,"byteLength":6}],
      "accessors":[
        {"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},
        {"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}],
      "meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],
      "nodes":[{"name":"A","mesh":0},{"name":"B","mesh":0}],
      "scenes":[{"nodes":[0,1]}]
    })";

    auto path = writeTempGlb(makeGlb(json, bin.data));
    GLTFLoader loader;
    auto res = loader.load(path);
    fs::remove(path);

    REQUIRE(res);
    std::vector<Mesh*> meshes;
    collectMeshes(res->scene.get(), meshes);
    REQUIRE(meshes.size() == 2);
    // Distinct Mesh objects, shared (same shared_ptr) geometry.
    CHECK(meshes[0] != meshes[1]);
    CHECK(meshes[0]->geometry() == meshes[1]->geometry());
}

TEST_CASE("GLTFLoader maps a NEAREST sampler to Filter::Nearest") {
    Bin bin;
    size_t posOff = bin.put<float>({0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f});
    size_t uvOff = bin.put<float>({0.f, 0.f, 1.f, 0.f, 0.f, 1.f});
    size_t idxOff = bin.put<uint16_t>({0, 1, 2});
    size_t pngOff = bin.putBytes(kPng2x2);

    std::string json = R"({
      "asset":{"version":"2.0"},
      "buffers":[{"byteLength":)" + std::to_string(bin.data.size()) + R"(}],
      "bufferViews":[
        {"buffer":0,"byteOffset":)" + std::to_string(posOff) + R"(,"byteLength":36},
        {"buffer":0,"byteOffset":)" + std::to_string(uvOff) + R"(,"byteLength":24},
        {"buffer":0,"byteOffset":)" + std::to_string(idxOff) + R"(,"byteLength":6},
        {"buffer":0,"byteOffset":)" + std::to_string(pngOff) + R"(,"byteLength":)" + std::to_string(kPng2x2.size()) + R"(}],
      "accessors":[
        {"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},
        {"bufferView":1,"componentType":5126,"count":3,"type":"VEC2"},
        {"bufferView":2,"componentType":5123,"count":3,"type":"SCALAR"}],
      "images":[{"bufferView":3,"mimeType":"image/png"}],
      "samplers":[{"magFilter":9728,"minFilter":9728}],
      "textures":[{"source":0,"sampler":0}],
      "materials":[{"pbrMetallicRoughness":{"baseColorTexture":{"index":0}}}],
      "meshes":[{"primitives":[{"attributes":{"POSITION":0,"TEXCOORD_0":1},"indices":2,"material":0}]}],
      "nodes":[{"mesh":0}],
      "scenes":[{"nodes":[0]}]
    })";

    auto path = writeTempGlb(makeGlb(json, bin.data));
    GLTFLoader loader;
    auto res = loader.load(path);
    fs::remove(path);

    REQUIRE(res);
    auto* mesh = firstMesh(res->scene.get());
    REQUIRE(mesh);
    auto mat = std::dynamic_pointer_cast<MeshStandardMaterial>(mesh->material());
    REQUIRE(mat);
    REQUIRE(mat->map);
    CHECK(mat->map->magFilter == Filter::Nearest);
    CHECK(mat->map->minFilter == Filter::Nearest);
    // A non-mipmap min filter disables mipmap generation.
    CHECK(mat->map->generateMipmaps == false);
}

TEST_CASE("GLTFLoader does not include an unreachable node's mesh") {
    Bin bin;
    size_t posOff = bin.put<float>({0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f});
    size_t idxOff = bin.put<uint16_t>({0, 1, 2});

    // Node 1 ("NoReach") carries a mesh but is in no scene and is no node's child.
    std::string json = R"({
      "asset":{"version":"2.0"},
      "buffers":[{"byteLength":)" + std::to_string(bin.data.size()) + R"(}],
      "bufferViews":[
        {"buffer":0,"byteOffset":)" + std::to_string(posOff) + R"(,"byteLength":36},
        {"buffer":0,"byteOffset":)" + std::to_string(idxOff) + R"(,"byteLength":6}],
      "accessors":[
        {"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},
        {"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}],
      "meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],
      "nodes":[{"name":"Reach","mesh":0},{"name":"NoReach","mesh":0}],
      "scenes":[{"nodes":[0]}]
    })";

    auto path = writeTempGlb(makeGlb(json, bin.data));
    GLTFLoader loader;
    auto res = loader.load(path);
    fs::remove(path);

    REQUIRE(res);
    std::vector<Mesh*> meshes;
    collectMeshes(res->scene.get(), meshes);
    REQUIRE(meshes.size() == 1);
    CHECK(meshes[0]->name != "NoReach");
}

TEST_CASE("GLTFLoader creates a SpotLight with nested spot cone angles") {
    std::string json = R"({
      "asset":{"version":"2.0"},
      "extensionsUsed":["KHR_lights_punctual"],
      "extensions":{"KHR_lights_punctual":{"lights":[
        {"name":"MySpot","type":"spot","intensity":5.0,"range":20.0,
         "spot":{"innerConeAngle":0.2,"outerConeAngle":0.6}},
        {"name":"MyPoint","type":"point","intensity":2.0}]}},
      "nodes":[
        {"name":"spotNode","extensions":{"KHR_lights_punctual":{"light":0}}},
        {"name":"pointNode","extensions":{"KHR_lights_punctual":{"light":1}}}],
      "scenes":[{"nodes":[0,1]}]
    })";

    auto path = writeTempGlb(makeGlb(json, {}));
    GLTFLoader loader;
    auto res = loader.load(path);
    fs::remove(path);

    REQUIRE(res);
    SpotLight* spot = nullptr;
    PointLight* point = nullptr;
    res->scene->traverse([&](Object3D& o) {
        if (auto* s = o.as<SpotLight>()) spot = s;
        else if (auto* p = o.as<PointLight>()) point = p;
    });

    REQUIRE(spot);
    CHECK(spot->name == "MySpot");
    CHECK(spot->distance == 20.0f);
    CHECK(spot->angle == 0.6f);
    CHECK_THAT(spot->penumbra, Catch::Matchers::WithinAbs(1.f - 0.2f / 0.6f, 1e-5));

    REQUIRE(point);
    CHECK(point->name == "MyPoint");
}

TEST_CASE("GLTFLoader preserves normalized uint8 COLOR_0 as a narrow attribute") {
    Bin bin;
    size_t posOff = bin.put<float>({0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f});
    size_t colOff = bin.putBytes({255, 0, 0, 0, 255, 0, 128, 128, 128});// u8 RGB per vertex

    std::string json = R"({
      "asset":{"version":"2.0"},
      "buffers":[{"byteLength":)" + std::to_string(bin.data.size()) + R"(}],
      "bufferViews":[
        {"buffer":0,"byteOffset":)" + std::to_string(posOff) + R"(,"byteLength":36},
        {"buffer":0,"byteOffset":)" + std::to_string(colOff) + R"(,"byteLength":9}],
      "accessors":[
        {"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},
        {"bufferView":1,"componentType":5121,"normalized":true,"count":3,"type":"VEC3"}],
      "meshes":[{"primitives":[{"attributes":{"POSITION":0,"COLOR_0":1}}]}],
      "nodes":[{"mesh":0}],
      "scenes":[{"nodes":[0]}]
    })";

    auto path = writeTempGlb(makeGlb(json, bin.data));
    GLTFLoader loader;
    auto res = loader.load(path);
    fs::remove(path);

    REQUIRE(res);
    auto* mesh = firstMesh(res->scene.get());
    REQUIRE(mesh);
    auto geom = mesh->geometry();

    const auto* col = geom->getAttribute("color");
    REQUIRE(col);
    CHECK(col->type() == AttributeType::UInt8);
    CHECK(col->normalized());
    CHECK(col->itemSize() == 3);
    CHECK(col->count() == 3);
    CHECK(col->byteLength() == 9);// was 36 as widened float

    // Raw stored bytes, denormalized on read through the view.
    FloatAttributeView view(col);
    CHECK(view[0] == 1.f);
    CHECK(view[1] == 0.f);
    CHECK(view[4] == 1.f);
    CHECK(std::abs(view[6] - 128.f / 255.f) < 1e-6f);
}

TEST_CASE("GLTFLoader widens COLOR_0 when preserveNarrowAttributes is off") {
    Bin bin;
    size_t posOff = bin.put<float>({0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f});
    size_t colOff = bin.put<uint16_t>({65535, 0, 0, 0, 65535, 0, 0, 0, 65535});

    std::string json = R"({
      "asset":{"version":"2.0"},
      "buffers":[{"byteLength":)" + std::to_string(bin.data.size()) + R"(}],
      "bufferViews":[
        {"buffer":0,"byteOffset":)" + std::to_string(posOff) + R"(,"byteLength":36},
        {"buffer":0,"byteOffset":)" + std::to_string(colOff) + R"(,"byteLength":18}],
      "accessors":[
        {"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},
        {"bufferView":1,"componentType":5123,"normalized":true,"count":3,"type":"VEC3"}],
      "meshes":[{"primitives":[{"attributes":{"POSITION":0,"COLOR_0":1}}]}],
      "nodes":[{"mesh":0}],
      "scenes":[{"nodes":[0]}]
    })";

    auto path = writeTempGlb(makeGlb(json, bin.data));

    GLTFLoader narrow;
    auto resNarrow = narrow.load(path);

    GLTFLoader wide;
    wide.preserveNarrowAttributes = false;
    auto resWide = wide.load(path);
    fs::remove(path);

    REQUIRE(resNarrow);
    REQUIRE(resWide);

    const auto* colNarrow = firstMesh(resNarrow->scene.get())->geometry()->getAttribute("color");
    REQUIRE(colNarrow);
    CHECK(colNarrow->type() == AttributeType::UInt16);
    CHECK(colNarrow->normalized());

    // Opt-out restores the old behaviour: denormalized floats.
    auto* colWide = firstMesh(resWide->scene.get())->geometry()->getAttribute<float>("color");
    REQUIRE(colWide);
    CHECK_FALSE(colWide->normalized());
    CHECK(colWide->array()[0] == 1.f);
    CHECK(colWide->array()[4] == 1.f);
}

// OpenCASCADE's RWGltf_CafWriter emits {"POSITION":-1,"indices":-1} primitives
// for faces it failed to triangulate. Those must not sink the whole document.
TEST_CASE("GLTFLoader skips primitives with out-of-range accessors") {
    Bin bin;
    size_t posOff = bin.put<float>({0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f});
    size_t idxOff = bin.put<uint16_t>({0, 1, 2});

    // Mesh 0: one good primitive plus a degenerate one. Mesh 1: nothing but
    // degenerate primitives, which must still yield an (empty) Group.
    std::string json = R"({
      "asset":{"version":"2.0"},
      "buffers":[{"byteLength":)" + std::to_string(bin.data.size()) + R"(}],
      "bufferViews":[
        {"buffer":0,"byteOffset":)" + std::to_string(posOff) + R"(,"byteLength":36},
        {"buffer":0,"byteOffset":)" + std::to_string(idxOff) + R"(,"byteLength":6}],
      "accessors":[
        {"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},
        {"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}],
      "meshes":[
        {"name":"Solid","primitives":[
          {"attributes":{"POSITION":0},"indices":1,"mode":4},
          {"attributes":{"POSITION":-1},"indices":-1,"mode":4}]},
        {"name":"BadFace","primitives":[
          {"attributes":{"POSITION":-1},"indices":-1,"mode":4}]}],
      "nodes":[{"mesh":0},{"mesh":1}],
      "scenes":[{"nodes":[0,1]}]
    })";

    auto path = writeTempGlb(makeGlb(json, bin.data));
    GLTFLoader loader;
    auto res = loader.load(path);
    fs::remove(path);

    REQUIRE(res);
    std::vector<Mesh*> meshes;
    collectMeshes(res->scene.get(), meshes);
    REQUIRE(meshes.size() == 1);

    // The survivor is the valid primitive, decoded intact.
    auto geom = meshes[0]->geometry();
    REQUIRE(geom->getAttribute<float>("position")->count() == 3);
    REQUIRE(geom->hasIndex());
    CHECK(geom->getIndex()->array().size() == 3);
}
