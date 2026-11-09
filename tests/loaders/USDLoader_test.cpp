
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/loaders/USDLoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

using namespace threepp;
using Catch::Matchers::WithinAbs;

namespace {

    // A 1x1 opaque white PNG.
    const unsigned char png[] = {
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
            0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59, 0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
            0x44, 0xAE, 0x42, 0x60, 0x82};

    void write(const std::filesystem::path& path, const std::string& text) {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream(path) << text;
    }

    void writePng(const std::filesystem::path& path) {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(png), sizeof(png));
    }

    // A triangle bound to a material whose shader names only its own .mdl file, as the SimReady
    // warehouse's flat boxes do.
    const std::string document = R"(#usda 1.0
(
    defaultPrim = "box"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "box" {
    def Mesh "mesh" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    ) {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (0, 1)] (
            interpolation = "vertex"
        )
        rel material:binding = </box/Looks/Decal>
    }

    def Scope "Looks" {
        def Material "Decal" {
            token outputs:mdl:surface.connect = </box/Looks/Decal/Shader.outputs:out>

            def Shader "Shader" {
                uniform token info:implementationSource = "sourceAsset"
                uniform asset info:mdl:sourceAsset = @./materials/Decal.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "Decal"
                token outputs:out
            }
        }
    }
}
)";

    const std::string mdl = R"(mdl 1.4;

using ::OmniPBR import OmniPBR;
import ::tex::gamma_mode;

export material Decal(*)
 = OmniPBR(
	diffuse_color_constant: color(0.2, 0.2, 0.2),
	diffuse_texture: texture_2d("./textures/Decal_Albedo.png", ::tex::gamma_srgb),
	enable_ORM_texture: true,
	ORM_texture: texture_2d("./textures/Decal_ORM.png", ::tex::gamma_linear),
	ao_texture: texture_2d(),
	enable_opacity: true,
	enable_opacity_texture: true,
	opacity_constant: 1.0,
	opacity_texture: texture_2d("./textures/Decal_Opacity.png", ::tex::gamma_srgb),
	opacity_threshold: 0.05,
	normalmap_texture: texture_2d("./textures/Decal_Normal.png", ::tex::gamma_linear),
	texture_scale: float2(1, 1));
)";

    // Two triangles, one in UDIM tile 1001 and one in 1002, under one OmniPBR material whose
    // albedo is a UDIM texture set, as the SimReady warehouse's racks and wooden crates are.
    const std::string udimDocument = R"(#usda 1.0
(
    defaultPrim = "crate"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "crate" {
    def Mesh "mesh" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    ) {
        int[] faceVertexCounts = [3, 3]
        int[] faceVertexIndices = [0, 1, 2, 3, 4, 5]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 0, 0), (3, 0, 0), (2, 1, 0)]
        texCoord2f[] primvars:st = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (1.1, 0.1), (1.9, 0.1), (1.1, 0.9)] (
            interpolation = "vertex"
        )
        rel material:binding = </crate/Looks/Crate>
    }

    def Scope "Looks" {
        def Material "Crate" {
            token outputs:mdl:surface.connect = </crate/Looks/Crate/Shader.outputs:out>

            def Shader "Shader" {
                uniform token info:implementationSource = "sourceAsset"
                uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
                asset inputs:diffuse_texture = @./textures/Crate_Albedo.<UDIM>.png@
                token outputs:out
            }
        }
    }
}
)";

    std::shared_ptr<MeshStandardMaterial> materialOf(Object3D& root) {
        std::shared_ptr<MeshStandardMaterial> found;
        root.traverse([&](Object3D& o) {
            if (auto* mesh = o.as<Mesh>(); mesh && !found) found = std::dynamic_pointer_cast<MeshStandardMaterial>(mesh->material());
        });
        return found;
    }

}// namespace

TEST_CASE("a material of its own reads its textures from its .mdl file", "[usd]") {
    const auto dir = std::filesystem::temp_directory_path() / "usd_mdl_defaults";
    write(dir / "box.usda", document);
    write(dir / "materials" / "Decal.mdl", mdl);
    for (const char* t : {"Decal_Albedo", "Decal_ORM", "Decal_Opacity", "Decal_Normal"}) {
        writePng(dir / "materials" / "textures" / (std::string(t) + ".png"));
    }

    USDLoader loader;
    const auto scene = loader.load(dir / "box.usda");
    REQUIRE(scene);
    const auto mat = materialOf(*scene);
    REQUIRE(mat);
    CHECK(mat->map);
    CHECK(mat->normalMap);
    CHECK(mat->roughnessMap);// the ORM texture: occlusion, roughness, metalness
    CHECK(mat->metalnessMap);
    CHECK(mat->aoMap);
    // the opacity mask cuts out below its threshold
    CHECK(mat->alphaMap);
    REQUIRE_THAT(mat->alphaTest, WithinAbs(0.05f, 1e-6f));
}

TEST_CASE("a UDIM texture set gives each tile's part of a mesh that tile's texture", "[usd]") {
    const auto dir = std::filesystem::temp_directory_path() / "usd_udim";
    std::filesystem::remove_all(dir);
    write(dir / "crate.usda", udimDocument);
    for (const char* t : {"Crate_Albedo.1001", "Crate_Albedo.1002"}) {
        writePng(dir / "textures" / (std::string(t) + ".png"));
    }

    USDLoader loader;
    const auto scene = loader.load(dir / "crate.usda");
    REQUIRE(scene);
    std::vector<Mesh*> meshes;
    scene->traverse([&](Object3D& o) {
        if (auto* mesh = o.as<Mesh>()) meshes.push_back(mesh);
    });
    REQUIRE(meshes.size() == 2);
    std::set<std::string> textures;
    for (auto* mesh : meshes) {
        const auto mat = std::dynamic_pointer_cast<MeshStandardMaterial>(mesh->material());
        REQUIRE(mat);
        REQUIRE(mat->map);
        textures.insert(mat->map->name);
        // the part's UVs moved into the square its tile's texture covers
        const auto* uv = mesh->geometry()->getAttribute<float>("uv");
        REQUIRE(uv);
        for (int i = 0; i < uv->count(); ++i) {
            CHECK(uv->getX(i) > 0.f);
            CHECK(uv->getX(i) < 1.f);
        }
    }
    CHECK(textures == std::set<std::string>{"Crate_Albedo.1001", "Crate_Albedo.1002"});
}

TEST_CASE("a material whose .mdl file is not beside the asset is left as the prim authors it", "[usd]") {
    const auto dir = std::filesystem::temp_directory_path() / "usd_mdl_missing";
    std::filesystem::remove_all(dir);
    write(dir / "box.usda", document);

    USDLoader loader;
    const auto scene = loader.load(dir / "box.usda");
    REQUIRE(scene);
    const auto mat = materialOf(*scene);
    REQUIRE(mat);
    CHECK_FALSE(mat->map);
    CHECK_FALSE(mat->alphaMap);
}

TEST_CASE("OmniPBR flags authored on the shader prim as bool or int are read", "[usd]") {
    // A library OmniPBR (no .mdl beside the asset to fall back on) whose decal flags the
    // prim authors itself, as Isaac Sim writes them: bool for the opacity flags, int for
    // the emission flag. Read as floats they failed, and the decal drew as an opaque quad.
    const std::string flagged = R"(#usda 1.0
(
    defaultPrim = "decal"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "decal" {
    def Mesh "mesh" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    ) {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (0, 1)] (
            interpolation = "vertex"
        )
        rel material:binding = </decal/Looks/Decal>
    }

    def Scope "Looks" {
        def Material "Decal" {
            token outputs:mdl:surface.connect = </decal/Looks/Decal/Shader.outputs:out>

            def Shader "Shader" {
                uniform token info:implementationSource = "sourceAsset"
                uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
                bool inputs:enable_opacity = 1
                bool inputs:enable_opacity_texture = true
                asset inputs:opacity_texture = @./textures/Decal_Opacity.png@
                float inputs:opacity_threshold = 0.2
                int inputs:enable_emission = 1
                color3f inputs:emissive_color = (1, 0.5, 0)
                float inputs:emissive_intensity = 2000
                token outputs:out
            }
        }
    }
}
)";
    const auto dir = std::filesystem::temp_directory_path() / "usd_mdl_prim_flags";
    std::filesystem::remove_all(dir);
    write(dir / "decal.usda", flagged);
    writePng(dir / "textures" / "Decal_Opacity.png");

    USDLoader loader;
    const auto scene = loader.load(dir / "decal.usda");
    REQUIRE(scene);
    const auto mat = materialOf(*scene);
    REQUIRE(mat);
    CHECK(mat->alphaMap);
    CHECK_THAT(mat->alphaTest, WithinAbs(0.2f, 1e-6f));
    CHECK(mat->emissive == Color(1, 0.5f, 0));
    CHECK_THAT(mat->emissiveIntensity, WithinAbs(2.f, 1e-6f));
}
