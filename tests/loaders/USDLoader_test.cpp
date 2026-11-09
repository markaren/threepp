
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/lights/SpotLight.hpp"
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

TEST_CASE("a UsdPreviewSurface UDIM texture gives each tile's part of a mesh that tile's image", "[usd]") {
    // As a Blender or Substance export writes it: a UsdUVTexture whose file is a UDIM set.
    // LightUSD resolves the texture's image to the set's first tile (1001), so every part
    // drew 1001's image unless the tile is looked up in the set's own list.
    const std::string preview = R"(#usda 1.0
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
            token outputs:surface.connect = </crate/Looks/Crate/Surface.outputs:surface>

            def Shader "Surface" {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </crate/Looks/Crate/Albedo.outputs:rgb>
                token outputs:surface
            }

            def Shader "Albedo" {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @./textures/Crate_Albedo.<UDIM>.png@
                float2 inputs:st.connect = </crate/Looks/Crate/Reader.outputs:result>
                float3 outputs:rgb
            }

            def Shader "Reader" {
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }
        }
    }
}
)";
    const auto dir = std::filesystem::temp_directory_path() / "usd_udim_preview";
    std::filesystem::remove_all(dir);
    write(dir / "crate.usda", preview);
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
    }
    CHECK(textures == std::set<std::string>{"Crate_Albedo.1001", "Crate_Albedo.1002"});
}

TEST_CASE("OmniPBR's texture_scale, translate and rotate place its maps, which repeat", "[usd]") {
    // OmniPBR samples every map with wrap_repeat at uv' = S * R * uv + T, where
    // rotation_translation_scale in OmniPBR.mdl builds R from texture_rotate in degrees.
    // A plain UsdPreviewSurface material on the same image shows the loader cloned the
    // shared, cached texture before placing it.
    const auto stage = [](const std::string& shaderInputs) {
        return R"(#usda 1.0
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
        rel material:binding = </box/Looks/Floor>
    }

    def Mesh "plain" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    ) {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(2, 0, 0), (3, 0, 0), (2, 1, 0)]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (0, 1)] (
            interpolation = "vertex"
        )
        rel material:binding = </box/Looks/Plain>
    }

    def Scope "Looks" {
        def Material "Floor" {
            token outputs:mdl:surface.connect = </box/Looks/Floor/Shader.outputs:out>

            def Shader "Shader" {
                uniform token info:implementationSource = "sourceAsset"
                uniform asset info:mdl:sourceAsset = @./materials/Floor.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "Floor"
)" + shaderInputs + R"(
                token outputs:out
            }
        }

        def Material "Plain" {
            token outputs:surface.connect = </box/Looks/Plain/Surface.outputs:surface>

            def Shader "Surface" {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </box/Looks/Plain/Albedo.outputs:rgb>
                token outputs:surface
            }

            def Shader "Albedo" {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @./materials/textures/Floor_Albedo.png@
                float2 inputs:st.connect = </box/Looks/Plain/Reader.outputs:result>
                float3 outputs:rgb
            }

            def Shader "Reader" {
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }
        }
    }
}
)";
    };
    const std::string floorMdl = R"(mdl 1.4;

using ::OmniPBR import OmniPBR;
import ::tex::gamma_mode;

export material Floor(*)
 = OmniPBR(
	diffuse_texture: texture_2d("./textures/Floor_Albedo.png", ::tex::gamma_srgb),
	normalmap_texture: texture_2d("./textures/Floor_Normal.png", ::tex::gamma_linear),
	project_uvw: false,
	texture_translate: float2(0.25, 0.5),
	texture_rotate: 90.f,
	texture_scale: float2(3, 2));
)";

    const auto load = [&](const std::string& name, const std::string& shaderInputs) {
        const auto dir = std::filesystem::temp_directory_path() / name;
        std::filesystem::remove_all(dir);
        write(dir / "box.usda", stage(shaderInputs));
        write(dir / "materials" / "Floor.mdl", floorMdl);
        writePng(dir / "materials" / "textures" / "Floor_Albedo.png");
        writePng(dir / "materials" / "textures" / "Floor_Normal.png");
        USDLoader loader;
        auto scene = loader.load(dir / "box.usda");
        REQUIRE(scene);
        std::shared_ptr<MeshStandardMaterial> floor, plain;
        scene->traverse([&](Object3D& o) {
            if (auto* mesh = o.as<Mesh>()) {
                auto mat = std::dynamic_pointer_cast<MeshStandardMaterial>(mesh->material());
                (mesh->name == "plain" || (mat && mat->name == "Plain") ? plain : floor) = mat;
            }
        });
        REQUIRE(floor);
        REQUIRE(plain);
        REQUIRE(floor->map);
        REQUIRE(floor->normalMap);
        REQUIRE(plain->map);
        return std::make_pair(floor, plain);
    };
    // Where a texture samples a mesh's uv, by the matrix a renderer would use.
    const auto uvOf = [](const Texture& tex, float u, float v) {
        Matrix3 m = tex.matrix;
        if (tex.matrixAutoUpdate) {
            m.setUvTransform(tex.offset.x, tex.offset.y, tex.repeat.x, tex.repeat.y, tex.rotation, tex.center.x, tex.center.y);
        }
        Vector2 uv(u, v);
        uv.applyMatrix3(m);
        return uv;
    };
    const auto checkUv = [](const Vector2& uv, float u, float v) {
        CHECK_THAT(uv.x, WithinAbs(u, 1e-5f));
        CHECK_THAT(uv.y, WithinAbs(v, 1e-5f));
    };

    SECTION("from the material's .mdl file") {
        const auto [floor, plain] = load("usd_mdl_transform", "");
        for (const auto& map : {floor->map, floor->normalMap}) {
            CHECK(map->wrapS == TextureWrapping::Repeat);
            CHECK(map->wrapT == TextureWrapping::Repeat);
            // S = (3, 2), R = 90 degrees, T = (0.25, 0.5)
            checkUv(uvOf(*map, 0, 0), 0.25f, 0.5f);
            checkUv(uvOf(*map, 1, 0), 0.25f, -1.5f);
            checkUv(uvOf(*map, 0, 1), 3.25f, 0.5f);
        }
        CHECK(plain->map != floor->map);
        CHECK(plain->map->wrapS == TextureWrapping::ClampToEdge);
        checkUv(uvOf(*plain->map, 1, 0), 1, 0);
    }

    SECTION("authored on the prim, over the file's") {
        const auto [floor, plain] = load("usd_mdl_transform_prim", R"(
                float2 inputs:texture_scale = (4, 4)
                float2 inputs:texture_translate = (0, 0)
                float inputs:texture_rotate = 0)");
        checkUv(uvOf(*floor->map, 1, 1), 4, 4);
        CHECK(floor->map->wrapS == TextureWrapping::Repeat);
    }

    SECTION("not with project_uvw, which projects the maps from 3D instead") {
        const auto [floor, plain] = load("usd_mdl_transform_projected", R"(
                bool inputs:project_uvw = 1)");
        checkUv(uvOf(*floor->map, 1, 0), 1, 0);
        CHECK(floor->map->wrapS == TextureWrapping::Repeat);
    }
}

TEST_CASE("a clone of a loaded stage aims its spot light at its own copy of the target", "[usd]") {
    // The loader aims a UsdLux DiskLight (a SpotLight here) at a target object. As the light's
    // sibling under the root, the target was outside the light, so a clone of the stage shared
    // the original's target, which dies with the original: the renderers then read freed memory.
    const std::string stage = R"(#usda 1.0
(
    defaultPrim = "stage"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "stage" {
    def Mesh "floor" {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 0, 1)]
    }

    def DiskLight "lamp" {
        float inputs:intensity = 10
        float inputs:radius = 0.25
        double3 xformOp:translate = (1, 2, 3)
        float3 xformOp:rotateXYZ = (-90, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
    }
}
)";
    const auto dir = std::filesystem::temp_directory_path() / "usd_spot_clone";
    std::filesystem::remove_all(dir);
    write(dir / "lamp.usda", stage);

    const auto worldPosition = [](const Object3D& o) {
        Vector3 at;
        at.setFromMatrixPosition(*o.matrixWorld);
        return at;
    };
    const auto spotIn = [](Object3D& root) {
        SpotLight* found = nullptr;
        root.traverse([&](Object3D& o) {
            if (auto* spot = o.as<SpotLight>()) found = spot;
        });
        return found;
    };

    USDLoader loader;
    auto scene = loader.load(dir / "lamp.usda");
    REQUIRE(scene);
    scene->updateMatrixWorld(true);
    auto* light = spotIn(*scene);
    REQUIRE(light);
    // It shines down from (1, 2, 3).
    const Vector3 lightAt = worldPosition(*light);
    const Vector3 aim = worldPosition(light->target()) - lightAt;
    CHECK_THAT(aim.x, WithinAbs(0, 1e-5));
    CHECK_THAT(aim.y, WithinAbs(-1, 1e-5));
    CHECK_THAT(aim.z, WithinAbs(0, 1e-5));

    auto copy = scene->clone<Group>();
    REQUIRE(copy);
    auto* copiedLight = spotIn(*copy);
    REQUIRE(copiedLight);
    // The clone's target is in the clone, not the original's.
    REQUIRE(&copiedLight->target() != &light->target());
    bool inCopy = false;
    copy->traverse([&](Object3D& o) { inCopy = inCopy || &o == &copiedLight->target(); });
    REQUIRE(inCopy);

    scene.reset();
    copy->updateMatrixWorld(true);
    const Vector3 copiedAim = worldPosition(copiedLight->target()) - worldPosition(*copiedLight);
    CHECK(worldPosition(*copiedLight) == lightAt);
    CHECK_THAT(copiedAim.x, WithinAbs(0, 1e-5));
    CHECK_THAT(copiedAim.y, WithinAbs(-1, 1e-5));
    CHECK_THAT(copiedAim.z, WithinAbs(0, 1e-5));
}
