import threepp as tp


def test_stl_loader_returns_geometry(stl_cube):
    geo = tp.STLLoader().load(stl_cube)
    assert isinstance(geo, tp.BufferGeometry)
    mesh = tp.Mesh(geo, tp.MeshStandardMaterial())
    assert mesh.geometry is not None


def test_model_loader_stl(stl_cube):
    model = tp.ModelLoader().load(stl_cube)
    assert isinstance(model, tp.Group)
    meshes = [c for c in model.children]
    assert len(meshes) >= 1


def test_model_loader_obj(obj_cube):
    model = tp.ModelLoader().load(obj_cube)
    assert isinstance(model, tp.Group)
    n_meshes = [0]
    model.traverse(lambda o: n_meshes.__setitem__(0, n_meshes[0] + (1 if isinstance(o, tp.Mesh) else 0)))
    assert n_meshes[0] >= 1


def test_box3_set_from_object_frames_model(stl_cube):
    model = tp.ModelLoader().load(stl_cube)
    bbox = tp.Box3().set_from_object(model)
    assert not bbox.is_empty()
    size = bbox.get_size()
    assert size.x > 0 and size.y > 0 and size.z > 0


def test_rgbe_loader_returns_texture(hdr_env):
    tex = tp.RGBELoader().load(hdr_env)
    assert tex is not None and isinstance(tex, tp.Texture)
    assert tex.name == "env"  # named after the file stem


def test_scene_environment_and_hdr_background(hdr_env):
    tex = tp.RGBELoader().load(hdr_env)
    scene = tp.Scene()
    # image-based-lighting env map: assignable and clearable
    scene.environment = tex
    assert scene.environment is not None
    scene.environment = None
    assert scene.environment is None
    # HDR as background (the Texture branch of the background setter)
    scene.background = tex
    assert scene.background.is_texture()
    # a solid color still works through the same property
    scene.background = 0x202830
    assert scene.background.is_color()


def test_loaded_model_can_be_retinted(stl_cube):
    model = tp.ModelLoader().load(stl_cube)
    mat = tp.MeshStandardMaterial()
    mat.color = 0xffaa33
    model.traverse(lambda o: o.set_material(mat) if isinstance(o, tp.Mesh) else None)
    # the loaded mesh now reports the new material
    found = []
    model.traverse(lambda o: found.append(o.material) if isinstance(o, tp.Mesh) else None)
    assert all(type(m).__name__ == "MeshStandardMaterial" for m in found)


def test_gltf_emissive_strength_on_plain_material(tmp_path):
    """KHR_materials_emissive_strength reaches a material that needs no
    physical extension (it used to be read only for MeshPhysicalMaterial)."""
    import json
    import struct

    (tmp_path / "tri.bin").write_bytes(struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0))
    gltf = {
        "asset": {"version": "2.0"},
        "extensionsUsed": ["KHR_materials_emissive_strength"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
        "materials": [{
            "name": "glow",
            "emissiveFactor": [1.0, 0.5, 0.0],
            "extensions": {"KHR_materials_emissive_strength": {"emissiveStrength": 5.0}},
        }],
        "buffers": [{"uri": "tri.bin", "byteLength": 36}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                       "min": [0, 0, 0], "max": [1, 1, 0]}],
    }
    path = tmp_path / "glow.gltf"
    path.write_text(json.dumps(gltf))

    result = tp.GLTFLoader().load(str(path))
    mats = []
    result.scene.traverse(lambda o: mats.append(o.material) if isinstance(o, tp.Mesh) else None)
    assert len(mats) == 1
    assert type(mats[0]).__name__ == "MeshStandardMaterial"
    assert abs(mats[0].emissive_intensity - 5.0) < 1e-6


def _write_tri_gltf(tmp_path, name, **extra):
    """A one-triangle .gltf with `extra` merged into the top-level JSON."""
    import json
    import struct

    (tmp_path / "tri.bin").write_bytes(struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0))
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "buffers": [{"uri": "tri.bin", "byteLength": 36}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                       "min": [0, 0, 0], "max": [1, 1, 0]}],
    }
    gltf.update(extra)
    path = tmp_path / name
    path.write_text(json.dumps(gltf))
    return path


def test_gltf_node_extras_become_user_data(tmp_path):
    import json

    path = _write_tri_gltf(
        tmp_path, "extras.gltf",
        nodes=[{"mesh": 0, "name": "handle", "extras": {"role": "grip", "mass": 2.5}}],
        meshes=[{"primitives": [{"attributes": {"POSITION": 0}}], "extras": {"lod": 0, "role": "mesh"}}],
    )
    result = tp.GLTFLoader().load(str(path))
    meshes = []
    result.scene.traverse(lambda o: meshes.append(o) if isinstance(o, tp.Mesh) else None)
    assert len(meshes) == 1
    m = meshes[0]
    assert m.get_user_data("role") == "grip"  # the node's extras win over the mesh's
    assert m.get_user_data("mass") == "2.5"
    assert m.get_user_data("lod") == "0"  # mesh-level extras land too
    assert json.loads(m.get_user_data("extras")) == {"role": "grip", "mass": 2.5}


def test_gltf_unhandled_material_extension_is_named(tmp_path, capfd):
    path = _write_tri_gltf(
        tmp_path, "aniso.gltf",
        extensionsUsed=["KHR_materials_anisotropy"],
        meshes=[{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
        materials=[{"extensions": {"KHR_materials_anisotropy": {"anisotropyStrength": 0.5}}}],
    )
    result = tp.GLTFLoader().load(str(path))
    assert result.scene is not None
    err = capfd.readouterr().err
    assert "KHR_materials_anisotropy" in err


def test_gltf_material_variants(tmp_path):
    path = _write_tri_gltf(
        tmp_path, "variants.gltf",
        extensionsUsed=["KHR_materials_variants"],
        extensions={"KHR_materials_variants": {"variants": [{"name": "summer"}, {"name": "autumn"}]}},
        meshes=[{"primitives": [{"attributes": {"POSITION": 0}, "material": 0,
                                 "extensions": {"KHR_materials_variants": {"mappings": [
                                     {"material": 1, "variants": [1]}]}}}]}],
        materials=[{"name": "leaf_green", "pbrMetallicRoughness": {"baseColorFactor": [0.1, 0.6, 0.1, 1.0]}},
                   {"name": "leaf_gold", "pbrMetallicRoughness": {"baseColorFactor": [0.9, 0.7, 0.1, 1.0]}}],
    )
    result = tp.GLTFLoader().load(str(path))
    assert result.variant_names == ["summer", "autumn"]
    meshes = []
    result.scene.traverse(lambda o: meshes.append(o) if isinstance(o, tp.Mesh) else None)
    m = meshes[0]
    assert m.material.name == "leaf_green"
    # looked up without applying: the variant's material, the default where the variant does not list the mesh
    assert result.variant_material("autumn", m).name == "leaf_gold"
    assert result.variant_material("summer", m).name == "leaf_green"
    assert m.material.name == "leaf_green"
    result.apply_variant("autumn")
    assert m.material.name == "leaf_gold"
    result.reset_variants()
    assert m.material.name == "leaf_green"
    import pytest
    with pytest.raises(ValueError):
        result.apply_variant("winter")


def test_gltf_variant_keeps_vertex_colors(tmp_path):
    # a COLOR_0 primitive draws its material's vertexColors copy; its variant materials must too
    path = _write_tri_gltf(
        tmp_path, "variants_color.gltf",
        extensionsUsed=["KHR_materials_variants"],
        extensions={"KHR_materials_variants": {"variants": [{"name": "autumn"}]}},
        accessors=[{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                    "min": [0, 0, 0], "max": [1, 1, 0]},
                   {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        meshes=[{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "material": 0,
                                 "extensions": {"KHR_materials_variants": {"mappings": [
                                     {"material": 1, "variants": [0]}]}}}]}],
        materials=[{"name": "leaf_green"}, {"name": "leaf_gold"}],
    )
    result = tp.GLTFLoader().load(str(path))
    meshes = []
    result.scene.traverse(lambda o: meshes.append(o) if isinstance(o, tp.Mesh) else None)
    m = meshes[0]
    assert m.material.vertex_colors
    gold = result.variant_material("autumn", m)
    assert gold.name == "leaf_gold" and gold.vertex_colors
    result.apply_variant("autumn")
    assert m.material.vertex_colors
