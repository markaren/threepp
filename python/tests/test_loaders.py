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


def test_obj_loader_direct(obj_cube):
    model = tp.OBJLoader().load(obj_cube)
    assert isinstance(model, tp.Group)


def test_box3_set_from_object_frames_model(stl_cube):
    model = tp.ModelLoader().load(stl_cube)
    bbox = tp.Box3().set_from_object(model)
    assert not bbox.is_empty()
    size = bbox.get_size()
    assert size.x > 0 and size.y > 0 and size.z > 0


def test_loaded_model_can_be_retinted(stl_cube):
    model = tp.ModelLoader().load(stl_cube)
    mat = tp.MeshStandardMaterial()
    mat.color = 0xffaa33
    model.traverse(lambda o: o.set_material(mat) if isinstance(o, tp.Mesh) else None)
    # the loaded mesh now reports the new material
    found = []
    model.traverse(lambda o: found.append(o.material) if isinstance(o, tp.Mesh) else None)
    assert all(type(m).__name__ == "MeshStandardMaterial" for m in found)
