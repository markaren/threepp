import threepp as tp


def _two_ledges():
    scene = tp.Scene()
    for name, x in (("keep", -8.0), ("skip", 8.0)):
        ledge = tp.Mesh(tp.BoxGeometry(6, 1, 6), tp.MeshStandardMaterial())
        ledge.name = name
        ledge.position.set(x, 2.0, 0.0)
        scene.add(ledge)
    return scene


def test_perch_filter_rejects_by_name():
    scene = _two_ledges()
    params = tp.FlockParams()
    params.bird_count = 4

    flock = tp.Flock(params)
    flock.bake_perches_blocking(scene)
    unfiltered = flock.perch_count()
    assert unfiltered > 0

    seen = []

    def only_keep(mesh):
        seen.append(mesh.name)
        return mesh.name == "keep"

    # The bake used to raise "return_value_policy = copy" here: the binding
    # tried to copy the non-copyable Mesh into Python.
    flock.set_perch_filter(only_keep)
    flock.bake_perches_blocking(scene)
    filtered = flock.perch_count()
    assert {"keep", "skip"} <= set(seen)
    assert 0 < filtered < unfiltered

    flock.set_perch_filter(None)
    flock.bake_perches_blocking(scene)
    assert flock.perch_count() == unfiltered
