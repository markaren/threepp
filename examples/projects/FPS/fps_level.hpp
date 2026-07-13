// ============================================================================
//  FPS demo — UE-editor-style prototype-grid level
//  Included inside namespace {} in main.cpp — not a standalone header.
//  Requires: fps_constants.hpp (kArena, frand), PhysX + threepp types
//
//  The look: the classic engine "prototype/blockout" material — a light
//  blue-grey checker with a crisp 1 m grid, a faint 10 cm sub-grid, and
//  orange accent blocks — generated procedurally into one tileable
//  DataTexture. Box/cylinder UVs are rescaled to WORLD size so one texel
//  is the same physical size on every surface (the UE trick, done in UVs).
// ============================================================================

// One tileable 512x512 grid tile spanning kGridTileM x kGridTileM metres:
// 2x2 checker of two greys, 1 m major lines, 10 cm sub-lines, per-pixel noise.
constexpr float kGridTileM = 2.f;// world metres per texture repeat

std::shared_ptr<DataTexture> makeGridTexture() {
    const int N = 512;               // texels per tile edge
    const int major = N / 2;         // 1 m line spacing
    const int sub = N / 20;          // 10 cm sub-line spacing (25.6 -> 25px approx is fine)
    std::vector<unsigned char> px(static_cast<size_t>(N) * N * 4);

    auto put = [&](int x, int y, float r, float g, float b) {
        const size_t i = (static_cast<size_t>(y) * N + x) * 4;
        px[i + 0] = static_cast<unsigned char>(std::clamp(r, 0.f, 255.f));
        px[i + 1] = static_cast<unsigned char>(std::clamp(g, 0.f, 255.f));
        px[i + 2] = static_cast<unsigned char>(std::clamp(b, 0.f, 255.f));
        px[i + 3] = 255;
    };

    std::mt19937 nrng{42};
    std::uniform_real_distribution<float> noise(-3.f, 3.f);

    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            // 2x2 checker: light / slightly darker blue-grey
            const bool check = ((x / major) + (y / major)) % 2 == 0;
            float r = check ? 181.f : 165.f;
            float g = check ? 186.f : 171.f;
            float b = check ? 193.f : 179.f;

            // faint 10 cm sub-grid (1px)
            if (x % sub == 0 || y % sub == 0) {
                r -= 10.f;
                g -= 10.f;
                b -= 9.f;
            }

            // crisp 1 m major grid (2px, darker slate)
            const int mx = x % major, my = y % major;
            if (mx < 2 || my < 2) {
                r = 118.f;
                g = 126.f;
                b = 138.f;
            }

            // 2 m tile border (3px, darkest) — reads as the "big" grid
            if (x < 3 || y < 3) {
                r = 96.f;
                g = 104.f;
                b = 118.f;
            }

            const float n = noise(nrng);
            put(x, y, r + n, g + n, b + n);
        }
    }

    auto tex = DataTexture::create(ImageData{std::move(px)},
                                   static_cast<unsigned int>(N), static_cast<unsigned int>(N));
    tex->colorSpace = ColorSpace::sRGB;
    tex->wrapS = TextureWrapping::Repeat;
    tex->wrapT = TextureWrapping::Repeat;
    tex->magFilter = Filter::Linear;
    tex->minFilter = Filter::LinearMipmapLinear;
    tex->generateMipmaps = true;
    tex->anisotropy = 8;// grid lines at grazing angles stay crisp
    return tex;
}

// Rescale BoxGeometry UVs from per-face 0..1 to world metres so the grid
// texture tiles at constant physical density. Face axis from the normal.
void worldUvBox(BufferGeometry& geo, float sx, float sy, float sz, float metersPerTile = kGridTileM) {
    auto* uv = geo.getAttribute<float>("uv");
    auto* nrm = geo.getAttribute<float>("normal");
    if (!uv || !nrm) return;
    auto& uvA = uv->array();
    auto& nA = nrm->array();
    const float s = 1.f / metersPerTile;
    for (size_t i = 0; i < uv->count(); ++i) {
        const float nx = std::abs(nA[i * 3 + 0]);
        const float ny = std::abs(nA[i * 3 + 1]);
        const float nz = std::abs(nA[i * 3 + 2]);
        float su, sv;
        if (nx >= ny && nx >= nz) {
            su = sz;
            sv = sy;
        } else if (ny >= nx && ny >= nz) {
            su = sx;
            sv = sz;
        } else {
            su = sx;
            sv = sy;
        }
        uvA[i * 2 + 0] *= su * s;
        uvA[i * 2 + 1] *= sv * s;
    }
    uv->needsUpdate();
}

// Cylinder UVs: u wraps the circumference, v runs the height.
void worldUvCylinder(BufferGeometry& geo, float radius, float height, float metersPerTile = kGridTileM) {
    auto* uv = geo.getAttribute<float>("uv");
    if (!uv) return;
    auto& uvA = uv->array();
    const float s = 1.f / metersPerTile;
    const float circ = 2.f * math::PI * radius;
    for (size_t i = 0; i < uv->count(); ++i) {
        uvA[i * 2 + 0] *= circ * s;
        uvA[i * 2 + 1] *= height * s;
    }
    uv->needsUpdate();
}

// The arena builder output: handles the game logic needs afterwards.
struct Level {
    std::vector<Dynamic> dynamics;                               // shootable / pushable props
    std::unordered_map<const PxRigidActor*, Mesh*> actorToMesh;  // collider -> visual (decals)
};

// Build the whole blockout arena into scene + world. Every static collider
// registers in actorToMesh so shots can stamp decals on it.
Level buildLevel(Scene& scene, PhysxWorld& world) {
    Level lvl;

    auto gridTex = makeGridTexture();

    // Tints multiply the grey grid map: near-white floor, cooler walls,
    // UE-selection-orange accents, amber crates.
    auto matFloor = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(0xffffff).roughness(0.62f).metalness(0.03f).map(gridTex));
    auto matWall = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(0xcdd4e0).roughness(0.7f).metalness(0.03f).map(gridTex));
    auto matAccent = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(0xff9a3c).roughness(0.55f).metalness(0.05f).map(gridTex));
    auto matCrate = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(0xe8c987).roughness(0.6f).metalness(0.05f).map(gridTex));
    auto matDark = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(0x5f6772).roughness(0.8f).metalness(0.06f).map(gridTex));
    auto matGlowCyan = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(0x0b1114).roughness(0.4f).metalness(0.f).emissive(0x35c2ff).emissiveIntensity(2.6f));
    auto matGlowOrange = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(0x140d06).roughness(0.4f).metalness(0.f).emissive(0xff7a1a).emissiveIntensity(2.4f));

    // world-UV box mesh + static collider, registered for decals
    auto addBox = [&](const Vector3& pos, const Vector3& size, const std::shared_ptr<Material>& mat,
                      float yaw = 0.f, float pitch = 0.f) {
        auto geo = BoxGeometry::create(size.x, size.y, size.z);
        worldUvBox(*geo, size.x, size.y, size.z);
        auto m = Mesh::create(geo, mat);
        m->position.copy(pos);
        m->rotation.set(pitch, yaw, 0.f);
        m->castShadow = true;
        m->receiveShadow = true;
        scene.add(m);
        lvl.actorToMesh[world.addStatic(*m)] = m.get();
        return m;
    };
    // purely decorative box — no collider (thin trims / glow strips)
    auto addTrim = [&](const Vector3& pos, const Vector3& size, const std::shared_ptr<Material>& mat, float yaw = 0.f) {
        auto geo = BoxGeometry::create(size.x, size.y, size.z);
        worldUvBox(*geo, size.x, size.y, size.z);
        auto m = Mesh::create(geo, mat);
        m->position.copy(pos);
        m->rotation.y = yaw;
        m->castShadow = false;
        m->receiveShadow = true;
        scene.add(m);
        return m;
    };
    auto addPillar = [&](float x, float z, float radius, float height, const std::shared_ptr<Material>& mat) {
        auto geo = CylinderGeometry::create(radius, radius, height, 24);
        worldUvCylinder(*geo, radius, height);
        auto m = Mesh::create(geo, mat);
        m->position.set(x, height * 0.5f, z);
        m->castShadow = true;
        m->receiveShadow = true;
        scene.add(m);
        if (auto* b = world.addStaticTrimesh(*m)) lvl.actorToMesh[b] = m.get();
        return m;
    };

    // ---- floor -------------------------------------------------------------
    addBox({0, -0.5f, 0}, {kArena * 2 + 8.f, 1.f, kArena * 2 + 8.f}, matFloor);

    // ---- perimeter walls with a dark base skirt + glow strip ----------------
    const float wallH = 4.5f, wallT = 1.f;
    const float wp = kArena + wallT * 0.5f;
    addBox({0, wallH * 0.5f, -wp}, {kArena * 2 + 2.f * wallT, wallH, wallT}, matWall);
    addBox({0, wallH * 0.5f, wp}, {kArena * 2 + 2.f * wallT, wallH, wallT}, matWall);
    addBox({-wp, wallH * 0.5f, 0}, {wallT, wallH, kArena * 2}, matWall);
    addBox({wp, wallH * 0.5f, 0}, {wallT, wallH, kArena * 2}, matWall);
    // skirt + glow strips on the inner face of each wall
    for (int side = 0; side < 4; ++side) {
        const bool zAxis = side < 2;
        const float sgn = (side % 2 == 0) ? -1.f : 1.f;
        const float in = kArena - 0.05f;
        const Vector3 skirtPos = zAxis ? Vector3(0, 0.3f, sgn * in) : Vector3(sgn * in, 0.3f, 0);
        const Vector3 skirtSize = zAxis ? Vector3(kArena * 2, 0.6f, 0.1f) : Vector3(0.1f, 0.6f, kArena * 2);
        addTrim(skirtPos, skirtSize, matDark);
        const Vector3 glowPos = zAxis ? Vector3(0, 0.72f, sgn * in) : Vector3(sgn * in, 0.72f, 0);
        const Vector3 glowSize = zAxis ? Vector3(kArena * 2, 0.08f, 0.06f) : Vector3(0.06f, 0.08f, kArena * 2);
        addTrim(glowPos, glowSize, matGlowCyan);
    }

    // ---- corner towers -------------------------------------------------------
    for (int sx = -1; sx <= 1; sx += 2)
        for (int sz = -1; sz <= 1; sz += 2) {
            addBox({sx * (kArena - 1.5f), 3.f, sz * (kArena - 1.5f)}, {4.f, 6.f, 4.f}, matWall);
            addTrim({sx * (kArena - 1.5f), 6.08f, sz * (kArena - 1.5f)}, {4.1f, 0.12f, 4.1f}, matGlowOrange);
        }

    // ---- central raised platform with two ramps (accent orange) -------------
    {
        const float ph = 1.5f, pw = 10.f, pd = 8.f;
        addBox({0, ph * 0.5f, 0}, {pw, ph, pd}, matAccent);
        addTrim({0, ph + 0.05f, 0}, {pw - 0.4f, 0.1f, pd - 0.4f}, matFloor);// lighter deck inlay
        // ramps on the ±Z edges, pitched about X (pure pitch keeps the slope sane)
        const float run = 6.f;
        const float ang = std::atan2(ph, run);
        addBox({0, ph * 0.5f - 0.12f, pd * 0.5f + run * 0.5f - 0.4f}, {4.f, 0.5f, run}, matAccent, 0.f, ang);
        addBox({0, ph * 0.5f - 0.12f, -(pd * 0.5f + run * 0.5f - 0.4f)}, {4.f, 0.5f, run}, matAccent, 0.f, -ang);
        // parapet blocks on the ±X platform edges (cover up top)
        addBox({-pw * 0.5f + 0.5f, ph + 0.5f, 0}, {1.f, 1.f, 5.f}, matWall);
        addBox({pw * 0.5f - 0.5f, ph + 0.5f, 0}, {1.f, 1.f, 5.f}, matWall);
    }

    // ---- pillar clusters ------------------------------------------------------
    for (int sx = -1; sx <= 1; sx += 2)
        for (int sz = -1; sz <= 1; sz += 2)
            addPillar(sx * 11.f, sz * 11.f, 0.6f, 5.f, matWall);

    // ---- L-covers + mid-lane blocks -------------------------------------------
    auto addCornerCover = [&](float cx, float cz) {
        const float ix = cx > 0 ? -1.f : 1.f, iz = cz > 0 ? -1.f : 1.f;
        addBox({cx + ix * 2.f, 0.75f, cz}, {4.5f, 1.5f, 1.f}, matDark);
        addBox({cx, 0.75f, cz + iz * 2.f}, {1.f, 1.5f, 4.5f}, matDark);
    };
    addCornerCover(-17, -17);
    addCornerCover(17, -17);
    addCornerCover(-17, 17);
    addCornerCover(17, 17);
    addBox({16, 0.6f, 0}, {1.2f, 1.2f, 5.f}, matDark);
    addBox({-16, 0.6f, 0}, {1.2f, 1.2f, 5.f}, matDark);
    addBox({0, 0.6f, 16}, {5.f, 1.2f, 1.2f}, matDark);
    addBox({0, 0.6f, -16}, {5.f, 1.2f, 1.2f}, matDark);

    // ---- floating step blocks up to the corner towers (parkour flavour) -------
    addBox({kArena - 6.5f, 0.5f, kArena - 3.f}, {2.f, 1.f, 2.f}, matAccent);
    addBox({kArena - 4.5f, 1.2f, kArena - 5.5f}, {2.f, 1.f, 2.f}, matAccent);

    // ---- dynamic crates (shootable / pushable) ---------------------------------
    auto spawnCrateStack = [&](float cx, float cz, int height, float side) {
        auto geo = BoxGeometry::create(side, side, side);
        worldUvBox(*geo, side, side, side);
        for (int y = 0; y < height; ++y) {
            auto m = Mesh::create(geo, matCrate);
            m->position.set(cx + frand(-0.04f, 0.04f), side * 0.5f + y * (side + 0.002f), cz + frand(-0.04f, 0.04f));
            m->castShadow = true;
            m->receiveShadow = true;
            scene.add(m);
            auto* body = world.add(*m, 55.f);
            lvl.actorToMesh[body] = m.get();
            lvl.dynamics.push_back({m, body, m->position.clone()});
        }
    };
    spawnCrateStack(8.5f, 8.5f, 2, 0.9f);
    spawnCrateStack(-8.5f, 8.5f, 3, 0.8f);
    spawnCrateStack(8.5f, -9.f, 2, 1.1f);
    spawnCrateStack(-9.f, -8.5f, 2, 0.9f);
    spawnCrateStack(13.f, -3.f, 1, 1.2f);
    spawnCrateStack(-13.f, 3.f, 1, 1.2f);

    return lvl;
}
