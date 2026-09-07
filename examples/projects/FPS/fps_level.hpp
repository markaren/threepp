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

// Build the whole blockout RANGE into scene + world: floor, perimeter, a
// backstop berm, three measured lanes, and the dynamic targets that fill them.
// Every static collider registers in actorToMesh so shots can stamp decals on
// it; every dynamic one lands in lvl.dynamics.
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

    // ---- backstop berm at the far end ---------------------------------------
    // Angled face, so rounds that get past a target bury themselves instead of
    // skipping back down the range.
    addBox({0, 2.2f, kRangeFarZ + 2.f}, {kArena * 2, 4.4f, 1.2f}, matDark);
    addBox({0, 1.1f, kRangeFarZ}, {kArena * 2, 3.2f, 0.6f}, matAccent, 0.f, -0.5f);
    addTrim({0, 4.45f, kRangeFarZ + 2.f}, {kArena * 2, 0.1f, 1.3f}, matGlowOrange);

    // ---- firing line: a bench you shoot over, and the lane it stands in -----
    addBox({0, 0.45f, kFiringLineZ}, {kLaneWidth * kLaneCount + 2.f, 0.9f, 0.5f}, matDark);
    addTrim({0, 0.92f, kFiringLineZ}, {kLaneWidth * kLaneCount + 2.f, 0.06f, 0.55f}, matGlowCyan);

    // ---- lane dividers + distance markers ------------------------------------
    // Low walls between lanes, and an orange kerb every 10 m with an emissive
    // strip on top: the range reads as measured rather than as an empty room.
    for (int i = 0; i <= static_cast<int>(kLaneCount); ++i) {
        const float x = (static_cast<float>(i) - kLaneCount * 0.5f) * kLaneWidth;
        addBox({x, 0.35f, (kFiringLineZ + kRangeFarZ) * 0.5f}, {0.35f, 0.7f, kRangeFarZ - kFiringLineZ},
               matWall);
    }
    for (int mark = 0; mark < 3; ++mark) {
        const float z = kRangeNearZ + static_cast<float>(mark) * 10.f;
        addTrim({0, 0.02f, z}, {kLaneWidth * kLaneCount, 0.04f, 0.18f}, matAccent);
        addTrim({0, 0.03f, z}, {kLaneWidth * kLaneCount, 0.05f, 0.05f}, matGlowOrange);
    }

    // ---- targets --------------------------------------------------------------
    // Everything below is a REAL dynamic body. Nothing scripts a knock-over:
    // the plates stand because a box on a flat post is stable, and they fall
    // because a bullet impulse at the contact point tips them. PhysxWorld::add
    // cooks Box / Sphere / Capsule, which is why the props are boxes and balls
    // rather than the drums you might expect.
    auto matPlate = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(0xf2f4f7).roughness(0.35f).metalness(0.55f).map(gridTex));

    // One steel plate standing on a static post.
    auto addPlate = [&](float x, float z, float plateW, float plateH, float postH) {
        addBox({x, postH * 0.5f, z}, {plateW * 0.75f, postH, 0.35f}, matDark);// post
        auto geo = BoxGeometry::create(plateW, plateH, 0.07f);
        worldUvBox(*geo, plateW, plateH, 0.07f);
        auto m = Mesh::create(geo, matPlate);
        m->position.set(x, postH + plateH * 0.5f + 0.005f, z);
        m->castShadow = true;
        m->receiveShadow = true;
        scene.add(m);
        // Light, so a single round actually moves it.
        auto* body = world.add(*m, 26.f);
        lvl.actorToMesh[body] = m.get();
        Dynamic d;
        d.mesh = m;
        d.body = body;
        d.home = m->position.clone();
        d.homeRot = m->quaternion.clone();
        d.kind = Dynamic::Kind::Plate;
        lvl.dynamics.push_back(std::move(d));
    };

    auto addProp = [&](const std::shared_ptr<Mesh>& m, float density) {
        m->castShadow = true;
        m->receiveShadow = true;
        scene.add(m);
        auto* body = world.add(*m, density);
        lvl.actorToMesh[body] = m.get();
        Dynamic d;
        d.mesh = m;
        d.body = body;
        d.home = m->position.clone();
        d.homeRot = m->quaternion.clone();
        d.kind = Dynamic::Kind::Prop;
        lvl.dynamics.push_back(std::move(d));
    };

    auto spawnCrateStack = [&](float cx, float cz, int height, float side) {
        auto geo = BoxGeometry::create(side, side, side);
        worldUvBox(*geo, side, side, side);
        for (int y = 0; y < height; ++y) {
            auto m = Mesh::create(geo, matCrate);
            m->position.set(cx + frand(-0.04f, 0.04f), side * 0.5f + y * (side + 0.002f),
                            cz + frand(-0.04f, 0.04f));
            addProp(m, 55.f);
        }
    };

    // Three lanes, each with a plate rack at 10 / 20 / 30 m and cover to knock
    // about. The lanes differ so the range is not three copies of one idea.
    for (int lane = 0; lane < static_cast<int>(kLaneCount); ++lane) {
        const float x = (static_cast<float>(lane) - (kLaneCount - 1) * 0.5f) * kLaneWidth;
        // near / mid / far plates, smaller the further out
        addPlate(x - 1.7f, kRangeNearZ, 0.8f, 0.95f, 0.75f);
        addPlate(x, kRangeNearZ, 0.8f, 0.95f, 0.75f);
        addPlate(x + 1.7f, kRangeNearZ, 0.8f, 0.95f, 0.75f);
        addPlate(x, kRangeNearZ + 9.f, 0.7f, 0.8f, 1.05f);
        addPlate(x - 1.9f, kRangeNearZ + 9.f, 0.55f, 0.65f, 1.35f);
        addPlate(x + 1.9f, kRangeNearZ + 9.f, 0.55f, 0.65f, 1.35f);
        addPlate(x, kRangeNearZ + 18.f, 0.5f, 0.6f, 1.5f);

        // cover: a crate stack per lane, and a ball that rolls when clipped
        spawnCrateStack(x - 2.2f, kRangeNearZ + 4.5f, lane == 1 ? 3 : 2, 0.85f);
        auto ball = Mesh::create(SphereGeometry::create(0.32f, 20, 14), matAccent);
        ball->position.set(x + 2.4f, 0.32f, kRangeNearZ + 4.f);
        addProp(ball, 90.f);
    }

    // A couple of stacks off to the sides, out of the lanes, for stray rounds.
    spawnCrateStack(-kArena + 5.f, 0.f, 2, 1.1f);
    spawnCrateStack(kArena - 5.f, 0.f, 2, 1.1f);

    return lvl;
}
