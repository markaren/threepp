// ============================================================================
//  FPS demo — skinned humanoid ragdoll
//  Included inside namespace {} in main.cpp — not a standalone header.
//  Requires: PhysX + threepp types, PhysxWorld helpers (toPxVec3, fromPxQuat...)
//
//  On death the animated skeleton is handed to physics: capsule bodies are
//  spawned over the major bones (pelvis, chest, head, upper/lower limbs) and
//  tied together with ANATOMICAL joints, then each frame the bones are written
//  back FROM the simulated bodies — so the actual skinned character (not
//  stand-in prop geometry) crumples, tumbles and drapes over the level.
//
//  What makes it read as a body and not a rag (all learned the hard way):
//    * Elbows and knees are HINGES, not cones. A knee that can fold sideways
//      or a forearm that can spin around its own axis is the single loudest
//      "cheap ragdoll" tell. The hinge axis and its flexion range come from
//      the CURRENT animated pose (n = upper x lower), so the limb can only
//      straighten back out and bend further the way it already bends.
//    * The torso is two bodies (pelvis + chest) with a stiff spine joint, so
//      the corpse folds at the waist instead of falling as one plank.
//    * Every joint runs a pure-damping SLERP drive — "muscle tone". Without it
//      limbs windmill freely and the whole thing looks like a bag of sticks.
//    * Self-collision is off (one PxAggregate). Crude capsules overlap in the
//      bind pose — shoulders inside the chest, thighs against each other — and
//      solving those overlaps is what made the old rig kick itself apart on
//      the first frame.
//    * ONE impulse, at the wound, via addForceAtPos. The old rig punched the
//      pelvis and then added random jitter to every limb: that jitter WAS the
//      unnatural flail.
// ============================================================================

Matrix4 pxToMat4(const PxTransform& t) {
    Matrix4 m;
    m.compose(fromPxVec3(t.p), fromPxQuat(t.q), Vector3(1.f, 1.f, 1.f));
    return m;
}

struct SkinnedRagdollPart {
    Object3D* bone = nullptr;
    PxRigidDynamic* body = nullptr;
    Matrix4 boneFromBody;// bone-world = body-world * boneFromBody (captured at build)
    int depth = 0;       // scene-graph depth; drive order = parents first
};

struct SkinnedRagdoll {
    std::vector<SkinnedRagdollPart> parts;
    std::vector<PxJoint*> joints;
    // All parts live in one self-collision-disabled aggregate (see header note).
    PxAggregate* aggregate = nullptr;
    PxMaterial* material = nullptr;// dead flesh: grippy, zero restitution

    [[nodiscard]] bool valid() const { return !parts.empty(); }

    // Case-insensitive bone lookup by name SUFFIX with separators stripped, so
    // "mixamorig:LeftForeArm" resolves for "leftforearm" but NOT for "leftarm".
    static Object3D* findBone(Object3D& root, const std::string& suffix) {
        Object3D* out = nullptr;
        root.traverse([&](Object3D& o) {
            if (out || o.name.empty()) return;
            std::string n = o.name;
            std::transform(n.begin(), n.end(), n.begin(), ::tolower);
            n.erase(std::remove_if(n.begin(), n.end(),
                                   [](unsigned char c) { return c == ':' || c == '_' || c == '.' || c == ' '; }),
                    n.end());
            if (n.size() >= suffix.size() &&
                n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0) out = &o;
        });
        return out;
    }

    // First bone matching any of `suffixes` (rig-naming fallback chain).
    static Object3D* findAny(Object3D& root, std::initializer_list<const char*> suffixes) {
        for (const char* s : suffixes)
            if (auto* b = findBone(root, s)) return b;
        return nullptr;
    }

    // Build the physics rig over `model`'s current pose. The corpse inherits
    // `inheritVel`; `impulse` is applied at `hitPoint` (the wound).
    void build(PhysxWorld& world, Object3D& model, const Vector3& inheritVel,
               const Vector3& hitPoint, const Vector3& impulse) {
        model.updateMatrixWorld(true);

        auto wpos = [](Object3D* o) {
            Vector3 p;
            p.setFromMatrixPosition(*o->matrixWorld);
            return p;
        };
        auto depthOf = [](Object3D* o) {
            int d = 0;
            for (auto* p = o->parent; p; p = p->parent) ++d;
            return d;
        };

        Object3D* hips = findBone(model, "hips");
        Object3D* spine = findAny(model, {"spine1", "spine2", "spine", "chest"});
        Object3D* neck = findBone(model, "neck");
        Object3D* head = findBone(model, "head");
        if (!neck) neck = head;
        if (!hips || !neck) return;// not a humanoid we understand — no ragdoll
        if (spine == hips) spine = nullptr;

        const Vector3 hipsP = wpos(hips), neckP = wpos(neck);

        // ---- character frame -------------------------------------------------
        // up from the spine, side from the shoulders (hips as fallback), and
        // forward completing a RIGHT-handed triple (side x up = forward) with
        // side = left - right. That handedness is what makes the fallback hinge
        // axes below bend a knee backwards and an elbow forwards regardless of
        // which way the rig happens to face.
        Vector3 up = neckP - hipsP;
        if (up.length() < 1e-4f) return;
        up.normalize();
        Vector3 side(1.f, 0.f, 0.f);
        {
            Object3D* l = findAny(model, {"leftarm", "leftupleg"});
            Object3D* r = findAny(model, {"rightarm", "rightupleg"});
            if (l && r) side = wpos(l) - wpos(r);
            side.addScaledVector(up, -side.dot(up));// orthogonalise against up
            if (side.length() < 1e-4f) side.set(1.f, 0.f, 0.f);
            side.normalize();
        }
        Vector3 forward;
        forward.crossVectors(side, up);
        forward.normalize();

        PxPhysics& phys = world.physics();
        material = phys.createMaterial(0.9f, 0.85f, 0.f);// no bounce: a bouncing corpse reads as rubber
        constexpr uint32_t kMaxParts = 12;
        aggregate = phys.createAggregate(kMaxParts, kMaxParts,
                                        PxGetAggregateFilterHint(PxAggregateType::eGENERIC, false));

        // Capsule body along from->to, NOT yet in the scene (aggregate members
        // must be inserted with the aggregate). The driven bone snapshots its
        // offset from the body now, while the pose is still the animated one.
        auto addCapsule = [&](Object3D* bone, const Vector3& from, const Vector3& to,
                              float r, float density) -> PxRigidDynamic* {
            Vector3 dir = to - from;
            float len = dir.length();
            if (len < 0.05f) len = 0.05f;
            dir.normalize();
            const float hh = std::max(0.008f, len * 0.5f - r);
            Quaternion q;
            q.setFromUnitVectors(Vector3(1, 0, 0), dir);// PhysX capsule axis is +X
            const Vector3 mid = from + dir * (len * 0.5f);

            PxRigidDynamic* body = phys.createRigidDynamic(PxTransform(toPxVec3(mid), toPxQuat(q)));
            PxShape* sh = phys.createShape(PxCapsuleGeometry(r, hh), *material, true);
            body->attachShape(*sh);
            sh->release();
            PxRigidBodyExt::updateMassAndInertia(*body, density);
            body->setLinearDamping(0.06f);
            body->setAngularDamping(0.35f);
            body->setMaxAngularVelocity(14.f);   // no windmilling limbs
            body->setMaxDepenetrationVelocity(2.5f);// spawned inside the floor must not pop
            body->setSolverIterationCounts(16, 4);// jointed chains need the iterations
            body->setSleepThreshold(0.06f);       // corpses settle instead of simmering
            body->setStabilizationThreshold(0.02f);
            body->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);
            body->setLinearVelocity(toPxVec3(inheritVel));
            aggregate->addActor(*body);

            SkinnedRagdollPart part;
            part.bone = bone;
            part.body = body;
            Matrix4 inv = pxToMat4(body->getGlobalPose());
            inv.invert();
            part.boneFromBody.multiplyMatrices(inv, *bone->matrixWorld);
            part.depth = depthOf(bone);
            parts.push_back(part);
            return body;
        };

        // ---- torso + head ----------------------------------------------------
        // Densities: limbs run heavier than water so the mass ratio to the
        // torso stays inside what the solver keeps stable (a 1 kg forearm on a
        // 17 kg chest jitters at every joint).
        const Vector3 spineP = spine ? wpos(spine) : hipsP + (neckP - hipsP) * 0.45f;
        auto* pelvis = addCapsule(hips, hipsP, spineP, 0.14f, 1000.f);
        auto* chest = spine ? addCapsule(spine, spineP, neckP, 0.15f, 1000.f) : pelvis;

        // ---- joint construction helpers --------------------------------------
        // D6 local frames: +X is the twist axis, Y/Z the swing axes. Both
        // actors get the SAME world anchor frame, so every limit is measured
        // relative to the pose the character died in.
        auto frameQuat = [](Vector3 x) {
            x.normalize();
            Vector3 ref = std::abs(x.y) < 0.9f ? Vector3(0, 1, 0) : Vector3(1, 0, 0);
            Vector3 z;
            z.crossVectors(x, ref);
            z.normalize();
            Vector3 y;
            y.crossVectors(z, x);
            y.normalize();
            Matrix4 m;
            m.makeBasis(x, y, z);
            Quaternion q;
            q.setFromRotationMatrix(m);
            return q;
        };
        auto makeD6 = [&](PxRigidDynamic* a, PxRigidDynamic* b,
                          const Vector3& anchorW, const Vector3& twistAxisW) {
            const PxTransform anchor(toPxVec3(anchorW), toPxQuat(frameQuat(twistAxisW)));
            auto* j = PxD6JointCreate(phys,
                                      a, a->getGlobalPose().transformInv(anchor),
                                      b, b->getGlobalPose().transformInv(anchor));
            // Pure-damping SLERP drive (stiffness 0, isAcceleration) = muscle
            // tone. PhysX recommends SLERP over separate swing/twist drives.
            j->setDrive(PxD6Drive::eSLERP, PxD6JointDrive(0.f, 1.f, PX_MAX_F32, true));
            joints.push_back(j);
            return j;
        };
        // Ball socket: symmetric cone off the current bone direction + limited
        // twist about it. Default limit restitution is 0 — a dead stop, which
        // is what flesh does.
        auto ball = [&](PxRigidDynamic* a, PxRigidDynamic* b, const Vector3& anchorW,
                        const Vector3& boneDirW, float swingDeg, float twistDeg, float damping) {
            auto* j = makeD6(a, b, anchorW, boneDirW);
            j->setMotion(PxD6Axis::eSWING1, PxD6Motion::eLIMITED);
            j->setMotion(PxD6Axis::eSWING2, PxD6Motion::eLIMITED);
            j->setMotion(PxD6Axis::eTWIST, PxD6Motion::eLIMITED);
            const float s = math::degToRad(swingDeg), t = math::degToRad(twistDeg);
            j->setSwingLimit(PxJointLimitCone(s, s));
            j->setTwistLimit(PxJointAngularLimitPair(-t, t));
            j->setDrive(PxD6Drive::eSLERP, PxD6JointDrive(0.f, damping, PX_MAX_F32, true));
        };
        // Hinge: one limited axis, the other two LOCKED (the D6 default).
        auto hinge = [&](PxRigidDynamic* a, PxRigidDynamic* b, const Vector3& anchorW,
                         const Vector3& axisW, float loRad, float hiRad, float damping) {
            auto* j = makeD6(a, b, anchorW, axisW);
            j->setMotion(PxD6Axis::eTWIST, PxD6Motion::eLIMITED);
            j->setTwistLimit(PxJointAngularLimitPair(loRad, hiRad));
            j->setDrive(PxD6Drive::eSLERP, PxD6JointDrive(0.f, damping, PX_MAX_F32, true));
        };
        // Hinge axis + flexion range read off the CURRENT pose: n = upper x
        // lower is the anatomical bend axis and +theta deepens the existing
        // bend, so the limb may straighten by theta0 and flex on to maxFlex.
        // A near-straight limb has no usable cross product — fall back to the
        // caller's axis (side for a knee, -side for an elbow).
        auto flexion = [](const Vector3& upperDir, const Vector3& lowerDir,
                          const Vector3& fallbackAxis, float maxFlexDeg,
                          Vector3& axisOut, float& loOut, float& hiOut) {
            Vector3 n;
            n.crossVectors(upperDir, lowerDir);
            const float s = n.length();
            const float theta0 = std::atan2(s, upperDir.dot(lowerDir));
            const float slack = math::degToRad(4.f);
            const float maxFlex = math::degToRad(maxFlexDeg);
            if (s > 0.20f) {// ~12 deg of bend: the cross product is trustworthy
                n.normalize();
                axisOut = n;
                loOut = -(theta0 + slack);
                hiOut = std::max(maxFlex - theta0, math::degToRad(10.f));
            } else {
                axisOut = fallbackAxis;
                loOut = -slack;
                hiOut = maxFlex;
            }
        };

        if (spine) {
            // waist: stiff. This is what lets the corpse fold and slump.
            ball(pelvis, chest, spineP, up, 22.f, 25.f, 30.f);
        }
        if (head) {
            const Vector3 headP = wpos(head);
            auto* hb = addCapsule(head, headP, headP + up * 0.20f, 0.10f, 1000.f);
            ball(chest, hb, headP, up, 38.f, 45.f, 22.f);
        }

        // ---- limbs -----------------------------------------------------------
        struct LimbSpec {
            const char* upper;
            const char* lower;
            const char* end;
            float rU, rL;    // capsule radii
            float density;
            bool arm;        // arm: hangs off the chest, elbow flexes forward
            float ballSwing; // shoulder / hip cone
            float ballTwist;
            float maxFlex;   // elbow / knee flexion limit
        };
        const LimbSpec limbs[] = {
                {"leftarm", "leftforearm", "lefthand", 0.065f, 0.05f, 1400.f, true, 80.f, 50.f, 145.f},
                {"rightarm", "rightforearm", "righthand", 0.065f, 0.05f, 1400.f, true, 80.f, 50.f, 145.f},
                {"leftupleg", "leftleg", "leftfoot", 0.09f, 0.07f, 1100.f, false, 55.f, 32.f, 135.f},
                {"rightupleg", "rightleg", "rightfoot", 0.09f, 0.07f, 1100.f, false, 55.f, 32.f, 135.f},
        };
        for (const auto& L : limbs) {
            Object3D* a = findBone(model, L.upper);
            Object3D* b = findBone(model, L.lower);
            Object3D* c = findBone(model, L.end);
            if (!a || !b || !c) continue;
            const Vector3 pa = wpos(a), pb = wpos(b), pc = wpos(c);
            auto* upper = addCapsule(a, pa, pb, L.rU, L.density);
            auto* lower = addCapsule(b, pb, pc, L.rL, L.density);

            Vector3 upperDir = pb - pa, lowerDir = pc - pb;
            if (upperDir.length() < 1e-4f || lowerDir.length() < 1e-4f) continue;
            upperDir.normalize();
            lowerDir.normalize();

            // shoulder off the chest, hip off the pelvis
            ball(L.arm ? chest : pelvis, upper, pa, upperDir, L.ballSwing, L.ballTwist,
                 L.arm ? 12.f : 16.f);

            // elbow bends forward, knee backward: both are +theta about
            // (arm ? -side : +side) once the frame is right-handed.
            Vector3 fallback = L.arm ? side * -1.f : side;
            Vector3 axis;
            float lo, hi;
            flexion(upperDir, lowerDir, fallback, L.maxFlex, axis, lo, hi);
            hinge(upper, lower, pb, axis, lo, hi, L.arm ? 8.f : 11.f);
        }

        // drive order: parents first, so a child's parent world is current
        std::sort(parts.begin(), parts.end(),
                  [](const SkinnedRagdollPart& x, const SkinnedRagdollPart& y) { return x.depth < y.depth; });

        world.scene().addAggregate(*aggregate);

        // ---- the killing blow ------------------------------------------------
        // One impulse at the wound on the part nearest to it: a shoulder hit
        // spins the corpse, a hip hit drops it, a headshot snaps the neck. No
        // per-limb jitter — that was the flail.
        PxRigidDynamic* wound = pelvis;
        float bestD = 1e9f;
        for (auto& p : parts) {
            const Vector3 c = fromPxVec3(p.body->getGlobalPose().p);
            const float dd = (c - hitPoint).lengthSq();
            if (dd < bestD) {
                bestD = dd;
                wound = p.body;
            }
        }
        PxRigidBodyExt::addForceAtPos(*wound, toPxVec3(impulse), toPxVec3(hitPoint),
                                      PxForceMode::eIMPULSE);
    }

    // After world.step(): write the simulated body poses back into the bones.
    void drive() {
        for (auto& p : parts) {
            if (!p.bone->parent) continue;
            p.bone->parent->updateWorldMatrix(true, false);
            Matrix4 boneW = pxToMat4(p.body->getGlobalPose());
            boneW.multiply(p.boneFromBody);// body-world * offset = bone-world
            Matrix4 invP;
            invP.copy(*p.bone->parent->matrixWorld).invert();
            Matrix4 local;
            local.multiplyMatrices(invP, boneW);
            local.decompose(p.bone->position, p.bone->quaternion, p.bone->scale);
        }
    }

    // joints must be released before the actors they constrain, and the
    // aggregate after the actors have left the scene with it
    void destroy(PhysxWorld& world) {
        for (auto* j : joints) j->release();
        joints.clear();
        if (aggregate) world.scene().removeAggregate(*aggregate, false);
        for (auto& p : parts) p.body->release();
        parts.clear();
        if (aggregate) {
            aggregate->release();
            aggregate = nullptr;
        }
        if (material) {
            material->release();
            material = nullptr;
        }
    }
};
