// ============================================================================
//  FPS demo — skinned humanoid ragdoll
//  Included inside namespace {} in main.cpp — not a standalone header.
//  Requires: PhysX + threepp types, PhysxWorld helpers (toPxVec3, fromPxQuat...)
//
//  On death the animated skeleton is handed to physics: capsule bodies are
//  spawned over the major bones (torso, head, upper/lower limbs) and tied
//  together with cone-limited spherical joints, then each frame the bones are
//  written back FROM the simulated bodies — so the actual skinned character
//  (not stand-in prop geometry) crumples, tumbles and drapes over the level.
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

    // Build the physics rig over `model`'s current pose. The corpse inherits
    // `inheritVel` and the killing shot flings the torso along `impulse`.
    void build(PhysxWorld& world, Object3D& model, const Vector3& inheritVel, const Vector3& impulse) {
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
        Object3D* neck = findBone(model, "neck");
        Object3D* head = findBone(model, "head");
        if (!neck) neck = head;
        if (!hips || !neck) return;// not a humanoid we understand — no ragdoll

        // capsule body along from->to; the driven bone snapshots its offset now
        auto addCapsule = [&](Object3D* bone, const Vector3& from, const Vector3& to, float r) {
            Vector3 dir = to - from;
            float len = dir.length();
            if (len < 0.05f) len = 0.05f;
            dir.normalize();
            const float hh = std::max(0.008f, len * 0.5f - r);
            Quaternion q;
            q.setFromUnitVectors(Vector3(1, 0, 0), dir);// PhysX capsule axis is +X
            const Vector3 mid = from + dir * (len * 0.5f);
            auto* body = world.addDynamic(PxCapsuleGeometry(r, hh),
                                          PxTransform(toPxVec3(mid), toPxQuat(q)), 900.f);
            body->setAngularDamping(0.6f);
            body->setLinearDamping(0.08f);
            body->setSolverIterationCounts(8, 2);// jointed chains need the extra iterations
            body->setLinearVelocity(toPxVec3(inheritVel));

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

        // cone-limited spherical joint at a shared world anchor, +X = cone axis
        auto link = [&](PxRigidDynamic* a, PxRigidDynamic* b, const Vector3& anchorW,
                        Vector3 axisW, float cone) {
            axisW.normalize();
            Quaternion qa;
            qa.setFromUnitVectors(Vector3(1, 0, 0), axisW);
            const PxTransform anchor(toPxVec3(anchorW), toPxQuat(qa));
            auto* j = PxSphericalJointCreate(world.physics(),
                                             a, a->getGlobalPose().transformInv(anchor),
                                             b, b->getGlobalPose().transformInv(anchor));
            j->setLimitCone(PxJointLimitCone(cone, cone));
            j->setSphericalJointFlag(PxSphericalJointFlag::eLIMIT_ENABLED, true);
            joints.push_back(j);
        };

        // ---- torso + head ---------------------------------------------------
        const Vector3 hipsP = wpos(hips), neckP = wpos(neck);
        auto* torso = addCapsule(hips, hipsP, neckP, 0.16f);
        if (head) {
            Vector3 up = neckP - hipsP;
            up.normalize();
            const Vector3 headP = wpos(head);
            auto* hb = addCapsule(head, headP, headP + up * 0.20f, 0.09f);
            link(torso, hb, headP, up, 0.55f);
        }

        // ---- limbs ------------------------------------------------------------
        struct LimbSpec {
            const char* upper;
            const char* lower;
            const char* end;
            float rU, rL;
        };
        const LimbSpec limbs[] = {
                {"leftarm", "leftforearm", "lefthand", 0.055f, 0.045f},
                {"rightarm", "rightforearm", "righthand", 0.055f, 0.045f},
                {"leftupleg", "leftleg", "leftfoot", 0.075f, 0.06f},
                {"rightupleg", "rightleg", "rightfoot", 0.075f, 0.06f},
        };
        for (const auto& L : limbs) {
            Object3D* a = findBone(model, L.upper);
            Object3D* b = findBone(model, L.lower);
            Object3D* c = findBone(model, L.end);
            if (!a || !b || !c) continue;
            const Vector3 pa = wpos(a), pb = wpos(b), pc = wpos(c);
            auto* upper = addCapsule(a, pa, pb, L.rU);
            auto* lower = addCapsule(b, pb, pc, L.rL);
            link(torso, upper, pa, pb - pa, 1.15f);// shoulder / hip
            link(upper, lower, pb, pc - pb, 0.85f);// elbow / knee
        }

        // drive order: parents first, so a child's parent world is current
        std::sort(parts.begin(), parts.end(),
                  [](const SkinnedRagdollPart& x, const SkinnedRagdollPart& y) { return x.depth < y.depth; });

        // the killing blow: fling the torso, jitter the limbs
        torso->addForce(toPxVec3(impulse), PxForceMode::eIMPULSE);
        for (auto& p : parts) {
            if (p.body == torso) continue;
            p.body->addForce(toPxVec3(impulse * 0.12f +
                                      Vector3(frand(-0.5f, 0.5f), frand(0.f, 0.6f), frand(-0.5f, 0.5f))),
                             PxForceMode::eIMPULSE);
        }
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

    // joints must be released before the actors they constrain
    void destroy(PhysxWorld& world) {
        for (auto* j : joints) j->release();
        joints.clear();
        for (auto& p : parts) {
            world.scene().removeActor(*p.body);
            p.body->release();
        }
        parts.clear();
    }
};
