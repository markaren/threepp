
#include "threepp/extras/editor/CharacterConfig.hpp"

#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/animation/AnimationClip.hpp"
#include "threepp/animation/KeyframeTrack.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/objects/SkinnedMesh.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string_view>

using namespace threepp;
using namespace threepp::editor;
using namespace threepp::editor::codec;

namespace {

    // A clip whose root bone travels slower than this over its whole length is
    // treated as standing still: idle, jump and the turn-on-the-spot clips all
    // measure a centimetre or two of drift, which is noise, not a direction.
    constexpr float kStillSpeed = 0.15f;

    // Early-vs-late speed ratio above which a clip is a TRANSITION rather than
    // a cycle. A locomotion cycle travels at one speed throughout; a "start
    // walking" clip begins at a standstill and ends at walking pace, so its
    // average speed is a number no controller should ever hold. Excluding it
    // by measurement rather than by name is what keeps this working on a pack
    // that spells its files differently.
    constexpr float kCycleRatio = 2.5f;

    std::string lowered(std::string text) {

        std::transform(text.begin(), text.end(), text.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return text;
    }

    bool mentions(const std::string& haystack, const char* needle) {

        return lowered(haystack).find(needle) != std::string::npos;
    }

    CharacterConfig::Facing facingFrom(std::string_view text, CharacterConfig::Facing fallback) {

        if (text == "camera") return CharacterConfig::Facing::Camera;
        if (text == "movement") return CharacterConfig::Facing::Movement;
        return fallback;
    }

    const char* facingToken(CharacterConfig::Facing facing) {

        switch (facing) {
            case CharacterConfig::Facing::Camera: return "camera";
            case CharacterConfig::Facing::Movement: return "movement";
        }
        return "camera";
    }

    // Where a mesh's geometry actually lands in world space.
    //
    // For an ordinary Mesh that is its world matrix. For a SKINNED one it is
    // not, and the difference is the whole reason this helper exists: glTF
    // says a skinned mesh's own node transform MUST be ignored (the joints
    // place every vertex), and threepp honours that — SkinnedMesh keeps
    // bindMatrixInverse equal to the inverse of its world matrix, so the two
    // cancel and the vertices land wherever the BONES put them. Measuring
    // through matrixWorld therefore measures a frame nothing is drawn in. On a
    // Mixamo rig, whose armature node carries the 0.01 unit scale, that reads a
    // 1.8 m character as 1.8 cm — and sizes its capsule to match.
    //
    // The transform the vertices really take is bone[0].matrixWorld *
    // boneInverse[0] (the bind-space-to-world map, exact while the rig is at
    // rest — which is when a model is measured) composed with bindMatrix,
    // which takes the geometry into bind space.
    Matrix4 geometryToWorld(Mesh& mesh) {

        auto* skinned = mesh.as<SkinnedMesh>();
        if (!skinned || !skinned->skeleton || skinned->skeleton->bones.empty() ||
            skinned->skeleton->boneInverses.empty()) {
            return Matrix4(*mesh.matrixWorld);
        }
        auto& bone = skinned->skeleton->bones.front();
        if (!bone) return Matrix4(*mesh.matrixWorld);
        bone->updateWorldMatrix(true, false);
        Matrix4 toWorld(*bone->matrixWorld);
        toWorld.multiply(skinned->skeleton->boneInverses.front());
        toWorld.multiply(skinned->bindMatrix);
        return toWorld;
    }

    // Union of every mesh's geometry bounds under `node`, mapped through
    // `intoFrame`. Same helper VehicleConfig uses, and for the same reason: a
    // character standing at an angle in the scene must measure the same as one
    // on the world axes.
    Box3 boundsInFrame(Object3D& node, const Matrix4& intoFrame) {

        Box3 bounds;
        bounds.makeEmpty();
        node.traverseType<Mesh>([&](Mesh& mesh) {
            const auto geometry = mesh.geometry();
            if (!geometry) return;
            if (!geometry->boundingBox) geometry->computeBoundingBox();
            if (!geometry->boundingBox) return;
            Matrix4 rel(intoFrame);
            rel.multiply(geometryToWorld(mesh));
            Box3 b = *geometry->boundingBox;
            b.applyMatrix4(rel);
            bounds.union_(b);
        });
        return bounds;
    }

    // The bone a locomotion clip moves the whole body with. "hips" by name
    // first (every mixamorig rig, and most others, spell it that way), else
    // the topmost Bone in the subtree — which is the same node for any rig
    // that has one root.
    Object3D* findRootBone(Object3D& root) {

        Object3D* named = nullptr;
        Object3D* topmost = nullptr;
        root.traverse([&](Object3D& node) {
            if (!node.as<Bone>()) return;
            if (!named && mentions(node.name, "hips")) named = &node;
            if (topmost) return;
            // Topmost = a Bone whose ancestors up to the root hold no Bone.
            for (const Object3D* p = node.parent; p && p != &root; p = p->parent) {
                if (p->as<Bone>()) return;
            }
            topmost = &node;
        });
        return named ? named : topmost;
    }

    // The root bone's position track in `clip`, or nullptr.
    const KeyframeTrack* positionTrack(const AnimationClip& clip, const std::string& boneName) {

        const std::string wanted = boneName + ".position";
        for (const auto& track : clip.getTracks()) {
            if (!track) continue;
            if (track->getName() == wanted) return track.get();
        }
        return nullptr;
    }

    // Linear sample of a 3-component track at `time`. Tracks arriving from
    // glTF are force-sampled and dense, so linear is exact enough for a
    // displacement measurement; a cubic sampler here would buy nothing.
    Vector3 sampleVec3(const KeyframeTrack& track, float time) {

        const auto& times = track.getTimes();
        const auto& values = track.getValues();
        if (times.empty() || values.size() < 3) return {};

        const auto at = [&](std::size_t i) {
            const std::size_t o = i * 3;
            return o + 2 < values.size() ? Vector3(values[o], values[o + 1], values[o + 2])
                                         : Vector3();
        };

        if (time <= times.front()) return at(0);
        if (time >= times.back()) return at(times.size() - 1);

        const auto upper = std::upper_bound(times.begin(), times.end(), time);
        const auto hi = static_cast<std::size_t>(upper - times.begin());
        const std::size_t lo = hi - 1;
        const float span = times[hi] - times[lo];
        const float t = span > 0.f ? (time - times[lo]) / span : 0.f;
        Vector3 a = at(lo);
        const Vector3 b = at(hi);
        return a.lerp(b, t);
    }

    // What one clip does, in the character's own frame.
    struct Travel {
        bool measured = false;
        float speed = 0.f;   // m/s, horizontal
        float angle = 0.f;   // radians from forward, positive towards the model's LEFT
        bool cycle = false;  // steady speed throughout (see kCycleRatio)
        float duration = 0.f;
    };

    Travel measureTravel(const AnimationClip& clip, const std::string& boneName,
                         const Matrix4& boneParentBasis,
                         const Vector3& forward, const Vector3& left) {

        Travel out;
        out.duration = clip.getDuration();
        const auto* track = positionTrack(clip, boneName);
        if (!track || out.duration <= 1e-4f) return out;

        // Local -> world metres. The track's values live in the bone's PARENT
        // frame, and that frame carries the rig's own unit scale (a Mixamo
        // armature is 0.01) — which is exactly why the basis is applied rather
        // than the numbers being read as metres.
        const auto worldDelta = [&](float t0, float t1) {
            Vector3 d = sampleVec3(*track, t1);
            d.sub(sampleVec3(*track, t0));
            d.applyMatrix4(boneParentBasis);
            return d;
        };

        const float T = out.duration;
        const Vector3 total = worldDelta(0.f, T);
        const float fwd = total.dot(forward);
        const float lft = total.dot(left);
        const float horizontal = std::sqrt(fwd * fwd + lft * lft);

        out.measured = true;
        out.speed = horizontal / T;
        out.angle = std::atan2(lft, fwd);

        // Steady, or ramping? Compare the first quarter with the last.
        const Vector3 early = worldDelta(0.f, 0.25f * T);
        const Vector3 late = worldDelta(0.75f * T, T);
        const auto flatLen = [&](const Vector3& v) {
            const float f = v.dot(forward), l = v.dot(left);
            return std::sqrt(f * f + l * l);
        };
        const float ve = flatLen(early), vl = flatLen(late);
        const float lo = std::min(ve, vl), hi = std::max(ve, vl);
        out.cycle = lo > 1e-4f && hi <= lo * kCycleRatio;
        return out;
    }

}// namespace


std::size_t CharacterGeometry::resolvedCount() const {

    std::size_t n = 0;
    for (const auto& slot : gaits) {
        if (slot.clip) ++n;
    }
    return n;
}

const char* CharacterConfig::label(Facing facing) {

    switch (facing) {
        case Facing::Camera: return "Camera";
        case Facing::Movement: return "Movement";
    }
    return "Camera";
}

std::string CharacterConfig::encode() const {

    std::string out;
    out += "facing=";
    out += facingToken(facing);
    // Every field rides along whatever the flags say, so flipping auto and
    // back does not quietly reset the ones the other mode hides — the rule
    // PhysicsConfig, JointConfig and VehicleConfig all follow.
    out += ";autogeom=";
    out += (autoGeometry ? "1" : "0");
    out += ";height=";
    out += number(height);
    out += ";radius=";
    out += number(radius);
    out += ";autospeeds=";
    out += (autoSpeeds ? "1" : "0");
    out += ";walkspeed=";
    out += number(walkSpeed);
    out += ";runspeed=";
    out += number(runSpeed);
    out += ";mass=";
    out += number(mass);
    out += ";jump=";
    out += number(jumpHeight);
    out += ";gravity=";
    out += number(gravity);
    out += ";step=";
    out += number(stepOffset);
    out += ";slope=";
    out += number(slopeLimit);
    out += ";turnrate=";
    out += number(turnRate);
    out += ";accel=";
    out += number(accel);
    out += ";blend=";
    out += number(blendTime);
    return out;
}

std::optional<CharacterConfig> CharacterConfig::decode(const std::string& text) {

    if (text.empty()) return std::nullopt;

    CharacterConfig config;

    codec::parsePairs(text, [&](std::string_view key, std::string_view value) {
        if (key == "facing") {
            config.facing = facingFrom(value, config.facing);
        } else if (key == "autogeom") {
            config.autoGeometry = toBool(value, config.autoGeometry);
        } else if (key == "height") {
            config.height = toFloat(value, config.height);
        } else if (key == "radius") {
            config.radius = toFloat(value, config.radius);
        } else if (key == "autospeeds") {
            config.autoSpeeds = toBool(value, config.autoSpeeds);
        } else if (key == "walkspeed") {
            config.walkSpeed = toFloat(value, config.walkSpeed);
        } else if (key == "runspeed") {
            config.runSpeed = toFloat(value, config.runSpeed);
        } else if (key == "mass") {
            config.mass = toFloat(value, config.mass);
        } else if (key == "jump") {
            config.jumpHeight = toFloat(value, config.jumpHeight);
        } else if (key == "gravity") {
            config.gravity = toFloat(value, config.gravity);
        } else if (key == "step") {
            config.stepOffset = toFloat(value, config.stepOffset);
        } else if (key == "slope") {
            config.slopeLimit = toFloat(value, config.slopeLimit);
        } else if (key == "turnrate") {
            config.turnRate = toFloat(value, config.turnRate);
        } else if (key == "accel") {
            config.accel = toFloat(value, config.accel);
        } else if (key == "blend") {
            config.blendTime = toFloat(value, config.blendTime);
        }
        // Unknown keys ignored on purpose: a document written by a newer
        // editor still loads here.
    });

    return config;
}

std::optional<CharacterConfig> CharacterConfig::read(const Object3D& object) {

    auto config = readEntry<CharacterConfig>(object, userDataKey);
    if (config) {
        for (std::size_t i = 0; i < kGaitCount; ++i) {
            config->clips[i] = readString(object, clipKeys[i]);
        }
    }
    return config;
}

bool CharacterConfig::isCharacter(const Object3D& object) {

    return hasEntry(object, userDataKey);
}

void CharacterConfig::write(Object3D& object) const {

    object.userData[userDataKey] = encode();
    for (std::size_t i = 0; i < kGaitCount; ++i) {
        if (clips[i].empty()) {
            object.userData.erase(clipKeys[i]);
        } else {
            object.userData[clipKeys[i]] = clips[i];
        }
    }
}

void CharacterConfig::erase(Object3D& object) {

    object.userData.erase(userDataKey);
    for (const auto* key : clipKeys) {
        object.userData.erase(key);
    }
}

CharacterGeometry CharacterConfig::derived(Object3D& root) const {

    return derived(root, root.animations);
}

CharacterGeometry CharacterConfig::derived(
        Object3D& root, const std::vector<std::shared_ptr<AnimationClip>>& clipList) const {

    CharacterGeometry geo;

    root.updateWorldMatrix(true, true);

    // --- the model's own frame ---------------------------------------------
    // +Z forward, +X left, +Y up: three.js's convention, the one every
    // atan2(x, z) yaw in this codebase already assumes, and the one a Mixamo
    // character lands in after the exporter's Y-up conversion.
    Vector3 rootPos, rootScale;
    Quaternion rootRot;
    root.matrixWorld->decompose(rootPos, rootRot, rootScale);

    Vector3 forward(0.f, 0.f, 1.f);
    forward.applyQuaternion(rootRot).normalize();
    Vector3 left(1.f, 0.f, 0.f);
    left.applyQuaternion(rootRot).normalize();

    Matrix4 frame;
    frame.compose(rootPos, rootRot, Vector3(1.f, 1.f, 1.f));
    Matrix4 frameInv(frame);
    frameInv.invert();

    // --- measure the capsule ------------------------------------------------
    const Box3 bounds = boundsInFrame(root, frameInv);
    if (bounds.isEmpty()) {
        geo.problem = "the model has no geometry to measure";
        return geo;
    }
    const Vector3 size = bounds.getSize();
    const Vector3 centre = bounds.getCenter();

    geo.height = std::max(size.y, 0.01f);
    // Half-DEPTH, floored at human proportion. Half-width would be an arm
    // span: a skinned character binds in a T-pose, so its X extent says
    // nothing about how wide a corridor it fits through.
    geo.radius = std::clamp(std::max(0.5f * size.z, 0.17f * geo.height),
                            0.02f, 0.45f * geo.height);

    Vector3 feetLocal(centre.x, bounds.min().y, centre.z);
    feetLocal.applyMatrix4(frame);
    geo.feet = feetLocal;

    // --- the root-motion bone ----------------------------------------------
    geo.rootBone = findRootBone(root);
    if (geo.rootBone) {
        geo.rootBoneBind = geo.rootBone->position;
    }

    geo.valid = true;
    if (!geo.rootBone || clipList.empty()) {
        // Still a valid character — a capsule with no clips walks around
        // T-posed, which is a better first Play than refusing to start.
        if (clipList.empty()) geo.problem = "the model carries no animation clips";
        else geo.problem = "the model has no skeleton, so its clips cannot be classified";
        return geo;
    }

    // The basis that turns a track value into world metres (see measureTravel).
    Matrix4 boneParentBasis;
    if (auto* parent = geo.rootBone->parent) {
        parent->updateWorldMatrix(true, false);
        boneParentBasis.copy(*parent->matrixWorld);
    }
    boneParentBasis.elements[12] = 0.f;
    boneParentBasis.elements[13] = 0.f;
    boneParentBasis.elements[14] = 0.f;

    // --- classify every clip by what it DOES --------------------------------
    struct Candidate {
        std::shared_ptr<AnimationClip> clip;
        Travel travel;
    };
    std::vector<Candidate> moving;   // steady-speed locomotion cycles
    std::vector<Candidate> standing; // idle / jump / turn on the spot

    for (const auto& clip : clipList) {
        if (!clip) continue;
        Candidate candidate{clip, measureTravel(*clip, geo.rootBone->name, boneParentBasis,
                                                forward, left)};
        if (!candidate.travel.measured) continue;
        if (candidate.travel.speed < kStillSpeed) {
            standing.push_back(std::move(candidate));
        } else if (candidate.travel.cycle) {
            moving.push_back(std::move(candidate));
        }
        // A clip that travels but ramps (a "start walking") is deliberately
        // dropped: its average speed is one no controller ever holds.
    }

    // Direction buckets, in the model's own frame.
    constexpr float kQuarter = 0.7853982f;      // 45 degrees
    constexpr float kThreeQuarters = 2.3561945f;// 135 degrees
    std::vector<Candidate> ahead, behind, portside, starboard;
    for (auto& candidate : moving) {
        const float a = candidate.travel.angle;
        const float mag = std::abs(a);
        if (mag <= kQuarter) ahead.push_back(candidate);
        else if (mag >= kThreeQuarters) behind.push_back(candidate);
        else if (a > 0.f) portside.push_back(candidate);
        else starboard.push_back(candidate);
    }

    // Slowest of a direction is its walk, fastest is its run. One clip fills
    // the slow slot only — the session time-scales it rather than pretending
    // a sprint exists.
    const auto fill = [&](std::vector<Candidate>& bucket, Gait slow, Gait fast) {
        if (bucket.empty()) return;
        std::sort(bucket.begin(), bucket.end(), [](const Candidate& a, const Candidate& b) {
            return a.travel.speed < b.travel.speed;
        });
        geo.gaits[gaitIndex(slow)] = {bucket.front().clip, bucket.front().travel.speed};
        if (bucket.size() > 1) {
            geo.gaits[gaitIndex(fast)] = {bucket.back().clip, bucket.back().travel.speed};
        }
    };
    fill(ahead, Gait::Walk, Gait::Run);
    fill(behind, Gait::WalkBack, Gait::RunBack);
    fill(portside, Gait::StrafeLeft, Gait::StrafeLeftFast);
    fill(starboard, Gait::StrafeRight, Gait::StrafeRightFast);

    // Idle and Jump ARE picked by name: standing still and jumping on the spot
    // look identical to a ruler.
    for (const auto& candidate : standing) {
        if (mentions(candidate.clip->name(), "jump") && !geo.slot(Gait::Jump).clip) {
            geo.gaits[gaitIndex(Gait::Jump)] = {candidate.clip, 0.f};
        }
        if (mentions(candidate.clip->name(), "idle") && !geo.slot(Gait::Idle).clip) {
            geo.gaits[gaitIndex(Gait::Idle)] = {candidate.clip, 0.f};
        }
    }
    if (!geo.slot(Gait::Idle).clip) {
        // No clip says "idle": take the LONGEST one that stands still and is
        // not the jump. A rig's idle is nearly always its longest loop.
        const AnimationClip* jump = geo.slot(Gait::Jump).clip.get();
        const Candidate* best = nullptr;
        for (const auto& candidate : standing) {
            if (candidate.clip.get() == jump) continue;
            if (!best || candidate.travel.duration > best->travel.duration) best = &candidate;
        }
        if (best) geo.gaits[gaitIndex(Gait::Idle)] = {best->clip, 0.f};
    }

    // --- explicit overrides win --------------------------------------------
    for (std::size_t i = 0; i < kGaitCount; ++i) {
        if (clips[i].empty()) continue;
        auto picked = AnimationClip::findByName(clipList, clips[i]);
        if (!picked) {
            // Named but absent: leave the slot as the auto-match found it and
            // say so, rather than silently playing something else.
            if (!geo.problem.empty()) geo.problem += "; ";
            geo.problem += std::string(gaitLabels[i]) + " clip \"" + clips[i] + "\" is not in this model";
            continue;
        }
        const Travel travel = measureTravel(*picked, geo.rootBone->name, boneParentBasis,
                                            forward, left);
        geo.gaits[i] = {picked, travel.measured ? travel.speed : 0.f};
    }

    return geo;
}
