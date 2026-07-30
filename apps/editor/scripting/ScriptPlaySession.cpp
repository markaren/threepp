
#include "Scripting.hpp"

#include "ScriptHost.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/scenes/Scene.hpp"

// fixed_update rides the physics world's substep loop, and the collision and
// trigger callbacks ride its contact and trigger reports, so this session has to
// reach the world the PhysicsPlaySession is playing. Same seam
// bind_editor_physics.cpp uses — the static active() on the session, plus its
// lifetime token — and gated on the same SDK: without PhysX there is no fixed
// clock to run on and nothing to report, and the session says so once instead of
// inventing either.
#ifdef THREEPP_EDITOR_WITH_PHYSX
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#endif

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace threepp;
using namespace threepp::editor;

namespace py = pybind11;

namespace {

    // What a message calls an object. The name if it has one — a uuid in the
    // console tells nobody anything.
    std::string labelOf(const Object3D& object) {

        return object.name.empty() ? ("<" + object.type() + ">") : object.name;
    }

    // What a message calls the code. A file has a name; inline source has
    // nothing but the object it belongs to, which the message already carries.
    std::string originOf(const ScriptConfig& config) {

        if (config.isInline()) return "inline script";
        return std::filesystem::path(config.path).filename().string();
    }

    // A script declares start(self, obj) or start(self, obj, scene); the scene
    // is passed only when asked for, so every existing script keeps working.
    // `start` arrives as a bound method, so `self` is already hidden and the
    // counts here are one positional versus two. *args reads as wanting it.
    bool startWantsScene(const py::object& start) {

        try {
            const auto inspect = py::module_::import("inspect");
            const auto parameters =
                    inspect.attr("signature")(start).attr("parameters").attr("values")();
            std::size_t positional = 0;
            for (auto parameter : parameters) {
                const auto kind = parameter.attr("kind");
                if (kind.equal(parameter.attr("VAR_POSITIONAL"))) return true;
                if (kind.equal(parameter.attr("POSITIONAL_ONLY")) ||
                    kind.equal(parameter.attr("POSITIONAL_OR_KEYWORD"))) {
                    ++positional;
                }
            }
            return positional >= 2;
        } catch (py::error_already_set&) {
            // signature() balks at some exotic callables; the safe reading is
            // the one every script had before this feature existed.
            return false;
        }
    }

}// namespace


struct ScriptPlaySession::Impl {

    struct Instance {
        std::string uuid;
        std::string label;
        // "spinner.py" or "inline script" — what the console calls the code.
        std::string origin;
        // The script object itself. Created and destroyed with the GIL held.
        py::object self;
        // The object the script is attached to. Weak, because a script instance
        // outlives nothing but the session and the scene is not ours: this is
        // only ever used to resolve the body at start().
        std::weak_ptr<Object3D> object;
        // Cached in phase 1 with the rest, so phase 2 is a pure dispatch loop
        // over instances that already exist.
        bool hasStart = false;
        bool hasUpdate = false;
        bool hasStop = false;
        // Cached like the others, and for the same reason: hasattr is a dict
        // lookup down the MRO, and this one would otherwise be paid per SUBSTEP.
        bool hasFixedUpdate = false;
        // Cached for the same reason one step harder: these are consulted from
        // inside PhysX's contact callback, where a dict lookup down the MRO is
        // paid per reported pair per substep.
        bool hasCollisionEnter = false;
        bool hasCollisionExit = false;
        // The trigger half of the same story, read from inside onTrigger().
        bool hasTriggerEnter = false;
        bool hasTriggerExit = false;
        // Set the first time this script raises. It stays set for the rest of
        // the session: a script that throws every frame must not fill the
        // console with the same traceback sixty times a second.
        bool failed = false;
    };

    std::vector<Instance> instances;
    // Kept across stop() so the inspector can still show what went wrong.
    std::unordered_map<std::string, std::string> errors;
    std::function<void(const std::string&)> logger;

    void log(const std::string& message) const {

        if (logger) logger(message);
    }

    // One report, one disable. Never rethrows: the editor keeps playing.
    void fail(Instance& instance, const std::string& what, const std::string& detail) {

        instance.failed = true;
        instance.self = py::object();
        errors[instance.uuid] = detail;
        log("script error in " + instance.origin + " on " + instance.label +
            " (" + what + "): " + detail);
    }

    // Applies the authored overrides to a fresh instance. Fields the document
    // has no value for keep the class attribute, which is what the inspector
    // showed as the default.
    void applyFields(const py::object& cls, const py::object& self, const ScriptConfig& config) {

        const auto discovered = py::cast<py::list>(scripting::exposedFields(cls));
        for (const auto& entry : discovered) {
            const auto tuple = py::cast<py::tuple>(entry);
            const auto name = py::cast<std::string>(tuple[0]);
            const auto stored = config.field(name);
            if (!stored) continue;

            const auto kind = py::cast<std::string>(tuple[1]);
            if (kind == "bool") {
                py::setattr(self, name.c_str(), py::cast(ScriptConfig::toBool(*stored)));
            } else if (kind == "int") {
                py::setattr(self, name.c_str(), py::cast(ScriptConfig::toInt(*stored)));
            } else if (kind == "float") {
                py::setattr(self, name.c_str(), py::cast(ScriptConfig::toFloat(*stored)));
            } else {
                py::setattr(self, name.c_str(), py::cast(*stored));
            }
        }
    }

    // --- one script reaching another ---------------------------------------
    //
    // threepp.editor.script_from_object(obj) hands back the live instance this
    // session is driving for `obj`. There is no event bus and no message type on
    // purpose: what comes back is the actual Python object, so calling a method
    // on it or setting an attribute IS the signalling. Unity's GetComponent,
    // minus the component-type dance — an object carries exactly one script.
    //
    // What makes it usable from start() is the two-phase start below: every
    // instance exists before any start() runs, so which order the scene happens
    // to be in stops mattering.
    //
    // A failed instance answers None rather than a corpse. fail() has already
    // dropped its `self`, and a script disabled halfway through a session is
    // disabled for the rest of it — handing one out would let a neighbour keep
    // driving something the console has already reported as dead.

    [[nodiscard]] py::object resolve(const std::string& uuid) const {

        // Linear, over the tens of scripts a scene has. The instance vector is
        // the authority on what is live; a uuid index beside it would be a
        // second thing to keep in step with fail(), for a lookup the doc tells
        // people to do once in start().
        for (const auto& instance : instances) {
            if (instance.uuid != uuid) continue;
            if (instance.failed) return py::none();
            return instance.self;
        }
        return py::none();
    }

    // Installed between the two phases of start() and taken back at stop, so a
    // resolver can never answer for a session that is not running. The lambda
    // captures `this` and nothing Python-shaped, which is what lets the release
    // below run from a destructor with no interpreter left.
    void installResolver() {

        auto& resolver = scripting::scriptResolver();
        resolver.owner = this;
        resolver.lookup = [this](const std::string& uuid) { return resolve(uuid); };
    }

    void clearResolver() {

        auto& resolver = scripting::scriptResolver();
        // Ours to take back only while it still is ours.
        if (resolver.owner != this) return;
        resolver.owner = nullptr;
        resolver.lookup = nullptr;
    }

    // --- the physics clock -------------------------------------------------
    //
    // fixed_update() is not a frame callback. It runs from inside
    // PhysxWorld::step(), once per fixed substep, with the constant substep dt —
    // which is the whole point: a controller that applies forces or drive
    // targets from update(dt) is frame-rate dependent, and a frame-rate
    // dependent controller is not a controller.
    //
    // ONE callback for the whole session, registered only when at least one live
    // instance defines the method: a scene whose scripts never ask must not pay
    // a per-substep call, nor a GIL acquisition, for the privilege.
    //
    // onPreSubstep, NOT onPostSubstep. Inside a substep PhysxWorld does:
    //     pre callbacks -> simulate -> fetchResults -> post callbacks -> sensors tick
    // so the registered sensors are sampled AFTER both hooks; neither one can see
    // the sample belonging to the substep it sits in. The pre hook of substep k+1
    // stands one tick later than the post hook of substep k, and both feed the
    // same simulate() — so pre is never staler and puts the write immediately
    // before the solve that consumes it, which is the ordering "read the sensor,
    // write the drive target" wants.
    //
    // (What a script's sensor HANDLE reads is one step removed from either: the
    // sensor session drains its rings once per frame, since drain() empties them
    // and there may be only one drainer. So a handle here reports the batch that
    // existed when the frame began, and two substeps of one frame see the same
    // reading. Documented in doc/editor.md rather than worked around — a second
    // drainer would starve the Sensors panel.)

#ifdef THREEPP_EDITOR_WITH_PHYSX
    // The world is BORROWED. Physics is registered first and therefore stops
    // first, so by the time stop() runs here the PhysxWorld is usually already
    // gone — the token (which PhysicsPlaySession drops before the world) is what
    // makes "is my registration still there to remove" answerable without ever
    // dereferencing a dead pointer.
    PhysxWorld* world = nullptr;
    PhysicsPlaySession* physics = nullptr;
    std::weak_ptr<const void> worldLife;
    PhysxWorld::SubstepHandle substep = 0;
    bool registered = false;

    // Take hold of the playing world, if there is one. Both the substep hook and
    // the contact watches want it, and both want the same token guarding it.
    bool bindWorld() {

        if (world) return true;
        // PhysicsPlaySession::start() always builds a world, even for a scene
        // with nothing physical in it — so "a session is playing" is the whole
        // condition. Sessions start in registration order and physics is first,
        // so it is already up by the time we get here.
        auto* session = PhysicsPlaySession::active();
        if (!session || !session->world()) return false;
        physics = session;
        world = session->world();
        worldLife = session->lifetime();
        return true;
    }
#endif

    // Live instances that define fixed_update.
    [[nodiscard]] std::size_t fixedUpdateWanted() const {

        std::size_t wanted = 0;
        for (const auto& instance : instances) {
            if (!instance.failed && instance.hasFixedUpdate) ++wanted;
        }
        return wanted;
    }

    // Hook the session onto the playing world's substep loop, or say once why
    // there is nothing to hook onto. Called at the end of start().
    void attachToPhysics() {

        const auto wanted = fixedUpdateWanted();
        if (wanted == 0) return;

#ifdef THREEPP_EDITOR_WITH_PHYSX
        if (bindWorld()) {
            substep = world->onPreSubstep([this](float dt) { fixedUpdate(dt); });
            registered = true;
            return;
        }
#endif
        // No world, no fixed clock. Faking one — a timer of our own, ticking at
        // some nominal rate — would be a lie about the one thing the name
        // promises, so the method simply never fires and the console says so.
        log("fixed_update needs a playing physics world - it never runs on " +
            std::to_string(wanted) + (wanted == 1 ? " script" : " scripts"));
    }

    // Give the registration back, but only while there is provably something to
    // give it back to. A world that died first took the callback with it, and
    // the handle we are still holding is then just a number.
    void detachFromPhysics() {

#ifdef THREEPP_EDITOR_WITH_PHYSX
        const bool alive = world && !worldLife.expired();
        if (registered && alive) {
            world->removeSubstepCallback(substep);
        }
        for (auto& watch : watches) {
            if (watch.handle && alive) world->unwatchContacts(watch.handle);
            // The actor's contact-report bit is deliberately LEFT SET. It is one
            // bit shared by every watcher of that body, so clearing it here
            // would silence a ContactSensor authored on the same object — the
            // same reasoning ContactSensor::detach() applies in the other
            // direction. It dies with the actor.
        }
        for (auto& watch : triggerWatches) {
            if (watch.handle && alive) world->unwatchTriggers(watch.handle);
            // Nothing to undo on the actor: trigger reporting was never turned
            // ON here. It is the shape's own eTRIGGER_SHAPE flag, authored in
            // the document and cooked by the physics session, and it belongs to
            // the volume rather than to any watcher of it.
        }
        // The queues hold nothing Python-shaped (see QueuedCollision), so this
        // needs no GIL and is safe from stop(), from the destructor, and from a
        // teardown where the interpreter was never started at all.
        watches.clear();
        triggerWatches.clear();
        registered = false;
        world = nullptr;
        physics = nullptr;
        worldLife.reset();
#endif
    }

    // One substep, every live instance that asked for it. Same error semantics
    // as the frame sweep: first raise disables the instance for the session, so
    // a script throwing at 60 Hz reports once and then stops costing anything.
    void fixedUpdate(float dt) {

        bool any = false;
        for (const auto& instance : instances) {
            if (!instance.failed && instance.hasFixedUpdate) any = true;
        }
        if (!any) return;

        // Inside the callback, not around the step: the physics session holds no
        // GIL, and a session whose scripts all failed pays nothing here.
        py::gil_scoped_acquire gil;

        for (auto& instance : instances) {
            if (instance.failed || !instance.hasFixedUpdate) continue;
            try {
                instance.self.attr("fixed_update")(dt);
            } catch (py::error_already_set& e) {
                fail(instance, "fixed_update", scripting::describeError(e));
            } catch (const std::exception& e) {
                fail(instance, "fixed_update", e.what());
            }
        }
    }

    // --- collisions --------------------------------------------------------
    //
    // on_collision_enter / on_collision_exit fire when the body governing a
    // script's object starts and stops touching another body. Three things make
    // that harder than it sounds, and all three are settled here.
    //
    // 1. PhysX reports contacts from inside fetchResults(), i.e. mid-simulate.
    //    Calling into Python there would let a script mutate the scene, or the
    //    world, while the solver still owns it — and would take the GIL inside
    //    the physics step. So the report is COPIED into a queue (nothing but
    //    values and a weak object reference; the manifold PhysX hands over is
    //    valid for the duration of the callback and no longer), and the queue is
    //    delivered from the frame sweep.
    //
    //    The queue is a LIST, never a state flag. A touch that begins and ends
    //    between two deliveries — a fast bounce, or two substeps in one frame —
    //    must still produce enter followed by exit, and collapsing the pair into
    //    "no change" would drop exactly the event a script cares most about.
    //
    // 2. Reporting is opt-in per actor (a bit in the shape's word3), and nobody
    //    ticks a box: start() turns it on for every live instance whose class
    //    defines either method, resolving the actor the same way
    //    rigid_body_from_object does. An object with the callbacks and no body
    //    gets one line and never fires.
    //
    // 3. One touch is one PAIR OF BODIES, not one shape pair. A box lands on the
    //    ground through four manifold points and possibly several shapes; that is
    //    ONE enter. The refcount below is what keeps a LOST on one shape from
    //    reporting an exit the others are still holding — the same bug
    //    ContactSensor's latch was written to avoid, for the same reason.

#ifdef THREEPP_EDITOR_WITH_PHYSX

    // One other body this watch is currently touching, and through how many
    // shape pairs.
    struct Touch {
        ::physx::PxRigidActor* other = nullptr;
        int pairs = 0;
    };

    // One reported edge, copied at report time and waiting for the sweep.
    // Deliberately free of py::object: it is built in PhysX's callback, where
    // this session holds no GIL, and cleared at stop, where there may be no
    // interpreter at all.
    struct QueuedCollision {
        bool enter = false;
        // The far side, as the object the physics was authored on. Weak because
        // the delivery is a sweep later; empty when that actor belongs to
        // nothing a script can see.
        std::weak_ptr<Object3D> other;
        Vector3 point;
        Vector3 normal;
        Vector3 impulse;
    };

    struct CollisionWatch {
        std::size_t instance = 0;// index into instances, stable after start()
        ::physx::PxRigidActor* actor = nullptr;
        PhysxWorld::ContactHandle handle = 0;
        std::vector<Touch> touching;
        std::vector<QueuedCollision> queue;
    };

    std::vector<CollisionWatch> watches;

    // 0 -> 1 for this body? (The caller only reports an enter on the edge.)
    static bool acquireTouch(std::vector<Touch>& touching, ::physx::PxRigidActor* other) {

        for (auto& touch : touching) {
            if (touch.other == other) {
                ++touch.pairs;
                return false;
            }
        }
        touching.push_back({other, 1});
        return true;
    }

    // ...and back to 0?
    static bool releaseTouch(std::vector<Touch>& touching, const ::physx::PxRigidActor* other) {

        for (auto it = touching.begin(); it != touching.end(); ++it) {
            if (it->other != other) continue;
            if (--it->pairs <= 0) {
                touching.erase(it);
                return true;
            }
            return false;
        }
        return false;
    }

    // The object on the far side, weakly. Resolved HERE rather than at delivery
    // because the actor pointer is only meaningful while the world that owns it
    // is; the object outlives both.
    [[nodiscard]] std::weak_ptr<Object3D> objectFor(const ::physx::PxRigidActor* actor) const {

        if (!physics || worldLife.expired()) return {};
        auto* object = physics->findObject(actor);
        // weak_from_this, not shared_from_this: a node somehow not owned by a
        // shared_ptr answers "nothing" instead of throwing into the solver.
        return object ? object->weak_from_this() : std::weak_ptr<Object3D>{};
    }

    // Called from inside fetchResults(), once per reported pair per substep.
    // Everything here is pointer arithmetic and vector pushes — no Python, no
    // allocation that can throw into PhysX's lap beyond the queue's own growth.
    void onContact(std::size_t watchIndex, const ContactEvent& event) {

        auto& watch = watches[watchIndex];
        const auto& instance = instances[watch.instance];
        // A disabled instance still gets its latch maintained — cheap, and it
        // keeps the state honest — but nothing is queued for it.
        const bool live = !instance.failed;

        if (event.touchFound && acquireTouch(watch.touching, event.other)) {
            if (live && instance.hasCollisionEnter) queueEnter(watch, event);
        }
        if (event.touchLost && releaseTouch(watch.touching, event.other)) {
            if (live && instance.hasCollisionExit) queueExit(watch, event);
        }
    }

    void queueEnter(CollisionWatch& watch, const ContactEvent& event) {

        QueuedCollision queued;
        queued.enter = true;
        queued.other = objectFor(event.other);

        // PhysX expresses a pair's normal and impulse relative to the pair's
        // FIRST actor. Flip when the watched body is the second, so `normal`
        // always points into the script's own body whichever order PhysX
        // happened to put the pair in.
        const float sign = event.selfIsFirst ? 1.f : -1.f;
        float hardest = -1.f;
        for (::physx::PxU32 i = 0; i < event.pointCount; ++i) {
            const auto& point = event.points[i];
            const Vector3 impulse(point.impulse.x * sign, point.impulse.y * sign,
                                  point.impulse.z * sign);
            queued.impulse.add(impulse);
            // One representative point, and the one worth having: where the hit
            // was hardest. (A first-point rule would pick a manifold corner.)
            const float magnitude = impulse.length();
            if (magnitude > hardest) {
                hardest = magnitude;
                queued.point.set(point.position.x, point.position.y, point.position.z);
                queued.normal.set(point.normal.x * sign, point.normal.y * sign,
                                  point.normal.z * sign);
            }
        }
        watch.queue.push_back(std::move(queued));
    }

    void queueExit(CollisionWatch& watch, const ContactEvent& event) {

        QueuedCollision queued;
        queued.enter = false;
        queued.other = objectFor(event.other);
        // No geometry: a lost touch has no manifold left to extract, which is
        // why PhysX does not offer one and why `contact` carries zeros here.
        watch.queue.push_back(std::move(queued));
    }

    // --- triggers ----------------------------------------------------------
    //
    // on_trigger_enter / on_trigger_exit are the collision callbacks' overlap
    // twin, and everything above about the queue holds here word for word: the
    // report arrives from inside fetchResults(), so it is COPIED and delivered
    // from the frame sweep; the queue is a LIST, so a body that crosses a thin
    // volume between two deliveries still yields enter then exit; one crossing
    // is one PAIR OF BODIES, refcounted per shape pair so a compound volume does
    // not report four of them.
    //
    // Three things differ, and each is deliberate.
    //
    // 1. There is nothing to turn on. Contact reporting is an opt-in bit per
    //    actor; trigger reporting is the SHAPE being a trigger, which the
    //    document authored and the physics session cooked. So the watch is
    //    delivery only, and a script's methods on a body that is not a trigger
    //    and never gets entered simply never fire.
    //
    // 2. BOTH SIDES are watched, and only one of them is the volume. The script
    //    on the trigger wants to know who walked in; the script on the walker
    //    wants to know it walked in. PhysX reports one PxTriggerPair and the
    //    dispatcher hands it to whichever side asked, so a watch is registered
    //    for every scripted body either way — the entering body is not itself a
    //    trigger and there is no way to tell in advance that it will enter one.
    //
    // 3. There is no `Collision`. A trigger generates no manifold: no point, no
    //    normal, no impulse — PhysX has none to give, and a struct full of
    //    zeroes would be a lie with fields. The callback takes the OBJECT.

    struct QueuedTrigger {
        bool enter = false;
        std::weak_ptr<Object3D> other;
    };

    struct TriggerWatch {
        std::size_t instance = 0;// index into instances, stable after start()
        ::physx::PxRigidActor* actor = nullptr;
        PhysxWorld::TriggerHandle handle = 0;
        std::vector<Touch> touching;
        std::vector<QueuedTrigger> queue;
    };

    std::vector<TriggerWatch> triggerWatches;

    // Called from inside fetchResults(), once per reported trigger pair per
    // substep. Same rules as onContact: values only, no Python, no GIL.
    void onTriggerPair(std::size_t watchIndex, const TriggerEvent& event) {

        auto& watch = triggerWatches[watchIndex];
        const auto& instance = instances[watch.instance];
        const bool live = !instance.failed;

        if (event.touchFound && acquireTouch(watch.touching, event.other)) {
            if (live && instance.hasTriggerEnter) {
                watch.queue.push_back(QueuedTrigger{true, objectFor(event.other)});
            }
        }
        if (event.touchLost && releaseTouch(watch.touching, event.other)) {
            if (live && instance.hasTriggerExit) {
                watch.queue.push_back(QueuedTrigger{false, objectFor(event.other)});
            }
        }
    }

#endif

    [[nodiscard]] bool collisionsPending() const {

#ifdef THREEPP_EDITOR_WITH_PHYSX
        for (const auto& watch : watches) {
            if (!watch.queue.empty()) return true;
        }
#endif
        return false;
    }

    // Live instances whose class defines either callback.
    [[nodiscard]] std::size_t collisionsWanted() const {

        std::size_t wanted = 0;
        for (const auto& instance : instances) {
            if (instance.failed) continue;
            if (instance.hasCollisionEnter || instance.hasCollisionExit) ++wanted;
        }
        return wanted;
    }

    // Turn contact reporting on for every scripted body that asked for it, in
    // instance order — which is the order the sweeps run in, so the whole
    // feature stays as deterministic as the substep it rides on.
    void attachCollisions() {

        const auto wanted = collisionsWanted();
        if (wanted == 0) return;

#ifdef THREEPP_EDITOR_WITH_PHYSX
        if (bindWorld()) {
            for (std::size_t i = 0; i < instances.size(); ++i) {
                const auto& instance = instances[i];
                if (instance.failed) continue;
                if (!instance.hasCollisionEnter && !instance.hasCollisionExit) continue;

                const auto object = instance.object.lock();
                // The same walk up the ancestry rigid_body_from_object does, so
                // a script on a child of a physics object watches the body that
                // governs it — and covers static bodies, which are never bound
                // and would be invisible to PhysxWorld::findActor.
                auto* actor = object ? physics->findActor(object.get()) : nullptr;
                if (!actor) {
                    log("collision callbacks need a physics body - " + instance.label +
                        " has none, so they never fire");
                    continue;
                }
                watches.push_back(CollisionWatch{i, actor, 0, {}, {}});
            }
            // Registered only once the vector is final: watchContacts takes a
            // callback that has to name its watch, and an index survives the
            // reallocation a pointer would not.
            for (std::size_t w = 0; w < watches.size(); ++w) {
                watches[w].handle = world->watchContacts(
                        watches[w].actor,
                        [this, w](const ContactEvent& event) { onContact(w, event); });
            }
            return;
        }
#endif
        // No world, no contacts. Same shape as fixed_update's line, and the same
        // reason: the methods are not quietly dropped, they are reported once.
        log("collision callbacks need a playing physics world - they never run on " +
            std::to_string(wanted) + (wanted == 1 ? " script" : " scripts"));
    }

    [[nodiscard]] bool triggersPending() const {

#ifdef THREEPP_EDITOR_WITH_PHYSX
        for (const auto& watch : triggerWatches) {
            if (!watch.queue.empty()) return true;
        }
#endif
        return false;
    }

    // Live instances whose class defines either trigger callback.
    [[nodiscard]] std::size_t triggersWanted() const {

        std::size_t wanted = 0;
        for (const auto& instance : instances) {
            if (instance.failed) continue;
            if (instance.hasTriggerEnter || instance.hasTriggerExit) ++wanted;
        }
        return wanted;
    }

    // Watch trigger overlaps for every scripted body that asked, in instance
    // order — the sweep order, so this stays as deterministic as the substep it
    // rides on. Nothing is enabled on the actor here (see the block comment
    // above): a body is only a trigger because the document says so.
    void attachTriggers() {

        const auto wanted = triggersWanted();
        if (wanted == 0) return;

#ifdef THREEPP_EDITOR_WITH_PHYSX
        if (bindWorld()) {
            for (std::size_t i = 0; i < instances.size(); ++i) {
                const auto& instance = instances[i];
                if (instance.failed) continue;
                if (!instance.hasTriggerEnter && !instance.hasTriggerExit) continue;

                const auto object = instance.object.lock();
                auto* actor = object ? physics->findActor(object.get()) : nullptr;
                if (!actor) {
                    // A body IS needed, on either side. The volume needs one to
                    // be a trigger at all; the entering body needs one to be
                    // something a trigger can notice. Ticking Trigger is NOT
                    // required — a script on an ordinary body that walks into
                    // somebody else's volume is exactly half of this feature.
                    log("trigger callbacks need a physics body - " + instance.label +
                        " has none, so they never fire");
                    continue;
                }
                triggerWatches.push_back(TriggerWatch{i, actor, 0, {}, {}});
            }
            // Registered only once the vector is final, for the reason
            // attachCollisions gives: the callback names its watch by index, and
            // an index survives the reallocation a pointer would not.
            for (std::size_t w = 0; w < triggerWatches.size(); ++w) {
                triggerWatches[w].handle = world->watchTriggers(
                        triggerWatches[w].actor,
                        [this, w](const TriggerEvent& event) { onTriggerPair(w, event); });
            }
            return;
        }
#endif
        log("trigger callbacks need a playing physics world - they never run on " +
            std::to_string(wanted) + (wanted == 1 ? " script" : " scripts"));
    }

    // deliverCollisions for overlaps: same order (instance, then queue), same
    // held GIL, same disable-once. What it hands over is the object rather than
    // a Collision, since there is no contact data to put beside it.
    void deliverTriggers() {

#ifdef THREEPP_EDITOR_WITH_PHYSX
        for (auto& watch : triggerWatches) {
            if (watch.queue.empty()) continue;
            const auto queued = std::move(watch.queue);
            watch.queue.clear();

            auto& instance = instances[watch.instance];
            for (const auto& event : queued) {
                if (instance.failed) break;// disabled by an earlier event
                const char* method = event.enter ? "on_trigger_enter" : "on_trigger_exit";
                if (event.enter ? !instance.hasTriggerEnter : !instance.hasTriggerExit) continue;
                try {
                    // The object itself, not a struct wrapping it: a trigger
                    // overlap has no contact data to carry alongside.
                    if (const auto other = event.other.lock()) {
                        instance.self.attr(method)(scripting::handleFor(*other));
                    } else {
                        instance.self.attr(method)(py::none());
                    }
                } catch (py::error_already_set& e) {
                    fail(instance, method, scripting::describeError(e));
                } catch (const std::exception& e) {
                    fail(instance, method, e.what());
                }
            }
        }
#endif
    }

    // One sweep of everything queued since the last one, in instance order and
    // in queue order within an instance. GIL must be held by the caller — this
    // shares the frame sweep's single acquisition.
    void deliverCollisions() {

#ifdef THREEPP_EDITOR_WITH_PHYSX
        for (auto& watch : watches) {
            if (watch.queue.empty()) continue;
            // Moved out first: whatever a callback does, it cannot be handed the
            // same event twice, and the queue is empty again from here on.
            const auto queued = std::move(watch.queue);
            watch.queue.clear();

            auto& instance = instances[watch.instance];
            for (const auto& event : queued) {
                if (instance.failed) break;// disabled by an earlier event
                const char* method = event.enter ? "on_collision_enter" : "on_collision_exit";
                if (event.enter ? !instance.hasCollisionEnter : !instance.hasCollisionExit) continue;
                try {
                    scripting::Collision contact;
                    contact.point = event.point;
                    contact.normal = event.normal;
                    contact.impulse = event.impulse;
                    if (const auto other = event.other.lock()) {
                        contact.other = scripting::handleFor(*other);
                    } else {
                        contact.other = py::none();
                    }
                    instance.self.attr(method)(py::cast(contact));
                } catch (py::error_already_set& e) {
                    fail(instance, method, scripting::describeError(e));
                } catch (const std::exception& e) {
                    fail(instance, method, e.what());
                }
            }
        }
#endif
    }

    // Drops every interpreter-side object. GIL is acquired here rather than by
    // the caller so this is safe from the destructor too.
    void release() {

        if (instances.empty()) return;
        if (!scripting::interpreterStarted()) {
            // Nothing was ever created; the py::objects are all null.
            instances.clear();
            return;
        }
        py::gil_scoped_acquire gil;
        instances.clear();
    }
};


ScriptPlaySession::ScriptPlaySession()
    : impl_(std::make_unique<Impl>()) {}

ScriptPlaySession::~ScriptPlaySession() {

    // The editor torn down mid-Play never gets a stop(), and a substep callback
    // — or a contact watch — outliving the object it captured is exactly what
    // PhysxWorld's "anything that registers must unregister in its destructor"
    // rule is about.
    impl_->detachFromPhysics();
    // Same rule: a resolver outliving the instance list it closes over would
    // answer out of freed memory the moment anything asked.
    impl_->clearResolver();
    impl_->release();
}

void ScriptPlaySession::setLogger(std::function<void(const std::string&)> logger) {

    impl_->logger = std::move(logger);
}

std::string ScriptPlaySession::errorFor(const std::string& uuid) const {

    const auto it = impl_->errors.find(uuid);
    return it == impl_->errors.end() ? std::string{} : it->second;
}

std::size_t ScriptPlaySession::errorCount() const {

    return impl_->errors.size();
}

void ScriptPlaySession::clearErrors() {

    impl_->errors.clear();
}

std::size_t ScriptPlaySession::instanceCount() const {

    return impl_->instances.size();
}

void ScriptPlaySession::start(Scene& scene) {

    // A previous session's registrations go before its instances do. Replaying
    // without that would leave the old callbacks in a world that has since been
    // rebuilt — and every handle stale for the new one — and would deliver the
    // last session's queued contacts to this one's scripts.
    impl_->detachFromPhysics();
    // Same reasoning one door over: a resolver still answering out of the list
    // release() is about to empty would hand this Play the last one's instances.
    impl_->clearResolver();
    impl_->release();
    impl_->errors.clear();

    // Collect first, so nothing Python-shaped happens on a scene with no
    // scripts — starting an interpreter costs a hundred milliseconds and every
    // Play would pay it.
    struct Attached {
        Object3D* object;
        ScriptConfig config;
    };
    std::vector<Attached> attached;
    scene.traverse([&attached](Object3D& object) {
        if (auto config = ScriptConfig::read(object)) {
            attached.push_back({&object, std::move(*config)});
        }
    });
    if (attached.empty()) return;

    std::string error;
    if (!scripting::ensureInterpreter(&error)) {
        impl_->log("scripts skipped - the Python interpreter did not start: " + error);
        return;
    }

    py::gil_scoped_acquire gil;

    // ---- phase 1: bring every instance up ---------------------------------
    //
    // Compile, construct, apply the authored fields — and stop there. Nothing
    // in a script's own code runs yet beyond its class body and its __init__.
    //
    // This is the whole reason the loop is split. Interleaved with start(), a
    // script looking a neighbour up in its own start() would find one only when
    // the traverse happened to reach the neighbour first, which is a scene
    // ordering nobody authored deliberately and the inspector does not show. Two
    // phases turn that into a guarantee worth writing down: BY THE TIME ANY
    // start() RUNS, EVERY SCRIPT INSTANCE EXISTS, with its authored field values
    // already on it.
    for (auto& entry : attached) {

        // Recorded before anything can go wrong, so a failure has somewhere to
        // be recorded against — and unconditionally, so instances[i] is
        // attached[i] for phase 2 below, failures included.
        const auto label = labelOf(*entry.object);
        impl_->instances.push_back(
                Impl::Instance{entry.object->uuid, label, originOf(entry.config)});
        auto& instance = impl_->instances.back();

        std::error_code ec;
        if (entry.config.isFile() && !std::filesystem::exists(entry.config.path, ec)) {
            impl_->fail(instance, "load", "file not found: " + entry.config.path);
            continue;
        }

        try {
            std::string className;
            // Inline source is compiled fresh here, exactly as a file is read
            // and compiled fresh — inline scripts are therefore inherently hot:
            // Apply in the editor, press Play, and that is what runs. The
            // module is named after the object so a traceback says which one.
            auto cls = entry.config.isInline()
                               ? scripting::loadInlineScriptClass(entry.config.source,
                                                                  entry.object->uuid, label, className)
                               : scripting::loadScriptClass(entry.config.path, className);
            auto self = cls();
            impl_->applyFields(cls, self, entry.config);

            instance.hasStart = py::hasattr(self, "start");
            instance.hasUpdate = py::hasattr(self, "update");
            instance.hasStop = py::hasattr(self, "stop");
            instance.hasFixedUpdate = py::hasattr(self, "fixed_update");
            instance.hasCollisionEnter = py::hasattr(self, "on_collision_enter");
            instance.hasCollisionExit = py::hasattr(self, "on_collision_exit");
            instance.hasTriggerEnter = py::hasattr(self, "on_trigger_enter");
            instance.hasTriggerExit = py::hasattr(self, "on_trigger_exit");
            instance.self = self;
            // Kept so attachCollisions() can resolve the body governing it,
            // without a second traversal of the scene to find the object again.
            instance.object = entry.object->weak_from_this();
        } catch (py::error_already_set& e) {
            // A constructor or a field that raised disables this instance
            // exactly as a failing start() does — one report, no instance, and
            // therefore None to anyone who looks it up.
            impl_->fail(instance, "load", scripting::describeError(e));
        } catch (const std::exception& e) {
            impl_->fail(instance, "load", e.what());
        }
    }

    // Between the phases, and not a line later: the first thing a start() may
    // want to do is find a neighbour, and every neighbour now exists.
    impl_->installResolver();

    // ---- phase 2: start them ----------------------------------------------
    //
    // Same order as phase 1, so the sequence a scene's start()s run in is still
    // the traverse order the sweeps use — deterministic, just no longer load
    // bearing. instances[i] is attached[i]; phase 1 pushes exactly one instance
    // per attachment.
    for (std::size_t i = 0; i < impl_->instances.size(); ++i) {

        auto& instance = impl_->instances[i];
        // Every method is optional; a script that only wants update() — or only
        // fixed_update() — is a perfectly good script.
        if (instance.failed || !instance.hasStart) continue;

        try {
            const auto start = instance.self.attr("start");
            if (startWantsScene(start)) {
                start(scripting::handleFor(*attached[i].object), scripting::handleFor(scene));
            } else {
                start(scripting::handleFor(*attached[i].object));
            }
        } catch (py::error_already_set& e) {
            impl_->fail(instance, "start", scripting::describeError(e));
        } catch (const std::exception& e) {
            impl_->fail(instance, "start", e.what());
        }
    }

    std::size_t live = 0;
    for (const auto& instance : impl_->instances) {
        if (!instance.failed) ++live;
    }
    if (live > 0) {
        impl_->log("scripts running: " + std::to_string(live) +
                   (live == 1 ? " instance" : " instances"));
    }

    // After BOTH phases, so they only ever consider instances that actually came
    // up: a script whose start() raised is disabled, and its fixed_update, its
    // collision callbacks and its trigger callbacks with it. (The hasattr
    // results they read are phase 1's, but `failed` is not final until phase 2
    // has run — which is exactly why these stay down here rather than moving up
    // with the resolver.)
    impl_->attachToPhysics();
    impl_->attachCollisions();
    impl_->attachTriggers();
}

void ScriptPlaySession::update(float dt) {

    if (impl_->instances.empty()) return;

    bool any = false;
    for (const auto& instance : impl_->instances) {
        if (!instance.failed && instance.hasUpdate) any = true;
    }
    // Contacts and trigger overlaps queued by this frame's substeps are
    // delivered here even when no script defines update() at all.
    const bool collisions = impl_->collisionsPending();
    const bool triggers = impl_->triggersPending();
    if (!any && !collisions && !triggers) return;

    // Once for the whole sweep, not once per script: acquiring the GIL is the
    // expensive part, and there is nothing else running that wants it.
    py::gil_scoped_acquire gil;

    // BEFORE update(dt), and in the same acquisition.
    //
    // Contacts are events, not a clock — which is the whole difference from
    // fixed_update. Delivered here they arrive where update() lives: after the
    // physics session has stepped and mirrored, so a callback reads the settled
    // world this frame will draw rather than a pose halfway through a solve.
    // Every edge this frame's substeps produced is delivered in this frame (a
    // per-substep delivery would strand the last substep's events until the next
    // one, and longer still on a frame that takes no substep at all), and the
    // whole sweep shares the acquisition above instead of taking the GIL inside
    // the physics step once per substep.
    if (collisions) impl_->deliverCollisions();
    // Then the overlaps, on the same terms and for the same reasons. Contacts
    // before triggers is an arbitrary but FIXED order: a script defining both
    // sees the same sequence every run.
    if (triggers) impl_->deliverTriggers();

    for (auto& instance : impl_->instances) {
        if (instance.failed || !instance.hasUpdate) continue;
        try {
            instance.self.attr("update")(dt);
        } catch (py::error_already_set& e) {
            impl_->fail(instance, "update", scripting::describeError(e));
        } catch (const std::exception& e) {
            impl_->fail(instance, "update", e.what());
        }
    }
}

void ScriptPlaySession::stop() {

    // Before anything Python-shaped, and unconditionally: the instance list
    // being empty says nothing about whether a callback is still registered.
    impl_->detachFromPhysics();

    if (!impl_->instances.empty() && scripting::interpreterStarted()) {
        py::gil_scoped_acquire gil;
        for (auto& instance : impl_->instances) {
            if (instance.failed || !instance.hasStop) continue;
            try {
                instance.self.attr("stop")();
            } catch (py::error_already_set& e) {
                impl_->fail(instance, "stop", scripting::describeError(e));
            } catch (const std::exception& e) {
                impl_->fail(instance, "stop", e.what());
            }
        }
        // The resolver goes down WITH the list it answers from, not before it: a
        // stop() is still inside the session, every instance is still alive
        // until the line below, and a script tidying up after a neighbour is a
        // reasonable last act. Nothing may resolve after this returns.
        impl_->clearResolver();
        // Inside the same GIL scope: dropping the last reference to a script
        // instance runs its __del__ and frees Python memory.
        impl_->instances.clear();
        return;
    }
    // No instances, or no interpreter to run stop() in. The release is still
    // unconditional, for the reason detachFromPhysics above is: an empty list
    // says nothing about what is still installed.
    impl_->clearResolver();
    impl_->instances.clear();
}
