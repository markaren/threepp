// Ambient bird flock: threepp::Flock from extras/fauna, exposed to Python.
//
// Flock is a Mesh subclass that owns its own simulation and rebakes one merged
// geometry every update() — so from Python it is the same three lines it is
// from C++: construct it, add it to the scene, call update(dt) once a frame.
//
// Bound as a subclass of the already-registered Mesh (Mesh is a non-virtual base
// of Flock, so the concrete Object3D API bound on Mesh comes through safely) —
// the same pattern as InstancedMesh and DisplacedMesh.
//
// Params is a wide struct (~80 fields). What is bound here is the subset a host
// actually reaches for: population, territory, flight envelope, the boids
// weights, perching, and the look. Everything else keeps its C++ default, which
// the header argues for at length at each site.
#include "bindings.hpp"

#include "threepp/extras/fauna/Flock.hpp"

#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace threepp_py {

    void init_fauna(py::module_& m) {

        using namespace threepp;

        py::enum_<Flock::BirdState>(m, "BirdState")
                .value("Cruise", Flock::BirdState::Cruise)
                .value("Approach", Flock::BirdState::Approach)
                .value("Flare", Flock::BirdState::Flare)
                .value("Perched", Flock::BirdState::Perched)
                .value("Launch", Flock::BirdState::Launch)
                .value("Evade", Flock::BirdState::Evade)
                .export_values();

        py::enum_<Flock::BirdRole>(m, "BirdRole")
                .value("Follower", Flock::BirdRole::Follower)
                .value("Leader", Flock::BirdRole::Leader)
                .value("Loner", Flock::BirdRole::Loner)
                .export_values();

        py::enum_<Flock::Gait>(m, "Gait")
                .value("Hop", Flock::Gait::Hop)
                .value("Walk", Flock::Gait::Walk)
                .export_values();

        // ---- look ------------------------------------------------------------
        py::class_<fauna::BirdShape>(m, "BirdShape")
                .def(py::init<>())
                .def_readwrite("body_length", &fauna::BirdShape::bodyLength, "m, bill tip to tail tip")
                .def_readwrite("body_radius", &fauna::BirdShape::bodyRadius, "m, max half-width of the body spindle")
                .def_readwrite("wing_span", &fauna::BirdShape::wingSpan, "m, tip to tip, fully extended")
                .def_readwrite("tail_fork", &fauna::BirdShape::tailFork, "-1 forked .. +1 wedge; 0 = square");

        py::class_<fauna::BirdPlumage>(m, "BirdPlumage")
                .def(py::init<>())
                .def_readwrite("back", &fauna::BirdPlumage::back, "dorsal colour (LINEAR rgb; multiplies albedo)")
                .def_readwrite("belly", &fauna::BirdPlumage::belly, "ventral colour")
                .def_readwrite("cap", &fauna::BirdPlumage::cap, "crown + bill")
                .def_readwrite("leg", &fauna::BirdPlumage::leg)
                .def_readwrite("wingtip_dark", &fauna::BirdPlumage::wingtipDark, "0..1 multiplier at the primaries")
                .def_readwrite("tail_band_dark", &fauna::BirdPlumage::tailBandDark)
                .def_readwrite("cap_strength", &fauna::BirdPlumage::capStrength)
                .def_readwrite("lightness_jitter", &fauna::BirdPlumage::lightnessJitter);

        // ---- params ----------------------------------------------------------
        py::class_<Flock::Params>(m, "FlockParams")
                .def(py::init<>())
                .def_readwrite("seed", &Flock::Params::seed)
                .def_readwrite("bird_count", &Flock::Params::birdCount,
                               "Hard-clamped to [0, 256]: neighbour search is O(N^2).")
                // territory
                .def_readwrite("home", &Flock::Params::home, "Centre of the loiter volume (world m); drifts at runtime.")
                .def_readwrite("roam_radius", &Flock::Params::roamRadius)
                .def_readwrite("cruise_altitude", &Flock::Params::cruiseAltitude, "m above the baked ground under home")
                .def_readwrite("altitude_spread", &Flock::Params::altitudeSpread)
                .def_readwrite("home_drift_rate", &Flock::Params::homeDriftRate)
                // species
                .def_readwrite("mass_kg", &Flock::Params::massKg, "drives wingbeat_hz allometrically")
                .def_readwrite("wingbeat_hz", &Flock::Params::wingbeatHz, "0 => derived from mass_kg")
                .def_readwrite("gait", &Flock::Params::gait)
                // flight
                .def_readwrite("cruise_speed", &Flock::Params::cruiseSpeed)
                .def_readwrite("min_speed", &Flock::Params::minSpeed)
                .def_readwrite("max_speed", &Flock::Params::maxSpeed)
                .def_readwrite("max_accel_along", &Flock::Params::maxAccelAlong)
                .def_readwrite("max_accel_lateral", &Flock::Params::maxAccelLateral)
                .def_readwrite("max_turn_rate", &Flock::Params::maxTurnRate)
                .def_readwrite("max_bank", &Flock::Params::maxBank)
                // boids
                .def_readwrite("neighbour_count", &Flock::Params::neighbourCount,
                               "Topological, not metric: a fixed NUMBER of nearest neighbours.")
                .def_readwrite("neighbour_radius", &Flock::Params::neighbourRadius)
                .def_readwrite("separation_distance", &Flock::Params::separationDistance)
                .def_readwrite("w_separation", &Flock::Params::wSeparation)
                .def_readwrite("w_alignment", &Flock::Params::wAlignment)
                .def_readwrite("w_cohesion", &Flock::Params::wCohesion)
                .def_readwrite("w_wander", &Flock::Params::wWander)
                .def_readwrite("w_bounds", &Flock::Params::wBounds)
                .def_readwrite("w_altitude", &Flock::Params::wAltitude)
                .def_readwrite("w_obstacle", &Flock::Params::wObstacle)
                .def_readwrite("w_ground", &Flock::Params::wGround)
                .def_readwrite("w_goal", &Flock::Params::wGoal)
                .def_readwrite("w_observer", &Flock::Params::wObserver)
                .def_readwrite("leader_fraction", &Flock::Params::leaderFraction)
                .def_readwrite("loner_fraction", &Flock::Params::lonerFraction)
                // obstacles / ground
                .def_readwrite("lookahead_time", &Flock::Params::lookaheadTime)
                .def_readwrite("obstacle_margin", &Flock::Params::obstacleMargin)
                .def_readwrite("min_ground_clearance", &Flock::Params::minGroundClearance)
                // perching
                .def_readwrite("perching", &Flock::Params::perching)
                .def_readwrite("perch_search_radius", &Flock::Params::perchSearchRadius)
                .def_readwrite("perch_interval_min", &Flock::Params::perchIntervalMin)
                .def_readwrite("perch_interval_max", &Flock::Params::perchIntervalMax)
                .def_readwrite("rest_interval_min", &Flock::Params::restIntervalMin)
                .def_readwrite("rest_interval_max", &Flock::Params::restIntervalMax)
                .def_readwrite("max_perched_fraction", &Flock::Params::maxPerchedFraction)
                .def_readwrite("perch_contagion", &Flock::Params::perchContagion)
                .def_readwrite("launch_contagion", &Flock::Params::launchContagion)
                .def_readwrite("ground_bias", &Flock::Params::groundBias)
                .def_readwrite("abort_chance", &Flock::Params::abortChance)
                // disturbance
                .def_readwrite("flight_initiation_distance", &Flock::Params::flightInitiationDistance)
                .def_readwrite("startle_wave_speed", &Flock::Params::startleWaveSpeed)
                // look
                .def_readwrite("shape", &Flock::Params::shape)
                .def_readwrite("plumage", &Flock::Params::plumage)
                .def_readwrite("size_variation", &Flock::Params::sizeVariation)
                .def_readwrite("birds_cast_shadow", &Flock::Params::birdsCastShadow)
                .def_readwrite("lod_far_distance", &Flock::Params::lodFarDistance)
                .def_readwrite("wind", &Flock::Params::wind, "world XZ; perched birds face into it");

        // ---- the flock -------------------------------------------------------
        py::class_<Flock, Mesh, std::shared_ptr<Flock>>(m, "Flock")
                .def(py::init([](const Flock::Params& p) { return Flock::create(p); }),
                     py::arg("params"),
                     "Build a flock from a FlockParams. Add it to a Scene and call "
                     "update(dt) once per frame.")
                .def(py::init([](int bird_count, const Vector3& home, float roam_radius,
                                 float cruise_altitude, float cruise_speed, bool perching,
                                 unsigned int seed) {
                         Flock::Params p;
                         p.birdCount = bird_count;
                         p.home = home;
                         p.roamRadius = roam_radius;
                         p.cruiseAltitude = cruise_altitude;
                         p.cruiseSpeed = cruise_speed;
                         p.perching = perching;
                         p.seed = seed;
                         return Flock::create(p);
                     }),
                     py::arg("bird_count") = 18, py::arg("home") = Vector3(0.f, 14.f, 0.f),
                     py::arg("roam_radius") = 42.0f, py::arg("cruise_altitude") = 14.0f,
                     py::arg("cruise_speed") = 9.0f, py::arg("perching") = true,
                     py::arg("seed") = 1337u,
                     "Convenience constructor over the handful of fields most hosts set. "
                     "For the rest, build a FlockParams and pass that instead.")
                .def("update", &Flock::update, py::arg("dt"),
                     "Advance the simulation and rebake the geometry. CALL ONCE PER FRAME. "
                     "dt is clamped internally to [0, 0.05] s; dt <= 0 is a no-op.")
                // Perching. bake_perches walks the scene for landable surfaces; the
                // async form spreads the work over frames, the blocking form does not.
                .def("bake_perches", &Flock::bakePerches, py::arg("scene_root"),
                     "Scan a scene for landable surfaces (async; poll bake_complete()).")
                .def("bake_perches_blocking", &Flock::bakePerchesBlocking, py::arg("scene_root"),
                     "Same, but finish before returning.")
                .def("bake_complete", &Flock::bakeComplete)
                .def("bake_progress", &Flock::bakeProgress)
                .def("perch_count", &Flock::perchCount)
                .def("add_perch", &Flock::addPerch,
                     py::arg("world_pos"), py::arg("world_normal"), py::arg("walkable"),
                     "Add one perch by hand, instead of (or as well as) baking.")
                .def("set_perch_filter", &Flock::setPerchFilter, py::arg("predicate"),
                     "predicate(mesh) -> bool, deciding which meshes may be landed on.")
                // Disturbance
                .def("startle", &Flock::startle,
                     py::arg("epicentre"), py::arg("radius") = 1e9f, py::arg("strength") = 1.0f,
                     "Flush the flock away from a world point.")
                .def("set_disturbance_source", [](Flock& f, Object3D* o) { f.setDisturbanceSource(o); },
                     py::arg("source"), py::keep_alive<1, 2>(),
                     "Birds take flight when this object comes within "
                     "flight_initiation_distance. Pass None to clear.")
                .def("set_observer", [](Flock& f, Camera* c) { f.setObserver(c); },
                     py::arg("camera"), py::keep_alive<1, 2>(),
                     "Soft repulsion from the camera, so birds do not fly through the lens.")
                .def("set_wind", [](Flock& f, float x, float z) { f.setWind(Vector2(x, z)); },
                     py::arg("x"), py::arg("z"),
                     "World XZ wind direction; perched birds turn to face into it.")
                // Read-out
                .def("bird_count", &Flock::birdCount)
                .def("perched_count", &Flock::perchedCount)
                .def("flying_count", &Flock::flyingCount)
                .def("state_of", &Flock::stateOf, py::arg("i"))
                .def("role_of", &Flock::roleOf, py::arg("i"))
                .def("bird_position", &Flock::birdPosition, py::arg("i"),
                     py::return_value_policy::copy)
                .def("bird_velocity", &Flock::birdVelocity, py::arg("i"),
                     py::return_value_policy::copy)
                .def("update_count", &Flock::updateCount)
                .def("stalled_updates", &Flock::stalledUpdates)
                .def_property_readonly("params", [](Flock& f) { return f.params(); },
                                       "A COPY of the construction params (read-only).");
    }

}// namespace threepp_py
