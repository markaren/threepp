// The cathedral's design: where every plank goes, from one seed.
//
// Every plank is the same board, 117 x 23.4 x 7.8 mm (1 : 3 : 15), always laid
// flat, so a building is a stack of courses 7.8 mm apart. The elements:
//
//  - Ring towers, in tiers. Each course is a ring of n planks set tangentially
//    around the tower's axis, the next course turned by half a plank (pi / n), so
//    that every plank bridges a joint below. Each tier is a little narrower than
//    the one below (fewer planks to the ring), carries a row of windows and ends
//    in a gallery, a ring stepping out by a few millimetres a course. A tower
//    closes as a spire of slowly shrinking rings, dropping planks per course as
//    the ring gets too small for them, or as a dome. A window is a plank left out
//    of every other course, its two neighbours in the courses between drawn back.
//
//  - Walls, a plank long thick, in English bond: courses of stretchers (two rows
//    of planks end to end along the wall, one on each face) alternate with
//    courses of headers (planks across the wall, side by side). Every joint is a
//    hairline; what a course cannot fill with whole stretchers it fills with
//    headers, so a wall reads as a surface of fine horizontal lines. Openings
//    are corbelled: above the springing each stretcher course reaches into the
//    opening by at most a fifth of a plank, the header course over it follows,
//    until a stretcher can bridge what is left. Arches are pointed (two arcs) or
//    round; a round window opens below its centre and closes above it.
//    A wall can lean: above its eaves it steps in by a few millimetres a course,
//    and two leaning walls meet over a nave as its roof. A gable steps in from
//    both ends at the same pitch.
//
//  - Ridges: courses of planks along a roof's ridge over two leaning walls,
//    narrowing to the top at the roof's pitch, every other course with a row
//    of planks across its middle that ties the two sides together.
//
//  - The plan: a west front with a portal and a round window under a gable,
//    between two towers with spires; a nave of tall bays standing over lower
//    aisles, with stepped buttresses; a crossing tower with a spire or a dome, a
//    transept with gabled ends, a choir and a round apse.
//
// Every course is checked as it is laid: a plank that would overlap another, or
// that does not rest on the course below on both sides of its middle, is left
// out. Whether the whole stands is for PhysX to say (Stability.hpp).

#ifndef PLANKCATHEDRAL_CATHEDRAL_HPP
#define PLANKCATHEDRAL_CATHEDRAL_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace planks {

    // the plank, in metres: x along its length, y its thickness, z its width
    constexpr float LEN = 0.117f;
    constexpr float THICK = 0.0078f;
    constexpr float WIDE = 0.0234f;
    constexpr float PI = 3.14159265358979f;

    struct Plank {
        float x, z;    // centre on the floor plan
        float angle;   // the length runs along (cos angle, 0, sin angle)
        int course;    // its centre is at (course + 0.5) * THICK
        int element;   // which element of the plan laid it
    };

    inline float courseY(int course) {
        return (static_cast<float>(course) + 0.5f) * THICK;
    }

    // splitmix64: the same numbers from the same seed on every compiler
    class Rng {
    public:
        explicit Rng(std::uint64_t seed): state_(seed * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull) {}

        std::uint64_t next() {
            std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            return z ^ (z >> 31);
        }

        float uniform(float a = 0, float b = 1) {
            return a + (b - a) * static_cast<float>(next() >> 40) / static_cast<float>(1ull << 24);
        }

        int range(int lo, int hi) {// inclusive
            return lo + static_cast<int>(next() % static_cast<std::uint64_t>(hi - lo + 1));
        }

        bool chance(float p) { return uniform() < p; }

    private:
        std::uint64_t state_;
    };

    // The smallest ring of n planks: their inner corners just clear each other.
    inline float ringMin(int n) {
        if (n <= 1) return 0.f;
        if (n == 2) return WIDE / 2 + 0.002f;
        return WIDE / 2 + (LEN / 2) / std::tan(PI / static_cast<float>(n));
    }

    // The largest: the gap between two planks of a course is at most half a
    // plank, so that the plank over it rests a quarter of its length on each side.
    inline float ringMax(int n) {
        if (n <= 2) return ringMin(n) + 0.02f;
        return 0.75f * LEN / std::tan(PI / static_cast<float>(n));
    }

    // how far out a ring of apothem a reaches, at its planks' corners
    inline float ringReach(float a) {
        return std::sqrt((a + WIDE / 2) * (a + WIDE / 2) + (LEN / 2) * (LEN / 2));
    }

    struct Vec2 {
        float x, z;
    };

    class Design {
    public:
        std::vector<Plank> planks;
        std::vector<std::string> elements;// names, for the report
        int rejectedOverlap = 0;
        int rejectedSupport = 0;
        std::vector<int> leftOut;     // per element
        std::vector<int> firstLeftOut;// the lowest course where it lost a plank

        bool explain = false;// print the first planks left out of each element, and why

        int element(const std::string& name) {
            elements.push_back(name);
            return static_cast<int>(elements.size()) - 1;
        }

        void add(float x, float z, float angle, int course, int element) {
            if (course < 0) return;
            candidates_.push_back({x, z, angle, course, element});
        }

        // Lays the candidates course by course, leaving out every plank that
        // would overlap one already laid, or that would not rest on the course
        // below on both sides of its middle.
        void finish() {
            std::stable_sort(candidates_.begin(), candidates_.end(), [](const Plank& a, const Plank& b) { return a.course < b.course; });
            planks.clear();
            Grid below, here;
            int current = -1;
            for (const auto& p : candidates_) {
                if (p.course != current) {
                    if (p.course == current + 1) below = std::move(here);
                    else below = Grid{};
                    here = Grid{};
                    current = p.course;
                }
                if (leftOut.size() < elements.size()) leftOut.resize(elements.size(), 0), firstLeftOut.resize(elements.size(), -1);
                const auto lose = [&](const char* why) {
                    if (explain && leftOut[p.element] < 3)
                        std::printf("      left out of %s: course %d at (%.4f, %.4f) angle %.4f: %s\n", elements[p.element].c_str(), p.course, p.x, p.z, p.angle, why);
                    ++leftOut[p.element];
                    if (firstLeftOut[p.element] < 0) firstLeftOut[p.element] = p.course;
                };
                Plank other{};
                if (overlaps(p, here, &other)) {
                    ++rejectedOverlap;
                    if (explain && leftOut[p.element] < 3)
                        std::printf("      overlapped %s plank at (%.4f, %.4f) angle %.4f\n", elements[other.element].c_str(), other.x, other.z, other.angle);
                    lose("overlaps");
                    continue;
                }
                if (p.course > 0 && !supported(p, below)) {
                    ++rejectedSupport;
                    if (explain && leftOut[p.element] < 3) std::printf("      support along %.4f..%.4f across %.4f..%.4f\n", lastSupport[0], lastSupport[1], lastSupport[2], lastSupport[3]);
                    lose("unsupported");
                    continue;
                }
                here.insert(p, static_cast<int>(planks.size()));
                planks.push_back(p);
            }
            candidates_.clear();
        }

    private:
        std::vector<Plank> candidates_;

        static constexpr float cell = 0.125f;
        static inline float lastSupport[4] = {0, 0, 0, 0};

        struct Grid {
            std::unordered_map<std::int64_t, std::vector<Plank>> cells;

            static std::int64_t key(int i, int j) {
                return (static_cast<std::int64_t>(i) << 32) ^ static_cast<std::uint32_t>(j);
            }

            void insert(const Plank& p, int) {
                cells[key(static_cast<int>(std::floor(p.x / cell)), static_cast<int>(std::floor(p.z / cell)))].push_back(p);
            }

            template<class F>
            void near(float x, float z, F&& f) const {
                const int ci = static_cast<int>(std::floor(x / cell)), cj = static_cast<int>(std::floor(z / cell));
                for (int i = ci - 1; i <= ci + 1; ++i)
                    for (int j = cj - 1; j <= cj + 1; ++j) {
                        const auto it = cells.find(key(i, j));
                        if (it == cells.end()) continue;
                        for (const auto& q : it->second) f(q);
                    }
            }
        };

        // the four corners of a plank's footprint
        static void corners(const Plank& p, Vec2 out[4]) {
            const float c = std::cos(p.angle), s = std::sin(p.angle);
            const float hl = LEN / 2, hw = WIDE / 2;
            const float ax = c * hl, az = s * hl, bx = -s * hw, bz = c * hw;
            out[0] = {p.x + ax + bx, p.z + az + bz};
            out[1] = {p.x + ax - bx, p.z + az - bz};
            out[2] = {p.x - ax - bx, p.z - az - bz};
            out[3] = {p.x - ax + bx, p.z - az + bz};
        }

        // How far two footprints overlap along the axis that separates them
        // best (negative: apart).
        static float penetration(const Plank& a, const Plank& b) {
            Vec2 ca[4], cb[4];
            corners(a, ca);
            corners(b, cb);
            const float axes[4][2] = {
                    {std::cos(a.angle), std::sin(a.angle)},
                    {-std::sin(a.angle), std::cos(a.angle)},
                    {std::cos(b.angle), std::sin(b.angle)},
                    {-std::sin(b.angle), std::cos(b.angle)}};
            float least = 1e9f;
            for (const auto& ax : axes) {
                float amin = 1e9f, amax = -1e9f, bmin = 1e9f, bmax = -1e9f;
                for (int i = 0; i < 4; ++i) {
                    const float pa = ca[i].x * ax[0] + ca[i].z * ax[1];
                    const float pb = cb[i].x * ax[0] + cb[i].z * ax[1];
                    amin = std::min(amin, pa), amax = std::max(amax, pa);
                    bmin = std::min(bmin, pb), bmax = std::max(bmax, pb);
                }
                least = std::min(least, std::min(amax, bmax) - std::max(amin, bmin));
            }
            return least;
        }

        static bool overlaps(const Plank& p, const Grid& here, Plank* with = nullptr) {
            bool hit = false;
            here.near(p.x, p.z, [&](const Plank& q) {
                if (!hit && penetration(p, q) > 0.0002f) {
                    hit = true;
                    if (with) *with = q;
                }
            });
            return hit;
        }

        // Rests on the course below: along its length there is support on both
        // sides of its middle (or under it), and the same across its width.
        static bool supported(const Plank& p, const Grid& below) {
            const float c = std::cos(p.angle), s = std::sin(p.angle);
            float lengthMin = 1e9f, lengthMax = -1e9f, widthMin = 1e9f, widthMax = -1e9f;
            below.near(p.x, p.z, [&](const Plank& q) {
                if (penetration(p, q) < 0.002f) return;// must really overlap
                Vec2 cq[4];
                corners(q, cq);
                float umin = 1e9f, umax = -1e9f, vmin = 1e9f, vmax = -1e9f;
                for (const auto& k : cq) {
                    const float dx = k.x - p.x, dz = k.z - p.z;
                    const float u = dx * c + dz * s, v = -dx * s + dz * c;
                    umin = std::min(umin, u), umax = std::max(umax, u);
                    vmin = std::min(vmin, v), vmax = std::max(vmax, v);
                }
                umin = std::max(umin, -LEN / 2), umax = std::min(umax, LEN / 2);
                vmin = std::max(vmin, -WIDE / 2), vmax = std::min(vmax, WIDE / 2);
                lengthMin = std::min(lengthMin, umin), lengthMax = std::max(lengthMax, umax);
                widthMin = std::min(widthMin, vmin), widthMax = std::max(widthMax, vmax);
            });
            const float alongMargin = 0.003f, acrossMargin = 0.0015f;
            lastSupport[0] = lengthMin, lastSupport[1] = lengthMax, lastSupport[2] = widthMin, lastSupport[3] = widthMax;
            return lengthMin < -alongMargin && lengthMax > alongMargin && widthMin < -acrossMargin && widthMax > acrossMargin;
        }
    };


    // ---------------------------------------------------------------- towers

    struct RingWindow {
        int course;  // first course without its plank, counted from the start of its tier
        int height;  // courses; even, so the course after it is a lintel
        float angle; // where, snapped to a plank of the first course
    };

    // One ring course: n planks at apothem a, the first at angle rot.
    // `skip` and `shift` carry the windows: planks left out, and planks turned
    // away from a window by the given angle.
    inline void ringCourse(Design& d, int element, Vec2 c, int course, int n, float a, float rot,
                           const std::vector<int>& skip = {}, const std::vector<std::pair<int, float>>& shift = {}) {
        for (int j = 0; j < n; ++j) {
            if (std::find(skip.begin(), skip.end(), j) != skip.end()) continue;
            float phi = rot + 2 * PI * static_cast<float>(j) / static_cast<float>(n);
            for (const auto& [index, by] : shift)
                if (index == j) phi += by;
            d.add(c.x + a * std::cos(phi), c.z + a * std::sin(phi), phi + PI / 2, course, element);
        }
    }

    // A stretch of a tower with the same ring: n planks a course at an apothem.
    struct Tier {
        int n = 10;
        float apothem = 0.2f;
        int courses = 40;
        bool gallery = false;// its last courses step out as a gallery
        std::vector<RingWindow> windows;
    };

    struct Tower {
        Vec2 centre{};
        int base = 0;               // first course
        std::vector<Tier> tiers;    // from the foot up, each a little narrower
        enum class Top { Cone, Lantern, Dome, None } top = Top::Cone;
        float coneStep = 0.0045f;   // inward per course in a cone or a spire
        float spireFoot = 0;        // a spire first steps in, a ring's step a course, to this apothem
    };

    constexpr float ringStep = WIDE / 4;// the most a ring may step in or out per course
    constexpr int galleryCourses = 5;
    constexpr float galleryOut = 3 * ringStep;

    inline float approach(float from, float to, float step) {
        if (to > from) return std::min(to, from + step);
        return std::max(to, from - step);
    }

    // The most planks a ring of apothem a can take, never more than n.
    inline int ringFit(float a, int n) {
        while (n > 3 && ringMin(n) + 0.0008f > a) --n;
        return n;
    }

    // how far from its axis any course of the tower reaches
    inline float towerReach(const Tower& t) {
        float r = 0;
        for (const auto& tier : t.tiers) r = std::max(r, ringReach(tier.apothem + (tier.gallery ? galleryOut : 0.f)));
        return r;
    }

    inline int buildTower(Design& d, const Tower& t, const std::string& name) {

        const int e = d.element(name);
        int n = t.tiers.front().n;
        float a = t.tiers.front().apothem;
        float rot = 0;
        int course = t.base;

        // A window, once begun, at the angle of the plank it leaves out of its
        // first course. The ring keeps its plank count while a window is open.
        struct Open {
            int first, height;
            float theta;
        };
        std::vector<Open> open;

        // One course of the ring at the current rotation; k counts the courses
        // that `windows` are given in.
        const auto lay = [&](int k, const std::vector<RingWindow>& windows) {
            const float pitch = 2 * PI / static_cast<float>(n);
            const auto index = [&](float angle) {
                const long j = std::lround((angle - rot) / pitch);
                return static_cast<int>(((j % n) + n) % n);
            };
            // how far the planks either side of a window can be drawn back from it
            const float corner = std::atan2(LEN / 2, a - WIDE / 2);
            const float room = std::max(0.f, 2 * (PI / static_cast<float>(n) - corner) - 0.0012f / a);
            const float by = std::min(room, 0.3f * PI / static_cast<float>(n));
            // a window opens only where they can be drawn back far enough to rest a few
            // millimetres past their middles on the planks beside them (that takes a loose ring)
            for (const auto& w : windows)
                if (w.course == k && by * a >= 0.0095f) {
                    const float theta = rot + pitch * std::floor((w.angle - rot) / pitch + 0.5f);
                    bool clear = true;
                    for (const auto& o : open) {
                        if (k - o.first >= o.height) continue;
                        const float apart = std::abs(std::remainder(theta - o.theta, 2 * PI));
                        if (apart < 2.5f * pitch) clear = false;
                    }
                    if (clear) open.push_back({k, w.height, theta});
                }

            std::vector<int> skip;
            std::vector<std::pair<int, float>> shift;
            for (const auto& w : open) {
                const int rel = k - w.first;
                if (rel < 0 || rel >= w.height) continue;
                if (rel % 2 == 0) {
                    skip.push_back(index(w.theta));
                } else {
                    // its two neighbours, half a plank either side, drawn back
                    shift.emplace_back(index(w.theta + PI / static_cast<float>(n)), by);
                    shift.emplace_back(index(w.theta - PI / static_cast<float>(n)), -by);
                }
            }
            ringCourse(d, e, t.centre, course, n, a, rot, skip, shift);
            rot += PI / static_cast<float>(n);
            ++course;
        };

        // the tiers, each stepping in to its own ring, the gallery at its top stepping out
        for (const auto& tier : t.tiers) {
            for (int k = 0; k < tier.courses; ++k) {
                float target = tier.apothem;
                if (tier.gallery && k >= tier.courses - galleryCourses) target += galleryOut;
                a = approach(a, target, ringStep);
                // fewer planks as the ring steps in, down to the tier's own count once it is loose enough for them
                if (n != tier.n) n = a <= ringMax(tier.n) ? tier.n : std::max(tier.n, ringFit(a, n));
                a = std::clamp(a, ringMin(n) + 0.0008f, ringMax(n));
                lay(k, tier.windows);
            }
            open.clear();
        }
        // back in from a last gallery, so that it stands out at the foot of the top
        if (t.top != Tower::Top::None) {
            const float last = t.tiers.back().apothem;
            while (a > last + 0.0001f) {
                a = approach(a, last, ringStep);
                n = ringFit(a, n);
                lay(-1, {});
            }
        }

        // a cone of shrinking rings, closed by a plank across the last ring
        const auto closeCone = [&](float step) {
            for (int guard = 0; guard < 600; ++guard) {
                a -= step;
                if (a < ringMin(n) + 0.0008f) {
                    const int fit = ringFit(a, n);
                    if (fit == n) {
                        d.add(t.centre.x, t.centre.z, rot + PI / 2, course, e);
                        ++course;
                        return;
                    }
                    n = fit;
                }
                lay(-1, {});
            }
        };

        // a short open lantern on a small ring
        const auto lantern = [&](int courses) {
            std::vector<RingWindow> windows;
            const int count = n >= 8 ? 4 : 3;
            for (int i = 0; i < count; ++i) windows.push_back({2, courses - 4, PI / 4 + 2 * PI * static_cast<float>(i) / static_cast<float>(count)});
            for (int k = 0; k < courses; ++k) lay(k, windows);
            open.clear();
        };

        switch (t.top) {
            case Tower::Top::None:
                break;
            case Tower::Top::Cone:
                // a shoulder stepping in to the spire's foot, then the spire
                while (t.spireFoot > 0 && a - ringStep > t.spireFoot) {
                    a -= ringStep;
                    n = ringFit(a, n);
                    lay(-1, {});
                }
                closeCone(t.coneStep);
                break;
            case Tower::Top::Lantern: {
                const int nl = std::max(6, n - 2);
                const float al = ringMin(nl) + 0.014f;
                while (a - ringStep > al) {
                    a -= ringStep;
                    n = ringFit(a, n);
                    lay(-1, {});
                }
                a = std::max(al, ringMin(n) + 0.0008f);
                n = ringFit(a, n);
                lantern(18);
                closeCone(t.coneStep);
                break;
            }
            case Tower::Top::Dome: {
                // shrinking rings on a quarter circle, steepened where a course
                // would step in further than a ring may
                const float radius = a;
                float rise = 0;
                const float al = ringMin(6) + 0.012f;
                while (a > al + 0.0005f) {
                    rise += THICK;
                    const float onCircle = rise < radius ? std::sqrt(radius * radius - rise * rise) : 0.f;
                    a = std::max({onCircle, a - ringStep, al});
                    n = ringFit(a, n);
                    lay(-1, {});
                }
                lantern(12);
                closeCone(t.coneStep);
                break;
            }
        }
        return course;
    }

    // ---------------------------------------------------------------- walls

    struct Opening {
        float at;        // centre, metres along the wall from its start
        float halfWidth;
        int sill;        // first open course
        int spring;      // where the arch starts to close; a round window's centre
        float rise = 1;  // a pointed arch's two arcs, in widths of the opening: 0.5 is a round arch, 1 equilateral
        bool round = false;// a round window, of radius halfWidth
    };

    struct Wall {
        Vec2 from{}, to{};
        int base = 0;
        int courses = 60;
        std::vector<Opening> openings;
        // an end against something: its stretchers start there; a free end ends where they do
        bool fixedFrom = true, fixedTo = true;
        // metres drawn in at either end, per course (gables, stepped buttresses, clearing a ring)
        std::function<float(int)> trimFrom, trimTo;
        // above leanFrom every course steps along the wall's left normal (-u.z, u.x) by this much
        float lean = 0;
        int leanFrom = 1 << 30;
    };

    constexpr float corbelStep = LEN / 6;// the most a stretcher course reaches into an opening
    constexpr float joint = 0.0008f;     // between the planks of a course: a hairline

    // Each opening's half-width per course, or -1 where the wall is closed.
    // Stretcher courses are the even ones (counted from the wall's base); a header
    // course always keeps the opening of the stretcher course below it.
    inline std::vector<float> openingProfile(const Opening& o, int courses) {
        std::vector<float> hw(courses, -1.f);
        const float closeAt = LEN / 4;// a stretcher can bridge what is left
        const float radius = std::max(o.halfWidth, o.rise * 2 * o.halfWidth);
        for (int k = 0; k < courses; ++k) {
            if (k < o.sill) continue;
            if (k % 2 == 1) {
                hw[k] = (k > 0 && hw[k - 1] > 0) ? hw[k - 1] : -1.f;
                continue;
            }
            float raw;
            if (o.round) {
                const float y = static_cast<float>(k - o.spring) * THICK;
                raw = std::abs(y) < o.halfWidth ? std::sqrt(o.halfWidth * o.halfWidth - y * y) : -1.f;
                if (raw < 0 && k < o.spring) continue;// not open yet
            } else if (k < o.spring) {
                raw = o.halfWidth;
            } else {
                // two arcs of the radius, each centred on the far side of the opening
                const float y = static_cast<float>(k - o.spring + 1) * THICK;
                raw = y < radius ? std::sqrt(radius * radius - y * y) - (radius - o.halfWidth) : 0.f;
            }
            const float before = k >= 2 ? hw[k - 2] : -1.f;
            if (before > 0) {
                // open below: close once what is left is narrow enough for any
                // stretcher laid across it to rest on both sides of its middle,
                // otherwise step in by no more than a corbel step
                if (before <= closeAt + 0.0001f) break;
                hw[k] = raw >= before ? raw : std::max({raw, before - corbelStep, closeAt});
                continue;
            }
            if (raw >= closeAt) hw[k] = raw;
            else if (k >= o.spring) break;
        }
        return hw;
    }

    // Every course fills its stretch [a, b] of a wall end to end, so that the
    // courses of a stretch end in the same place, stretchers and headers alike,
    // and no plank hangs over an end; what does not come out even is spread over
    // the joints.

    // Headers side by side over [a, b], spread to meet both ends.
    template<class Put>
    void layHeaders(float a, float b, Put&& put) {
        const float length = b - a;
        if (length < WIDE - 0.0001f) return;
        const int n = static_cast<int>(std::floor((length + joint) / (WIDE + joint) + 1e-4f));
        if (n == 1) {
            put(0.5f * (a + b), true);
            return;
        }
        const float pitch = (length - WIDE) / static_cast<float>(n - 1);
        for (int i = 0; i < n; ++i) put(a + WIDE / 2 + static_cast<float>(i) * pitch, true);
    }

    // A stretcher course on [a, b]: whole stretchers from the fixed ends (from
    // the middle between two free ones), headers in what is left. As many
    // stretchers as leave every joint a hairline; failing that, the count that
    // leaves the tightest joints.
    template<class Put>
    void layStretchers(float a, float b, bool fixedA, bool fixedB, Put&& put) {
        const float length = b - a;
        if (length < WIDE - 0.0001f) return;
        const int most = std::max(0, static_cast<int>(std::floor((length + joint) / (LEN + joint) + 1e-4f)));
        // a stretcher against every fixed end (a jamb corbels out on stretchers), where they fit
        const int ends = (fixedA ? 1 : 0) + (fixedB ? 1 : 0);
        const int least = std::min(most, ends);
        // the joint each count of stretchers leaves (spread over the joints; a lone plank leaves it at an end)
        const auto joints = [&](int c, int& h) {
            h = std::max(0, static_cast<int>(std::floor((length + joint - static_cast<float>(c) * (LEN + joint)) / (WIDE + joint) + 1e-4f)));
            const int planks = c + h;
            if (planks == 0) return 1e9f;
            const float free = length - static_cast<float>(c) * LEN - static_cast<float>(h) * WIDE;
            return planks > 1 ? free / static_cast<float>(planks - 1) : free;
        };
        int count = -1, headers = 0;
        // the most stretchers, with one at every fixed end, that leave hairline joints
        for (int c = most; c >= least && count < 0; --c) {
            int h;
            if (joints(c, h) <= 0.004f) count = c, headers = h;
        }
        // failing that, the tightest joints of any count
        if (count < 0) {
            float best = 1e9f;
            for (int c = most; c >= std::max(0, most - 8); --c) {
                int h;
                const float g = joints(c, h);
                if (g < best - 0.0005f) best = g, count = c, headers = h;
            }
        }
        if (count < 0) return;
        const float gap = std::max(0.f, joints(count, headers));
        if (count + headers == 0) return;
        // the order along the course: stretchers against the fixed ends
        std::vector<bool> stretcher;
        const auto add = [&](int n, bool isStretcher) { stretcher.insert(stretcher.end(), static_cast<size_t>(n), isStretcher); };
        if (fixedA && !fixedB) {
            add(count, true), add(headers, false);
        } else if (!fixedA && fixedB) {
            add(headers, false), add(count, true);
        } else if (fixedA && fixedB) {
            add((count + 1) / 2, true), add(headers, false), add(count / 2, true);
        } else {
            add(headers / 2, false), add(count, true), add(headers - headers / 2, false);
        }
        if (stretcher.size() == 1) {
            const float w = stretcher[0] ? LEN : WIDE;
            put(fixedA && !fixedB ? a + w / 2 : !fixedA && fixedB ? b - w / 2 : 0.5f * (a + b), !stretcher[0]);
            return;
        }
        float at = a;
        for (const bool isStretcher : stretcher) {
            const float w = isStretcher ? LEN : WIDE;
            put(at + w / 2, !isStretcher);
            at += w + gap;
        }
    }

    inline void buildWall(Design& d, const Wall& w, const std::string& name) {
        const int e = d.element(name);
        const float dx = w.to.x - w.from.x, dz = w.to.z - w.from.z;
        const float length = std::sqrt(dx * dx + dz * dz);
        const Vec2 u{dx / length, dz / length};
        const Vec2 nrm{-u.z, u.x};
        const float along = std::atan2(u.z, u.x);

        // an opening that would leave less than two headers of wall beside it is left out
        std::vector<Opening> openings;
        for (const auto& o : w.openings)
            if (o.at - o.halfWidth >= 2 * WIDE && o.at + o.halfWidth <= length - 2 * WIDE) openings.push_back(o);
        std::sort(openings.begin(), openings.end(), [](const Opening& a, const Opening& b) { return a.at < b.at; });
        std::vector<std::vector<float>> profiles;
        for (const auto& o : openings) profiles.push_back(openingProfile(o, w.courses));
        // the first stretcher course over each opening once it has closed
        std::vector<int> closedFrom(openings.size(), w.courses + 1);
        for (size_t i = 0; i < openings.size(); ++i) {
            bool opened = false;
            for (int k = 0; k < w.courses; k += 2) {
                if (profiles[i][k] > 0) opened = true;
                else if (opened) {
                    closedFrom[i] = k;
                    break;
                }
            }
        }

        for (int k = 0; k < w.courses; ++k) {
            const float s0 = w.trimFrom ? w.trimFrom(k) : 0.f;
            const float s1 = length - (w.trimTo ? w.trimTo(k) : 0.f);
            if (s1 - s0 < (w.trimFrom && w.trimTo ? 3 * WIDE : WIDE)) break;
            const float shift = k >= w.leanFrom ? w.lean * static_cast<float>(k - w.leanFrom + 1) : 0.f;
            const int course = w.base + k;
            const auto put = [&](float s, bool header) {
                if (header) {
                    d.add(w.from.x + u.x * s + nrm.x * shift, w.from.z + u.z * s + nrm.z * shift, along + PI / 2, course, e);
                    return;
                }
                for (int row = -1; row <= 1; row += 2) {
                    const float off = shift + static_cast<float>(row) * (LEN / 2 - WIDE / 2);
                    d.add(w.from.x + u.x * s + nrm.x * off, w.from.z + u.z * s + nrm.z * off, along, course, e);
                }
            };

            // the solid stretches of this course; an end against an opening is fixed
            struct Span {
                float a, b;
                bool fixedA, fixedB;
            };
            std::vector<Span> spans{{s0, s1, w.fixedFrom, w.fixedTo}};
            for (size_t i = 0; i < openings.size(); ++i) {
                const float hw = profiles[i][k];
                const float c = openings[i].at;
                if (hw <= 0) continue;
                std::vector<Span> next;
                for (const auto& s : spans) {
                    if (c + hw <= s.a || c - hw >= s.b) {
                        next.push_back(s);
                        continue;
                    }
                    if (c - hw > s.a + WIDE) next.push_back({s.a, c - hw, s.fixedA, true});
                    if (c + hw < s.b - WIDE) next.push_back({c + hw, s.b, true, s.fixedB});
                }
                spans = std::move(next);
            }

            for (const auto& s : spans) {
                if (k % 2 == 1) {
                    layHeaders(s.a, s.b, put);
                    continue;
                }
                // A closed arch keeps a key stretcher over its middle in every
                // stretcher course above it, and the stretchers run from the keys
                // as from the jambs: the course that closes an arch rests on both sides.
                float from = s.a;
                bool fixed = s.fixedA;
                for (size_t i = 0; i < openings.size(); ++i) {
                    const float key = openings[i].at;
                    if (k < closedFrom[i] || key - LEN / 2 < from - 0.0001f || key + LEN / 2 > s.b + 0.0001f) continue;
                    layStretchers(from, key - LEN / 2 - joint, fixed, true, put);
                    put(key, false);
                    from = key + LEN / 2 + joint;
                    fixed = true;
                }
                layStretchers(from, s.b, fixed, s.fixedB, put);
            }
        }
    }

    // ---------------------------------------------------------------- ridges

    // Courses along a roof's ridge, over the last courses of two leaning walls
    // that have met: rows of planks along the ridge, narrowing by `step` on each
    // side per course; every other course has a row of planks across its
    // middle, resting on both sides, to tie them together.
    struct Ridge {
        Vec2 from{}, to{};// along the ridge
        int base = 0;
        float halfWidth = LEN;
        float step = 0.0045f;
        // metres drawn in at either end for a row whose nearest edge is at a distance from the ridge line (clearing a ring)
        std::function<float(float)> trimFrom, trimTo;
    };

    inline int buildRidge(Design& d, const Ridge& r, const std::string& name) {
        const int e = d.element(name);
        const float dx = r.to.x - r.from.x, dz = r.to.z - r.from.z;
        const float length = std::sqrt(dx * dx + dz * dz);
        const Vec2 u{dx / length, dz / length};
        const Vec2 nrm{-u.z, u.x};
        const float along = std::atan2(u.z, u.x);
        const auto at = [&](float s, float off) { return Vec2{r.from.x + u.x * s + nrm.x * off, r.from.z + u.z * s + nrm.z * off}; };
        int k = 0;
        for (;; ++k) {
            const float hw = r.halfWidth - r.step * static_cast<float>(k);
            if (hw < WIDE + joint / 2) break;// ends two rows wide: a lone top row slides
            const int course = r.base + k;
            const auto ends = [&](float reach) {
                return std::pair{r.trimFrom ? r.trimFrom(reach) : 0.f, length - (r.trimTo ? r.trimTo(reach) : 0.f)};
            };
            // a row of stretchers at a distance off from the ridge line
            const auto row = [&](float off) {
                const auto [s0, s1] = ends(std::max(0.f, std::abs(off) - WIDE / 2));
                for (float s = s0 + LEN / 2; s + LEN / 2 <= s1 + 0.0001f; s += LEN + joint) {
                    const auto p = at(s, off);
                    d.add(p.x, p.z, along, course, e);
                }
            };
            float inner = -1;// the half-width taken by a tie across the middle
            if (k % 2 == 1 && hw >= LEN / 2 + 0.0005f) {
                const auto [s0, s1] = ends(0.f);
                const int n = static_cast<int>(std::floor((s1 - s0 + joint) / (WIDE + joint)));
                const float pitch = n > 1 ? (s1 - s0 - WIDE) / static_cast<float>(n - 1) : 0.f;
                for (int i = 0; i < n; ++i) {
                    const auto p = at(s0 + WIDE / 2 + static_cast<float>(i) * pitch, 0);
                    d.add(p.x, p.z, along + PI / 2, course, e);
                }
                inner = LEN / 2 + joint;
            }
            if (inner < 0) {
                const int rows = static_cast<int>(std::floor((2 * hw + joint) / (WIDE + joint)));
                for (int i = 0; i < rows; ++i) row((static_cast<float>(i) - 0.5f * static_cast<float>(rows - 1)) * (WIDE + joint));
            } else {
                const int rows = static_cast<int>(std::floor((hw - inner + joint) / (WIDE + joint)));
                for (int i = 0; i < rows; ++i) {
                    const float off = inner + WIDE / 2 + static_cast<float>(i) * (WIDE + joint);
                    row(off);
                    row(-off);
                }
            }
        }
        return r.base + k;
    }

    // ---------------------------------------------------------------- the plan

    struct Plan {
        float naveInner;   // between the nave walls' inner faces
        int bays;
        float bayLength;
        int naveCourses;   // to the eaves
        float pitch;       // of the roofs, degrees
        bool aisles;
        int aisleCourses;
        float aisleGap;    // between an aisle wall and the nave wall
        int towerPlanks;   // the west towers' first tier
        int tiers;
        bool galleries;
        float spireStep;   // in per course
        bool lantern;
        int crossingPlanks;
        float crossingApothem;
        bool dome;
        int drumCourses;
        int armStretchers;
        int choirStretchers;

        // where things ended up, for the cameras and the ball
        float naveStart = 0, naveEnd = 0, crossingX = 0, apseX = 0, zWall = 0, towerZ = 0, towerReach = 0;
        int ridgeCourse = 0;// the top of the nave's ridge
    };

    // scale: 1 is the full building; less makes it lower and shorter, for quick probes
    inline Plan makePlan(std::uint64_t seed, float scale) {
        Rng rng(seed * 7919 + 17);
        Plan p{};
        p.naveInner = rng.uniform(0.19f, 0.22f);
        p.bays = std::max(2, static_cast<int>(std::lround(static_cast<float>(rng.range(4, 5)) * scale)));
        p.bayLength = rng.uniform(0.33f, 0.37f);
        p.naveCourses = 2 * static_cast<int>(std::lround(static_cast<float>(rng.range(46, 52)) * scale));
        p.pitch = rng.uniform(56.f, 62.f);
        p.aisles = rng.chance(0.85f);
        p.aisleCourses = 2 * static_cast<int>(static_cast<float>(p.naveCourses) * rng.uniform(0.40f, 0.46f) / 2);
        p.aisleGap = rng.uniform(0.075f, 0.09f);
        p.towerPlanks = rng.chance(0.5f) ? 9 : 10;
        p.tiers = rng.range(3, 4);
        p.galleries = rng.chance(0.8f);
        p.spireStep = rng.uniform(0.0008f, 0.0011f);
        p.lantern = false;
        rng.uniform();
        p.dome = rng.chance(0.4f);
        p.drumCourses = 2 * rng.range(8, 14);
        p.armStretchers = 3 + (rng.chance(0.3f) ? 1 : 0);
        p.choirStretchers = 3;

        // The crossing must be wide enough that the nave's and the transept's
        // walls, ending against it, clear each other at the corners between them.
        const float zFront = p.naveInner / 2 + LEN, gap = 0.004f;
        const float reach = std::sqrt((zFront + gap) * (zFront + gap) + (p.naveInner / 2) * (p.naveInner / 2)) - gap + 0.003f;
        const float need = std::sqrt(reach * reach - (LEN / 2) * (LEN / 2)) - WIDE / 2;
        int n = 24;
        while (n > 8 && ringMin(n) + 0.026f > need) --n;
        p.crossingPlanks = n;
        p.crossingApothem = std::clamp(need, ringMin(n) + 0.026f, ringMax(n));
        return p;
    }

    // Single elements, to try the physics on: "stack", "ring", "wall", "tower", "roof".
    inline Design buildTest(const std::string& what, int courses) {
        Design d;
        if (what == "stack") {
            const int e = d.element("stack");
            for (int k = 0; k < courses; ++k) d.add(0, 0, (k % 2) ? PI / 2 : 0.f, k, e);
        } else if (what == "ring") {
            Tower t;
            t.tiers.push_back({12, ringMin(12) + 0.01f, courses, false, {}});
            t.top = Tower::Top::None;
            buildTower(d, t, "ring");
        } else if (what == "tower") {
            Tower t;
            t.tiers.push_back({10, ringMin(10) + 0.022f, courses / 2, true, {{courses / 2 - 22, 12, PI / 4}, {courses / 2 - 22, 12, 3 * PI / 4}}});
            t.tiers.push_back({9, ringMin(9) + 0.021f, courses / 4, true, {{courses / 4 - 18, 10, PI / 4}}});
            t.tiers.push_back({8, ringMin(8) + 0.020f, courses / 4, false, {}});
            t.top = Tower::Top::Cone;
            t.coneStep = 0.0015f;
            buildTower(d, t, "tower");
        } else if (what == "roof") {
            // two leaning walls meeting under a ridge
            const float W = 0.2f, s = (W / 2 - 0.0003f) / 22;
            for (int side = -1; side <= 1; side += 2) {
                Wall w;
                w.from = {0, static_cast<float>(side) * (W / 2 + LEN / 2)};
                w.to = {0.8f, static_cast<float>(side) * (W / 2 + LEN / 2)};
                w.courses = courses + 22;
                w.lean = -static_cast<float>(side) * s;
                w.leanFrom = courses;
                w.fixedFrom = w.fixedTo = false;
                buildWall(d, w, side < 0 ? "north wall" : "south wall");
            }
            Ridge r;
            r.from = {0, 0};
            r.to = {0.8f, 0};
            r.base = courses + 22;
            r.halfWidth = LEN - s;
            r.step = s;
            buildRidge(d, r, "ridge");
        } else {
            Wall w;
            w.from = {0, 0};
            w.to = {1.0f, 0};
            w.courses = courses;
            w.fixedFrom = w.fixedTo = false;
            w.openings.push_back({0.5f, 0.06f, courses / 5 & ~1, courses / 2 & ~1, 1.f, false});
            buildWall(d, w, "wall");
        }
        d.finish();
        return d;
    }

    inline Design buildCathedral(std::uint64_t seed, float scale, Plan* planOut = nullptr, bool explain = false) {
        Plan p = makePlan(seed, scale);
        Rng rng(seed * 104729 + 3);
        Design d;
        d.explain = explain;

        const float gap = 0.004f;
        const float W = p.naveInner;
        const float zWall = W / 2 + LEN / 2;// nave wall centreline
        const float zF = W / 2 + LEN;       // the nave walls' outer faces
        const int H = p.naveCourses;        // the eaves
        const float step = THICK / std::tan(p.pitch * PI / 180.f);

        // The roofs: the walls lean in from the eaves until their inner faces
        // meet over the middle, the ridge closes the rest. An even number of
        // courses to the meeting, so that the last leaning course is one of
        // headers, a whole course under the ridge.
        int leanCourses = std::max(4, static_cast<int>(std::lround((W / 2) / step)));
        if ((H + leanCourses) % 2) ++leanCourses;
        const float s = (W / 2 - 0.0003f) / static_cast<float>(leanCourses);
        const int top = H + leanCourses;// first course of a ridge
        const int ridgeCourses = static_cast<int>(std::ceil((LEN - s - WIDE / 2 - 0.001f) / s)) + 1;

        // the crossing ring, and where the walls meet it
        const float crossReach = ringReach(p.crossingApothem) + gap;
        const float meet = std::sqrt(crossReach * crossReach - (W / 2) * (W / 2));
        // a leaning wall's end against a ring is drawn back as the wall leans in
        const auto ringTrim = [=](int k) -> float {
            if (k < H) return 0.f;
            const float inner = std::max(0.f, W / 2 - s * static_cast<float>(k - H + 1));
            return std::sqrt(crossReach * crossReach - inner * inner) - meet;
        };
        // and a ridge's rows, by how far they reach from the ridge line
        const auto ridgeTrim = [=](float reach) -> float {
            if (reach >= crossReach) return 0.f;
            return std::max(0.f, std::sqrt(crossReach * crossReach - reach * reach) - meet);
        };

        const float naveStart = LEN / 2 + gap;
        const float naveEnd = naveStart + static_cast<float>(p.bays) * p.bayLength;
        const Vec2 crossing{naveEnd + meet, 0};
        const float pier = 2 * LEN + joint;
        const float bayHalf = std::max(0.04f, (p.bayLength - pier) / 2);// a window's half-width

        // the aisles' walls and their lean-to roofs, leaning in toward the nave wall
        const int Ha = p.aisleCourses;
        const float G = p.aisleGap;
        int aisleLean = std::max(4, static_cast<int>(std::lround(G / 0.0058f)));
        const float sa = (G - 0.0012f) / static_cast<float>(aisleLean);
        const int aisleTop = Ha + aisleLean;

        // --- the crossing tower first: where walls meet it, its planks come first
        {
            Tower t;
            t.centre = crossing;
            const int ridgeTop = top + ridgeCourses;
            Tier drum{p.crossingPlanks, p.crossingApothem, ridgeTop + p.drumCourses, p.galleries, {}};
            const int windows = p.crossingPlanks >= 10 ? 4 : 3;
            for (int i = 0; i < windows; ++i)
                drum.windows.push_back({ridgeTop + 4, std::max(4, (p.drumCourses - 14) & ~1), PI / 4 + 2 * PI * static_cast<float>(i) / static_cast<float>(windows)});
            t.tiers.push_back(drum);
            if (!p.dome && p.crossingPlanks >= 10) {
                const int n2 = p.crossingPlanks - 3;
                Tier belfry{n2, ringMin(n2) + 0.021f, 2 * (p.drumCourses / 3 + 12), false, {}};
                for (int i = 0; i < 4; ++i) belfry.windows.push_back({16, belfry.courses - 24, PI / 4 + PI / 2 * static_cast<float>(i)});
                t.tiers.push_back(belfry);
            }
            t.top = p.dome ? Tower::Top::Dome : Tower::Top::Cone;
            t.coneStep = p.dome ? 0.0045f : p.spireStep * 1.15f;
            t.spireFoot = ringMin(6) + 0.003f;
            buildTower(d, t, "crossing");
        }

        // --- the apse, a ring at the east end with a cone at the roofs' pitch
        const float choirLength = static_cast<float>(p.choirStretchers) * (LEN + joint) - joint;
        const float choirX0 = crossing.x + meet, choirX1 = choirX0 + choirLength;
        const Vec2 apse{choirX1 + meet, 0};
        {
            Tower t;
            t.centre = apse;
            Tier shaft{p.crossingPlanks, p.crossingApothem, H, false, {}};
            const int sill = (H * 3 / 10) & ~1;
            for (int i = 0; i < 3; ++i) shaft.windows.push_back({sill, (H * 2 / 5) & ~1, PI * (-0.3f + 0.3f * static_cast<float>(i))});
            t.tiers.push_back(shaft);
            t.top = Tower::Top::Cone;
            t.coneStep = step;
            buildTower(d, t, "apse");
        }

        // --- the west front: a portal, a round window over it, a gable above the roof
        {
            Wall w;
            w.from = {0, -zF};
            w.to = {0, zF};
            w.fixedFrom = w.fixedTo = false;
            const int gableFrom = (H + 6) & ~1;
            w.courses = gableFrom + static_cast<int>(std::ceil(zF / s)) + 2;
            w.trimFrom = w.trimTo = [=](int k) { return k < gableFrom ? 0.f : s * static_cast<float>(k - gableFrom + 1); };
            Opening portal;
            portal.at = zF;
            portal.halfWidth = rng.uniform(0.072f, 0.088f);
            portal.sill = 0;
            portal.spring = static_cast<int>(static_cast<float>(H) * rng.uniform(0.34f, 0.40f)) & ~1;
            portal.rise = 1.f;
            w.openings.push_back(portal);
            Opening rose;
            rose.round = true;
            rose.at = zF;
            rose.halfWidth = rng.uniform(0.068f, 0.082f);
            rose.spring = (H - 2) & ~1;
            rose.sill = rose.spring - static_cast<int>(rose.halfWidth / THICK) - 2;
            w.openings.push_back(rose);
            buildWall(d, w, "west front");
        }

        // --- the west towers: tiers narrowing up to a spire
        Tower west;
        {
            Rng trng(seed * 31 + 7);
            int n = p.towerPlanks;
            const int t1 = (H + 8) & ~1;
            const std::vector<int> heights{t1, static_cast<int>(static_cast<float>(H) * 0.40f) & ~1,
                                           static_cast<int>(static_cast<float>(H) * 0.32f) & ~1, static_cast<int>(static_cast<float>(H) * 0.26f) & ~1};
            for (int i = 0; i < p.tiers; ++i) {
                Tier tier{n, ringMin(n) + 0.022f, heights[std::min(i, 3)], p.galleries && i + 1 < p.tiers + (p.lantern ? 0 : 1), {}};
                const int windows = n >= 9 ? 4 : 3;
                const int h = (i == 0 ? 14 : 12);
                const int at = (tier.courses - (tier.gallery ? galleryCourses : 0) - h - 6) & ~1;
                for (int j = 0; j < windows; ++j) tier.windows.push_back({at, h, PI / 4 + 2 * PI * static_cast<float>(j) / static_cast<float>(windows)});
                if (i == 0) {// a lower row on the foot tier
                    const int low = (tier.courses * 2 / 5) & ~1;
                    for (int j = 0; j < windows; ++j) tier.windows.push_back({low, 10, PI / 4 + 2 * PI * static_cast<float>(j) / static_cast<float>(windows)});
                }
                west.tiers.push_back(tier);
                n = std::max(6, n - (i == 0 ? 2 : 1));
            }
            west.top = p.lantern ? Tower::Top::Lantern : Tower::Top::Cone;
            west.coneStep = p.spireStep;
            west.spireFoot = ringMin(6) + 0.003f;
        }
        const float westReach = towerReach(west);
        const float zT = zF + gap + westReach;
        for (int side = -1; side <= 1; side += 2) {
            Tower t = west;
            t.centre = {0, static_cast<float>(side) * zT};
            buildTower(d, t, side < 0 ? "north-west tower" : "south-west tower");
        }

        // --- the nave: walls with arcades below and tall windows above, leaning in to the roof
        for (int side = -1; side <= 1; side += 2) {
            Wall w;
            w.from = {naveStart, static_cast<float>(side) * zWall};
            w.to = {naveEnd, static_cast<float>(side) * zWall};
            w.courses = top;
            w.lean = -static_cast<float>(side) * s;
            w.leanFrom = H;
            w.trimTo = ringTrim;
            for (int b = 0; b < p.bays; ++b) {
                const float at = (static_cast<float>(b) + 0.5f) * p.bayLength;
                const int arch = static_cast<int>(std::ceil(1.8f * bayHalf / THICK));
                if (p.aisles) {
                    w.openings.push_back({at, bayHalf, 0, (Ha * 3 / 5) & ~1, 1.f, false});
                    w.openings.push_back({at, bayHalf, (aisleTop + 4) & ~1, (H - arch - 4) & ~1, 1.f, false});
                } else {
                    w.openings.push_back({at, bayHalf, (H * 3 / 10) & ~1, (H - arch - 4) & ~1, 1.f, false});
                }
            }
            buildWall(d, w, side < 0 ? "north nave wall" : "south nave wall");
        }
        {
            Ridge r;
            r.from = {naveStart, 0};
            r.to = {naveEnd, 0};
            r.base = top;
            r.halfWidth = LEN - s;
            r.step = s;
            r.trimTo = ridgeTrim;
            p.ridgeCourse = buildRidge(d, r, "nave roof");
        }

        // --- the aisles, behind the west towers to the transept, with buttresses
        if (p.aisles) {
            const float x0 = westReach + gap, x1 = crossing.x - zF - gap;
            const float zA = zF + G + LEN / 2;
            for (int side = -1; side <= 1; side += 2) {
                const float sz = static_cast<float>(side);
                Wall w;
                w.from = {x0, sz * zA};
                w.to = {x1, sz * zA};
                w.courses = aisleTop;
                w.lean = -sz * sa;
                w.leanFrom = Ha;
                w.fixedFrom = false;
                w.trimFrom = [=](int k) { return k < Ha ? 0.f : static_cast<float>(k - Ha + 1) * (WIDE + joint); };
                for (int b = 0; b < p.bays; ++b) {
                    const float at = naveStart + (static_cast<float>(b) + 0.5f) * p.bayLength - x0;
                    const int arch = static_cast<int>(std::ceil(1.8f * bayHalf / THICK));
                    w.openings.push_back({at, bayHalf, 10, (Ha - arch - 4) & ~1, 1.f, false});
                }
                buildWall(d, w, side < 0 ? "north aisle" : "south aisle");

                for (int b = 1; b < p.bays; ++b) {
                    const float xb = naveStart + static_cast<float>(b) * p.bayLength;
                    if (xb - LEN / 2 < x0 + 0.01f) continue;
                    const float z0 = zA + LEN / 2 + gap;
                    Wall bt;
                    bt.from = {xb, sz * z0};
                    bt.to = {xb, sz * (z0 + 3 * WIDE + 2 * joint)};
                    bt.courses = (aisleTop + 4) & ~1;
                    const int stepAt = (Ha - 6) & ~1;// stepping back a header above the aisle windows
                    bt.trimTo = [=](int k) { return k >= stepAt ? WIDE + joint : 0.f; };
                    bt.fixedTo = false;
                    buildWall(d, bt, "buttress");
                }
            }
        }

        // --- the transept: two arms, gabled ends with a tall window and a round one
        const float armLength = static_cast<float>(p.armStretchers) * (LEN + joint) - joint;
        for (int side = -1; side <= 1; side += 2) {
            const float sz = static_cast<float>(side);
            const float armStart = meet, armEnd = meet + armLength;
            for (int wside = -1; wside <= 1; wside += 2) {
                Wall w;
                const float x = crossing.x + static_cast<float>(wside) * zWall;
                w.from = {x, sz * armStart};
                w.to = {x, sz * armEnd};
                // to the eaves: short walls leaning in as a roof did not stand
                w.courses = H;
                if (armLength > 2 * (0.045f + LEN)) {
                    const int arch = static_cast<int>(std::ceil(1.8f * 0.045f / THICK));
                    w.openings.push_back({armLength / 2, 0.045f, (H * 3 / 10) & ~1, (H - arch - 4) & ~1, 1.f, false});
                }
                buildWall(d, w, side < 0 ? "north transept" : "south transept");
            }
            Wall end;
            const float zEnd = sz * (armEnd + gap + LEN / 2);
            end.from = {crossing.x - zF, zEnd};
            end.to = {crossing.x + zF, zEnd};
            end.fixedFrom = end.fixedTo = false;
            const int gableFrom = (H + 4) & ~1;
            end.courses = gableFrom + static_cast<int>(std::ceil(zF / s)) + 2;
            end.trimFrom = end.trimTo = [=](int k) { return k < gableFrom ? 0.f : s * static_cast<float>(k - gableFrom + 1); };
            {
                const float hw = 0.062f;
                const int arch = static_cast<int>(std::ceil(1.8f * hw / THICK));
                end.openings.push_back({zF, hw, (H / 5) & ~1, (H * 3 / 5 - arch) & ~1, 1.f, false});
                Opening rose;
                rose.round = true;
                rose.at = zF;
                rose.halfWidth = 0.06f;
                rose.spring = (H - 4) & ~1;
                rose.sill = rose.spring - 10;
                end.openings.push_back(rose);
            }
            buildWall(d, end, side < 0 ? "north transept end" : "south transept end");

        }

        // --- the choir, between the crossing and the apse
        for (int side = -1; side <= 1; side += 2) {
            Wall w;
            w.from = {choirX0, static_cast<float>(side) * zWall};
            w.to = {choirX1, static_cast<float>(side) * zWall};
            w.courses = H;
            buildWall(d, w, side < 0 ? "north choir wall" : "south choir wall");
        }

        d.finish();
        p.naveStart = naveStart;
        p.naveEnd = naveEnd;
        p.crossingX = crossing.x;
        p.apseX = apse.x;
        p.zWall = zWall;
        p.towerZ = zT;
        p.towerReach = westReach;
        if (planOut) *planOut = p;
        return d;
    }

}// namespace planks

#endif//PLANKCATHEDRAL_CATHEDRAL_HPP
