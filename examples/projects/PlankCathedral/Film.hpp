// The film's harness: the words on screen, the stamps that let a render in
// chunks resume, ffmpeg, and the way a building rises course by course.
// main.cpp plans the shots and cameras from the buildings and draws the frames.
//
// The film is drawn headless into a clean master: the frames go raw to ffmpeg,
// one chunk of the film at a time (chunks/chunk_NNN.mkv). Each chunk carries a
// stamp of everything that went into its pixels: every frame's shot, camera and
// moment, both buildings' designs, the recording (its header, its size and a
// sample of its frames), the frame size, the sampling and the look. A chunk
// whose stamp differs is drawn again, so a stopped render resumes and a stale
// chunk never comes back after a change to the building or the recording.
// The film's light is a table (main.cpp's FilmLook) and is in the stamp; a
// change to the look's code instead (Look.hpp's shaders, the passes) must bump
// lookCode below, or the render be started with --film-fresh.
//
// No word is in the master. captions.ass, written from the one table below
// together with captions.md, is burned in by ffmpeg in a separate pass, so a
// changed word never means drawing again.

#ifndef PLANKCATHEDRAL_FILM_HPP
#define PLANKCATHEDRAL_FILM_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace planks::film {

    namespace fs = std::filesystem;

    // Bump when the look's code changes (Look.hpp's shaders, the passes): it is in the chunks' stamps.
    constexpr int lookCode = 1;

    constexpr int fps = 60;
    constexpr int chunkFrames = 4 * fps;

    // ------------------------------------------------------------------ stamps

    struct Hash {
        std::uint64_t h = 1469598103934665603ull;

        void bytes(const void* data, std::size_t n) {
            const auto* p = static_cast<const unsigned char*>(data);
            for (std::size_t i = 0; i < n; ++i) {
                h ^= p[i];
                h *= 1099511628211ull;
            }
        }
        template<class T>
        void value(const T& v) { bytes(&v, sizeof(T)); }
        void text(const std::string& s) { bytes(s.data(), s.size()); }

        [[nodiscard]] std::string hex() const {
            char b[24];
            std::snprintf(b, sizeof(b), "%016llx", static_cast<unsigned long long>(h));
            return b;
        }
    };

    inline std::string contentsOf(const fs::path& p) {
        std::ifstream in(p, std::ios::binary);
        std::stringstream s;
        s << in.rdbuf();
        return s.str();
    }

    // ------------------------------------------------------------------ ffmpeg

    inline std::string q(const fs::path& p) { return "\"" + p.string() + "\""; }

    inline int runCommand(std::string line) {
#ifdef _WIN32
        // cmd /c strips one pair of outer quotes from a line that starts with one
        line = "\"" + line + "\"";
#endif
        return std::system(line.c_str());
    }

    inline std::FILE* openPipe(const std::string& line) {
#ifdef _WIN32
        return _popen(("\"" + line + "\"").c_str(), "wb");
#else
        return popen(line.c_str(), "w");
#endif
    }

    inline int closePipe(std::FILE* f) {
#ifdef _WIN32
        return _pclose(f);
#else
        return pclose(f);
#endif
    }

    // "cd" into a folder first, so that the ass filters name their files without a drive's colon
    inline std::string inFolder(const fs::path& dir) {
#ifdef _WIN32
        return "cd /d " + q(dir) + " && ";
#else
        return "cd " + q(dir) + " && ";
#endif
    }

    // Raw RGB frames, bottom row first (as GL reads them), into a chunk of the clean master.
    inline std::string encoder(const fs::path& ffmpeg, const fs::path& out, int w, int h) {
        return q(ffmpeg) + " -y -loglevel error -f rawvideo -pix_fmt rgb24 -s " + std::to_string(w) + "x" + std::to_string(h) + " -framerate " +
               std::to_string(fps) + " -i - -vf vflip -c:v libx264 -preset fast -crf 8 -pix_fmt yuv444p " + q(out);
    }

    // ------------------------------------------------------------------ the shots

    struct Shot {
        int number;
        std::string name;
        double t0, t1;// film seconds
    };

    inline std::string mmss(double t) {
        char b[32];
        std::snprintf(b, sizeof(b), "%d:%05.2f", static_cast<int>(t) / 60, std::fmod(t, 60.0));
        return b;
    }

    // ------------------------------------------------------------------ the words

    // Every word on screen, in one table: the captions (plain white, a soft
    // shadow, the lower third) and the end card's lines.
    struct Caption {
        int shot;          // the shot it belongs to
        double in, out;    // film seconds
        std::string text;
        std::string style; // Caption, Title, Line, Link
        int y = 0;         // the end card's lines: their centre, in 1080p pixels
    };

    inline std::string thousands(long long n) {
        std::string s = std::to_string(n), out;
        for (std::size_t i = 0; i < s.size(); ++i) {
            if (i > 0 && (s.size() - i) % 3 == 0) out += ',';
            out += s[i];
        }
        return out;
    }

    // The drafts, word for word (Lars approves them before the render). Each
    // is up for 2-4 s inside its shot, never over the impact.
    inline std::vector<Caption> captionTable(const std::vector<Shot>& shots, long long planks, double slowHit) {
        const auto shot = [&](int n) -> const Shot& {
            for (const auto& s : shots)
                if (s.number == n) return s;
            return shots.front();
        };
        std::vector<Caption> c;
        const auto& s1 = shot(1);
        c.push_back({1, s1.t0 + 0.6, std::min(s1.t1 - 0.2, s1.t0 + 4.1), "This cathedral is made of " + thousands(planks) + " wooden planks.", "Caption"});
        const auto& s2 = shot(2);
        c.push_back({2, s2.t0 + 0.3, std::min(s2.t1 - 0.2, s2.t0 + 4.2), "No one placed them. A program designed it from a single number.", "Caption"});
        const auto& s3 = shot(3);
        c.push_back({3, s3.t0 + 1.0, s3.t0 + 4.8, "PhysX checks that every design stands before it is shown.", "Caption"});
        // it announces the replay: up while the ball comes in when there are 2 s for it before the hit; otherwise
        // over the settled rubble at the end of shot 5, running a moment across the cut and gone before the hit
        const auto& s5 = shot(5);
        const auto& s6 = shot(6);
        if (slowHit - 0.1 - (s6.t0 + 0.15) >= 2.0)
            c.push_back({6, s6.t0 + 0.15, std::min(slowHit - 0.1, s6.t0 + 3.6), "The same collapse, four times slower.", "Caption"});
        else
            c.push_back({5, s5.t1 - 2.4, std::min(s6.t0 + 0.4, slowHit - 0.3), "The same collapse, four times slower.", "Caption"});
        const auto& s7 = shot(7);
        c.push_back({7, s7.t0 + 0.6, std::min(s7.t1 - 0.1, s7.t0 + 3.8), "Another number, another cathedral.", "Caption"});
        const auto& s8 = shot(8);
        c.push_back({8, s8.t0 + 0.2, s8.t1, "threepp", "Title", 470});
        c.push_back({8, s8.t0 + 0.4, s8.t1, "A C++ 3D library. Everything here is made in code.", "Line", 572});
        c.push_back({8, s8.t0 + 0.6, s8.t1, "github.com/markaren/threepp", "Link", 646});
        return c;
    }

    inline std::string assTime(double t) {
        t = std::max(0.0, t);
        const int cs = static_cast<int>(std::lround(t * 100));
        char b[32];
        std::snprintf(b, sizeof(b), "%d:%02d:%02d.%02d", cs / 360000, (cs / 6000) % 60, (cs / 100) % 60, cs % 100);
        return b;
    }

    // The end card: the frame dimmed, a dark panel, three lines on it.
    constexpr int panelX0 = 330, panelX1 = 1590, panelY0 = 380, panelY1 = 712;

    inline std::string assFile(const std::vector<Caption>& captions, const std::vector<Shot>& shots) {
        std::string s =
                "[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
                "[V4+ Styles]\n"
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, "
                "StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
                "Style: Caption,Segoe UI,50,&H00FFFFFF,&H00FFFFFF,&HB4000000,&H90000000,0,0,0,0,100,100,0.4,0,1,2.6,1.6,2,180,180,150,1\n"
                "Style: Title,Segoe UI Semibold,96,&H00FFFFFF,&H00FFFFFF,&HC8000000,&HB4000000,0,0,0,0,100,100,1,0,1,2,1.5,5,0,0,0,1\n"
                "Style: Line,Segoe UI,42,&H00FFFFFF,&H00FFFFFF,&HC8000000,&HB4000000,0,0,0,0,100,100,0.4,0,1,1.6,1.2,5,0,0,0,1\n"
                "Style: Link,Segoe UI,36,&H00D8DEE4,&H00D8DEE4,&HC8000000,&HB4000000,0,0,0,0,100,100,0.6,0,1,1.4,1.0,5,0,0,0,1\n"
                "Style: Box,Segoe UI,20,&H00000000,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n\n"
                "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n";
        const auto add = [&](int layer, double t0, double t1, const std::string& style, const std::string& text) {
            s += "Dialogue: " + std::to_string(layer) + "," + assTime(t0) + "," + assTime(t1) + "," + style + ",,0,0,0,," + text + "\n";
        };
        // the end card's dimmed frame and its panel
        for (const auto& shot : shots) {
            if (shot.number != 8) continue;
            const double end = shot.t1 + 1.0;
            add(0, shot.t0, end, "Box", "{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c&H000000&\\1a&H70&\\fad(500,0)}m 0 0 l 1920 0 1920 1080 0 1080{\\p0}");
            char b[256];
            std::snprintf(b, sizeof(b), "{\\an7\\pos(%d,%d)\\p1\\bord0\\shad0\\blur3\\1c&H0B0C0E&\\1a&H38&\\fad(500,0)}m 0 0 l %d 0 %d %d 0 %d{\\p0}", panelX0,
                          panelY0, panelX1 - panelX0, panelX1 - panelX0, panelY1 - panelY0, panelY1 - panelY0);
            add(1, shot.t0, end, "Box", b);
        }
        for (const auto& c : captions) {
            // a caption reads over pale wood on a soft dark band across the lower third, its top edge blurred away
            if (c.style == "Caption")
                add(1, c.in, c.out, "Box", "{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\blur60\\1c&H000000&\\1a&H90&\\fad(300,300)}m -100 800 l 2020 800 2020 1180 -100 1180{\\p0}");
            if (c.style == "Caption") add(2, c.in, c.out, "Caption", "{\\blur5\\fad(300,300)}" + c.text);
            else add(2, c.in, c.out + 1.0, c.style, "{\\an5\\pos(960," + std::to_string(c.y) + ")\\blur3\\fad(400,0)}" + c.text);
        }
        return s;
    }

    inline std::string captionsMarkdown(const std::vector<Caption>& captions, const std::vector<Shot>& shots, double end) {
        std::string s = "# Plank Cathedral: every word on screen\n\n"
                        "Drafts for Lars's approval. Burned in by ffmpeg from `captions.ass` (written with this file from one table in "
                        "`Film.hpp`); the clean master has no words, so a change here never means rendering again.\n\n"
                        "| Shot | From | To | Text |\n|---|---|---|---|\n";
        for (const auto& c : captions) {
            const double out = c.style == "Caption" ? c.out : end;
            s += "| " + std::to_string(c.shot) + " | " + mmss(c.in) + " | " + mmss(out) + " | " + c.text + " |\n";
        }
        s += "\nThe end card's three lines stand on a dark panel over the dimmed last frame until the film fades out at " + mmss(end) + ".\n\n";
        s += "## Shots\n\n| # | Shot | From | To |\n|---|---|---|---|\n";
        for (const auto& sh : shots) s += "| " + std::to_string(sh.number) + " | " + sh.name + " | " + mmss(sh.t0) + " | " + mmss(sh.t1) + " |\n";
        return s;
    }

    // ------------------------------------------------------------------ the build

    // When each plank of a design starts to drop into its place: the courses
    // rise from the floor one after another, each plank of a course starting a
    // little before or after its neighbours, so that a few courses are always
    // on their way. A course of many planks takes longer than a course of few
    // (by the square root of its count), so the spires do not drag.
    // `seconds` is the whole build, the last plank in its place at its end.
    struct Build {
        std::vector<float> start;// per plank
        float drop = 0.4f;       // seconds a plank takes to fall into its place
        float height = 0.025f;   // metres it falls

        // how far above its place a plank is at `t` (seconds into the build); < 0: not there yet
        [[nodiscard]] float lift(std::size_t i, float t) const {
            const float u = (t - start[i]) / drop;
            if (u < 0) return -1.f;
            if (u >= 1) return 0.f;
            const float v = 1.f - u;
            return height * v * v;// it falls, slows and settles
        }
    };

    template<class CourseOf>
    Build makeBuild(std::size_t planks, CourseOf courseOf, float seconds, float drop, float height) {
        Build b;
        b.drop = drop;
        b.height = height;
        int top = 0;
        for (std::size_t i = 0; i < planks; ++i) top = std::max(top, courseOf(i));
        std::vector<double> count(top + 1, 0.0), from(top + 2, 0.0);
        for (std::size_t i = 0; i < planks; ++i) count[courseOf(i)] += 1;
        for (int c = 0; c <= top; ++c) from[c + 1] = from[c] + std::sqrt(count[c]);
        const double total = std::max(1e-9, from[top + 1]);
        const float span = std::max(0.1f, seconds - drop);
        b.start.resize(planks);
        for (std::size_t i = 0; i < planks; ++i) {
            const int c = courseOf(i);
            // a hash of the plank: where in its course's turn it starts
            std::uint64_t x = (static_cast<std::uint64_t>(i) + 1) * 0x9E3779B97F4A7C15ull;
            x ^= x >> 31;
            x *= 0xBF58476D1CE4E5B9ull;
            x ^= x >> 29;
            const double jitter = static_cast<double>(x >> 11) / 9007199254740992.0;
            const double a = from[c] / total, w = (from[c + 1] - from[c]) / total;
            // spread over its own turn and the next two
            b.start[i] = static_cast<float>(std::min(1.0, a + jitter * 3.0 * w) * span);
        }
        return b;
    }

    inline float smooth(float u) {
        u = std::clamp(u, 0.f, 1.f);
        return u * u * (3 - 2 * u);
    }

}// namespace planks::film

#endif//PLANKCATHEDRAL_FILM_HPP
