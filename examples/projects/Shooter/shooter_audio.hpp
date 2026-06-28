// ============================================================================
//  Shooter — procedural audio: DSP helpers, synth functions, and sound bank
//  Included inside namespace {} in main.cpp — not a standalone header.
//  Requires: shooter_constants.hpp (frand, math::PI)
// ============================================================================

// ========================================================================
//  Procedural placeholder sound effects
// ========================================================================

std::vector<float> synthShot(int sr = 44100) {
    const int n = sr * 18 / 100;// 0.18s
    std::vector<float> s(n);
    for (int i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / sr;
        const float env = std::exp(-t * 22.f);
        const float noise = frand(-1.f, 1.f);
        const float thump = std::sin(2.f * math::PI * 70.f * t) * std::exp(-t * 12.f);
        s[i] = std::clamp((noise * 0.8f + thump * 0.6f) * env, -1.f, 1.f);
    }
    return s;
}

std::vector<float> synthClick(int sr = 44100) {
    const int n = sr * 3 / 100;
    std::vector<float> s(n);
    for (int i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / sr;
        s[i] = frand(-1.f, 1.f) * std::exp(-t * 120.f) * 0.5f;
    }
    return s;
}

std::vector<float> synthReload(int sr = 44100) {
    const int n = sr * 35 / 100;
    std::vector<float> s(n, 0.f);
    auto clickAt = [&](float at, float amp) {
        const int start = static_cast<int>(at * sr);
        for (int i = 0; i < sr * 3 / 100 && start + i < n; ++i) {
            const float t = static_cast<float>(i) / sr;
            s[start + i] += frand(-1.f, 1.f) * std::exp(-t * 90.f) * amp;
        }
    };
    clickAt(0.f, 0.5f);
    clickAt(0.16f, 0.4f);
    clickAt(0.30f, 0.6f);
    return s;
}

// ---- DSP helpers for the impact/step synths ----------------------------
// One-pole low-pass; band-limited noise = lp(high cut) - lp(low cut).
// Raw frand() noise reads as static hiss — every "physical" sound below
// band-shapes it first.
struct OnePole {
    float y = 0.f;
    float operator()(float x, float a) {
        y += a * (x - y);
        return y;
    }
};
float lpAlpha(float cutoffHz, int sr) {
    return 1.f - std::exp(-2.f * math::PI * cutoffHz / static_cast<float>(sr));
}
// Scale to a known peak so layering tweaks can't silently clip or vanish.
std::vector<float> normalized(std::vector<float> s, float peak) {
    float m = 0.f;
    for (float x : s) m = std::max(m, std::abs(x));
    if (m > 1e-6f)
        for (float& x : s) x *= peak / m;
    return s;
}

// Bullet-into-body thwack: band-passed noise crack + a pitch-dropping body
// thump + a duller low "wet" layer. Seeded so the bank renders a few
// distinct variants (the old sound was a single 1.4 kHz sine ping — pure
// arcade beep). Doubles as the hit-marker audio cue, which is why the
// crack keeps some mid-band brightness.
std::vector<float> synthHit(uint32_t seed, int sr = 44100) {
    std::mt19937 r(seed);
    auto rf = [&](float a, float b) { return a + (b - a) * std::uniform_real_distribution<float>(0.f, 1.f)(r); };
    auto rn = [&] { return std::uniform_real_distribution<float>(-1.f, 1.f)(r); };
    const int n = sr * 9 / 100;
    std::vector<float> s(n);
    const float f0 = rf(190.f, 240.f);
    OnePole lpHi, lpLo, lpWet;
    const float aHi = lpAlpha(rf(2600.f, 3400.f), sr);
    const float aLo = lpAlpha(800.f, sr);
    const float aWet = lpAlpha(550.f, sr);
    float phase = 0.f;
    for (int i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / sr;
        const float f = 80.f + f0 * std::exp(-t * 35.f);
        phase += 2.f * math::PI * f / sr;
        const float w = rn();
        const float thump = std::sin(phase) * std::exp(-t * 38.f) * 0.8f;
        const float crack = (lpHi(w, aHi) - lpLo(w, aLo)) * std::exp(-t * 70.f) * 1.6f;
        const float wet = lpWet(w, aWet) * std::exp(-t * 22.f) * 0.9f;
        s[i] = thump + crack + wet;
    }
    return normalized(std::move(s), 0.75f);
}

// Kill confirm: a deeper double knock (second hit ~70 ms behind the first)
// with a low noise tail, so a lethal hit reads instantly different from a
// normal one.
std::vector<float> synthKill(uint32_t seed, int sr = 44100) {
    std::mt19937 r(seed);
    auto rf = [&](float a, float b) { return a + (b - a) * std::uniform_real_distribution<float>(0.f, 1.f)(r); };
    auto rn = [&] { return std::uniform_real_distribution<float>(-1.f, 1.f)(r); };
    const int n = sr * 30 / 100;
    std::vector<float> s(n, 0.f);
    auto knock = [&](float at, float f0, float amp) {
        const int start = static_cast<int>(at * sr);
        float phase = 0.f;
        for (int i = 0; start + i < n; ++i) {
            const float t = static_cast<float>(i) / sr;
            const float f = f0 * (0.45f + 0.55f * std::exp(-t * 28.f));
            phase += 2.f * math::PI * f / sr;
            s[start + i] += std::sin(phase) * std::exp(-t * 22.f) * amp;
        }
    };
    knock(0.f, rf(120.f, 145.f), 1.f);
    knock(rf(0.06f, 0.08f), rf(90.f, 110.f), 0.65f);
    OnePole lp;
    const float aLp = lpAlpha(420.f, sr);
    for (int i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / sr;
        s[i] += lp(rn(), aLp) * std::exp(-t * 14.f) * 0.7f;
    }
    return normalized(std::move(s), 0.8f);
}

std::vector<float> synthThud(int sr = 44100) {
    const int n = sr * 14 / 100;
    std::vector<float> s(n);
    for (int i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / sr;
        const float tone = std::sin(2.f * math::PI * 180.f * t);
        s[i] = (tone * 0.6f + frand(-1.f, 1.f) * 0.4f) * std::exp(-t * 24.f);
    }
    return s;
}

std::vector<float> synthBoom(int sr = 44100) {
    const int n = sr * 90 / 100;// 0.9s
    std::vector<float> s(n);
    for (int i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / sr;
        const float env = std::exp(-t * 5.f);
        const float sub = std::sin(2.f * math::PI * (90.f - 50.f * t) * t);// downward sweep
        const float crack = frand(-1.f, 1.f) * std::exp(-t * 18.f);        // initial crack
        const float rumble = frand(-1.f, 1.f) * env;
        s[i] = std::clamp((sub * 0.7f + crack * 0.6f + rumble * 0.4f) * env, -1.f, 1.f);
    }
    return s;
}

// Player-hit cue: a body thump under a falling tone (phase-accumulated so
// the sweep is clean) with a slight vibrato and a breathy band-passed
// layer — less "game-over beep" than the old bare descending sine.
std::vector<float> synthHurt(int sr = 44100) {
    const int n = sr * 30 / 100;
    std::vector<float> s(n);
    OnePole lpHi, lpLo;
    const float aHi = lpAlpha(1400.f, sr);
    const float aLo = lpAlpha(500.f, sr);
    float phase = 0.f, phaseT = 0.f;
    for (int i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / sr;
        const float f = 150.f + 180.f * std::exp(-t * 7.f) + 12.f * std::sin(2.f * math::PI * 9.f * t);
        phaseT += 2.f * math::PI * f / sr;
        const float tone = std::sin(phaseT) * std::exp(-t * 8.f) * 0.7f;
        phase += 2.f * math::PI * (70.f + 100.f * std::exp(-t * 40.f)) / sr;
        const float thump = std::sin(phase) * std::exp(-t * 30.f) * 0.8f;
        const float w = frand(-1.f, 1.f);
        const float breath = (lpHi(w, aHi) - lpLo(w, aLo)) * std::exp(-t * 12.f) * 0.6f;
        s[i] = tone + thump + breath;
    }
    return normalized(std::move(s), 0.7f);
}

// One footstep on gritty sand/concrete: a soft pitch-dropping heel thump,
// then a band-passed scuff whose amplitude is re-modulated by slow noise
// (the "grit" crunch). Seeded — the bank renders several distinct variants
// and cycles them, so successive steps never replay one identical sample
// (the old step was a single 70 ms white-noise tick).
std::vector<float> synthStep(uint32_t seed, int sr = 44100) {
    std::mt19937 r(seed);
    auto rf = [&](float a, float b) { return a + (b - a) * std::uniform_real_distribution<float>(0.f, 1.f)(r); };
    auto rn = [&] { return std::uniform_real_distribution<float>(-1.f, 1.f)(r); };
    const int n = static_cast<int>(static_cast<float>(sr) * rf(0.10f, 0.13f));
    std::vector<float> s(n);
    const float f0 = rf(110.f, 150.f);
    const float scuffAt = rf(0.008f, 0.018f);// sole contact lags the heel strike
    OnePole lpHi, lpHi2, lpLo, grit;
    const float aHi = lpAlpha(rf(1900.f, 2700.f), sr);
    const float aLo = lpAlpha(rf(320.f, 460.f), sr);
    const float aGrit = lpAlpha(rf(60.f, 90.f), sr);
    float phase = 0.f;
    for (int i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / sr;
        phase += 2.f * math::PI * (55.f + f0 * std::exp(-t * 30.f)) / sr;
        const float heel = std::sin(phase) * std::exp(-t * 45.f) * 0.8f;
        const float w = rn();
        // cascade the high cut (12 dB/oct): one pole leaves enough leakage
        // above the cutoff that the scuff hisses instead of crunching
        const float band = lpHi2(lpHi(w, aHi), aHi) - lpLo(w, aLo);
        // slow-noise modulation makes the scuff crunch instead of hiss
        const float tex = std::clamp(0.35f + 9.f * std::abs(grit(rn(), aGrit)), 0.f, 1.f);
        const float ts = t - scuffAt;
        const float scuff = ts > 0.f ? band * tex * std::exp(-ts * 32.f) * 1.4f : 0.f;
        s[i] = heel + scuff;
    }
    return normalized(std::move(s), 0.6f);
}

// A pooled, retriggerable sound: round-robins a few voices so rapid fire
// overlaps instead of cutting itself off. Voices may hold different synth
// variants of the same sound, so the rotation also cycles variants. Each
// play() takes a volume scale + playback rate — re-rolling those per
// trigger is what keeps repeated one-shots (steps, hits) from sounding
// machine-gunned. Degrades to a no-op if the audio device or file failed
// to initialise.
struct Sound {
    std::vector<std::unique_ptr<Audio>> voices;
    size_t next = 0;
    float volume = 0.6f;
    void play(float volScale = 1.f, float rate = 1.f) {
        if (voices.empty()) return;
        auto& v = voices[next];
        v->stop();
        v->seekToStart();// rewind so re-fire restarts from frame 0
        v->setVolume(volume * volScale);
        v->setPlaybackRate(rate);
        v->play();
        next = (next + 1) % voices.size();
    }
};

// Like Sound, but spatialised: playAt() drops the source at a world position so
// the blast pans + attenuates relative to the camera-mounted AudioListener.
struct PositionalSound {
    std::vector<std::unique_ptr<PositionalAudio>> voices;
    size_t next = 0;
    void playAt(const Vector3& p, float rate = 1.f) {
        if (voices.empty()) return;
        auto& v = voices[next];
        v->stop();
        v->seekToStart();
        v->setPlaybackRate(rate);
        v->position.copy(p);
        v->updateMatrixWorld(true);// push the new source position to the audio engine
        v->play();
        next = (next + 1) % voices.size();
    }
};

struct SoundBank {
    std::unique_ptr<AudioListener> listener;
    Sound shot, empty, reload, hit, thud, hurt, step, metal;
    PositionalSound boom;// grenade blast — spatialised at the detonation point
    bool ok = false;

    void init(Object3D& attachTo) {
        try {
            const fs::path dir = fs::temp_directory_path() / "threepp_tps_sounds";
            fs::create_directories(dir);
            struct Spec {
                const char* name;                         // temp WAV base name (synth fallback)
                std::vector<std::vector<float>> variants;// synth renders; voice i loads variant i % N
                Sound* dst;
                int voices;
                std::string file;// external audio file; used instead of synth when set
            };

            // Real submachine-gun sample for the gun; the rest stay
            // procedural. Falls back to the synth shot if the file is absent.
            const std::string assets = std::string(DATA_FOLDER) + "/sounds/";
            const std::string gunFile = assets + "freesound_community-submachine-gun-79846.mp3";
            const std::string reloadFile = assets + "freesound_community-1911-reload-6248.mp3";
            const std::string metalFile = assets + "freesound_community-hard-metal-impact-43052.mp3";
            const std::string boomFile = assets + "grenade_explosion.mp3";
            std::vector<Spec> specs{
                    {"shot", {synthShot()}, &shot, 6, fs::exists(gunFile) ? gunFile : std::string{}},
                    {"empty", {synthClick()}, &empty, 2, {}},
                    {"reload", {synthReload()}, &reload, 2, fs::exists(reloadFile) ? reloadFile : std::string{}},
                    {"hit", {synthHit(11), synthHit(22), synthHit(33)}, &hit, 6, {}},
                    {"metal", {synthThud()}, &metal, 4, fs::exists(metalFile) ? metalFile : std::string{}},
                    {"thud", {synthKill(41), synthKill(42)}, &thud, 4, {}},
                    {"hurt", {synthHurt()}, &hurt, 2, {}},
                    {"step", {synthStep(1), synthStep(2), synthStep(3), synthStep(4)}, &step, 4, {}}};

            listener = std::make_unique<AudioListener>();
            attachTo.addRef(*listener);
            for (auto& sp : specs) {
                std::vector<std::string> paths;
                if (!sp.file.empty()) {
                    paths.push_back(sp.file);// external sample (e.g. the gun MP3)
                } else {
                    for (size_t k = 0; k < sp.variants.size(); ++k) {// render the synth fallback(s)
                        auto p = (dir / (sp.name + std::to_string(k) + ".wav")).string();
                        threepp::audio::writeWav(p, sp.variants[k]);
                        paths.push_back(std::move(p));
                    }
                }
                for (int i = 0; i < sp.voices; ++i) {
                    auto a = std::make_unique<Audio>(*listener, paths[i % paths.size()]);
                    a->setVolume(0.6f);
                    sp.dst->voices.push_back(std::move(a));
                }
            }

            // grenade blast: spatialised. Stays at full volume within ~15 m of the
            // camera (covers most of the arena), then a shallow inverse rolloff so
            // far blasts are clearly quieter + directional without dropping out.
            {
                std::string boomPath = fs::exists(boomFile) ? boomFile : (dir / "boom.wav").string();
                if (!fs::exists(boomFile)) threepp::audio::writeWav(boomPath, synthBoom());
                for (int i = 0; i < 4; ++i) {
                    auto a = std::make_unique<PositionalAudio>(*listener, boomPath);
                    a->setVolume(0.9f);// blast is the loudest cue in the game
                    a->setDistanceModel(PositionalAudio::DistanceModel::Inverse);
                    a->setMinDistance(15.f);  // full-volume radius
                    a->setRolloffFactor(0.35f);// shallow falloff beyond it
                    boom.voices.push_back(std::move(a));
                }
            }
            ok = true;
        } catch (const std::exception& e) {
            std::cerr << "[audio] disabled: " << e.what() << "\n";
        }
    }
};
