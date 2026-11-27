// MotorControl: PID position control of a simulated DC motor.
//
// The slider sets a target angle and the PID sets the motor voltage from the
// error between target and shaft:
//
//   u = kp * e  +  ki * integral(e dt)  +  kd * de/dt
//
// The dial shows the shaft, the target (blue marker) and the error between
// them (orange band). The upper chart plots target and shaft over the last ten
// seconds, the lower one the three terms and their sum, the voltage the motor
// gets (clamped to its 12 V supply). Things to try: ki = kd = 0 (P only), kd up
// to damp the overshoot, a load torque to see P alone leave an offset and the
// integral term remove it, a large kp to drive the voltage into its limit.
//
// The simulation runs at a fixed 1 kHz, apart from the frame rate. Steps the
// length of a frame would make the motion depend on it, and one slow frame
// (the first one, which also builds the window, the UI and the pipelines) would
// be a single long, inaccurate step that throws the shaft forward.

#include "DCMotor.hpp"
#include "renderer_factory.hpp"
#include "utility/Regulator.hpp"

#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/objects/TextSprite.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr double kRadToDeg = 180.0 / math::PI;
    constexpr double kDegToRad = math::PI / 180.0;

    // The degree sign in UTF-8, spelled as bytes: MSVC builds of this example
    // do not read the source as UTF-8.
    const std::string kDeg = "\xC2\xB0";

    // Palette. The error band and the P trace share a colour: P is kp times
    // that error.
    constexpr int kInk = 0x1e293b;   // shaft, measured trace, total voltage, text
    constexpr int kMuted = 0x64748b; // ticks and axis labels
    constexpr int kFace = 0xf1f5f9;  // dial face, chart background
    constexpr int kGrid = 0xe2e8f0;  // dial track, chart grid
    constexpr int kTarget = 0x2563eb;// target
    constexpr int kP = 0xf97316;     // error, P term
    constexpr int kI = 0x16a34a;     // I term
    constexpr int kD = 0x9333ea;     // D term
    constexpr int kShaft = 0x94a3b8; // the motor shaft in the hub

    std::string format(const char* fmt, double value) {
        char buf[32];
        std::snprintf(buf, sizeof buf, fmt, value);
        return buf;
    }

    // Everything here is flat and lies in z = 0, so renderOrder alone decides
    // what covers what, the same on both backends. Double sided, so no shape
    // depends on its winding to be seen.
    std::shared_ptr<Mesh> flatMesh(std::shared_ptr<BufferGeometry> geometry, int color, int order) {
        auto material = MeshBasicMaterial::create(MeshBasicMaterial::Params{}.color(color));
        material->side = Side::Double;
        auto mesh = Mesh::create(std::move(geometry), material);
        mesh->renderOrder = order;
        return mesh;
    }

    std::shared_ptr<TextSprite> makeLabel(const Font& font, float height, int color,
                                          TextSprite::HorizontalAlignment h = TextSprite::HorizontalAlignment::Center) {
        // TextBox::Line: one line of the font is `height` tall whatever the
        // string, so "0°" and "180°" get the same glyph size.
        auto label = TextSprite::create(font, height, TextBox::Line);
        label->setColor(Color(color));
        label->setHorizontalAlignment(h);
        label->setVerticalAlignment(TextSprite::VerticalAlignment::Center);
        return label;
    }

    // Static flat shapes collected into one triangle list.
    class Triangles {

    public:
        void triangle(const Vector2& a, const Vector2& b, const Vector2& c) {
            for (const auto* p : {&a, &b, &c}) xyz_.insert(xyz_.end(), {p->x, p->y, 0.f});
        }

        void quad(const Vector2& a, const Vector2& b, const Vector2& c, const Vector2& d) {
            triangle(a, b, c);
            triangle(a, c, d);
        }

        void rect(float x0, float y0, float x1, float y1) {
            quad({x0, y0}, {x1, y0}, {x1, y1}, {x0, y1});
        }

        // A bar `width` wide along `angle`, from radius r0 out to r1.
        void radialBar(float angle, float r0, float r1, float width) {
            const Vector2 dir(std::cos(angle), std::sin(angle));
            const Vector2 side(-dir.y * width / 2, dir.x * width / 2);
            const Vector2 a(dir.x * r0, dir.y * r0), b(dir.x * r1, dir.y * r1);
            quad({a.x - side.x, a.y - side.y}, {b.x - side.x, b.y - side.y},
                 {b.x + side.x, b.y + side.y}, {a.x + side.x, a.y + side.y});
        }

        [[nodiscard]] std::shared_ptr<BufferGeometry> geometry() const {
            auto g = BufferGeometry::create();
            g->setAttribute("position", FloatBufferAttribute::create(xyz_, 3));
            return g;
        }

    private:
        std::vector<float> xyz_;
    };

    // A mesh with a fixed vertex count and index list whose positions are
    // rewritten in place: no allocation per frame, and the renderer re-uploads
    // one buffer instead of meeting a new geometry every frame.
    class DynamicMesh {

    public:
        DynamicMesh(size_t vertices, std::vector<unsigned int> index, int color, int order) {
            auto g = BufferGeometry::create();
            position_ = FloatBufferAttribute::create(std::vector<float>(vertices * 3, 0.f), 3);
            g->setAttribute("position", position_);
            g->setIndex(std::move(index));
            mesh_ = flatMesh(g, color, order);
            mesh_->frustumCulled = false;// the bounding sphere is never recomputed
        }

        void set(size_t i, float x, float y) { position_->setXYZ(i, x, y, 0.f); }

        void commit() { position_->needsUpdate(); }

        [[nodiscard]] const std::shared_ptr<Mesh>& mesh() const { return mesh_; }

    private:
        std::shared_ptr<FloatBufferAttribute> position_;
        std::shared_ptr<Mesh> mesh_;
    };

    // A ring sector whose start and end angles change every frame.
    class ArcBand {

    public:
        ArcBand(float inner, float outer, int color, int order)
            : inner_(inner), outer_(outer), mesh_(2 * (kSegments + 1), makeIndex(), color, order) {}

        void update(float from, float to) {
            for (unsigned k = 0; k <= kSegments; ++k) {
                const float a = from + (to - from) * static_cast<float>(k) / kSegments;
                mesh_.set(2 * k, inner_ * std::cos(a), inner_ * std::sin(a));
                mesh_.set(2 * k + 1, outer_ * std::cos(a), outer_ * std::sin(a));
            }
            mesh_.commit();
        }

        [[nodiscard]] const std::shared_ptr<Mesh>& mesh() const { return mesh_.mesh(); }

    private:
        static constexpr unsigned kSegments = 96;

        float inner_, outer_;
        DynamicMesh mesh_;

        static std::vector<unsigned int> makeIndex() {
            std::vector<unsigned int> index;
            for (unsigned k = 0; k < kSegments; ++k) {
                const unsigned i0 = 2 * k, o0 = i0 + 1, i1 = i0 + 2, o1 = i0 + 3;
                index.insert(index.end(), {i0, o0, o1, i0, o1, i1});
            }
            return index;
        }
    };

    // A polyline drawn as a ribbon of the given width: a 1 px GL line is a
    // hairline on a high-DPI screen.
    class Trace {

    public:
        Trace(size_t points, float width, int color, int order)
            : width_(width), mesh_(2 * points, makeIndex(points), color, order) {}

        void update(const std::vector<Vector2>& points) {
            const size_t n = points.size();
            for (size_t i = 0; i < n; ++i) {
                // Offset across the direction through the neighbours.
                const Vector2& prev = points[i > 0 ? i - 1 : i];
                const Vector2& next = points[i + 1 < n ? i + 1 : i];
                const Vector2 dir(next.x - prev.x, next.y - prev.y);
                const float len = std::max(dir.length(), 1e-6f);
                const Vector2 side(-dir.y / len * width_ / 2, dir.x / len * width_ / 2);
                mesh_.set(2 * i, points[i].x + side.x, points[i].y + side.y);
                mesh_.set(2 * i + 1, points[i].x - side.x, points[i].y - side.y);
            }
            mesh_.commit();
        }

        [[nodiscard]] const std::shared_ptr<Mesh>& mesh() const { return mesh_.mesh(); }

    private:
        float width_;
        DynamicMesh mesh_;

        static std::vector<unsigned int> makeIndex(size_t points) {
            std::vector<unsigned int> index;
            for (unsigned i = 0; i + 1 < points; ++i) {
                const unsigned l0 = 2 * i, r0 = l0 + 1, l1 = l0 + 2, r1 = l0 + 3;
                index.insert(index.end(), {r0, r1, l1, r0, l1, l0});
            }
            return index;
        }
    };

    // The motor seen from the front, on a protractor: 0° to the right,
    // counter-clockwise positive, 0° to 180° marked.
    class Dial: public Object3D {

    public:
        explicit Dial(const Font& font)
            : errorBand_(kTrackInner, kTrackOuter, kP, 2) {

            add(flatMesh(CircleGeometry::create(0.9f, 96, 0, math::PI), kFace, 0));
            add(flatMesh(RingGeometry::create(kTrackInner, kTrackOuter, 128, 1, 0, math::PI), kGrid, 1));
            add(errorBand_.mesh());

            Triangles ticks;
            for (int deg = 0; deg <= 180; deg += 5) {
                const bool major = deg % 30 == 0;
                const auto a = static_cast<float>(deg * kDegToRad);
                ticks.radialBar(a, 0.69f, major ? 0.755f : 0.72f, major ? 0.008f : 0.004f);
                if (major) {
                    auto label = makeLabel(font, 0.06f, kMuted);
                    label->setText(std::to_string(deg) + kDeg);
                    label->position.set(0.82f * std::cos(a), 0.82f * std::sin(a), 0);
                    add(label);
                }
            }
            add(flatMesh(ticks.geometry(), kMuted, 3));

            Triangles marker;// points at the track from outside it
            marker.triangle({0.675f, 0.f}, {0.765f, -0.034f}, {0.765f, 0.034f});
            target_ = flatMesh(marker.geometry(), kTarget, 4);
            add(target_);

            Shape arm;// a bar with a rounded tip, along +x
            constexpr float w = 0.035f, length = 0.58f;
            arm.moveTo(0, -w).lineTo(length, -w);
            arm.absarc(length, 0, w, -math::PI / 2, math::PI / 2, false);
            arm.lineTo(0, w).lineTo(0, -w);
            shaft_ = flatMesh(ShapeGeometry::create(arm, 16), kInk, 5);
            add(shaft_);

            add(flatMesh(CircleGeometry::create(0.11f, 64), kInk, 6));
            add(flatMesh(CircleGeometry::create(0.04f, 32), kShaft, 7));

            readout_ = makeLabel(font, 0.13f, kInk);
            readout_->position.set(0, -0.19f, 0);
            add(readout_);
        }

        void update(double measured, double target) {
            shaft_->rotation.z = static_cast<float>(measured);
            target_->rotation.z = static_cast<float>(target);
            errorBand_.update(static_cast<float>(measured), static_cast<float>(target));
        }

        void setReadout(double measured) {
            readout_->setText(format("%.1f", measured * kRadToDeg) + kDeg);
        }

    private:
        static constexpr float kTrackInner = 0.62f;
        static constexpr float kTrackOuter = 0.67f;

        ArcBand errorBand_;
        std::shared_ptr<Mesh> shaft_;
        std::shared_ptr<Mesh> target_;
        std::shared_ptr<TextSprite> readout_;
    };

    // The last `seconds` of a few signals, newest at the right, with a legend
    // of their current values above. The plot's lower-left corner is the
    // object's origin.
    class StripChart: public Object3D {

    public:
        static constexpr float kSampleInterval = 0.02f;// [s] of simulated time

        struct Series {
            std::string name;
            int color;
            float width;
        };

        struct Axis {
            double lo, hi;                // plotted range; values beyond it are pinned to the edge
            double gridFrom, gridTo, step;// horizontal grid lines
            double labelStep;             // labelled every labelStep from gridFrom
            std::string unit;             // after labels and legend values
            const char* valueFormat;      // printf format of a legend value
        };

        StripChart(const Font& font, float width, float height, float seconds,
                   Axis axis, const std::vector<Series>& series, bool timeLabels)
            : width_(width), height_(height), axis_(std::move(axis)),
              slots_(static_cast<size_t>(seconds / kSampleInterval) + 1),
              samples_(slots_ * series.size()), points_(slots_) {

            Triangles background;
            background.rect(0, 0, width, height);
            add(flatMesh(background.geometry(), kFace, 0));

            Triangles grid;
            for (double v = axis_.gridFrom; v <= axis_.gridTo + 1e-9; v += axis_.step) {
                const float y = yOf(v);
                grid.rect(0, y - 0.0015f, width, y + 0.0015f);
                if (std::fmod(v - axis_.gridFrom, axis_.labelStep) < 1e-9) {
                    auto label = makeLabel(font, 0.05f, kMuted, TextSprite::HorizontalAlignment::Right);
                    label->setText(format("%g", v) + axis_.unit);
                    label->position.set(-0.02f, y, 0);
                    add(label);
                }
            }
            for (int s = 0; s <= static_cast<int>(seconds); s += 2) {
                const float x = width * (1.f - static_cast<float>(s) / seconds);
                grid.rect(x - 0.0015f, 0, x + 0.0015f, height);
            }
            add(flatMesh(grid.geometry(), kGrid, 1));

            if (timeLabels) {
                const auto timeLabel = [&](const std::string& text, float x) {
                    auto label = makeLabel(font, 0.05f, kMuted);
                    label->setText(text);
                    label->position.set(x, -0.05f, 0);
                    add(label);
                };
                timeLabel(format("-%g s", seconds), 0);
                timeLabel(format("-%g s", seconds / 2), width / 2);
                timeLabel("now", width);
            }

            // Legend above the plot: a swatch, the name and the current value.
            const float spacing = std::min(0.62f, width / static_cast<float>(series.size()));
            const float y = height + 0.06f;
            for (size_t i = 0; i < series.size(); ++i) {
                const Series& s = series[i];
                traces_.emplace_back(slots_, s.width, s.color, 2 + static_cast<int>(i));
                add(traces_.back().mesh());

                const float x = spacing * static_cast<float>(i);
                Triangles swatch;
                swatch.rect(x, y - 0.006f, x + 0.05f, y + 0.006f);
                add(flatMesh(swatch.geometry(), s.color, 2 + static_cast<int>(series.size())));

                auto text = makeLabel(font, 0.05f, kInk, TextSprite::HorizontalAlignment::Left);
                text->position.set(x + 0.07f, y, 0);
                add(text);
                legend_.push_back({s.name, text});
            }

            reset(std::vector<double>(series.size(), 0.0));
        }

        // One sample: a value per series, in the axis unit.
        void push(const std::vector<double>& values) {
            std::copy(values.begin(), values.end(), samples_.begin() + static_cast<long>(head_ * traces_.size()));
            head_ = (head_ + 1) % slots_;
        }

        // Fill the whole history with one sample.
        void reset(const std::vector<double>& values) {
            for (size_t i = 0; i < slots_; ++i) {
                std::copy(values.begin(), values.end(), samples_.begin() + static_cast<long>(i * traces_.size()));
            }
            head_ = 0;
        }

        // Rewrite the traces from the history.
        void update() {
            for (size_t k = 0; k < traces_.size(); ++k) {
                for (size_t i = 0; i < slots_; ++i) {
                    const size_t slot = (head_ + i) % slots_;// oldest first
                    points_[i].set(width_ * static_cast<float>(i) / static_cast<float>(slots_ - 1),
                                   yOf(samples_[slot * traces_.size() + k]));
                }
                traces_[k].update(points_);
            }
        }

        // The legend shows the newest sample.
        void updateLegend() {
            const size_t newest = (head_ + slots_ - 1) % slots_;
            for (size_t k = 0; k < legend_.size(); ++k) {
                const double v = samples_[newest * traces_.size() + k];
                legend_[k].text->setText(legend_[k].name + " " + format(axis_.valueFormat, v) + axis_.unit);
            }
        }

    private:
        struct LegendEntry {
            std::string name;
            std::shared_ptr<TextSprite> text;
        };

        [[nodiscard]] float yOf(double value) const {
            return height_ * static_cast<float>(std::clamp((value - axis_.lo) / (axis_.hi - axis_.lo), 0.0, 1.0));
        }

        float width_, height_;
        Axis axis_;
        size_t slots_;
        std::vector<double> samples_;// slots_ x series, a ring buffer of slots
        size_t head_{};              // the slot the next sample goes in
        std::vector<Trace> traces_;
        std::vector<LegendEntry> legend_;
        std::vector<Vector2> points_;
    };

    // The motor and its controller, stepped at a fixed rate.
    struct Servo {
        DCMotor motor;
        PIDRegulator pid{3, 1, 0.3f};

        double target{};    // [rad]
        double loadTorque{};// [N m]
        double time{};      // [s] simulated

        Servo() {
            const auto supply = static_cast<float>(motor.params().maxVoltage);
            pid.setOutputLimits(-supply, supply);
            pid.setDerivativeFilter(0.01f);
        }

        void step(double dt) {
            const double error = target - motor.position();
            const double voltage = pid.regulate(static_cast<float>(error), static_cast<float>(dt));
            motor.update(voltage, loadTorque, dt);
            time += dt;
        }

        void reset() {
            motor.reset(0);
            pid.reset();
        }
    };

}// namespace

int main() {

    Canvas canvas("Motor control", {{"aa", 6}});
    auto renderer = createRenderer(canvas);

    Scene scene;
    scene.background = Color::white;

    // One unit is half the window height; x spans +-aspect.
    OrthographicCamera camera(-canvas.size().aspect(), canvas.size().aspect(), 1, -1);
    camera.position.z = 1;

    FontLoader fontLoader;
    const auto font = fontLoader.defaultFont();

    // Dial and charts sit to the right of the settings panel, as one group
    // kept against the window's right edge.
    auto content = Group::create();
    scene.add(content);

    auto dial = std::make_shared<Dial>(font);
    dial->position.y = 0.22f;
    dial->scale.setScalar(0.8f);
    content->add(dial);

    constexpr float chartWidth = 1.87f;
    auto angleChart = std::make_shared<StripChart>(
            font, chartWidth, 0.34f, 10.f,
            StripChart::Axis{-20, 200, 0, 180, 45, 90, kDeg, "%.1f"},
            std::vector<StripChart::Series>{{"target", kTarget, 0.006f}, {"measured", kInk, 0.008f}},
            false);
    angleChart->position.set(-0.87f, -0.42f, 0);
    content->add(angleChart);

    auto voltageChart = std::make_shared<StripChart>(
            font, chartWidth, 0.30f, 10.f,
            StripChart::Axis{-15, 15, -12, 12, 6, 12, " V", "%+.1f"},
            std::vector<StripChart::Series>{{"P", kP, 0.006f}, {"I", kI, 0.006f}, {"D", kD, 0.006f}, {"u", kInk, 0.008f}},
            true);
    voltageChart->position.set(-0.87f, -0.88f, 0);
    content->add(voltageChart);

    const auto layout = [&](float aspect) {
        camera.left = -aspect;
        camera.right = aspect;
        camera.updateProjectionMatrix();
        content->position.x = aspect - 1.08f;
    };
    layout(canvas.size().aspect());

    canvas.onWindowResize([&](WindowSize size) {
        layout(size.aspect());
        renderer->setSize(size);
    });

    Servo servo;
    float targetDeg = 120;
    float loadTorque = 0;

    auto& gains = servo.pid.params();
    RendererSettingsUi ui(canvas, *renderer, [&] {
        // Narrow widgets keep the panel clear of the dial and the charts.
        ImGui::PushItemWidth(ImGui::GetFontSize() * 10);

        ImGui::SeparatorText("Setpoint");
        ImGui::SliderFloat("target", &targetDeg, 0, 180, "%.0f deg");

        ImGui::SeparatorText("PID gains");
        ImGui::SliderFloat("kp", &gains.kp, 0, 20, "%.1f V/rad");
        ImGui::SliderFloat("ki", &gains.ki, 0, 10, "%.2f V/(rad s)");
        ImGui::SliderFloat("kd", &gains.kd, 0, 2, "%.2f V s/rad");

        ImGui::SeparatorText("Disturbance");
        ImGui::SliderFloat("load", &loadTorque, -1, 1, "%.2f N m");

        ImGui::SeparatorText("Motor voltage");
        const double volts = servo.motor.voltage();
        char overlay[32];
        std::snprintf(overlay, sizeof overlay, "%+.1f V", volts);
        ImGui::ProgressBar(static_cast<float>(std::abs(volts) / servo.motor.params().maxVoltage),
                           ImVec2(ImGui::CalcItemWidth(), 0), overlay);

        if (ImGui::Button("Reset")) {
            servo.reset();
            angleChart->reset({0, 0});
            voltageChart->reset({0, 0, 0, 0});
        }

        ImGui::PopItemWidth();
    }, "Motor control");

    constexpr double kStep = 0.001;// [s], the simulation's fixed step
    double pending = 0;            // simulated time owed to the wall clock
    double nextSample = 0;
    int frame = 0;

    Clock clock;
    canvas.animate([&] {
        servo.target = targetDeg * kDegToRad;
        servo.loadTorque = loadTorque;

        // A frame longer than 0.1 s (start-up, a dragged window) is not caught
        // up: the motion pauses instead of jumping.
        pending += std::min(static_cast<double>(clock.getDelta()), 0.1);
        while (pending >= kStep) {
            servo.step(kStep);
            pending -= kStep;
            if (servo.time >= nextSample) {
                const auto& t = servo.pid.terms();
                angleChart->push({servo.target * kRadToDeg, servo.motor.position() * kRadToDeg});
                voltageChart->push({t.p, t.i, t.d, servo.motor.voltage()});
                nextSample += StripChart::kSampleInterval;
            }
        }

        dial->update(servo.motor.position(), servo.target);
        angleChart->update();
        voltageChart->update();
        if (frame++ % 4 == 0) {// text is rasterized on change: a few times a second is plenty
            dial->setReadout(servo.motor.position());
            angleChart->updateLegend();
            voltageChart->updateLegend();
        }

        renderer->render(scene, camera);
        ui.render();
    });
}
