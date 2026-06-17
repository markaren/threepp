// An InstancedMesh of grass blades animated by a GPU wind vertex shader.
//
// Each blade is a small tapered quad-strip; a ShaderMaterial bends it toward
// the wind in the vertex stage (base planted, tip sways), so the whole field
// animates for free on the GPU — no per-frame CPU matrix rewrites. Placement
// is up to the caller: build the field with a blade count, then position each
// blade with the inherited InstancedMesh::setMatrixAt(). Advance the wind clock
// once per frame with setTime().
//
// Backends: GL and WebGPU (raster). The blades sway via the instanced
// ShaderMaterial path. For the Vulkan path tracer use GrassMesh instead — it
// has no generic ShaderMaterial path, so a GrassField renders there as a static
// (non-swaying) instanced mesh.

#ifndef THREEPP_GRASSFIELD_HPP
#define THREEPP_GRASSFIELD_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/InstancedMesh.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace threepp {

    class GrassField: public InstancedMesh {

    public:
        struct Params {
            Vector2 windDir{0.8f, 0.6f};      // horizontal wind direction (world XZ)
            float windStrength = 0.18f;       // sway amplitude
            Vector3 topColor{0.30f, 0.42f, 0.14f};
            Vector3 bottomColor{0.08f, 0.16f, 0.05f};
            Vector3 sunColor{0.55f, 0.55f, 0.50f};
            Vector3 ambient{0.30f, 0.34f, 0.30f};
            Vector3 fogColor{0.70f, 0.75f, 0.80f};
            float fogNear = 30.f;
            float fogFar = 120.f;
            int segments = 4;                 // blade tessellation (default geometry)
        };

        GrassField(const std::shared_ptr<BufferGeometry>& blade,
                   const std::shared_ptr<ShaderMaterial>& material,
                   size_t bladeCount)
            : InstancedMesh(blade, material, bladeCount), windMaterial_(material) {}

        // Advance the wind animation clock (seconds). Call once per frame.
        void setTime(float seconds) {
            windMaterial_->uniforms.at("time").setValue(seconds);
        }

        void setWind(float strength, const Vector2& dir) {
            windMaterial_->uniforms.at("windStrength").setValue(strength);
            windMaterial_->uniforms.at("windDir").setValue(dir);
        }

        void setFog(const Vector3& color, float near, float far) {
            windMaterial_->uniforms.at("fogColor").setValue(color);
            windMaterial_->uniforms.at("fogNear").setValue(near);
            windMaterial_->uniforms.at("fogFar").setValue(far);
        }

        [[nodiscard]] ShaderMaterial& windMaterial() const { return *windMaterial_; }

        static std::shared_ptr<GrassField> create(size_t bladeCount, const Params& params = {}) {
            return std::make_shared<GrassField>(
                    bladeGeometry(params.segments), makeMaterial(params), bladeCount);
        }

        // A single tapered blade (origin at the base, unit height along +Y;
        // scale per instance). Carries position / normal / uv / color so it also
        // works with a standard vertexColors material if reused elsewhere.
        static std::shared_ptr<BufferGeometry> bladeGeometry(int segments = 4) {
            const int seg = segments < 1 ? 1 : segments;
            constexpr float wBase = 0.05f;// half-width at the base
            const Vector3 bottom{0.06f, 0.13f, 0.04f};
            const Vector3 top{0.20f, 0.34f, 0.11f};

            std::vector<float> pos, nrm, uv, col;
            std::vector<unsigned int> idx;
            for (int i = 0; i <= seg; ++i) {
                const float t = static_cast<float>(i) / static_cast<float>(seg);
                const float w = wBase * (1.f - t);// taper to a point at the tip
                const Vector3 c = bottom.clone().lerp(top, t);
                for (int s = 0; s < 2; ++s) {
                    pos.push_back(s == 0 ? -w : w);
                    pos.push_back(t);// unit height
                    pos.push_back(0.f);
                    // Up-biased normal: catches sky/sun light regardless of facing.
                    nrm.insert(nrm.end(), {0.f, 0.85f, 0.53f});
                    uv.push_back(s == 0 ? 0.f : 1.f);
                    uv.push_back(t);
                    col.insert(col.end(), {c.x, c.y, c.z});
                }
            }
            for (int i = 0; i < seg; ++i) {
                const auto a = static_cast<unsigned int>(i * 2);
                const unsigned int b = a + 1, c = a + 2, d = a + 3;
                idx.insert(idx.end(), {a, b, c, b, d, c});
            }

            auto geo = BufferGeometry::create();
            geo->setIndex(idx);
            geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
            geo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
            geo->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
            geo->setAttribute("color", FloatBufferAttribute::create(col, 3));
            return geo;
        }

    private:
        std::shared_ptr<ShaderMaterial> windMaterial_;

        static std::shared_ptr<ShaderMaterial> makeMaterial(const Params& p) {
            auto m = ShaderMaterial::create();
            m->vertexShader = vertexShader();
            m->fragmentShader = fragmentShader();
            m->side = Side::Double;
            m->uniforms["time"].setValue(0.f);
            m->uniforms["windStrength"].setValue(p.windStrength);
            m->uniforms["windDir"].setValue(p.windDir);
            m->uniforms["topColor"].setValue(p.topColor);
            m->uniforms["bottomColor"].setValue(p.bottomColor);
            m->uniforms["sunColor"].setValue(p.sunColor);
            m->uniforms["ambient"].setValue(p.ambient);
            m->uniforms["fogColor"].setValue(p.fogColor);
            m->uniforms["fogNear"].setValue(p.fogNear);
            m->uniforms["fogFar"].setValue(p.fogFar);
            return m;
        }

        // The blade sways toward windDir, weighted by height² (base planted, tip
        // free). instanceMatrix is supplied by the InstancedMesh path on GL/WGPU.
        static const char* vertexShader() {
            return R"(
                uniform float time;
                uniform float windStrength;
                uniform vec2  windDir;
                varying float vHeight;
                varying float vFog;
                void main() {
                    vHeight = uv.y;
                    vec3 p = position;
                #ifdef USE_INSTANCING
                    vec3 instPos = vec3(instanceMatrix[3][0], instanceMatrix[3][1], instanceMatrix[3][2]);
                #else
                    vec3 instPos = vec3(0.0);
                #endif
                    float phase = time * 1.6 + instPos.x * 0.25 + instPos.z * 0.25;
                    float gust  = sin(phase) * 0.6 + sin(phase * 2.3 + 1.7) * 0.25;
                    float bend  = gust * windStrength * vHeight * vHeight;
                    p.x += windDir.x * bend;
                    p.z += windDir.y * bend;
                #ifdef USE_INSTANCING
                    vec4 mv = modelViewMatrix * instanceMatrix * vec4(p, 1.0);
                #else
                    vec4 mv = modelViewMatrix * vec4(p, 1.0);
                #endif
                    vFog = -mv.z;
                    gl_Position = projectionMatrix * mv;
                }
            )";
        }

        // Self-contained shading (no IBL) keeps the thin blades from blowing out
        // to white under a bright sky: a soft wrap + ambient, then distance fog.
        static const char* fragmentShader() {
            return R"(
                uniform vec3  topColor;
                uniform vec3  bottomColor;
                uniform vec3  sunColor;
                uniform vec3  ambient;
                uniform vec3  fogColor;
                uniform float fogNear;
                uniform float fogFar;
                varying float vHeight;
                varying float vFog;
                void main() {
                    vec3 base = mix(bottomColor, topColor, vHeight);
                    vec3 lit = base * (ambient + sunColor * 0.7);
                    float f = clamp((vFog - fogNear) / (fogFar - fogNear), 0.0, 1.0);
                    gl_FragColor = vec4(mix(lit, fogColor, f), 1.0);
                }
            )";
        }
    };

}// namespace threepp

#endif// THREEPP_GRASSFIELD_HPP
