
#include "threepp/extras/terrain/MountainBackdrop.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/renderers/shaders/ShaderChunk.hpp"
#include "threepp/renderers/shaders/ShaderLib.hpp"
#include "threepp/utils/ThreadPool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

using namespace threepp;
using namespace threepp::terrain;

namespace {

    constexpr float kTau = 6.28318530717959f;

    float smoothstepf(float e0, float e1, float x) {
        const float t = std::clamp((x - e0) / (e1 - e0), 0.f, 1.f);
        return t * t * (3.f - 2.f * t);
    }

    float mixf(float a, float b, float t) { return a + (b - a) * t; }

    std::uint32_t hashU(std::uint32_t x) {
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;
        return x;
    }

    float hash3(int x, int y, int seed) {
        const std::uint32_t h = hashU(static_cast<std::uint32_t>(x) * 0x9E3779B1U ^
                                      hashU(static_cast<std::uint32_t>(y) + 0x7F4A7C15U * static_cast<std::uint32_t>(seed + 1)));
        return static_cast<float>(h & 0xffffffU) / 16777215.f;
    }

    // Gradient noise, roughly [-1, 1].
    float gnoise(float x, float y, int seed) {
        const float fx = std::floor(x), fy = std::floor(y);
        const int ix = static_cast<int>(fx), iy = static_cast<int>(fy);
        const float tx = x - fx, ty = y - fy;
        auto grad = [&](int a, int b, float dx, float dy) {
            const float ang = hash3(a, b, seed) * kTau;
            return std::cos(ang) * dx + std::sin(ang) * dy;
        };
        const float ux = tx * tx * tx * (tx * (tx * 6 - 15) + 10);
        const float uy = ty * ty * ty * (ty * (ty * 6 - 15) + 10);
        const float n00 = grad(ix, iy, tx, ty), n10 = grad(ix + 1, iy, tx - 1, ty);
        const float n01 = grad(ix, iy + 1, tx, ty - 1), n11 = grad(ix + 1, iy + 1, tx - 1, ty - 1);
        return 1.4142f * mixf(mixf(n00, n10, ux), mixf(n01, n11, ux), uy);
    }

    // Value noise in [-1, 1] with its derivatives.
    void vnoiseD(float x, float y, int seed, float& v, float& dx, float& dy) {
        const float fx = std::floor(x), fy = std::floor(y);
        const int ix = static_cast<int>(fx), iy = static_cast<int>(fy);
        const float tx = x - fx, ty = y - fy;
        const float ux = tx * tx * (3 - 2 * tx), uy = ty * ty * (3 - 2 * ty);
        const float dux = 6 * tx * (1 - tx), duy = 6 * ty * (1 - ty);
        const float a = hash3(ix, iy, seed) * 2 - 1, b = hash3(ix + 1, iy, seed) * 2 - 1;
        const float c = hash3(ix, iy + 1, seed) * 2 - 1, d = hash3(ix + 1, iy + 1, seed) * 2 - 1;
        const float k1 = b - a, k2 = c - a, k3 = a - b - c + d;
        v = a + k1 * ux + k2 * uy + k3 * ux * uy;
        dx = dux * (k1 + k3 * uy);
        dy = duy * (k2 + k3 * ux);
    }

    // Fractal noise whose finer octaves die away where the ground is already steep: crests
    // stay sharp, flanks smooth out and gullies run down them, roughly as erosion leaves a range.
    float erodedFbm(float x, float y, int octaves, int seed) {
        float a = 0, b = 1, dx = 0, dy = 0;
        for (int i = 0; i < octaves; ++i) {
            float v, nx, ny;
            vnoiseD(x, y, seed + i, v, nx, ny);
            dx += nx;
            dy += ny;
            a += b * v / (1.f + dx * dx + dy * dy);
            b *= 0.5f;
            const float x2 = 1.6f * x - 1.2f * y, y2 = 1.2f * x + 1.6f * y;
            x = x2;
            y = y2;
        }
        return a;
    }

    // Plain value-noise fBm in [0, 1] (the per-vertex snow line's wiggle).
    float fbm01(float x, float y, int seed) {
        float s = 0, a = 0.5f, norm = 0;
        for (int i = 0; i < 4; ++i) {
            float v, dx, dy;
            vnoiseD(x, y, seed + 31 * i, v, dx, dy);
            s += a * (0.5f + 0.5f * v);
            norm += a;
            a *= 0.5f;
            x *= 2.f;
            y *= 2.f;
        }
        return s / norm;
    }

    std::string hook(std::string s, const std::string& what, const std::string& with) {
        const auto p = s.find(what);
        if (p == std::string::npos) {
            std::cerr << "MountainBackdrop: shader hook not found: " << what << std::endl;
            return s;
        }
        s.replace(p, what.size(), with);
        return s;
    }

    const char* kVertPars = R"(
attribute float aCavity;
varying vec3 vMountW;
varying float vCavity;
)";

    const char* kFragPars = R"(
uniform vec3 uHorizon;
uniform vec3 uSkySun;
uniform vec3 uSunDirW;
uniform float uHaze;
uniform float uSnowSlope;
uniform vec3 uSnowColor;
uniform vec3 uRockColor;
uniform vec3 uRockBandColor;
varying vec3 vMountW;
varying float vCavity;
float rHash( vec2 p ) {
	vec3 p3 = fract( vec3( p.xyx ) * 0.1031 );
	p3 += dot( p3, p3.yzx + 33.33 );
	return fract( ( p3.x + p3.y ) * p3.z );
}
float rNoise( vec2 p ) {
	vec2 i = floor( p ), f = fract( p );
	vec2 u = f * f * ( 3.0 - 2.0 * f );
	return mix( mix( rHash( i ), rHash( i + vec2( 1.0, 0.0 ) ), u.x ),
	            mix( rHash( i + vec2( 0.0, 1.0 ) ), rHash( i + vec2( 1.0, 1.0 ) ), u.x ), u.y );
}
float rFbm( vec2 p ) {
	float s = 0.0, a = 0.5;
	for ( int i = 0; i < 4; ++ i ) {
		s += a * rNoise( p );
		p = mat2( 1.6, 1.2, - 1.2, 1.6 ) * p;
		a *= 0.5;
	}
	return s;
}
// relief finer than the mesh: buttresses, gullies, ledges (metres)
float rRelief( vec2 p, float f1, float f2 ) {
	return f1 * 30.0 * rFbm( p / 80.0 ) + f2 * 5.0 * rFbm( p / 15.0 + 3.0 );
}
)";

    const char* kSurface = R"(
	vec3 rockN = inverseTransformDirection( normal, viewMatrix );
	vec2 rp = vMountW.xz;
	float rPx = length( fwidth( rp ) );
	float rF1 = 1.0 - smoothstep( 5.0, 20.0, rPx );
	float rF2 = 1.0 - smoothstep( 1.0, 5.0, rPx );
	const float rE = 2.5;
	float r0 = rRelief( rp, rF1, rF2 );
	vec2 rg = vec2( rRelief( rp + vec2( rE, 0.0 ), rF1, rF2 ) - r0, rRelief( rp + vec2( 0.0, rE ), rF1, rF2 ) - r0 ) / rE;
	float steep0 = smoothstep( 0.12, 0.45, 1.0 - rockN.y );
	rockN = normalize( rockN + vec3( - rg.x, 0.0, - rg.y ) * ( 0.2 + 0.8 * steep0 ) );
	float slope = 1.0 - rockN.y;
	float alt = vMountW.y;
	float rN = rFbm( rp / 170.0 );
	float cav = clamp( vCavity, -1.0, 1.0 );
	// snow holds on gentle ground, higher up and in hollows; crests and steep faces go bare
	float snowLine = uSnowSlope + 0.12 * rN + 0.05 * smoothstep( 200.0, 1200.0, alt ) + 0.24 * max( cav, 0.0 ) - 0.2 * max( - cav, 0.0 );
	float snowAmt = 1.0 - smoothstep( snowLine - 0.03, snowLine + 0.03, slope );
	float strata = smoothstep( 0.25, 0.75, 0.5 + 0.5 * sin( alt / 12.0 + rN * 7.0 ) );
	vec3 rockC = mix( uRockColor, uRockBandColor, strata ) * ( 0.7 + 0.6 * rNoise( rp / 30.0 ) );
	// snow caught on the ledges between strata
	float ledge = ( 1.0 - strata ) * smoothstep( 0.55, 0.75, rNoise( rp / 10.0 ) ) * ( 1.0 - smoothstep( 0.45, 0.75, slope ) );
	snowAmt = max( snowAmt, 0.8 * ledge );
	diffuseColor.rgb = mix( rockC, uSnowColor, snowAmt );
	roughnessFactor = mix( 0.92, 0.78, snowAmt );
	normal = normalize( ( viewMatrix * vec4( rockN, 0.0 ) ).xyz );
)";

    // Aerial perspective: range behind range fades into the sky's horizon colour, warmer
    // toward the sun.
    const char* kHaze = R"(
	{
		vec3 rv = vMountW - cameraPosition;
		float rd = length( rv );
		float haze = 1.0 - exp( - rd / uHaze );
		float mu = max( dot( rv / rd, uSunDirW ), 0.0 );
		vec3 hc = mix( uHorizon, uSkySun * 0.75, 0.45 * pow( mu, 3.0 ) );
		gl_FragColor.rgb = mix( gl_FragColor.rgb, hc, min( haze, 0.95 ) );
	}
	#include <tonemapping_fragment>
)";

    std::shared_ptr<ShaderMaterial> perPixelMaterial(const MountainBackdropOptions& o) {
        const auto& lib = shaders::ShaderLib::instance().get("physical");
        auto m = ShaderMaterial::create();
        m->uniforms = lib.uniforms;
        m->uniforms["diffuse"].setValue(Color(1, 1, 1));
        m->uniforms["roughness"].setValue(0.9f);
        m->uniforms["metalness"].setValue(0.f);
        m->uniforms["emissive"].setValue(Color(0, 0, 0));
        m->uniforms["opacity"].setValue(1.f);
        m->uniforms["uHorizon"] = Uniform(o.hazeColor);
        m->uniforms["uSkySun"] = Uniform(o.hazeSunColor);
        m->uniforms["uSunDirW"] = Uniform(Vector3(o.sunDirection).normalize());
        m->uniforms["uHaze"] = Uniform(o.hazeDistance);
        m->uniforms["uSnowSlope"] = Uniform(o.snowSlope);
        m->uniforms["uSnowColor"] = Uniform(o.snowColor);
        m->uniforms["uRockColor"] = Uniform(o.rockColor);
        m->uniforms["uRockBandColor"] = Uniform(o.rockBandColor);

        std::string vs = shaders::ShaderChunk::instance().meshphysical_vert();
        vs = hook(vs, "void main() {", std::string(kVertPars) + "\nvoid main() {");
        vs = hook(vs, "#include <begin_vertex>", "#include <begin_vertex>\n\tvMountW = ( modelMatrix * vec4( transformed, 1.0 ) ).xyz;\n\tvCavity = aCavity;");
        std::string fs = shaders::ShaderChunk::instance().meshphysical_frag();
        fs = hook(fs, "void main() {", std::string(kFragPars) + "\nvoid main() {");
        fs = hook(fs, "#include <normal_fragment_maps>", std::string("#include <normal_fragment_maps>\n") + kSurface);
        fs = hook(fs, "#include <tonemapping_fragment>", kHaze);
        m->vertexShader = vs;
        m->fragmentShader = fs;
        m->lights = true;
        m->fog = false;// the haze above replaces it
        return m;
    }

}// namespace

MountainBackdrop::MountainBackdrop(MountainBackdropOptions options)
    : options_(std::move(options)) {

    const auto& o = options_;
    const int S = std::max(16, o.segments);
    std::vector<float> radii;
    for (float r = std::min(o.innerRadius, o.fullRadius) - 50.f; r < o.outerRadius; r += r * kTau / static_cast<float>(S)) {
        radii.push_back(std::max(r, 1.f));
    }
    const int R = static_cast<int>(radii.size());

    std::vector<float> pos(static_cast<size_t>(R) * S * 3);
    ThreadPool::shared().parallelFor(0, R, [&](int j) {
        for (int i = 0; i < S; ++i) {
            const float a = (static_cast<float>(i) + 0.5f) / static_cast<float>(S) * kTau;
            const float x = radii[j] * std::cos(a), z = radii[j] * std::sin(a);
            const size_t k = (static_cast<size_t>(j) * S + i) * 3;
            pos[k] = x;
            pos[k + 1] = heightAt(x, z);
            pos[k + 2] = z;
        } }, 1);

    // hollows (above zero) and crests (below) against the ground a few rings and segments away
    std::vector<float> cavity(static_cast<size_t>(R) * S, 0.f);
    const int k = std::min(8, std::max(1, (R - 1) / 2));
    for (int j = k; j < R - k; ++j)
        for (int i = 0; i < S; ++i) {
            auto H = [&](int jj, int ii) { return pos[(static_cast<size_t>(jj) * S + ((ii + S) % S)) * 3 + 1]; };
            const float avg = 0.25f * (H(j - k, i) + H(j + k, i) + H(j, i - k) + H(j, i + k));
            cavity[static_cast<size_t>(j) * S + i] = std::clamp((avg - H(j, i)) / 40.f, -1.f, 1.f);
        }

    std::vector<unsigned int> idx;
    idx.reserve(static_cast<size_t>(std::max(0, R - 1)) * S * 6);
    for (int j = 0; j + 1 < R; ++j)
        for (int i = 0; i < S; ++i) {
            const auto a = static_cast<unsigned int>(j * S + i), b = static_cast<unsigned int>(j * S + (i + 1) % S);
            const auto c = static_cast<unsigned int>((j + 1) * S + i), d = static_cast<unsigned int>((j + 1) * S + (i + 1) % S);
            idx.insert(idx.end(), {a, b, c, b, d, c});
        }

    auto geometry = BufferGeometry::create();
    geometry->setAttribute("position", FloatBufferAttribute::create(pos, 3));
    geometry->setAttribute("aCavity", FloatBufferAttribute::create(cavity, 1));
    geometry->setIndex(std::move(idx));
    geometry->computeVertexNormals();
    geometry->computeBoundingSphere();

    std::shared_ptr<Material> material;
    if (o.shading == MountainBackdropOptions::Shading::PerPixel) {
        material = perPixelMaterial(o);
    } else {
        // the per-pixel rule, per vertex: no relief finer than the mesh, no strata
        const auto& n = geometry->getAttribute<float>("normal")->array();
        std::vector<float> col(pos.size());
        for (size_t v = 0; v < pos.size() / 3; ++v) {
            const float x = pos[3 * v], y = pos[3 * v + 1], z = pos[3 * v + 2];
            const float slope = 1.f - n[3 * v + 1];
            const float cav = cavity[v];
            const float line = o.snowSlope + 0.12f * fbm01(x / 170.f, z / 170.f, static_cast<int>(o.seed) + 5) +
                               0.05f * smoothstepf(200.f, 1200.f, y) + 0.24f * std::max(cav, 0.f) - 0.2f * std::max(-cav, 0.f);
            const float snow = 1.f - smoothstepf(line - 0.04f, line + 0.04f, slope);
            const float band = fbm01(x / 30.f, z / 30.f, static_cast<int>(o.seed) + 9);
            Color rock(mixf(o.rockColor.r, o.rockBandColor.r, band), mixf(o.rockColor.g, o.rockBandColor.g, band), mixf(o.rockColor.b, o.rockBandColor.b, band));
            col[3 * v] = mixf(rock.r, o.snowColor.r, snow);
            col[3 * v + 1] = mixf(rock.g, o.snowColor.g, snow);
            col[3 * v + 2] = mixf(rock.b, o.snowColor.b, snow);
        }
        geometry->setAttribute("color", FloatBufferAttribute::create(std::move(col), 3));
        auto m = MeshStandardMaterial::create();
        m->vertexColors = true;
        m->roughness = 0.9f;
        m->metalness = 0.f;
        material = m;
    }

    mesh_ = Mesh::create(geometry, material);
    mesh_->receiveShadow = false;
    add(mesh_);
}

std::shared_ptr<MountainBackdrop> MountainBackdrop::create(MountainBackdropOptions options) {
    return std::make_shared<MountainBackdrop>(std::move(options));
}

float MountainBackdrop::heightAt(float x, float z) const {
    const auto& o = options_;
    const float ground = o.ground ? o.ground(x, z) : 0.f;
    const float r = std::sqrt(x * x + z * z);
    const float env = smoothstepf(o.innerRadius, o.fullRadius, r);
    if (env <= 0.f) return ground - o.sink;

    const int s = static_cast<int>(o.seed);
    const float sx = x / o.featureScale, sz = z / o.featureScale;
    const float wx = sx + 0.35f * gnoise(sx * 0.6f + 3.1f, sz * 0.6f, s - 9);
    const float wz = sz + 0.35f * gnoise(sx * 0.6f, sz * 0.6f + 7.7f, s - 8);
    const float e = erodedFbm(wx * 1.4f, wz * 1.4f, 9, s);
    float ridge = 1.f - std::abs(gnoise(wx * 0.9f, wz * 0.9f, s - 7));
    ridge *= ridge;
    const float mass = smoothstepf(-0.35f, 0.55f, gnoise(x / (2.f * o.featureScale), z / (2.f * o.featureScale), s - 6));
    float crest = 1.f - std::abs(gnoise(wx * 2.3f + 1.7f, wz * 2.3f, s - 5));
    crest = crest * crest * crest;

    float H = (0.3f + 0.7f * mass) * ((0.55f + 0.45f * ridge) * std::max(0.f, 0.45f + e) * o.height +
                                      crest * o.crestHeight * std::max(0.f, 0.3f + e));
    for (const auto& m : o.massifs) {
        const float d = std::hypot(x - m.centre.x, z - m.centre.y);
        H += std::exp(-d * d / (2.f * m.radius * m.radius)) * std::max(0.f, 0.8f + 0.6f * e) * m.height;
    }
    return ground + env * H - o.sink * (1.f - smoothstepf(0.f, 0.12f, env));
}

const std::shared_ptr<Mesh>& MountainBackdrop::mesh() const {
    return mesh_;
}

const MountainBackdropOptions& MountainBackdrop::options() const {
    return options_;
}

void MountainBackdrop::setHaze(const Color& color, const Color& sunColor, const Vector3& sunDirection, float distance) {
    options_.hazeColor = color;
    options_.hazeSunColor = sunColor;
    options_.sunDirection = sunDirection;
    options_.hazeDistance = distance;
    if (auto* m = mesh_->material()->as<ShaderMaterial>()) {
        m->uniforms["uHorizon"].setValue(color);
        m->uniforms["uSkySun"].setValue(sunColor);
        m->uniforms["uSunDirW"].setValue(Vector3(sunDirection).normalize());
        m->uniforms["uHaze"].setValue(distance);
    }
}
