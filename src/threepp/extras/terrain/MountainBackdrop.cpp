
#include "threepp/extras/terrain/MountainBackdrop.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/core/NoiseUtils.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/renderers/shaders/ShaderChunk.hpp"
#include "threepp/renderers/shaders/ShaderLib.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/utils/Log.hpp"
#include "threepp/utils/ThreadPool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

using namespace threepp;
using namespace threepp::terrain;

namespace {

    constinit log::Category kLog{"extras.terrain"};

    constexpr float kTau = 6.28318530717959f;



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
        const float ux = noise::fade(tx);
        const float uy = noise::fade(ty);
        const float n00 = grad(ix, iy, tx, ty), n10 = grad(ix + 1, iy, tx - 1, ty);
        const float n01 = grad(ix, iy + 1, tx, ty - 1), n11 = grad(ix + 1, iy + 1, tx - 1, ty - 1);
        return 1.4142f * noise::lerp(noise::lerp(n00, n10, ux), noise::lerp(n01, n11, ux), uy);
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

    // CPU copies of the per-pixel shader's rHash / rNoise / rFbm / rRelief (below), for the
    // textured mode, which bakes the same rule.
    float fract(float x) { return x - std::floor(x); }
    float rHash(float px, float py) {
        float x = fract(px * 0.1031f), y = fract(py * 0.1031f), z = x;
        const float d = x * (y + 33.33f) + y * (z + 33.33f) + z * (x + 33.33f);
        x += d;
        y += d;
        z += d;
        return fract((x + y) * z);
    }
    float rNoise(float px, float py) {
        const float ix = std::floor(px), iy = std::floor(py), fx = px - ix, fy = py - iy;
        const float ux = noise::smooth(fx), uy = noise::smooth(fy);
        return noise::lerp(noise::lerp(rHash(ix, iy), rHash(ix + 1, iy), ux), noise::lerp(rHash(ix, iy + 1), rHash(ix + 1, iy + 1), ux), uy);
    }
    float rFbm(float px, float py) {
        float s = 0, a = 0.5f;
        for (int i = 0; i < 4; ++i) {
            s += a * rNoise(px, py);
            const float x2 = 1.6f * px - 1.2f * py, y2 = 1.2f * px + 1.6f * py;
            px = x2;
            py = y2;
            a *= 0.5f;
        }
        return s;
    }
    float rRelief(float px, float py, float f1, float f2) {
        return f1 * 30.f * rFbm(px / 80.f, py / 80.f) + f2 * 5.f * rFbm(px / 15.f + 3.f, py / 15.f + 3.f);
    }


    std::string hook(std::string s, const std::string& what, const std::string& with) {
        const auto p = s.find(what);
        if (p == std::string::npos) {
            THREEPP_LOG(kLog, Warn) << "shader hook not found: " << what;
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
    const bool textured = o.shading == MountainBackdropOptions::Shading::Textured;
    // Textured: one more column, a copy of the first at u = 1, so the maps do not wrap
    // backwards across the last segment.
    const int cols = textured ? S + 1 : S;
    std::vector<float> radii;
    for (float r = std::min(o.innerRadius, o.fullRadius) - 50.f; r < o.outerRadius; r += r * kTau / static_cast<float>(S)) {
        radii.push_back(std::max(r, 1.f));
    }
    const int R = static_cast<int>(radii.size());

    std::vector<float> pos(static_cast<size_t>(R) * cols * 3);
    ThreadPool::shared().parallelFor(0, R, [&](int j) {
        for (int i = 0; i < cols; ++i) {
            const float a = (static_cast<float>(i % S) + 0.5f) / static_cast<float>(S) * kTau;
            const float x = radii[j] * std::cos(a), z = radii[j] * std::sin(a);
            const size_t k = (static_cast<size_t>(j) * cols + i) * 3;
            pos[k] = x;
            pos[k + 1] = heightAt(x, z);
            pos[k + 2] = z;
        } }, 1);

    // hollows (above zero) and crests (below) against the ground a few rings and segments away
    std::vector<float> cavity(static_cast<size_t>(R) * cols, 0.f);
    const int k = std::min(8, std::max(1, (R - 1) / 2));
    for (int j = k; j < R - k; ++j)
        for (int i = 0; i < cols; ++i) {
            auto H = [&](int jj, int ii) { return pos[(static_cast<size_t>(jj) * cols + ((ii + S) % S)) * 3 + 1]; };
            const float avg = 0.25f * (H(j - k, i) + H(j + k, i) + H(j, i - k) + H(j, i + k));
            cavity[static_cast<size_t>(j) * cols + i] = std::clamp((avg - H(j, i % S)) / 40.f, -1.f, 1.f);
        }

    std::vector<unsigned int> idx;
    idx.reserve(static_cast<size_t>(std::max(0, R - 1)) * S * 6);
    for (int j = 0; j + 1 < R; ++j)
        for (int i = 0; i < S; ++i) {
            const int i1 = textured ? i + 1 : (i + 1) % S;
            const auto a = static_cast<unsigned int>(j * cols + i), b = static_cast<unsigned int>(j * cols + i1);
            const auto c = static_cast<unsigned int>((j + 1) * cols + i), d = static_cast<unsigned int>((j + 1) * cols + i1);
            idx.insert(idx.end(), {a, b, c, b, d, c});
        }

    auto geometry = BufferGeometry::create();
    geometry->setAttribute("position", FloatBufferAttribute::create(pos, 3));
    geometry->setAttribute("aCavity", FloatBufferAttribute::create(cavity, 1));
    geometry->setIndex(std::move(idx));
    geometry->computeVertexNormals();
    geometry->computeBoundingSphere();

    std::shared_ptr<Material> material;
    if (!textured) {
        material = perPixelMaterial(o);
    } else {
        // The per-pixel rule baked into a colour and a normal map: u round the ring, v by
        // ring index, so a texel is as square as a segment at every distance.
        const int tps = std::max(1, o.texelsPerSegment);
        const int W = S * tps, H = std::max(2, (R - 1) * tps);
        std::vector<float> uv(static_cast<size_t>(R) * cols * 2);
        for (int j = 0; j < R; ++j)
            for (int i = 0; i < cols; ++i) {
                uv[(static_cast<size_t>(j) * cols + i) * 2] = static_cast<float>(i) / static_cast<float>(S);
                uv[(static_cast<size_t>(j) * cols + i) * 2 + 1] = static_cast<float>(j) / static_cast<float>(R - 1);
            }
        geometry->setAttribute("uv", FloatBufferAttribute::create(std::move(uv), 2));

        const auto& nrm = geometry->getAttribute<float>("normal")->array();
        std::vector<unsigned char> colPx(static_cast<size_t>(W) * H * 4, 255), nrmPx(static_cast<size_t>(W) * H * 4, 255);
        const float snowR = o.snowColor.r, snowG = o.snowColor.g, snowB = o.snowColor.b;
        ThreadPool::shared().parallelFor(0, H, [&](int y) {
            const float jf = (static_cast<float>(y) + 0.5f) / static_cast<float>(H) * static_cast<float>(R - 1);
            const int j0 = std::min(static_cast<int>(jf), R - 2);
            const float tj = jf - static_cast<float>(j0);
            const float r = noise::lerp(radii[j0], radii[j0 + 1], tj);
            const float texel = r * kTau / static_cast<float>(W);
            // relief finer than a texel is left out, as the shader leaves out what is finer than a pixel
            const float f1 = 1.f - math::smoothstep(5.f, 20.f, texel), f2 = 1.f - math::smoothstep(1.f, 5.f, texel);
            for (int x = 0; x < W; ++x) {
                const float uf = (static_cast<float>(x) + 0.5f) / static_cast<float>(tps);
                const int i0 = std::min(static_cast<int>(uf), S - 1);
                const float ti = uf - static_cast<float>(i0);
                auto at = [&](const std::vector<float>& a, int stride, int c) {
                    auto v = [&](int jj, int ii) { return a[(static_cast<size_t>(jj) * cols + ii) * stride + c]; };
                    return noise::lerp(noise::lerp(v(j0, i0), v(j0, i0 + 1), ti), noise::lerp(v(j0 + 1, i0), v(j0 + 1, i0 + 1), ti), tj);
                };
                const float a = (uf + 0.5f) / static_cast<float>(S) * kTau;
                const float ca = std::cos(a), sa = std::sin(a);
                const float px = r * ca, pz = r * sa, alt = at(pos, 3, 1);
                Vector3 N(at(nrm, 3, 0), at(nrm, 3, 1), at(nrm, 3, 2));
                N.normalize();

                const float e = 2.5f;
                const float r0 = rRelief(px, pz, f1, f2);
                const float gx = (rRelief(px + e, pz, f1, f2) - r0) / e, gz = (rRelief(px, pz + e, f1, f2) - r0) / e;
                const float steep0 = math::smoothstep(0.12f, 0.45f, 1.f - N.y);
                Vector3 n = N + Vector3(-gx, 0.f, -gz) * (0.2f + 0.8f * steep0);
                n.normalize();
                const float slope = 1.f - n.y;
                const float rN = rFbm(px / 170.f, pz / 170.f);
                const float cav = std::clamp(at(cavity, 1, 0), -1.f, 1.f);
                const float line = o.snowSlope + 0.12f * rN + 0.05f * math::smoothstep(200.f, 1200.f, alt) + 0.24f * std::max(cav, 0.f) - 0.2f * std::max(-cav, 0.f);
                float snow = 1.f - math::smoothstep(line - 0.03f, line + 0.03f, slope);
                const float strata = math::smoothstep(0.25f, 0.75f, 0.5f + 0.5f * std::sin(alt / 12.f + rN * 7.f));
                const float grain = 0.7f + 0.6f * rNoise(px / 30.f, pz / 30.f);
                const float ledge = (1.f - strata) * math::smoothstep(0.55f, 0.75f, rNoise(px / 10.f, pz / 10.f)) * (1.f - math::smoothstep(0.45f, 0.75f, slope));
                snow = std::max(snow, 0.8f * ledge);
                const float rr = noise::lerp(o.rockColor.r, o.rockBandColor.r, strata) * grain;
                const float rg = noise::lerp(o.rockColor.g, o.rockBandColor.g, strata) * grain;
                const float rb = noise::lerp(o.rockColor.b, o.rockBandColor.b, strata) * grain;

                // the normal in the frame the renderer builds from the uvs: T round the ring
                // (u), B = N x T outward (v)
                Vector3 T(-sa, 0.f, ca);
                T.sub(Vector3(N).multiplyScalar(T.dot(N))).normalize();
                Vector3 B = Vector3().crossVectors(N, T);
                const float nx = n.dot(T), ny = n.dot(B), nz = n.dot(N);

                const size_t k = (static_cast<size_t>(y) * W + x) * 4;
                auto srgb = [](float c) {
                    c = std::clamp(c, 0.f, 1.f);
                    c = c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.f / 2.4f) - 0.055f;
                    return static_cast<unsigned char>(std::lround(c * 255.f));
                };
                colPx[k] = srgb(noise::lerp(rr, snowR, snow));
                colPx[k + 1] = srgb(noise::lerp(rg, snowG, snow));
                colPx[k + 2] = srgb(noise::lerp(rb, snowB, snow));
                auto unit8 = [](float c) { return static_cast<unsigned char>(std::lround(std::clamp(0.5f + 0.5f * c, 0.f, 1.f) * 255.f)); };
                nrmPx[k] = unit8(nx);
                nrmPx[k + 1] = unit8(ny);
                nrmPx[k + 2] = unit8(nz);
            }
        });
        auto makeTex = [&](std::vector<unsigned char> px, ColorSpace cs) {
            auto tex = DataTexture::create(ImageData(std::move(px)), W, H);
            tex->format = Format::RGBA;
            tex->type = Type::UnsignedByte;
            tex->colorSpace = cs;
            tex->wrapS = TextureWrapping::Repeat;
            tex->wrapT = TextureWrapping::ClampToEdge;
            tex->magFilter = Filter::Linear;
            tex->minFilter = Filter::LinearMipmapLinear;
            tex->generateMipmaps = true;
            tex->anisotropy = 8;
            tex->needsUpdate();
            return tex;
        };
        auto m = MeshStandardMaterial::create();
        m->map = makeTex(std::move(colPx), ColorSpace::sRGB);
        m->normalMap = makeTex(std::move(nrmPx), ColorSpace::NoColorSpace);
        m->roughness = 0.88f;
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
    const float env = math::smoothstep(o.innerRadius, o.fullRadius, r);
    if (env <= 0.f) return ground - o.sink;

    const int s = static_cast<int>(o.seed);
    const float sx = x / o.featureScale, sz = z / o.featureScale;
    const float wx = sx + 0.35f * gnoise(sx * 0.6f + 3.1f, sz * 0.6f, s - 9);
    const float wz = sz + 0.35f * gnoise(sx * 0.6f, sz * 0.6f + 7.7f, s - 8);
    const float e = erodedFbm(wx * 1.4f, wz * 1.4f, 9, s);
    float ridge = 1.f - std::abs(gnoise(wx * 0.9f, wz * 0.9f, s - 7));
    ridge *= ridge;
    const float mass = math::smoothstep(-0.35f, 0.55f, gnoise(x / (2.f * o.featureScale), z / (2.f * o.featureScale), s - 6));
    float crest = 1.f - std::abs(gnoise(wx * 2.3f + 1.7f, wz * 2.3f, s - 5));
    crest = crest * crest * crest;

    float H = (0.3f + 0.7f * mass) * ((0.55f + 0.45f * ridge) * std::max(0.f, 0.45f + e) * o.height +
                                      crest * o.crestHeight * std::max(0.f, 0.3f + e));
    for (const auto& m : o.massifs) {
        const float d = std::hypot(x - m.centre.x, z - m.centre.y);
        H += std::exp(-d * d / (2.f * m.radius * m.radius)) * std::max(0.f, 0.8f + 0.6f * e) * m.height;
    }
    return ground + env * H - o.sink * (1.f - math::smoothstep(0.f, 0.12f, env));
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
