#include "threepp/objects/ParticleSprites.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/ParticleSystem.hpp"
#include "threepp/renderers/shaders/UniformsLib.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

using namespace threepp;

namespace {

    // The billboard layout ParticleSystem uses, which VulkanRenderer's particle pass reads:
    // four vertices per sprite, all at its centre, with
    //   normal = {size, angle, opacity}   uv = the corner (-0.5..0.5)   color = rgb
    const char* kVertex = R"(
varying vec4 vColor;
varying vec2 vUv;
varying vec3 vViewDir;
#include <common>
#include <fog_pars_vertex>
void main() {
	float pSize = normal.x;
	float pAngle = normal.y;
	float pOpacity = normal.z;
	vColor = vec4( color, pOpacity );
	vUv = uv + vec2( 0.5 );

	float c = cos( pAngle ), s = sin( pAngle );
	vec2 corner = vec2( c * uv.x - s * uv.y, s * uv.x + c * uv.y );

	vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
	vViewDir = normalize( mvPosition.xyz );
	gl_Position = projectionMatrix * mvPosition;
	// pSize world units across, square on screen: each axis takes its own projection scale
	vec2 scale = vec2( projectionMatrix[ 0 ][ 0 ], projectionMatrix[ 1 ][ 1 ] ) * pSize / length( mvPosition.xyz );
	gl_Position.xy += corner * scale * gl_Position.w;
	if ( pOpacity <= 0.0 ) gl_Position = vec4( 2.0, 2.0, 2.0, 1.0 );// hidden: outside the clip volume
	#include <fog_vertex>
}
)";

    const char* kFragment = R"(
uniform sampler2D tex;
uniform float lit;
uniform float anisotropy;
uniform float selfShadow;
varying vec4 vColor;
varying vec2 vUv;
varying vec3 vViewDir;
#include <common>
#include <lights_pars_begin>
#include <fog_pars_fragment>

// Henyey-Greenstein, x4pi so an isotropic lobe is 1 (the same lobe VulkanRenderer lights
// its particles with, which reads the same uniform).
float spritePhase( float mu ) {
	float g = anisotropy;
	float d = 1.0 + g * g - 2.0 * g * mu;
	return ( 1.0 - g * g ) / ( d * sqrt( d ) );
}

void main() {
	vec4 t = texture2D( tex, vUv );
	float alpha = vColor.a * t.a;
	if ( alpha < 1.0 / 255.0 ) discard;
	vec3 col = vColor.rgb * t.rgb;
	if ( lit > 0.5 ) {
		// a puff has no surface normal: light from every side counts
		vec3 ambient = ambientLightColor;
		#if NUM_HEMI_LIGHTS > 0
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			ambient += 0.5 * ( hemisphereLights[ i ].skyColor + hemisphereLights[ i ].groundColor );
		}
		#endif
		vec3 direct = vec3( 0.0 );
		#if NUM_DIR_LIGHTS > 0
		for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
			direct += directionalLights[ i ].color * spritePhase( dot( vViewDir, directionalLights[ i ].direction ) );
		}
		#endif
		// a dense puff keeps part of the light from one direction from itself
		direct *= 1.0 - selfShadow;
		#ifdef USE_LEGACY_LIGHTS
		ambient *= PI;
		direct *= PI;
		#endif
		col *= ( ambient + direct ) * RECIPROCAL_PI;
	}
	gl_FragColor = vec4( col, alpha );
	#include <tonemapping_fragment>
	#include <encodings_fragment>
	#include <fog_fragment>
}
)";

}// namespace

struct ParticleSprites::Impl {
    size_t capacity;
    std::shared_ptr<Texture> texture;
    std::shared_ptr<BufferGeometry> geometry;
    std::shared_ptr<ShaderMaterial> material;
    std::shared_ptr<Mesh> mesh;
    std::vector<float>* position{};
    std::vector<float>* normal{};
    std::vector<float>* color{};

    Impl(size_t n, std::shared_ptr<Texture> tex)
        : capacity(n), texture(tex ? std::move(tex) : softDisc()) {

        if (n == 0) throw std::invalid_argument("ParticleSprites: capacity must be > 0");

        const size_t verts = n * 4;
        std::vector<float> uv(verts * 2);
        std::vector<unsigned int> index(n * 6);
        const float cu[] = {-0.5f, 0.5f, 0.5f, -0.5f};
        const float cv[] = {-0.5f, -0.5f, 0.5f, 0.5f};
        for (size_t i = 0; i < n; ++i) {
            for (int j = 0; j < 4; ++j) {
                uv[(i * 4 + j) * 2] = cu[j];
                uv[(i * 4 + j) * 2 + 1] = cv[j];
            }
            const auto v = static_cast<unsigned int>(i * 4);
            const unsigned int quad[] = {v, v + 1, v + 2, v, v + 2, v + 3};
            std::copy(std::begin(quad), std::end(quad), index.begin() + static_cast<std::ptrdiff_t>(i * 6));
        }

        geometry = BufferGeometry::create();
        geometry->setAttribute("position", FloatBufferAttribute::create(std::vector<float>(verts * 3, 0.f), 3));
        geometry->setAttribute("normal", FloatBufferAttribute::create(std::vector<float>(verts * 3, 0.f), 3));
        geometry->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        geometry->setAttribute("color", FloatBufferAttribute::create(std::vector<float>(verts * 3, 1.f), 3));
        geometry->setIndex(index);
        position = &geometry->getAttribute<float>("position")->array();
        normal = &geometry->getAttribute<float>("normal")->array();
        color = &geometry->getAttribute<float>("color")->array();

        material = ShaderMaterial::create();
        // VulkanRenderer has no general ShaderMaterial path; this name sends the mesh to its
        // particle pass, which reads the layout above.
        material->name = kParticleMaterialName;
        material->vertexShader = kVertex;
        material->fragmentShader = kFragment;
        material->uniforms = shaders::UniformsLib::instance().lights;
        for (const auto& [k, u] : shaders::UniformsLib::instance().fog) material->uniforms[k] = u;
        material->uniforms["tex"] = Uniform(texture.get());
        material->uniforms["lit"] = Uniform(1.f);
        material->uniforms["anisotropy"] = Uniform(0.35f);
        material->uniforms["selfShadow"] = Uniform(0.f);
        material->lights = true;
        material->fog = true;
        material->vertexColors = true;
        material->transparent = true;
        material->depthWrite = false;
        material->side = Side::Double;

        mesh = Mesh::create(geometry, material);
        mesh->frustumCulled = false;// the corners only exist on the GPU
    }

    void write(std::vector<float>& a, size_t i, float x, float y, float z) const {
        float* p = a.data() + i * 12;
        for (int j = 0; j < 4; ++j) {
            p[3 * j] = x;
            p[3 * j + 1] = y;
            p[3 * j + 2] = z;
        }
    }
};

ParticleSprites::ParticleSprites(size_t capacity, std::shared_ptr<Texture> texture)
    : pimpl_(std::make_unique<Impl>(capacity, std::move(texture))) {
    add(pimpl_->mesh);
}

std::shared_ptr<ParticleSprites> ParticleSprites::create(size_t capacity, std::shared_ptr<Texture> texture) {
    return std::shared_ptr<ParticleSprites>(new ParticleSprites(capacity, std::move(texture)));
}

size_t ParticleSprites::capacity() const {
    return pimpl_->capacity;
}

void ParticleSprites::set(size_t i, const Vector3& centre, float size, float opacity, const Color& color, float angle) {
    if (i >= pimpl_->capacity) return;
    pimpl_->write(*pimpl_->position, i, centre.x, centre.y, centre.z);
    pimpl_->write(*pimpl_->normal, i, size, angle, std::clamp(opacity, 0.f, 1.f));
    pimpl_->write(*pimpl_->color, i, color.r, color.g, color.b);
}

void ParticleSprites::hide(size_t i) {
    if (i >= pimpl_->capacity) return;
    float* p = pimpl_->normal->data() + i * 12;
    for (int j = 0; j < 4; ++j) p[3 * j + 2] = 0.f;
}

void ParticleSprites::hideAll() {
    for (size_t i = 0; i < pimpl_->capacity; ++i) hide(i);
}

void ParticleSprites::setAnisotropy(float g) {
    pimpl_->material->uniforms["anisotropy"].setValue(std::clamp(g, -0.9f, 0.9f));
}

float ParticleSprites::anisotropy() const {
    return pimpl_->material->uniforms.at("anisotropy").value<float>();
}

void ParticleSprites::setSelfShadowing(float k) {
    pimpl_->material->uniforms["selfShadow"].setValue(std::clamp(k, 0.f, 0.95f));
}

float ParticleSprites::selfShadowing() const {
    return pimpl_->material->uniforms.at("selfShadow").value<float>();
}

void ParticleSprites::commit() {
    pimpl_->material->uniforms["lit"].setValue(pimpl_->material->blending == Blending::Normal ? 1.f : 0.f);
    pimpl_->geometry->getAttribute<float>("position")->needsUpdate();
    pimpl_->geometry->getAttribute<float>("normal")->needsUpdate();
    pimpl_->geometry->getAttribute<float>("color")->needsUpdate();
}

Mesh& ParticleSprites::mesh() const {
    return *pimpl_->mesh;
}

std::shared_ptr<Texture> ParticleSprites::softDisc(int size) {
    size = std::max(size, 4);
    std::vector<unsigned char> px(static_cast<size_t>(size) * size * 4);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(size) * 2.f - 1.f;
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(size) * 2.f - 1.f;
            const float k = std::max(0.f, 1.f - (u * u + v * v));
            unsigned char* p = px.data() + (static_cast<size_t>(y) * size + x) * 4;
            p[0] = p[1] = p[2] = 255;
            p[3] = static_cast<unsigned char>(std::lround(k * k * 255.f));
        }
    }
    auto tex = DataTexture::create(ImageData(std::move(px)), size, size);
    tex->format = Format::RGBA;
    tex->type = Type::UnsignedByte;
    tex->magFilter = Filter::Linear;
    tex->minFilter = Filter::LinearMipmapLinear;
    tex->needsUpdate();
    return tex;
}

ParticleSprites::~ParticleSprites() = default;
