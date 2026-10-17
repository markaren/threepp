
#define STANDARD

#ifdef PHYSICAL
	#define REFLECTIVITY
	#define CLEARCOAT
#endif

uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;

#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform vec3 attenuationColor;
	uniform float attenuationDistance;
#endif

#ifdef REFLECTIVITY
	uniform float reflectivity;
#endif

#ifdef CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif

#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
#endif

#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
#endif

#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessNm;
#endif

varying vec3 vViewPosition;

#ifndef FLAT_SHADED

	varying vec3 vNormal;

	#ifdef USE_TANGENT

		varying vec3 vTangent;
		varying vec3 vBitangent;

	#endif

#endif

#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <uv2_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <bsdfs>
#include <iridescence_pars_fragment>
#include <transmission_pars_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>

#ifdef USE_PLANAR_REFLECTION

	// A planar mirror of the near scene (renderers/PlanarReflection), read in
	// <lights_fragment_maps> in place of the environment where it was drawn.
	uniform sampler2D planarReflection;
	uniform mat4 planarReflectionMatrix;
	uniform float planarReflectionDistortion;
	varying vec3 vPlanarReflectionPosition;

	// The mirror's radiance for this fragment, alpha 0 where it holds nothing
	// (nothing drawn, or off its frame). The surface point is projected through
	// the mirrored camera after a push along the shading normal's tilt off the
	// geometric normal: a wave slope moves the mirrored image as it would move
	// the reflected ray, the way objects/Water shifts its lookup by the normal.
	// A short fade at the frame's edges hands over to the environment softly.
	vec4 getPlanarReflection( const in vec3 shadingNormal, const in vec3 flatNormal ) {

		vec3 n = inverseTransformDirection( shadingNormal, viewMatrix );
		vec3 g = inverseTransformDirection( flatNormal, viewMatrix );
		vec3 tilt = n - g * dot( n, g );
		vec4 coord = planarReflectionMatrix * vec4( vPlanarReflectionPosition + tilt * planarReflectionDistortion, 1.0 );

		if ( coord.w <= 0.0 ) return vec4( 0.0 );

		vec2 uv = coord.xy / coord.w;
		if ( any( lessThan( uv, vec2( 0.0 ) ) ) || any( greaterThan( uv, vec2( 1.0 ) ) ) ) return vec4( 0.0 );

		vec4 mirror = texture2D( planarReflection, uv );
		float edge = min( min( uv.x, 1.0 - uv.x ), min( uv.y, 1.0 - uv.y ) );
		mirror.a *= smoothstep( 0.0, 0.05, edge );
		return mirror;

	}

#endif

#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <lights_physical_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>

void main() {


	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;

	#ifdef USE_TRANSMISSION
		float totalTransmission = transmission;
		float thicknessFactor = thickness;
	#endif

	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>

	// accumulation
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>

	// modulation
	#include <aomap_fragment>

	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;

	#include <transmission_fragment>

	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;

	#ifdef CLEARCOAT

		// See clearcoatSpecular in <lights_physical_pars_fragment>.
		float dotNVcc = saturate( dot( geometry.clearcoatNormal, geometry.viewDir ) );

		vec3 Fcc = F_Schlick( vec3( DEFAULT_SPECULAR_COEFFICIENT ), 1.0, dotNVcc );

		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + clearcoatSpecular * material.clearcoat;

	#endif

	#include <opaque_fragment>

	#include <tonemapping_fragment>
	#include <encodings_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>

}
