
#include <envmap_common_pars_fragment>
uniform float opacity;
// three.js r146 / r147 scene.backgroundBlurriness and scene.backgroundIntensity.
// A blurred background is read from the PMREM strip atlas (ENVMAP_TYPE_CUBE_UV)
// at roughness = blurriness; envMapRotation carries scene.backgroundRotation.
uniform float backgroundBlurriness;
uniform float backgroundIntensity;

varying vec3 vWorldDirection;

#include <cube_uv_reflection_fragment>

void main() {

	#ifdef ENVMAP_TYPE_CUBE_UV

		vec4 envColor = textureCubeUV( envMap, envMapRotation * vWorldDirection, backgroundBlurriness );

	#else

		vec3 vReflect = vWorldDirection;
		#include <envmap_fragment>

	#endif

	envColor.rgb *= backgroundIntensity;

	gl_FragColor = envColor;
	gl_FragColor.a *= opacity;

	#include <tonemapping_fragment>
	#include <encodings_fragment>

}
