
uniform sampler2D t2D;
uniform float backgroundIntensity;

varying vec2 vUv;

void main() {

	vec4 texColor = texture2D( t2D, vUv );

	// GLBackground defines this for an sRGB texture: ShaderMaterial carries no
	// `map`, so the program's mapTexelToLinear cannot know the encoding.
	#ifdef BACKGROUND_SRGB
		texColor = sRGBToLinear( texColor );
	#endif

	texColor.rgb *= backgroundIntensity;

	gl_FragColor = texColor;

	#include <tonemapping_fragment>
	#include <encodings_fragment>

}
