
uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;

#include <packing>

void main() {

	float mean = 0.0;
	float squared_mean = 0.0;

	// This seems totally useless but it's a crazy work around for a Adreno compiler bug
	float depth = unpackRGBAToDepth( texture2D( shadow_pass, ( gl_FragCoord.xy ) / resolution ) );

	for ( float i = -1.0; i < 1.0 ; i += SAMPLE_RATE) {

		#ifdef HORIZONTAL_PASS

			// Raw, not unpackRGBATo2Half: the vertical pass wrote the moments as
			// floats. Packing them into RGBA8 is what made VSM unusable at any
			// depth range the shadow camera was not hand-fitted to.
			//
			// Both channels are already the moments E[z] and E[z^2], so this is
			// a plain sum of each - no reconstructing E[z^2] from a deviation.
			vec2 distribution = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( i, 0.0 ) * radius ) / resolution ).xy;
			mean += distribution.x;
			squared_mean += distribution.y;

		#else

			float depth = unpackRGBAToDepth( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, i ) * radius ) / resolution ) );
			mean += depth;
			squared_mean += depth * depth;

		#endif

	}

	mean = mean * HALF_SAMPLE_RATE;
	squared_mean = squared_mean * HALF_SAMPLE_RATE;

	// E[z] and E[z^2], not a mean and a standard deviation. Both are linear in
	// the samples, so any later averaging of this target - a mip level, a
	// bilinear tap - stays a valid distribution over the wider footprint, and
	// the receiver recovers the variance itself. A standard deviation does NOT
	// average that way: mip-filtering one silently understates the spread, and
	// on a receiver sloped away from the light that reads as everything
	// shadowing itself.
	gl_FragColor = vec4( mean, squared_mean, 0.0, 1.0 );

}

