
#ifdef USE_IRIDESCENCE

// Thin-film iridescence, KHR_materials_iridescence.
//
// "A Practical Extension to Microfacet Theory for the Modeling of Varying
// Iridescence", Belcour & Barla 2017.
// https://belcour.github.io/blog/research/publication/2017/05/01/brdf-thin-film.html
//
// The lobe shape is untouched: only the Fresnel base shifts per channel, as
// wavelength-dependent interference between the two film interfaces. Kept
// numerically identical to the Vulkan deferred path in
// renderers/vulkan/shaders/deferred_shade_10_lighting_utils.glsl, which is
// itself a verbatim port of the three.js chunk.
//
// Needs only PI from <common>, and is reached solely by #include from
// meshphysical_frag.

vec3 iridFresnel0ToIor( vec3 F0 ) {

	vec3 sqrtF0 = sqrt( F0 );
	return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );

}

vec3 iridIorToFresnel0_v( vec3 transmittedIor, float incidentIor ) {

	return pow( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ), vec3( 2.0 ) );

}

float iridIorToFresnel0_s( float transmittedIor, float incidentIor ) {

	return pow( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ), 2.0 );

}

// Fitted analytic spectral integral of the XYZ colour matching functions
// against a cosine of the given optical path difference. This is what turns a
// film thickness into a hue.
vec3 iridSensitivity( float OPD, vec3 shift ) {

	float phase = 2.0 * PI * OPD * 1.0e-9;
	vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
	vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
	vec3 vr  = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );

	vec3 xyz = val * sqrt( 2.0 * PI * vr ) * cos( pos * phase + shift ) * exp( - ( phase * phase ) * vr );
	xyz.x   += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift.x ) * exp( - 4.5282e+09 * ( phase * phase ) );
	xyz     /= 1.0685e-7;

	// XYZ to linear sRGB, D65. Nine scalars fill a mat3 COLUMN-major, so this
	// is the transpose of how the matrix is usually written down.
	mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252 );

	return XYZ_TO_REC709 * xyz;

}

vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {

	// Force iridescenceIOR -> outsideIOR when thinFilmThickness == 0, which
	// reduces the whole thing to the base F0.
	float iridescenceIor = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );

	// Snell, to the angle inside the film.
	float sinTheta2Sq = pow( outsideIOR / iridescenceIor, 2.0 ) * ( 1.0 - pow( cosTheta1, 2.0 ) );
	float cosTheta2Sq = 1.0 - sinTheta2Sq;

	// Total internal reflection at the film's top interface.
	if ( cosTheta2Sq < 0.0 ) return vec3( 1.0 );

	float cosTheta2 = sqrt( cosTheta2Sq );

	// First interface, above the film.
	float R0 = iridIorToFresnel0_s( iridescenceIor, outsideIOR );
	float R12 = R0 + ( 1.0 - R0 ) * pow( 1.0 - cosTheta1, 5.0 );
	float T121 = 1.0 - R12;
	float phi12 = 0.0;
	if ( iridescenceIor < outsideIOR ) phi12 = PI;
	float phi21 = PI - phi12;

	// Second interface, the substrate. Its IOR is recovered from baseF0.
	vec3 baseIOR = iridFresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );
	vec3 R1 = iridIorToFresnel0_v( baseIOR, iridescenceIor );
	vec3 R23 = R1 + ( vec3( 1.0 ) - R1 ) * pow( 1.0 - cosTheta2, 5.0 );
	vec3 phi23 = vec3( 0.0 );
	if ( baseIOR.x < iridescenceIor ) phi23.x = PI;
	if ( baseIOR.y < iridescenceIor ) phi23.y = PI;
	if ( baseIOR.z < iridescenceIor ) phi23.z = PI;

	// Optical path difference, and the phase it induces.
	float OPD = 2.0 * iridescenceIor * thinFilmThickness * cosTheta2;
	vec3 phi = vec3( phi21 ) + phi23;

	// Compound reflectance, Belcour 2017 section 3.
	vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
	vec3 r123 = sqrt( R123 );
	vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );

	// Zeroth Fourier term: the achromatic base reflectance. Above roughly
	// 2000 nm the exp in iridSensitivity has annihilated everything else and
	// the result collapses to exactly this.
	vec3 C0 = R12 + Rs;
	vec3 I = C0;

	// Higher-order Fourier terms: the spectral cosine integrals that make the
	// hue.
	vec3 Cm = Rs - T121;
	for ( int m = 1; m <= 2; ++ m ) {

		Cm *= r123;
		vec3 Sm = 2.0 * iridSensitivity( float( m ) * OPD, float( m ) * phi );
		I += Cm * Sm;

	}

	return max( I, vec3( 0.0 ) );

}

#endif
