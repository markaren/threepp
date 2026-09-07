
#ifdef ENVMAP_TYPE_CUBE_UV

	// GL IBL samples a prefiltered *equirect LOD-strip* atlas produced by
	// GLPMREM::fromEquirectangular(). The atlas is EQ_STRIP_W x (EQ_STRIP_H *
	// EQ_N_LODS): EQ_N_LODS full-equirect strips stacked vertically, each
	// GGX-prefiltered at roughness L/(EQ_N_LODS-1). The texture uses hardware
	// bilinear with Repeat on U, so the azimuth seam (atan2 at -X) wraps
	// seamlessly and there are no cube-face boundaries to streak. The function
	// is still called `textureCubeUV` so the material shaders
	// (envmap_physical_pars_fragment / envmap_fragment) need no changes.
	//
	// These MUST match the constants in GLPMREM.cpp.
	#define EQ_STRIP_W 512.0
	#define EQ_STRIP_H 256.0
	#define EQ_N_LODS  7.0

	#define EQ_RECIP_2PI 0.15915494309
	#define EQ_RECIP_PI  0.31830988618

	vec2 eqUvFromDir( vec3 dir ) {
		return vec2( atan( dir.z, dir.x ) * EQ_RECIP_2PI + 0.5,
		             asin( clamp( dir.y, -1.0, 1.0 ) ) * EQ_RECIP_PI + 0.5 );
	}

	// Sample one roughness strip. u uses the texture's Repeat wrap (seamless
	// azimuth); v is clamped to this strip's texel-centre range so hardware
	// bilinear cannot bleed into the adjacent strip.
	vec3 eqSampleStrip( sampler2D envMap, vec2 eqUv, float lod ) {
		const float atlasH = EQ_STRIP_H * EQ_N_LODS;
		float vTexel = clamp( eqUv.y * EQ_STRIP_H, 0.5, EQ_STRIP_H - 0.5 );
		float atlasV = ( lod * EQ_STRIP_H + vTexel ) / atlasH;
		return texture2D( envMap, vec2( eqUv.x, atlasV ) ).rgb;
	}

	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		vec2 eqUv = eqUvFromDir( normalize( sampleDir ) );

		float lod = clamp( roughness, 0.0, 1.0 ) * ( EQ_N_LODS - 1.0 );
		float l0 = floor( lod );
		float l1 = min( l0 + 1.0, EQ_N_LODS - 1.0 );
		float f  = lod - l0;

		vec3 c0 = eqSampleStrip( envMap, eqUv, l0 );
		vec3 c1 = eqSampleStrip( envMap, eqUv, l1 );

		return vec4( mix( c0, c1, f ), 1.0 );
	}

#endif
