
#ifdef USE_TET_SKIN

	// Current (deformed) collision-tet vertex positions, one RGB texel per tet,
	// packed row-major into a square tetTextureSize x tetTextureSize float texture.
	uniform highp sampler2D tetTexture;
	uniform int tetTextureSize;

	vec3 fetchTet( const in sampler2D tex, const in float i ) {

		float size = float( tetTextureSize );
		float x = mod( i, size );
		float y = floor( i / size );
		float d = 1.0 / size;

		return texture2D( tex, vec2( d * ( x + 0.5 ), d * ( y + 0.5 ) ) ).xyz;

	}

	vec3 getTetPos( const in float i ) { return fetchTet( tetTexture, i ); }

	// Deformation gradient F (rest -> current) of the bound tetrahedron, constant
	// within a tet. The rest edge matrix Dr is constant, so its inverse is baked
	// once at bind time into the tetRestInv0/1/2 vertex attributes (the columns of
	// Dr^-1) — no per-vertex inverse or rest-texture fetch here. F = Dc * Dr^-1.
	mat3 tetDeformationGradient() {

		vec3 c0 = getTetPos( tetIndex.x );

		mat3 Dc = mat3( getTetPos( tetIndex.y ) - c0,
		                getTetPos( tetIndex.z ) - c0,
		                getTetPos( tetIndex.w ) - c0 );

		mat3 restInv = mat3( tetRestInv0, tetRestInv1, tetRestInv2 );

		return Dc * restInv;

	}

#endif
