
#ifdef USE_TET_SKIN

	// Deformed (current) and rest collision-tet vertex positions, one RGB texel per
	// tet, packed row-major into a square tetTextureSize x tetTextureSize float texture.
	uniform highp sampler2D tetTexture;
	uniform highp sampler2D restTetTexture;
	uniform int tetTextureSize;

	vec3 fetchTet( const in sampler2D tex, const in float i ) {

		float size = float( tetTextureSize );
		float x = mod( i, size );
		float y = floor( i / size );
		float d = 1.0 / size;

		return texture2D( tex, vec2( d * ( x + 0.5 ), d * ( y + 0.5 ) ) ).xyz;

	}

	vec3 getTetPos( const in float i ) { return fetchTet( tetTexture, i ); }
	vec3 getRestTetPos( const in float i ) { return fetchTet( restTetTexture, i ); }

	// Deformation gradient F (rest -> current) of the bound tetrahedron. Constant
	// within a tet, so it skins the normal: n' = normalize( (F^-1)^T * n_rest ).
	mat3 tetDeformationGradient() {

		vec3 c0 = getTetPos( tetIndex.x );
		vec3 r0 = getRestTetPos( tetIndex.x );

		mat3 Dc = mat3( getTetPos( tetIndex.y ) - c0,
		                getTetPos( tetIndex.z ) - c0,
		                getTetPos( tetIndex.w ) - c0 );
		mat3 Dr = mat3( getRestTetPos( tetIndex.y ) - r0,
		                getRestTetPos( tetIndex.z ) - r0,
		                getRestTetPos( tetIndex.w ) - r0 );

		return Dc * inverse( Dr );

	}

#endif
