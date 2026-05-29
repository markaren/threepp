
#ifdef USE_TET_SKIN

	// Skin the rest normal (and tangent) by the bound tet's deformation gradient F.
	// The normal transforms by the inverse-transpose of F, which up to a positive
	// scale equals the cofactor matrix (cross products of F's columns), so we get
	// (F^-1)^T with no per-vertex matrix inverse. The det(F) magnitude scales out
	// under normalize; sign(det) preserves orientation through a (rare) inverted
	// tet, and the guard keeps the rest normal for degenerate / collapsed tets.
	{

		mat3 tetF = tetDeformationGradient();

		vec3 f0 = tetF[ 0 ];
		vec3 f1 = tetF[ 1 ];
		vec3 f2 = tetF[ 2 ];
		vec3 cof0 = cross( f1, f2 );
		float detF = dot( f0, cof0 );

		if ( abs( detF ) > 1e-12 ) {

			mat3 cofF = mat3( cof0, cross( f2, f0 ), cross( f0, f1 ) );
			objectNormal = normalize( sign( detF ) * ( cofF * objectNormal ) );

			#ifdef USE_TANGENT

				objectTangent = normalize( tetF * objectTangent );

			#endif

		}

	}

#endif
