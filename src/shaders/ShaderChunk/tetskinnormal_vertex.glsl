
#ifdef USE_TET_SKIN

	// Skin the rest normal (and tangent) by the bound tet's deformation gradient F.
	// A deforming tet rotates/shears, so the rest normal is carried along by the
	// inverse-transpose of F: n' = normalize( (F^-1)^T * n_rest ). Guard against
	// degenerate / near-singular tets (collapsed, or rest == current) where F is not
	// invertible — in that case keep the original normal.
	{

		mat3 tetF = tetDeformationGradient();

		if ( abs( determinant( tetF ) ) > 1e-12 ) {

			objectNormal = normalize( transpose( inverse( tetF ) ) * objectNormal );

			#ifdef USE_TANGENT

				objectTangent = normalize( tetF * objectTangent );

			#endif

		}

	}

#endif
