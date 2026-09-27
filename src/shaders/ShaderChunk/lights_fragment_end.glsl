
#if defined( RE_IndirectDiffuse )

	#if defined( LAMBERT ) || defined( PHONG )

		// r184: Lambert and Phong have no RE_IndirectSpecular to fold the IBL in, so the
		// environment's irradiance joins the ambient one.
		irradiance += iblIrradiance;

	#endif

	RE_IndirectDiffuse( irradiance, geometry, material, reflectedLight );

#endif

#if defined( RE_IndirectSpecular )

	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometry, material, reflectedLight );

#endif

