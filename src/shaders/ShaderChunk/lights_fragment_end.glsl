
#if defined( RE_IndirectDiffuse )

	#if defined( PHONG )

		// r184: Phong has no RE_IndirectSpecular to fold the IBL in, so the
		// environment's irradiance joins the ambient one.
		irradiance += iblIrradiance;

	#endif

	RE_IndirectDiffuse( irradiance, geometry, material, reflectedLight );

#endif

#if defined( RE_IndirectSpecular )

	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometry, material, reflectedLight );

#endif

