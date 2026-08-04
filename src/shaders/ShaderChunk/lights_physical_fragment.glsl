
PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb * ( 1.0 - metalnessFactor );

vec3 dxy = max( abs( dFdx( geometryNormal ) ), abs( dFdy( geometryNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );

material.specularRoughness = max( roughnessFactor, 0.0525 );// 0.0525 corresponds to the base mip of a 256 cubemap.
material.specularRoughness += geometryRoughness;
material.specularRoughness = min( material.specularRoughness, 1.0 );

#ifdef REFLECTIVITY

	// threepp's setIor maps reflectivity = 2.5(ior-1)/(ior+1), so
	// 0.16 * reflectivity^2 == ((ior-1)/(ior+1))^2 — algebraically the same
	// dielectric F0 three.js derives from an `ior` uniform. No second uniform
	// is needed for KHR_materials_specular.
	float dielectricF0 = MAXIMUM_SPECULAR_COEFFICIENT * pow2( reflectivity );

#else

	float dielectricF0 = DEFAULT_SPECULAR_COEFFICIENT;

#endif

#ifdef USE_SPECULAR

	// KHR_materials_specular. specularColor tints the dielectric F0 and
	// specularIntensity scales it; both leave the metal branch alone, since a
	// metal's F0 is its albedo. F90 drops with intensity so grazing angles dim
	// too — that is the whole point of intensity 0 meaning "no specular".
	float specularIntensityFactor = specularIntensity;
	vec3 specularColorFactor = specularColor;

	material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );

	material.specularF0 = mix(
		min( vec3( dielectricF0 ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor,
		diffuseColor.rgb,
		metalnessFactor );

#else

	// 1.0 is the F90 every lobe assumed implicitly before this extension existed.
	material.specularF90 = 1.0;
	material.specularF0 = mix( vec3( dielectricF0 ), diffuseColor.rgb, metalnessFactor );

#endif

#ifdef CLEARCOAT

	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;

	#ifdef USE_CLEARCOATMAP

		material.clearcoat *= texture2D( clearcoatMap, vUv ).x;

	#endif

	#ifdef USE_CLEARCOAT_ROUGHNESSMAP

		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vUv ).y;

	#endif

	material.clearcoat = saturate( material.clearcoat ); // Burley clearcoat model
	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );

#endif

#ifdef USE_SHEEN

	// No clamp: Vulkan hands sheenRoughness through raw and lets D_Charlie's
	// max(alpha, 1e-4) be the only guard. Kept identical here on purpose.
	material.sheenColor = sheenColor;
	material.sheenRoughness = sheenRoughness;

#endif

