
#ifdef USE_TRANSMISSION

	#ifdef USE_TRANSMISSIONMAP

		totalTransmission *= texture2D( transmissionMap, vUv ).r;

	#endif

	#ifdef USE_THICKNESSMAP

		thicknessFactor *= texture2D( thicknessMap, vUv ).g;

	#endif

	vec3 pos = vWorldPosition.xyz / vWorldPosition.w;
	vec3 v = normalize( cameraPosition - pos );
	vec3 viewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
	float ior = ( 1.0 + 0.4 * reflectivity ) / ( 1.0 - 0.4 * reflectivity );

	// Reuse the F0/F90 the lighting lobes already resolved rather than deriving
	// a second one here. For a plain dielectric the two are algebraically the
	// same value — threepp's reflectivity->ior mapping makes
	// ((ior-1)/(ior+1))^2 == 0.16*reflectivity^2 — so this is a no-op unless
	// KHR_materials_specular (or metalness) is in play, in which case
	// transmission should follow them. The local `ior` above stays: it drives
	// the refraction geometry, not the Fresnel.
	vec3 f0 = material.specularF0;
	vec3 f90 = vec3( material.specularF90 );

	// KHR_materials_transmission: the base colour filters what comes through, so
	// a black base colour is BLACK glass, not clear glass (glTF sample
	// SunglassesKhronos: lenses at 0.009 and 0.016, alphaMode OPAQUE). This used
	// to lerp a near-black tint to white, for assets that pair a black base
	// colour with alphaMode BLEND and a low alpha (smoked car windows) — and it
	// cleared every dark glass along with them.
	//
	// What those assets ask for is COVERAGE. The spec composite under alpha is
	//   a * [ (1-t) * diffuse + t * tint * behind ] + (1-a) * behind
	// and this pass writes alpha = 1, so fold it into the two knobs the mix
	// below consumes, exactly as the Vulkan host does:
	//   t' = 1 - a (1-t),   tint' = ( a t tint + 1-a ) / t'.
	// Only for an alpha-blended material: glTF ignores alpha in OPAQUE mode.
	vec3 transmissionAlbedo = diffuseColor.rgb;

	#ifdef TRANSMISSION_COVERAGE

		float coverage = saturate( diffuseColor.a );
		float foldedTransmission = 1.0 - coverage * ( 1.0 - totalTransmission );
		transmissionAlbedo = ( coverage * totalTransmission * transmissionAlbedo + vec3( 1.0 - coverage ) ) / max( foldedTransmission, 1e-4 );
		totalTransmission = foldedTransmission;

	#endif

	vec3 f_transmission = getIBLVolumeRefraction(
		normal, v, viewDir, roughnessFactor, transmissionAlbedo, f0, f90,
		pos, modelMatrix, viewMatrix, projectionMatrix, ior, thicknessFactor,
		attenuationColor, attenuationDistance);

	// Fresnel-weighted transmission: edges reflect more, center transmits more
	float NdotV = saturate( dot( normal, viewDir ) );
	float fresnel = f0.x + ( 1.0 - f0.x ) * pow( 1.0 - NdotV, 5.0 );
	float transmissionFactor = totalTransmission * ( 1.0 - fresnel );

	totalDiffuse = mix( totalDiffuse, f_transmission, transmissionFactor );

	// Force opaque output — transmission handles see-through, not alpha blending
	diffuseColor.a = 1.0;

#endif
