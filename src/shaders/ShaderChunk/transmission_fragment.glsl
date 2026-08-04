
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

	// Per glTF KHR_materials_transmission, transmitted light is tinted by volume
	// attenuation only. Some assets ship baseColor=(0,0,0) as "clear glass" relying
	// on alphaMode=BLEND; lerp the albedo toward white when it is near-black so the
	// refracted background still passes through.
	float albedoLum = max( max( diffuseColor.r, diffuseColor.g ), diffuseColor.b );
	vec3 transmissionAlbedo = mix( vec3( 1.0 ), diffuseColor.rgb, smoothstep( 0.0, 0.1, albedoLum ) );

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
