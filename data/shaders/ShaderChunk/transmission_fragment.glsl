
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

	// From https://google.github.io/filament/Filament.html#materialsystem/parameterization/remapping
	vec3 f0 = vec3( pow( ior - 1.0, 2.0 ) / pow( ior + 1.0, 2.0 ) );
	vec3 f90 = vec3( 1.0 );

	vec3 f_transmission = getIBLVolumeRefraction(
		normal, v, viewDir, roughnessFactor, diffuseColor.rgb, f0, f90,
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
