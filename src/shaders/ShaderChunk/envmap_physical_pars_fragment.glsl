
#if defined( USE_ENVMAP )

	#ifdef ENVMAP_MODE_REFRACTION
		uniform float refractionRatio;
	#endif

	vec3 getLightProbeIndirectIrradiance( /*const in SpecularLightProbe specularLightProbe,*/ const in GeometricContext geometry, const in int maxMIPLevel ) {

		vec3 worldNormal = inverseTransformDirection( geometry.normal, viewMatrix );

		#ifdef ENVMAP_TYPE_CUBE

			vec3 queryVec = vec3( flipEnvMap * worldNormal.x, worldNormal.yz );

			// TODO: replace with properly filtered cubemaps and access the irradiance LOD level, be it the last LOD level
			// of a specular cubemap, or just the default level of a specially created irradiance cubemap.

			#ifdef TEXTURE_LOD_EXT

				vec4 envMapColor = textureCubeLodEXT( envMap, queryVec, float( maxMIPLevel ) );

			#else

				// force the bias high to get the last LOD level as it is the most blurred.
				vec4 envMapColor = textureCube( envMap, queryVec, float( maxMIPLevel ) );

			#endif

			envMapColor.rgb = envMapTexelToLinear( envMapColor ).rgb;

		#elif defined( ENVMAP_TYPE_CUBE_UV )

			vec4 envMapColor = textureCubeUV( envMap, worldNormal, 1.0 );

		#else

			vec4 envMapColor = vec4( 0.0 );

		#endif

		// PI multiplier: the PMREM roughness-1 strip stores the NdotL-weighted
		// hemispheric MEAN, i.e. E/π (for a unit furnace env it reads exactly
		// 1.0). Downstream, RE_IndirectSpecular folds in Lambert's 1/π via
		// cosineWeightedIrradiance, so this function must return the full
		// irradiance E = π · sample for the chain to come out at ρE/π — the
		// white-furnace identity L_o = 1 for ρ = 1 (CrossRenderer_furnace_test).
		// This factor was once removed to match WGPU, but WGPU was the π-dark
		// outlier (its diffuse env lacked the π-scaling); both now carry it.
		return PI * envMapColor.rgb * envMapIntensity;

	}

	// Linear PMREM mapping (matches WGPU raster + three.js r155+ PMREMGenerator convention).
	// The legacy log/sigma heuristic over-blurred mid-roughness reflections vs the WGPU path.
	float getSpecularMIPLevel( const in float roughness, const in int maxMIPLevel ) {

		float maxMIPLevelScalar = float( maxMIPLevel );
		return clamp( roughness * maxMIPLevelScalar, 0.0, maxMIPLevelScalar );

	}

	vec3 getLightProbeIndirectRadiance( /*const in SpecularLightProbe specularLightProbe,*/ const in vec3 viewDir, const in vec3 normal, const in float roughness, const in int maxMIPLevel ) {

		#ifdef ENVMAP_MODE_REFLECTION

			// Pure mirror reflection. WGPU raster does the same; the prior
			// roughness-bend mix toward the normal made GL sample sharper env
			// regions on glossy/mid-rough surfaces, breaking parity.
			vec3 reflectVec = reflect( -viewDir, normal );

		#else

			vec3 reflectVec = refract( -viewDir, normal, refractionRatio );

		#endif

		reflectVec = inverseTransformDirection( reflectVec, viewMatrix );

		float specularMIPLevel = getSpecularMIPLevel( roughness, maxMIPLevel );

		#ifdef ENVMAP_TYPE_CUBE

			vec3 queryReflectVec = vec3( flipEnvMap * reflectVec.x, reflectVec.yz );

			#ifdef TEXTURE_LOD_EXT

				vec4 envMapColor = textureCubeLodEXT( envMap, queryReflectVec, specularMIPLevel );

			#else

				vec4 envMapColor = textureCube( envMap, queryReflectVec, specularMIPLevel );

			#endif

			envMapColor.rgb = envMapTexelToLinear( envMapColor ).rgb;

		#elif defined( ENVMAP_TYPE_CUBE_UV )

			vec4 envMapColor = textureCubeUV( envMap, reflectVec, roughness );

		#endif

		return envMapColor.rgb * envMapIntensity;

	}

#endif

