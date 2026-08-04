
struct PhysicalMaterial {

	vec3 diffuseColor;
	float specularRoughness;

	// Named specularF0 rather than specularColor so the KHR_materials_specular
	// uniform of that name (meshphysical_frag, under USE_SPECULAR) stays
	// unambiguous — this field is the F0 the two are combined INTO.
	vec3 specularF0;

	// Unconditional, not #ifdef USE_SPECULAR: it is read at four call sites and
	// fencing all of them buys nothing. Defaulted to 1.0 in
	// <lights_physical_fragment>, which reproduces the old implicit F90 = 1
	// bit-for-bit.
	float specularF90;

#ifdef CLEARCOAT
	float clearcoat;
	float clearcoatRoughness;
#endif
#ifdef USE_SHEEN
	vec3 sheenColor;
	float sheenRoughness;
#endif
#ifdef USE_IRIDESCENCE
	float iridescence;
	float iridescenceIOR;
	float iridescenceThickness;// nanometres
	vec3 iridescenceFresnel;   // evaluated at dotNVi, filled in <lights_fragment_begin>
	vec3 iridescenceF0;        // the same, folded back through Schlick_to_F0
#endif

};

#define MAXIMUM_SPECULAR_COEFFICIENT 0.16
#define DEFAULT_SPECULAR_COEFFICIENT 0.04

// Clear coat directional hemishperical reflectance (this approximation should be improved)
float clearcoatDHRApprox( const in float roughness, const in float dotNL ) {

	return DEFAULT_SPECULAR_COEFFICIENT + ( 1.0 - DEFAULT_SPECULAR_COEFFICIENT ) * ( pow( 1.0 - dotNL, 5.0 ) * pow( 1.0 - roughness, 2.0 ) );

}

#if NUM_RECT_AREA_LIGHTS > 0

	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {

		vec3 normal = geometry.normal;
		vec3 viewDir = geometry.viewDir;
		vec3 position = geometry.position;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.specularRoughness;

		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight; // counterclockwise; light shines in local neg z direction
		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;

		vec2 uv = LTC_Uv( normal, viewDir, roughness );

		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );

		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);

		// LTC Fresnel Approximation by Stephen Hill
		// http://blog.selfshadow.com/publications/s2016-advances/s2016_ltc_fresnel.pdf
		// vec3( 1.0 ) was the implicit F90; specularF90 defaults to 1.0, so this
		// is unchanged for every material that does not set KHR specular.
		vec3 fresnel = ( material.specularF0 * t2.x + ( vec3( material.specularF90 ) - material.specularF0 ) * t2.y );

		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );

		reflectedLight.directDiffuse += lightColor * material.diffuseColor * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );

	}

#endif

void RE_Direct_Physical( const in IncidentLight directLight, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {

	float dotNL = saturate( dot( geometry.normal, directLight.direction ) );

	vec3 irradiance = dotNL * directLight.color;

	#ifdef USE_LEGACY_LIGHTS

		irradiance *= PI; // punctual light

	#endif

	#ifdef CLEARCOAT

		float ccDotNL = saturate( dot( geometry.clearcoatNormal, directLight.direction ) );

		vec3 ccIrradiance = ccDotNL * directLight.color;

		#ifdef USE_LEGACY_LIGHTS

			ccIrradiance *= PI; // punctual light

		#endif

		float clearcoatDHR = material.clearcoat * clearcoatDHRApprox( material.clearcoatRoughness, ccDotNL );

		reflectedLight.directSpecular += ccIrradiance * material.clearcoat * BRDF_Specular_GGX( directLight, geometry.viewDir, geometry.clearcoatNormal, vec3( DEFAULT_SPECULAR_COEFFICIENT ), 1.0, material.clearcoatRoughness );

	#else

		float clearcoatDHR = 0.0;

	#endif

	#ifdef USE_IRIDESCENCE

		// three.js parity: the thin-film Fresnel is evaluated once at the VIEW
		// angle but mixed in at the HALF vector, per light. Vulkan instead
		// substitutes a single F0 and hardcodes F90 = 1; this is the more
		// correct of the two, so the backends differ here on purpose.
		//
		// Iridescence does not apply to the clearcoat or LTC lobes.
		vec3 halfDir = normalize( directLight.direction + geometry.viewDir );
		float dotVH = saturate( dot( directLight.direction, halfDir ) );
		vec3 F = mix( F_Schlick( material.specularF0, material.specularF90, dotVH ), material.iridescenceFresnel, material.iridescence );

		reflectedLight.directSpecular += ( 1.0 - clearcoatDHR ) * irradiance * BRDF_Specular_GGX_Fresnel( directLight, geometry.viewDir, geometry.normal, F, material.specularRoughness );

	#else

		reflectedLight.directSpecular += ( 1.0 - clearcoatDHR ) * irradiance * BRDF_Specular_GGX( directLight, geometry.viewDir, geometry.normal, material.specularF0, material.specularF90, material.specularRoughness );

	#endif

	#ifdef USE_SHEEN
		// KHR_materials_sheen: the Charlie lobe sits ON TOP of the base BRDF. The
		// r119 chunk this replaces put it in an #else and swapped GGX out entirely,
		// which cost sheen materials their whole specular highlight. No albedo
		// scaling, matching the Vulkan deferred path.
		reflectedLight.directSpecular += ( 1.0 - clearcoatDHR ) * irradiance * BRDF_Specular_Sheen(
			material.sheenRoughness,
			directLight.direction,
			geometry,
			material.sheenColor
		);
	#endif

	reflectedLight.directDiffuse += ( 1.0 - clearcoatDHR ) * irradiance * BRDF_Diffuse_Lambert( material.diffuseColor );
}

void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {

	reflectedLight.indirectDiffuse += irradiance * BRDF_Diffuse_Lambert( material.diffuseColor );

}

void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {

	#ifdef CLEARCOAT

		float ccDotNV = saturate( dot( geometry.clearcoatNormal, geometry.viewDir ) );

		reflectedLight.indirectSpecular += clearcoatRadiance * material.clearcoat * BRDF_Specular_GGX_Environment( geometry.viewDir, geometry.clearcoatNormal, vec3( DEFAULT_SPECULAR_COEFFICIENT ), 1.0, material.clearcoatRoughness );

		float ccDotNL = ccDotNV;
		float clearcoatDHR = material.clearcoat * clearcoatDHRApprox( material.clearcoatRoughness, ccDotNL );

	#else

		float clearcoatDHR = 0.0;

	#endif

	float clearcoatInv = 1.0 - clearcoatDHR;

	// Both indirect specular and indirect diffuse light accumulate here.
	// Uses three.js r155+ EnvironmentBRDF (split-sum F0*brdf.x + F90*brdf.y)
	// rather than the older roughness-dependent Fresnel + multi-scattering
	// approach (both functions are gone from <bsdfs> now). That Fresnel pumped
	// grazing-angle reflection on rough surfaces (e.g. asphalt) far above what
	// looks correct.
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;

	#ifdef USE_IRIDESCENCE
		vec3 iblF0 = mix( material.specularF0, material.iridescenceF0, material.iridescence );
	#else
		vec3 iblF0 = material.specularF0;
	#endif

	vec3 envBRDF = BRDF_Specular_GGX_Environment( geometry.viewDir, geometry.normal, iblF0, material.specularF90, material.specularRoughness );

	reflectedLight.indirectSpecular += clearcoatInv * radiance * envBRDF;
	reflectedLight.indirectDiffuse += material.diffuseColor * cosineWeightedIrradiance;

	#ifdef USE_SHEEN

		// Env/IBL sheen — the grazing rim glow that carries a fabric lit only by an
		// environment. Driven by the same cosine-weighted irradiance the diffuse
		// lobe uses, as in the Vulkan gather.
		float sheenDotNV = saturate( dot( geometry.normal, geometry.viewDir ) );
		reflectedLight.indirectSpecular += material.sheenColor * IBLSheenBRDF( sheenDotNV, material.sheenRoughness ) * cosineWeightedIrradiance;

	#endif

}

#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical

// ref: https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {

	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );

}

