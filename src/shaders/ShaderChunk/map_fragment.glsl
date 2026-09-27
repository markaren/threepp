
#ifdef USE_MAP

	vec4 texelColor = texture2D( map, vMapUv );

	texelColor = mapTexelToLinear( texelColor );
	diffuseColor *= texelColor;

#endif

