
#ifdef ALPHATEST

	#ifdef ALPHA_TO_COVERAGE

	// three.js r161 (#22172): with alpha to coverage on, the cutout edge is a
	// one-pixel ramp that the MSAA coverage mask turns into an antialiased edge.
	diffuseColor.a = smoothstep( float( ALPHATEST ), float( ALPHATEST ) + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;

	#else

	if ( diffuseColor.a < ALPHATEST ) discard;

	#endif

#endif
