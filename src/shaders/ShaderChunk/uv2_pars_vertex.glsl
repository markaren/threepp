
#if defined( USE_LIGHTMAP ) || defined( USE_AOMAP )

	// The uv2 attribute itself is declared in the program prefix, since any
	// map may now read it (Texture::channel = 1).
	varying vec2 vUv2;

	uniform mat3 uv2Transform;

#endif
