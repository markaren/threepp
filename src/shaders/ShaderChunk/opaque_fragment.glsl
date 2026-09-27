// three.js r132 (#22424) / r186 opaque_fragment: an opaque material writes
// alpha 1, so an opacity below 1 cannot leak into the target's alpha while
// blending is off.
#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif

gl_FragColor = vec4( outgoingLight, diffuseColor.a );
