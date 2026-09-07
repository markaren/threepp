
#ifdef USE_TET_SKIN

	// Barycentric blend of the 4 influencing tet vertices. The soft-body mesh has an
	// identity model matrix (world-space rest positions), and the tet positions are
	// world-space, so `transformed` lands directly in world space — exactly what
	// project_vertex / worldpos_vertex expect.
	transformed = getTetPos( tetIndex.x ) * tetWeight.x
	            + getTetPos( tetIndex.y ) * tetWeight.y
	            + getTetPos( tetIndex.z ) * tetWeight.z
	            + getTetPos( tetIndex.w ) * tetWeight.w;

#endif
