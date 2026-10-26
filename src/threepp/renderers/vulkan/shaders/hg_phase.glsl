// The Henyey-Greenstein phase function the fog family scatters with,
// normalised over the sphere (1/(4π) at g = 0): mu = cosine of the scattering
// angle, g = anisotropy. One copy for the deferred shade's medium, beams and
// water column (deferred_shade_60_fog_volumetrics.glsl includes it,
// deferred_shade_50_water_glass.glsl forward-declares it), froxel_inject.comp,
// cloud_march.comp's dual lobe and particle_light.comp. The includer defines
// PI. sky_preetham.comp keeps its own, differently evaluated Mie lobe.

#ifndef HG_PHASE_GLSL
#define HG_PHASE_GLSL

float hgPhase(float mu, float g) {
    const float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * mu, 1e-4), 1.5));
}

#endif// HG_PHASE_GLSL
