#version 460
#extension GL_EXT_ray_tracing : require

// Path-traced LIDAR — miss shader.
//
// Mark the beam as a non-detection by writing instanceId = -1 and zero
// distance. The raygen converts this into a "no-hit" LidarResult.

struct Payload {
    vec3  hitPos;
    vec3  hitNormal;
    float distance;
    float intensity;
    int   instanceId;
    float continueFraction;
};
layout(location = 0) rayPayloadInEXT Payload pl;

void main() {
    pl.hitPos           = vec3(0.0);
    pl.hitNormal        = vec3(0.0);
    pl.distance         = 0.0;
    pl.intensity        = 0.0;
    pl.instanceId       = -1;
    pl.continueFraction = 0.0;
}
