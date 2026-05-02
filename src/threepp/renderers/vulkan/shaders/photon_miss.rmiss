#version 460
#extension GL_EXT_ray_tracing : require

struct PhotonPayload {
    vec3  pos;
    vec3  normal;
    float roughness;
    float metalness;
    float transmission;
    float ior;
    uint  hitValid;
};

layout(location = 2) rayPayloadInEXT PhotonPayload pp;

void main() {
    pp.hitValid = 0u;
}
