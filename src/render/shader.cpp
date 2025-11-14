#include "rasterizer_renderer.h"
#include "../utils/math.hpp"
#include <cstdio>
#include <algorithm>
#include <cmath>

#ifdef _WIN32
    #undef min
    #undef max
#endif

using Eigen::Vector3f;
using Eigen::Vector4f;

// vertex shader
VertexShaderPayload vertex_shader(const VertexShaderPayload& payload)
{
    VertexShaderPayload output_payload = payload;
    // Vertex position transformation
    output_payload.viewport_position = Uniforms::MVP * payload.world_position;
    // Viewport transformation
    output_payload.viewport_position.x() =
        0.5f * Uniforms::width * (output_payload.viewport_position.x() / output_payload.viewport_position.w() + 1.0f);
    output_payload.viewport_position.y() =
        0.5f * Uniforms::height * (output_payload.viewport_position.y() / output_payload.viewport_position.w() + 1.0f);
    output_payload.viewport_position.z() = output_payload.viewport_position.z() / output_payload.viewport_position.w();
    // Vertex normal transformation
    Vector4f normal_homogeneous(payload.normal.x(), payload.normal.y(), payload.normal.z(), 0.0f);
    Vector4f transformed_normal = Uniforms::inv_trans_M * normal_homogeneous;
    output_payload.normal = Vector3f(transformed_normal.x(), transformed_normal.y(), transformed_normal.z()).normalized();
    return output_payload;
}

Vector3f phong_fragment_shader(
    const FragmentShaderPayload& payload, const GL::Material& material,
    const std::list<Light>& lights, const Camera& camera
)
{
    Vector3f result = {0, 0, 0};

    // ka,kd,ks can be got from material.ambient,material.diffuse,material.specular
    Vector3f ka = material.ambient;
    Vector3f kd = material.diffuse;
    Vector3f ks = material.specular;

    // set ambient light intensity
    Vector3f ambient_light_intensity = {0.001f, 0.001f, 0.001f};
    Vector3f ambient = ka.cwiseProduct(ambient_light_intensity);
    result += ambient;
    for(auto& light : lights) {
    // Light Direction
    Vector3f light_dir = (light.position - payload.world_pos).normalized();
    // View Direction
    Vector3f view_dir  = (camera.position - payload.world_pos).normalized();
    // Half Vector
    Vector3f half_vec = (light_dir + view_dir).normalized();
    // Light Attenuation
    float distance = (light.position - payload.world_pos).norm();
    float attenuation = 1.0f / (1.0f + 0.1f * distance + 0.01f * distance * distance);
    // Ambient
    
    // Diffuse
    float diffuse_factor = std::max(0.0f, payload.world_normal.dot(light_dir));
        Vector3f diffuse = kd * diffuse_factor * light.intensity * attenuation;
    // Specular
    float specular_factor = std::pow(std::max(0.0f, payload.world_normal.dot(half_vec)), material.shininess);
    Vector3f specular = ks * specular_factor * light.intensity * attenuation;
    result +=  diffuse + specular;
    }
     // set rendering result max threshold to 255
     for(int i = 0; i < 3; i++) {
         result[i] = std::min(1.0f, result[i]);
     }
    return result * 255.f;
}
