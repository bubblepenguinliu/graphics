// shader.cpp
// 顶点着色器与片元着色器（Phong / Blinn-Phong）实现
// 注意：在顶点着色器中做透视除法时必须对 w 做保护，避免除以 0 导致 NaN/Inf。

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

// ---------------------------------------------------------------------------
// 顶点着色器：把 world_position 变换到裁剪空间 -> NDC -> 再做视口变换
// 输出：viewport_position (x,y in screen space, z for depth, w keep original clip.w)
// ---------------------------------------------------------------------------
VertexShaderPayload vertex_shader(const VertexShaderPayload& payload)
{
    VertexShaderPayload output_payload = payload;

    // 1) Clip space = MVP * position
    Vector4f clip_pos = Uniforms::MVP * payload.world_position;

    // 2) 保护性检查 w，避免除以 0 导致 NaN
    float w = clip_pos.w();
    if (std::abs(w) < 1e-6f) {
        // 将该顶点标记为“不可用”/已裁剪：设置 viewport_position.w() = 0 表示无效（Rasterizer 会检查）
        output_payload.viewport_position = Vector4f(0.0f, 0.0f, 0.0f, 0.0f);
        // 仍保留世界坐标与法线（若需要调试）
        output_payload.world_position = payload.world_position;
        output_payload.normal         = payload.normal;
        return output_payload;
    }

    // 3) 透视除法 -> NDC space
    Vector4f ndc;
    float    invw = 1.0f / w;
    ndc.x()       = clip_pos.x() * invw;
    ndc.y()       = clip_pos.y() * invw;
    ndc.z()       = clip_pos.z() * invw;
    ndc.w()       = 1.0f;

    // 4) 视口变换：NDC [-1,1] -> screen [0,width],[0,height]
    Vector4f view_pos;
    view_pos.x() = 0.5f * Uniforms::width * (ndc.x() + 1.0f);
    view_pos.y() = 0.5f * Uniforms::height * (ndc.y() + 1.0f);
    view_pos.z() = ndc.z();      // 存储归一化后的深度（用于插值/深度测试）
    view_pos.w() = clip_pos.w(); // 保存 clip space 的原始 w 以便之后做透视校正插值（若需要）

    output_payload.viewport_position = view_pos;

    // 法线变换（使用 model 的逆转置）
    Vector4f normal_h(payload.normal.x(), payload.normal.y(), payload.normal.z(), 0.0f);
    Vector4f transformed = Uniforms::inv_trans_M * normal_h;
    Vector3f out_normal(transformed.x(), transformed.y(), transformed.z());
    if (out_normal.norm() > 1e-6f)
        out_normal.normalize();
    output_payload.normal = out_normal;

    // 保留世界坐标给片段着色器使用
    output_payload.world_position = payload.world_position;

    return output_payload;
}

// ---------------------------------------------------------------------------
// Phong / Blinn-Phong 风格的片元着色器
// - 使用材质的 ka/kd/ks，以及场景灯光列表
// - 对每个光源计算漫反射与镜面反射，并使用距离衰减
// - 返回的颜色值范围被限制到 [0,1]，最终乘 255 返回 8-bit 色值
// ---------------------------------------------------------------------------
Eigen::Vector3f phong_fragment_shader(
    const FragmentShaderPayload& payload, const GL::Material& material,
    const std::list<Light>& lights, const Camera& camera
)
{
    Vector3f result = Vector3f::Zero();

    Vector3f ka        = material.ambient;
    Vector3f kd        = material.diffuse;
    Vector3f ks        = material.specular;
    float    shininess = material.shininess;

    // 环境光强度（较小的全局光）
    Vector3f ambient_intensity(0.001f, 0.001f, 0.001f);
    Vector3f ambient = ka.cwiseProduct(ambient_intensity);
    result += ambient;

    // 视线方向（从片元指向相机）
    Vector3f view_dir = (camera.position - payload.world_pos);
    if (view_dir.norm() > 1e-6f)
        view_dir.normalize();

    // 遍历光源
    for (const auto& light: lights) {
        Vector3f light_dir = (light.position - payload.world_pos);
        float    distance  = light_dir.norm();
        if (distance > 1e-6f)
            light_dir.normalize();
        else
            continue; // 极端保护：光源与片元重合，跳过

        // 半程向量（Blinn-Phong）
        Vector3f half_vec = (light_dir + view_dir);
        if (half_vec.norm() > 1e-6f)
            half_vec.normalize();

        // 距离衰减（一个合理的二次衰减模型）
        float attenuation = 1.0f / (1.0f + 0.1f * distance + 0.01f * distance * distance);

        // 漫反射因子
        float    diff_factor = std::max(0.0f, payload.world_normal.dot(light_dir));
        Vector3f diffuse     = kd * diff_factor * light.intensity * attenuation;

        // 镜面反射因子
        float spec_factor = std::pow(std::max(0.0f, payload.world_normal.dot(half_vec)), shininess);
        Vector3f specular = ks * spec_factor * light.intensity * attenuation;

        result += diffuse + specular;
    }

    // 限幅到 [0,1]，然后映射到 [0,255]
    for (int i = 0; i < 3; ++i) result[i] = std::min(1.0f, result[i]);
    return result * 255.0f;
}
