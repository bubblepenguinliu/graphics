#include "ray.h"

#include <cmath>
#include <array>

#include <Eigen/Dense>
#include <spdlog/spdlog.h>

#include "../utils/math.hpp"

using Eigen::Matrix3f;
using Eigen::Matrix4f;
using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Vector4f;
using std::numeric_limits;
using std::optional;
using std::size_t;

constexpr float infinity = 1e5f;
constexpr float eps      = 1e-5f;

Intersection::Intersection() : t(numeric_limits<float>::infinity()), face_index(0)
{
}

Ray generate_ray(int width, int height, int x, int y, Camera& camera, float depth)
{
    // 在像素中心采样，先得到成像平面上相对于中心的位置
    Vector2f pos(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
    Vector2f center(static_cast<float>(width) * 0.5f, static_cast<float>(height) * 0.5f);

    // 根据相机视场角计算归一化成像平面的高度，再换算每个像素在相机坐标系中的物理尺寸
    float    fov_y              = radians(camera.fov_y_degrees);
    float    image_plane_height = 2.0f * depth * std::tan(fov_y * 0.5f);
    float    pixel_size         = image_plane_height / static_cast<float>(height);
    float    view_x             = (pos.x() - center.x()) * pixel_size;
    float    view_y             = -(pos.y() - center.y()) * pixel_size;
    Vector4f view_pos(view_x, view_y, -depth, 1.0f);

    // 将相机坐标系下的点变换到世界坐标系，得到该像素对应的世界空间位置
    Matrix4f inv_view   = camera.view().inverse();
    Vector4f world_pos4 = inv_view * view_pos;
    Vector3f world_pos  = world_pos4.head<3>() / world_pos4.w();

    Vector3f origin    = camera.position;
    Vector3f direction = (world_pos - origin).normalized();
    return {origin, direction};
}

optional<Intersection> ray_triangle_intersect(const Ray& ray, const GL::Mesh& mesh, size_t index)
{
    // these lines below are just for compiling and can be deleted
    (void)ray;
    (void)mesh;
    (void)index;
    // these lines above are just for compiling and can be deleted
    Intersection result;

    if (result.t - infinity < -eps) {
        return result;
    } else {
        return std::nullopt;
    }
}

optional<Intersection> naive_intersect(const Ray& ray, const GL::Mesh& mesh, const Matrix4f model)
{
    Matrix4f inv_model = model.inverse();
    Vector3f local_origin =
        (inv_model * Vector4f(ray.origin.x(), ray.origin.y(), ray.origin.z(), 1.0f)).head<3>();
    Vector3f local_direction =
        (inv_model * Vector4f(ray.direction.x(), ray.direction.y(), ray.direction.z(), 0.0f))
            .head<3>();
    Matrix3f normal_matrix = inv_model.block<3, 3>(0, 0).transpose();

    auto result = naive_intersect(Ray{local_origin, local_direction}, mesh, normal_matrix);

    // 如果有相交结果，需要将局部空间的 t 值转换为世界空间的 t 值
    if (result.has_value()) {
        // 计算局部空间的交点
        Vector3f local_hit = local_origin + local_direction * result->t;
        // 将局部空间交点转换回世界空间
        Vector3f world_hit =
            (model * Vector4f(local_hit.x(), local_hit.y(), local_hit.z(), 1.0f)).head<3>();
        // 重新计算世界空间的 t 值
        result->t = (world_hit - ray.origin).norm();
    }

    return result;
}

optional<Intersection>
naive_intersect(const Ray& local_ray, const GL::Mesh& mesh, const Matrix3f& normal_matrix)
{
    Intersection result;
    result.t = infinity;

    for (size_t i = 0; i < mesh.faces.count(); ++i) {
        // 获取三角形顶点索引
        size_t idx0 = mesh.faces.data[i * 3];
        size_t idx1 = mesh.faces.data[i * 3 + 1];
        size_t idx2 = mesh.faces.data[i * 3 + 2];

        // 直接使用模型坐标系下的顶点数据
        Vector3f v0(
            mesh.vertices.data[idx0 * 3], mesh.vertices.data[idx0 * 3 + 1],
            mesh.vertices.data[idx0 * 3 + 2]
        );
        Vector3f v1(
            mesh.vertices.data[idx1 * 3], mesh.vertices.data[idx1 * 3 + 1],
            mesh.vertices.data[idx1 * 3 + 2]
        );
        Vector3f v2(
            mesh.vertices.data[idx2 * 3], mesh.vertices.data[idx2 * 3 + 1],
            mesh.vertices.data[idx2 * 3 + 2]
        );

        // Möller–Trumbore 算法 (在模型坐标系下进行)
        Vector3f E1 = v1 - v0;
        Vector3f E2 = v2 - v0;
        Vector3f S  = local_ray.origin - v0;
        Vector3f S1 = local_ray.direction.cross(E2);
        Vector3f S2 = S.cross(E1);

        float det = S1.dot(E1);
        if (std::abs(det) < eps)
            continue; // 光线平行于三角形

        float inv_det = 1.0f / det;
        float b1      = S1.dot(S) * inv_det;
        float b2      = S2.dot(local_ray.direction) * inv_det;
        float b0      = 1.0f - b1 - b2;

        // 判断交点是否在三角形内部
        if (b1 < 0.0f || b1 > 1.0f || b2 < 0.0f || b2 > 1.0f || b0 < 0.0f || b0 > 1.0f)
            continue;

        float t = S2.dot(E2) * inv_det;

        // 如果相交，且比当前记录的最近交点更近，且在射线正方向
        if (t > eps && t < result.t) {
            result.t                 = t;
            result.face_index        = i;
            result.barycentric_coord = Vector3f(b0, b1, b2);

            // 插值计算局部法线
            Vector3f n0(
                mesh.normals.data[idx0 * 3], mesh.normals.data[idx0 * 3 + 1],
                mesh.normals.data[idx0 * 3 + 2]
            );
            Vector3f n1(
                mesh.normals.data[idx1 * 3], mesh.normals.data[idx1 * 3 + 1],
                mesh.normals.data[idx1 * 3 + 2]
            );
            Vector3f n2(
                mesh.normals.data[idx2 * 3], mesh.normals.data[idx2 * 3 + 1],
                mesh.normals.data[idx2 * 3 + 2]
            );

            Vector3f local_normal = b0 * n0 + b1 * n1 + b2 * n2;
            // 将法线变换回世界坐标系并归一化
            result.normal = (normal_matrix * local_normal).normalized();
        }
    }

    // 确保结果有效
    if (result.t < infinity) {
        return result;
    }
    return std::nullopt;
}
