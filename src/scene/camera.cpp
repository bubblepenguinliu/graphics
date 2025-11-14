#include "camera.h"

#include <cmath>
#include <Eigen/Geometry>
#include "../utils/formatter.hpp"
#include <spdlog/spdlog.h>

#include "../utils/math.hpp"

using Eigen::Affine3f;
using Eigen::Matrix3f;
using Eigen::Matrix4f;
using Eigen::Vector3f;
using Eigen::Vector4f;

Camera::Camera(
    const Eigen::Vector3f& position, const Eigen::Vector3f& target, float near_plane,
    float far_plane, float fov_y_degrees, float aspect_ratio
) :
    position(position), target(target), near_plane(near_plane), far_plane(far_plane),
    fov_y_degrees(fov_y_degrees), aspect_ratio(aspect_ratio)
{
    world_up.x() = 0.0f;
    world_up.y() = 1.0f;
    world_up.z() = 0.0f;
}

Matrix4f Camera::view()
{
    // Compute the inverted view direction, up and right vectors related to
    // the camera itself.
    Vector3f inv_direction = (position - target).normalized();
    Vector3f right         = (world_up).cross(inv_direction).normalized();
    Vector3f up            = inv_direction.cross(right);
    // The view matrix is multiplication of a rotation and a translation
    // matrices: rotation represented by [right, up, inv_direction]^T and
    // translation represented by -position.
    Matrix4f view_matrix          = Matrix4f::Identity();
    view_matrix.block<1, 3>(0, 0) = right;
    view_matrix.block<1, 3>(1, 0) = up;
    view_matrix.block<1, 3>(2, 0) = inv_direction;
    view_matrix(0, 3)             = -right.dot(position);
    view_matrix(1, 3)             = -up.dot(position);
    view_matrix(2, 3)             = -inv_direction.dot(position);
    return view_matrix;
}

Matrix4f Camera::projection()
{
    const float fov_y  = radians(fov_y_degrees);
    const float aspect = aspect_ratio;

    // 定义视图截锥体的近平面和远平面距离（均为正值）
    const float n = near_plane; // 近平面距离 n
    const float f = far_plane;  // 远平面距离 f

    // 计算 Y 轴上的缩放因子，基于垂直视场角（FOV_y）
    // y_scale = cot(FOV_y / 2)
    const float y_scale = 1.0f / std::tan(fov_y / 2.0f);
    // 计算 X 轴上的缩放因子，基于 y_scale 和宽高比
    // x_scale = y_scale / aspect_ratio
    const float x_scale = y_scale / aspect;

    // 初始化一个 4x4 零矩阵用于构建透视投影矩阵
    Matrix4f projection = Matrix4f::Zero();

    // 设置矩阵元素：[0][0] 为 X 轴缩放因子
    projection(0, 0) = x_scale;

    // 设置矩阵元素：[1][1] 为 Y 轴缩放因子
    projection(1, 1) = y_scale;

    // 设置矩阵元素：[2][2] 用于 Z 坐标的缩放和平移，将 [n, f] 映射到 [-1, 1]（或 [0, 1] 取决于约定）
    // 主要是执行 Z 值的非线性变换：- (f + n) / (f - n)
    projection(2, 2) = -(f + n) / (f - n);

    // 设置矩阵元素：[2][3] 也是 Z 坐标变换的一部分（平移项）
    // -2 * f * n / (f - n)
    projection(2, 3) = -2.0f * f * n / (f - n);

    // 设置矩阵元素：[3][2] 用于 W 坐标，实现透视除法（将 -z 放入 w）
    projection(3, 2) = -1.0f;

    // 返回构建好的透视投影矩阵
    return projection;
}
