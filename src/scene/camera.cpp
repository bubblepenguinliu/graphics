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
    const float fov_y = radians(fov_y_degrees);
    const float n = near_plane;
    const float f = far_plane;
    const float a = aspect_ratio;
    const float y_scale = 1.0f / std::tan(fov_y / 2.0f);
    const float x_scale = y_scale / a;

    Matrix4f projection = Matrix4f::Zero();
    projection(0, 0) = x_scale;
    projection(1, 1) = y_scale;
    projection(2, 2) = - (f + n) / (f - n);
    projection(2, 3) = - 2.0f * f * n / (f - n);
    projection(3, 2) = -1.0f;

    // fov_y 是垂直方向的视场角（角度转弧度），决定了“看到多高”
    // n 和 f 分别是近平面和远平面的位置
    // a 是宽高比，决定了画面横向拉伸程度
    // x_scale 和 y_scale 控制投影后 x/y 方向的缩放，使视锥体正好映射到标准立方体
    // 矩阵的第三行和第四行负责把 z 坐标和透视除法处理成 OpenGL 需要的格式

    return projection;
}
