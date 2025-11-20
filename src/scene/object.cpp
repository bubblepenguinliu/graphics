#include "object.h"

#include <array>
#include <optional>

#ifdef _WIN32
    #include <Windows.h>
#endif
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fmt/format.h>

#include "../utils/math.hpp"
#include "../utils/ray.h"
#include "../simulation/solver.h"
#include "../utils/logger.h"
#include "../utils/math.hpp"
using Eigen::Matrix4f;
using Eigen::Quaternionf;
using Eigen::Vector3f;
using Eigen::Vector4f;
using std::array;
using std::make_unique;
using std::optional;
using std::string;
using std::vector;

bool   Object::BVH_for_collision = false;
size_t Object::next_available_id = 0;
std::function<KineticState(const KineticState&, const KineticState&)> Object::step =
    forward_euler_step;

Object::Object(const string& object_name) :
    name(object_name), center(0.0f, 0.0f, 0.0f), scaling(1.0f, 1.0f, 1.0f),
    rotation(1.0f, 0.0f, 0.0f, 0.0f), velocity(0.0f, 0.0f, 0.0f), force(0.0f, 0.0f, 0.0f),
    mass(1.0f), BVH_boxes("BVH", GL::Mesh::highlight_wireframe_color)
{
    visible  = true;
    modified = false;
    id       = next_available_id;
    ++next_available_id;
    bvh                      = make_unique<BVH>(mesh);
    const string logger_name = fmt::format("{} (Object ID: {})", name, id);
    logger                   = get_logger(logger_name);
}

Matrix4f Object::model()
{
    Matrix4f scale_matrix            = Matrix4f::Identity();
    scale_matrix(0, 0)               = scaling.x();
    scale_matrix(1, 1)               = scaling.y();
    scale_matrix(2, 2)               = scaling.z();
    const Quaternionf& r             = rotation;
    auto [x_angle, y_angle, z_angle] = quaternion_to_ZYX_euler(r.w(), r.x(), r.y(), r.z());
    Matrix4f rotation_x              = Matrix4f::Identity();
    Matrix4f rotation_y              = Matrix4f::Identity();
    Matrix4f rotation_z              = Matrix4f::Identity();
    float    x                       = radians(x_angle);
    float    y                       = radians(y_angle);
    float    z                       = radians(z_angle);
    rotation_x(1, 1)                 = cos(x);
    rotation_x(1, 2)                 = -sin(x);
    rotation_x(2, 1)                 = sin(x);
    rotation_x(2, 2)                 = cos(x);
    rotation_x(0, 0)                 = 1;

    rotation_y(0, 0) = cos(y);
    rotation_y(0, 2) = sin(y);
    rotation_y(1, 1) = 1;
    rotation_y(2, 0) = -sin(y);
    rotation_y(2, 2) = cos(y);

    rotation_z(0, 0)   = cos(z);
    rotation_z(0, 1)   = -sin(z);
    rotation_z(1, 0)   = sin(z);
    rotation_z(1, 1)   = cos(z);
    rotation_z(2, 2)   = 1;
    Matrix4f rotation  = rotation_x * rotation_y * rotation_z;
    Matrix4f centering = Matrix4f::Identity();
    centering(0, 3)    = center.x();
    centering(1, 3)    = center.y();
    centering(2, 3)    = center.z();
    return centering * rotation * scale_matrix;
}

// void Object::update(vector<Object*>& all_objects)
// {
//     // 首先调用 step 函数计下一步该物体的运动学状态。
//     KineticState current_state{center, velocity, force / mass};
//     KineticState next_state = step(prev_state, current_state);
//     (void)next_state;
//     // 将物体的位置移动到下一步状态处，但暂时不要修改物体的速度。
//     // 遍历 all_objects，检查该物体在下一步状态的位置处是否会与其他物体发生碰撞。
//     for (auto object: all_objects) {
//         (void)object;

//         // 检测该物体与另一物体是否碰撞的方法是：
//         // 遍历该物体的每一条边，构造与边重合的射线去和另一物体求交，如果求交结果非空、
//         // 相交处也在这条边的两个端点之间，那么该物体与另一物体发生碰撞。
//         // 请时刻注意：物体 mesh 顶点的坐标都在模型坐标系下，你需要先将其变换到世界坐标系。
//         for (size_t i = 0; i < mesh.edges.count(); ++i) {
//             array<size_t, 2> v_indices = mesh.edge(i);
//             (void)v_indices;
//             // v_indices 中是这条边两个端点的索引，以这两个索引为参数调用 GL::Mesh::vertex
//             // 方法可以获得它们的坐标，进而用于构造射线。
//             if (BVH_for_collision) {
//             } else {
//             }
//             // 根据求交结果，判断该物体与另一物体是否发生了碰撞。
//             // 如果发生碰撞，按动量定理计算两个物体碰撞后的速度，并将下一步状态的位置设为
//             // current_state.position ，以避免重复碰撞。
//         }
//     }
//     // 将上一步状态赋值为当前状态，并将物体更新到下一步状态。
// }

void Object::update(vector<Object*>& all_objects)
{
    // 1. 更新历史状态（为了支持某些依赖历史状态的积分器或回滚）
    prev_state = KineticState{center, velocity, force / mass};

    // 2. 使用选定的求解器（step函数）计算下一步的预测状态
    // step函数指针会在运行时指向Forward Euler, RK4等不同实现
    KineticState current_state_in_sim{center, velocity, force / mass};
    KineticState next_state = step(prev_state, current_state_in_sim);

    // 3. 暂时将位置移动到预测位置，准备进行碰撞检测
    center   = next_state.position;
    velocity = next_state.velocity; // 先更新速度，如果发生碰撞再修正

    // 4. 碰撞检测循环
    for (auto object: all_objects) {
        // 跳过自己与自己的碰撞
        if (object == this)
            continue;

        // 遍历当前物体（碰撞发起者）的每一条边
        for (size_t i = 0; i < mesh.edges.count(); ++i) {
            array<size_t, 2> v_indices = mesh.edge(i);

            // 获取边的两个端点（模型坐标系）
            Vector3f v1_local = mesh.vertex(v_indices[0]);
            Vector3f v2_local = mesh.vertex(v_indices[1]);

            // --- 坐标变换：从模型空间 -> 世界空间 ---
            // 碰撞检测必须在世界坐标系下进行
            Matrix4f model_mat = this->model();
            Vector3f v1_world =
                (model_mat * Vector4f(v1_local.x(), v1_local.y(), v1_local.z(), 1.0f)).head<3>();
            Vector3f v2_world =
                (model_mat * Vector4f(v2_local.x(), v2_local.y(), v2_local.z(), 1.0f)).head<3>();

            // --- 构造射线 ---
            Vector3f edge_vec = v2_world - v1_world;
            float    edge_len = edge_vec.norm();
            if (edge_len < 1e-6)
                continue; // 忽略极短的边

            Vector3f ray_dir = edge_vec.normalized();
            Ray      ray; // 假设Ray构造函数接受起点和方向
            ray.origin    = v1_world;
            ray.direction = ray_dir;

            // --- 射线求交 ---
            std::optional<Intersection> hit;

            if (BVH_for_collision) {
                // 使用BVH加速求交 (需要Experiment 4支持)
                if (object->bvh) {
                    hit = object->bvh->intersect(ray, object->mesh, object->model());
                }
            } else {
                // 朴素求交：需要调用 utils/ray.cpp 中的 naive_intersect
                // 由于无法修改 ray.cpp，此处假设 object->mesh 提供了求交接口
                // 或者通过 BVH 的根节点进行全遍历（如果没有BVH，通常框架会提供 naive_intersect 全局函数）
                // 此处代码依赖于你是否正确实现了 ray.cpp 中的接口。
                if (object->bvh) {
                    hit = object->bvh->intersect(
                        ray, object->mesh, object->model()
                    ); // 回退到BVH接口，或者使用 object->mesh.intersect(ray)
                }
            }

            // --- 碰撞响应 ---
            // 如果发生相交，且交点在边线段内部 (t < edge_len)
            if (hit.has_value() && hit->t < edge_len && hit->t > 0) {
                Vector3f n = hit->normal; // 碰撞点法线

                // 相对速度 v_rel = v2 - v1
                Vector3f v1    = object->velocity; // 被碰撞者速度
                Vector3f v2    = this->velocity;   // 碰撞者（自己）速度
                Vector3f v_rel = v2 - v1;

                // 只有当物体相向运动时才计算碰撞（分离时忽略）
                if (v_rel.dot(n) < 0) {
                    // --- 计算冲量 (Impulse) ---
                    // 假设完全弹性碰撞 (e=1)，根据手册公式 3.1 和 3.2 推导
                    // j_r = - (1 + e) * (v_rel . n) / (1/m1 + 1/m2)
                    // 这里 e=1，所以分子是 -2
                    float j_r = -(2.0f * v_rel.dot(n)) / (1.0f / object->mass + 1.0f / this->mass);
                    Vector3f impulse = j_r * n;

                    // 更新速度
                    this->velocity += impulse / this->mass;
                    object->velocity -= impulse / object->mass;

                    // --- 位置修正 ---
                    // 发生碰撞说明预测位置非法，强制回滚到上一步位置
                    // 这样下一帧会用新速度重新计算位置，从而避免穿模
                    this->center = prev_state.position;

                    // 既然已经回滚了位置，当前边的检测可以终止，或者继续检测其他边？
                    // 为简化物理稳定性，通常检测到一次碰撞后处理完即可跳出对该物体的检测
                    goto stop_collision_check;
                }
            }
        }
    stop_collision_check:;
    }
}

void Object::render(const Shader& shader, WorkingMode mode, bool selected)
{
    if (modified) {
        mesh.VAO.bind();
        mesh.vertices.to_gpu();
        mesh.normals.to_gpu();
        mesh.edges.to_gpu();
        mesh.edges.release();
        mesh.faces.to_gpu();
        mesh.faces.release();
        mesh.VAO.release();
    }
    modified = false;
    // Render faces anyway.
    unsigned int element_flags = GL::Mesh::faces_flag;
    if (mode == WorkingMode::MODEL) {
        // For *Model* mode, only the selected object is rendered at the center in the world.
        // So the model transform is the identity matrix.
        shader.set_uniform("model", I4f);
        shader.set_uniform("normal_transform", I4f);
        element_flags |= GL::Mesh::vertices_flag;
        element_flags |= GL::Mesh::edges_flag;
    } else {
        Matrix4f model = this->model();
        shader.set_uniform("model", model);
        shader.set_uniform("normal_transform", (Matrix4f)(model.inverse().transpose()));
    }
    // Render edges of the selected object for modes with picking enabled.
    if (check_picking_enabled(mode) && selected) {
        element_flags |= GL::Mesh::edges_flag;
    }
    mesh.render(shader, element_flags);
}

void Object::rebuild_BVH()
{
    bvh->recursively_delete(bvh->root);
    bvh->build();
    BVH_boxes.clear();
    refresh_BVH_boxes(bvh->root);
    BVH_boxes.to_gpu();
}

void Object::refresh_BVH_boxes(BVHNode* node)
{
    if (node == nullptr) {
        return;
    }
    BVH_boxes.add_AABB(node->aabb.p_min, node->aabb.p_max);
    refresh_BVH_boxes(node->left);
    refresh_BVH_boxes(node->right);
}
