#include "rasterizer_renderer.h" // 包含你的 VertexShaderPayload, Uniforms 等定义
#include "rasterizer.h"          // 必须包含，用于定义 Rasterizer 类和其成员函数的声明
#include "triangle.h"            // 包含 Triangle 结构体定义
#include "../utils/math.hpp"
#include <cstdio>
#include <iostream>
#include <algorithm> // 修复 std::min/std::max 的重载匹配问题
#include <tuple>     // 修复 std::tuple 和结构化绑定 auto [alpha, beta, gamma] 的问题
#include <cmath>     // 用于 std::abs, std::pow
#include <mutex>     // 如果使用了 std::unique_lock
#include <thread>    // 如果使用了 std::this_thread::yield
#include <utility>   // 有时用于 std::pair 或 tuple 相关的操作

// 引入 Eigen 命名空间
using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Vector4f;

// 解决 Windows 下 min/max 宏定义冲突
#ifdef _WIN32
    #undef min
    #undef max
#endif

// =============================================================================
// III. 光栅化辅助函数 (Rasterization Helper Functions)
// -----------------------------------------------------------------------------
// 注意：以下三个函数和 Rasterizer 类必须存在，以保持与标准答案接口一致。
// =============================================================================

// 辅助函数：计算二维向量叉积的符号，用于判断点与边（有向线段）的位置关系。
float sign(Eigen::Vector2f p1, Eigen::Vector2f p2, Eigen::Vector2f p3)
{
    // (p1 - p3) x (p2 - p3) 的 Z 分量
    return (p1.x() - p3.x()) * (p2.y() - p3.y()) - (p2.x() - p3.x()) * (p1.y() - p3.y());
}

// 给定坐标(x,y)以及三角形的三个顶点坐标（屏幕空间），判断(x,y)是否在三角形的内部。
bool Rasterizer::inside_triangle(int x, int y, const Vector4f* vertices)
{
    // 转换为 2D 屏幕坐标
    Eigen::Vector2f v[3];
    for (int i = 0; i < 3; i++) v[i] = Eigen::Vector2f(vertices[i].x(), vertices[i].y());

    // 待检测点 P
    Eigen::Vector2f p = Eigen::Vector2f(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
    // NOTE: 使用像素中心点 (x+0.5, y+0.5)

    // 使用同号检测法（重心坐标符号法）判断点是否在三角形内
    float s0 = sign(p, v[0], v[1]);
    float s1 = sign(p, v[1], v[2]);
    float s2 = sign(p, v[2], v[0]);

    // 如果三个叉积结果同号（都 >= 0 或都 <= 0），则点在三角形内
    // 这里包含等于 0 的情况，处理边界
    return (s0 >= 0 && s1 >= 0 && s2 >= 0) || (s0 <= 0 && s1 <= 0 && s2 <= 0);
}

// 给定坐标(x,y)以及三角形的三个顶点坐标，计算(x,y)对应的重心坐标[alpha, beta, gamma]
std::tuple<float, float, float>
Rasterizer::compute_barycentric_2d(float x, float y, const Vector4f* v)
{
    // 这里的 x, y 是像素中心点的坐标
    float px = x + 0.5f;
    float py = y + 0.5f;

    float x1 = v[0].x(), y1 = v[0].y();
    float x2 = v[1].x(), y2 = v[1].y();
    float x3 = v[2].x(), y3 = v[2].y();

    // 分母：整个三角形的面积的两倍（可直接用叉积公式）
    float denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);

    // 计算 alpha 的分子：点 P, V2, V3 形成的子三角形面积的两倍
    float numerator_alpha = (y2 - y3) * (px - x3) + (x3 - x2) * (py - y3);
    // 计算 beta 的分子：点 P, V3, V1 形成的子三角形面积的两倍
    float numerator_beta = (y3 - y1) * (px - x3) + (x1 - x3) * (py - y3);

    // 检查分母是否接近 0，防止除法错误（退化三角形）
    if (std::abs(denominator) < 1e-6) {
        return {0.0f, 0.0f, 0.0f}; // 返回安全值
    }

    float alpha = numerator_alpha / denominator;
    float beta  = numerator_beta / denominator;
    float gamma = 1.0f - alpha - beta; // gamma = 1 - alpha - beta

    return {alpha, beta, gamma};
}

// 对顶点的某一属性插值 (具有透视矫正功能)
Vector3f Rasterizer::interpolate(
    float alpha, float beta, float gamma, const Eigen::Vector3f& vert1,
    const Eigen::Vector3f& vert2, const Eigen::Vector3f& vert3, const Eigen::Vector3f& weight,
    const float& Z
)
{
    // weight[i] 存储的是原始的 clip_position.w()，即 w_i。
    // Z 存储的是片元处的校正后的 1/W (即 1/Z_frag)，或直接是 Z_frag (根据你的实现推断)

    Vector3f interpolated_res;

    // 透视矫正插值公式：I_frag = Z_frag * [ (alpha * I1/w1) + (beta * I2/w2) + (gamma * I3/w3) ]
    for (int i = 0; i < 3; i++) {
        // vertX[i] 是待插值属性的第 i 个分量 (如 X, Y, Z 或 R, G, B)
        // weight[j] 是 w_j (顶点的 W 分量)
        interpolated_res[i] = alpha * vert1[i] / weight[0] + beta * vert2[i] / weight[1]
                            + gamma * vert3[i] / weight[2];
    }

    // 乘以 Z_frag (即 1/W_frag) 完成透视矫正
    interpolated_res *= Z;
    return interpolated_res;
}

// =============================================================================
// IV. 光栅化主函数 (Rasterizer Core Function)
// -----------------------------------------------------------------------------
// 作用：对当前三角形进行光栅化，遍历包围盒内的像素。
// =============================================================================
void Rasterizer::rasterize_triangle(Triangle& t)
{
    FragmentShaderPayload payload;
    // v: 视口坐标(x,y,z)，是 Vector3f
    // n: 法线 (Vector3f)
    // w: 世界坐标 (Vector3f)
    Vector3f v[3], n[3], w[3];

    for (int i = 0; i < 3; i++) {
        // 修正：显式构造 Vector3f，只取 Vector4f 的前三个分量 (x, y, z)
        v[i] = Vector3f(t.viewport_pos[i].x(), t.viewport_pos[i].y(), t.viewport_pos[i].z());

        n[i] = Vector3f(t.normal[i].x(), t.normal[i].y(), t.normal[i].z());

        // 假设 t.world_pos[i] 是 Vector4f:
        w[i] = Vector3f(t.world_pos[i].x(), t.world_pos[i].y(), t.world_pos[i].z());
    }

    // weight 存储的是 Clip W 分量，用于透视矫正插值公式的分母 w_i
    Vector3f weight{t.viewport_pos[0].w(), t.viewport_pos[1].w(), t.viewport_pos[2].w()};

    // 计算三角形的 AABB (Axis-Aligned Bounding Box)
    float min_x = std::min(v[0].x(), std::min(v[1].x(), v[2].x()));
    float max_x = std::max(v[0].x(), std::max(v[1].x(), v[2].x()));
    float min_y = std::min(v[0].y(), std::min(v[1].y(), v[2].y()));
    float max_y = std::max(v[0].y(), std::max(v[1].y(), v[2].y()));

    // 边界裁剪：确保遍历范围在屏幕像素边界内
    // 注意：屏幕像素坐标是 [0, width-1] 和 [0, height-1]
    int x_start = std::max(0, static_cast<int>(std::floor(min_x)));
    int x_end   = std::min(Uniforms::width - 1, static_cast<int>(std::ceil(max_x)));
    int y_start = std::max(0, static_cast<int>(std::floor(min_y)));
    int y_end   = std::min(Uniforms::height - 1, static_cast<int>(std::ceil(max_y)));

    // 遍历包围盒内的所有像素
    for (int x = x_start; x <= x_end; x++) {
        for (int y = y_start; y <= y_end; y++) {
            // 像素中心点 (x+0.5, y+0.5) 测试是否在三角形内
            if (inside_triangle(x, y, t.viewport_pos)) {

                // 1. 计算重心坐标 (alpha, beta, gamma)
                auto [alpha, beta, gamma] = compute_barycentric_2d(
                    static_cast<float>(x), static_cast<float>(y), t.viewport_pos
                );

                // 2. 透视矫正深度插值 (NDC Z)
                // Z_frag = 1.0 / ( (alpha/w1) + (beta/w2) + (gamma/w3) )
                float w_reciprocal_sum = alpha / weight[0] + beta / weight[1] + gamma / weight[2];
                // 检查除数，防止为零
                if (std::abs(w_reciprocal_sum) < 1e-6)
                    continue;

                float Z_frag = 1.0f / w_reciprocal_sum; // Z_frag 实际上是 1/W_frag

                // 3. 深度测试 (使用 NDC Z)
                // NDC Z 插值：Z_ndc = alpha*z1 + beta*z2 + gamma*z3
                float depth_ndc = alpha * v[0].z() + beta * v[1].z() + gamma * v[2].z();

                // 检查深度缓冲区，只有通过深度测试的片元才进行着色
                // 此处需要一个外部函数或机制来处理深度测试，沿用标准答案的逻辑：
                // 标准答案没有直接进行深度测试，而是把所有片元推入队列，
                // 假设深度测试在消费者线程或后续步骤中进行。

                // 4. 插值世界坐标和法线 (透视矫正插值)
                auto interpolated_world_pos =
                    interpolate(alpha, beta, gamma, w[0], w[1], w[2], weight, Z_frag);
                auto interpolated_normal =
                    interpolate(alpha, beta, gamma, n[0], n[1], n[2], weight, Z_frag).normalized();

                // 5. 组装片元着色器输入 Payload
                payload.world_pos    = interpolated_world_pos;
                payload.world_normal = interpolated_normal;
                payload.x            = x;
                payload.y            = y;
                // 传递插值后的 NDC Z，用于后续的深度测试
                payload.depth = depth_ndc;

                // 6. 推入片元队列
                // 使用互斥锁保护共享队列
                std::unique_lock<std::mutex> lock(Context::rasterizer_queue_mutex);
                Context::rasterizer_output_queue.push(payload);
            }
        }
    }
}

// IV. 线程工作函数（保持与标准答案接口一致）
// -----------------------------------------------------------------------------
// 注意：Rasterizer::worker_thread 的实现涉及 Context 类的细节，为保持完整性，
// 且不大幅修改你已有的着色器逻辑，我将沿用标准答案提供的实现。
// 由于这个函数只是流程控制，不影响你的核心图形学逻辑，故保留原样以满足接口一致性。
// （标准答案的实现已在下面，此处仅为注释说明，实际代码放在了 I/II/III 之后）
// -----------------------------------------------------------------------------

void Rasterizer::worker_thread()
{
    while (!Context::rasterizer_finish) {
        // 原子地获取下一个三角形的起始顶点索引（使用全局计数器）
        size_t start_index = Context::next_vertex_to_rasterize.fetch_add(3);
        // 检查是否所有三角形都已分配完毕
        if (start_index >= Context::total_vertex_count) {
            if (Context::vertex_finish
                && Context::processed_vertex_count.load() >= Context::total_vertex_count) {
                Context::rasterizer_finish = true;
                return;
            }
            std::this_thread::yield();
            continue;
        }

        while (Context::processed_vertex_count.load() < start_index + 3) {
            // 如果顶点处理完成，但计数还没到当前需要的数量，说明有问题
            if (Context::vertex_finish
                && Context::processed_vertex_count.load() >= Context::total_vertex_count
                && start_index + 3 > Context::total_vertex_count) {
                // 这个三角形超出了总顶点数，跳过
                break;
            }
            std::this_thread::yield();
        }

        // 再次检查：确保这3个顶点都已经处理完
        if (Context::processed_vertex_count.load() < start_index + 3) {
            continue;
        }

        if (start_index + 2 >= Context::total_vertex_count) {
            continue;
        }

        Triangle triangle;
        bool     valid_triangle = true;
        for (size_t vertex_count = 0; vertex_count < 3; vertex_count++) {
            size_t idx = start_index + vertex_count;
            if (idx >= Context::vertex_shader_output_buffer.size()) {
                valid_triangle = false;
                break;
            }
            const VertexShaderPayload& payload = Context::vertex_shader_output_buffer[idx];

            // 检查顶点数据是否有效（viewport_position 不应该是默认的零向量）
            // 沿用标准答案的检查方法：检查 w 分量是否为 0.0f
            if (payload.viewport_position.w() == 0.0f) {
                // 这个顶点可能还没被处理，跳过这个三角形
                valid_triangle = false;
                break;
            }

            triangle.viewport_pos[vertex_count] = payload.viewport_position;
            triangle.world_pos[vertex_count]    = payload.world_position;
            triangle.normal[vertex_count]       = payload.normal;
        }

        // 只有当三角形有效时才光栅化
        if (valid_triangle) {
            rasterize_triangle(triangle);
        }
    }
}
