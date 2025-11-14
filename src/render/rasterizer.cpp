#include <array>
#include <limits>
#include <tuple>
#include <vector>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <thread>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <spdlog/spdlog.h>

#include "rasterizer.h"
#include "triangle.h"
#include "../utils/math.hpp"

using Eigen::Matrix4f;
using Eigen::Vector2i;
using Eigen::Vector3f;
using Eigen::Vector4f;
using std::fill;
using std::tuple;

void Rasterizer::worker_thread()
{
    while (!Context::rasterizer_finish) {
        // 原子地获取下一个三角形的起始顶点索引（使用全局计数器）
        size_t start_index = Context::next_vertex_to_rasterize.fetch_add(3);
        // 检查是否所有三角形都已分配完毕
        if (start_index >= Context::total_vertex_count) {
            if (Context::vertex_finish && 
                Context::processed_vertex_count.load() >= Context::total_vertex_count) {
                Context::rasterizer_finish = true;
                return;
            }
            std::this_thread::yield();
            continue;
        }
        
        while (Context::processed_vertex_count.load() < start_index + 3) {
            // 如果顶点处理完成，但计数还没到当前需要的数量，说明有问题
            if (Context::vertex_finish && 
                Context::processed_vertex_count.load() >= Context::total_vertex_count &&
                start_index + 3 > Context::total_vertex_count) {
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
        bool valid_triangle = true;
        for (size_t vertex_count = 0; vertex_count < 3; vertex_count++) {
            size_t idx = start_index + vertex_count;
            if (idx >= Context::vertex_shader_output_buffer.size()) {
                valid_triangle = false;
                break;
            }
            const VertexShaderPayload& payload = Context::vertex_shader_output_buffer[idx];
            
            // 检查顶点数据是否有效（viewport_position 不应该是默认的零向量）
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

float sign(Eigen::Vector2f p1, Eigen::Vector2f p2, Eigen::Vector2f p3)
{
    return (p1.x() - p3.x()) * (p2.y() - p3.y()) - (p2.x() - p3.x()) * (p1.y() - p3.y());
}

// 给定坐标(x,y)以及三角形的三个顶点坐标，判断(x,y)是否在三角形的内部
bool Rasterizer::inside_triangle(int x, int y, const Vector4f* vertices)
{
    Eigen::Vector2f v[3];
    for (int i = 0; i < 3; i++) v[i] = Eigen::Vector2f(vertices[i].x(), vertices[i].y());

    Eigen::Vector2f p = Eigen::Vector2f(float(x), float(y));
    if((sign(p, v[0], v[1]) >= 0 && sign(p, v[1], v[2]) >= 0 && sign(p, v[2], v[0]) >= 0)||
       (sign(p, v[0], v[1]) <= 0 && sign(p, v[1], v[2]) <= 0 && sign(p, v[2], v[0]) <= 0)) {
        return true;
    }
    return false;
}

// 给定坐标(x,y)以及三角形的三个顶点坐标，计算(x,y)对应的重心坐标[alpha, beta, gamma]
tuple<float, float, float> Rasterizer::compute_barycentric_2d(float x, float y, const Vector4f* v)
{
    float c1 = 0.f, c2 = 0.f, c3 = 0.f;
    float x1=v[0].x(), y1=v[0].y();
    float x2=v[1].x(), y2=v[1].y();
    float x3=v[2].x(), y3=v[2].y();
    // these lines below are just for compiling and can be deleted
    (void)x;
    (void)y;
    (void)v;
    // these lines above are just for compiling and can be deleted
    c1=((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3));
    c2=((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3));
    c3=1-c1-c2;
    return {c1, c2, c3};
}

// 对顶点的某一属性插值
Vector3f Rasterizer::interpolate(
    float alpha, float beta, float gamma, const Eigen::Vector3f& vert1,
    const Eigen::Vector3f& vert2, const Eigen::Vector3f& vert3, const Eigen::Vector3f& weight,
    const float& Z
)
{
    Vector3f interpolated_res;
    for (int i = 0; i < 3; i++) {
        interpolated_res[i] = alpha * vert1[i] / weight[0] + beta * vert2[i] / weight[1]
                            + gamma * vert3[i] / weight[2];
    }
    interpolated_res *= Z;
    return interpolated_res;
}

// 对当前三角形进行光栅化
void Rasterizer::rasterize_triangle(Triangle& t)
{
    
    FragmentShaderPayload payload;
    Vector3f v[3], n[3], w[3]; // w for world positions
    for (int i = 0; i < 3; i++){
        v[i]={t.viewport_pos[i].x(), t.viewport_pos[i].y(), t.viewport_pos[i].z()};
        n[i]={t.normal[i].x(), t.normal[i].y(), t.normal[i].z()};
        w[i]={t.world_pos[i].x(), t.world_pos[i].y(), t.world_pos[i].z()}; // Extract world positions
    }
    float min_x=std::min(v[0].x(), std::min(v[1].x(), v[2].x()));
    float max_x=std::max(v[0].x(), std::max(v[1].x(), v[2].x()));
    float min_y=std::min(v[0].y(), std::min(v[1].y(), v[2].y()));
    float max_y=std::max(v[0].y(), std::max(v[1].y(), v[2].y()));
    Vector3f weight{v[0].z(), v[1].z(), v[2].z()};

    // 边界裁剪：确保遍历范围在屏幕内
    int x_start = std::max(0, static_cast<int>(min_x));
    int x_end = std::min(Uniforms::width - 1, static_cast<int>(max_x));
    int y_start = std::max(0, static_cast<int>(min_y));
    int y_end = std::min(Uniforms::height - 1, static_cast<int>(max_y));

    for(int x = x_start; x <= x_end; x++){
        for(int y = y_start; y <= y_end; y++){
            if(inside_triangle(x, y, t.viewport_pos)){
              // if current pixel is in current triange:
                auto[alpha, beta, gamma] = compute_barycentric_2d(static_cast<float>(x), static_cast<float>(y), t.viewport_pos);
                 // 1. interpolate depth(use projection correction algorithm)
                 float Z = 1.0f / (alpha / v[0].z() + beta / v[1].z() + gamma / v[2].z()); 
                 // 2. interpolate world position & normal(use function:interpolate())
                auto interpolated_world_pos=interpolate(alpha, beta, gamma, w[0], w[1], w[2], weight, Z);
                auto interpolated_normal=interpolate(alpha, beta, gamma, n[0], n[1], n[2], weight, Z).normalized();
                // 3. push primitive into fragment queue
                payload.world_pos = interpolated_world_pos;
                payload.world_normal = interpolated_normal;
                payload.x = x;
                payload.y = y;
                payload.depth = Z;
                std::unique_lock<std::mutex> lock(Context::rasterizer_queue_mutex);
                Context::rasterizer_output_queue.push(payload);
            }
        }
    }
}
