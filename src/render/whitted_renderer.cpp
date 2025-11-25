#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <vector>
#include <optional>
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "render_engine.h"
#include "../scene/light.h"
#include "../utils/math.hpp"
#include "../utils/ray.h"
#include "../utils/logger.h"

using std::chrono::steady_clock;
using duration   = std::chrono::duration<float>;
using time_point = std::chrono::time_point<steady_clock, duration>;
using Eigen::Matrix3f;
using Eigen::Matrix4f;
using Eigen::Vector3f;
using Eigen::Vector4f;

// 最大的反射次数
constexpr int   MAX_DEPTH      = 5;
constexpr float INFINITY_FLOAT = std::numeric_limits<float>::max();
// 考虑物体与光线相交点的偏移值
constexpr float EPSILON = 0.00001f;

// 当前物体的材质类型，根据不同材质类型光线会有不同的反射情况
enum class MaterialType
{
    DIFFUSE_AND_GLOSSY,
    REFLECTION
};

// 显示渲染的进度条
void update_progress(float progress)
{
    int barwidth = 70;
    std::cout << "[";
    int pos = static_cast<int>(barwidth * progress);
    for (int i = 0; i < barwidth; i++) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "]" << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

WhittedRenderer::WhittedRenderer(RenderEngine& engine) :
    width(engine.width), height(engine.height), n_threads(engine.n_threads), use_bvh(false),
    rendering_res(engine.rendering_res)
{
    logger = get_logger("Whitted Renderer");
}

// whitted-style渲染的实现
void WhittedRenderer::render(Scene& scene)
{
    time_point begin_time = steady_clock::now();
    width                 = std::floor(width);
    height                = std::floor(height);

    // initialize frame buffer
    std::vector<Vector3f> framebuffer(static_cast<size_t>(width * height));
    for (auto& v: framebuffer) {
        v = Vector3f(0.0f, 0.0f, 0.0f);
    }

    int num_threads = n_threads;
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0)
            num_threads = 1;
    }

    std::vector<std::thread> threads;
    std::atomic<int>         completed_rows(0);
    std::mutex               progress_mutex;

    auto render_rows = [&](int start_row, int end_row) {
        for (int j = start_row; j < end_row; j++) {
            for (int i = 0; i < width; i++) {
                // generate ray
                Ray ray = generate_ray(
                    static_cast<int>(width), static_cast<int>(height), i, j, scene.camera, 1.0f
                );
                // cast ray
                framebuffer[j * static_cast<int>(width) + i] = cast_ray(ray, scene, 0);
            }

            int current_completed = ++completed_rows;
            if (current_completed % (static_cast<int>(height) / 100 + 1) == 0) {
                std::lock_guard<std::mutex> lock(progress_mutex);
                update_progress(static_cast<float>(current_completed) / height);
            }
        }
    };

    int rows_per_thread = static_cast<int>(height) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start_row = t * rows_per_thread;
        int end_row = (t == num_threads - 1) ? static_cast<int>(height) : (t + 1) * rows_per_thread;
        threads.emplace_back(render_rows, start_row, end_row);
    }

    for (auto& thread: threads) {
        thread.join();
    }
    update_progress(1.0f); // Ensure 100% at the end

    static unsigned char color_res[3];
    rendering_res.clear();
    for (long unsigned int i = 0; i < framebuffer.size(); i++) {
        color_res[0] = static_cast<unsigned char>(255 * clamp(0.f, 1.f, framebuffer[i][0]));
        color_res[1] = static_cast<unsigned char>(255 * clamp(0.f, 1.f, framebuffer[i][1]));
        color_res[2] = static_cast<unsigned char>(255 * clamp(0.f, 1.f, framebuffer[i][2]));
        rendering_res.push_back(color_res[0]);
        rendering_res.push_back(color_res[1]);
        rendering_res.push_back(color_res[2]);
    }
    time_point end_time           = steady_clock::now();
    duration   rendering_duration = end_time - begin_time;
    logger->info("rendering takes {:.6f} seconds", rendering_duration.count());
}

// 菲涅尔定理计算反射光线
float WhittedRenderer::fresnel(const Vector3f& I, const Vector3f& N, const float& ior)
{
    float cosi = std::clamp(I.dot(N), -1.0f, 1.0f);
    float etai = 1, etat = ior;
    if (cosi > 0) {
        std::swap(etai, etat);
    }
    // Compute sini using Snell's law
    float sint = etai / etat * sqrtf(std::max(0.f, 1 - cosi * cosi));
    // Total internal reflection
    if (sint >= 1) {
        return 1.0f;
    } else {
        float cost = sqrtf(std::max(0.f, 1 - sint * sint));
        cosi       = std::abs(cosi);
        float Rs   = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
        float Rp   = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
        return (Rs * Rs + Rp * Rp) / 2;
    }
}

// 如果相交返回Intersection结构体，如果不相交则返回false
std::optional<std::tuple<Intersection, GL::Material>>
WhittedRenderer::trace(const Ray& ray, const Scene& scene)
{
    std::optional<Intersection> closest_intersection;
    GL::Material                closest_material;
    float                       min_t = INFINITY_FLOAT;

    for (const auto& group: scene.groups) {
        for (const auto& object: group->objects) {
            // 直接使用世界坐标系的光线与模型求交
            // naive_intersect 内部会处理坐标变换
            auto result = naive_intersect(ray, object->mesh, object->model());
            if (result.has_value()) {
                if (result->t < min_t && result->t > EPSILON) {
                    min_t                = result->t;
                    closest_intersection = result;
                    closest_material     = object->mesh.material;
                }
            }
        }
    }

    if (!closest_intersection.has_value()) {
        return std::nullopt;
    }
    return std::make_tuple(closest_intersection.value(), closest_material);
}

// Whitted-style的光线传播算法实现
Vector3f WhittedRenderer::cast_ray(const Ray& ray, const Scene& scene, int depth)
{
    if (depth > MAX_DEPTH) {
        return Vector3f(0.0f, 0.0f, 0.0f);
    }

    auto result = trace(ray, scene);
    if (!result.has_value()) {
        return RenderEngine::background_color;
    }

    auto [intersection, material] = result.value();
    Vector3f hit_point            = ray.origin + ray.direction * intersection.t;
    Vector3f N                    = intersection.normal;
    Vector3f V                    = -ray.direction;

    Vector3f hitcolor = Vector3f(0.0f, 0.0f, 0.0f);

    if (material.shininess > mirror_threshold) {
        Vector3f R = ray.direction - 2 * ray.direction.dot(N) * N;
        Vector3f reflection_origin =
            (R.dot(N) < 0) ? Vector3f(hit_point - N * EPSILON) : Vector3f(hit_point + N * EPSILON);
        float kr = fresnel(ray.direction, N, 1.5f);
        hitcolor = cast_ray(Ray{reflection_origin, R}, scene, depth + 1) * kr;
    } else {
        // Ambient
        hitcolor += material.ambient * 0.1f;

        for (const auto& light: scene.lights) {
            Vector3f L              = (light.position - hit_point).normalized();
            float    light_distance = (light.position - hit_point).norm();

            Vector3f shadow_origin = (L.dot(N) < 0) ? Vector3f(hit_point - N * EPSILON)
                                                    : Vector3f(hit_point + N * EPSILON);
            Ray      shadow_ray{shadow_origin, L};
            auto     shadow_result = trace(shadow_ray, scene);

            bool in_shadow = false;
            if (shadow_result.has_value()) {
                auto [shadow_hit, _] = shadow_result.value();
                if (shadow_hit.t < light_distance) {
                    in_shadow = true;
                }
            }

            if (!in_shadow) {
                float intensity = light.intensity;
                // Diffuse
                float diff = std::max(0.0f, N.dot(L));
                hitcolor += material.diffuse * intensity * diff;

                // Specular
                Vector3f H    = (L + V).normalized();
                float    spec = std::pow(std::max(0.0f, N.dot(H)), material.shininess);
                hitcolor += material.specular * intensity * spec;
            }
        }
    }

    return hitcolor;
}
