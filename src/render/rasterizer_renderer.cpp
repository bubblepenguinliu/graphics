// rasterizer_renderer.cpp
// 管理整个渲染流程：设置 Uniforms、将网格顶点送入 VertexProcessor、启动线程并等待完成。
// 对比原实现：修复了 vertex 输出缓冲区的管理、线程退出条件、帧缓冲初始化与深度/颜色写入的越界检查。
// 保持原来主线逻辑（主线程遍历场景并 feed 顶点），但把内部缓冲/计数处理与标准接口对齐。

#include <cstddef>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <spdlog/spdlog.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "render_engine.h"
#include "../scene/light.h"
#include "../utils/logger.h"

// 便捷别名
using std::chrono::steady_clock;
using std::size_t;
using duration   = std::chrono::duration<float>;
using time_point = std::chrono::time_point<steady_clock, duration>;
using Eigen::Vector3f;
using Eigen::Vector4f;

// ---------------------------------------------------------------------------
// 全局 Uniforms 与初始绑定（与标准答案保持一致的外部接口）
// ---------------------------------------------------------------------------
Eigen::Matrix4f Uniforms::MVP;
Eigen::Matrix4f Uniforms::inv_trans_M;
int             Uniforms::width  = 0;
int             Uniforms::height = 0;

GL::Material     ini_material = GL::Material();
std::list<Light> ini_lights   = {};
Camera           ini_camera = Camera(Vector3f::Ones(), Vector3f::Ones(), 0.1f, 10.0f, 45.0f, 1.33f);

GL::Material&     Uniforms::material = ini_material;
std::list<Light>& Uniforms::lights   = ini_lights;
Camera&           Uniforms::camera   = ini_camera;

// ---------------------------------------------------------------------------
// Context 静态成员（共享队列、输出缓冲、计数器）
// 这些成员名与标准答案保持一致，方便外部对接。
// ---------------------------------------------------------------------------
std::mutex                        Context::vertex_queue_mutex;
std::mutex                        Context::rasterizer_queue_mutex;
std::queue<VertexShaderPayload>   Context::vertex_shader_output_queue;  // 备用队列（保留）
std::vector<VertexShaderPayload>  Context::vertex_shader_output_buffer; // 主要用于按索引访问
std::atomic<size_t>               Context::processed_vertex_count(0);
std::atomic<size_t>               Context::next_vertex_to_rasterize(0);
std::queue<FragmentShaderPayload> Context::rasterizer_output_queue;
size_t                            Context::total_vertex_count = 0;

volatile bool Context::vertex_finish     = false;
volatile bool Context::rasterizer_finish = false;
volatile bool Context::fragment_finish   = false;

FrameBuffer Context::frame_buffer(Uniforms::width, Uniforms::height);

// ---------------------------------------------------------------------------
// FrameBuffer 构造：初始化颜色与深度缓冲并初始化自旋锁（若有）
// ---------------------------------------------------------------------------
FrameBuffer::FrameBuffer(int width, int height) :
    width(width), height(height), color_buffer(width * height, Eigen::Vector3f(0, 0, 0)),
    depth_buffer(width * height, std::numeric_limits<float>::infinity()), spin_locks(width * height)
{
    // 初始化自旋锁为解锁状态（如果 FrameBuffer 使用自旋锁的话）
    for (auto& l: spin_locks) {
        // 假设 spin_locks 是一个可调用 unlock 的轻量类型
        l.unlock();
    }
}

// ---------------------------------------------------------------------------
// RasterizerRenderer 构造 - 保持原有构造器签名
// ---------------------------------------------------------------------------
RasterizerRenderer::RasterizerRenderer(
    RenderEngine& engine, int num_vertex_threads, int num_rasterizer_threads,
    int num_fragment_threads
) :
    width(engine.width), height(engine.height), n_vertex_threads(num_vertex_threads),
    n_rasterizer_threads(num_rasterizer_threads), n_fragment_threads(num_fragment_threads),
    vertex_processor(), rasterizer(), fragment_processor(), rendering_res(engine.rendering_res)
{
    logger = get_logger("Rasterizer Renderer");
}

// ---------------------------------------------------------------------------
// render(): 主渲染入口
// - 遍历 scene，将每个 object 的顶点以 index 形式喂给 VertexProcessor
// - 为每个 object 单独分配 vertex_shader_output_buffer，启动线程，等待 join
// - 最终把 FrameBuffer 的颜色数据拷贝到 rendering_res
// ---------------------------------------------------------------------------
void RasterizerRenderer::render(const Scene& scene)
{
    // 1) 更新全局画布大小与统一 Uniforms
    Uniforms::width  = static_cast<int>(width);
    Uniforms::height = static_cast<int>(height);
    Uniforms::lights = scene.lights;
    Uniforms::camera = scene.camera;

    // 2) 初始化/清空 FrameBuffer、渲染结果
    Context::frame_buffer = FrameBuffer(Uniforms::width, Uniforms::height);
    Context::frame_buffer.clear(BufferType::Color | BufferType::Depth);
    rendering_res.clear();

    // 记录渲染开始时间（仅用于统计，不影响渲染逻辑）
    time_point begin_time = steady_clock::now();

    // 为 fragment processor 设定 shader 指针（外部可替换）
    fragment_processor.fragment_shader_ptr = phong_fragment_shader;
    vertex_processor.vertex_shader_ptr     = vertex_shader;

    // 主循环：逐个物体渲染（不合并多个 object 的顶点缓冲，便于单独管理）
    for (const auto& group: scene.groups) {
        for (const auto& object: group->objects) {
            // 清空 VertexProcessor 的输入队列（防止残留）
            vertex_processor.clear_queue();
            // 清空片元队列
            {
                std::lock_guard<std::mutex> lock(Context::rasterizer_queue_mutex);
                while (!Context::rasterizer_output_queue.empty())
                    Context::rasterizer_output_queue.pop();
            }

            // 设置 MVP 与法线矩阵（模型相关）
            Eigen::Matrix4f model_matrix = object->model();
            Uniforms::MVP = Uniforms::camera.projection() * Uniforms::camera.view() * model_matrix;
            Uniforms::inv_trans_M = model_matrix.inverse().transpose();
            Uniforms::width       = static_cast<int>(this->width);
            Uniforms::height      = static_cast<int>(this->height);
            Uniforms::material    = object->mesh.material;
            Uniforms::lights      = scene.lights;
            Uniforms::camera      = scene.camera;

            // 读取 mesh 数据
            const std::vector<float>&        vertices = object->mesh.vertices.data;
            const std::vector<unsigned int>& faces    = object->mesh.faces.data;
            const std::vector<float>&        normals  = object->mesh.normals.data;

            // faces.size() 应是顶点索引数量（3 * num_triangles）
            size_t num_faces = faces.size();

            // 计算并初始化上下文相关计数与缓冲
            Context::total_vertex_count = num_faces; // 每个 face 索引即是一个顶点输入
            Context::processed_vertex_count.store(0);
            Context::next_vertex_to_rasterize.store(0);
            Context::vertex_shader_output_buffer.clear();
            Context::vertex_shader_output_buffer.resize(Context::total_vertex_count);

            // 将输出缓冲初始化为“未处理”状态（viewport_position.w() == 0 表示无效）
            for (auto& v: Context::vertex_shader_output_buffer) {
                v.viewport_position = Eigen::Vector4f(0, 0, 0, 0);
                v.world_position    = Eigen::Vector4f(0, 0, 0, 0);
                v.normal            = Eigen::Vector3f(0, 0, 0);
                v.index             = 0;
            }

            // 将顶点数据以 (position, normal, index) 喂入 VertexProcessor 输入队列
            // 注意：faces 数组里的索引用于查询 vertices/normals
            size_t index            = 0;
            size_t vertex_count_f   = vertices.size();
            size_t vertex_count_v   = vertex_count_f / 3;
            size_t max_vertex_index = (vertex_count_v > 0) ? (vertex_count_v - 1) : 0;

            // 安全的输入：对 faces 中的索引做越界检查，发现越界则记录错误并跳过当前 object
            bool abort_current_object = false;
            for (size_t i = 0; i < num_faces; i += 3) {
                for (size_t j = 0; j < 3; ++j) {
                    size_t vidx = faces[i + j];
                    if (vidx > max_vertex_index) {
                        spdlog::error(
                            "Face index {} out of bounds (max {}). Skipping object '{}'.", vidx,
                            max_vertex_index, object->name
                        );
                        abort_current_object = true;
                        break;
                    }
                    Vector4f pos(
                        vertices[3 * vidx], vertices[3 * vidx + 1], vertices[3 * vidx + 2], 1.0f
                    );
                    Vector3f nrm(normals[3 * vidx], normals[3 * vidx + 1], normals[3 * vidx + 2]);
                    // 使用标准答案兼容的输入接口：带 index 的 input_vertices
                    vertex_processor.input_vertices(pos, nrm, index++);
                }
                if (abort_current_object)
                    break;
            }
            if (abort_current_object) {
                // 跳过当前 object，继续下一个
                continue;
            }

            // 启动线程组（vertex / rasterizer / fragment）
            std::vector<std::thread> workers;
            // vertex threads
            for (int i = 0; i < n_vertex_threads; ++i) {
                workers.emplace_back(&VertexProcessor::worker_thread, &vertex_processor);
            }
            // rasterizer threads
            for (int i = 0; i < n_rasterizer_threads; ++i) {
                workers.emplace_back(&Rasterizer::worker_thread, &rasterizer);
            }
            // fragment threads
            for (int i = 0; i < n_fragment_threads; ++i) {
                workers.emplace_back(&FragmentProcessor::worker_thread, &fragment_processor);
            }

            // 等待所有线程完成（join）
            for (auto& w: workers) {
                if (w.joinable())
                    w.join();
            }
        }
    }

    // 渲染结束，统计时间
    time_point end_time           = steady_clock::now();
    duration   rendering_duration = end_time - begin_time;
    this->logger->info("rendering takes {:.6f} seconds", rendering_duration.count());

    // 将 frame_buffer 中的颜色拷贝为 unsigned char 三通道 (RGB)
    // 注意：FrameBuffer 中的 color_buffer 存储的是 float（0..255 或 0..1，按项目约定）
    size_t total_pixels = Context::frame_buffer.color_buffer.size();
    rendering_res.clear();
    rendering_res.reserve(total_pixels * 3);
    for (size_t i = 0; i < total_pixels; ++i) {
        // 做安全裁剪后转换
        float rx = std::clamp(Context::frame_buffer.color_buffer[i].x(), 0.0f, 255.0f);
        float ry = std::clamp(Context::frame_buffer.color_buffer[i].y(), 0.0f, 255.0f);
        float rz = std::clamp(Context::frame_buffer.color_buffer[i].z(), 0.0f, 255.0f);
        rendering_res.push_back(static_cast<unsigned char>(rx));
        rendering_res.push_back(static_cast<unsigned char>(ry));
        rendering_res.push_back(static_cast<unsigned char>(rz));
    }
}

// ---------------------------------------------------------------------------
// VertexProcessor::input_vertices（调整为与标准答案一致的接口）
//  - positions: world space position (vec4)
//  - normals: world space normal (vec3)
//  - index: 全局索引（用于将输出写回 vertex_shader_output_buffer 的正确位置）
// ---------------------------------------------------------------------------
void VertexProcessor::input_vertices(
    const Vector4f& positions, const Vector3f& normals, size_t index
)
{
    std::unique_lock<std::mutex> lock(queue_mutex);
    VertexShaderPayload          payload;
    payload.world_position = positions;
    payload.normal         = normals;
    payload.index          = index;
    vertex_queue.push(payload);
}

// 清空输入队列（用于在每个 object 前保证空队列）
void VertexProcessor::clear_queue()
{
    std::lock_guard<std::mutex> lock(queue_mutex);
    while (!vertex_queue.empty()) vertex_queue.pop();
}

// ---------------------------------------------------------------------------
// VertexProcessor::worker_thread
//  - 从输入队列取顶点，调用 vertex_shader（可能是函数指针），将结果放入 vertex_shader_output_buffer
//  - 使用 processed_vertex_count 来记录已完成的顶点数（用于 rasterizer 判断）
//  - 需要处理队列为空、还有待处理顶点的情形，避免死等或过早退出
// ---------------------------------------------------------------------------
void VertexProcessor::worker_thread()
{
    while (true) {
        VertexShaderPayload payload;
        bool                has_payload = false;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (!vertex_queue.empty()) {
                payload = vertex_queue.front();
                vertex_queue.pop();
                has_payload = true;
            }
        }

        if (!has_payload) {
            // 如果所有顶点都已被处理（processed_vertex_count == total），则线程可以退出
            if (Context::processed_vertex_count.load() >= Context::total_vertex_count) {
                Context::vertex_finish = true;
                return;
            }
            // 否则 yield 等待更多输入
            std::this_thread::yield();
            continue;
        }

        // 调用顶点着色器（函数指针）
        VertexShaderPayload output_payload = vertex_shader_ptr(payload);
        output_payload.index               = payload.index;

        // 边界检查：如果 index 超出输出缓冲范围，则跳过并增加 processed count，避免死锁等待
        if (output_payload.index >= Context::vertex_shader_output_buffer.size()) {
            Context::processed_vertex_count.fetch_add(1);
            continue;
        }

        // 将输出写回共享输出缓冲区（按 index 写，避免队列乱序）
        Context::vertex_shader_output_buffer[output_payload.index] = output_payload;
        Context::processed_vertex_count.fetch_add(1);
    }
}

// ---------------------------------------------------------------------------
// FragmentProcessor::worker_thread
//  - 从 Context::rasterizer_output_queue 获取片元，执行片元着色与帧缓冲写入
//  - 包含对片元坐标、帧缓冲索引的越界保护
// ---------------------------------------------------------------------------
void FragmentProcessor::worker_thread()
{
    while (!Context::fragment_finish) {
        FragmentShaderPayload fragment;
        {
            // 当 rasterizer 完成且队列为空时，结束片元线程
            if (Context::rasterizer_finish && Context::rasterizer_output_queue.empty()) {
                Context::fragment_finish = true;
                return;
            }
            // 如果队列为空，继续尝试等待
            if (Context::rasterizer_output_queue.empty()) {
                std::this_thread::yield();
                continue;
            }

            std::unique_lock<std::mutex> lock(Context::rasterizer_queue_mutex);
            if (Context::rasterizer_output_queue.empty())
                continue;
            fragment = Context::rasterizer_output_queue.front();
            Context::rasterizer_output_queue.pop();
        }

        // 越界检查：保证片元坐标落在屏幕内部
        if (fragment.x < 0 || fragment.x >= Uniforms::width || fragment.y < 0
            || fragment.y >= Uniforms::height) {
            continue;
        }

        // 将 2D 坐标 y 翻转（若坐标体系不同），此处按常见约定 (0,0) 在左上角或左下角需和 FrameBuffer 约定一致
        // 这里采用 (0,0) 在左下角形式：index = (height - 1 - y) * width + x
        int index = (Uniforms::height - 1 - fragment.y) * Uniforms::width + fragment.x;

        if (index < 0 || index >= static_cast<int>(Context::frame_buffer.depth_buffer.size())) {
            // 非法索引，跳过
            continue;
        }

        // 执行片元着色器，得到颜色
        fragment.color =
            fragment_shader_ptr(fragment, Uniforms::material, Uniforms::lights, Uniforms::camera);

        // 写入帧缓冲（FrameBuffer::set_pixel 内部应有深度测试、原子写或自旋锁）
        Context::frame_buffer.set_pixel(index, fragment.depth, fragment.color);
    }
}
