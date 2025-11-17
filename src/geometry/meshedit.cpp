#include "halfedge.h"

#include <set>
#include <map>
#include <vector>
#include <string>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <spdlog/spdlog.h>

using Eigen::Matrix3f;
using Eigen::Matrix4f;
using Eigen::Vector3f;
using Eigen::Vector4f;
using std::optional;
using std::set;
using std::size_t;
using std::string;
using std::unordered_map;
using std::vector;

HalfedgeMesh::EdgeRecord::EdgeRecord(unordered_map<Vertex*, Matrix4f>& vertex_quadrics, Edge* e) :
    edge(e)
{
    // 确保半边和反向半边存在
    if (!e->halfedge || !e->halfedge->inv) {
        optimal_pos = Vector3f(0.0f, 0.0f, 0.0f);
        cost        = std::numeric_limits<float>::infinity();
        return;
    }

    Vertex* v1 = e->halfedge->from;
    Vertex* v2 = e->halfedge->inv->from;

    // 1. 计算合并的二次误差矩阵 K = Kv1 + Kv2 [cite: 177]
    Matrix4f K = vertex_quadrics.at(v1) + vertex_quadrics.at(v2);

    // 2. 将 K 分解为 K' (3x3), b (3x1), c (1x1)
    // K = [ K' | b ]
    //     [ b^T| c ]
    Matrix3f K_prime = K.block<3, 3>(0, 0);
    Vector3f b       = K.block<3, 1>(0, 3);

    // 3. 寻找最优位置 x_opt：解线性方程组 K' * x_opt = -b
    bool     solved = false;
    Vector4f u_opt;

    // 检查 K' 是否可逆，以求得解析最优解
    // 使用容差检查可逆性
    if (K_prime.determinant() != 0.0f) {
        // x_opt = - (K')^-1 * b
        // Eigen::Inverse 求解，得到最优 3D 坐标
        Vector3f x_opt = K_prime.inverse() * (-b);

        // 齐次坐标 u_opt = (x_opt, 1)
        u_opt       = Vector4f(x_opt.x(), x_opt.y(), x_opt.z(), 1.0f);
        optimal_pos = x_opt;
        solved      = true;
    }

    // 4. Fallback：如果 K' 不可逆，则使用边的中点作为近似最优位置
    if (!solved) {
        Vector3f midpoint = (v1->pos + v2->pos) * 0.5f;
        u_opt             = Vector4f(midpoint.x(), midpoint.y(), midpoint.z(), 1.0f);
        optimal_pos       = midpoint;
    }

    // 5. 计算代价： cost = u_opt^T * K * u_opt [cite: 176]
    cost = u_opt.transpose() * K * u_opt;
}

bool HalfedgeMesh::find_and_set_outgoing_halfedge(Vertex* v)
{
    if (!v)
        return false;
    // 如果当前指针已经合法，直接返回
    if (v->halfedge && v->halfedge->from == v)
        return true;

    // 1) 尝试在局部环上找到（通过从任何相邻半边绕圈）
    Halfedge* start = v->halfedge;
    if (start) {
        Halfedge* it = start;
        do {
            // 看看 it 的 inv 是否从 v 发出
            if (it->inv && it->inv->from == v) {
                v->halfedge = it->inv;
                return true;
            }
            // 继续沿着 v 的一圈前进
            if (it->inv && it->inv->next) {
                it = it->inv->next;
            } else {
                break;
            }
        } while (it != start && it);
    }

    // 2) 局部没找到，遍历全局半边列表找一个 from==v 的半边
    for (Halfedge* h = halfedges.head; h != nullptr; h = h->next_node) {
        if (h->from == v) {
            v->halfedge = h;
            logger->info("  Fixed vertex {} halfedge to {}", v->id, h->id);
            return true;
        }
    }

    // 无法修复
    logger->error("  Could not find valid halfedge for vertex {}", v->id);
    return false;
}

bool operator<(const HalfedgeMesh::EdgeRecord& a, const HalfedgeMesh::EdgeRecord& b)
{
    if (a.cost == b.cost) {
        // Sort by edge id if cost are the same
        return a.edge->id < b.edge->id;
    }
    return a.cost < b.cost;
}

// ------------------------ flip_edge ------------------------
optional<Edge*> HalfedgeMesh::flip_edge(Edge* e)
{
    if (!e)
        return std::nullopt;

    Halfedge* h = e->halfedge;
    if (!h)
        return std::nullopt;

    Halfedge* h_inv = h->inv;
    if (!h_inv) {
        logger->error("flip_edge: edge {} is on boundary, cannot flip", e->id);
        return std::nullopt;
    }

    Face* f1 = h->face;
    Face* f2 = h_inv->face;

    if (!f1 || !f2 || f1->is_boundary || f2->is_boundary) {
        logger->error("flip_edge: edge {} has invalid or boundary faces", e->id);
        return std::nullopt;
    }

    // 获取四个顶点
    Vertex* v1 = h->from;
    Vertex* v2 = h_inv->from;
    Vertex* v3 = h->next->from;
    Vertex* v4 = h_inv->next->from;

    if (!v1 || !v2 || !v3 || !v4) {
        logger->error("flip_edge: cannot get all vertices", e->id);
        return std::nullopt;
    }

    // 获取相邻的半边
    Halfedge* h_v1_v3 = h->next;
    Halfedge* h_v3_v2 = h_v1_v3->next;
    Halfedge* h_v2_v4 = h_inv->next;
    Halfedge* h_v4_v1 = h_v2_v4->next;

    if (!h_v1_v3 || !h_v3_v2 || !h_v2_v4 || !h_v4_v1) {
        logger->error("flip_edge: cannot get adjacent halfedges", e->id);
        return std::nullopt;
    }

    // 翻转后的边将连接 v3 和 v4
    // 原始配置：
    //     v3
    //    /  \
    //   / f1 \
    //  v1 --- v2
    //   \ f2 /
    //    \  /
    //     v4
    //
    // 翻转后：
    //     v3
    //    /  \
    //   / f1 \
    //  v4 --- v1
    //   \ f2 /
    //    \  /
    //     v2

    // 重新连接 f1: (v1, v2, v3) -> (v3, v2, v4)
    h->from = v3;
    h->next = h_v2_v4;
    h->prev = h_v4_v1;

    h_v2_v4->next = h_v4_v1;
    h_v2_v4->prev = h;

    h_v4_v1->next = h;
    h_v4_v1->prev = h_v2_v4;

    h_v1_v3->face = f2;
    h_v3_v2->face = f1;

    // 重新连接 f2: (v2, v1, v4) -> (v2, v4, v3)
    h_inv->from = v4;
    h_inv->next = h_v3_v2;
    h_inv->prev = h_v1_v3;

    h_v3_v2->next = h_v1_v3;
    h_v3_v2->prev = h_inv;

    h_v1_v3->next = h_inv;
    h_v1_v3->prev = h_v3_v2;

    f1->halfedge = h;
    f2->halfedge = h_inv;

    // ==================== 关键：更新顶点的 halfedge 指针 ====================
    // 对于每个顶点，检查其当前 halfedge 是否仍然有效
    // 如果无效，找一个新的有效半边

    auto fix_vertex_halfedge = [this](Vertex* v) {
        if (!v || !v->halfedge)
            return;

        // 检查当前半边是否从 v 出发
        if (v->halfedge->from == v) {
            return; // 有效
        }

        // 尝试找一个从 v 出发的半边
        Halfedge* start = v->halfedge;
        Halfedge* it    = start;

        do {
            if (it->inv && it->inv->from == v) {
                v->halfedge = it->inv;
                return;
            }

            // 移动到下一个相邻的半边
            if (it->inv && it->inv->next) {
                it = it->inv->next;
            } else {
                break;
            }
        } while (it != start && it);

        // 如果还是找不到，遍历所有半边
        logger->warn("Vertex {} halfedge pointer corrupted, searching all halfedges", v->id);
        for (Halfedge* h_search = halfedges.head; h_search != nullptr;
             h_search           = h_search->next_node) {
            if (h_search->from == v) {
                v->halfedge = h_search;
                logger->info("  Fixed vertex {} halfedge to {}", v->id, h_search->id);
                return;
            }
        }

        logger->error("  Could not find valid halfedge for vertex {}", v->id);
    };

    find_and_set_outgoing_halfedge(v1);
    find_and_set_outgoing_halfedge(v2);
    find_and_set_outgoing_halfedge(v3);
    find_and_set_outgoing_halfedge(v4);

    global_inconsistent = true;
    return e;
}

// ------------------------ split_edge ------------------------
optional<Vertex*> HalfedgeMesh::split_edge(Edge* e)
{
    if (!e)
        return std::nullopt;

    Halfedge* h = e->halfedge;
    if (!h)
        return std::nullopt;

    Halfedge* h_inv = h->inv;

    // 检查是否为边界边
    if (!h_inv) {
        logger->error("split_edge: edge {} is on boundary, cannot split", e->id);
        return std::nullopt;
    }

    // 获取两个相邻的面
    Face* f1 = h->face;
    Face* f2 = h_inv->face;

    if (!f1 || !f2) {
        logger->error("split_edge: edge {} has invalid faces", e->id);
        return std::nullopt;
    }

    if (f1->is_boundary || f2->is_boundary) {
        logger->error("split_edge: edge {} adjacent to boundary face", e->id);
        return std::nullopt;
    }

    // 获取四个顶点
    Vertex* v1 = h->from;
    Vertex* v2 = h_inv->from;

    if (!v1 || !v2) {
        logger->error("split_edge: edge {} has invalid vertices", e->id);
        return std::nullopt;
    }

    // 获取相邻的半边
    Halfedge* h_next     = h->next;
    Halfedge* h_prev     = h->prev;
    Halfedge* h_inv_next = h_inv->next;
    Halfedge* h_inv_prev = h_inv->prev;

    if (!h_next || !h_prev || !h_inv_next || !h_inv_prev) {
        logger->error("split_edge: edge {} has incomplete face loops", e->id);
        return std::nullopt;
    }

    // v3 是 f1 中的第三个顶点
    Vertex* v3 = h_prev->from;
    // v4 是 f2 中的第三个顶点
    Vertex* v4 = h_inv_prev->from;

    if (!v3 || !v4) {
        logger->error("split_edge: edge {} cannot find third vertices", e->id);
        return std::nullopt;
    }

    // 验证这些半边确实有反向边
    if (!h_next->inv || !h_prev->inv || !h_inv_next->inv || !h_inv_prev->inv) {
        logger->error("split_edge: edge {} adjacent halfedges missing inverses", e->id);
        return std::nullopt;
    }

    // ==================== 创建新顶点 ====================
    Vertex* v_new = new_vertex();
    v_new->pos    = (v1->pos + v2->pos) * 0.5f;
    v_new->is_new = true;

    // ==================== 创建新的半边和边 ====================
    // 边 v1-v_new
    Edge*     e1     = new_edge();
    Halfedge* h1     = new_halfedge();
    Halfedge* h1_inv = new_halfedge();

    // 边 v_new-v2
    Edge*     e2     = new_edge();
    Halfedge* h2     = new_halfedge();
    Halfedge* h2_inv = new_halfedge();

    // 边 v_new-v3
    Edge*     e3     = new_edge();
    Halfedge* h3     = new_halfedge();
    Halfedge* h3_inv = new_halfedge();

    // 边 v_new-v4
    Edge*     e4     = new_edge();
    Halfedge* h4     = new_halfedge();
    Halfedge* h4_inv = new_halfedge();

    // ==================== 创建新的面 ====================
    Face* f_new1 = new_face(false); // 分割 f1 产生的新面
    Face* f_new2 = new_face(false); // 分割 f2 产生的新面

    // ==================== 设置半边的 from 指针 ====================
    h1->from     = v1;
    h1_inv->from = v_new;
    h2->from     = v_new;
    h2_inv->from = v2;
    h3->from     = v_new;
    h3_inv->from = v3;
    h4->from     = v_new;
    h4_inv->from = v4;

    // ==================== 设置边和反向关系 ====================
    e1->halfedge = h1;
    h1->edge     = e1;
    h1->inv      = h1_inv;
    h1_inv->edge = e1;
    h1_inv->inv  = h1;

    e2->halfedge = h2;
    h2->edge     = e2;
    h2->inv      = h2_inv;
    h2_inv->edge = e2;
    h2_inv->inv  = h2;

    e3->halfedge = h3;
    h3->edge     = e3;
    h3->inv      = h3_inv;
    h3_inv->edge = e3;
    h3_inv->inv  = h3;

    e4->halfedge = h4;
    h4->edge     = e4;
    h4->inv      = h4_inv;
    h4_inv->edge = e4;
    h4_inv->inv  = h4;

    // ==================== 重新连接 f1 的三角形 ====================
    // 原始 f1: (v1, v2, v3)
    // 分割后:
    //   - 三角形 A: (v1, v_new, v3)
    //   - 三角形 B: (v_new, v2, v3)

    // 三角形 A: (v1, v_new, v3)
    h1->next     = h3;
    h3->next     = h_prev;
    h_prev->next = h1;

    h1->prev     = h_prev;
    h3->prev     = h1;
    h_prev->prev = h3;

    h1->face     = f1;
    h3->face     = f1;
    h_prev->face = f1;
    f1->halfedge = h1;

    // 三角形 B: (v_new, v2, v3)
    h2->next     = h_next;
    h_next->next = h3_inv;
    h3_inv->next = h2;

    h2->prev     = h3_inv;
    h_next->prev = h2;
    h3_inv->prev = h_next;

    h2->face         = f_new1;
    h_next->face     = f_new1;
    h3_inv->face     = f_new1;
    f_new1->halfedge = h2;

    // ==================== 重新连接 f2 的三角形 ====================
    // 原始 f2: (v2, v1, v4)
    // 分割后:
    //   - 三角形 C: (v2, v_new, v4)
    //   - 三角形 D: (v_new, v1, v4)

    // 三角形 C: (v2, v_new, v4)
    h2_inv->next     = h4;
    h4->next         = h_inv_prev;
    h_inv_prev->next = h2_inv;

    h2_inv->prev     = h_inv_prev;
    h4->prev         = h2_inv;
    h_inv_prev->prev = h4;

    h2_inv->face     = f2;
    h4->face         = f2;
    h_inv_prev->face = f2;
    f2->halfedge     = h2_inv;

    // 三角形 D: (v_new, v1, v4)
    h1_inv->next     = h_inv_next;
    h_inv_next->next = h4_inv;
    h4_inv->next     = h1_inv;

    h1_inv->prev     = h4_inv;
    h_inv_next->prev = h1_inv;
    h4_inv->prev     = h_inv_next;

    h1_inv->face     = f_new2;
    h_inv_next->face = f_new2;
    h4_inv->face     = f_new2;
    f_new2->halfedge = h1_inv;

    // ==================== 关键：更新所有顶点的 halfedge 指针 ====================
    v_new->halfedge = h1_inv;

    // 对于 v1：需要找一个从 v1 出发的有效半边
    // 优先选择新创建的半边
    v1->halfedge = h1;

    // 对于 v2：需要找一个从 v2 出发的有效半边
    v2->halfedge = h2_inv;

    // 对于 v3：需要找一个从 v3 出发的有效半边
    // h_prev 现在在三角形 A 中，从 v3 指向 v1
    // h3_inv 在三角形 B 中，从 v3 指向 v_new
    // 我们需要找一个从 v3 出发的半边
    Halfedge* h_v3_out = h_prev->inv; // 这个半边从 v3 出发
    if (h_v3_out) {
        v3->halfedge = h_v3_out;
    }

    // 对于 v4：需要找一个从 v4 出发的有效半边
    Halfedge* h_v4_out = h_inv_prev->inv; // 这个半边从 v4 出发
    if (h_v4_out) {
        v4->halfedge = h_v4_out;
    }

    v1->is_new = true;
    v2->is_new = true;
    v3->is_new = true;
    v4->is_new = true;

    v_new->halfedge =
        h1_inv; // 保守设置一个与 v_new 相关的半边（再由 find_and_set_outgoing_halfedge 校验）

    // 现在对 v1, v2, v3, v4, v_new 都做稳健修复（保证 v->halfedge->from == v）
    find_and_set_outgoing_halfedge(v1);
    find_and_set_outgoing_halfedge(v2);
    find_and_set_outgoing_halfedge(v3);
    find_and_set_outgoing_halfedge(v4);
    find_and_set_outgoing_halfedge(v_new);
    // ==================== 删除原始的半边和边 ====================
    erase(h);
    erase(h_inv);
    erase(e);

    global_inconsistent = true;
    return v_new;
}

// ------------------------ collapse_edge ------------------------
optional<Vertex*> HalfedgeMesh::collapse_edge(Edge* e)
{
    if (!e)
        return std::nullopt;
    Halfedge* h = e->halfedge;
    if (!h)
        return std::nullopt;
    Halfedge* h_inv = h->inv;
    if (!h_inv)
        return std::nullopt;

    Vertex* v1 = h->from;
    Vertex* v2 = h_inv->from;

    // If either endpoint is null, abort
    if (!v1 || !v2)
        return std::nullopt;

    // If edge is boundary or one adjacent face is boundary, handle separately as allowed by spec.
    // The handout requires collapse_edge to consider boundary case. We'll allow collapsing boundary edges,
    // but must avoid breaking manifoldness. We'll enforce the neighborhood intersection check:
    // N1(v1) ∩ N1(v2) must have size 2 (per handout)
    // Collect 1-ring neighbors of v1 and v2 (excluding virtual boundary faces as per representation)
    std::set<Vertex*> nbrs1, nbrs2;

    // traverse 1-ring for v1
    {
        Halfedge* it = v1->halfedge;
        if (it) {
            Halfedge* start = it;
            do {
                Vertex* nv = it->inv->from;
                if (nv && nv != v1)
                    nbrs1.insert(nv);
                it = it->inv->next;
            } while (it != start);
        }
    }
    // traverse 1-ring for v2
    {
        Halfedge* it = v2->halfedge;
        if (it) {
            Halfedge* start = it;
            do {
                Vertex* nv = it->inv->from;
                if (nv && nv != v2)
                    nbrs2.insert(nv);
                it = it->inv->next;
            } while (it != start);
        }
    }

    // compute intersection size
    size_t inter_count = 0;
    for (Vertex* vv: nbrs1) {
        if (nbrs2.count(vv))
            ++inter_count;
    }
    if (inter_count != 2) {
        logger->trace(
            "collapse_edge: N1 intersection size = {} != 2, abort collapse for edge {}",
            inter_count, e->id
        );
        return std::nullopt;
    }

    logger->trace("---start collapsing edge {} (v1={}, v2={})---", e->id, v1->id, v2->id);

    // We'll collapse v2 into v1 (i.e., v1 remains, v2 removed). Move all halfedges originating from v2 to originate from v1.
    // 1) Reassign 'from' of all halfedges that had v2 as from to v1
    {
        Halfedge* it = v2->halfedge;
        if (it) {
            Halfedge* start = it;
            do {
                Halfedge* next = it->inv->next; // step to next around v2
                // reassign from (except those halfedges that are part of faces that will be deleted because they're adjacent to the collapsing edge)
                // If face adjacent to this halfedge becomes degenerate (has duplicated vertex), we'll erase that face below.
                it->from = v1;
                // also update v1->halfedge to some outgoing halfedge (keep last seen)
                v1->halfedge = it;
                it           = next;
            } while (it != start);
        }
    }

    // 2) Find faces that will become degenerate (triangles containing both v1 and v2) — those adjacent to the collapsing edge will be removed.
    // Faces adjacent to h and h_inv are likely to be removed if they are triangles (per handout)
    Face* fA = h->face;
    Face* fB = h_inv->face;
    if (fA && !fA->is_boundary)
        erase(fA);
    if (fB && !fB->is_boundary)
        erase(fB);

    // 3) Erase the edge and vertex v2 (and associated halfedges)
    // First erase the halfedges on edge e
    erase(h);
    erase(h_inv);

    // Erase edge object
    erase(e);

    // Finally erase vertex v2
    erase(v2);

    // Some surrounding edges may now have halfedges whose inv/next/prev relationships are stale.
    // Handout suggests validate() call at the end; also set global_inconsistent.
    global_inconsistent = true;
    logger->trace("---end collapsing edge {} -> keep vertex {}---", e->id, v1->id);

    return std::optional<Vertex*>(v1);
}

void HalfedgeMesh::loop_subdivide()
{
    logger->info("========== LOOP SUBDIVIDE START ==========");

    // 诊断网格
    diagnose_mesh();

    // 尝试修复网格
    if (!repair_mesh()) {
        logger->error("❌ Failed to repair mesh");
        return;
    }

    optional<HalfedgeMeshFailure> check_result = validate();
    if (check_result.has_value()) {
        logger->error("❌ Validation failed after repair");
        return;
    }
    logger->info("✓ Initial validation passed");
    logger->info(
        fmt::format(
            " Original: {} vertices, {} edges, {} faces", vertices.size, edges.size, faces.size
        )
    );

    // ======================== Step 1 ========================
    logger->info("Step 1: Computing new positions for original vertices...");

    // Helper: ensure v->halfedge is set by scanning global halfedges (best-effort)
    auto ensure_vertex_halfedge = [this](Vertex* v) {
        if (!v)
            return;
        if (v->halfedge)
            return;
        for (Halfedge* hh = halfedges.head; hh != nullptr; hh = hh->next_node) {
            if (hh->from == v) {
                v->halfedge = hh;
                logger->debug("  set missing halfedge for vertex {} to {}", v->id, hh->id);
                return;
            }
        }
    };

    for (Vertex* v = vertices.head; v != nullptr; v = v->next_node) {
        v->is_new  = false;
        v->new_pos = v->pos; // default

        // Attempt to ensure v->halfedge exists for local ops
        ensure_vertex_halfedge(v);

        // -----------------------
        // Robust neighbor collection:
        // do NOT rely only on local next/inv traversal which can fail temporarily;
        // instead gather neighbors by scanning halfedges and deduplicating.
        // -----------------------
        bool                 is_boundary_vertex = false;
        std::vector<Vertex*> neighbors;
        neighbors.reserve(16);

        for (Halfedge* h = halfedges.head; h != nullptr; h = h->next_node) {
            // If this halfedge originates from v, the opposite vertex is h->inv->from (if inv exists)
            if (h->from == v) {
                if (!h->inv) {
                    is_boundary_vertex = true;
                } else {
                    Vertex* nb = h->inv->from;
                    if (nb && nb != v) {
                        if (std::find(neighbors.begin(), neighbors.end(), nb) == neighbors.end()) {
                            neighbors.push_back(nb);
                        }
                    }
                }
            }
            // Also consider halfedges whose inv originates from v (covers alternate orientation)
            if (h->inv && h->inv->from == v) {
                Vertex* nb = h->from;
                if (nb && nb != v) {
                    if (std::find(neighbors.begin(), neighbors.end(), nb) == neighbors.end()) {
                        neighbors.push_back(nb);
                    }
                }
            }
        }

        // Collect boundary-neighbors (two adjacent verts along boundary loop) if any
        std::vector<Vertex*> boundary_neighbors;
        if (is_boundary_vertex) {
            for (Halfedge* h = halfedges.head; h != nullptr; h = h->next_node) {
                if (h->from == v && h->is_boundary()) {
                    Vertex* nb = h->inv ? h->inv->from : nullptr;
                    if (nb && nb != v) {
                        if (std::find(boundary_neighbors.begin(), boundary_neighbors.end(), nb)
                            == boundary_neighbors.end()) {
                            boundary_neighbors.push_back(nb);
                        }
                    }
                }
                if (h->inv && h->inv->from == v && h->is_boundary()) {
                    Vertex* nb = h->from;
                    if (nb && nb != v) {
                        if (std::find(boundary_neighbors.begin(), boundary_neighbors.end(), nb)
                            == boundary_neighbors.end()) {
                            boundary_neighbors.push_back(nb);
                        }
                    }
                }
            }
        }

        // -----------------------
        // Fault-tolerant boundary handling:
        // If vertex is flagged boundary but has >= 3 neighbors (typical internal valence),
        // this is likely a mis-detection due to temporary inv/next inconsistency.
        // In that case, force internal smoothing (log a warning).
        // Only when boundary_neighbors has >= 2 (and neighbors.size() < 3) do we apply the boundary rule.
        // -----------------------
        if (is_boundary_vertex) {
            if (neighbors.size() >= 3) {
                logger->warn(
                    "Vertex {} incorrectly flagged as boundary but has {} neighbors. Forcing "
                    "internal smoothing.",
                    v->id, neighbors.size()
                );
                // fallthrough to internal smoothing below
            } else if (boundary_neighbors.size() >= 2) {
                // legitimate boundary with two neighbors -> apply Loop boundary rule
                Vertex* v1 = boundary_neighbors[0];
                Vertex* v2 = boundary_neighbors[1];
                if (v1 && v2) {
                    v->new_pos = 0.75f * v->pos + 0.125f * (v1->pos + v2->pos);
                    continue; // done for this vertex
                }
                // if something odd, fall through to internal smoothing
            } else {
                // boundary flagged but insufficient boundary neighbors: fall back to internal smoothing
                logger->warn(
                    "Vertex {} boundary flagged but couldn't collect two boundary neighbors; "
                    "falling back to internal smoothing.",
                    v->id
                );
            }
        }

        // -----------------------
        // Internal vertex smoothing (or fallback from ambiguous boundary case)
        // Use Loop formula with mu = 3/16 if n==3 else mu = 3/(8n)
        // -----------------------
        size_t n = neighbors.size();
        if (n == 0) {
            // degenerate: keep the original position
            v->new_pos = v->pos;
            continue;
        }

        Vector3f neighbor_sum = Vector3f::Zero();
        for (Vertex* nb: neighbors) {
            neighbor_sum += nb->pos;
        }

        float u    = (n == 3) ? (3.0f / 16.0f) : (3.0f / (8.0f * static_cast<float>(n)));
        v->new_pos = (1.0f - n * u) * v->pos + u * neighbor_sum;
    }
    logger->info("✓ Step 1 done");

    // ======================== Step 2 ========================
    logger->info("Step 2: Computing positions for edge midpoints...");

    vector<Edge*> original_edges;
    for (Edge* e = edges.head; e != nullptr; e = e->next_node) {
        original_edges.push_back(e);
    }

    for (Edge* e: original_edges) {
        if (!e)
            continue;

        Halfedge* h = e->halfedge;
        if (!h)
            continue;

        Vertex* v1 = h->from;
        Vertex* v2 = h->inv ? h->inv->from : nullptr;

        if (!v1 || !v2)
            continue;

        if (!h->inv) {
            e->new_pos = 0.5f * (v1->pos + v2->pos);
        } else {
            Halfedge* h_prev = h->prev;
            Vertex*   v3     = h_prev ? h_prev->from : nullptr;

            Halfedge* h_inv_prev = h->inv->prev;
            Vertex*   v4         = h_inv_prev ? h_inv_prev->from : nullptr;

            if (v3 && v4) {
                e->new_pos = 0.375f * (v1->pos + v2->pos) + 0.125f * (v3->pos + v4->pos);
            } else {
                e->new_pos = 0.5f * (v1->pos + v2->pos);
            }
        }
    }
    logger->info("✓ Step 2 done");

    // ======================== Step 3 ========================
    logger->info("Step 3: Splitting all original edges...");

    vector<Vertex*> new_vertices;
    int             split_count = 0;

    for (size_t i = 0; i < original_edges.size(); ++i) {
        Edge* e = original_edges[i];
        if (!e)
            continue;

        Vector3f edge_new_pos = e->new_pos;
        e->is_new             = false;

        optional<Vertex*> v_new_opt = split_edge(e);
        if (!v_new_opt.has_value()) {
            logger->warn("Failed to split edge {}", e->id);
            continue;
        }

        Vertex* v_new = v_new_opt.value();
        if (!v_new)
            continue;

        v_new->is_new = true;
        v_new->pos    = edge_new_pos;
        new_vertices.push_back(v_new);
        split_count++;
    }

    logger->info("✓ Step 3 done: {} edges split", split_count);

    // ======================== Step 4 ========================
    logger->info("Step 4: Flipping edges connecting old and new vertices...");

    vector<Edge*> edges_to_flip;
    for (Edge* e = edges.head; e != nullptr; e = e->next_node) {
        if (!e)
            continue;
        if (!e->is_new)
            continue; // flip new edges created by split
        edges_to_flip.push_back(e);
    }

    int flip_count = 0;
    for (Edge* e: edges_to_flip) {
        if (!e)
            continue;

        Halfedge* h = e->halfedge;
        if (!h || !h->inv)
            continue;

        if (h->is_boundary() || h->inv->is_boundary())
            continue;

        Vertex* v1 = h->from;
        Vertex* v2 = h->inv->from;
        if (!v1 || !v2)
            continue;

        if (v1->is_new == v2->is_new)
            continue;

        Face* f1 = h->face;
        Face* f2 = h->inv->face;
        if (!f1 || !f2 || f1->is_boundary || f2->is_boundary)
            continue;

        auto flipped = flip_edge(e);
        if (flipped.has_value()) {
            flip_count++;
        }
    }

    logger->info("✓ Step 4 done: {} edges flipped", flip_count);

    // ======================== Step 5 ========================
    logger->info("Step 5: Applying new positions to original vertices...");

    int update_count = 0;
    for (Vertex* v = vertices.head; v != nullptr; v = v->next_node) {
        if (!v->is_new && v->new_pos.allFinite()) {
            v->pos = v->new_pos;
            update_count++;
        }
    }
    logger->info("✓ Step 5 done: {} vertices updated", update_count);

    // ======================== Validation ========================
    logger->info("Final validation...");
    global_inconsistent = true;

    optional<HalfedgeMeshFailure> final_check = validate();
    if (final_check.has_value()) {
        logger->error("❌ Final validation FAILED");
        return;
    }

    logger->info("✓ Final validation passed");
    logger->info(
        fmt::format(
            " Final mesh: {} vertices, {} edges, {} faces", vertices.size, edges.size, faces.size
        )
    );

    logger->info("========== LOOP SUBDIVIDE SUCCESS ==========\n");
}

void HalfedgeMesh::simplify()
{
    optional<HalfedgeMeshFailure> check_result = validate();
    if (check_result.has_value()) {
        return;
    }
    logger->info("simplify object {} (ID: {})", object.name, object.id);
    logger->info("original mesh: {} vertices, {} faces", vertices.size, faces.size);

    // 存储面片、顶点、边的二次误差矩阵
    unordered_map<Vertex*, Matrix4f> vertex_quadrics;
    unordered_map<Face*, Matrix4f>   face_quadrics;
    unordered_map<Edge*, EdgeRecord> edge_records;
    // 使用 std::set 作为优先队列，实现动态排序和更新
    set<EdgeRecord> edge_queue;

    // ==================== Step 1: 计算面片的二次误差矩阵 (Kf) ====================
    // K_f = v * v^T, v = (N, -N . p) [cite: 175, 176]
    for (Face* f = faces.head; f != nullptr; f = f->next_node) {
        if (f->is_boundary)
            continue;

        Halfedge* h_start = f->halfedge;
        if (!h_start)
            continue;

        // 假设网格是三角形面片。获取三个顶点：v1, v2, v3
        Vertex* v1 = h_start->from;
        Vertex* v2 = h_start->next->from;
        Vertex* v3 = h_start->next->next->from;

        if (!v1 || !v2 || !v3)
            continue;

        // 计算面片法向量 N
        Vector3f v21 = v2->pos - v1->pos;
        Vector3f v31 = v3->pos - v1->pos;
        Vector3f N   = v21.cross(v31).normalized();

        // 计算齐次平面方程向量 v = (N, d)，其中 d = -N . p
        float    d = -N.dot(v1->pos);
        Vector4f v(N.x(), N.y(), N.z(), d);

        // 计算面片二次误差矩阵 Kf = v * v^T
        Matrix4f Kf      = v * v.transpose();
        face_quadrics[f] = Kf;
    }
    logger->trace("Step 1: Face quadrics computed.");

    // ==================== Step 2: 计算顶点的二次误差矩阵 (Ki) ====================
    // K_i = Sum(K_f) for incident faces [cite: 177, 180]
    for (Vertex* v = vertices.head; v != nullptr; v = v->next_node) {
        Matrix4f  Ki      = Matrix4f::Zero();
        Halfedge* h_start = v->halfedge;
        if (!h_start)
            continue;

        Halfedge* it = h_start;
        // 遍历 v 周围的一环邻域面片
        do {
            if (it->face && !it->face->is_boundary) {
                Ki += face_quadrics.at(it->face);
            }

            // 沿着顶点 v 的一圈，移动到下一个从 v 出发的半边
            if (it->inv && it->inv->next) {
                it = it->inv->next;
            } else {
                break;
            }
        } while (it != h_start);

        vertex_quadrics[v] = Ki;
    }
    logger->trace("Step 2: Vertex quadrics computed.");

    // ==================== Step 3: 构建边的优先队列  ====================
    vector<Edge*> all_edges;
    for (Edge* e = edges.head; e != nullptr; e = e->next_node) {
        // 确保边有反向半边（非边界边）
        if (e->halfedge && e->halfedge->inv) {
            all_edges.push_back(e);
        }
    }

    for (Edge* e: all_edges) {
        // EdgeRecord 构造函数会计算 K=Ki+Kj, optimal_pos, cost
        EdgeRecord record(vertex_quadrics, e);
        edge_records[e] = record;
        edge_queue.insert(record);
    }
    logger->trace("Step 3: Edge priority queue built.");

    size_t initial_faces = faces.size;
    // 目标面数：原始面数的 1/4
    size_t target_faces = (initial_faces > 4) ? (initial_faces / 4) : 1;

    logger->info(
        "Start simplification loop: initial faces={}, target faces={}", initial_faces, target_faces
    );

    // ==================== Step 4: 循环坍缩最优边 ====================
    while (faces.size > target_faces && !edge_queue.empty()) {
        // a) 从优先队列中取出代价最小的边
        EdgeRecord record = *edge_queue.begin();
        edge_queue.erase(edge_queue.begin());

        Edge* e = record.edge;
        // 必须确保边有效，且没有在之前的坍缩中被移除
        if (!e || !e->halfedge || !e->halfedge->inv) {
            continue;
        }

        Vertex*  v1    = e->halfedge->from;
        Vertex*  v2    = e->halfedge->inv->from;
        Vector3f x_opt = record.optimal_pos;

        // b) 收集 v1 和 v2 所有相邻的边（即坍缩后需要更新代价的边）
        std::set<Edge*> nbr_edges;
        auto            collect_nbr_edges = [&](Vertex* v) {
            Halfedge* it = v->halfedge;
            if (it) {
                Halfedge* start = it;
                do {
                    if (it->edge)
                        nbr_edges.insert(it->edge);
                    // 沿着顶点 v 的一圈，移动到下一个从 v 出发的半边
                    if (it->inv && it->inv->next) {
                        it = it->inv->next;
                    } else {
                        break;
                    }
                } while (it != start);
            }
        };

        collect_nbr_edges(v1);
        collect_nbr_edges(v2);

        // c) 从队列中移除所有受影响的旧 EdgeRecord
        for (Edge* nbr_e: nbr_edges) {
            if (edge_records.count(nbr_e)) {
                edge_queue.erase(edge_records.at(nbr_e)); // 使用 EdgeRecord 对象移除
                edge_records.erase(nbr_e);
            }
        }

        // d) 尝试坍缩边 (v2 坍缩到 v1)
        optional<Vertex*> v_new_opt = collapse_edge(e);
        if (!v_new_opt.has_value()) {
            logger->trace("Collapse failed for edge {}, skipping.", e->id);
            continue;
        }

        // 坍缩成功，v_new 就是保留下来的 v1 顶点
        Vertex* v_new = v_new_opt.value();

        // e) 更新保留下来的顶点 (v_new) 的位置和二次矩阵
        // K_new = K_old_v1 + K_old_v2 = K_edge
        Matrix4f K_new = vertex_quadrics.at(v1) + vertex_quadrics.at(v2);

        // 更新顶点的位置为最优位置
        v_new->pos = x_opt;

        // 更新顶点的二次矩阵
        vertex_quadrics[v_new] = K_new;
        vertex_quadrics.erase(v2);
        // 移除已删除的 v2 的二次矩阵

        // f) 重新计算并插入与 v_new 相邻的所有新边记录
        std::set<Edge*> post_collapse_nbr_edges;
        collect_nbr_edges(v_new); // 重新收集 v_new 的邻边

        for (Edge* nbr_e: post_collapse_nbr_edges) {
            // 重新计算并插入一个新的 EdgeRecord
            EdgeRecord new_record(vertex_quadrics, nbr_e);
            edge_records[nbr_e] = new_record;
            edge_queue.insert(new_record);
        }

        logger->trace("Collapsed edge {} -> {} faces remaining", e->id, faces.size);
    }

    // 清理资源
    edge_records.clear();
    edge_queue.clear();
    vertex_quadrics.clear();
    face_quadrics.clear();

    logger->info("simplified mesh: {} vertices, {} faces", vertices.size, faces.size);
    logger->info("simplification done\n");
    global_inconsistent = true;
    validate();
}

void HalfedgeMesh::isotropic_remesh()
{
    optional<HalfedgeMeshFailure> check_result = validate();
    if (check_result.has_value()) {
        return;
    }
    logger->info(
        "remesh the object {} (ID: {}) with strategy Isotropic Remeshing", object.name, object.id
    );
    logger->info("original mesh: {} vertices, {} faces", vertices.size, faces.size);
    // Compute the mean edge length.

    // Repeat the four main steps for 5 or 6 iterations
    // -> Split edges much longer than the target length (being careful about
    //    how the loop is written!)
    // -> Collapse edges much shorter than the target length.  Here we need to
    //    be EXTRA careful about advancing the loop, because many edges may have
    //    been destroyed by a collapse (which ones?)
    // -> Now flip each edge if it improves vertex degree
    // -> Finally, apply some tangential smoothing to the vertex positions
    static const size_t iteration_limit = 5;
    set<Edge*>          selected_edges;
    for (size_t i = 0; i != iteration_limit; ++i) {
        // Split long edges.

        // Collapse short edges.

        // Flip edges.

        // Vertex averaging.
    }
    logger->info("remeshed mesh: {} vertices, {} faces\n", vertices.size, faces.size);
    global_inconsistent = true;
    validate();
}
