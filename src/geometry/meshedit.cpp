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
    (void)vertex_quadrics;
    optimal_pos = Vector3f(0.0f, 0.0f, 0.0f);
    cost        = 0.0f;
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
    // Defensive checks
    if (!e)
        return std::nullopt;
    Halfedge* h = e->halfedge;
    if (!h)
        return std::nullopt;
    Halfedge* h_inv = h->inv;
    if (!h_inv)
        return std::nullopt;

    // If either adjacent face is a boundary virtual face, flip is not defined.
    Face* f1 = h->face;
    Face* f2 = h_inv->face;
    if (!f1 || !f2)
        return std::nullopt; // safety
    if (f1->is_boundary || f2->is_boundary) {
        // flip not allowed on boundary
        logger->trace(
            "flip_edge: edge {} is on boundary (one adjacent face is virtual) - abort", e->id
        );
        return std::nullopt;
    }

    // Identify the six halfedges as in the handout:
    // h is the halfedge from v1 -> v2
    // h_inv is the opposite halfedge from v2 -> v1
    // Using local names consistent with handout/figures:
    Halfedge* h_2_3 = h->next;     // halfedge from v2 -> v3
    Halfedge* h_3_1 = h_2_3->next; // halfedge from v3 -> v1
    Halfedge* h_1_4 = h_inv->next; // halfedge from v1 -> v4 (in the other face)
    Halfedge* h_4_2 = h_1_4->next; // halfedge from v4 -> v2

    // Vertex names:
    Vertex* v1 = h->from;     // one endpoint of e
    Vertex* v2 = h_inv->from; // the other endpoint
    Vertex* v3 = h_3_1->from; // opposite vertex in face f1
    Vertex* v4 = h_4_2->from; // opposite vertex in face f2

    logger->trace("---start flipping edge {}---", e->id);
    logger->trace("(v1, v2) ({}, {})", v1->id, v2->id);
    logger->trace("(v3, v4) ({}, {})", v3->id, v4->id);

    // After flip, diagonal will be between v3 and v4.
    // We will rewire the two triangles:
    // before: (v1, v3, v2) and (v2, v4, v1)
    // after:  (v3, v4, v1) and (v4, v3, v2)  (consistent CCW orientation)

    // We'll reuse existing halfedges but reassign their neighbors, from and face pointers.
    // For clarity set up names of the 6 halfedges in final configuration:
    // face f1 new cycle: (h' = c->d, h_d_a, h_a_c)
    // Let's match reusing pointers to existing ones:
    Halfedge* h_c_d = h;     // will become halfedge from v3 -> v4 (reusing h)
    Halfedge* h_d_c = h_inv; // will become halfedge from v4 -> v3 (reusing h_inv)

    // choose other halfedges to reuse for cycles:
    // For face f1 (the one that contained h originally), we want cycle (v3 -> v4 -> v1):
    Halfedge* h_d_a =
        h_4_2; // currently v4 -> v2, we will change 'from' to v4 and set next to v_a_c
    Halfedge* h_a_c = h_3_1; // currently v3 -> v1 (good for v1->?), we'll update accordingly

    // For face f2 (the other), we want cycle (v4 -> v3 -> v2):
    Halfedge* h_c_b =
        h_2_3; // currently v2 -> v3; will become v3 -> v2 or v3->b depending on reassign
    Halfedge* h_b_d = h_1_4; // currently v1 -> v4; will become v2-related halfedge

    // To avoid confusion, we will explicitly set 'from' for reused halfedges:
    // Set h_c_d (was v1->v2) now v3->v4
    h_c_d->from = v3;
    // Set h_d_c (was v2->v1) now v4->v3
    h_d_c->from = v4;

    // Set the other 'from' as indicated by final triangles:
    h_d_a->from = v4; // will point v4 -> v1 (we will rewire next so its end is v1)
    h_a_c->from = v1; // will be v1 -> v3
    h_c_b->from = v3; // will be v3 -> v2
    h_b_d->from = v2; // will be v2 -> v4

    // Now set next/prev cycles for two faces:
    // f1 cycle: v3 -> v4 -> v1  (h_c_d -> h_d_a -> h_a_c)
    h_c_d->next = h_d_a;
    h_d_a->next = h_a_c;
    h_a_c->next = h_c_d;

    h_c_d->prev = h_a_c;
    h_d_a->prev = h_c_d;
    h_a_c->prev = h_d_a;

    // f2 cycle: v4 -> v3 -> v2  (h_d_c -> h_c_b -> h_b_d)
    h_d_c->next = h_c_b;
    h_c_b->next = h_b_d;
    h_b_d->next = h_d_c;

    h_d_c->prev = h_b_d;
    h_c_b->prev = h_d_c;
    h_b_d->prev = h_c_b;

    // Update faces on halfedges
    h_c_d->face = f1;
    h_d_a->face = f1;
    h_a_c->face = f1;

    h_d_c->face = f2;
    h_c_b->face = f2;
    h_b_d->face = f2;

    // Update representative halfedge pointers on faces
    f1->halfedge = h_c_d;
    f2->halfedge = h_d_c;

    // update edge pointers if necessary (edge objects owning halfedges)
    // e is the edge object; after flip it corresponds to halfedges between v3<->v4 so keep e->halfedge = h_c_d;
    e->halfedge = h_c_d;

    // For safety, ensure vertices' halfedge pointers still point to an outgoing halfedge
    // If any vertex's halfedge was one of modified halfedges, keep it; otherwise leave unchanged.
    v1->halfedge = h_a_c;
    v2->halfedge = h_b_d;
    v3->halfedge = h_c_d;
    v4->halfedge = h_d_c;

    logger->trace(
        "face 1 2 3: {}->{}->{}", f1->halfedge->from->id, f1->halfedge->next->from->id,
        f1->halfedge->next->next->from->id
    );
    logger->trace(
        "face 2 1 4: {}->{}->{}", f2->halfedge->from->id, f2->halfedge->next->from->id,
        f2->halfedge->next->next->from->id
    );
    logger->trace("---end---");

    global_inconsistent = true;
    return std::optional<Edge*>(e);
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
    if (!h_inv)
        return std::nullopt;

    Face* f1 = h->face;
    Face* f2 = h_inv->face;

    if (!f1 || !f2 || f1->is_boundary || f2->is_boundary)
        return std::nullopt;

    // ------------------------------
    // Extract original vertices
    // ------------------------------
    Vertex* v1 = h->from;
    Vertex* v2 = h_inv->from;

    Halfedge* h_2_3 = h->next;
    Halfedge* h_3_1 = h_2_3->next;
    Vertex*   v3    = h_3_1->from;

    Halfedge* h_1_4 = h_inv->next;
    Halfedge* h_4_2 = h_1_4->next;
    Vertex*   v4    = h_4_2->from;

    logger->trace(
        "split_edge {}: (v1,v2)=({},{}) v3={}, v4={}", e->id, v1->id, v2->id, v3->id, v4->id
    );

    // ------------------------------
    // Create new vertex at midpoint
    // ------------------------------
    Vertex* v_new = new_vertex();
    v_new->pos    = (v1->pos + v2->pos) * 0.5f;
    v_new->is_new = true;

    // ------------------------------
    // Create 3 new edges:
    // e = v1--v_new  (reuse original edge e)
    // e_nv_v2 = v_new--v2
    // e_nv_v3 = v_new--v3
    // e_nv_v4 = v_new--v4
    // ------------------------------
    Edge* e_nv_v2 = new_edge();
    Edge* e_nv_v3 = new_edge();
    Edge* e_nv_v4 = new_edge();

    // ------------------------------
    // Create 8 halfedges forming the 4 triangles
    // triangle 1: v1, v3, v_new
    Halfedge* a1 = new_halfedge(); // v1 -> v3
    Halfedge* a2 = new_halfedge(); // v3 -> v_new
    Halfedge* a3 = new_halfedge(); // v_new -> v1

    // triangle 2: v_new, v3, v2
    Halfedge* b1 = new_halfedge(); // v_new -> v3
    Halfedge* b2 = new_halfedge(); // v3 -> v2
    Halfedge* b3 = new_halfedge(); // v2 -> v_new

    // triangle 3: v1, v_new, v4
    Halfedge* c1 = new_halfedge(); // v1 -> v_new
    Halfedge* c2 = new_halfedge(); // v_new -> v4
    Halfedge* c3 = new_halfedge(); // v4 -> v1

    // triangle 4: v_new, v2, v4
    Halfedge* d1 = new_halfedge(); // v_new -> v2
    Halfedge* d2 = new_halfedge(); // v2 -> v4
    Halfedge* d3 = new_halfedge(); // v4 -> v_new

    // ------------------------------
    // Create 4 faces
    // ------------------------------
    Face* fA = new_face(false);
    Face* fB = new_face(false);
    Face* fC = new_face(false);
    Face* fD = new_face(false);

    // ------------------------------
    // Assign "from" for all halfedges
    // ------------------------------
    a1->from = v1;
    a2->from = v3;
    a3->from = v_new;
    b1->from = v_new;
    b2->from = v3;
    b3->from = v2;
    c1->from = v1;
    c2->from = v_new;
    c3->from = v4;
    d1->from = v_new;
    d2->from = v2;
    d3->from = v4;

    // ------------------------------
    // Assign edges (every edge has 2 halfedges)
    // ------------------------------

    // Edge e: v1 -- v_new
    e->halfedge = c1; // representative halfedge
    c1->edge    = e;
    a3->edge    = e; // v_new -> v1, inv of c1
    c1->inv     = a3;
    a3->inv     = c1;

    // Edge v_new -- v2
    e_nv_v2->halfedge = d1;
    d1->edge          = e_nv_v2;
    b3->edge          = e_nv_v2;
    d1->inv           = b3;
    b3->inv           = d1;

    // Edge v_new -- v3
    e_nv_v3->halfedge = a2;
    a2->edge          = e_nv_v3;
    b1->edge          = e_nv_v3;
    a2->inv           = b1;
    b1->inv           = a2;

    // Edge v_new -- v4
    e_nv_v4->halfedge = c2;
    c2->edge          = e_nv_v4;
    d3->edge          = e_nv_v4;
    c2->inv           = d3;
    d3->inv           = c2;

    // Edge v3 -- v1  (reuse original h_3_1 direction: v3->v1)
    a1->edge   = h_3_1->edge;
    a1->inv    = h_3_1;
    h_3_1->inv = a1;

    // Edge v3 -- v2  (reuse h_2_3 / h_3_1 set)
    b2->edge   = h_2_3->edge;
    b2->inv    = h_2_3;
    h_2_3->inv = b2;

    // Edge v4 -- v1  (reuse h_1_4)
    c3->edge   = h_1_4->edge;
    c3->inv    = h_1_4;
    h_1_4->inv = c3;

    // Edge v4 -- v2  (reuse h_4_2)
    d2->edge   = h_4_2->edge;
    d2->inv    = h_4_2;
    h_4_2->inv = d2;

    // ------------------------------
    // Build next/prev cycles for 4 faces
    // ------------------------------

    // Face A: v1 -> v3 -> v_new
    a1->next = a2;
    a2->next = a3;
    a3->next = a1;
    a1->prev = a3;
    a2->prev = a1;
    a3->prev = a2;
    a1->face = a2->face = a3->face = fA;
    fA->halfedge                   = a1;

    // Face B: v_new -> v3 -> v2
    b1->next = b2;
    b2->next = b3;
    b3->next = b1;
    b1->prev = b3;
    b2->prev = b1;
    b3->prev = b2;
    b1->face = b2->face = b3->face = fB;
    fB->halfedge                   = b1;

    // Face C: v1 -> v_new -> v4
    c1->next = c2;
    c2->next = c3;
    c3->next = c1;
    c1->prev = c3;
    c2->prev = c1;
    c3->prev = c2;
    c1->face = c2->face = c3->face = fC;
    fC->halfedge                   = c1;

    // Face D: v_new -> v2 -> v4
    d1->next = d2;
    d2->next = d3;
    d3->next = d1;
    d1->prev = d3;
    d2->prev = d1;
    d3->prev = d2;
    d1->face = d2->face = d3->face = fD;
    fD->halfedge                   = d1;

    // ------------------------------
    // Reset vertex halfedge pointers
    // ------------------------------
    v1->halfedge    = a1;
    v2->halfedge    = b3;
    v3->halfedge    = a2;
    v4->halfedge    = c3;
    v_new->halfedge = a3;

    // ------------------------------
    // Finally remove ALL old halfedges + faces
    // ------------------------------
    erase(h);
    erase(h_inv);
    erase(h_2_3);
    erase(h_3_1);
    erase(h_1_4);
    erase(h_4_2);
    erase(f1);
    erase(f2);

    global_inconsistent = true;
    logger->trace("split_edge {} done, new vertex {}", e->id, v_new->id);

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
    optional<HalfedgeMeshFailure> check_result = validate();
    if (check_result.has_value()) {
        return;
    }
    logger->info(
        "subdivide object {} (ID: {}) with Loop Subdivision strategy", object.name, object.id
    );
    logger->info("original mesh: {} vertices, {} faces in total", vertices.size, faces.size);
    // Each vertex and edge of the original mesh can be associated with a vertex
    // in the new (subdivided) mesh.
    // Therefore, our strategy for computing the subdivided vertex locations is to
    // *first* compute the new positions using the connectivity of the original
    // (coarse) mesh. Navigating this mesh will be much easier than navigating
    // the new subdivided (fine) mesh, which has more elements to traverse.
    // We will then assign vertex positions in the new mesh based on the values
    // we computed for the original mesh.

    // Compute new positions for all the vertices in the input mesh using
    // the Loop subdivision rule and store them in Vertex::new_pos.
    //    At this point, we also want to mark each vertex as being a vertex of the
    //    original mesh. Use Vertex::is_new for this.

    // Next, compute the subdivided vertex positions associated with edges, and
    // store them in Edge::new_pos.

    // Next, we're going to split every edge in the mesh, in any order.
    // We're also going to distinguish subdivided edges that came from splitting
    // an edge in the original mesh from new edges by setting the boolean Edge::is_new.
    // Note that in this loop, we only want to iterate over edges of the original mesh.
    // Otherwise, we'll end up splitting edges that we just split (and the
    // loop will never end!)
    // I use a vector to store iterators of original because there are three kinds of
    // edges: original edges, edges split from original edges and newly added edges.
    // The newly added edges are marked with Edge::is_new property, so there is not
    // any other property to mark the edges I just split.

    // Now flip any new edge that connects an old and new vertex.

    // Finally, copy new vertex positions into the Vertex::pos.

    // Once we have successfully subdivided the mesh, set global_inconsistent
    // to true to trigger synchronization with GL::Mesh.
    global_inconsistent = true;
    logger->info("subdivided mesh: {} vertices, {} faces in total", vertices.size, faces.size);
    logger->info("Loop Subdivision done");
    logger->info("");
    validate();
}

void HalfedgeMesh::simplify()
{
    optional<HalfedgeMeshFailure> check_result = validate();
    if (check_result.has_value()) {
        return;
    }
    logger->info("simplify object {} (ID: {})", object.name, object.id);
    logger->info("original mesh: {} vertices, {} faces", vertices.size, faces.size);
    unordered_map<Vertex*, Matrix4f> vertex_quadrics;
    unordered_map<Face*, Matrix4f>   face_quadrics;
    unordered_map<Edge*, EdgeRecord> edge_records;
    set<EdgeRecord>                  edge_queue;

    // Compute initial quadrics for each face by simply writing the plane equation
    // for the face in homogeneous coordinates. These quadrics should be stored
    // in face_quadrics

    // -> Compute an initial quadric for each vertex as the sum of the quadrics
    //    associated with the incident faces, storing it in vertex_quadrics

    // -> Build a priority queue of edges according to their quadric error cost,
    //    i.e., by building an Edge_Record for each edge and sticking it in the
    //    queue. You may want to use the above PQueue<Edge_Record> for this.

    // -> Until we reach the target edge budget, collapse the best edge. Remember
    //    to remove from the queue any edge that touches the collapsing edge
    //    BEFORE it gets collapsed, and add back into the queue any edge touching
    //    the collapsed vertex AFTER it's been collapsed. Also remember to assign
    //    a quadric to the collapsed vertex, and to pop the collapsed edge off the
    //    top of the queue.

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
