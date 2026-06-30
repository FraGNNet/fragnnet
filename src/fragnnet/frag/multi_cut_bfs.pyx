# cython: profile=False
# cython: linetrace=False
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
"""Multi-bond-cut BFS fragmentation — Cython implementation.

Cython port of ``multi_cut_bfs.py``; identical public interface, faster execution.
See ``multi_cut_bfs.py`` for the full PPGB_MS2 reaction coverage table and design
rationale.

Key implementation improvements over the Python version
-------------------------------------------------------
- ``_cc_uf``: union-find CC using C stack arrays — zero heap allocation per call.
- ``bond_to_sys_mv``: ``int[::1]`` typed array for O(1) ring-system lookup vs dict.
- ``disconnected_mv``: ``char[::1]`` bit flags (avoid Python ``set`` overhead).
- Single CC call per candidate cut (Python version called ``_union_find_ccs`` twice).
- In-place mask modify/restore — no extra array copy per pair/triple.
- Pre-allocated working buffers: bytes keys ARE the mask data; masks are reconstructed
  into fixed C buffers via ``PyBytes_AS_STRING + memcpy``, eliminating all per-node
  ``malloc``/``free`` and dict storage of mask arrays.
- ``PyBytes_FromStringAndSize`` for child key creation — direct C API, no buffer
  protocol overhead from ``bytes(memoryview_slice)``.
- ``queued_set`` prevents duplicate BFS queue entries: each ``(node_key, edge_key)``
  pair is enqueued at most once, eliminating O(r²) duplicates for fused ring systems.
- ``cur_node_id`` cached once per BFS node (avoids repeated dict lookup).
- ``n_css_unique`` C-int counter replaces ``len(css_to_id_dict)`` in the hot path.
- Cut=1 replaced by Tarjan's bridge-finding algorithm: O(V+E) DFS identifies all
  bridges, then O(V+E) per bridge recovers fragment masks from DFS entry/exit times.
  Total cut=1 cost O(V+E + bridges*(V+E)) vs original O(live_bonds*(V+E)).
- ``_build_adj_c``: CSR adjacency list built in O(E) from live edges — stack only.
- ``_find_bridges_c``: iterative (stack-based) Tarjan bridge DFS — fully ``nogil``.
"""

import numpy as np

from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize
from libc.string cimport memcpy, memset
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
from rdkit import Chem

cimport numpy as cnp

MASK_DTYPE = np.uint8
MAX_NUM_NODES = 128
MAX_NUM_EDGES = 4 * MAX_NUM_NODES   # 512

# ---------------------------------------------------------------------------
# Time helper (same as compute_frags.pyx)
# ---------------------------------------------------------------------------

cdef long _get_time() nogil:
    cdef timespec ts
    clock_gettime(CLOCK_REALTIME, &ts)
    return ts.tv_sec


# ---------------------------------------------------------------------------
# Union-Find find with path-halving (operates on a C int pointer).
# ---------------------------------------------------------------------------

cdef inline int _uf_find(int* parent, int x) nogil:
    while parent[x] != x:
        parent[x] = parent[parent[x]]   # path halving
        x = parent[x]
    return x


# ---------------------------------------------------------------------------
# Zero scratch arrays before each CC call.
#
# _zero_scratch_full: memset-based full clear (used for first call per BFS node).
# _zero_scratch_live: selective clear — only live nodes/edges (subsequent calls).
# ---------------------------------------------------------------------------

cdef inline void _zero_scratch_full(
    int num_nodes,
    int num_edges,
    char[:, ::1] cc_nodes_s,
    char[:, ::1] cc_edges_s,
) nogil:
    memset(&cc_nodes_s[0, 0], 0, 2 * num_nodes)
    memset(&cc_edges_s[0, 0], 0, 2 * num_edges)


cdef inline void _zero_scratch_live(
    int n_live_nodes,
    int n_live_edges,
    int* live_node_arr,
    int* live_edge_arr,
    char[:, ::1] cc_nodes_s,
    char[:, ::1] cc_edges_s,
) nogil:
    """Reset only entries that _cc_uf can write — avoids clearing unvisited slots."""
    cdef int idx
    for idx in range(n_live_nodes):
        cc_nodes_s[0, live_node_arr[idx]] = 0
        cc_nodes_s[1, live_node_arr[idx]] = 0
    for idx in range(n_live_edges):
        cc_edges_s[0, live_edge_arr[idx]] = 0
        cc_edges_s[1, live_edge_arr[idx]] = 0


# ---------------------------------------------------------------------------
# Connected components via union-find.
#
# Writes results into pre-allocated, caller-zeroed scratch arrays.
# Returns num_ccs: 0, 1, 2, or 99 (> 2 components → bail early).
#
# Uses C stack arrays for the union-find structure — no heap allocation.
# ---------------------------------------------------------------------------

cdef int _cc_uf(
    int num_nodes,
    int num_edges,
    char[::1] node_mask,
    int[:, ::1] edges,
    char[::1] edge_mask,
    char[:, ::1] cc_nodes_out,   # (2, num_nodes) — must be zeroed by caller
    char[:, ::1] cc_edges_out,   # (2, num_edges) — must be zeroed by caller
) nogil:
    # Stack-allocated union-find arrays (MAX_NUM_NODES = 128)
    cdef int parent[128]
    # cc_root[atom_root] → CC index assigned to that root (-1 = not seen yet)
    cdef int cc_root[128]
    cdef int i, u, v, pu, pv, r, num_ccs, cc_idx

    for i in range(num_nodes):
        parent[i] = i
        cc_root[i] = -1

    # Union connected atoms via live edges
    for i in range(num_edges):
        if edge_mask[i] == 0:
            continue
        u = edges[i, 0]
        v = edges[i, 1]
        pu = _uf_find(parent, u)
        pv = _uf_find(parent, v)
        if pu != pv:
            parent[pu] = pv

    # Assign CC indices to live atoms and populate cc_nodes_out
    num_ccs = 0
    for i in range(num_nodes):
        if node_mask[i] == 0:
            continue
        r = _uf_find(parent, i)
        if cc_root[r] == -1:
            if num_ccs == 2:
                return 99   # > 2 connected components → invalid cut
            cc_root[r] = num_ccs
            num_ccs += 1
        cc_nodes_out[cc_root[r], i] = 1

    # Assign live edges to their CC and populate cc_edges_out
    for i in range(num_edges):
        if edge_mask[i] == 0:
            continue
        u = edges[i, 0]
        if node_mask[u] == 0:
            continue
        r = _uf_find(parent, u)
        cc_idx = cc_root[r]
        if cc_idx >= 0:
            cc_edges_out[cc_idx, i] = 1

    return num_ccs


# ---------------------------------------------------------------------------
# Build CSR adjacency list for live edges (used by bridge-finding).
#
# Fills ``adj_start[0..num_nodes]`` (prefix-sum of degrees) and
# ``adj_list_[0..2*num_live_edges-1]`` (edge indices per node).
# Both arrays must be pre-allocated by the caller (stack arrays in practice).
# ---------------------------------------------------------------------------

cdef void _build_adj_c(
    int num_nodes,
    int num_edges,
    int[:, ::1] edges,
    char[::1] edge_mask,
    int* adj_start,
    int* adj_list_,
) nogil:
    """Build CSR adjacency list from live edges (edge_mask == 1).

    Args:
        num_nodes: Number of nodes (atoms).
        num_edges: Number of bond slots to scan.
        edges: Shape ``(MAX_NUM_EDGES, 2)`` — bond endpoints.
        edge_mask: Shape ``(num_edges,)`` — 1 for live bonds.
        adj_start: Output prefix-sum array, size ``num_nodes + 1``.
        adj_list_: Output edge-index list, size ``2 * num_edges``.
    """
    cdef int i, u, v
    cdef int degree[128]

    for i in range(num_nodes):
        degree[i] = 0

    for i in range(num_edges):
        if edge_mask[i] == 0:
            continue
        u = edges[i, 0]
        v = edges[i, 1]
        degree[u] += 1
        degree[v] += 1

    adj_start[0] = 0
    for i in range(num_nodes):
        adj_start[i + 1] = adj_start[i] + degree[i]
        degree[i] = 0

    for i in range(num_edges):
        if edge_mask[i] == 0:
            continue
        u = edges[i, 0]
        v = edges[i, 1]
        adj_list_[adj_start[u] + degree[u]] = i
        degree[u] += 1
        adj_list_[adj_start[v] + degree[v]] = i
        degree[v] += 1


# ---------------------------------------------------------------------------
# Tarjan's bridge algorithm — iterative, stack-based, fully nogil.
#
# Returns the number of bridges found.  For each bridge i:
#   ``bridge_bonds[i]`` — edge index of the bridge bond.
#   ``bridge_child[i]`` — the child-side node (subtree root in DFS tree).
#   ``entry_t[v]``      — DFS entry (discovery) time for each node v.
#   ``exit_t[v]``       — DFS exit (finish) time for each node v.
#
# Membership in the child-side CC for bridge i:
#   node k is in child side  iff  entry_t[child] <= entry_t[k] <= exit_t[k] <= exit_t[child]
#   node k is in parent side iff  entry_t[k] < entry_t[child]  or  exit_t[k] > exit_t[child]
#
# This O(V+E) pass replaces O(live_bonds) calls to _cc_uf for cut=1.
# ---------------------------------------------------------------------------

cdef int _find_bridges_c(
    int num_nodes,
    int num_edges,
    int n_live_nodes,
    int* live_node_arr,
    int[:, ::1] edges,
    int* adj_start,    # CSR from _build_adj_c
    int* adj_list_,    # CSR from _build_adj_c
    int* bridge_bonds,
    int* bridge_child,
    int* entry_t,
    int* exit_t,
) nogil:
    """Iterative Tarjan bridge-finding DFS.

    Args:
        num_nodes: Total node count (for initialisation).
        num_edges: Total edge count (unused here; kept for symmetry).
        n_live_nodes: Number of live (active) nodes.
        live_node_arr: C array of live node indices; DFS starts from index 0.
        edges: Bond endpoints, shape ``(MAX_NUM_EDGES, 2)``.
        adj_start: CSR adjacency prefix-sum (size ``num_nodes + 1``).
        adj_list_: CSR adjacency edge list (size ``2 * num_live_edges``).
        bridge_bonds: Output — bridge edge indices.
        bridge_child: Output — child-side node for each bridge.
        entry_t: Output — DFS entry time per node (-1 if unvisited).
        exit_t: Output — DFS exit time per node (-1 if unvisited).

    Returns:
        Number of bridges found.
    """
    cdef int disc[128]
    cdef int low[128]
    cdef int par_node_[128]
    cdef int par_edge_[128]
    cdef int stk_node[128]
    cdef int stk_cursor[128]
    cdef int stk_top = 0
    cdef int timer = 0
    cdef int n_bridges = 0
    cdef int i, v, w, e, u_e, v_e, cur, p, pe

    for i in range(num_nodes):
        disc[i] = -1
        entry_t[i] = -1
        exit_t[i] = -1

    if n_live_nodes == 0:
        return 0

    # Start DFS from the first live node
    v = live_node_arr[0]
    disc[v] = low[v] = timer
    entry_t[v] = timer
    timer += 1
    par_node_[v] = -1
    par_edge_[v] = -1
    stk_node[0] = v
    stk_cursor[0] = adj_start[v]
    stk_top = 1

    while stk_top > 0:
        v = stk_node[stk_top - 1]
        cur = stk_cursor[stk_top - 1]

        if cur >= adj_start[v + 1]:
            # All neighbours of v visited — finish v and propagate low upward
            stk_top -= 1
            exit_t[v] = timer
            timer += 1
            p = par_node_[v]
            pe = par_edge_[v]
            if p != -1:
                if low[v] < low[p]:
                    low[p] = low[v]
                if low[v] > disc[p]:
                    # Edge (p → v) is a bridge
                    bridge_bonds[n_bridges] = pe
                    bridge_child[n_bridges] = v
                    n_bridges += 1
            continue

        # Advance to the next adjacent edge
        e = adj_list_[cur]
        stk_cursor[stk_top - 1] = cur + 1

        u_e = edges[e, 0]
        v_e = edges[e, 1]
        w = v_e if u_e == v else u_e

        if disc[w] == -1:
            # Tree edge — push w onto the DFS stack
            disc[w] = low[w] = timer
            entry_t[w] = timer
            timer += 1
            par_node_[w] = v
            par_edge_[w] = e
            stk_node[stk_top] = w
            stk_cursor[stk_top] = adj_start[w]
            stk_top += 1
        elif e != par_edge_[v]:
            # Back edge (skip the exact tree edge that brought us here so that
            # parallel edges are treated correctly and not masked as back edges)
            if disc[w] < low[v]:
                low[v] = disc[w]

    return n_bridges


# ---------------------------------------------------------------------------
# Public Python helper: build ring-edge mask from RDKit mol
# (kept for API compatibility with multi_cut_bfs.py)
# ---------------------------------------------------------------------------

def get_ring_edge_mask(mol, int num_bonds):
    """Build a uint8 array marking which bonds are ring bonds.

    Args:
        mol: RDKit molecule (same one used to build mol_d).
        num_bonds: Number of bonds in the molecule (``len(mol_d["bonds"])``).

    Returns:
        Shape ``(num_bonds,)`` uint8 array; 1 if the bond is in a ring, 0 otherwise.
    """
    ring_mask = np.zeros(num_bonds, dtype=MASK_DTYPE)
    for bond in mol.GetBonds():
        if bond.IsInRing():
            ring_mask[bond.GetIdx()] = 1
    return ring_mask


# ---------------------------------------------------------------------------
# Main BFS — same interface as compute_ccs_multi_cut in multi_cut_bfs.py
# ---------------------------------------------------------------------------

def compute_ccs_multi_cut(
    int num_nodes,
    int num_edges,
    cnp.ndarray node_mask_arr,
    cnp.ndarray edges_arr,
    cnp.ndarray edge_mask_arr,
    cnp.ndarray node_to_edge_idx_arr,
    int max_depth = 3,
    long time_limit = 0,
    cnp.ndarray ring_edge_mask = None,
    int max_cut_size = 2,
    smarts_seed_pairs = None,
    ring_bond_groups = None,
    int min_frag_atoms = 0,
):
    """Multi-bond-cut BFS fragmentation (Cython).

    Identical interface and output format to
    ``multi_cut_bfs.compute_ccs_multi_cut`` and ``compute_frags.compute_ccs``.

    Args:
        num_nodes: Number of atoms.
        num_edges: Number of bonds.
        node_mask_arr: Shape ``(num_nodes,)`` uint8 — active atoms.
        edges_arr: Shape ``(MAX_NUM_EDGES, 2)`` int32 — bond endpoints.
        edge_mask_arr: Shape ``(num_edges,)`` uint8 — active bonds.
        node_to_edge_idx_arr: Kept for interface compatibility; unused here.
        max_depth: Maximum BFS depth.
        time_limit: Wall-clock seconds before timeout.  ``0`` (default) means
            no limit.
        ring_edge_mask: Shape ``(num_edges,)`` uint8 — 1 for ring bonds.
            If ``None``, behaves like single-bond BFS.
        max_cut_size: Max simultaneous bond breaks per step (1, 2, or 3).
            Cut=3 is restricted to depth 0 to prevent O(R³) blowup.
        smarts_seed_pairs: Optional list of ``(node_mask_a, node_mask_b)``
            uint8 numpy array pairs pre-computed by ``_apply_smarts_prepass``
            in Python.  Each pair is injected as a depth-1 DAG child pair
            identical to BFS cut=1/2 children.  Pass ``None`` (default) to
            skip the SMARTS prepass.
        ring_bond_groups: List of lists — one entry per SSSR ring, each
            containing the bond indices of that ring (from
            ``mol.GetRingInfo().BondRings()``).  Used by cut=2 and cut=3 to
            pair bonds within the same individual ring rather than the entire
            ring system, reducing O(r²) candidate pairs for fused ring systems.
            If ``None``, cut=2/3 are skipped even when ``ring_edge_mask`` is
            provided.
        min_frag_atoms: Minimum number of heavy atoms a child fragment must
            have to be registered in the DAG.  Fragments smaller than this
            threshold are silently dropped (and their subtrees never explored,
            since all children would be even smaller).  ``0`` disables the
            filter.  Default ``0`` disables the filter.
    Returns:
        Tuple ``(nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix,
        dag_frag_meta)`` — same format as ``compute_frags.compute_ccs``.
    """
    assert num_nodes <= MAX_NUM_NODES, num_nodes
    assert num_edges <= MAX_NUM_EDGES, num_edges

    cdef long start_time = _get_time()
    cdef long cur_time

    # Whether multi-bond cuts are active
    cdef int effective_max_cut = max_cut_size if ring_edge_mask is not None else 1

    # -----------------------------------------------------------------------
    # Typed memoryviews into the input arrays
    # -----------------------------------------------------------------------
    cdef char[::1] node_mask_mv = np.asarray(node_mask_arr, dtype=MASK_DTYPE)
    cdef int[:, ::1] edges_mv = np.asarray(edges_arr, dtype=np.intc)
    cdef char[::1] base_edge_mask_mv = np.asarray(edge_mask_arr, dtype=MASK_DTYPE)

    # -----------------------------------------------------------------------
    # Ring-edge mask memoryview (used to count live ring bonds per BFS node).
    # -----------------------------------------------------------------------
    cdef char[::1] ring_edge_mask_mv

    if ring_edge_mask is not None:
        ring_edge_mask_mv = np.asarray(ring_edge_mask, dtype=MASK_DTYPE)

    # -----------------------------------------------------------------------
    # Pre-allocated scratch arrays for CC computation (reused every cut attempt)
    # -----------------------------------------------------------------------
    cdef cnp.ndarray cc_nodes_np = np.zeros((2, num_nodes), dtype=MASK_DTYPE)
    cdef cnp.ndarray cc_edges_np = np.zeros((2, num_edges), dtype=MASK_DTYPE)
    cdef char[:, ::1] cc_nodes_s = cc_nodes_np
    cdef char[:, ::1] cc_edges_s = cc_edges_np

    # -----------------------------------------------------------------------
    # Pre-allocated working mask buffers.
    # -----------------------------------------------------------------------
    cdef cnp.ndarray cur_node_mask_buf = np.empty(num_nodes, dtype=MASK_DTYPE)
    cdef cnp.ndarray cur_edge_mask_buf = np.empty(num_edges, dtype=MASK_DTYPE)
    cdef char[::1] cur_node_mask_mv = cur_node_mask_buf
    cdef char[::1] cur_edge_mask_mv = cur_edge_mask_buf

    cdef char* key_ptr

    # -----------------------------------------------------------------------
    # Bridge-finding buffers (stack-allocated, reused each BFS node).
    #
    # adj_start_c[129]  — CSR prefix-sum (num_nodes + 1 entries).
    # adj_list_c[1024]  — CSR edge list (2 × MAX_NUM_EDGES entries).
    # bridge_bonds_c / bridge_child_c — one entry per bridge (≤ num_edges).
    # entry_t_c / exit_t_c            — DFS timestamps per node.
    # -----------------------------------------------------------------------
    cdef int adj_start_c[129]
    cdef int adj_list_c[1024]
    cdef int bridge_bonds_c[512]
    cdef int bridge_child_c[512]
    cdef int entry_t_c[128]
    cdef int exit_t_c[128]

    # -----------------------------------------------------------------------
    # BFS state.
    # -----------------------------------------------------------------------
    cdef bytes root_node_key = PyBytes_FromStringAndSize(&node_mask_mv[0], num_nodes)
    cdef bytes root_edge_key = PyBytes_FromStringAndSize(&base_edge_mask_mv[0], num_edges)

    css_to_id_dict  = {root_node_key: 0}
    ccs_depth_bits  = {root_node_key: 1}   # bit 0 = depth 0 for root
    ccs_min_depth   = {root_node_key: 0}
    # queued_set tracks every (node_key, edge_key) pair ever appended to the
    # BFS queue.  Each unique pair is enqueued at most once, eliminating the
    # O(r²) duplicate entries that fused ring systems previously produced.
    queued_set      = {(root_node_key, root_edge_key)}
    # seen_set kept as a safety net (effectively a no-op with queued_set).
    seen_set        = set()
    dag_edge_dict   = {}

    # depth_queue[d] = list of (node_key, edge_key) pairs to process at depth d.
    # A cut=k bond break at depth d places children at depth d+k (cut weight = depth
    # consumed), preventing compounding multi-cuts across depths.
    depth_queue = [[(root_node_key, root_edge_key)]] + [[] for _ in range(max_depth)]

    # -----------------------------------------------------------------------
    # C-level loop variables
    # -----------------------------------------------------------------------
    cdef int current_depth
    cdef int i, j, k, ii, jj, kk
    cdef int n_live, n_ring, n_live_nodes
    cdef int n_ccs
    cdef int c

    # Bridge-finding loop variables
    cdef int n_bridges, bi_b, bridge_e, child_v, ev, xv, node_k, edge_k
    cdef int child_n_atoms
    cdef int cur_node_id    # cached css_to_id_dict[cur_node_key] per BFS node
    cdef int n_css_unique = 1  # counter: replaces len(css_to_id_dict) in hot path

    # C array for live-edge indices (avoid Python list per node)
    cdef int live_arr[512]    # MAX_NUM_EDGES
    # Live atom indices — used for selective scratch reset
    cdef int live_node_arr[128]   # MAX_NUM_NODES

    # Endpoint temporaries for bridge mask computation
    cdef int u_ep, v_ep

    # Per-bond flag: did cutting this bond alone disconnect the molecule?
    cdef cnp.ndarray disconnected_np = np.zeros(num_edges, dtype=MASK_DTYPE)
    cdef char[::1] disconnected_mv = disconnected_np

    # Seed injection working buffers (smarts_seed_pairs)
    cdef cnp.ndarray seed_edge_buf
    cdef char[::1] seed_node_mv
    cdef char[::1] seed_edge_mv
    cdef int u_s, v_s

    force_stop = False
    reached_depth = 0

    # Pre-allocate seed edge buffer once if needed.
    if smarts_seed_pairs is not None:
        seed_edge_buf = np.empty(num_edges, dtype=MASK_DTYPE)
        seed_edge_mv = seed_edge_buf

    # -----------------------------------------------------------------------
    # BFS depth loop
    # -----------------------------------------------------------------------
    for current_depth in range(max_depth):

        for cur_node_key, cur_edge_key in depth_queue[current_depth]:
            cur_time = _get_time()
            if time_limit > 0 and cur_time - start_time > time_limit:
                force_stop = True
                break

            # Safety dedup (effectively no-op when queued_set is correct).
            seen_key = (cur_node_key, cur_edge_key)
            if seen_key in seen_set:
                continue
            seen_set.add(seen_key)

            # Cache the DAG node-id for cur_node_key — used in all child
            # edge-pair registrations below.
            cur_node_id = css_to_id_dict[cur_node_key]

            # -----------------------------------------------------------
            # Reconstruct masks from bytes keys into pre-allocated buffers.
            # -----------------------------------------------------------
            key_ptr = PyBytes_AS_STRING(cur_node_key)
            memcpy(&cur_node_mask_mv[0], key_ptr, num_nodes)

            key_ptr = PyBytes_AS_STRING(cur_edge_key)
            memcpy(&cur_edge_mask_mv[0], key_ptr, num_edges)

            # Build live-edge and live-node C arrays; count live ring bonds.
            n_live = 0
            n_ring = 0
            n_live_nodes = 0
            for i in range(num_edges):
                if cur_edge_mask_mv[i] == 0:
                    continue
                live_arr[n_live] = i
                n_live += 1
                if ring_edge_mask is not None and ring_edge_mask_mv[i] == 1:
                    n_ring += 1
            for i in range(num_nodes):
                if cur_node_mask_mv[i] != 0:
                    live_node_arr[n_live_nodes] = i
                    n_live_nodes += 1

            if n_live == 0:
                continue

            # Reset disconnected flags for this BFS node
            for i in range(n_live):
                disconnected_mv[live_arr[i]] = 0

            # -----------------------------------------------------------
            # Cut = 1: Tarjan bridge-finding (O(V+E)) replaces O(m) CC tests.
            #
            # Build CSR adjacency from live edges, run iterative DFS to find
            # all bridges, then recover fragment masks for each bridge from
            # DFS entry/exit times in O(V+E) per bridge.
            #
            # For each bridge (u, child_v):
            #   CC 0 (child side)  — subtree of child_v in DFS tree.
            #   CC 1 (parent side) — all remaining live nodes.
            # Both sides are contiguous because non-bridge edges cannot span
            # the two sides (that would create a cycle, contradicting bridge).
            # -----------------------------------------------------------
            _build_adj_c(
                num_nodes, num_edges,
                edges_mv, cur_edge_mask_mv,
                adj_start_c, adj_list_c,
            )
            n_bridges = _find_bridges_c(
                num_nodes, num_edges, n_live_nodes, live_node_arr,
                edges_mv, adj_start_c, adj_list_c,
                bridge_bonds_c, bridge_child_c,
                entry_t_c, exit_t_c,
            )

            reached_depth = current_depth + 1
            child_depth_bit = 1 << (current_depth + 1)

            for bi_b in range(n_bridges):
                bridge_e = bridge_bonds_c[bi_b]
                child_v  = bridge_child_c[bi_b]
                ev = entry_t_c[child_v]
                xv = exit_t_c[child_v]

                disconnected_mv[bridge_e] = 1

                # Selective-reset only live positions (same cost as the
                # original _zero_scratch_live before each _cc_uf call).
                _zero_scratch_live(
                    n_live_nodes, n_live, live_node_arr, live_arr,
                    cc_nodes_s, cc_edges_s,
                )

                # Assign each live node to child (CC 0) or parent (CC 1)
                # based on whether it falls inside the DFS subtree of child_v.
                for kk in range(n_live_nodes):
                    node_k = live_node_arr[kk]
                    if entry_t_c[node_k] >= ev and exit_t_c[node_k] <= xv:
                        cc_nodes_s[0, node_k] = 1
                    else:
                        cc_nodes_s[1, node_k] = 1

                # Assign each live non-bridge edge to the CC of its first
                # endpoint.  Because bridge_e is the only cut, all non-bridge
                # live edges have BOTH endpoints in the same CC — so checking
                # just one endpoint is sufficient and correct.
                for kk in range(n_live):
                    edge_k = live_arr[kk]
                    if edge_k == bridge_e:
                        continue
                    u_ep = edges_mv[edge_k, 0]
                    if cc_nodes_s[0, u_ep]:
                        cc_edges_s[0, edge_k] = 1
                    else:
                        cc_edges_s[1, edge_k] = 1

                # Register both child fragments
                for c in range(2):
                    child_n_atoms = 0
                    for kk in range(n_live_nodes):
                        if cc_nodes_s[c, live_node_arr[kk]]:
                            child_n_atoms += 1
                    if child_n_atoms == 0 or child_n_atoms < min_frag_atoms:
                        continue
                    child_node_key = PyBytes_FromStringAndSize(
                        &cc_nodes_s[c, 0], num_nodes)
                    child_edge_key = PyBytes_FromStringAndSize(
                        &cc_edges_s[c, 0], num_edges)

                    if child_node_key not in css_to_id_dict:
                        css_to_id_dict[child_node_key] = n_css_unique
                        n_css_unique += 1

                    if child_node_key not in ccs_min_depth:
                        ccs_min_depth[child_node_key] = current_depth + 1
                        ccs_depth_bits[child_node_key] = child_depth_bit
                    else:
                        ccs_depth_bits[child_node_key] |= child_depth_bit

                    child_queue_key = (child_node_key, child_edge_key)
                    if child_queue_key not in queued_set:
                        depth_queue[current_depth + 1].append((child_node_key, child_edge_key))
                        queued_set.add(child_queue_key)

                    if cur_node_key != child_node_key:
                        edge_pair = (cur_node_id,
                                     css_to_id_dict[child_node_key])
                        if edge_pair not in dag_edge_dict:
                            dag_edge_dict[edge_pair] = current_depth + 1

            # -----------------------------------------------------------
            # Cut = 2: pairs of ring bonds within the same SSSR ring.
            # Cut=2 consumes 2 depth units: children land at current_depth+2.
            #
            # Uses ring_bond_groups (one list per SSSR ring) instead of the
            # old ring-system grouping.  Cross-ring pairs within a fused
            # system are skipped entirely — they never produce n_ccs==2 and
            # the CC test on them was pure overhead.
            # -----------------------------------------------------------
            if effective_max_cut >= 2 and ring_bond_groups is not None and n_ring >= 2 and current_depth + 2 <= max_depth:
                child_depth_bit = 1 << (current_depth + 2)
                for ring_bonds in ring_bond_groups:
                    # Collect bonds from this SSSR ring that are live and
                    # not already identified as bridges (cut=1 candidates).
                    live_ring = []
                    for i in ring_bonds:
                        if cur_edge_mask_mv[i] == 1 and not disconnected_mv[i]:
                            live_ring.append(i)
                    n_sys = len(live_ring)
                    if n_sys < 2:
                        continue
                    for ii in range(n_sys):
                        i = live_ring[ii]
                        for jj in range(ii + 1, n_sys):
                            j = live_ring[jj]

                            cur_edge_mask_mv[i] = 0
                            cur_edge_mask_mv[j] = 0
                            _zero_scratch_live(
                                n_live_nodes, n_live, live_node_arr, live_arr,
                                cc_nodes_s, cc_edges_s,
                            )
                            n_ccs = _cc_uf(
                                num_nodes, num_edges,
                                cur_node_mask_mv, edges_mv, cur_edge_mask_mv,
                                cc_nodes_s, cc_edges_s,
                            )
                            cur_edge_mask_mv[i] = 1
                            cur_edge_mask_mv[j] = 1

                            if n_ccs != 2:
                                continue

                            for c in range(2):
                                child_n_atoms = 0
                                for kk in range(n_live_nodes):
                                    if cc_nodes_s[c, live_node_arr[kk]]:
                                        child_n_atoms += 1
                                if child_n_atoms == 0 or child_n_atoms < min_frag_atoms:
                                    continue

                                child_node_key = PyBytes_FromStringAndSize(
                                    &cc_nodes_s[c, 0], num_nodes)
                                child_edge_key = PyBytes_FromStringAndSize(
                                    &cc_edges_s[c, 0], num_edges)

                                if child_node_key not in css_to_id_dict:
                                    css_to_id_dict[child_node_key] = n_css_unique
                                    n_css_unique += 1

                                if child_node_key not in ccs_min_depth:
                                    ccs_min_depth[child_node_key] = current_depth + 2
                                    ccs_depth_bits[child_node_key] = child_depth_bit
                                else:
                                    ccs_depth_bits[child_node_key] |= child_depth_bit

                                child_queue_key = (child_node_key, child_edge_key)
                                if child_queue_key not in queued_set:
                                    depth_queue[current_depth + 2].append((child_node_key, child_edge_key))
                                    queued_set.add(child_queue_key)

                                if cur_node_key != child_node_key:
                                    edge_pair = (cur_node_id,
                                                 css_to_id_dict[child_node_key])
                                    if edge_pair not in dag_edge_dict:
                                        dag_edge_dict[edge_pair] = current_depth + 2

            # -----------------------------------------------------------
            # Cut = 3: triples of ring bonds within the same SSSR ring —
            # depth 0 only.
            # Cut=3 consumes 3 depth units: children land at current_depth+3.
            # -----------------------------------------------------------
            if effective_max_cut >= 3 and ring_bond_groups is not None and current_depth == 0 and n_ring >= 3 and current_depth + 3 <= max_depth:
                child_depth_bit = 1 << (current_depth + 3)
                for ring_bonds in ring_bond_groups:
                    live_ring = []
                    for i in ring_bonds:
                        if cur_edge_mask_mv[i] == 1 and not disconnected_mv[i]:
                            live_ring.append(i)
                    n_sys = len(live_ring)
                    if n_sys < 3:
                        continue
                    for ii in range(n_sys):
                        i = live_ring[ii]
                        for jj in range(ii + 1, n_sys):
                            j = live_ring[jj]
                            for kk in range(jj + 1, n_sys):
                                k = live_ring[kk]

                                cur_edge_mask_mv[i] = 0
                                cur_edge_mask_mv[j] = 0
                                cur_edge_mask_mv[k] = 0
                                _zero_scratch_live(
                                    n_live_nodes, n_live, live_node_arr, live_arr,
                                    cc_nodes_s, cc_edges_s,
                                )
                                n_ccs = _cc_uf(
                                    num_nodes, num_edges,
                                    cur_node_mask_mv, edges_mv, cur_edge_mask_mv,
                                    cc_nodes_s, cc_edges_s,
                                )
                                cur_edge_mask_mv[i] = 1
                                cur_edge_mask_mv[j] = 1
                                cur_edge_mask_mv[k] = 1

                                if n_ccs != 2:
                                    continue

                                for c in range(2):
                                    child_n_atoms = 0
                                    for kk in range(n_live_nodes):
                                        if cc_nodes_s[c, live_node_arr[kk]]:
                                            child_n_atoms += 1
                                    if child_n_atoms == 0 or child_n_atoms < min_frag_atoms:
                                        continue

                                    child_node_key = PyBytes_FromStringAndSize(
                                        &cc_nodes_s[c, 0], num_nodes)
                                    child_edge_key = PyBytes_FromStringAndSize(
                                        &cc_edges_s[c, 0], num_edges)

                                    if child_node_key not in css_to_id_dict:
                                        css_to_id_dict[child_node_key] = n_css_unique
                                        n_css_unique += 1

                                    if child_node_key not in ccs_min_depth:
                                        ccs_min_depth[child_node_key] = current_depth + 3
                                        ccs_depth_bits[child_node_key] = child_depth_bit
                                    else:
                                        ccs_depth_bits[child_node_key] |= child_depth_bit

                                    child_queue_key = (child_node_key, child_edge_key)
                                    if child_queue_key not in queued_set:
                                        depth_queue[current_depth + 3].append((child_node_key, child_edge_key))
                                        queued_set.add(child_queue_key)

                                    if cur_node_key != child_node_key:
                                        edge_pair = (cur_node_id,
                                                     css_to_id_dict[child_node_key])
                                        if edge_pair not in dag_edge_dict:
                                            dag_edge_dict[edge_pair] = current_depth + 3

        # -------------------------------------------------------------------
        # SMARTS prepass injection (depth 0 only).
        # -------------------------------------------------------------------
        if current_depth == 0 and smarts_seed_pairs is not None:
            for pair in smarts_seed_pairs:
                pair_mask_a, pair_mask_b, _pair_rule_idx = pair
                for child_mask_np in (pair_mask_a, pair_mask_b):
                    if int(child_mask_np.sum()) < min_frag_atoms:
                        continue
                    seed_node_mv = np.ascontiguousarray(child_mask_np, dtype=MASK_DTYPE)
                    child_node_key = PyBytes_FromStringAndSize(&seed_node_mv[0], num_nodes)
                    if child_node_key == root_node_key:
                        continue
                    for i in range(num_edges):
                        if base_edge_mask_mv[i] == 0:
                            seed_edge_mv[i] = 0
                        else:
                            u_s = edges_mv[i, 0]
                            v_s = edges_mv[i, 1]
                            seed_edge_mv[i] = (
                                1 if (seed_node_mv[u_s] == 1 and seed_node_mv[v_s] == 1) else 0
                            )
                    child_edge_key = PyBytes_FromStringAndSize(&seed_edge_mv[0], num_edges)

                    if child_node_key not in css_to_id_dict:
                        css_to_id_dict[child_node_key] = n_css_unique
                        n_css_unique += 1
                    if child_node_key not in ccs_min_depth:
                        ccs_min_depth[child_node_key] = 1
                        ccs_depth_bits[child_node_key] = 1 << 1
                    else:
                        ccs_depth_bits[child_node_key] |= (1 << 1)

                    child_queue_key = (child_node_key, child_edge_key)
                    if child_queue_key not in queued_set:
                        depth_queue[1].append((child_node_key, child_edge_key))
                        queued_set.add(child_queue_key)

                    edge_pair = (css_to_id_dict[root_node_key],
                                 css_to_id_dict[child_node_key])
                    if edge_pair not in dag_edge_dict:
                        dag_edge_dict[edge_pair] = 1

        if force_stop:
            reached_depth = current_depth - 1 if current_depth > 0 else 0
            valid_bits = (1 << (reached_depth + 1)) - 1
            ccs_depth_bits = {
                k: (v & valid_bits) for k, v in ccs_depth_bits.items() if v & valid_bits
            }
            ccs_min_depth = {k: v for k, v in ccs_min_depth.items() if v <= reached_depth}
            css_to_id_dict = {k: css_to_id_dict[k] for k in ccs_min_depth}
            id_to_css = {v: k for k, v in css_to_id_dict.items()}
            dag_edge_dict = {
                (s, d): dep
                for (s, d), dep in dag_edge_dict.items()
                if dep <= reached_depth and s in id_to_css and d in id_to_css
            }
            break

    # -----------------------------------------------------------------------
    # Build output matrices — same format as compute_frags.compute_ccs
    # -----------------------------------------------------------------------
    num_un_ccs = len(css_to_id_dict)
    ordered_ccs = [b""] * num_un_ccs
    for cc, idx in css_to_id_dict.items():
        ordered_ccs[idx] = cc

    nodes_mask_matrix = np.stack(
        [np.frombuffer(cc, dtype=MASK_DTYPE) for cc in ordered_ccs],
        dtype=MASK_DTYPE,
    )
    nodes_depth_matrix = np.zeros((num_un_ccs, max_depth + 1), dtype=MASK_DTYPE)
    nodes_min_depth    = np.zeros(num_un_ccs, dtype=MASK_DTYPE)

    for idx, cc in enumerate(ordered_ccs):
        bits = ccs_depth_bits[cc]
        for d in range(max_depth + 1):
            if (bits >> d) & 1:
                nodes_depth_matrix[idx, d] = 1
        nodes_min_depth[idx] = ccs_min_depth[cc]

    if dag_edge_dict:
        dag_edges_matrix = np.array(list(dag_edge_dict.keys()), dtype=np.int64)
        edges_min_depth  = np.array(list(dag_edge_dict.values()), dtype=MASK_DTYPE)
    else:
        dag_edges_matrix = np.empty((0, 2), dtype=np.int64)
        edges_min_depth  = np.empty(0, dtype=MASK_DTYPE)

    dag_frag_meta = {
        "reached_depth":      reached_depth,
        "edges_min_depth":    edges_min_depth,
        "nodes_min_depth":    nodes_min_depth,
        "force_stopped":      force_stop,
    }
    return nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta
