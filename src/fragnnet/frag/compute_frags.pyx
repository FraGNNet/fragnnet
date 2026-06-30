# cython: profile=False
# cython: linetrace=False
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
import signal
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize
from libc.string cimport memcpy, memset
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

cimport numpy as cnp

# Global interrupt flag
interrupted = False

def signal_handler(signum, frame):
	global interrupted
	interrupted = True

# Register the handler
signal.signal(signal.SIGINT, signal_handler)

MAX_NUM_NODES = 128  # hard cap on 80!!
MAX_NUM_EDGES = 4 * MAX_NUM_NODES
MASK_DTYPE = np.uint8
MAX_EDGE_PER_NODE = 6  # Sulfur

class TimeoutError(RuntimeError):
	pass

# ---------------------------------------------------------------------------
# Legacy fill helpers — kept for compute_cc_h_floor / update_bonds
# ---------------------------------------------------------------------------

cdef void np_fill(int [::1] array, int val) nogil:
	cdef Py_ssize_t i, num_elems = array.shape[0]
	for i in range(num_elems):
		array[i] = val

cdef void char_fill(char [::1] array, char val) nogil:
	cdef Py_ssize_t i, num_elems = array.shape[0]
	for i in range(num_elems):
		array[i] = val

cdef void char2d_fill(char [:,::1] array, char val) nogil:
	cdef Py_ssize_t i, j
	cdef Py_ssize_t rows = array.shape[0]
	cdef Py_ssize_t cols = array.shape[1]
	for i in range(rows):
		for j in range(cols):
			array[i, j] = val

cdef int np_sum(char [::1] array) nogil:
	cdef Py_ssize_t i, num_elems = array.shape[0]
	cdef int total = 0
	for i in range(num_elems):
		total += array[i]
	return total

cdef long get_time():
	cdef timespec ts
	cdef long current
	clock_gettime(CLOCK_REALTIME, &ts)
	current = ts.tv_sec
	return current

def mask_to_binary_hash(char [::1] mask):
	return np.asarray(mask).tobytes()

# ---------------------------------------------------------------------------
# Union-Find CC — replaces BFS-based connected_components.
#
# Key improvements over the old BFS implementation:
#   - O(α·n) union-find vs O(V+E) BFS per cut attempt.
#   - C stack arrays (no heap allocation per call).
#   - Caller-zeroed scratch arrays + selective reset eliminates 5×
#     numpy allocations per CC call that the BFS version required.
#   - Returns early with sentinel 99 when > 2 CCs found.
# ---------------------------------------------------------------------------

cdef inline int _uf_find(int* parent, int x) nogil:
	"""Path-halving find — O(α) amortised."""
	while parent[x] != x:
		parent[x] = parent[parent[x]]
		x = parent[x]
	return x


cdef inline void _zero_scratch_full(
	int num_nodes,
	int num_edges,
	char[:, ::1] cc_nodes_s,
	char[:, ::1] cc_edges_s,
) nogil:
	"""Full memset of CC scratch arrays before the first cut attempt per BFS node."""
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
	"""Selective reset — only clears entries _cc_uf can write, avoids full memset."""
	cdef int idx
	for idx in range(n_live_nodes):
		cc_nodes_s[0, live_node_arr[idx]] = 0
		cc_nodes_s[1, live_node_arr[idx]] = 0
	for idx in range(n_live_edges):
		cc_edges_s[0, live_edge_arr[idx]] = 0
		cc_edges_s[1, live_edge_arr[idx]] = 0


cdef int _cc_uf(
	int num_nodes,
	int num_edges,
	char[::1]    node_mask,
	int[:, ::1]  edges,
	char[::1]    edge_mask,
	char[:, ::1] cc_nodes_out,   # (2, num_nodes) — zeroed by caller
	char[:, ::1] cc_edges_out,   # (2, num_edges) — zeroed by caller
) nogil:
	"""Connected components via union-find.

	Returns num_ccs (0, 1, 2) or 99 if > 2 components (invalid cut).
	Results written into caller-zeroed cc_nodes_out / cc_edges_out.
	Uses C stack arrays — zero heap allocation.
	"""
	cdef int parent[128]   # MAX_NUM_NODES
	cdef int cc_root[128]  # atom root → CC index (-1 = unseen)
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

	# Assign CC indices to live atoms
	num_ccs = 0
	for i in range(num_nodes):
		if node_mask[i] == 0:
			continue
		r = _uf_find(parent, i)
		if cc_root[r] == -1:
			if num_ccs == 2:
				return 99   # > 2 CCs → bail early
			cc_root[r] = num_ccs
			num_ccs += 1
		cc_nodes_out[cc_root[r], i] = 1

	# Assign live edges to their CC
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
# node_to_edge_idx — kept for interface compatibility with frag_utils.py
# ---------------------------------------------------------------------------

cdef int [:,::1] compute_node_to_edge_idx(int num_nodes, int num_edges, int [:,::1] edges):
	cdef int [:,::1] node_to_edge_idx = np.full((num_nodes, MAX_EDGE_PER_NODE), -1, dtype=np.intc)
	cdef int src_node, dst_node, j, k
	for j in range(num_edges):
		src_node = edges[j,0]
		dst_node = edges[j,1]
		for k in range(MAX_EDGE_PER_NODE):
			if node_to_edge_idx[src_node,k] == -1:
				node_to_edge_idx[src_node,k] = j
				break
		for k in range(MAX_EDGE_PER_NODE):
			if node_to_edge_idx[dst_node,k] == -1:
				node_to_edge_idx[dst_node,k] = j
				break
	return node_to_edge_idx

def py_compute_node_to_edge_idx(int num_nodes, int num_edges, int [:,::1] edges):
	return np.asarray(compute_node_to_edge_idx(num_nodes,num_edges,edges),dtype=np.intc)


# ---------------------------------------------------------------------------
# Main BFS — optimised rewrite of the original compute_ccs.
#
# Changes vs the old implementation
# -----------------------------------
# - _cc_uf (union-find) replaces connected_components (BFS):
#     old: O(V+E) BFS + 5× numpy allocation per cut.
#     new: O(α·n) UF + zero heap allocation per cut.
# - cc_node_mask_dict / cc_edge_mask_dict eliminated:
#     masks are now reconstructed from bytes keys on demand via
#     PyBytes_AS_STRING + memcpy (no per-fragment numpy copy).
# - cc_edge_active_dict eliminated:
#     live edges collected into a C int[] per BFS node.
# - PyBytes_FromStringAndSize replaces np.asarray(...).tobytes():
#     direct C API, no buffer-protocol round-trip.
# - ccs_depth_dict (Python set per fragment) → int bitmask:
#     bit-OR update is faster than set.add(), no set object created.
# - has_seen_dict replaced by seen_set (plain set):
#     BFS processes depths in order → once-seen = always-skip is correct.
# - Selective scratch reset (_zero_scratch_live) after first cut:
#     avoids re-memset'ing the whole (2×N / 2×E) buffer per edge.
# ---------------------------------------------------------------------------

def compute_ccs(
	int num_nodes,
	int num_edges,
	char [::1] node_mask,
	int [:,::1] edges,
	char [::1] edge_mask,
	int [:,::1] node_to_edge_idx,   # kept for API compat; unused internally
	int max_depth,
	long time_limit,
	int min_frag_atoms = 3,
):
	assert num_nodes <= MAX_NUM_NODES, num_nodes
	assert num_edges <= MAX_NUM_EDGES, num_edges
	cdef long start_time = get_time()
	cdef long cur_time = 0

	# Pre-allocated CC scratch arrays (reused every cut attempt)
	cdef cnp.ndarray cc_nodes_np = np.zeros((2, num_nodes), dtype=MASK_DTYPE)
	cdef cnp.ndarray cc_edges_np = np.zeros((2, num_edges), dtype=MASK_DTYPE)
	cdef char[:, ::1] cc_nodes_s = cc_nodes_np
	cdef char[:, ::1] cc_edges_s = cc_edges_np

	# Working buffers: masks reconstructed from bytes keys via memcpy.
	# Eliminates cc_node_mask_dict / cc_edge_mask_dict entirely.
	cdef cnp.ndarray cur_node_mask_buf = np.empty(num_nodes, dtype=MASK_DTYPE)
	cdef cnp.ndarray cur_edge_mask_buf = np.empty(num_edges, dtype=MASK_DTYPE)
	cdef char[::1] cur_node_mask_mv = cur_node_mask_buf
	cdef char[::1] cur_edge_mask_mv = cur_edge_mask_buf
	cdef char* key_ptr

	# Root keys via direct C API (avoids .tobytes() numpy round-trip)
	cdef bytes root_node_key = PyBytes_FromStringAndSize(&node_mask[0], num_nodes)
	cdef bytes root_edge_key = PyBytes_FromStringAndSize(&edge_mask[0], num_edges)

	css_to_id_dict = {root_node_key: 0}
	# int bitmask per fragment: bit d = depth d seen.
	# Replaces Python set per fragment — bit-OR is faster than set.add().
	ccs_depth_bits = {root_node_key: 1}   # bit 0 = depth 0 for root
	ccs_min_depth  = {root_node_key: 0}
	# Plain set replaces has_seen_dict: BFS depth-order guarantees
	# once-seen → always-skip is safe (no depth value needed).
	seen_set       = set()
	dag_edge_dict  = {}

	cdef list ccs           = [root_node_key]
	cdef list cc_edges_list = [root_edge_key]

	cdef int ccs_idx = 0, ccs_end = 0, node_idx
	cdef int current_depth, i, ii, c, kk
	cdef int n_live, n_live_nodes, n_ccs
	cdef int child_n_atoms

	# C arrays for live edge / node indices — no Python list per BFS node
	cdef int live_arr[512]       # MAX_NUM_EDGES
	cdef int live_node_arr[128]  # MAX_NUM_NODES

	force_stop    = False
	reached_depth = 0

	for current_depth in range(max_depth):
		ccs_end = len(ccs)

		for node_idx in range(ccs_idx, ccs_end):
			cur_time = get_time()
			if interrupted:
				raise KeyboardInterrupt("Execution interrupted by user (Ctrl+C)")
			if cur_time - start_time > time_limit:
				force_stop = True
				break

			cur_node_key = ccs[node_idx]
			cur_edge_key = cc_edges_list[node_idx]

			# Tuple key reuses cached per-element hashes — cheaper than bytes concat.
			seen_key = (cur_node_key, cur_edge_key)
			if seen_key in seen_set:
				continue
			seen_set.add(seen_key)

			# Reconstruct masks from bytes keys via C memcpy — zero heap allocation,
			# no buffer-protocol type checks, no numpy intermediary.
			key_ptr = PyBytes_AS_STRING(cur_node_key)
			memcpy(&cur_node_mask_mv[0], key_ptr, num_nodes)
			key_ptr = PyBytes_AS_STRING(cur_edge_key)
			memcpy(&cur_edge_mask_mv[0], key_ptr, num_edges)

			# Build C arrays of live edges and live nodes
			n_live = 0
			n_live_nodes = 0
			for i in range(num_edges):
				if cur_edge_mask_mv[i] == 1:
					live_arr[n_live] = i
					n_live += 1
			for i in range(num_nodes):
				if cur_node_mask_mv[i] != 0:
					live_node_arr[n_live_nodes] = i
					n_live_nodes += 1

			if n_live == 0:
				continue

			# Full memset before first CC call; selective reset for subsequent cuts.
			_zero_scratch_full(num_nodes, num_edges, cc_nodes_s, cc_edges_s)

			child_depth_bit = 1 << (current_depth + 1)

			for ii in range(n_live):
				i = live_arr[ii]
				cur_edge_mask_mv[i] = 0
				if ii > 0:
					_zero_scratch_live(
						n_live_nodes, n_live,
						live_node_arr, live_arr,
						cc_nodes_s, cc_edges_s,
					)
				n_ccs = _cc_uf(
					num_nodes, num_edges,
					cur_node_mask_mv, edges, cur_edge_mask_mv,
					cc_nodes_s, cc_edges_s,
				)
				cur_edge_mask_mv[i] = 1

				if n_ccs == 99:
					continue   # >2 CCs -- impossible for single-cut; skip

				# n_ccs==1: ring bond opened (same atoms, chain) -> enqueue for depth+1
				# n_ccs==2: bridge bond -> genuine split, add DAG edge
				for c in range(n_ccs):
					child_n_atoms = 0
					for kk in range(n_live_nodes):
						if cc_nodes_s[c, live_node_arr[kk]]:
							child_n_atoms += 1
					if child_n_atoms < min_frag_atoms:
						continue
					child_node_key = PyBytes_FromStringAndSize(&cc_nodes_s[c, 0], num_nodes)
					child_edge_key = PyBytes_FromStringAndSize(&cc_edges_s[c, 0], num_edges)

					if child_node_key not in css_to_id_dict:
						css_to_id_dict[child_node_key] = len(css_to_id_dict)

					if child_node_key not in ccs_min_depth:
						ccs_min_depth[child_node_key] = current_depth + 1
						ccs_depth_bits[child_node_key] = child_depth_bit
					else:
						ccs_depth_bits[child_node_key] |= child_depth_bit

					ccs.append(child_node_key)
					cc_edges_list.append(child_edge_key)

					if cur_node_key != child_node_key:
						edge_pair = (css_to_id_dict[cur_node_key],
						             css_to_id_dict[child_node_key])
						if edge_pair not in dag_edge_dict:
							dag_edge_dict[edge_pair] = current_depth + 1

			reached_depth = current_depth + 1

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

		ccs_idx = ccs_end

	# -----------------------------------------------------------------------
	# Build output matrices — same format as the old compute_ccs
	# -----------------------------------------------------------------------
	num_un_ccs = len(css_to_id_dict)
	ordered_ccs = [b""] * num_un_ccs
	for cc, idx in css_to_id_dict.items():
		ordered_ccs[idx] = cc

	# np.frombuffer avoids a copy (bytes are immutable, view is safe read-only)
	nodes_mask_matrix = np.stack(
		[np.frombuffer(cc, dtype=MASK_DTYPE) for cc in ordered_ccs],
		dtype=MASK_DTYPE,
	)
	nodes_depth_matrix = np.zeros((num_un_ccs, max_depth + 1), dtype=MASK_DTYPE)
	nodes_min_depth    = np.zeros(num_un_ccs, dtype=MASK_DTYPE)

	# Unpack bitmask into depth matrix without creating Python set intermediaries
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
		"reached_depth":   reached_depth,
		"edges_min_depth": edges_min_depth,
		"nodes_min_depth": nodes_min_depth,
		"force_stopped":   force_stop,
	}
	return nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta


def compute_cc_h_floor(
		cnp.ndarray[cnp.int32_t, ndim=1] cc_atom_ids,
		cnp.ndarray[cnp.int32_t, ndim=1] ve_arr,
		cnp.ndarray[cnp.int32_t, ndim=1] sbond_arr,
		int num_radicals,
		cnp.ndarray[cnp.int32_t, ndim=2] bonds,
		dict atoms_to_bonds,
		cnp.ndarray[cnp.uint8_t, ndim=1] bond_mask_arr):  # Fix: uint8_t for boolean array

	"""Compute the minimum number of Hs a connected component (cc) can have."""

	assert num_radicals == 0

	# Compute the difference array (initial hydrogen deficit)P
	cdef cnp.ndarray[cnp.int32_t, ndim=1] diff_arr = np.maximum(ve_arr - sbond_arr, 0)
	cdef cnp.ndarray[cnp.int32_t, ndim=1] h_arr = diff_arr.copy()

	# Define Cython integer variables
	cdef Py_ssize_t atom, bond_idx, other
	cdef list bond_idxs
	cdef tuple bond

	# Iterate over atoms in the connected component
	for atom in cc_atom_ids:
		bond_idxs = atoms_to_bonds[atom]  # List of bond indices for this atom

		for bond_idx in bond_idxs:
			if h_arr[atom] == 0:
				break
			if bond_mask_arr[bond_idx] == 0:  # Use explicit comparison for uint8_t
				continue

			other = bonds[bond_idx, 1] if bonds[bond_idx, 0] == atom else bonds[bond_idx, 0]

			# Ensure we don't form more than 3 bonds
			h_arr[atom] = max(0, h_arr[atom] - min(diff_arr[other], 2))

	# Compute the lower bound of hydrogen count
	cdef int cc_floor = max(h_arr[cc_atom_ids].sum() - num_radicals, 0)

	return cc_floor

def update_bonds(
		cnp.ndarray[cnp.int32_t, ndim=1] cc_atom_ids,
		cnp.ndarray[cnp.int32_t, ndim=1] sbond_arr,
		cnp.ndarray[cnp.uint8_t, ndim=1] bond_mask_arr,  # Using uint8_t for boolean
		cnp.ndarray[cnp.int32_t, ndim=2] bonds,
		dict atoms_to_bonds):

	"""Updates single bond counts and bond masks for a given connected component."""

	# Create new arrays with the same shape as sbond_arr and bond_mask_arr
	cdef cnp.ndarray[cnp.int32_t, ndim=1] new_sbond_arr = np.zeros_like(sbond_arr, dtype=np.int32)
	cdef cnp.ndarray[cnp.uint8_t, ndim=1] new_bond_mask_arr = np.zeros_like(bond_mask_arr, dtype=np.uint8)

	# Boolean lookup array: O(1) membership test instead of O(n) linear scan per atom
	cdef Py_ssize_t num_atoms = sbond_arr.shape[0]
	cdef cnp.ndarray[cnp.uint8_t, ndim=1] in_cc = np.zeros(num_atoms, dtype=np.uint8)
	cdef Py_ssize_t k
	for k in range(cc_atom_ids.shape[0]):
		in_cc[cc_atom_ids[k]] = 1

	# Define Cython integer variables
	cdef Py_ssize_t atom, other, bond_idx, i, j
	cdef list bond_idxs

	# Iterate through each atom in the connected component
	for i in range(cc_atom_ids.shape[0]):
		atom = cc_atom_ids[i]
		bond_idxs = atoms_to_bonds[atom]  # Get bond indices for this atom

		for j in range(len(bond_idxs)):
			bond_idx = bond_idxs[j]

			# Get bond endpoints
			other = bonds[bond_idx, 1] if bonds[bond_idx, 0] == atom else bonds[bond_idx, 0]

			# O(1) lookup instead of O(n) linear scan through cc_atom_ids
			if in_cc[other]:
				new_sbond_arr[atom] += 1
				new_bond_mask_arr[bond_idx] = 1  # Use 1 instead of True for uint8_t

	return new_sbond_arr, new_bond_mask_arr
