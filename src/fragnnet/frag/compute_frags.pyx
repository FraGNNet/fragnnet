# cython: profile=False
# cython: linetrace=False
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from cysignals.signals cimport sig_check
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

MAX_NUM_NODES = 128  # hard cap on 64!!
MAX_NUM_EDGES = 4 * MAX_NUM_NODES  # ~4*MAX_NUM_NODES
MASK_DTYPE = np.uint8
MAX_EDGE_PER_NODE = 6  # Sulfur

class TimeoutError(RuntimeError):
	pass

cdef void np_fill(int [::1] array, int val) nogil:
	cdef Py_ssize_t i, num_elems = array.shape[0]
	for i in range(num_elems):
		array[i] = val

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

#cdef long node_mask_to_int(int num_nodes, char [::1] node_mask):

#	assert MAX_NUM_NODES < 64, MAX_NUM_NODES
#	cdef long node_mask_int = 0
#	cdef int i
#	for i in range(num_nodes):
#		assert node_mask[i] >= 0, node_mask[i]
#		node_mask_int += node_mask[i]*(2**i)
#	assert node_mask_int >= 0, node_mask_int
#	return node_mask_int

def mask_to_binary_hash(char [::1] mask):
	return np.asarray(mask).tobytes()

cdef int [:,::1] compute_node_to_edge_idx(int num_nodes, int num_edges, int [:,::1] edges):
	# use np fill instead of np.ones to speed up
	cdef int [:,::1] node_to_edge_idx = np.full((num_nodes, MAX_EDGE_PER_NODE), -1, dtype=np.intc)
	cdef int src_node, dst_node, j, k
	for j in range(num_edges):
		src_node = edges[j,0]
		dst_node = edges[j,1]
		#assert src_node < num_nodes and dst_node < num_nodes
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

cdef connected_components(
		int num_nodes,
		int num_edges,
		char [::1] node_mask,
		int [:,::1] edges,
		char [::1] edge_mask,
		int [:,::1] node_to_edge_idx
	):
	# print('>connected_components')
	# return the number of connected components
	cdef Py_ssize_t num_visited = 0
	cdef Py_ssize_t num_nonmask = np_sum(node_mask)
	cdef Py_ssize_t cur_node
	cdef Py_ssize_t num_ccs = 0
	cdef Py_ssize_t start_queue = 0
	cdef Py_ssize_t end_queue = 0
	cdef Py_ssize_t cur_edge
	cdef Py_ssize_t i
	cdef Py_ssize_t j
	cdef Py_ssize_t neighbour

	cdef int [::1] node_visited = np.zeros(num_nodes,dtype=np.intc)
	cdef int [::1] node_enqueued = np.zeros(num_nodes,dtype=np.intc)
	cdef int [::1] node_queue = np.full(num_nodes, -1, dtype=np.intc)

	cdef char [:,::1] cc_nodes = np.zeros((2,num_nodes),dtype= MASK_DTYPE)
	cdef char [:,::1] cc_edges = np.zeros((2,num_edges),dtype= MASK_DTYPE)

	# run BFS
	while num_visited < num_nonmask:
		if start_queue != end_queue:
			# dequeue
			cur_node = node_queue[start_queue]
			start_queue = (start_queue + 1) % num_nodes
			# mark as visited
			#assert node_enqueued[cur_node] == 1, cur_node
			#assert node_visited[cur_node] == 0, cur_node
			#assert node_mask[cur_node] == 1, cur_node
			node_visited[cur_node] = 1
			num_visited += 1
			# add node to cc
			#assert num_ccs > 0 and num_ccs <= 2, num_ccs
			#assert cur_node < MAX_NUM_NODES, cur_node
			cc_nodes[num_ccs-1,cur_node] = 1
			# add neighbours
			for j in range(MAX_EDGE_PER_NODE):
				cur_edge = node_to_edge_idx[cur_node,j]
				# check if null
				if cur_edge == -1:
					break
				# check if edge is masked
				if edge_mask[cur_edge] == 0:
					continue
				# add unvisited neighbours
				if edges[cur_edge,0] == cur_node:
					neighbour = edges[cur_edge,1]
				else:
					#assert edges[cur_edge,1] == cur_node
					neighbour = edges[cur_edge,0]

				# if we have not see node before
				if node_enqueued[neighbour] == 0:
					#assert node_visited[neighbour] == 0
					node_queue[end_queue] = neighbour
					end_queue = (end_queue + 1) % num_nodes
					node_enqueued[neighbour] = 1

				# add edge to cc
				cc_edges[num_ccs-1,cur_edge] = 1

		else:
			# check for new nodes to seed BFS
			for i in range(num_nodes):
				if node_mask[i] == 1 and node_visited[i] == 0:
					# enqueue
					node_queue[end_queue] = i
					end_queue = (end_queue + 1) % num_nodes
					#print(f"new end node {end_queue}")
					#assert node_enqueued[i] == 0
					node_enqueued[i] = 1
					num_ccs += 1
					break
	
	if num_visited != num_nonmask:
		raise ValueError(f'num_visited ({num_visited}) != num_nonmask ({num_nonmask})')
	
	if start_queue != end_queue:
		raise ValueError(f'start_queue ({start_queue}) != end_queue ({end_queue}), num_ccs {num_ccs}')

	if num_ccs < 0 or num_ccs > 2:
		raise ValueError(f'{num_ccs} num_ccs <0 or num_ccs > 2')
	
	return cc_nodes, cc_edges, num_ccs

def compute_ccs(
		int num_nodes,
		int num_edges,
		char [::1] node_mask,
		int [:,::1] edges,
		char [::1] edge_mask,
		int [:,::1] node_to_edge_idx,
		int max_depth,
		long time_limit
	):

	# and a time check
	assert num_nodes <= MAX_NUM_NODES, num_nodes
	assert num_edges <= MAX_NUM_EDGES, num_edges
	cdef long start_time = get_time()

	# this need happen within loop
	cdef long cur_time = 0
	
	# define a load of things
	cdef list child_node_mask_list = []
	cdef list child_edge_mask_list = []
	cdef int node_idx
	cdef int num_edges_nomask = 0
	
	cdef int ccs_idx = 0
	cdef int ccs_end = 0
	cdef int c = 0
	cdef int i = 0
	cdef int current_depth = 0

	# start with return_parents
	node_mask_key = mask_to_binary_hash(node_mask)
	edge_mask_key = mask_to_binary_hash(edge_mask)

	cdef list ccs = [node_mask_key]
	cdef list cc_edges = [edge_mask_key]
	cdef list ccs_depths = [0]
	cdef dict cc_node_mask_dict = {node_mask_key:node_mask}
	cdef dict cc_edge_mask_dict = {edge_mask_key:edge_mask}
	cdef dict has_seen_dict = {}
	cdef dict ccs_depth_dict = {node_mask_key:set({0})}
	cdef int num_ccs = 0
	
	# data for gaphs
	cdef dict css_to_id_dict = {node_mask_key:0}
	# needed by pyg
	# dag_edge_dict, used to keep track dag edges and min depth
	dag_edge_dict = {} #set({})

	#print("test_list", test_list[0])
	force_stop = False
	reached_depth = 0
	# for each depth
	for current_depth in range(max_depth):
		# keep all the ccs mask for next round
		child_edge_mask_list = []		
		ccs_end = len(ccs)
		
		for node_idx in range(ccs_idx,ccs_end):
			# let us check time
			cur_time = get_time()
			sig_check()
			if cur_time - start_time > time_limit:
				force_stop = True
				break
			
			# get mask hash and mask
			curent_node_mask_key = ccs[node_idx]
			current_edge_mask_key = cc_edges[node_idx]
			current_node_mask = cc_node_mask_dict[curent_node_mask_key]
			current_edge_mask = cc_edge_mask_dict[current_edge_mask_key]

			# let us check mask, there is more than one way to get each node
			# thus, we need to check if we have seen this node mask and edge mask before
			node_full_mask_str = curent_node_mask_key + b'|' +  current_edge_mask_key

			# check if we have seen has configure before
			need_skip = False
			if node_full_mask_str not in has_seen_dict:
				has_seen_dict[node_full_mask_str] = current_depth
			elif has_seen_dict[node_full_mask_str] > current_depth:
				has_seen_dict[node_full_mask_str] = current_depth
			else:
				need_skip = True

			# check if there any bond left to kill
			num_edges_nomask = np_sum(current_edge_mask)
			if need_skip or num_edges_nomask == 0:
				continue

			# now we need to break more bond
			#i = 0
			for i in range(num_edges):
				# check if edge is already masked
				# sorry Mr Bond, you already dead
				if current_edge_mask[i] == 0:
					continue
				# mask edge, kill bond
				current_edge_mask[i] = 0
				# compute ccs
				new_cc_nodes, new_cc_edges, new_num_ccs = connected_components(
					num_nodes,
					num_edges,
					current_node_mask,
					edges,
					current_edge_mask,
					node_to_edge_idx
				)

				# restore edge
				current_edge_mask[i] = 1
				# save data and save edge and node configure
				for c in range(new_num_ccs):
					# save cc_nodes
					child_node_mask_key = mask_to_binary_hash(new_cc_nodes[c,:])
					child_edge_mask_key = mask_to_binary_hash(new_cc_edges[c,:])

					# add to dict to track node and edges
					if child_node_mask_key not in cc_node_mask_dict:
						cc_node_mask_dict[child_node_mask_key] = new_cc_nodes[c,:]
						# track ccs to idx 
						css_to_id_dict[child_node_mask_key] = len(css_to_id_dict)
						
					if child_edge_mask_key not in cc_edge_mask_dict:
						cc_edge_mask_dict[child_edge_mask_key] = new_cc_edges[c,:]

					# track depth for each cc node
					if child_node_mask_key not in ccs_depth_dict:
						ccs_depth_dict[child_node_mask_key] = set([current_depth + 1 ])
					else:
						ccs_depth_dict[child_node_mask_key].add(current_depth + 1)

					ccs.append(child_node_mask_key)
					cc_edges.append(child_edge_mask_key)
					
					# add edge from node to node if mask are differen
					# NOTE: it is possiable we have parent and child with the same mask
					# eg. first bond break on a ring
					if curent_node_mask_key != child_node_mask_key:
						
						edge_pair = (css_to_id_dict[curent_node_mask_key],css_to_id_dict[child_node_mask_key])
						#if edge_pair_ids not in 
						if edge_pair not in  dag_edge_dict:
							dag_edge_dict[edge_pair] = current_depth + 1
					
				num_ccs += new_num_ccs
				reached_depth = current_depth + 1
		# break outer loop
		if force_stop:
			# comupute what can be used
			# NOTE this has not been tested yet
			filtered_dag_edge_dict = {k: v for k, v in dag_edge_dict.items() if v <= reached_depth}
			dag_edge_dict = filtered_dag_edge_dict

			filtered_ccs_depth_dict = {k: set([i for i in v if i <= reached_depth]) for k,v in ccs_depth_dict.items() if min(v) <= reached_depth}
			ccs_depth_dict = filtered_ccs_depth_dict

			filtered_css_to_id_dict = {cc: css_to_id_dict[cc] for cc in filtered_ccs_depth_dict}
			css_to_id_dict =  filtered_css_to_id_dict
			#original.update(filtered)
			break
		# finishing current depth, move to next
		ccs_idx = ccs_end

	# build graph
	num_un_ccs = len(css_to_id_dict)
	# create sorted node matirx with masks
	cdef list ordered_ccs = [0] * num_un_ccs
	for cc in css_to_id_dict:
		ordered_ccs[css_to_id_dict[cc]] = cc

	# create node masks
	nodes_mask_matrix = np.stack([np.asarray(cc_node_mask_dict[cc]) for cc in ordered_ccs], dtype = MASK_DTYPE)

	# create node depth matrix
	nodes_depth_matrix = np.zeros((num_un_ccs, max_depth + 1),dtype = MASK_DTYPE)
	nodes_min_depth = np.zeros((num_un_ccs),dtype = MASK_DTYPE)

	for idx,cc in enumerate(ordered_ccs):
		for d_id in ccs_depth_dict[cc]:
			nodes_depth_matrix[idx,d_id] = 1
		# set lowest depth on the dag for each node
		nodes_min_depth[idx] = min(ccs_depth_dict[cc])

	# create edge set:
	dag_edges_matrix = np.asarray(list(dag_edge_dict.keys()), dtype = np.int64)
	edges_min_depth = np.asarray(list(dag_edge_dict.values()), dtype = MASK_DTYPE)
	dag_frag_meta = {"reached_depth": reached_depth, "edges_min_depth": edges_min_depth, "nodes_min_depth": nodes_min_depth }
	# both edges_min_depth and node_min_depth should be ordered

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

			#bond = bonds[bond_idx]
			#other = bond[1] if bond[0] == atom else bond[0]
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

	# Define Cython integer variables
	cdef Py_ssize_t  atom, other, bond_idx, i, j
	cdef list bond_idxs

	# Iterate through each atom in the connected component
	for i in range(cc_atom_ids.shape[0]):
		atom = cc_atom_ids[i]
		bond_idxs = atoms_to_bonds[atom]  # Get bond indices for this atom

		for j in range(len(bond_idxs)):
			bond_idx = bond_idxs[j]

			# Get bond endpoints
			other = bonds[bond_idx, 1] if bonds[bond_idx, 0] == atom else bonds[bond_idx, 0]

			# Check if both atoms are in the connected component
			if other in cc_atom_ids:
				new_sbond_arr[atom] += 1
				new_bond_mask_arr[bond_idx] = 1  # Use 1 instead of True for uint8_t

	return new_sbond_arr, new_bond_mask_arr
