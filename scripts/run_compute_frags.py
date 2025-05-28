import numpy as np
import argparse
import torch_geometric as pyg
from pprint import pprint

import fragnnet.utils.frag_utils as frag_utils
from fragnnet.utils.misc_utils import timeit, booltype, get_pyg_memory_usage

def run_frag(smiles, **kwargs):
	
	print(f"> smiles = {smiles}")
	pprint(kwargs)
	
	mol_d = frag_utils.extract_mol_info(smiles)
	num_nodes = mol_d["atom_mask_arr"].shape[0]
	print(f"> num heavy atoms {num_nodes}")
	dag_d = {}
	dag_d = timeit(frag_utils.compute_dags)(mol_d,**kwargs)

	dag = dag_d["dag"]
	
	print("> dag")
	dag_pyg = dag_d["dag"]
	dag_nn = dag_d["dag_num_nodes"]
	dag_ne = dag_d["dag_num_edges"]
	dag_nn_nb = dag_d["dag_num_nodes_nb"]
	
	dag_indeg = pyg.utils.degree(dag.edge_index[1]).numpy()
	dag_outdeg = pyg.utils.degree(dag.edge_index[0]).numpy()
	dag_memory = get_pyg_memory_usage(dag)

	print(f"reached_depth = {dag_d['reached_depth']}")
	print(f"num_nodes = {dag_nn}, num_edges = {dag_ne}, edge_frac = {2*dag_ne/(dag_nn*(dag_nn-1))}")
	print(f"num_nodes_nb = {dag_nn_nb}")
	print(f"avg_indeg = {np.mean(dag_indeg)}, std_indeg = {np.std(dag_indeg)}")
	print(f"avg_outdeg = {np.mean(dag_outdeg)}, std_outdeg = {np.std(dag_outdeg)}")
	print(f"num unique forumla = {len(dag_d['idx_to_formula'])}")
	print(f"node feature size = {dag_d['node_feature_size']}, edge feature size = {dag_d['edge_feature_size']}")
	print(f"node edges by depth: {dag_d['dag_num_edges_by_depth']}")
	print(f"node node by depth: {dag_d['dag_num_nodes_by_depth']}")
	print(f"is directed = {dag_d['is_directed']}")
	print(f"memory = {dag_memory/1e6} MB")
	print(f"dag_pyg = {dag_pyg}")
	return dag_d

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--compound_idx",type=int,required=False)
	parser.add_argument("--compound_smiles",type=str,required=False)
	parser.add_argument("--max_depth",type=int,default=4)
	parser.add_argument("--h_prior",type=booltype,default=True)
	parser.add_argument("--max_h_transfer",type=int,default=4)
	parser.add_argument("--isotopes",type=booltype,default=True)
	parser.add_argument("--nb_isomorphic",type=booltype,default=False)
	parser.add_argument("--b_isomorphic",type=booltype,default=False)
	parser.add_argument("--max_iterations",type=int,default=2)
	args = parser.parse_args()

	np.random.seed(420)

	IDX_TO_SMILES = {
		# ethanol
		0: "CCO",
		# benzene
		1: "c1ccccc1",
		# aspirin
		2: "CC(=O)OC1=CC=CC=C1C(=O)O",
		# testosterone
		3: "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
		# geddic acid
		4: "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
		# methionine
		5: "CSCCC(C(=O)O)N",
		# sulfanilamide
		6: "C1=CC(=CC=C1N)S(=O)(=O)N",
		# methoxydiphenylphosphine
		7: "COP(C1=CC=CC=C1)C2=CC=CC=C2",
		# phosphomycin
		8: "CC1C(O1)P(=O)(O)O",
		# 1H-Benzotriazole
		9: 'c1ccc2[nH]nnc2c1',
		# big one (47 heavy atoms)
		10: 'CC(=O)NC1C(OC2CCC3(C)C(CCC4(C)C3CC=C3C5CC(C)(C)CCC5(C(=O)O)CCC34C)C2(C)C)OC(CO)C(O)C1O',
		# big one (60 heavy atoms)
		11: 'CC(=O)NC1C(OC2OC(CC(O)C3OC(n4ccc(=O)[nH]c4=O)C(O)C3O)C(O)C(O)C2NC(=O)C=CCCCCCCCCCCCC(C)C)OC(CO)C(O)C1O',
		# another big one
		12: 'CC(=O)OC12COC1CC(O)C1(C)C(=O)C(=O)C3C(C)C(OC(=O)C(O)C(NC(=O)c4ccccc4)c4ccccc4)CC(O)(C(OC(=O)c4ccccc4)C21)C3(C)C',
		# C20 fullerene
		13: 'C12=C3C4=C5C6=C3C7=C1C8=C9C2=C4C1=C5C2=C6C7=C8C2=C91',
		# C60 fullerene
		14: 'c12c3c4c5c1c1c6c7c2c2c8c3c3c9c4c4c%10c5c5c1c1c6c6c%11c7c2c2c7c8c3c3c8c9c4c4c9c%10c5c5c1c1c6c6c%11c2c2c7c3c3c8c4c4c9c5c1c1c6c2c3c41',
		# C70 fullerene
		15: 'C12=C3C4=C5C6=C7C8=C9C%10=C%11C%12=C%13C%10=C%10C8=C5C1=C%10C1=C%13C5=C8C1=C2C1=C3C2=C3C%10=C%13C%14=C3C1=C8C1=C3C5=C%12C5=C8C%11=C%11C9=C7C7=C9C6=C4C2=C2C%10=C4C(=C29)C2=C6C(=C8C8=C9C6=C4C%13=C9C(=C%141)C3=C85)C%11=C27'
	}

	assert not (args.compound_idx is None and args.compound_smiles is None), "must provide either --compound_idx or --compound_smiles"
	assert not (args.compound_idx is not None and args.compound_smiles is not None), "must provide only one of --compound_idx or --compound_smiles"
	if args.compound_idx is not None:
		smiles = IDX_TO_SMILES[args.compound_idx]
	else:
		smiles = args.compound_smiles

	frag_kw_d = dict(
		max_depth=args.max_depth,
		h_prior=args.h_prior,
		max_h_transfer=args.max_h_transfer,
		frag_max_time=None,
		isotopes=args.isotopes,
		nb_isomorphic=args.nb_isomorphic,
		b_isomorphic=args.b_isomorphic,
		max_iterations=args.max_iterations
	)

	dag_d = run_frag(smiles, **frag_kw_d)
	
