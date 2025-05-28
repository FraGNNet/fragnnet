import torch as th
import torch.nn as nn
import numpy as np
from pprint import pprint
from pyteomics.mass import Composition

from fragnnet.utils.frag_utils import get_node_feats, get_edge_feats, th_long_to_mask
from fragnnet.utils.misc_utils import scatter_logsumexp, scatter_logsoftmax, scatter_reduce, scatter_masked_softmax, scatter_masked_logsumexp
from fragnnet.utils.feat_utils import get_mol_feats_sizes, get_mol_fp_size
from fragnnet.utils.nn_utils import *
from fragnnet.form_embedder import get_embedder
from fragnnet.utils.misc_utils import check_pyg_compile, LOG_ZERO, check_pyg_full_compile
from fragnnet.utils.spec_utils import transform_ce, batched_bin_func
from fragnnet.utils.data_utils import combine_formulae
from fragnnet.utils.formula_utils import PREC_TYPE_TO_FORMULA_DIFF


class CEModel:
	""" class for handling collision engery embedding
	"""
	def _ce_init(
		self,
		int_embedder,
		ce_insert_location: str,
		ce_insert_type: str,
		ce_insert_merge: bool,
		ce_insert_size: int,
		ce_mean: float,
  		ce_std: float,
    	ce_max: float,):

		# ce stuff
		assert ce_insert_type in ["id","lin","embed","bin"]
		assert ce_insert_location in ["none","mol","frag","mlp"]
		self.ce_insert_type = ce_insert_type
		self.ce_insert_location = ce_insert_location
		self.ce_insert_merge = ce_insert_merge
		self.ce_insert_size = ce_insert_size
		self.int_embedder = int_embedder
		self.ce_max = ce_max
		self.ce_mean = ce_mean
		self.ce_std = ce_std
		self._ce_location_check()
		self._setup_ce()

	def _ce_location_check(self):

		raise NotImplementedError

	def _setup_ce(self):

		# embedding type
		if self.ce_insert_type == "id":
			def ce_transform(ce):
				ce = transform_ce(ce, self.ce_mean, self.ce_std)
				ce = ce.reshape(-1,1)
				ce = th.repeat_interleave(ce, self.ce_insert_size, dim=1)
				return ce
			ce_embedder = nn.Identity()
			ce_input_dim = self.ce_insert_size
		elif self.ce_insert_type == "lin":
			def ce_transform(ce):
				ce = transform_ce(ce, self.ce_mean, self.ce_std)
				ce = ce.reshape(-1,1)
				return ce
			ce_embedder = nn.Linear(1,self.ce_insert_size)
			ce_input_dim = self.ce_insert_size
		elif self.ce_insert_type == "embed":
			def ce_transform(ce):
				ce = th.clamp(ce, min=0, max=int(self.ce_max)-1)
				ce = th.round(ce, decimals=0).long()
				ce = ce.reshape(-1,1)
				return ce
			embedder = get_embedder(self.int_embedder, max_count_int=int(self.ce_max))
			ce_embedder = nn.Sequential(
				embedder,
				nn.Linear(embedder.num_dim,self.ce_insert_size)
			)	
			ce_input_dim = self.ce_insert_size
		elif self.ce_insert_type == "bin":
			def ce_transform(ce):
				ce = th.clamp(ce, min=0, max=int(self.ce_max)-10)
				ce = th.round(ce, decimals=-1).long() // 10
				ce = F.one_hot(ce, num_classes=int(self.ce_max)//10).float()
				return ce
			ce_embedder = nn.Linear(int(self.ce_max)//10,self.ce_insert_size)
			ce_input_dim = self.ce_insert_size
		# location
		if self.ce_insert_location == "mol":
			ce_mol_input_dim = ce_input_dim
			ce_mlp_input_dim = 0
		elif self.ce_insert_location == "mlp":
			ce_mol_input_dim = 0
			ce_mlp_input_dim = ce_input_dim
		else:
			assert self.ce_insert_location == "none"
			ce_mol_input_dim = 0
			ce_mlp_input_dim = 0
		self.ce_transform = ce_transform
		self.ce_embedder = ce_embedder
		self.ce_mol_input_dim = ce_mol_input_dim
		self.ce_mlp_input_dim = ce_mlp_input_dim

	def embed_ce(self, ce, ce_batch_idxs, batch_size):

		if self.ce_insert_location != "none":
			ce_embed = self.ce_transform(ce)
			ce_embed = self.ce_embedder(ce_embed)
			# possibly merge the embeddings
			if self.ce_insert_merge:
				ce_embed = scatter_reduce(
					src=ce_embed,
					index=ce_batch_idxs.unsqueeze(1).expand_as(ce_embed),
					reduce="mean",
					dim_size=batch_size,
					include_self=False
				)
		else:
			ce_embed = None
		return ce_embed

class PrecModel:

	def _prec_init(
		self,
		prec_insert_location: str,
		prec_insert_size: int,
		prec_num_types: int):

		self.prec_insert_location = prec_insert_location
		self.prec_embedder = nn.Embedding(prec_num_types+1, prec_insert_size)	
		prec_dim = prec_insert_size

		self._prec_location_check()

		if self.prec_insert_location == "mol":
			prec_mol_input_dim = prec_dim
			prec_mlp_input_dim = 0
		elif self.prec_insert_location == "mlp":
			prec_mol_input_dim = 0
			prec_mlp_input_dim = prec_dim
		else:
			assert self.prec_insert_location == "none"
			prec_mol_input_dim = 0
			prec_mlp_input_dim = 0
		self.prec_mol_input_dim = prec_mol_input_dim
		self.prec_mlp_input_dim = prec_mlp_input_dim

	def _prec_location_check(self):

		raise NotImplementedError

	def embed_prec(self, prec_type):

		if self.prec_insert_location != "none":
			prec_embed = self.prec_embedder(prec_type)
		else:
			prec_embed = None
		return prec_embed

class InstModel:

	def _inst_init(
		self,
		inst_insert_location: str,
		inst_insert_size: int,
		inst_num_types: int):

		self.inst_insert_location = inst_insert_location
		self.inst_embedder = nn.Embedding(inst_num_types+1, inst_insert_size)	
		inst_dim = inst_insert_size

		self._inst_location_check()

		if self.inst_insert_location == "mol":
			inst_mol_input_dim = inst_dim
			inst_mlp_input_dim = 0
		elif self.inst_insert_location == "mlp":
			inst_mol_input_dim = 0
			inst_mlp_input_dim = inst_dim
		else:
			assert self.inst_insert_location == "none"
			inst_mol_input_dim = 0
			inst_mlp_input_dim = 0
		self.inst_mol_input_dim = inst_mol_input_dim
		self.inst_mlp_input_dim = inst_mlp_input_dim

	def _inst_location_check(self):

		raise NotImplementedError

	def embed_inst(self, inst_type):

		if self.inst_insert_location != "none":
			inst_embed = self.inst_embedder(inst_type)
		else:
			inst_embed = None
		return inst_embed

class FragGNNModel(nn.Module, CEModel, PrecModel, InstModel):

	def __init__(
		self,
		num_depth: int,
		num_hs: int,
		num_elements: int,
		int_embedder: str,
		int_embedder_tight: bool,
		mol_node_feats: list[str],
		mol_edge_feats: list[str],
		mol_pe_embed_k: int,
		mol_hidden_size: int,
		mol_num_layers: int,
		mol_gnn_type: str,
		mol_normalization: str,
		mol_dropout: float,
		mol_pool_type: str,
		frag_node_feats: list[str],
		frag_edge_feats: list[str],
		frag_hidden_size: int,
		frag_num_layers: int,
		frag_gnn_type: str,
		frag_normalization: str,
		frag_dropout: float,
		frag_pool_type: str,
		frag_embed_combine: str,
		frag_pool_combine: str,
		mlp_output_format: str,
		mlp_hidden_size: int,
		mlp_normalization: str,
		mlp_dropout: float,
		mlp_num_layers: int,
		mlp_use_residuals: bool,
		cc_interstage_type: str,
		nb_iso: bool,
		skip_edge_loss: bool,
		mask_null_formula: bool,
		predict_oos: bool,
		bin_output: bool,
		mz_bin_res: float,
		mz_max: float,
		ce_insert_location: str,
		ce_insert_type: str,
		ce_insert_merge: bool,
		ce_insert_size: int,
  		ce_max: float,
		ce_mean: float,
		ce_std: float,
		prec_insert_location: str,
		prec_insert_size: int,
		prec_types: list[str],
		inst_insert_location: str,
		inst_insert_size: int,
		inst_types: list[str],
		output_formula_str: bool):

		# nn.Module init
		super().__init__()
		
		self._ce_init(
			int_embedder=int_embedder,
			ce_insert_location=ce_insert_location,
			ce_insert_type=ce_insert_type,
			ce_insert_merge=ce_insert_merge,
			ce_insert_size=ce_insert_size,
			ce_max=ce_max,
			ce_mean=ce_mean,
			ce_std=ce_std
		)

		self._prec_init(
			prec_insert_location=prec_insert_location,
			prec_insert_size=prec_insert_size,
			prec_num_types=len(prec_types)
		)

		self._inst_init(
			inst_insert_location=inst_insert_location,
			inst_insert_size=inst_insert_size,
			inst_num_types=len(inst_types)
		)

		self.num_depth = num_depth
		self.num_hs = num_hs
		self.num_elements = num_elements

		# calculate node/edge feats sizes
		self.mol_node_feats = mol_node_feats
		self.mol_edge_feats = mol_edge_feats
		self.mol_pe_embed_k = mol_pe_embed_k
		self._compute_mol_feats_sizes()

		# setup mol gnn
		self.mol_node_feats_size += self.ce_mol_input_dim + self.prec_mol_input_dim + self.inst_mol_input_dim
		mol_kwargs = {
			"node_feats_size": self.mol_node_feats_size,
			"edge_feats_size": self.mol_edge_feats_size,
			"hidden_size": mol_hidden_size,
			"num_layers": mol_num_layers,
			"gnn_type": mol_gnn_type,
			"dropout": mol_dropout,
			"normalization": mol_normalization,
		}
		
		# Mol GNN
		self.mol_embedder = GNN(**mol_kwargs)
		self.mol_pool_type = mol_pool_type
		self.mol_pool = build_pool_module(mol_pool_type,mol_hidden_size)
		if int_embedder_tight:
			formula_d = {"max_count_int": 255}
			depth_d = {"max_count_int": num_depth+1}
			complement_d = {"max_count_int": 2}
		else:
			formula_d = depth_d = complement_d = {}
		self.formula_embedder = get_embedder(int_embedder,**formula_d)
		self.depth_embedder = get_embedder(int_embedder,**depth_d)
		self.complement_embedder = get_embedder(int_embedder,**complement_d)
		self.frag_node_feats = frag_node_feats
		self.frag_edge_feats = frag_edge_feats
		self._compute_frag_feats_sizes()

		# define interstage
		assert cc_interstage_type in ["add","sub","linear","direct"]
		self.cc_interstage_type = cc_interstage_type
		if self.cc_interstage_type == "linear":
			self.cc_interstage = nn.Linear(mol_hidden_size * 2, mol_hidden_size)

		frag_kwargs = {
			"node_feats_size": self.frag_node_feats_size,
			"edge_feats_size": self.frag_edge_feats_size,
			"hidden_size": frag_hidden_size,
			"num_layers": frag_num_layers,
			"gnn_type": frag_gnn_type,
			"dropout": frag_dropout,
			"normalization": frag_normalization
		}

		self.frag_embedder = GNN(**frag_kwargs)
		self.frag_pool_type = frag_pool_type
		self.frag_pool = build_pool_module(frag_pool_type,frag_hidden_size)
		self.frag_embed_combine = frag_embed_combine
		self.frag_pool_combine = frag_pool_combine
		self.mlp_output_format = mlp_output_format

		# determine mlp input dims
		if self.frag_embed_combine == "cat":
			mlp_input_dim = 2*self.frag_embedder.hidden_size
		else:
			assert self.frag_embed_combine == "avg", self.frag_embed_combine
			mlp_input_dim = self.frag_embedder.hidden_size
		mlp_input_dim += self.ce_mlp_input_dim + self.prec_mlp_input_dim + self.inst_mlp_input_dim

		if self.mlp_output_format in ["formula","node_formula"]:
			formula_mlp_kwargs = {
				"input_size": mlp_input_dim,
				"output_size": 2*self.num_hs+1,
				"hidden_size": mlp_hidden_size,
				"num_layers": mlp_num_layers,
				"dropout": mlp_dropout,
				"use_residuals": mlp_use_residuals,
				"normalization": mlp_normalization
			}
			self.formula_module = MLPBlocks(**formula_mlp_kwargs)

		if self.mlp_output_format in ["node_formula"]:
			node_mlp_kwargs = {
				"input_size": 2*self.frag_embedder.hidden_size,
				"output_size": 1,
				"hidden_size": mlp_hidden_size,
				"num_layers": mlp_num_layers,
				"dropout": mlp_dropout,
				"use_residuals": mlp_use_residuals,
				"normalization": mlp_normalization
			}
			self.node_module = MLPBlocks(**node_mlp_kwargs)
		else:
			self.node_module = None
		
		self.predict_oos = predict_oos
		if self.predict_oos:
			oos_mlp_kwargs = {
				"input_size": self.mol_embedder.hidden_size+self.frag_embedder.hidden_size,
				"output_size": 1,
				"hidden_size": mlp_hidden_size,
				"num_layers": mlp_num_layers,
				"dropout": mlp_dropout,
				"use_residuals": mlp_use_residuals,
				"normalization": "none" # never normalized
			}
			self.oos_module = MLPBlocks(**oos_mlp_kwargs)
		else:
			self.oos_module = None

		self.skip_edge_loss = skip_edge_loss
		self.mask_null_formula = mask_null_formula
		self.nb_iso = nb_iso
		self.bin_output = bin_output
		self.mz_bin_res = mz_bin_res
		self.mz_max = mz_max
		self.output_formula_str = output_formula_str

		if self.bin_output:
			self.mz_bins = th.arange(mz_bin_res,mz_max+mz_bin_res,mz_bin_res)

		# this is required
		assert "h_formulae_idx" in self.frag_node_feats

	def _ce_location_check(self):

		assert not self.ce_insert_location == "frag", "ce_insert_location=frag not supported"

	def _prec_location_check(self):

		assert not self.prec_insert_location == "frag", "prec_insert_location=frag not supported"

	def _inst_location_check(self):

		assert not self.inst_insert_location == "frag", "inst_insert_location=frag not supported"

	def _compute_mol_feats_sizes(self):
		""" method compute mol feature size
			these features don't rely on any model parameters
		"""
		self.mol_node_feats_size, self.mol_edge_feats_size = get_mol_feats_sizes(
			self.mol_node_feats, 
			self.mol_edge_feats, 
			self.mol_pe_embed_k
		)

	def _compute_frag_feats_sizes(self):
		""" method compute frag-graph feature size
			these features do depend on model parameters
		"""
		# nodes
		self.frag_node_feats_size = 0
		if "cc" in self.frag_node_feats:
			self.frag_node_feats_size += self.mol_embedder.hidden_size
		if "base_formula" in self.frag_node_feats:
			self.frag_node_feats_size += self.num_elements*self.formula_embedder.num_dim
		if "depth" in self.frag_node_feats:
			self.frag_node_feats_size += self.num_depth*self.depth_embedder.num_dim
		# edges
		self.frag_edge_feats_size = 0
		if "cc" in self.frag_edge_feats:
			self.frag_edge_feats_size += self.mol_embedder.hidden_size
		if "base_formula" in self.frag_edge_feats:
			self.frag_edge_feats_size += self.num_elements*self.formula_embedder.num_dim
		if "complement" in self.frag_edge_feats:
			self.frag_edge_feats_size += self.complement_embedder.num_dim

	def get_compile(self, **kwargs):

		if check_pyg_full_compile():
			return th.compile(self,**kwargs)
		else:
			self.compile_submodules(**kwargs)
			return self

	def compile_submodules(self,**kwargs):
		""" pyg does not support dynamic shape compiling """
		self.formula_embedder = th.compile(self.formula_embedder,**kwargs)
		self.depth_embedder = th.compile(self.depth_embedder,**kwargs)
		self.complement_embedder = th.compile(self.complement_embedder,**kwargs)
		if hasattr(self,"ce_embedder"):
			self.ce_embedder = th.compile(self.ce_embedder,**kwargs)
		if hasattr(self,"m_ce_embedder"):
			self.m_ce_embedder = th.compile(self.m_ce_embedder,**kwargs)
		if check_pyg_compile():
			self.mol_embedder = pyg.compile(self.mol_embedder,**kwargs)
			self.frag_embedder = pyg.compile(self.frag_embedder,**kwargs)

	def forward(
		self, 
		mol_pyg: pyg.data.Data,
		frag_pyg: pyg.data.Data,
		mol_num_nodes: th.Tensor,
		frag_num_nodes: th.Tensor,
		frag_formula_peak_idxs: th.Tensor,
		frag_formula_peak_mzs: th.Tensor,
		frag_formula_peak_probs: th.Tensor,
		frag_formula_sizes: th.Tensor,
		frag_formula_cumsizes: th.Tensor,
		frag_formula_peak_sizes: th.Tensor,
		frag_formula_str: np.ndarray = None,
		spec_ce: th.Tensor = None,
		spec_ce_batch_idxs: th.Tensor = None,
		spec_prec_type: th.Tensor = None,
		spec_inst_type: th.Tensor = None,
		spec_prec_type_str: np.ndarray = None,
		**kwargs
	):
		"""forward methods for joint predictor

		Args:
			mol_pyg (pyg.data.Data): molecule pyg data object
			frag_pyg (pyg.data.Data): fragmentation graph pyg data object
			mol_num_nodes (th.Tensor): number of nodes in molecule graph
			frag_num_nodes (th.Tensor): number of nodes in fragmentation graph
			frag_formula_peak_idxs (th.Tensor): _description_
			frag_formula_peak_mzs (th.Tensor): _description_
			frag_formula_peak_probs (th.Tensor): _description_
			frag_formula_sizes (th.Tensor): _description_
			frag_formula_cumsizes (th.Tensor): _description_
			frag_formula_peak_sizes (th.Tensor): _description_

		Returns:
			_type_: _description_
		"""

		# mol_x: mol level node feature matrix
		# mol_edge_index: mol graph connectivity in COO format with shape [2, num_edges]
		# edge_attr: mol graph edge feature matrix with shape [num_edges, num_edge_features]
		# batch: sample idx repsect to current batch
		mol_x, mol_edge_index, mol_edge_attr, mol_batch = mol_pyg.x, mol_pyg.edge_index, mol_pyg.edge_attr, mol_pyg.batch
		# frag_x: frag-graph level node feature matrix
		# frag_edge_index: frag graph connectivity in COO format with shape [2, num_edges]
		# frag_edge_attr: frag graph edge feature matrix with shape [num_edges, num_edge_features]
		# batch: sample idx repsect to current batch
		frag_x, frag_edge_index, frag_edge_attr, frag_batch = frag_pyg.x, frag_pyg.edge_index, frag_pyg.edge_attr, frag_pyg.batch

		device = mol_num_nodes.device
		# int_dtype = mol_edge_index.dtype
		float_dtype = mol_edge_attr.dtype
		frag_node_feat_idxs = frag_pyg.node_feat_idxs[0]
		frag_edge_feat_idxs = frag_pyg.edge_feat_idxs[0]
		batch_frag_num_nodes = frag_x.shape[0]
		batch_frag_num_edges = frag_edge_index.shape[1]
		batch_frag_num_formulae = frag_formula_cumsizes[-1]
		batch_size = frag_batch[-1]+1
		
		# get ce value
		ce = spec_ce
		ce_batch_idxs = spec_ce_batch_idxs
		ce_embed = self.embed_ce(ce, ce_batch_idxs, batch_size)
		# get prec value
		prec_embed = self.embed_prec(spec_prec_type)
		# get inst value
		inst_embed = self.embed_inst(spec_inst_type)

		if self.ce_insert_location == "mol":
			mol_ce_embed = th.repeat_interleave(ce_embed,th.unique(mol_batch,return_counts=True)[1],dim=0)
			mol_x = th.cat([mol_x,mol_ce_embed],dim=1)
		if self.prec_insert_location == "mol":
			mol_prec_embed = th.repeat_interleave(prec_embed,th.unique(mol_batch,return_counts=True)[1],dim=0)
			mol_x = th.cat([mol_x,mol_prec_embed],dim=1)
		if self.inst_insert_location == "mol":
			mol_inst_embed = th.repeat_interleave(inst_embed,th.unique(mol_batch,return_counts=True)[1],dim=0)
			mol_x = th.cat([mol_x,mol_inst_embed],dim=1)

		# get per-atom embeddings
		mol_embed_gnn = self.mol_embedder(
			mol_x,
			mol_batch,
			mol_edge_index,
			mol_edge_attr
		)
		mol_embed_gnn_pool = self.mol_pool(mol_embed_gnn,mol_batch)

		# process dag
		# create interstage
		frag_ndata, frag_edata = [], []
		# node atom embeddings
		if "cc" in self.frag_node_feats:
			frag_node_mask = th_long_to_mask(get_node_feats(frag_x,frag_node_feat_idxs,"cc").to(device))
			frag_node_mask_idxs = th.nonzero(frag_node_mask).long()
			frag_node_offsets = mol_num_nodes[th.bucketize(frag_node_mask_idxs[:,0],frag_num_nodes,right=True)-1]
			frag_node_mask_idxs[:,1] = frag_node_mask_idxs[:,1] + frag_node_offsets
			frag_node_mask_embed = scatter_reduce(
				src=mol_embed_gnn[frag_node_mask_idxs[:,1]],
				index=frag_node_mask_idxs[:,0:1].expand(-1,mol_embed_gnn.shape[1]),
				reduce="sum",
				dim_size=batch_frag_num_nodes
			)
			# frag_node_mask_embed can be one of following
			# 1. frag_node_mask_embed plus mol_embed_gnn_pool
			# 2. frag_node_mask_embed only, with out pooled embed for entire mol
			# 3. a linear layer between masked embed and pooled embed
			if self.cc_interstage_type == "add":
				frag_node_mask_embed = frag_node_mask_embed + mol_embed_gnn_pool[frag_batch]
			elif self.cc_interstage_type == "sub":
				frag_node_mask_embed = frag_node_mask_embed - mol_embed_gnn_pool[frag_batch]
			elif self.cc_interstage_type == "linear":
				frag_node_mask_embed = self.cc_interstage(th.cat([frag_node_mask_embed,mol_embed_gnn_pool[frag_batch]], dim = 1))
			else:
				assert self.cc_interstage_type == "direct", self.cc_interstage_type
				frag_node_mask_embed = frag_node_mask_embed
			frag_ndata.append(frag_node_mask_embed)
		# edge atom embeddings
		if "cc" in self.frag_edge_feats:
			# connected competents
			frag_edge_mask = th_long_to_mask(get_edge_feats(frag_edge_attr,frag_edge_feat_idxs,"cc").to(device))
			frag_edge_mask_idxs = th.nonzero(frag_edge_mask).long()
			frag_edge_node_idxs = frag_edge_index[0][frag_edge_mask_idxs[:,0]]
			frag_edge_offsets = mol_num_nodes[th.bucketize(frag_edge_node_idxs,frag_num_nodes,right=True)-1]
			frag_edge_mask_idxs[:,1] = frag_edge_mask_idxs[:,1] + frag_edge_offsets
			frag_edge_mask_embed = scatter_reduce(
				src=mol_embed_gnn[frag_edge_mask_idxs[:,1]],
				index=frag_edge_mask_idxs[:,0:1].expand(-1,mol_embed_gnn.shape[1]),
				reduce="sum",
				dim_size=batch_frag_num_edges
			)
			frag_edata.append(frag_edge_mask_embed)
		# node formulae
		if "base_formula" in self.frag_node_feats:
			frag_node_formula = self.formula_embedder(
				get_node_feats(frag_x,frag_node_feat_idxs,"base_formula").reshape(batch_frag_num_nodes,-1)
			)
			frag_ndata.append(frag_node_formula)
		# edge formulae
		if "base_formula" in self.frag_edge_feats:
			frag_edge_formula = self.formula_embedder(
				get_edge_feats(frag_edge_attr,frag_edge_feat_idxs,"base_formula").reshape(batch_frag_num_edges,-1)
			)
			frag_edata.append(frag_edge_formula)
		# node depth
		if "depth" in self.frag_node_feats:
			frag_depth = self.depth_embedder(
				get_node_feats(frag_x,frag_node_feat_idxs,"depth").reshape(batch_frag_num_nodes,-1)
			)
			frag_ndata.append(frag_depth)
		# edge complement
		if "complement" in self.frag_edge_feats:
			frag_edge_complement = self.complement_embedder(
				get_edge_feats(frag_edge_attr,frag_edge_feat_idxs,"complement").reshape(batch_frag_num_edges,-1)
			)
			frag_edata.append(frag_edge_complement)
		# empty feats check
		if len(frag_ndata) == 0:
			assert self.frag_node_feats_size == 0, self.frag_node_feats_size
			frag_ndata.append(th.zeros([batch_frag_num_nodes,0],dtype=float_dtype,device=device))
		if len(frag_edata) == 0:
			assert self.frag_edge_feats_size == 0, self.frag_edge_feats_size
			frag_edata.append(th.zeros([batch_frag_num_edges,0],dtype=float_dtype,device=device))

		# get output formula aggregation 
		frag_node_batch_idxs = frag_batch
		frag_formula_batch_idxs = th.repeat_interleave(th.arange(batch_size,device=device),frag_formula_cumsizes[1:]-frag_formula_cumsizes[:-1])
		frag_node_offsets = frag_formula_cumsizes[frag_node_batch_idxs]
		frag_joint_formula_idxs = (get_node_feats(frag_x,frag_node_feat_idxs,"h_formulae_idx")+frag_node_offsets.unsqueeze(-1)).flatten()
		frag_formula_idxs = th.unique(frag_joint_formula_idxs)
		# remove formulae if necessary
		frag_joint_formula_idxs = frag_joint_formula_idxs.reshape(batch_frag_num_nodes,-1)
		num_hs_diff = (frag_joint_formula_idxs.shape[1]-1)//2 - self.num_hs
		assert num_hs_diff >= 0 and num_hs_diff <= (frag_joint_formula_idxs.shape[1]-1)//2, num_hs_diff
		if num_hs_diff > 0:
			# remove formulae
			frag_joint_formula_idxs = frag_joint_formula_idxs[:,:-2*num_hs_diff].flatten()
			frag_joint_formula_idxs_un, frag_joint_formula_idxs_inv = th.unique(th.cat([frag_joint_formula_idxs,frag_formula_cumsizes[:-1]],dim=0),return_inverse=True)
			frag_joint_formula_idxs_inv = frag_joint_formula_idxs_inv[:frag_joint_formula_idxs.shape[0]]
			batch_frag_num_formulae = frag_joint_formula_idxs_un.shape[0]
			frag_formula_batch_idxs = frag_formula_batch_idxs[frag_joint_formula_idxs_un]
			frag_joint_formula_idxs = th.arange(batch_frag_num_formulae,device=device)[frag_joint_formula_idxs_inv]
			frag_formula_peak_mask = th.isin(frag_formula_idxs[~th.isin(frag_formula_idxs,frag_formula_cumsizes[:-1])],frag_joint_formula_idxs_un)
			frag_formula_idxs = th.arange(batch_frag_num_formulae,device=device)
			frag_formula_sizes = scatter_reduce(
				th.ones_like(frag_joint_formula_idxs_un),
				frag_formula_batch_idxs,
				reduce="sum",
				dim_size=batch_size
			)
			assert not th.any(frag_formula_sizes <= 1), frag_formula_sizes
			frag_formula_cumsizes = th.cat([th.zeros([1],device=device,dtype=frag_formula_sizes.dtype),frag_formula_sizes],dim=0)
			frag_formula_cumsizes = th.cumsum(frag_formula_cumsizes,dim=0)
			frag_node_offsets = frag_formula_cumsizes[frag_node_batch_idxs]
			# peak stuff
			frag_formula_peak_idxs = frag_formula_idxs[~th.isin(frag_formula_idxs,frag_formula_cumsizes[:-1])]
			frag_formula_peak_idxs = frag_formula_peak_idxs-frag_formula_cumsizes[:-1][frag_formula_batch_idxs[~th.isin(frag_formula_idxs,frag_formula_cumsizes[:-1])]]
			frag_formula_peak_probs = frag_formula_peak_probs[frag_formula_peak_mask]
			frag_formula_peak_mzs = frag_formula_peak_mzs[frag_formula_peak_mask]
			frag_formula_peak_sizes = frag_formula_sizes - 1
		else:
			# no removal required
			frag_joint_formula_idxs = frag_joint_formula_idxs.flatten()

		# get isomorphism aggregation
		if self.nb_iso:
			frag_nb_idxs = get_node_feats(frag_x,frag_node_feat_idxs,"nb_iso_idx").flatten()
			frag_nb_offsets = scatter_reduce(
				frag_nb_idxs,
				frag_node_batch_idxs,
				reduce="amax",
				dim_size=batch_size
			)
			frag_nb_offsets = th.cat(
				[
					th.zeros([1],dtype=frag_nb_offsets.dtype,device=device),
					frag_nb_offsets+1
				], dim=0
			)
			frag_nb_offsets = th.cumsum(frag_nb_offsets,dim=0)
			batch_frag_nb_num_nodes = frag_nb_offsets[-1].item()
			frag_nb_offsets = th.gather(
				input=frag_nb_offsets[:-1],
				index=frag_node_batch_idxs,
				dim=0
			)
			frag_nb_idxs = frag_nb_idxs + frag_nb_offsets
			assert th.max(frag_nb_idxs) < batch_frag_nb_num_nodes, (th.max(frag_nb_idxs),batch_frag_nb_num_nodes)
			frag_nb_un_idxs, frag_nb_inv_idxs = th.unique(frag_nb_idxs,return_inverse=True)

		# assemble all features for dag
		# concatenate everything 
		frag_x_embed = th.cat(frag_ndata,dim=-1)
		# concatenate everything 
		frag_edge_attr_embed = th.cat(frag_edata,dim=-1)

		# define frag network
		frag_embed_gnn = self.frag_embedder(
			frag_x_embed,
			frag_node_batch_idxs,
			frag_edge_index,
			frag_edge_attr_embed
		)
		frag_embed_node = self.frag_embedder.input_project(frag_x_embed)
		frag_embed_gnn_pool = self.frag_pool(frag_embed_gnn,frag_batch)
		frag_embed_node_pool = self.frag_pool(frag_embed_node,frag_batch)
		if self.frag_pool_combine == "subtract":
			frag_embed_gnn = frag_embed_gnn - frag_embed_gnn_pool[frag_batch]
			frag_embed_node = frag_embed_node - frag_embed_node_pool[frag_batch]
		elif self.frag_pool_combine == "add":
			frag_embed_gnn = frag_embed_gnn + frag_embed_gnn_pool[frag_batch]
			frag_embed_node = frag_embed_node + frag_embed_node_pool[frag_batch]
		else:
			assert self.frag_pool_combine == "none", self.frag_pool_combine

		# get frag dag embedding
		if self.frag_embed_combine == "cat":
			frag_embed_parts = [frag_embed_gnn, frag_embed_node]
		else:
			assert self.frag_embed_combine == "avg", self.frag_embed_combine
			frag_embed_parts = [0.5*frag_embed_gnn + 0.5*frag_embed_node]
		
		if self.ce_insert_location == "mlp":
			mlp_ce_embed = th.repeat_interleave(ce_embed,th.unique(frag_node_batch_idxs,return_counts=True)[1],dim=0)
			frag_embed_parts.append(mlp_ce_embed)
		if self.prec_insert_location == "mlp":
			mlp_prec_embed = th.repeat_interleave(prec_embed,th.unique(frag_node_batch_idxs,return_counts=True)[1],dim=0)
			frag_embed_parts.append(mlp_prec_embed)
		if self.inst_insert_location == "mlp":
			mlp_inst_embed = th.repeat_interleave(inst_embed,th.unique(frag_node_batch_idxs,return_counts=True)[1],dim=0)
			frag_embed_parts.append(mlp_inst_embed)

		frag_embed = th.cat(frag_embed_parts, dim=1)

		frag_joint_batch_idxs = th.repeat_interleave(frag_node_batch_idxs,2*self.num_hs+1)
		frag_joint_mask = (~th.isin(frag_joint_formula_idxs,frag_formula_cumsizes[:-1])).float()
		h_counts = th.zeros([2*self.num_hs+1],device=device,dtype=frag_joint_batch_idxs.dtype)
		h_counts[1+2*th.arange(self.num_hs,device=device)] = -th.arange(1,self.num_hs+1,device=device)
		h_counts[2+2*th.arange(self.num_hs,device=device)] = th.arange(1,self.num_hs+1,device=device)
		frag_joint_h_counts = h_counts.repeat(frag_node_batch_idxs.shape[0])

		if self.mlp_output_format == "formula":

			# log p(f,n)
			frag_joint_logits = self.formula_module(frag_embed)
			assert frag_joint_logits.shape[1] == 2*self.num_hs+1, (frag_joint_logits.shape[1],2*self.num_hs+1)
			frag_joint_logits = frag_joint_logits.flatten()

			# compute total NULL probability (before renormalization)
			frag_joint_logprobs = scatter_logsoftmax(
				frag_joint_logits,
				frag_joint_batch_idxs
			)
			frag_null_formula_logprob = scatter_logsumexp(
				(1.-frag_joint_mask) * frag_joint_logprobs + frag_joint_mask * LOG_ZERO(frag_joint_logprobs.dtype),
				frag_joint_batch_idxs
			)
			frag_null_formula_logprob = th.clamp(frag_null_formula_logprob, max=0.)

			if self.mask_null_formula:
				# compute non-NULL renormalized intensity
				frag_joint_logprobs = scatter_masked_softmax(
					frag_joint_logits,
					frag_joint_mask,
					frag_joint_batch_idxs,
					log=True
				)
			
			# reshape
			frag_joint_logprobs = frag_joint_logprobs.reshape(-1,2*self.num_hs+1)

			# aggregate formula logits by node, then normalize
			# log p(n) = logsumexp_f log p(f,n)
			frag_node_logits = th.logsumexp(frag_joint_logprobs,dim=1)
			frag_node_logprobs = scatter_logsoftmax(
				frag_node_logits,
				frag_node_batch_idxs
			)
			
			# calculate conditional probability
			# log p(f|n) = log p(f,n) - log p(n)
			frag_node_formula_logprobs = frag_joint_logprobs - frag_node_logprobs.unsqueeze(-1)
			frag_node_formula_logprobs = th.log_softmax(frag_node_formula_logprobs,dim=1)

		else:

			assert self.mlp_output_format == "node_formula", self.mlp_output_format

			# log p(f|n)
			frag_node_formula_logits = self.formula_module(frag_embed)
			assert frag_node_formula_logits.shape[1] == 2*self.num_hs+1, (frag_node_formula_logits.shape[1],2*self.num_hs+1)
			frag_node_formula_logprobs = th.log_softmax(frag_node_formula_logits,dim=1)

			# log p(n)
			frag_node_logits = self.node_module(frag_embed).squeeze(1)
			frag_node_logprobs = scatter_logsoftmax(
				frag_node_logits,
				frag_node_batch_idxs
			)

			# log p(f,n) = log p(f|n) + log p(n)
			frag_joint_logprobs = frag_node_formula_logprobs + frag_node_logprobs.unsqueeze(-1)
			frag_joint_logprobs = frag_joint_logprobs.flatten()

			# compute total NULL probability (before renormalization)
			frag_null_formula_logprob = scatter_logsumexp(
				(1.-frag_joint_mask) * frag_joint_logprobs + frag_joint_mask * LOG_ZERO(frag_joint_logprobs.dtype),
				frag_joint_batch_idxs
			)
			frag_null_formula_logprob = th.clamp(frag_null_formula_logprob, max=0.)

			if self.mask_null_formula:
				# compute non-NULL renormalized intensity
				frag_joint_logprobs = scatter_masked_softmax(
					frag_joint_logprobs,
					frag_joint_mask,
					frag_joint_batch_idxs,
					log=True
				)

		# aggregate by formula
		# log p(f) = logsumexp_n log p(f,n)
		frag_formula_mask = th.ones_like(frag_formula_batch_idxs,dtype=float_dtype)
		frag_formula_mask[frag_formula_cumsizes[:-1]] = 0.
		# aggregate by formula
		frag_formula_logprobs = scatter_logsumexp(
			frag_joint_logprobs.flatten(),
			frag_joint_formula_idxs
		)

		# softmax over formulae
		if self.mask_null_formula:
			frag_formula_logprobs = scatter_masked_softmax(
				frag_formula_logprobs,
				frag_formula_mask,
				frag_formula_batch_idxs,
				log=True
			)
		else:
			frag_formula_logprobs = scatter_masked_softmax(
				frag_formula_logprobs,
				th.ones_like(frag_formula_logprobs),
				frag_formula_batch_idxs,
				log=True
			)

		if self.predict_oos:
			# get OOS logits and logprobs
			oos_logits = self.oos_module(th.cat([mol_embed_gnn_pool,frag_embed_gnn_pool],dim=1))
			oos_logits = oos_logits.flatten()
			oos_logprobs = F.logsigmoid(oos_logits)
			not_oos_logprobs = F.logsigmoid(-oos_logits)
		else:
			# set them to 0
			oos_logprobs = LOG_ZERO(frag_formula_logprobs.dtype)*th.ones([batch_size],device=device)
			not_oos_logprobs = th.zeros([batch_size],device=device)
		
		# adjust frag_formula_logprobs
		frag_formula_oos_logprobs = frag_formula_logprobs + \
			th.repeat_interleave(not_oos_logprobs, frag_formula_sizes, dim=0)

		# convert to spectrum
		frag_formula_offsets = th.repeat_interleave(
			frag_formula_cumsizes[:-1],
			frag_formula_sizes-1 # -1 is for NULL formulae
		)
		spec_mzs = frag_formula_peak_mzs
		spec_logprobs = frag_formula_oos_logprobs[frag_formula_peak_idxs+frag_formula_offsets] + th.log(frag_formula_peak_probs)

		# get batch idxs
		spec_batch_idxs = th.repeat_interleave(
			th.arange(frag_formula_peak_sizes.shape[0],device=device),
			frag_formula_peak_sizes
		)

		if not self.skip_edge_loss:

			frag_edge_batch_idxs = frag_batch[frag_edge_index[0]]
			frag_edge_logits = frag_node_logprobs[frag_edge_index[0]] + frag_node_logprobs[frag_edge_index[1]]
			frag_edge_logprobs = scatter_logsoftmax(
				frag_edge_logits,
				frag_edge_batch_idxs
			)
			# print(scatter_logsumexp(frag_edge_logprobs,frag_edge_batch_idxs))
			frag_node_h_counts = get_node_feats(frag_x,frag_node_feat_idxs,"h_counts")
			frag_edge_h_ranges = get_edge_feats(frag_edge_attr,frag_edge_feat_idxs,"h_range")
			frag_edge_h_diffs = frag_node_h_counts[frag_edge_index[0]].unsqueeze(1) - frag_node_h_counts[frag_edge_index[1]].unsqueeze(2)
			frag_edge_h_diffs = frag_edge_h_diffs.reshape(frag_edge_h_diffs.shape[0],-1)
			frag_edge_h_range_masks = th.logical_or(
				frag_edge_h_diffs<frag_edge_h_ranges[:,0].unsqueeze(-1), 
				frag_edge_h_diffs>frag_edge_h_ranges[:,1].unsqueeze(-1)
			)
			frag_edge_h_logprobs = (frag_node_formula_logprobs[frag_edge_index[0]]).unsqueeze(1)  \
				+ (frag_node_formula_logprobs[frag_edge_index[1]]).unsqueeze(2)
			frag_edge_h_logprobs = frag_edge_h_logprobs.reshape(frag_edge_h_logprobs.shape[0],-1) 

		else:

			frag_edge_logprobs = None
			frag_edge_h_diffs = None
			frag_edge_h_range_masks = None
			frag_edge_h_logprobs = None
			frag_edge_batch_idxs = None

		frag_joint_node_idxs = th.repeat_interleave(th.arange(frag_joint_mask.shape[0]//(2*self.num_hs+1),device=device),2*self.num_hs+1)
		# select (will remove all NULL formula idxs, potentially some node idxs too if they contain only NULLs)
		frag_real_joint_logits = frag_joint_logits[frag_joint_mask.bool()]
		frag_real_joint_h_counts = frag_joint_h_counts[frag_joint_mask.bool()]
		frag_real_joint_node_idxs = frag_joint_node_idxs[frag_joint_mask.bool()]
		frag_real_joint_formula_idxs = frag_joint_formula_idxs[frag_joint_mask.bool()]
		frag_real_joint_batch_idxs = frag_joint_batch_idxs[frag_joint_mask.bool()]
		# P(f,n)
		frag_real_joint_logprobs = scatter_logsoftmax(
			frag_real_joint_logits,
			frag_real_joint_batch_idxs
		)
		# P(n) - sum, renormalize, but keep all-NULL nodes (as zeros)
		frag_real_node_node_idxs = th.arange(batch_frag_num_nodes,device=device)
		frag_real_node_logprobs = scatter_logsumexp(
			frag_real_joint_logprobs,
			frag_real_joint_node_idxs,
			dim_size=batch_frag_num_nodes
		)
		frag_real_node_logprobs = scatter_logsoftmax(
			frag_real_node_logprobs,
			frag_node_batch_idxs
		)
		# P(f) - sum, renormalize, but keep NULL formulae (as zeros)
		frag_real_formula_logprobs = scatter_logsumexp(
			frag_real_joint_logprobs,
			frag_real_joint_formula_idxs,
			dim_size=batch_frag_num_formulae
		)
		frag_real_formula_logprobs = scatter_logsoftmax(
			frag_real_formula_logprobs,
			frag_formula_batch_idxs,
		)
		frag_real_formula_formula_idxs = th.arange(batch_frag_num_formulae,device=device)
		# P(f|n) - remove all NULLs from conditionals
		frag_real_node_formula_logprobs = frag_real_joint_logprobs - frag_real_node_logprobs[frag_real_joint_node_idxs]
		frag_real_node_formula_logprobs = scatter_logsoftmax(
			frag_real_node_formula_logprobs,
			frag_real_joint_node_idxs
		)
		# P(n|f) - remove all NULLs from conditionals
		frag_real_formula_node_logprobs = frag_real_joint_logprobs - frag_real_formula_logprobs[frag_real_joint_formula_idxs]
		frag_real_formula_node_logprobs = scatter_logsoftmax(
			frag_real_formula_node_logprobs,
			frag_real_joint_formula_idxs
		)
		# hydrogens
		frag_real_joint_h_idxs = (frag_real_joint_h_counts + self.num_hs) + frag_real_joint_batch_idxs * (2*self.num_hs+1)
		frag_real_h_logprobs = scatter_logsumexp(
			frag_real_joint_logprobs,
			frag_real_joint_h_idxs,
			dim_size=batch_size*(2*self.num_hs+1)
		)
		frag_real_h_logprobs = th.clamp(frag_real_h_logprobs, max=0.)
		frag_real_h_counts = th.arange(-self.num_hs,self.num_hs+1,device=device).repeat(batch_size)
		frag_real_h_batch_idxs = th.repeat_interleave(th.arange(batch_size,device=device),2*self.num_hs+1)

		# calculate isomorphic distributions
		if self.nb_iso:
			# P(n')
			frag_nb_node_logprobs = scatter_logsumexp(
				frag_real_node_logprobs,
				frag_nb_idxs,
				dim_size=batch_frag_nb_num_nodes
			)
			frag_nb_node_batch_idxs = scatter_reduce(
				frag_node_batch_idxs,
				frag_nb_idxs,
				reduce="amax",
				dim_size=batch_frag_nb_num_nodes
			)
			frag_nb_node_node_idxs = frag_nb_un_idxs
			# P(n'|f)
			frag_nb_joint_idxs = frag_nb_idxs[frag_real_joint_node_idxs]
			frag_nb_joint_both_idxs = th.stack(
				[
					frag_nb_joint_idxs,
					frag_real_joint_formula_idxs
				], dim=1
			)
			frag_nb_joint_both_un_idxs, frag_nb_joint_both_inv_idxs = th.unique(
				frag_nb_joint_both_idxs,
				return_inverse=True,
				dim=0
			)
			frag_nb_joint_node_idxs = frag_nb_joint_both_un_idxs[:,0]
			frag_nb_joint_formula_idxs = frag_nb_joint_both_un_idxs[:,1]
			frag_nb_joint_batch_idxs = frag_nb_node_batch_idxs[frag_nb_joint_node_idxs]
			frag_nb_formula_node_logprobs = scatter_logsumexp(
				frag_real_formula_node_logprobs,
				frag_nb_joint_both_inv_idxs,
				dim_size=frag_nb_joint_both_un_idxs.shape[0]
			)
			frag_nb_formula_node_logprobs = th.clamp(frag_nb_formula_node_logprobs, max=0.)
			# P(n|n')
			frag_nb_node_node_logprobs = scatter_logsoftmax(
				frag_real_node_logprobs,
				frag_nb_idxs
			)
			frag_nb_node_node_node_idxs = frag_nb_idxs # frag_nb_node_node_idxs[frag_nb_inv_idxs]
			frag_nb_node_node_batch_idxs = frag_nb_node_batch_idxs[frag_nb_inv_idxs]
			assert th.all(frag_nb_node_node_logprobs <= 0.)
			# P(f|n')
			frag_nb_node_formula_logprobs = frag_real_node_formula_logprobs + frag_nb_node_node_logprobs[frag_real_joint_node_idxs]
			frag_nb_node_formula_logprobs = scatter_logsumexp(
				frag_nb_node_formula_logprobs,
				frag_nb_joint_both_inv_idxs,
				dim_size=frag_nb_joint_both_un_idxs.shape[0]
			)
			frag_nb_node_formula_logprobs = th.clamp(frag_nb_node_formula_logprobs, max=0.)
			# P(f,n')
			frag_nb_joint_logprobs = scatter_logsumexp(
				frag_real_joint_logprobs,
				frag_nb_joint_both_inv_idxs,
				dim_size=frag_nb_joint_both_un_idxs.shape[0]
			)
		else:
			frag_nb_node_logprobs = None
			frag_nb_node_formula_logprobs = None
			frag_nb_formula_node_logprobs = None
			frag_nb_node_node_logprobs = None
			frag_nb_node_node_idxs = None
			frag_nb_node_batch_idxs = None
			frag_nb_joint_node_idxs = None
			frag_nb_joint_formula_idxs = None
			frag_nb_joint_batch_idxs = None
			frag_nb_joint_logprobs = None
			frag_nb_node_node_node_idxs = None
			frag_nb_node_node_batch_idxs = None

		assert th.unique(frag_real_node_node_idxs).shape[0] == frag_real_node_node_idxs.max()+1 == batch_frag_num_nodes
		assert th.unique(frag_real_formula_formula_idxs).shape[0] == frag_real_formula_formula_idxs.max()+1 == batch_frag_num_formulae
		if self.nb_iso:
			assert th.unique(frag_nb_node_node_idxs).shape[0] == frag_nb_node_node_idxs.max()+1 == batch_frag_nb_num_nodes

		if self.bin_output:
			# import pdb; pdb.set_trace()
			spec_bin_mzs, spec_bin_logprobs, spec_bin_batch_idxs = batched_bin_func(
				mzs=spec_mzs,
				ints=spec_logprobs,
				batch_idxs=spec_batch_idxs,
				mz_max=self.mz_max,
				mz_bin_res=self.mz_bin_res,
				agg="lse",
				sparse=True,
				return_mzs=True
			)
			spec_mzs = spec_bin_mzs
			spec_logprobs = spec_bin_logprobs
			spec_batch_idxs = spec_bin_batch_idxs

		assert th.all(
			th.isclose(
				scatter_logsumexp(spec_logprobs, spec_batch_idxs).exp() + oos_logprobs.exp() + (not_oos_logprobs+frag_null_formula_logprob).exp(),
				th.ones_like(oos_logprobs),
				rtol=0.,
				atol=1e-2
			)
		), (scatter_logsumexp(spec_logprobs, spec_batch_idxs).exp(), oos_logprobs.exp(), (not_oos_logprobs+frag_null_formula_logprob).exp())

		out_d = {
			"pred_mzs": spec_mzs,
			"pred_logprobs": spec_logprobs,
			"pred_batch_idxs": spec_batch_idxs,
			"pred_formula_logprobs": frag_real_formula_logprobs,
			"pred_formula_formula_idxs": frag_real_formula_formula_idxs,
			"pred_formula_batch_idxs": frag_formula_batch_idxs,
			"pred_node_logprobs": frag_real_node_logprobs,
			"pred_node_node_idxs": frag_real_node_node_idxs,
			"pred_node_batch_idxs": frag_node_batch_idxs,
			"pred_node_formula_logprobs": frag_real_node_formula_logprobs,
			"pred_formula_node_logprobs": frag_real_formula_node_logprobs,
			"pred_joint_logprobs": frag_real_joint_logprobs,
			"pred_joint_node_idxs": frag_real_joint_node_idxs,
			"pred_joint_formula_idxs": frag_real_joint_formula_idxs,
			"pred_joint_batch_idxs": frag_real_joint_batch_idxs,
			"pred_joint_h_counts": frag_real_joint_h_counts,
			"pred_joint_h_idxs": frag_real_joint_h_idxs,
			"pred_null_formula_logprob": frag_null_formula_logprob,
			"pred_edge_logprobs": frag_edge_logprobs,
			"pred_edge_h_diffs": frag_edge_h_diffs,
			"pred_edge_h_range_masks": frag_edge_h_range_masks,
			"pred_edge_h_logprobs": frag_edge_h_logprobs,
			"pred_edge_batch_idxs": frag_edge_batch_idxs,
			"pred_oos_logprobs": oos_logprobs,
			"pred_h_counts": frag_real_h_counts,
			"pred_h_batch_idxs": frag_real_h_batch_idxs,
			"pred_h_logprobs": frag_real_h_logprobs,
			"pred_nb_node_logprobs": frag_nb_node_logprobs,
			"pred_nb_node_formula_logprobs": frag_nb_node_formula_logprobs,
			"pred_nb_formula_node_logprobs": frag_nb_formula_node_logprobs,
			"pred_nb_node_node_logprobs": frag_nb_node_node_logprobs,
			"pred_nb_node_node_idxs": frag_nb_node_node_idxs,
			"pred_nb_node_batch_idxs": frag_nb_node_batch_idxs,
			"pred_nb_joint_logprobs": frag_nb_joint_logprobs,
			"pred_nb_joint_node_idxs": frag_nb_joint_node_idxs,
			"pred_nb_joint_formula_idxs": frag_nb_joint_formula_idxs,
			"pred_nb_joint_batch_idxs": frag_nb_joint_batch_idxs,
			"pred_nb_node_node_node_idxs": frag_nb_node_node_node_idxs,
			"pred_nb_node_node_batch_idxs": frag_nb_node_node_batch_idxs,
		}

		if self.output_formula_str:

			assert frag_formula_str is not None, "frag_formula_strs must be provided if output_formula_str=True"
			assert spec_prec_type_str is not None, "spec_prec_type_strs must be provided if output_formula_str=True"			
			assert frag_formula_str.shape[0] == frag_real_formula_logprobs.shape[0]
			prec_type_str = spec_prec_type_str
			prec_type_delta_comp = np.array([PREC_TYPE_TO_FORMULA_DIFF[prec_type_str[i]] for i in range(len(prec_type_str))])
			prec_type_delta_comp = prec_type_delta_comp[frag_formula_batch_idxs.cpu().numpy()]
			pred_formula_str = [
				combine_formulae(frag_formula_str[i], prec_type_delta_comp[i]) if frag_formula_str[i] != "" else "" for i in range(len(frag_formula_str))
			]
			pred_formula_str = np.array(pred_formula_str)
			out_d["pred_formula_str"] = pred_formula_str

		for k, v in out_d.items():
			if "logprob" in k and v is not None and v.max() > 0.:
				print(f"> Warning: {k} has value {v.max()} > 0")
			if "batch_idx" in k and v is not None:
				if v.numel() == 0:
					raise ValueError(f"Empty batch index: {k}")
				elif th.unique(v).shape[0] != batch_size:
					raise ValueError(f"Missing items in batch: {k}")

		return out_d

class NeimsModel(nn.Module, CEModel, PrecModel, InstModel):

	def __init__(
		self,
		mol_fingerprint_morgan: bool,
		mol_fingerprint_rdkit: bool,
		mol_fingerprint_maccs: bool,
		mlp_hidden_size: int,
		mlp_dropout: float,
		mlp_num_layers: int,
		mlp_use_residuals: bool,
		mz_max: int,
		mz_bin_res: float,
		ff_prec_mz_offset: int,
		ff_bidirectional: bool,
		ff_output_map_size: int,
		ff_output_activation: str,
		int_embedder: str,
		ce_insert_type: str,
		ce_insert_location: str,
		ce_insert_merge: bool,
		ce_insert_size: int,
  		ce_max: float,
		ce_mean: float,
		ce_std: float,
		prec_insert_location: str,
		prec_insert_size: int,
		prec_types: list[str],
		inst_insert_location: str,
		inst_insert_size: int,
		inst_types: list[str],
		log_min: float):

		# nn.Module init
		super().__init__()
		
		self.mol_fingerprint_morgan = mol_fingerprint_morgan
		self.mol_fingerprint_rdkit = mol_fingerprint_rdkit
		self.mol_fingerprint_maccs = mol_fingerprint_maccs
		
		# input size
		self.mol_fp_dim = get_mol_fp_size(self.mol_fingerprint_morgan, self.mol_fingerprint_rdkit, self.mol_fingerprint_maccs)
		self.mlp_input_dim = self.mol_fp_dim

		# ce stuff
		self._ce_init(
			int_embedder=int_embedder,
			ce_insert_type=ce_insert_type,
			ce_insert_location=ce_insert_location,
			ce_insert_merge=ce_insert_merge,
			ce_insert_size=ce_insert_size,
   			ce_max=ce_max,
			ce_mean=ce_mean,
			ce_std=ce_std)
		self.mlp_input_dim += self.ce_mlp_input_dim

		# prec stuff
		self._prec_init(
			prec_insert_location=prec_insert_location,
			prec_insert_size=prec_insert_size,
			prec_num_types=len(prec_types))
		self.mlp_input_dim += self.prec_mlp_input_dim

		# inst stuff
		self._inst_init(
			inst_insert_location=inst_insert_location,
			inst_insert_size=inst_insert_size,
			inst_num_types=len(inst_types))
		self.mlp_input_dim += self.inst_mlp_input_dim

		self.ffn = SpecFFN(
			input_size=self.mlp_input_dim,
			hidden_size=mlp_hidden_size,
			mz_max=mz_max,
			mz_bin_res=mz_bin_res,
			num_layers=mlp_num_layers,
			dropout=mlp_dropout,
			use_residuals=mlp_use_residuals,
			bidirectional=ff_bidirectional,
			prec_mz_offset=ff_prec_mz_offset,
			output_map_size=ff_output_map_size,
			output_activation=ff_output_activation,
			log_min=log_min
		)

	def _ce_location_check(self):

		assert self.ce_insert_location in ["mlp","none"], f"ce_insert_location={self.ce_insert_location} not supported"

	def _prec_location_check(self):
		
		assert self.prec_insert_location in ["mlp","none"], f"prec_insert_location={self.prec_insert_location} not supported"

	def _inst_location_check(self):

		assert self.inst_insert_location in ["mlp","none"], f"prec_insert_location={self.inst_insert_location} not supported"

	def forward(
		self,
		mol_fingerprint: th.Tensor, 
		spec_prec_mz: th.Tensor,
		spec_ce: th.Tensor = None,
		spec_ce_batch_idxs: th.Tensor = None,
		spec_prec_type: th.Tensor = None,
		spec_inst_type: th.Tensor = None,
		**kwargs
	):

		fh = mol_fingerprint.reshape(-1,self.mol_fp_dim)
		batch_size = fh.shape[0]
		# get ce
		ce = spec_ce
		ce_batch_idxs = spec_ce_batch_idxs
		ce_embed = self.embed_ce(ce, ce_batch_idxs, batch_size)
		prec_embed = self.embed_prec(spec_prec_type)
		inst_embed = self.embed_inst(spec_inst_type)
		if self.ce_insert_location == "mlp":
			fh = th.cat([fh,ce_embed],dim=1)
		if self.prec_insert_location == "mlp":
			fh = th.cat([fh,prec_embed],dim=1)
		if self.inst_insert_location == "mlp":
			fh = th.cat([fh,inst_embed],dim=1)

		# apply ffn
		pred_mzs, pred_logprobs, pred_batch_idxs, pred_specs = self.ffn(fh,spec_prec_mz)
		out_d = {
			"pred_mzs": pred_mzs,
			"pred_logprobs": pred_logprobs,
			"pred_batch_idxs": pred_batch_idxs,
			"pred_specs": pred_specs
		}
		return out_d
	
class PrecursorModel(nn.Module):

	def __init__(self):

		super().__init__()
		self.dummy_params = nn.Parameter(th.zeros((1,), dtype=th.float32))

	def forward(
		self, 
		spec_prec_mz: th.Tensor,
		**kwargs):

		pred_mzs = spec_prec_mz
		pred_logprobs = 0.*self.dummy_params + th.zeros_like(pred_mzs)
		pred_batch_idxs = th.arange(pred_mzs.shape[0],device=pred_mzs.device)

		out_d = {
			"pred_mzs": pred_mzs,
			"pred_logprobs": pred_logprobs,
			"pred_batch_idxs": pred_batch_idxs
		}
		return out_d

class GNNModel(nn.Module, CEModel, PrecModel, InstModel):

	def __init__(
		self,
		mol_node_feats: list[str],
		mol_edge_feats: list[str],
		mol_pe_embed_k: int,
		mol_hidden_size: int,
		mol_num_layers: int,
		mol_gnn_type: str,
		mol_normalization: str,
		mol_dropout: float,
		mol_pool_type: str,
		mlp_hidden_size: int,
		mlp_dropout: float,
		mlp_num_layers: int,
		mlp_use_residuals: bool,
		mz_max: int,
		mz_bin_res: float,
		ff_prec_mz_offset: int,
		ff_bidirectional: bool,
		ff_output_map_size: int,
		ff_output_activation: str,
		int_embedder: str,
		ce_insert_type: str,
		ce_insert_location: str,
		ce_insert_merge: bool,
		ce_insert_size: int,
		ce_max: float,
		ce_mean: float,
		ce_std: float,
		prec_insert_location: str,
		prec_insert_size: int,
		prec_types: list[str],
		inst_insert_location: str,
		inst_insert_size: int,
		inst_types: list[str],
		log_min: float
	):
		# nn.Module init
		super().__init__()
		# collision energy
		self._ce_init(
			int_embedder=int_embedder,
			ce_insert_location=ce_insert_location,
			ce_insert_type=ce_insert_type,
			ce_insert_merge=ce_insert_merge,
			ce_insert_size=ce_insert_size,
			ce_max=ce_max,
			ce_mean=ce_mean,
			ce_std=ce_std
		)
		# precursor
		self._prec_init(
			prec_insert_location=prec_insert_location,
			prec_insert_size=prec_insert_size,
			prec_num_types=len(prec_types)
		)
		# instrument
		self._inst_init(
			inst_insert_location=inst_insert_location,
			inst_insert_size=inst_insert_size,
			inst_num_types=len(inst_types))

		# calculate node/edge feats sizes
		self.mol_node_feats = mol_node_feats
		self.mol_edge_feats = mol_edge_feats
		self.mol_pe_embed_k = mol_pe_embed_k
		self._compute_mol_feats_sizes()

		# setup mol gnn
		self.mol_node_feats_size += self.ce_mol_input_dim + self.prec_mol_input_dim + self.inst_mol_input_dim
		mol_kwargs = {
			"node_feats_size": self.mol_node_feats_size,
			"edge_feats_size": self.mol_edge_feats_size,
			"hidden_size": mol_hidden_size,
			"num_layers": mol_num_layers,
			"gnn_type": mol_gnn_type,
			"dropout": mol_dropout,
			"normalization": mol_normalization,
		}
		# Mol GNN
		self.mol_embedder = GNN(**mol_kwargs)
		self.mol_pool_type = mol_pool_type
		self.mol_pool = build_pool_module(mol_pool_type,mol_hidden_size)

		# MLP input = GNN output
		self.mlp_input_dim = mol_hidden_size
		# metadata
		self.mlp_input_dim += self.ce_mlp_input_dim + self.prec_mlp_input_dim + self.inst_mlp_input_dim

		self.ffn = SpecFFN(
			input_size=self.mlp_input_dim,
			hidden_size=mlp_hidden_size,
			mz_max=mz_max,
			mz_bin_res=mz_bin_res,
			num_layers=mlp_num_layers,
			dropout=mlp_dropout,
			use_residuals=mlp_use_residuals,
			bidirectional=ff_bidirectional,
			prec_mz_offset=ff_prec_mz_offset,
			output_map_size=ff_output_map_size,
			output_activation=ff_output_activation,
			log_min=log_min
		)

	def forward(
		self, 
		mol_pyg: pyg.data.Data,
		spec_prec_mz: th.Tensor,
		spec_nce: th.Tensor = None,
		spec_nce_batch_idxs: th.Tensor = None,
		spec_prec_type: th.Tensor = None,
		spec_inst_type: th.Tensor = None,
		**kwargs
	):
		# mol features
		# mol_x: mol level node feature matrix
		# mol_edge_index: mol graph connectivity in COO format with shape [2, num_edges]
		# edge_attr: mol graph edge feature matrix with shape [num_edges, num_edge_features]
		# batch: sample idx repsect to current batch
		mol_x, mol_edge_index, mol_edge_attr, mol_batch = mol_pyg.x, mol_pyg.edge_index, mol_pyg.edge_attr, mol_pyg.batch

		# int_dtype = mol_edge_index.dtype
		batch_size = mol_batch[-1]+1

		# metadata embedders
		# get ce value
		ce = spec_nce
		ce_batch_idxs = spec_nce_batch_idxs
		ce_embed = self.embed_ce(ce, ce_batch_idxs, batch_size)
		# get prec value
		prec_embed = self.embed_prec(spec_prec_type)
		# get inst value
		inst_embed = self.embed_inst(spec_inst_type)		

		# metadata embeddings at the node feature level
		if self.ce_insert_location == "mol":
			mol_ce_embed = th.repeat_interleave(ce_embed,th.unique(mol_batch,return_counts=True)[1],dim=0)
			mol_x = th.cat([mol_x,mol_ce_embed],dim=1)
		if self.prec_insert_location == "mol":
			mol_prec_embed = th.repeat_interleave(prec_embed,th.unique(mol_batch,return_counts=True)[1],dim=0)
			mol_x = th.cat([mol_x,mol_prec_embed],dim=1)
		if self.inst_insert_location == "mol":
			mol_inst_embed = th.repeat_interleave(inst_embed,th.unique(mol_batch,return_counts=True)[1],dim=0)
			mol_x = th.cat([mol_x,mol_inst_embed],dim=1)
		
		# get per-atom embeddings
		mol_embed_gnn = self.mol_embedder(
			mol_x,
			mol_batch,
			mol_edge_index,
			mol_edge_attr
		)
		mol_embed_gnn_pool = self.mol_pool(mol_embed_gnn,mol_batch)
		ffn_input = mol_embed_gnn_pool

		if self.ce_insert_location == "mlp":
			ffn_input = th.cat([ffn_input,ce_embed],dim=1)
		if self.prec_insert_location == "mlp":
			ffn_input = th.cat([ffn_input,prec_embed],dim=1)
		if self.inst_insert_location == "mlp":
			ffn_input = th.cat([ffn_input,inst_embed],dim=1)

		# apply ffn
		pred_mzs, pred_logprobs, pred_batch_idxs, pred_specs = self.ffn(ffn_input,spec_prec_mz)
		out_d = {
			"pred_mzs": pred_mzs,
			"pred_logprobs": pred_logprobs,
			"pred_batch_idxs": pred_batch_idxs,
			"pred_specs": pred_specs
		}
		return out_d
		
	def _compute_mol_feats_sizes(self):
		""" method compute mol feature size
			these features don't rely on any model parameters
		"""
		self.mol_node_feats_size, self.mol_edge_feats_size = get_mol_feats_sizes(
			self.mol_node_feats, 
			self.mol_edge_feats, 
			self.mol_pe_embed_k
		)

	def _ce_location_check(self):

		assert self.ce_insert_location in ["mlp","mol","none"], f"ce_insert_location={self.ce_insert_location} not supported"

	def _prec_location_check(self):
		
		assert self.prec_insert_location in ["mlp","mol","none"], f"prec_insert_location={self.prec_insert_location} not supported"

	def _inst_location_check(self):

		assert self.inst_insert_location in ["mlp","mol","none"], f"prec_insert_location={self.inst_insert_location} not supported"