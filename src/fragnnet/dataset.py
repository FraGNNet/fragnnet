import pandas as pd
import os
import torch as th
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler
from torch.utils.data import Dataset
import numpy as np
import torch_geometric as pyg
from torch_geometric.data import Batch
import sys
from tqdm import tqdm
from typing import Iterator, List
import copy

from fragnnet.utils.feat_utils import batch_mols_frags, get_mol_fp, get_mol_graph, get_frag_graph
from fragnnet.utils.spec_utils import batch_func
from fragnnet.utils.proc_utils import merge_spec_df
from fragnnet.utils.frag_utils import load_frag_d
from fragnnet.utils.misc_utils import flatten_lol, get_pyg_memory_usage, get_tensor_dict_memory_usage, none_or_nan
from fragnnet.utils.data_utils import fill_missing_ace, fill_missing_nce, seq_apply_df_rows
import fragnnet.massformer.data_utils as mf_data_utils
from fragnnet.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF

class BaseDataset(Dataset):

	def _base_init(
		self,
		spec_fp: str,
		mol_fp: str,
		split_dp: str,
		split: str,
		subsample_params: dict,
		spec_params: dict):

		self.split = split
		self.subsample_params = subsample_params
		self.spec_params = spec_params
		spec_df, mol_df, um_spec_df, split_df, id_key, ce_key = BaseDataset._setup_dfs(
			spec_fp_or_df=spec_fp,
			mol_fp_or_df=mol_fp,
			split_dp=split_dp,
			splits=[split],
			subsample_params=subsample_params,
			spec_params=spec_params
		)
		self.spec_df = spec_df
		self.mol_df = mol_df
		self.um_spec_df = um_spec_df
		self.split_df = split_df
		self.id_key = id_key
		self.ce_key = ce_key
		self._compute_counts()
		self._setup_prec_type_to_idx()
		self._setup_inst_type_to_idx()

	@staticmethod
	def _setup_dfs( 
		spec_fp_or_df: str|pd.DataFrame,
		mol_fp_or_df: str|pd.DataFrame,
		split_dp: str,
		splits: List[str],
		subsample_params: dict,
		spec_params: dict):
		
		spec_df = spec_fp_or_df if isinstance(spec_fp_or_df, pd.DataFrame) else pd.read_pickle(spec_fp_or_df)
		mol_df = mol_fp_or_df if isinstance(mol_fp_or_df, pd.DataFrame) else pd.read_pickle(mol_fp_or_df)
  
		split_dfs = []
		for split in splits:
			# assert split in ["train","val","test","secondary","predict_only"], split
			if split == "predict_only":
				assert len(splits) == 1, splits
				# predict_all split, just include everything, this is used for prediction
				split_df = pd.DataFrame()
				# fill these to keep compatible
				split_df["spec_id"] = spec_df["spec_id"]
				split_df["mol_id"] = spec_df["mol_id"]
				split_df["group_id"] = spec_df["spec_id"]
			else:	
				split_fp = os.path.join(split_dp,f"{split}_ids.csv")
				assert os.path.isfile(split_fp), split_fp
				split_df = pd.read_csv(split_fp)
			split_dfs.append(split_df)
		split_df = pd.concat(split_dfs,ignore_index=True).reset_index(drop=True)
   
		# select spectra
		spec_df = spec_df[spec_df["spec_id"].isin(split_df["spec_id"])]
		# select molecules
		mol_df = mol_df[mol_df["mol_id"].isin(split_df["mol_id"])]
		assert np.all(np.unique(mol_df["mol_id"]) == np.unique(spec_df["mol_id"]))
		
		# covert ace and nce if there is data
		assert not (spec_params["ace"] and spec_params["nce"])
		if spec_params["ace"]:
			ce_key = "ace"
			spec_df.loc[:,"ace"] = seq_apply_df_rows(spec_df, fill_missing_ace)
		elif spec_params["nce"]:
			ce_key = "nce"
			spec_df.loc[:,"nce"] = seq_apply_df_rows(spec_df, fill_missing_nce)
		else:
			ce_key = None

		if spec_params["test_ces"] is not None and ce_key is not None and "test" in splits:
			test_split_fp = os.path.join(split_dp,"test_ids.csv")
			test_split_df = pd.read_csv(test_split_fp)
			orig_test_spec_df = spec_df[spec_df["spec_id"].isin(test_split_df["spec_id"])]
			test_drop_ids = orig_test_spec_df[~(orig_test_spec_df[ce_key].isin(np.array(spec_params["test_ces"])))][["spec_id","group_id"]]
			spec_df = spec_df[~(spec_df["spec_id"].isin(test_drop_ids["spec_id"]))]
			new_test_spec_df = spec_df[spec_df["spec_id"].isin(test_split_df["spec_id"])]
			orig_test_spec_count = orig_test_spec_df["spec_id"].nunique()
			orig_test_group_count = orig_test_spec_df["group_id"].nunique()
			new_test_spec_count = new_test_spec_df["spec_id"].nunique()
			new_test_group_count = new_test_spec_df["group_id"].nunique()
			print(f">> Dropping spectra unless {ce_key} is one of {spec_params['test_ces']}")
			print(f"> Before drop: {orig_test_spec_count} spectra, {orig_test_group_count} groups")
			print(f"> After drop: {new_test_spec_count} spectra, {new_test_group_count} groups")

		# merge spectra
		um_spec_df = spec_df[["dset","dset_spec_id","spec_id","group_id","mol_id","peaks"]].copy()
		if spec_params["merge"]:
			spec_df = merge_spec_df(spec_df,keep_ces=spec_params["merge_keep_ces"])
			id_key = "group_id"
		else:
			id_key = "spec_id"
   
		# subsample
		if subsample_params.get(split,False) and subsample_params["subsample_size"] > 0:
			if isinstance(subsample_params["subsample_size"],int):
				n = subsample_params["subsample_size"]
				frac = None
			else:
				assert isinstance(subsample_params["subsample_size"],float)
				n = None
				frac = subsample_params["subsample_size"]
			spec_df = spec_df.sample(
				n=n,
				frac=frac,
				random_state=subsample_params["subsample_seed"],
				replace=False)
			mol_df = mol_df[mol_df["mol_id"].isin(spec_df["mol_id"])]
			um_spec_df = um_spec_df[um_spec_df[id_key].isin(spec_df[id_key])]

		# reset indices
		spec_df = spec_df.reset_index(drop=True)
		mol_df = mol_df.reset_index(drop=True)
		# use mol_id as index for speedy access
		mol_df = mol_df.set_index("mol_id",drop=False).sort_index().rename_axis(None)
		return spec_df, mol_df, um_spec_df, split_df, id_key, ce_key

	def _compute_counts(self):

		self.group_per_mol = self.spec_df[["mol_id","group_id"]].drop_duplicates().groupby("mol_id").size().to_dict()
		if self.spec_params["merge"]:
			self.spec_per_mol = copy.deepcopy(self.group_per_mol)
			self.spec_per_group = {group_id: 1 for group_id in self.spec_df["group_id"].unique()}
		else:
			self.spec_per_mol = self.spec_df[["mol_id","spec_id"]].drop_duplicates().groupby("mol_id").size().to_dict()
			self.spec_per_group = self.spec_df[["group_id","spec_id"]].drop_duplicates().groupby("group_id").size().to_dict()

	def get_group_mol_stats(self):

		group_ids = []
		mol_ids = []
		spec_per_group_stats = []
		spec_per_mol_stats = []
		group_per_mol_stats = []
		for row_idx, row in self.spec_df.iterrows():
			spec_entry = row
			mol_id = spec_entry["mol_id"]
			group_id = spec_entry["group_id"]
			spec_per_group = self.spec_per_group[group_id]
			spec_per_mol = self.spec_per_mol[mol_id]
			group_per_mol = self.group_per_mol[mol_id]
			group_ids.append(group_id)
			spec_per_group_stats.append(spec_per_group)
			mol_ids.append(mol_id)
			spec_per_mol_stats.append(spec_per_mol)
			group_per_mol_stats.append(group_per_mol)
		group_ids = th.tensor(group_ids)
		mol_ids = th.tensor(mol_ids)
		spec_per_group_stats = th.tensor(spec_per_group_stats)
		spec_per_mol_stats = th.tensor(spec_per_mol_stats)
		group_per_mol_stats = th.tensor(group_per_mol_stats)
		return group_ids, mol_ids, spec_per_group_stats, spec_per_mol_stats, group_per_mol_stats

	@staticmethod
	def get_data_dict_types():
		return ["spec_pp_sd"]

	def _preprocess_spec(self,spec_pp_sd: dict):

		# preload and pre-process spectra
		if self.spec_params["preprocess"]:
			self.spec_datas = spec_pp_sd
			total_spec_data_size = 0
			for idx, spec_entry in tqdm(self.spec_df.iterrows(),desc="> preprocess spec",total=len(self.spec_df)):
				spec_data = self._process_spec(spec_entry)
				total_spec_data_size += get_tensor_dict_memory_usage(**spec_data)
				self.spec_datas[idx] = spec_data
			print(f"> total_spec_data_size: {total_spec_data_size/1e6:.2f} MB")

	@staticmethod
	def _get_mzs_ints(peaks:list):
		"""
  		convert peaks to tensors, it is caller's responsibility to make sure data is valid 
		"""
		mzs, ints = [], []
		for peak in peaks:
			p_mz, p_int = peak
			mzs.append(p_mz)
			ints.append(p_int)

		mzs = th.tensor(mzs,dtype=th.float)
		ints = th.tensor(ints,dtype=th.float)
		# mzs, ints = filter_func(mzs, ints, self.spec_params["ints_thresh"], self.spec_params["mz_max"])
		return mzs, ints
	
	def _setup_prec_type_to_idx(self):

		prec_types = sorted(self.spec_params["prec_types"])
		assert all(prec_type in PREC_TYPE_TO_MASS_DIFF for prec_type in prec_types), prec_types
		self.prec_type_to_idx = {prec_type: idx for idx, prec_type in enumerate(prec_types)}
		self.idx_to_prec_type = {idx: prec_type for idx, prec_type in enumerate(prec_types)}
		self.num_prec_types = len(prec_types)

	def _setup_inst_type_to_idx(self):

		inst_types = sorted(self.spec_params["inst_types"])
		self.inst_type_to_idx = {inst_type: idx for idx, inst_type in enumerate(inst_types)}
		self.idx_to_inst_type = {idx: inst_type for idx, inst_type in enumerate(inst_types)}
		self.num_inst_types = len(inst_types)

	def _process_spec(self,spec_entry):
		spec_data = {}
		# peak data
		mzs, ints = BaseDataset._get_mzs_ints(spec_entry["peaks"])
		if self.spec_params["sparse"]:
			# get sparse spectrum
			spec_data["spec_mzs"] = mzs
			spec_data["spec_ints"] = ints
		# metadata
		if self.spec_params["prec_type"]:
			prec_type = spec_entry["prec_type"]
			prec_type = th.tensor([self.prec_type_to_idx[prec_type]],dtype=th.long)
			spec_data["spec_prec_type"] = prec_type
		if self.spec_params["prec_type_str"]:
			prec_type_str = spec_entry["prec_type"]
			spec_data["spec_prec_type_str"] = np.array([prec_type_str])
		if self.spec_params["inst_type"]:
			inst_type = spec_entry["inst_type"]
			inst_type = th.tensor([self.inst_type_to_idx[inst_type]],dtype=th.long)
			spec_data["spec_inst_type"] = inst_type
		if self.spec_params["prec_mass_diff"]:
			prec_type = spec_entry["prec_type"]
			mass_diff = th.tensor([PREC_TYPE_TO_MASS_DIFF[prec_type]],dtype=th.float)
			spec_data["spec_prec_mass_diff"] = mass_diff
		if self.spec_params["nce"] or self.spec_params["ace"]:
			assert self.ce_key is not None
			assert (not self.spec_params["merge"]) or self.spec_params["merge_keep_ces"]
			assert not (self.spec_params["ace"] and self.spec_params["nce"])
			ce = spec_entry[self.ce_key]
			if self.spec_params["merge"]:
				assert self.spec_params["merge_keep_ces"]
				assert isinstance(ce,list), type(ce)
				ce = th.tensor(ce,dtype=th.float)
				spec_data["spec_ce"] = ce
			else:
				assert isinstance(ce,float), type(ce)
				ce = th.tensor([ce],dtype=th.float)
				spec_data["spec_ce"] = ce
		if self.spec_params["prec_mz"]:
			prec_mz = spec_entry["prec_mz"]
			prec_mz = th.tensor([float(prec_mz)],dtype=th.float)
			spec_data["spec_prec_mz"] = prec_mz
		if self.spec_params["unique_id"]:
			unique_id = spec_entry[self.id_key]
			unique_id = th.tensor([unique_id],dtype=th.long)
			spec_data["spec_unique_id"] = unique_id
			spec_data['group_id'] =  th.tensor([spec_entry['group_id']],dtype=th.long)
			spec_data['mol_id'] = spec_entry['mol_id'] # mol id does not need to be an int #th.tensor([spec_entry['mol_id']],dtype=th.long)
		if self.spec_params["counts"]:
			spec_per_mol = self.spec_per_mol[spec_entry["mol_id"]]
			spec_per_mol = th.tensor([spec_per_mol],dtype=th.long)
			spec_data["spec_per_mol"] = spec_per_mol
			group_per_mol = self.group_per_mol[spec_entry["mol_id"]]
			group_per_mol = th.tensor([group_per_mol],dtype=th.long)
			spec_data["group_per_mol"] = group_per_mol
			spec_per_group = self.spec_per_group[spec_entry["group_id"]]
			spec_per_group = th.tensor([spec_per_group],dtype=th.long)
			spec_data["spec_per_group"] = spec_per_group
		return spec_data

	def __getitem__(self, idx):

		raise NotImplementedError()

	def __len__(self):

		return len(self.spec_df)

	@staticmethod
	def get_collate_fn():
		
		return BaseDataset.collate_fn

	@staticmethod
	def _setup_collate(data_list):

		batch_size = len(data_list)
		keys = list(data_list[0].keys())
		collate_data = {key: [] for key in keys}
		for data in data_list:
			for key in keys:
				collate_data[key].append(data[key])
		return batch_size, keys, collate_data

	@staticmethod
	def _special_collate(keys, collate_data):

		# handle sparse spectra
		if "spec_ints" in keys and "spec_mzs" in keys:
			# create batch_idxs
			mzs, ints, batch_idxs = batch_func(
				collate_data["spec_mzs"],
				collate_data["spec_ints"]
			)
			collate_data["spec_mzs"] = mzs
			collate_data["spec_ints"] = ints
			collate_data["spec_batch_idxs"] = batch_idxs
			# remove from list
			keys.remove("spec_ints")
			keys.remove("spec_mzs")
   
		# handle sparse ces
		if 'spec_ce' in keys:
			# create batch_idxs
			ces, _, batch_idxs = batch_func(
				collate_data['spec_ce'],
				collate_data['spec_ce'] # duplicate for compatibility
			)
			collate_data['spec_ce'] = ces
			collate_data["spec_ce_batch_idxs"] = batch_idxs
			# remove from list
			keys.remove('spec_ce')
				
	@staticmethod
	def _standard_collate(batch_size,keys,collate_data):
		""" mutates keys and collate_data """

		# handle generic data
		for key in keys:
			values = collate_data[key]
			if isinstance(values[0],list):
				# flatten
				values = flatten_lol(values)
				collate_data[key] = values
			elif isinstance(values[0],th.Tensor):
				# cat
				values = th.cat(values,dim=0)
				collate_data[key] = values
			elif isinstance(values[0],np.ndarray):
				# cat
				values = np.concatenate(values,axis=0)
				collate_data[key] = values
			elif key in ["mol_id"]:
				collate_data[key] = values
			else:
				raise TypeError(f"Unsupported type: {key} {type(values[0])}")
		# remove everything
		keys.clear()
		# add batch size
		collate_data["batch_size"] = th.tensor(batch_size, dtype=th.long)

	@staticmethod
	def collate_fn(data_list):

		raise NotImplementedError()

class SpecMolDataset(BaseDataset):

	def __init__(
		self,
		spec_fp: str,
		mol_fp: str,
		split_dp: str,
		split: str,
		subsample_params: dict,
		spec_params: dict,
		mol_params: dict,
		spec_pp_sd: dict = None,
		mol_pp_sd: dict = None,
		**kwargs
	):

		BaseDataset.__init__(self)
		self._base_init(
			spec_fp=spec_fp,
			mol_fp=mol_fp,
			split_dp=split_dp,
			split=split,
			subsample_params=subsample_params,
			spec_params=spec_params
		)
  
		if spec_pp_sd is None:
			spec_pp_sd = dict()
		if mol_pp_sd  is None:
			mol_pp_sd = dict()
   
		self.mol_params = mol_params
		self._preprocess_spec(spec_pp_sd)
		self._preprocess_mol(mol_pp_sd)
	
	@staticmethod
	def get_data_dict_types():
		return ["spec_pp_sd", "mol_pp_sd"]

	def _get_mol_graph_size(self, mol_data):

		if self.mol_params["pyg"]:
			mol_pyg = mol_data["mol_pyg"]
			mol_graph_size = get_pyg_memory_usage(mol_pyg)
		else:
			mol_graph_size = 0
		return mol_graph_size

	def _preprocess_mol(self, mol_pp_sd: dict):

		# preload and pre-process molecules
		if self.mol_params["preprocess"]:
			self.mol_datas = mol_pp_sd
			total_mol_graph_size = 0
			for idx, mol_entry in tqdm(self.mol_df.iterrows(),desc="> preprocess mol",total=len(self.mol_df)):
				mol_data = self._process_mol(mol_entry)
				total_mol_graph_size += self._get_mol_graph_size(mol_data)
				self.mol_datas[mol_entry["mol_id"]] = mol_data
			print(f"> total_mol_graph_size: {total_mol_graph_size/1e6:.2f} MB")

	def __getitem__(self,idx):
		spec_entry = self.spec_df.iloc[idx]
		mol_id = spec_entry["mol_id"]
		mol_entry = self.mol_df.loc[mol_id]
		if self.spec_params["preprocess"]:
			spec_data = self.spec_datas[idx].copy()
		else:
			spec_data = self._process_spec(spec_entry)
		if self.mol_params["preprocess"]:
			mol_data = self.mol_datas[mol_id].copy()
		else:
			mol_data = self._process_mol(mol_entry)
		data = {**spec_data,**mol_data}
		return data

	def _process_mol(self,mol_entry):

		mol_data = {}
		mol = mol_entry["mol"]
		if self.mol_params["smiles"]:
			smiles = mol_entry["smiles"]
			mol_data["mol_smiles"] = [smiles]
		if self.mol_params["fingerprint"]:
			fingerprint = get_mol_fp(
				mol,
				self.mol_params["fingerprint_morgan"],
				self.mol_params["fingerprint_rdkit"],
				self.mol_params["fingerprint_maccs"]
			)
			mol_data["mol_fingerprint"] = fingerprint
		if self.mol_params["pyg"]:
			mol_pyg = get_mol_graph(
				mol,
				self.mol_params["pyg_node_feats"],
				self.mol_params["pyg_edge_feats"],
				self.mol_params["pyg_pe_embed_k"],
				self.mol_params["pyg_bigraph"]
			)
			mol_data["mol_pyg"] = mol_pyg
		if self.mol_params["mf"]:
			mol_mf = mf_data_utils.gf_preprocess(mol,-1)
			mol_data["mol_mf"] = mol_mf
		return mol_data

	@staticmethod
	def get_collate_fn():

		return SpecMolDataset.collate_fn

	@staticmethod
	def _special_collate(keys, collate_data):

		if "mol_pyg" in keys:
			# batch
			collate_data["mol_pyg"] = Batch.from_data_list(collate_data["mol_pyg"])
			# remove from list
			keys.remove("mol_pyg")
		if "mol_mf" in keys:
			# batch
			mol_mf_d = mf_data_utils.collator(collate_data["mol_mf"])
			for k, v in mol_mf_d.items():
				collate_data["mol_mf_"+k] = v
			# remove from list
			collate_data.pop("mol_mf")
			keys.remove("mol_mf")
		if "mol_graff" in keys:
			# batch
			collate_data["mol_graff"] = Batch.from_data_list(collate_data["mol_graff"])
			# remove from list
			keys.remove("mol_graff")
		BaseDataset._special_collate(keys,collate_data)

	@staticmethod
	def collate_fn(data_list):
		
		batch_size, keys, collate_data = SpecMolDataset._setup_collate(data_list)
		SpecMolDataset._special_collate(keys,collate_data)
		SpecMolDataset._standard_collate(batch_size,keys,collate_data)
		return collate_data

	def training_data_sanity_check(self):
		"""
  		basic data sanity check for training time only 
		"""
		assert self.spec_df['peaks'].isna().any() == False
		assert (self.spec_df['peaks'] == '').any() == False
  	
class SpecMolFragDataset(SpecMolDataset):

	def __init__(
		self,
		spec_fp: str,
		mol_fp: str,
		split_dp: str,
		split: str,
		subsample_params: dict,
		spec_params: dict,
		mol_params: dict,
		frag_dp: str,
		frag_params: dict,
		spec_pp_sd: dict = None,
		mol_pp_sd: dict = None,
		frag_pl_sd: dict = None,
		frag_pp_sd: dict = None,
		**kwargs
	):

		BaseDataset.__init__(self)
		self._base_init(
			spec_fp=spec_fp,
			mol_fp=mol_fp,
			split_dp=split_dp,
			split=split,
			subsample_params=subsample_params,
			spec_params=spec_params
		)
		self.mol_params = mol_params
		self.frag_dp = frag_dp
		self.frag_params = frag_params
		
		if spec_pp_sd is None:
			spec_pp_sd = dict()
		if mol_pp_sd  is None:
			mol_pp_sd = dict()
		if frag_pl_sd is None:
			frag_pl_sd = dict()
		if frag_pp_sd is None:
			frag_pp_sd = dict()
   
		self._preprocess_spec(spec_pp_sd)
		self._preprocess_mol(mol_pp_sd)
		self._preprocess_frag(frag_pl_sd, frag_pp_sd)
	
	@staticmethod
	def get_data_dict_types():
		return ["spec_pp_sd", "mol_pp_sd", "frag_pl_sd", "frag_pp_sd"]

	def _preprocess_frag(self, frag_pl_sd: dict, frag_pp_sd: dict):

		# preload frag dags
		if self.frag_params["preload"]:
			self.frag_entries = frag_pl_sd
			total_frag_entry_size = 0
			for mol_id in tqdm(self.mol_df["mol_id"].values,desc="> preload frag",total=len(self.mol_df)):
				frag_entry = load_frag_d(mol_id,self.frag_dp,self.frag_params["compressed"])
				total_frag_entry_size += get_pyg_memory_usage(frag_entry["dag"])
				self.frag_entries[mol_id] = frag_entry
			print(f"> total_frag_entry_size: {total_frag_entry_size/1e6:.2f} MB")
   
		# preprocess frag dags
		if self.frag_params["preprocess"]:
			assert self.frag_params["preload"]
			self.frag_data = frag_pp_sd
			total_frag_data_size = 0
			for k in tqdm(list(self.frag_entries.keys()),desc="> preprocess frag",total=len(self.frag_entries)):
				frag_data = self._process_frag(self.frag_entries.pop(k),None)
				total_frag_data_size += get_pyg_memory_usage(frag_data["frag_pyg"])
				self.frag_data[k] = frag_data
			print(f"> total_frag_data_size: {total_frag_data_size/1e6:.2f} MB")
			# remove them from the entries
			self.frag_entries = {}

	def __getitem__(self, idx):

		spec_entry = self.spec_df.iloc[idx]
		mol_id = spec_entry["mol_id"]
		mol_entry = self.mol_df.loc[mol_id]
		if self.spec_params["preprocess"]:
			spec_data = self.spec_datas[idx].copy()
		else:
			spec_data = self._process_spec(spec_entry)
		if self.mol_params["preprocess"]:
			mol_data = self.mol_datas[mol_id].copy()
		else:
			mol_data = self._process_mol(mol_entry)
		# frag stuff
		if self.frag_params["preprocess"]:
			frag_data = self.frag_data[mol_id].copy()
			# update prec_mass_diff
			prec_type_mass_diff = PREC_TYPE_TO_MASS_DIFF[spec_entry["prec_type"]]
			frag_data["frag_formula_peak_mzs"] = frag_data["frag_formula_peak_mzs"] + prec_type_mass_diff
		elif self.frag_params["preload"]:
			frag_entry = self.frag_entries[mol_id].copy()
			frag_data = self._process_frag(frag_entry,spec_entry)
		else:
			frag_entry = self._load_frag_entry(mol_id)
			frag_data = self._process_frag(frag_entry,spec_entry)
		data = {**spec_data,**mol_data,**frag_data}
		return data

	def _load_frag_entry(self,mol_id):

		frag_entry = load_frag_d(
			mol_id,
			self.frag_dp,
			self.frag_params["compressed"]
		)
		return frag_entry

	def _process_frag(self,frag_entry,spec_entry):

		frag_data = {}
		if self.frag_params["pyg"]:
			frag_pyg = frag_entry["dag"]
			frag_pyg = get_frag_graph(
				frag_pyg,
				self.frag_params["pyg_node_feats"],
				self.frag_params["pyg_edge_feats"],
				self.frag_params["pyg_edges"],
				self.frag_params["pyg_bigraph"]
			)
			frag_data["frag_pyg"] = frag_pyg
		if self.frag_params["formula_peak_mzs"]:
			formula_peak_mzs = frag_entry["formula_peak_mzs"]
			formula_peak_mzs = formula_peak_mzs[:,:self.frag_params["num_isotopes"]]
			if spec_entry is not None:
				prec_type_mass_diff = PREC_TYPE_TO_MASS_DIFF[spec_entry["prec_type"]]
				formula_peak_mzs = formula_peak_mzs + prec_type_mass_diff
			frag_data["frag_formula_peak_mzs"] = formula_peak_mzs
		if self.frag_params["formula_peak_probs"]:	
			formula_peak_probs = frag_entry["formula_peak_probs"]
			formula_peak_probs = F.normalize(formula_peak_probs[:,:self.frag_params["num_isotopes"]],dim=1,p=1)
			frag_data["frag_formula_peak_probs"] = formula_peak_probs
		if self.frag_params["formula_str"]:
			# import pdb; pdb.set_trace()
			formula_str = list(frag_entry["idx_to_formula"].values())
			assert formula_str[0] == "", formula_str[0]
			formula_str = np.array(formula_str)
			frag_data["frag_formula_str"] = formula_str
		return frag_data

	@staticmethod
	def get_collate_fn():

		return SpecMolFragDataset.collate_fn

	@staticmethod
	def _special_collate(keys, collate_data):

		if "frag_pyg" in keys:
			assert "mol_pyg" in keys
			assert "frag_formula_peak_mzs" in keys
			assert "frag_formula_peak_probs" in keys
			# process
			batch_mol_frag_data = batch_mols_frags(
				collate_data["mol_pyg"],
				collate_data["frag_pyg"],
				collate_data["frag_formula_peak_mzs"],
				collate_data["frag_formula_peak_probs"]
			)
			for k,v in batch_mol_frag_data.items():
				collate_data[k] = v
			# remove from list
			keys.remove("frag_pyg")
			keys.remove("mol_pyg")
			keys.remove("frag_formula_peak_mzs")
			keys.remove("frag_formula_peak_probs")
		SpecMolDataset._special_collate(keys,collate_data)

	@staticmethod
	def collate_fn(data_list):

		# prevent edge case causing crash
		if len(data_list) == 0:
			return  {"batch_size": th.tensor(0)}
		batch_size, keys, collate_data = SpecMolFragDataset._setup_collate(data_list)
		# special handling
		SpecMolFragDataset._special_collate(keys,collate_data)
		SpecMolFragDataset._standard_collate(batch_size,keys,collate_data)
		return collate_data

def get_batch_memory(batch):

	batch_mem_d = {}
	for k,v in batch.items():
		if isinstance(v, th.Tensor):
			batch_mem_d[k] = v.element_size()*v.nelement()
		elif isinstance(v, Batch):
			batch_mem_d[k] = pyg.profile.get_data_size(v)
		elif isinstance(v, list):
			batch_mem_d[k] = sum(sys.getsizeof(x) for x in v)
		else:
			raise ValueError(f"Unsupported type: {type(v)}")
	batch_mem_total = sum(batch_mem_d.values())
	return batch_mem_d, batch_mem_total

def find_largest_batch(dl):

	batch_mem_ds, batch_mem_totals = [], []
	for batch in iter(dl):
		batch_mem_d, batch_mem_total = get_batch_memory(batch)
		batch_mem_ds.append(batch_mem_d)
		batch_mem_totals.append(batch_mem_total)
	argmax_idx = np.argmax(batch_mem_totals)
	return batch_mem_ds[argmax_idx], batch_mem_totals[argmax_idx]
		
class SpecMolFragDynamicBatchSampler(BatchSampler):
	"""Dynamically adds samples to a mini-batch up to a maximum size  either based on number of nodes on frag DAG or number of edges on frag DAG.
	 This is used to avoid CUDA OOM errors, implmentaion is inspired by PyG DynamicBatchSampler, and this should be used to replace default BatchSampler
	 This should have the same random sampling beheivor as RandomSampler
	"""
	def __init__(self, data_source: SpecMolFragDataset, max_num: int, limited_by: str = 'frag_edge',
			  	 	skip_too_big: bool = False, num_samples  = None, 
					return_batch_at = 0, sampler=None) -> None:
		"""_summary_

		Args:
			dataset (Dataset): 
			max_num (int): _description_
			mode (str, optional): _description_. Defaults to 'node'.
			shuffle (bool, optional): Samples elements randomly each epoch. Defaults to False.
			skip_too_big (bool, optional): _description_. Defaults to False.
			num_samples (Optional[int], optional): num of samples to draw. Defaults to None. if None set to all samples in dataset
			generator
			return_batch_at 
			max_batch_sample_size
		Raises:
			ValueError: _description_
			ValueError: _description_
		"""
		if not isinstance(max_num, int) or max_num <= 0:
			raise ValueError("`dag_node` should be a positive integer value "
							"(got {max_num}).")
		if limited_by not in ['frag_node', 'frag_edge']:
			raise ValueError("`limited_by` choice should be either "
							f"'frag_node' or 'frag_edge' (got '{limited_by}').")

		if num_samples is None:
			num_samples = len(data_source)

		self.data_source = data_source
		self._max_num = max_num
		self._limited_by = limited_by
		self._skip_too_big = skip_too_big
		self._max_sampling_step = num_samples
		self._return_batch_at = return_batch_at
		self._batches = []
		self._data_meta = []
		self.sampler = sampler
		self._pre_load_batches()
		self._pre_compute_batches()

	def _pre_load_batches(self):

		# get data meta once and cache them
		assert len(self._data_meta) == 0, len(self._data_meta)
		expected_total = 0
		warning_msg = ""
		for dataset_idx in tqdm(range(len(self.data_source)), desc="SpecMolFragDynamicBatchSampler:pre_load_batches"):
			data = self.data_source[dataset_idx]
			self._data_meta.append((data['frag_pyg'].num_nodes,data['frag_pyg'].num_edges))
			n = self._data_meta[dataset_idx][0] if self._limited_by == 'frag_node' else self._data_meta[dataset_idx][1]
			if not (n > self._max_num and self._skip_too_big):
				expected_total += 1
			else:
				warning_msg += "Size of data sample at index " +\
					f"{dataset_idx} is larger than " +\
					f"{self._max_num} {self._limited_by}s " +\
					f"Got {n} {self._limited_by}s." +\
					"This sample can not fit into batch. "
				if self._skip_too_big:
					warning_msg += "Sampler will skip this to prevent CUDA OOM ERROR \n"
				else:
					warning_msg += "Attempting to fit this in to batch, this may cause CUDA OOM ERROR  \n"
				#warnings.warn(warning_msg)
				#print(warning_msg)
		if warning_msg != "":
			print("[specmolfrag_dynamic_batch_sampler]", warning_msg)
		print(f"[SpecMolFragDynamicBatchSampler] Expecting {expected_total}/{len(self.data_source)} samples with skip_too_big:{self._skip_too_big}")

	def _pre_compute_batches(self):
		
		self._batches = []

		if self.sampler is not None:
			indices = th.tensor([idx for idx in self.sampler])
		else:
			indices = th.arange(len(self.data_source), dtype=th.long)

		# limited index to _max_sampling_step
		indices = indices[:self._max_sampling_step]

		num_processed = 0
		batch = []
		batch_n = 0
		batch_filled = False

		# Fill batch
		for idx in indices:
			# Size of sample
			n = self._data_meta[idx.item()][0] if self._limited_by == 'frag_node' else self._data_meta[idx.item()][1]
			if n > self._max_num and self._skip_too_big:
				continue 
			# check batch_filled condition
			if batch_n + n > self._max_num:
				# no more budget left, mini-batch filled
				batch_filled = True	
			# check we need return at this point for ga
			if self._return_batch_at > 0 \
				and num_processed > 0 \
				and num_processed % self._return_batch_at == 0:
				# Mini-batch filled
				batch_filled = True
			if batch_filled:
				self._batches.append(batch)
				batch_n = 0
				batch = []
				batch_filled = False
			# Add sample to current batch
			batch.append(idx.item())
			num_processed += 1
			batch_n += n
			
		if len(batch) > 0:
			self._batches.append(batch)
		print(f"[SpecMolFragDynamicBatchSampler] Batch indices computed, Expecting {len(self._batches)} mini-batch for next epoch")

	def __iter__(self) -> Iterator[List[int]]:
		""" we use a pre computed batche list, this way we could have a correct batch size
			PL uses last batch in progress to toggle val step in training

		Yields:
			Iterator[List[int]]: _description_
		"""
		for batch in self._batches:
			yield batch

	def __len__(self) -> int:
		""" Note The __len__() method isn't strictly required by DataLoader, but is expected in any calculation involving the length of a DataLoader.
			ref: https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler
		Returns:
			int: length of datasource
		"""
		return len(self._batches)

class GroupSampler(Sampler):

	def __init__(self, data_source: SpecMolFragDataset, sample_k=None, generator=None) -> None:
		"""_summary_

		Args:
			data_source (SpecMolFragDataset): _description_
			sample_k (_type_, optional): _description_. Defaults to None.
		"""
		self.data_source = data_source
		self.num_samples = None
		self._data_meta_d = {}
		self.sample_k = 3 if sample_k is None else sample_k
		self.generator = generator
		self._pre_compute_meta()
		self._pre_compute_batches()

	def _pre_compute_meta(self):

		for dataset_idx in range(len(self.data_source)):
			data = self.data_source[dataset_idx]
			group_id = data['group_id'].item()
			if group_id not in self._data_meta_d:
				self._data_meta_d[group_id] = []
			self._data_meta_d[group_id].append(dataset_idx)
		
		for group_id in self._data_meta_d:
			self._data_meta_d[group_id] =  th.tensor(self._data_meta_d[group_id])

	def _pre_compute_batches(self):

		if self.generator is None:
			seed = int(th.empty((), dtype=th.int64).random_().item())
			generator = th.Generator()
			generator.manual_seed(seed)
		else:
			generator = self.generator

		sampled_indices = []
		for group_id in self._data_meta_d:
			group_indices = self._data_meta_d[group_id]
			sampled_group_indices = group_indices[th.randperm(min(len(group_indices),self.sample_k),generator=generator)]
			sampled_indices.append(sampled_group_indices)
		sampled_indices = th.cat(sampled_indices)
		self.sampled_indices = th.randperm(len(sampled_indices),generator=generator)
		self.num_samples = len(sampled_indices)

	def __iter__(self) -> Iterator[int]:
		""" 
		Yields:
			Iterator[List[int]]: _description_
		"""
		
		for i in range(self.num_samples):
			yield self.sampled_indices[i].item()

	def __len__(self) -> int:
		""" Note The __len__() method isn't strictly required by DataLoader
			ref: https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler
		Returns:
			int: length of datasource
		"""
		assert self.num_samples is not None
		return self.num_samples 
	
def get_group_sampler(ds: SpecMolFragDataset, sampler_type: str, avg_per_group: int, generator) -> WeightedRandomSampler:
	"""get WeightedRandomSampler based on input

	Args:
		ds (SpecMolFragDataset): _description_
		config_d (dict): _description_

	Returns:
		_type_: _description_
	"""
	group_ids, mol_ids, spec_per_group, spec_per_mol, group_per_mol = ds.get_group_mol_stats()

	if sampler_type == "group":
		sample_weights = 1.0 / spec_per_group
	elif sampler_type  == "mol":
		sample_weights = 1.0 / spec_per_mol
	elif sampler_type  == "group_mol":
		sample_weights = 1.0 / (spec_per_group * group_per_mol)
	else:
		return None
	
	spec_per_group_2 = th.mean(th.unique(group_ids,return_counts=True)[1].float(),dim=0)
	num_samples = th.ceil(len(ds) / spec_per_group_2 * avg_per_group).long()
	num_samples = min(num_samples.item(), len(ds))
	sampler = WeightedRandomSampler(
		sample_weights,
		num_samples=num_samples,
		replacement=False,
		generator=generator)
	return sampler

