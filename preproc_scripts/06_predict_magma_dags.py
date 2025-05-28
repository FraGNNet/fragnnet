import sys
import argparse
import logging
import json
from pathlib import Path
import numpy as np
import copy
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from pprint import pformat
import torch as th
import yaml
import rdkit
from rdkit import Chem
from rdkit import RDLogger
import shutil

from fragnnet.iceberg.pl_model import IcebergGenPL
from fragnnet.utils.misc_utils import booltype
from fragnnet.iceberg.dataset import SpecMolMagmaGenDataset
import fragnnet.utils.frag_utils as frag_utils
from fragnnet.utils.proc_utils import filter_spec_mol, merge_spec_df
from fragnnet.utils.misc_utils import booltype
from fragnnet.utils.data_utils import seq_apply, par_apply, rdkit_import

def run_magma_gen(
	num_entries: int,
	magma_dp: str,
	proc_dp: str,
	gen_ckpt_fp: str,
	allowed_elements: list[str],
	dsets: list[str],
	threshold: float,
	max_nodes: int,
	parallel: bool,
	group_ids: list[int],
	gpu: bool,
	rseed: int,
):

	th.manual_seed(rseed)
	device = "cuda" if gpu else "cpu"

	# setup model
	gen_ckpt_file = Path(gen_ckpt_fp)
	assert gen_ckpt_file.exists(), f"Missing gen ckpt file: {gen_ckpt_file}"
	pl_model = IcebergGenPL.load_from_checkpoint(gen_ckpt_fp, map_location=device)

	magma_gen_dir = Path(magma_dp)
	magma_gen_dir.mkdir(exist_ok=True)
	magma_gen_args_file = magma_gen_dir / "args.yml"
	magma_gen_tree_dir = magma_gen_dir / "magma_tree"
	magma_gen_tree_dir.mkdir(exist_ok=True)
	magma_gen_ckpt_file = magma_gen_dir / "gen_ckpt.ckpt"
	shutil.copy(gen_ckpt_file, magma_gen_ckpt_file)

	args = dict(
		num_entries=num_entries,
		magma_dp=magma_dp,
		proc_dp=proc_dp,
		gen_ckpt_fp=gen_ckpt_fp,
		threshold=threshold,
		max_nodes=max_nodes,
		parallel=parallel,
		gpu=gpu,
		rseed=rseed
	)
	yaml_args = yaml.dump(args)
	with open(magma_gen_args_file, "w") as f:
		f.write(yaml_args)

	proc_dir = Path(proc_dp)
	mol_df_file = proc_dir / "mol_df.pkl"
	assert mol_df_file.exists(), f"Missing mol_df file: {mol_df_file}"
	mol_df = pd.read_pickle(mol_df_file)
	spec_df_file = proc_dir / "spec_df.pkl"
	assert spec_df_file.exists(), f"Missing spec_df file: {spec_df_file}"
	spec_df = pd.read_pickle(spec_df_file)

	# filter and merge
	spec_df, mol_df = filter_spec_mol(
		spec_df,
		mol_df,
		elements=allowed_elements,
		dsets=dsets,
		prec_types=["[M+H]+"],
		num_entries=num_entries
	)
	m_spec_df = merge_spec_df(spec_df,renormalize=False,sum_ints=False)
	if len(group_ids) > 0:
		m_spec_df = m_spec_df[m_spec_df["group_id"].isin(group_ids)].reset_index(drop=True)
	both_df = m_spec_df[["group_id","mol_id","prec_type"]].merge(mol_df[["mol_id","smiles"]],on="mol_id",how="inner")
	spec_entries = both_df[["group_id","prec_type","smiles"]].to_dict("records")

	tree_processor = SpecMolMagmaGenDataset.init_tree_processor(
		pe_embed_k=pl_model.hparams.magma_params["pe_embed_k"],
		root_encode=pl_model.hparams.magma_params["root_encode"],
		add_hs=pl_model.hparams.magma_params["add_hs"]
	)

	with th.no_grad():
		pl_model.eval()
		pl_model.freeze()
		pl_model.to(device)

		def single_predict_mol(entry):
			Chem = rdkit_import("rdkit.Chem")[0]
			th.set_num_threads(8)
			smi = entry["smiles"]
			name = str(entry["group_id"])
			adduct = entry["prec_type"]
			inchi = Chem.MolToInchi(Chem.MolFromSmiles(smi))
			pred = pl_model.predict_mol(
				smi,
				adduct=adduct,
				threshold=threshold,
				device=device,
				max_nodes=max_nodes,
				tree_processor=tree_processor
			)
			output = {
				"root_inchi": inchi,
				"name": name,
				"frags": pred,
			}
			out_file = magma_gen_tree_dir / f"{name}.json"
			with open(out_file, "w") as fp:
				json.dump(output, fp, indent=2)
			return output

		spec_entry_iter = tqdm(
			spec_entries,
			desc=pformat(single_predict_mol),
			total=len(spec_entries)
		)
		if not parallel:
			results = seq_apply(spec_entry_iter,single_predict_mol)
		else:
			results = par_apply(spec_entry_iter,single_predict_mol)

def get_args():
	"""get args"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--num_entries", type=int, default=-1)
	parser.add_argument("--magma_dp", type=str, default="data/magma/inten/nist_2")
	parser.add_argument("--proc_dp", type=str, default="data/proc/nist")
	parser.add_argument("--gen_ckpt_fp", type=str, default="data/magma/ckpt/nist_2.ckpt")
	parser.add_argument("--allowed_elements", type=str, nargs="+", default=frag_utils.ELEMENTS)
	parser.add_argument("--dsets", type=str, nargs="+", default=["nist"])
	parser.add_argument("--parallel", type=booltype, default=False)
	parser.add_argument("--gpu", type=booltype, default=True)
	parser.add_argument("--rseed", type=int, default=420)
	parser.add_argument("--group_ids", type=int, nargs="+", default=[])
	### gen params
	parser.add_argument("--threshold", type=float, default=0.0)
	parser.add_argument("--max-nodes", type=int, default=100)
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	# Define basic logger
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s: %(message)s",
		handlers=[
			logging.StreamHandler(sys.stdout),
		],
	)
	RDLogger.DisableLog("rdApp.*")
	args = get_args()
	kwargs = args.__dict__
	run_magma_gen(**kwargs)
