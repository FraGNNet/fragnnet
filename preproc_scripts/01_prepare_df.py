import zipfile
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import argparse
import glob
from fragnnet.utils.data_utils import rdkit_import, seq_apply, par_apply, parse_mass_gym_ce_str
from fragnnet.utils.misc_utils import flatten_lol

MSP_KEY_DICT = {
	"Precursor_type": "prec_type",
	"Spectrum_type": "spec_type",
	"PrecursorMZ": "prec_mz",
	"Instrument_type": "inst_type",
	"Collision_energy": "col_energy",
	"Ion_mode": "ion_mode",
	"Ionization": "ion_type",
	"ID": "spec_id",
	"Collision_gas": "col_gas",
	"Pressure": "pressure",
	"Num peaks": "num_peaks",
	"MW": "mw",
	"ExactMass": "exact_mass",
	"CASNO": "cas_num",
	# "NISTNO": "dset_spec_id",
	"Name": "name",
	"MS": "peaks",
	"SMILES": "smiles",
	"Rating": "rating",
	"Frag_mode": "frag_mode",
	"Instrument": "inst",
	"RI": "ri",
	# "DB#": "dset_spec_id_2",
	"Notes": "notes", # NIST only
	"Formula": "formula",
	"InChIKey": "inchikey"
}

MS_KEY_DICT = {
	">compound": "name",
	# ">ionization": "prec_type",
	# ">formula": "formula",
	">parentmass": "prec_mz",
	# "#smiles": "smiles",
	"#instrumentation": "inst_type",
	# ">collision": "col_energy",
}
META_KEYS = list(MS_KEY_DICT.keys())
SPEC_KEYS = [">ms1peaks",">ms1merged",">ms2peaks",">ms2merged",">collision"]


def extract_info_from_comments(comments,key):

	start_idx = comments.find(key)
	if start_idx == -1:
		return None
	start_idx += len(key)+1 # +1 is for =
	end_idx = start_idx+1
	cur_char = comments[end_idx]
	while cur_char != "\"":
		end_idx += 1
		cur_char = comments[end_idx]
	value = comments[start_idx:end_idx]
	return value


"""
Convert data from MSMS database format to pandas dataframe with JSON
No type conversions or filtering: all of that is done downstream
"""

def preproc_msp(msp_fp,keys,num_entries):
	""" """

	with open(msp_fp) as f:
		raw_data_lines = f.readlines()
	# split CAS# and NIST# on different lines
	_raw_data_lines = []
	for line in raw_data_lines:
		if "CAS#" in line and "NIST#" in line:
			assert ";" in line, line
			split_lines = line.split(";")
			_raw_data_lines.append(split_lines[0].rstrip(";")+"\n")
			_raw_data_lines.append(split_lines[1].lstrip())
		else:
			_raw_data_lines.append(line)
	raw_data_lines = _raw_data_lines
	raw_data_list = []
	raw_data_item = {key: None for key in keys}
	read_ms = False
	for raw_l in tqdm(raw_data_lines,desc=f"> processing {msp_fp}",total=len(raw_data_lines)):
		if num_entries > -1 and len(raw_data_list) == num_entries:
			break
		raw_l = raw_l.replace('\n', '')
		if raw_l == '':
			# check if double line
			if all(v is None for v in raw_data_item.values()):
				assert not read_ms	
			else:
				raw_data_list.append(raw_data_item.copy())
				raw_data_item = {key: None for key in keys}
				read_ms = False
		elif read_ms:
			raw_data_item['MS'] = raw_data_item['MS'] + raw_l + '\n'
		else:
			if "RI:" in raw_l:
				raw_l_split = raw_l.split(':')
			else:
				raw_l_split = raw_l.split(': ')
			assert len(raw_l_split) >= 2, len(raw_l_split)
			key = raw_l_split[0]
			if key == "Num peaks" or key == "Num Peaks":
				assert len(raw_l_split) == 2, raw_l_split
				value = raw_l_split[1]
				raw_data_item['Num peaks'] = int(value)
				raw_data_item['MS'] = ''
				read_ms = True
			elif key == "Comments":
				comments = ": ".join(raw_l_split[1:])
				smiles = extract_info_from_comments(comments,"computed SMILES")
				rating = extract_info_from_comments(comments,"MoNA Rating")
				frag_mode = extract_info_from_comments(comments,"fragmentation mode")
				if not (smiles is None):
					raw_data_item["SMILES"] = smiles
				if not (rating is None):
					raw_data_item["Rating"] = rating
				if not (frag_mode is None):
					raw_data_item["Frag_mode"] = frag_mode
			elif key in keys:
				value = raw_l_split[1]
				raw_data_item[key] = value
	if num_entries > -1:
		import pdb; pdb.set_trace()
	msp_df = pd.DataFrame(raw_data_list)
	# drop all-NaN rows
	msp_df = msp_df.dropna(axis=0,how="all")
	return msp_df

def preproc_nist_mol(mol_dp):
	""" read in all .MOL files and return a df """

	mol_fp_list = glob.glob(os.path.join(mol_dp,"*.MOL"))
	def proc_mol_file(mol_fp):
		modules = rdkit_import("rdkit.Chem","rdkit.Chem.rdinchi","rdkit.Chem.AllChem")
		Chem = modules[0]
		mol_fn = os.path.basename(os.path.normpath(mol_fp))
		spec_id = mol_fn.lstrip("ID").rstrip(".MOL")
		mol = Chem.MolFromMolFile(mol_fp, sanitize=True)
		if not (mol is None):
			smiles = Chem.MolToSmiles(mol)
		else:
			smiles = None
		entry = dict(
			spec_id=spec_id,
			smiles=smiles
		)
		return entry
	mol_fp_iter = tqdm(mol_fp_list,desc="> proc_mol_files",total=len(mol_fp_list))
	mol_df_entries = par_apply(mol_fp_iter,proc_mol_file)
	mol_df = pd.DataFrame(mol_df_entries)
	return mol_df

def merge_and_check(msp_df,mol_df,rename_dict):

	# get rid of the columns that you don't care about
	msp_bad_cols = set(msp_df.columns)-set(rename_dict.keys())
	msp_df = msp_df.drop(columns=msp_bad_cols)
	# rename to be consistent
	msp_df = msp_df.rename(columns=rename_dict)
	if mol_df is None:
		assert not msp_df["smiles"].isna().all()
		msp_df.loc[:,"spec_id"] = np.arange(msp_df.shape[0])
		spec_df = msp_df
	else:
		assert msp_df["smiles"].isna().all()
		assert not msp_df["spec_id"].isna().all()
		# merge with mol on spec_id
		msp_df = msp_df.drop(columns=["smiles"])
		spec_df = pd.merge(msp_df,mol_df,how="inner",on="spec_id")
	if "dset_spec_id" not in spec_df.columns:
		spec_df.loc[:,"dset_spec_id"] = spec_df["spec_id"]
	print(spec_df.isna().sum())
	spec_df = spec_df.reset_index(drop=True)
	return spec_df


def preproc_ms_files(ms_dp,ms_meta_fp,keys,num_entries):

	# read in meta file
	ms_meta_df = pd.read_csv(ms_meta_fp,sep="\t")
	ms_meta_df = ms_meta_df.rename(columns={"ionization":"prec_type","spec":"dset_spec_id_base"}).drop(columns=["name"])
	assert not ms_meta_df.isna().any().any(), ms_meta_df.isna().any()
	# get all .ms files
	ms_fp_list = glob.glob(os.path.join(ms_dp,"*.ms"))
	if num_entries > -1:
		ms_fp_list = ms_fp_list[:num_entries]
	# read in all .ms files
	def proc_ms_file(ms_fp):
		ms_fn = os.path.basename(os.path.normpath(ms_fp))
		dset_spec_id = ms_fn.removesuffix(".ms")
		with open(ms_fp,'r') as f:
			ms_lines = f.readlines()
		ms_meta_entry = {}
		ms_levels, ms_ces, ms_peakses = [], [], []
		cur_level, cur_ce, cur_peaks = None, None, []
		in_spec = False
		for ms_line in ms_lines:
			ms_line = ms_line.strip()
			if in_spec:
				if ms_line == "":
					in_spec = False
					ms_levels.append(cur_level)
					ms_ces.append(cur_ce)
					ms_peakses.append("\n".join(cur_peaks))
					cur_level, cur_ce, cur_peaks = None, None, []
				else:
					mz_ints = ms_line
					cur_peaks.append(mz_ints)
			else:
				for meta_key in META_KEYS:
					if ms_line.startswith(meta_key):
						value = ms_line.removeprefix(meta_key+" ")
						# if meta_key == ">ionization":
						# 	value = value.replace(" ","")
						ms_meta_entry[MS_KEY_DICT[meta_key]] = value
				for spec_key in SPEC_KEYS:
					if ms_line.startswith(spec_key):
						in_spec = True
						if ms_line.startswith(">ms1peaks") or ms_line.startswith(">ms1merged"):
							cur_level = 1
						else:
							cur_level = 2
						if ms_line.startswith(">collision"):
							cur_ce = ms_line.removeprefix(">collision ")
		# check if still in_spec
		if in_spec:
			in_spec = False
			ms_levels.append(cur_level)
			ms_ces.append(cur_ce)
			ms_peakses.append("\n".join(cur_peaks))
			cur_level, cur_ce, cur_peaks = None, None, []
		# flatten entries
		ms_entries = []
		for idx, (ms_level, ms_ce, ms_peaks) in enumerate(zip(ms_levels,ms_ces,ms_peakses)):
			if ms_level == 2:
				ms_entry = dict(
					dset_spec_id_base=dset_spec_id, # for debugging
					dset_spec_id=dset_spec_id+f"_{idx}",
					col_energy=ms_ce,
					peaks=ms_peaks,
					**ms_meta_entry
				)
				ms_entries.append(ms_entry)
		assert len(ms_entries) > 0, (ms_fp, ms_lines)
		return ms_entries
	ms_fp_iter = tqdm(ms_fp_list,desc="> proc_ms_files",total=len(ms_fp_list))
	ms_df_entries = seq_apply(ms_fp_iter,proc_ms_file)
	ms_df_entries = flatten_lol(ms_df_entries)
	ms_df = pd.DataFrame(ms_df_entries)
	# drop all-NaN rows
	ms_df = ms_df.dropna(axis=0,how="all")
	# add metadata
	ms_df = ms_df.merge(ms_meta_df[["dset_spec_id_base","prec_type","formula","smiles"]],on=["dset_spec_id_base"],how="inner")
	# add ion_mode
	ms_df.loc[:,"ion_mode"] = "P"
	# add spec_type
	ms_df.loc[:,"spec_type"] = "MS2"
	# add spec_id (this will be relabeled later)
	ms_df.loc[:,"spec_id"] = np.arange(len(ms_df))
	# add extra columns for compatibility
	for value in MSP_KEY_DICT.values():
		if value not in ms_df.columns:
			ms_df.loc[:,value] = np.nan
	# drop unnecessary columns
	ms_df = ms_df.drop(columns=["dset_spec_id_base"])
	ms_df = ms_df.reset_index(drop=True)
	return ms_df

def process_ms_gym(ms_gym_tsv_fp, subset):
	ms_df = pd.read_csv(ms_gym_tsv_fp, sep="\t")
 
	# only use on in simulation_challenge
	ms_df = ms_df.sort_values(by='simulation_challenge', ascending=False)
	print(f"> processs ms gym file {ms_gym_tsv_fp}")
	print(f"> using ms gym subset {subset}")
	print("> if you are doing simulation challenge make sure use only simulation_challenge subset")
 
	if subset == 'simulation_challenge':
		ms_df = ms_df[ms_df['simulation_challenge']]
		print(f"> {len(ms_df)} rows selected for subset {subset}")
	elif subset == 'extra':
		ms_df = ms_df[~ms_df['simulation_challenge']]
		print(f"> {len(ms_df)} rows selected for subset {subset}")
	elif subset != 'all':
		print(f">warningm unknown subset [{subset}] to msygm. Every row will be included")
  
	ms_df["peaks"] = ms_df.apply(lambda x: "\n".join([f"{m} {i}" for m,i in zip(x["mzs"].split(","),x["intensities"].split(","))]),axis=1)
	ms_df = ms_df.drop(columns=["mzs","intensities"])
	ms_df.rename(columns={"precursor_mz":"prec_mz", 
						   "adduct":"prec_type",
							"precursor_mz" :"prec_mz",
						   "parent_mass":"exact_mass",
						   "instrument_type":"inst_type",
						   "collision_energy":"col_energy",
						   "identifier":"dset_spec_id"},
				  	inplace=True)
	
	ms_df["col_energy"], ms_df["normalized"], ms_df["ramped"] = zip(*ms_df["col_energy"].apply(parse_mass_gym_ce_str))

	#ms_df = ms_df.astype({"prec_mz":float,"exact_mass":float})

	# add ion_mode
	ms_df.loc[:,"ion_mode"] = "P"
	# add spec_type
	ms_df.loc[:,"spec_type"] = "MS2"
	# add ion_type
	ms_df.loc[:,"ion_type"] = "ESI"
	# add extra columns for compatibility
	for value in MSP_KEY_DICT.values():
		if value not in ms_df.columns:
			ms_df.loc[:,value] = np.nan
	
	split_df = ms_df[["dset_spec_id","fold"]]

	return ms_df, split_df
 
if __name__ == "__main__":

	## TODO: add support for NPLLIB
	parser = argparse.ArgumentParser()
	parser.add_argument('--msp_file', type=str, required=False)
	parser.add_argument('--mol_dir', type=str, required=False)
	parser.add_argument('--pkl_file', type=str, required=False)
	parser.add_argument('--ms_dir', type=str, required=False)
	parser.add_argument('--ms_meta_file', type=str, required=False)
	parser.add_argument('--input_format', type=str, choices=["msp","msp+mol","ms+meta", "msp+pkl", "ms_gym", "ms_gym_extra"], required=True)
	parser.add_argument('--output_name', type=str, required=True)
	parser.add_argument('--raw_data_dp', type=str, default='data/raw')
	parser.add_argument('--output_dp', type=str, default='data/df')
	parser.add_argument('--num_entries', type=int, default=-1)
	parser.add_argument('--output_format', type=str, default="json", choices=["json","csv"])
	parser.add_argument('--msp_dset_spec_id', type=str, default="NISTNO", choices=["NISTNO","DB#","NIST#","ID"])
	parser.add_argument('--ms_gym_tsv', type=str, required=False)
	args = parser.parse_args()

	os.makedirs(args.output_dp,exist_ok=True)

	if args.input_format in ["msp","msp+mol","msp+pkl"]:
		# select dset_spec_id
		if args.msp_dset_spec_id not in MSP_KEY_DICT:
			MSP_KEY_DICT[args.msp_dset_spec_id] = "dset_spec_id"
		else:
			assert "dset_spec_id" not in MSP_KEY_DICT.values()
		# nist or mona
		msp_fp = os.path.join(args.raw_data_dp,args.msp_file)
		assert os.path.isfile(msp_fp), msp_fp
		msp_df = preproc_msp(msp_fp,MSP_KEY_DICT.keys(),args.num_entries)
		if args.input_format == "msp+mol":
			mol_dp = os.path.join(args.raw_data_dp,args.mol_dir)
			assert os.path.isdir(mol_dp), mol_dp
			mol_df = preproc_nist_mol(mol_dp)
		elif args.input_format == "msp+pkl":
			pkl_fp = os.path.join(args.raw_data_dp,args.pkl_file)
			assert os.path.isfile(pkl_fp), pkl_fp
			nist_id_map = pd.read_pickle(pkl_fp)
			nist_id_map	= nist_id_map.rename(columns={"nist_id":args.msp_dset_spec_id,"smiles":"SMILES"})
			msp_df = msp_df.drop(columns=["SMILES"]).merge(nist_id_map,on=[args.msp_dset_spec_id],how="left")
			mol_df = None
		else:
			mol_df = None
		spec_df = merge_and_check(msp_df,mol_df,MSP_KEY_DICT)
	elif args.input_format == "ms+meta":
		# npllib
		# just use NISTNO as default
		MSP_KEY_DICT["NISTNO"] = "dset_spec_id"
		ms_dp = os.path.join(args.raw_data_dp,args.ms_dir)
		assert os.path.isdir(ms_dp), ms_dp
		ms_meta_fp = os.path.join(args.raw_data_dp,args.ms_meta_file)
		assert os.path.isfile(ms_meta_fp), ms_meta_fp
		spec_df = preproc_ms_files(ms_dp,ms_meta_fp,MSP_KEY_DICT.keys(),args.num_entries)
	elif args.input_format == "ms_gym":
		spec_df, split_df = process_ms_gym(os.path.join(args.raw_data_dp,args.ms_gym_tsv), 'simulation_challenge')
		split_df_fp = os.path.join(args.output_dp,f"{args.output_name}_fold.csv")
		split_df.to_csv(split_df_fp,index=False)
	elif args.input_format == "ms_gym_extra":
		spec_df, split_df = process_ms_gym(os.path.join(args.raw_data_dp,args.ms_gym_tsv), 'extra')
		split_df_fp = os.path.join(args.output_dp,f"{args.output_name}_fold.csv")
		split_df.to_csv(split_df_fp,index=False)
	else:
		raise ValueError(f"Invalid input format: {args.input_format}")
	
	# save files
	spec_df_fp = os.path.join(args.output_dp,f"{args.output_name}_df.{args.output_format}")
	if args.output_format == "json":
		spec_df.to_json(spec_df_fp)
	else:
		assert args.output_format == "csv"
		spec_df.to_csv(spec_df_fp,index=False)