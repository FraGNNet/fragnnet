import importlib
import re
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from pprint import pformat, pprint
from typing import Tuple, Union, List, TYPE_CHECKING
import pyteomics
from pyteomics.mass import Composition
import rdkit.RDLogger as RDLogger

from fragnnet.utils.misc_utils import np_temp_seed, none_or_nan
from fragnnet.utils.formula_utils import MASS, ELECTRON_MASS, parse_formula, NEUTRON_MASS

# don't do expesive import in runtime
if TYPE_CHECKING:
	import rdkit

JOBLIB_BACKEND = "loky"
JOBLIB_N_JOBS = joblib.cpu_count()
JOBLIB_TIMEOUT = 10800 # default is 300 seconds for "loky"

def rdkit_import(*module_strs:List[str]) -> Tuple:
	""" until function for import rdkit modules

	Returns:
		tuple: tuple of loaded modules
	"""
	RDLogger = importlib.import_module("rdkit.RDLogger")
	RDLogger.DisableLog('rdApp.*')
	modules = []
	for module_str in module_strs:
		modules.append(importlib.import_module(module_str))
	return tuple(modules)

def normalize_ints(ints:List[float]) -> List[float]:
	"""normalize intensities to 0-1.0

	Args:
		ints (List[float]): list of intensities

	Returns:
		List[float]: normalized intensities
	"""
	total_ints = sum(ints)
	ints = [ints[i] / total_ints for i in range(len(ints))]
	return ints

def randomize_smiles(smiles, rseed, isomeric=False, kekule=False):
	"""Perform a randomization of a SMILES string must be RDKit sanitizable"""
	if rseed == -1:
		return smiles
	modules = rdkit_import("rdkit.Chem")
	Chem = modules[0]
	m = Chem.MolFromSmiles(smiles)
	assert not (m is None)
	ans = list(range(m.GetNumAtoms()))
	with np_temp_seed(rseed):
		np.random.shuffle(ans)
	nm = Chem.RenumberAtoms(m,ans)
	smiles = Chem.MolToSmiles(nm, canonical=False, isomericSmiles=isomeric, kekuleSmiles=kekule)
	assert not (smiles is None)
	return smiles

def split_smiles(smiles_str:str) -> List:
	"""Tokenize a SMILES molecule use regex

	Args:
		smiles_str (str): _description_

	Returns:
		List: tokens
	"""
	token_list = []
	pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
	regex = re.compile(pattern)
	token_list = [token for token in regex.findall(smiles_str)]
	assert smiles_str == ''.join(token_list)
	return token_list

def list_replace(l,d):
	return [d[data] for data in l]

def mol_from_inchi(inchi:str, standardize:bool = True) -> "rdkit.Chem.rdchem.Mol":
	"""return a rdkit mol from inchi str

	Args:
		inchi (str): inchi string, np.nan if fails

	Returns:
		rdkit.Chem.rdchem.Mol: rdkit mol object
	"""
	modules = rdkit_import("rdkit.Chem")
	Chem = modules[0]
	try:
		mol = Chem.MolFromInchi(inchi)
		if standardize:
			mol = rdkit_standardize(mol)
	except:
		mol = np.nan
	if none_or_nan(mol):
		mol = np.nan
	return mol

def rdkit_standardize(mol:"rdkit.Chem.rdchem.Mol") -> "rdkit.Chem.rdchem.Mol":
	"""standardize given mol

	Args:
		mol (rdkit.Chem.rdchem.Mol): mol object

	Returns:
		rdkit.Chem.rdchem.Mol: rdkit mol object after standardize
	"""
	# adapted from https://bitsilla.com/blog/2021/06/standardizing-a-molecule-using-rdkit/
	modules = rdkit_import("rdkit.Chem","rdkit.Chem.MolStandardize","rdkit.Chem.MolStandardize.rdMolStandardize")
	rdMolStandardize = modules[-1]
	# removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
	mol = rdMolStandardize.Cleanup(mol) 
	# if many fragments, get the "parent" (the actual mol we are interested in) 
	mol = rdMolStandardize.FragmentParent(mol)
	# try to neutralize molecule
	uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
	mol = uncharger.uncharge(mol)
	# note that no attempt is made at reionization at this step
	# nor at ionization at some pH (rdkit has no pKa caculator)
	# the main aim to to represent all molecules from different sources
	# in a (single) standard way, for use in ML, catalogue, etc.
	te = rdMolStandardize.TautomerEnumerator() # idem
	mol = te.Canonicalize(mol)
	return mol

def mol_from_smiles(smiles,standardize:bool=True) -> "rdkit.Chem.rdchem.Mol":
	"""return a rdkit mol from smiles str

	Args:
		smiles (_type_): smiles str
		standardize (bool, optional): flag to apply standardize. Defaults to True.

	Returns:
		rdkit.Chem.rdchem.Mol: mol object, np.nan if fails
	"""
	modules = rdkit_import("rdkit.Chem")
	Chem = modules[0]
	try:
		mol = Chem.MolFromSmiles(smiles)
		if standardize:
			mol = rdkit_standardize(mol)
	except Exception as e:
		mol = np.nan
	if none_or_nan(mol):
		mol = np.nan
	return mol

def mol_to_smiles(mol:"rdkit.Chem.rdchem.Mol",canonical:bool=True,isomericSmiles:bool=False,kekuleSmiles:bool=False) -> str:
	""" get smiles string from give mol object

	Args:
		mol (rdkit.Chem.rdchem.Mol): mol object
		canonical (bool, optional):flag to get canonical smiles. Defaults to True.
		isomericSmiles (bool, optional): flag to get isomeric smiles. Defaults to False.
		kekuleSmiles (bool, optional): flag to get kekule version of smiles. Defaults to False.

	Returns:
		str: smile string, np.nan if fails
	"""
	modules = rdkit_import("rdkit.Chem")
	Chem = modules[0]
	try:
		smiles = Chem.MolToSmiles(mol,canonical=canonical,isomericSmiles=isomericSmiles,kekuleSmiles=kekuleSmiles)
	except Exception as e:
		smiles = np.nan
	if none_or_nan(smiles):
		smiles = np.nan
	return smiles

def mol_to_formula(mol:"rdkit.Chem.rdchem.Mol") -> str:
	"""get formula string from give mol object

	Args:
		mol (rdkit.Chem.rdchem.Mol): mol object

	Returns:
		str: formula, np.nan if fails
	"""
	modules = rdkit_import("rdkit.Chem.AllChem")
	AllChem = modules[0]
	try:
		formula = AllChem.CalcMolFormula(mol)
	except Exception as e:
		formula = np.nan
	return formula

def mol_to_inchikey(mol:"rdkit.Chem.rdchem.Mol") -> str:
	"""get inchikey string from give mol object

	Args:
		mol (rdkit.Chem.rdchem.Mol): mol object

	Returns:
		str: inchikey, np.nan if fails
	"""
	modules = rdkit_import("rdkit.Chem.inchi")
	inchi = modules[0]
	try:
		inchikey = inchi.MolToInchiKey(mol)
	except Exception as e:
		inchikey = np.nan
	return inchikey

def mol_to_inchikey_s(mol:"rdkit.Chem.rdchem.Mol") -> str:
	"""get inchikey string from give mol object

	Args:
		mol (rdkit.Chem.rdchem.Mol): mol object

	Returns:
		str: inchikey, np.nan if fails
	"""
	modules = rdkit_import("rdkit.Chem.inchi")
	inchi = modules[0]
	try:
		inchikey_s = inchi.MolToInchiKey(mol)[:14]
	except Exception as e:
		inchikey_s = np.nan
	return inchikey_s

def mol_to_inchi(mol:"rdkit.Chem.rdchem.Mol")  -> str:
	"""get inchi string from give mol object

	Args:
		mol (rdkit.Chem.rdchem.Mol): mol object

	Returns:
		str: inchi, np.nan if fails
	"""
	modules = rdkit_import("rdkit.Chem.inchi")
	inchi = modules[0]
	try:
		inchikey = inchi.MolToInchiKey(mol)
		inchikey_s = inchikey[:14]
	except Exception as e:
		inchikey_s = np.nan
	return inchikey_s

def mol_to_inchi(mol)  -> str:
	"""get mol weight from mol object

	Args:
		mol (_type_): mol object
		
	Returns:
		str: inchi
	"""
	modules = rdkit_import("rdkit.Chem.rdinchi")
	rdinchi = modules[0]
	try:
		inchi = rdinchi.MolToInchi(mol,options='-SNon')[0]
	except:
		inchi = np.nan
	return inchi

def mol_to_mol_weight(mol,exact=True):
	"""get mol weight from mol object
	Args:
		mol (_type_): mol object
		exact (bool, optional): flag to get extact mass. Defaults to True.

	Returns:
		float: mw
	"""
	modules = rdkit_import("rdkit.Chem.Descriptors")
	Desc = modules[0]
	try:
		if exact:
			mol_weight = float(Desc.ExactMolWt(mol))
		else:
			mol_weight = float(Desc.MolWt(mol))
	except Exception as e:
		mol_weight = np.nan
	return mol_weight

def mol_to_charge(mol:"rdkit.Chem.rdchem.Mol") -> int:
	""" get number of charges from mol object

	Returns:
		int: total number of charges, np.nan if fails
	"""
	modules = rdkit_import("rdkit.Chem.rdmolops")
	rdmolops = modules[0]
	try:
		charge = rdmolops.GetFormalCharge(mol)
	except Exception as e:
		charge = np.nan
	return charge

def mol_to_num_atoms(mol:"rdkit.Chem.rdchem.Mol", heavy:bool=True) -> int:
	""" get total number of atoms in get mol object

	Args:
		mol (rdkit.Chem.rdchem.Mol): mol object
		heavy (bool, optional): count heavy atom only. Defaults to True.

	Returns:
		int: total number of atoms, np.nan if fails
	"""
	try:
		num_atoms = mol.GetNumAtoms(onlyHeavy=heavy)
	except Exception as e:
		num_atoms = np.nan
	return num_atoms

def mol_to_num_bonds(mol:"rdkit.Chem.rdchem.Mol",heavy:bool=True) -> int:
	""" get total number of bondss in get mol object

	Args:
		mol (rdkit.Chem.rdchem.Mol): mol object
		heavy (bool, optional): count bond between heavy atom only. Defaults to True.

	Returns:
		int: total number of bonds, np.nan if fails
	"""
	try:
		num_bonds = mol.GetNumBonds(onlyHeavy=heavy)
	except:
		num_bonds = np.nan
	return num_bonds

def mol_to_num_radicals(mol:"rdkit.Chem.rdchem.Mol") -> int:
	"""get total number of radicals in the mol object

	Args:
		mol (rdkit.Chem.rdchem.Mol): mol object

	Returns:
		int: number of radicals, np.nan if fails
	"""
	try:
		num_radicals = 0
		for atom in mol.GetAtoms():
			num_radicals += atom.GetNumRadicalElectrons()
	except:
		num_radicals = np.nan
	return num_radicals

def check_neutral_charge(mol:"rdkit.Chem.rdchem.Mol")-> bool:
	"""check if mol is nertral

	Args:
		mol (rdkit.Chem.rdchem.Mol): _description_

	Returns:
		bool: _description_
	"""
	valid = mol_to_charge(mol) == 0
	return valid

def check_single_mol(mol:"rdkit.Chem.rdchem.Mol") -> bool:
	""" check if mol object is a single mol (connected graph)
	Args:
		mol (rdkit.Chem.rdchem.Mol): mol object

	Returns:
		bool: True if is single mol, else false
	"""
	modules = rdkit_import("rdkit.Chem.rdmolops")
	rdmolops = modules[0]
	try:
		num_frags = len(rdmolops.GetMolFrags(mol))
	except Exception as e:
		num_frags = np.nan
	valid = num_frags == 1
	return valid

def inchi_to_smiles(inchi:str) -> str:
	"""convert between inchi and smiles

	Args:
		inchi (str): inchi string

	Returns:
		str: smiles string
	"""
	try:
		mol = mol_from_inchi(inchi)
		smiles = mol_to_smiles(mol)
	except Exception as e:
		smiles = np.nan
	return smiles

def smiles_to_selfies(smiles:str) -> str:
	"""convert from smiles to selfies string

	Args:
		smiles (str): smiles string

	Returns:
		str: selfies
	"""
	sf, Chem = rdkit_import("selfies","rdkit.Chem")
	try:
		# canonicalize, strip isomeric information, kekulize
		mol = Chem.MolFromSmiles(smiles)
		smiles = Chem.MolToSmiles(mol,canonical=False,isomericSmiles=False,kekuleSmiles=True)
		selfies = sf.encoder(smiles)
	except Exception as e:
		selfies = np.nan
	return selfies

def get_tanmimoto(fp, other_fp):
	"""return tanmimoto distance, does not work with np yet

	Args:
		fp (_type_): _description_
		other_fp (_type_): _description_

	Returns:
		_type_: _description_
	"""
	ds = rdkit_import("rdkit.DataStructs")[0]
	return ds.DiceSimilarity(fp, other_fp)

def make_morgan_fingerprint(mol:"rdkit.Chem.rdchem.Mol", radius:int=3, covert_to_np = True)  -> np.array:
	"""make morgan fingerpirnt from mol object

	Args:
		mol (rdkit mol): mol object

	Returns:
		fp_arr
	"""
	modules = rdkit_import("rdkit.Chem.rdMolDescriptors","rdkit.DataStructs")
	rmd = modules[0]
	ds = modules[1]
	fp = rmd.GetHashedMorganFingerprint(mol,radius)
	if covert_to_np:
		fp_arr = np.zeros(1)
		ds.ConvertToNumpyArray(fp, fp_arr)
	else:
		fp_arr = fp
	return fp_arr

def make_rdkit_fingerprint(mol:"rdkit.Chem.rdchem.Mol") -> np.array:
	"""make rdkit fingerpirnt from mol object

	Args:
		mol (rdkit mol): mol object

	Returns:
		fp_arr:  np.array
	"""

	chem, ds = rdkit_import("rdkit.Chem","rdkit.DataStructs")
	fp = chem.RDKFingerprint(mol)
	fp_arr = np.zeros(1)
	ds.ConvertToNumpyArray(fp,fp_arr)
	return fp_arr

def make_maccs_fingerprint(mol:"rdkit.Chem.rdchem.Mol") -> np.array:
	"""make maccs fingerpirnt from mol object

	Args:
		mol (rdkit mol): mol object

	Returns:
		fp_arr:  np.array
	"""

	maccs, ds = rdkit_import("rdkit.Chem.MACCSkeys","rdkit.DataStructs")
	fp = maccs.GenMACCSKeys(mol)
	fp_arr = np.zeros(1)
	ds.ConvertToNumpyArray(fp,fp_arr)
	return fp_arr

def remove_stereochemistry(mol:"rdkit.Chem.rdchem.Mol"):
	"""_summary_

	Args:
		mol (rdkit.Chem.rdchem.Mol): _description_
	"""
	chem = rdkit_import("rdkit.Chem")[0]
	mol = chem.RemoveStereochemistry(mol)
	
def split_selfies(selfies_str:str) -> List[str]:
	"""method to splt selfies strngs to list of tokens

	Args:
		selfies_str (str): selfies string

	Returns:
		 List[str]: list of tokens
	"""
	selfies = importlib.import_module("selfies")
	selfies_tokens = list(selfies.split_selfies(selfies_str))
	return selfies_tokens

def seq_apply(iterator,func,need_unpack=False):

	result = []
	for i in iterator:
		if not need_unpack:
			result.append(func(i))
		else:
			result.append(func(*i))
	return result

def par_apply(iterator,func,need_unpack=False, return_as_generator=False, n_jobs=JOBLIB_N_JOBS):

	par_func = joblib.delayed(func)
	parallel = joblib.Parallel(
		backend=JOBLIB_BACKEND,
		n_jobs=n_jobs,
		timeout=JOBLIB_TIMEOUT,
		return_as="list" if not return_as_generator else "generator"
	)
	if not need_unpack:
		result = parallel(par_func(i) for i in iterator)
	else:
		result = parallel(par_func(*i) for i in iterator)
	return result

def par_apply_series(series,func, tqdm_leave = True):

	series_iter = tqdm(series.items(),desc=pformat(func),total=series.shape[0],leave=tqdm_leave)
	series_func = lambda tup: func(tup[1])
	result_list = par_apply(series_iter,series_func)
	result_series = pd.Series(result_list,index=series.index)
	return result_series

def seq_apply_series(series,func, tqdm_leave=True):

	series_iter = tqdm(series.items(),desc=pformat(func),total=series.shape[0],leave=tqdm_leave)
	series_func = lambda tup: func(tup[1])
	result_list = seq_apply(series_iter,series_func)
	result_series = pd.Series(result_list,index=series.index)
	return result_series

def par_apply_df_rows(df,func):

	df_iter = tqdm(df.iterrows(),desc=pformat(func),total=df.shape[0])
	df_func = lambda tup: func(tup[1])
	result_list = par_apply(df_iter,df_func)
	if isinstance(result_list[0],tuple):
		result_series = tuple([pd.Series(rl,index=df.index) for rl in zip(*result_list)])
	else:
		result_series = pd.Series(result_list,index=df.index)
	return result_series

def seq_apply_df_rows(df,func):

	df_iter = tqdm(df.iterrows(),desc=pformat(func),total=df.shape[0])
	df_func = lambda tup: func(tup[1])
	result_list = seq_apply(df_iter,df_func)
	if isinstance(result_list[0],tuple):
		result_series = tuple([pd.Series(rl,index=df.index) for rl in zip(*result_list)])
	else:
		result_series = pd.Series(result_list,index=df.index)
	return result_series

def parse_ace_str(ce_str):

	if none_or_nan(ce_str):
		return np.nan
	matches = {
		# nist ones
		r"^[\d]+[.]?[\d]*$": lambda x: float(x), # this case is ambiguous (float(x) >= 2. or float(x) == 0.)
		r"^[\d]+[.]?[\d]*[ ]?eV$": lambda x: float(x.rstrip(" eV")),
		r"^NCE=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*eV$": lambda x: float(x.split()[1].rstrip("eV")),
		# other ones
		r"^[\d]+[.]?[\d]*HCD$": lambda x: float(x.rstrip("HCD")),
		r"^CE [\d]+[.]?[\d]*$": lambda x: float(x.lstrip("CE ")),
	}
	for k,v in matches.items():
		# try:
		if re.match(k,ce_str):
			return v(ce_str)
		# except:
		# 	import pdb; pdb.set_trace()
	return np.nan

def parse_nce_str(ce_str):

	if none_or_nan(ce_str):
		return np.nan
	matches = {
		# nist ones
		r"^NCE=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*eV$": lambda x: float(x.split()[0].lstrip("NCE=").rstrip("%")),
		r"^NCE=[\d]+[.]?[\d]*%$": lambda x: float(x.lstrip("NCE=").rstrip("%")),
		r"^[\d]+% resonant relative/normalized$": lambda x: float(x.split("%")[0]),
		# other ones
		r"^[\d]+[.]?[\d]*$": lambda x: 100.*float(x) if float(x) < 2. else np.nan, # this case is ambiguous
		r"^[\d]+[.]?[\d]*[ ]?[%]? \(nominal\)$": lambda x: float(x.rstrip(" %(nominal)")),
		r"^HCD [\d]+[.]?[\d]*%$": lambda x: float(x.lstrip("HCD ").rstrip("%")),
		r"^[\d]+[.]?[\d]* NCE$": lambda x: float(x.rstrip("NCE")),
		r"^[\d]+[.]?[\d]*\(NCE\)$": lambda x: float(x.rstrip("(NCE)")),
		r"^[\d]+[.]?[\d]*[ ]?%$": lambda x: float(x.rstrip(" %")),
		r"^HCD \(NCE [\d]+[.]?[\d]*%\)$": lambda x: float(x.lstrip("HCD (NCE").rstrip("%)")),
	}
	for k,v in matches.items():
		if re.match(k,ce_str):
			return v(ce_str)
	return np.nan

# def parse_ramp_ce_str(ce_str):

# 	r"^[\d]+[.]?[\d]*->[\d]+[.]?[\d]*%$": (float(x.split("->")[0]),float(x.split("->")[1].rstrip("%"))),
# 	r"[\d]+[.]?[\d]*-[\d]+[.]?[\d]*": lambda x: (float(x.split("-")[0]),float(x.split("-")[1])),

def parse_inst_info(df):
	# TODO: separate FT from Orbitrap

	inst_type_str = df["inst_type"]
	inst_str = df["inst"] if "inst" in df else np.nan
	frag_mode_str = df["frag_mode"] if "frag_mode" in df else np.nan
	col_energy_str = df["col_energy"] if "col_energy" in df else np.nan
	# instrument type
	if inst_type_str == "EI":
		assert inst_str == "EI"
		assert frag_mode_str == "EI"
		assert col_energy_str == "100"
		return "EI", "EI"
	if none_or_nan(inst_type_str):
		# resort to instrument
		inst_map = {
			"Maxis II HD Q-TOF Bruker": "QTOF",
			"qToF": "QTOF",
			"Orbitrap": "FT"
		}
		if none_or_nan(inst_str):
			inst_type = np.nan
		elif inst_str in inst_map:
			inst_type = inst_map[inst_str]
		else:
			inst_type = "Other"
	else:
		inst_type_map = {
			"QTOF": "QTOF",
			"FT": "FT",
			"Q-TOF": "QTOF",
			"HCD": "FT",
			"QqQ": "QQQ",
			"QqQ/triple quadrupole": "QQQ",
			"IT/ion trap": "IT",
			"IT-FT/ion trap with FTMS": "FT",
			"Q-ToF (LCMS)": "QTOF",
			"Bruker Q-ToF (LCMS)": "QTOF",
			"ESI-QTOF": "QTOF",
			"ESI-QFT": "FT",
			"ESI-ITFT": "FT",
			"Linear Ion Trap": "IT",
			"LC-ESI-QTOF": "QTOF",
			"LC-ESI-QFT": "FT",
			"LC-ESI-QQQ": "QQQ",
			"LC-Q-TOF/MS": "QTOF",
			"LC-ESI-ITFT": "FT",
			"LC-ESI-IT": "IT",
			"LC-QTOF": "QTOF",
			"qToF": "QTOF",
			# NPLLIB stuff
			# "Q-ToF (LCMS)": "QTOF",
			# "Bruker Q-ToF (LCMS)": "QTOF",
			"Unknown (LCMS)": "Other",
			"Ion Trap (LCMS)": "IT",
			"Orbitrap (LCMS)": "FT",
			"FTICR (LCMS)": "FT",
			"ITFT": "FT",
			"Orbitrap": "FT",
			"QFT": "FT",
		}
		if inst_type_str in inst_type_map:
			inst_type = inst_type_map[inst_type_str]
		else:
			inst_type = "Other"
	# fragmentation mode
	if inst_type_str == "HCD":
		frag_mode = "HCD"
	elif type(col_energy_str) == str and "HCD" in col_energy_str:
		frag_mode = "HCD"
	elif none_or_nan(frag_mode_str) or frag_mode_str == "CID":
		# default is CID if we don't know
		frag_mode = "CID"
	elif frag_mode_str == "HCD":
		frag_mode = "HCD"
	else:
		frag_mode = np.nan
	return inst_type, frag_mode

def parse_ion_mode_str(ion_mode_str):

	if none_or_nan(ion_mode_str):
		return np.nan
	if ion_mode_str in ["P","N","E","EI"]:
		return ion_mode_str
	elif ion_mode_str == "POSITIVE":
		return "P"
	elif ion_mode_str == "NEGATIVE":
		return "N"
	else:
		return np.nan

def parse_ri_str(ri_str):

	if none_or_nan(ri_str):
		return np.nan
	else:
		return float(ri_str)

def parse_prec_type_str(prec_type_str):

	if none_or_nan(prec_type_str):
		return np.nan
	if prec_type_str == "EI":
		return "EI"
	if prec_type_str.endswith("1+"):
		prec_type_str = prec_type_str.replace("1+","+")
	elif prec_type_str.endswith("1-"):
		prec_type_str = prec_type_str.replace("1-","-")
	# manually replace some
	prec_type_dict = {
		# "M+H": "[M+H]+",
		# "M-H": "[M-H]-",
		"[M+FA-H]-": "[M+CH2O2-H]-"
	}
	if prec_type_str in prec_type_dict:
		prec_type_str = prec_type_dict[prec_type_str]
	return prec_type_str


def infer_prec_mz(df):

	# these were observed to be missing prec_mz in MoNA
	prec_type_to_mass_diff = {
		"[M+HCOO]-": MASS("H")+MASS("C")+2*MASS("O")+ELECTRON_MASS,
		"[M]+": -ELECTRON_MASS,
		"[M+HOO]-": MASS("H")+2*MASS("O")+ELECTRON_MASS,
		"[M-OH]+": -MASS("H")-MASS("O")-ELECTRON_MASS,
		"[M+H-C12H20O9]+": -19*MASS("H")-12*MASS("C")-9*MASS("O")-ELECTRON_MASS,
		"[M-H]-": -MASS("H")+ELECTRON_MASS,
		"[M+H]+": MASS("H")-ELECTRON_MASS,
		"[M+Na]+": MASS("Na")-ELECTRON_MASS,
	}

	prec_mz = df["prec_mz"]
	spec_type = df["spec_type"]
	if none_or_nan(prec_mz) and spec_type == "MS2":
		dset = df["dset"]
		assert "nist" not in dset, dset
		prec_type = df["prec_type"]
		exact_mw = df["exact_mw"]
		if not none_or_nan(prec_type):
			prec_mz = exact_mw + prec_type_to_mass_diff[prec_type]
	return prec_mz

def parse_peaks_str(peaks_str):
	# peaks still represented as string
	if none_or_nan(peaks_str):
		return np.nan
	lines = peaks_str.split("\n")
	peaks = []
	for line in lines:
		if len(line) == 0:
			continue
		line = line.split(" ")
		mz = line[0]
		ints = line[1]
		peaks.append((mz,ints))
	return peaks

def composition_to_string(x):
	return ''.join([a+str(x[a]) for a in sorted(x)])

def combine_formulae(formula_1, formula_2):
	if isinstance(formula_1,str):
		formula_1 = Composition(formula_1)
	if isinstance(formula_2,str):
		formula_2 = Composition(formula_2)
	combine_comp = Composition(formula_1) + Composition(formula_2)
	return composition_to_string(combine_comp)

def parse_product_isotope(annot, precursor):
	formula = Composition('')
	precursor = Composition(formula=precursor)
	adducts = []
	losses = []
	isotope = 0
	charge = 1
	chunks = [''] + re.split(r'([\+-])',annot)
	for prefix, chunk in zip(chunks[::2],chunks[1::2]):
		if len(chunk) == 0:
			continue
		elif prefix == '^':
			charge = int(chunk)
		else:
			if chunk[-1] == 'i' and chunk[-2:] != "Si":
				chunk = chunk[:-1]
				isotope = int(chunk) if chunk else 1
				if prefix == '-':
					isotope = -isotope
			else:
				if chunk[0].isdigit():
					n = int(chunk[0])
					chunk = chunk[1:]
				else:
					n = 1
				if chunk == 'p':
					chunk = precursor
				else:
					chunk = Composition(formula=chunk) * n
				if prefix == '':
					formula += chunk
				elif prefix == '+':
					adducts.append(chunk)
					formula += chunk
				elif prefix == '-':
					losses.append(chunk)
					formula -= chunk
	return formula, adducts, losses, isotope, charge

def is_pep_annot(annot):
	pattern = r"(?:a|b|y)\d+|pi"
	return re.match(pattern,annot) is not None

def parse_annotations(row):
	
	RDLogger = importlib.import_module("rdkit.RDLogger")
	RDLogger.DisableLog('rdApp.*')
	peaks_str = row["peaks"]
	prec_type = row["prec_type"]
	dset = row["dset"]
	formula = row["formula"]
	notes = row["notes"]
	error = None
	if dset not in ["nist20_hr"]:
		error = ValueError("dataset not supported")
	elif "glycan" in notes.lower():
		error = ValueError("glycans not supported (1)")
	elif "peptide" in notes.lower():
		error = ValueError("peptide not supported (1)")
	elif prec_type not in ["[M+H]+","[M-H]-"]:
		error = ValueError("prec_type not supported")
	elif none_or_nan(peaks_str):
		error = ValueError("peaks_str is invalid")
	if error is not None:	
		error = repr(error)
		return np.nan, np.nan, np.nan, np.nan, np.nan, error
	precursor = Composition(formula)
	if prec_type == "[M+H]+":
		precursor["H"] += 1
	elif prec_type == "[M-H]-":
		precursor["H"] -= 1
	precursor = composition_to_string(precursor)
	lines = peaks_str.split("\n")
	peak_mzs, products, losses, isotopes, exact_mzs = [], [], [], [], []
	all_good = True
	for line in lines:
		if not all_good:
			break
		if len(line) == 0:
			continue
		line = line.split(" ")
		if len(line) < 3:
			continue
		peak_mz = line[0]
		annots = " ".join(line[2:]).strip('\"')
		# glycan parsing not implemented
		if "|" in annots:
			error = ValueError("glycans not supported (2)")
			all_good = False
			break
		# peptide parsing not implemented
		if is_pep_annot(annots):
			error = ValueError("peptides not supported (2)")
			all_good = False
			break
		annots = annots.split(";")
		for annot in annots:
			# strip the peak count
			annot = re.sub(r' ?\d+/\d+$','',annot)
			if len(annot) == 0:
				error = ValueError("empty annotation")
				all_good = False
				break
			if annot in ('?','more'):
				continue
			if '/' in annot:
				annot, ppm = annot.split('/')
			if '=' in annot:
				annot, _ = annot.split('=')
			try:
				product, _, _, isotope, _ = parse_product_isotope(annot, precursor)
			except pyteomics.auxiliary.structures.PyteomicsError as e:
				error = e
				all_good = False
				break
			except ValueError as e:
				error = e
				all_good = False
				break
			try:
				exact_mz = composition_to_mass(product,isotope)
			except RuntimeError as e:
				# something wrong with the formula
				error = e
				all_good = False
				break
			loss = Composition(formula=precursor) - product
			product = composition_to_string(product)
			loss = composition_to_string(loss)
			peak_mzs.append(peak_mz)
			products.append(product)
			losses.append(loss)
			isotopes.append(isotope)
			exact_mzs.append(exact_mz)
	error = repr(error)
	if all_good:
		return peak_mzs, products, losses, isotopes, exact_mzs, error
	else:
		return np.nan, np.nan, np.nan, np.nan, np.nan, error

def composition_to_mass(formula,isotope):
	mass = isotope * NEUTRON_MASS
	for k,v in formula.items():
		mass += MASS(k) * v
	return mass

def formula_to_mass(formula):
	formula = Composition(formula=formula)
	return composition_to_mass(formula,0)

def convert_peaks_to_float(peaks):
	# assumes no nan
	float_peaks = []
	for peak in peaks:
		float_peaks.append((float(peak[0]),float(peak[1])))
	return float_peaks

def get_res(peaks):
	# assumes no nan
	ress = []
	for mz,ints in peaks:
		dec_idx = mz.find(".")
		if dec_idx == -1:
			res = 0
		else:
			res = len(mz) - (dec_idx+1)
		ress.append(res)
	highest_res = max(ress)
	return highest_res

def get_murcko_scaffold(mol,output_type="smiles",include_chirality=False):

	if none_or_nan(mol):
		return np.nan
	MurckoScaffold = importlib.import_module("rdkit.Chem.Scaffolds.MurckoScaffold")
	if output_type == "smiles":
		scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=include_chirality)
	else:
		raise NotImplementedError
	return scaffold

def analyze_mol(mol):

	import rdkit
	from rdkit.Chem.Descriptors import MolWt
	import rdkit.Chem as Chem
	mol_dict = {}
	mol_dict["num_atoms"] = mol.GetNumHeavyAtoms()
	mol_dict["num_bonds"] = mol.GetNumBonds(onlyHeavy=True)
	mol_dict["mol_weight"] = MolWt(mol)
	mol_dict["num_rings"] = len(list(Chem.GetSymmSSSR(mol)))
	mol_dict["max_ring_size"] = max([-1]+[len(list(atom_iter)) for atom_iter in Chem.GetSymmSSSR(mol)])
	cnops_counts = {"C": 0, "N": 0, "O": 0, "P": 0, "S": 0, "Cl": 0, "other": 0}
	bond_counts = {"single": 0, "double": 0, "triple": 0, "aromatic": 0}
	cnops_bond_counts = {"C": [-1], "N": [-1], "O": [-1], "P": [-1], "S": [-1], "Cl": [-1]}
	h_counts = 0
	p_num_bonds = [-1]
	s_num_bonds = [-1]
	other_atoms = set()
	for atom in mol.GetAtoms():
		atom_symbol = atom.GetSymbol()
		if atom_symbol in cnops_counts:
			cnops_counts[atom_symbol] += 1
			cnops_bond_counts[atom_symbol].append(len(atom.GetBonds()))
		else:
			cnops_counts["other"] += 1
			other_atoms.add(atom_symbol)
		h_counts += atom.GetNumImplicitHs()
	for bond in mol.GetBonds():
		bond_type = bond.GetBondType()
		if bond_type == rdkit.Chem.rdchem.BondType.SINGLE:
			bond_counts["single"] += 1
		elif bond_type == rdkit.Chem.rdchem.BondType.DOUBLE:
			bond_counts["double"] += 1
		elif bond_type == rdkit.Chem.rdchem.BondType.TRIPLE:
			bond_counts["triple"] += 1
		else:
			assert bond_type == rdkit.Chem.rdchem.BondType.AROMATIC
			bond_counts["aromatic"] += 1
	mol_dict["other_atoms"] = ",".join(sorted(list(other_atoms)))
	mol_dict["H_counts"] = h_counts
	for k,v in cnops_counts.items():
		mol_dict[f"{k}_counts"] = v
	for k,v in bond_counts.items():
		mol_dict[f"{k}_counts"] = v
	for k,v in cnops_bond_counts.items():
		mol_dict[f"{k}_max_bond_counts"] = max(v)
	return mol_dict

def check_num_bonds(mol):
	rdkit = importlib.import_module("rdkit")
	valid = mol.GetNumBonds() > 0
	return valid

CHARGE_FACTOR_MAP = {
	1: 1.00,
	2: 0.90,
	3: 0.85,
	4: 0.80,
	5: 0.75,
	"large": 0.75
}

def get_charge(prec_type_str):

	if prec_type_str == "EI":
		return 1
	end_brac_idx = prec_type_str.index("]")
	charge_str = prec_type_str[end_brac_idx+1:]
	if charge_str == "-":
		charge_str = "1-"
	elif charge_str == "+":
		charge_str = "1+"
	assert len(charge_str) >= 2
	sign = charge_str[-1]
	assert sign in ["+","-"], prec_type_str
	magnitude = int(charge_str[:-1])
	if sign == "+":
		charge = magnitude
	else:
		charge = -magnitude
	return charge

def nce_to_ace_helper(nce,charge,prec_mz):

	if charge in CHARGE_FACTOR_MAP:
		charge_factor = CHARGE_FACTOR_MAP[charge]
	else:
		charge_factor = CHARGE_FACTOR_MAP["large"]
	ace = (nce * prec_mz * charge_factor) / 500.
	return ace

def ace_to_nce_helper(ace,charge,prec_mz):

	if charge in CHARGE_FACTOR_MAP:
		charge_factor = CHARGE_FACTOR_MAP[charge]
	else:
		charge_factor = CHARGE_FACTOR_MAP["large"]
	nce = (ace * 500.) / (prec_mz * charge_factor)
	return nce

def nce_to_ace(row):

	prec_mz = row["prec_mz"]
	nce = row["nce"]
	prec_type = row["prec_type"]
	charge = np.abs(get_charge(prec_type))
	ace = nce_to_ace_helper(nce,charge,prec_mz)
	return ace

def ace_to_nce(row):

	prec_mz = row["prec_mz"]
	ace = row["ace"]
	prec_type = row["prec_type"]
	charge = np.abs(get_charge(prec_type))
	nce = ace_to_nce_helper(ace,charge,prec_mz)
	return nce

def fill_missing_nce(row):
	nce = row["nce"]
	ace = row["ace"]
	if none_or_nan(nce) and not none_or_nan(ace):
		return ace_to_nce(row)
	else:
		return nce

def fill_missing_ace(row):
	nce = row["nce"]
	ace = row["ace"]
	if none_or_nan(ace) and not none_or_nan(nce):
		return nce_to_ace(row)
	else:
		return ace

def parse_mass_gym_ce_str(ce_str):
	"""make it comparable to NIST so we can parse it

	Args:
		ce_str (_type_): _description_

	Returns:
		_type_: _description_
	"""
	if pd.isna(ce_str):
		return "", False, False

	ce_str = str(ce_str)
	normalized = "normalized=True" in ce_str
	ramped = "ramped=True" in ce_str

	ce = ce_str.split(" ")[0]
	if normalized:
		ce = f"NCE={ce}%"
	else:
		ce = f"{ce} eV"

	return ce, normalized, ramped