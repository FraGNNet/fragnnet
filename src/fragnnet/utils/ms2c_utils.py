import copy
import sqlite3
import numpy as np
import pandas as pd
import torch as th
import multiprocessing
import os
from tqdm import tqdm

from fragnnet.pl_model import FragGNNPL, NeimsPL, GNNPL
from fragnnet.massformer.pl_model import MassFormerPL
from fragnnet.iceberg.pl_model import IcebergGenPL, IcebergIntenPL
from fragnnet.iceberg.dataset import SpecMolMagmaIntenDataset,SpecMolMagmaGenDataset
from fragnnet.runner import init_dataloader, load_config
from fragnnet.utils.misc_utils import to_device
from fragnnet.utils.nn_utils import get_pl_hparams, decompile_jit_ckpt
from fragnnet.dataset import SpecMolFragDataset, SpecMolDataset

class MolCandidateDB():

	def __init__(self, db_config: str):
		self.conn = None
		if type(db_config) == str:
			self.db_file = db_config
		else:
			self.db_file = db_config.db_file

	def __enter__(self):
		self.connect()
		return self

	def __exit__(self, exception_type, exception_value, traceback):
		# save cache on exit
		self.disconnect()

	def connect(self):
		self.disconnect()
		# Try to connect
		# conn = None
		try:
			self.conn = sqlite3.connect(self.db_file, isolation_level=None, detect_types=sqlite3.PARSE_DECLTYPES)
		except:
			print(f"Unable to connect to the database: {self.db_file} ")
		else:
			# print("Database Connection Established")
			# use wal model
			self.conn.execute('pragma journal_mode=wal2')

	def disconnect(self):
		if self.conn is not None:
			self.conn.cursor().close()
			self.conn.close()
			self.conn = None

	def check_connection(self):
		if self.conn is None:
			raise ValueError('No Connection')

	def execute_query(self, query: str, commit: bool = False, fetch: bool = True):
		self.check_connection()
		fetched = None
		try:
			cursor = self.conn.cursor()
			cursor.execute(query)
		except Exception as err:
			# pass exception to function
			print(f'{err}')
			self.conn.rollback()
			return None
		else:
			if fetch:
				fetched = cursor.fetchall()
			if commit:
				self.conn.commit()
		#print("size of fetched", sys.getsizeof(fetched) >> 20, "MB")
		return fetched 
	
	def execute_many_query(self, query: str, data):
		self.check_connection()
		try:
			cursor = self.conn.cursor()
			cursor.executemany(query, data)
		except Exception as err:
			# pass exception to function
			#DatabaseUtilities.print_psycopg2_exception(err)
			print(f'{err}')
			print(query)
			print(data)
			self.conn.rollback()
			return None
		else:
			self.conn.commit()

	def _go_fast_at_all_cost(self):
		"""_summary_
		"""
		# Turning off journal_mode will result in no rollback journal
		self.conn.execute('PRAGMA journal_mode = OFF;')
		# SQLite will not care about writing to disk reliably and hands off that responsibility to the OS
		self.conn.execute('PRAGMA synchronous = 0;')
		# The cache_size specifies how many memory pages SQLite is allowed to hold in the memory
		self.conn.execute('PRAGMA cache_size = 4000000;')  # give it  4GB
		self.conn.execute('PRAGMA locking_mode = EXCLUSIVE;')
		self.conn.execute('PRAGMA temp_store = MEMORY;')

	def _create_compound_table(self):
		query = """
			CREATE TABLE IF NOT EXISTS compound (
				id INTEGER primary key,
				inchikey TEXT,
				smiles TEXT,
				formula TEXT,
				exact_mass FLOAT);
			"""
		self.execute_query(query)

		for index_col in ['exact_mass']:
			self.execute_query(f"CREATE INDEX IF NOT EXISTS {index_col}_idx on compound ({index_col});")

	def create_tables(self):
		self._create_compound_table()


	def get_compounds_by_exact_mass_range(self, mw_min:float, mw_max:float, verbose:bool=False):
		"""_summary_

		Args:
			smiles_list (List[str]): _description_

		Returns:
			_type_: return -1 for not found
		"""

		selection_query = f"SELECT COALESCE(id,-1),inchikey,smiles,formula,exact_mass FROM compound WHERE exact_mass >= {mw_min} AND exact_mass <= {mw_max}"
		if verbose:
			print(f">get_compounds_by_colname {selection_query}")
		return self.execute_query(selection_query, fetch=True)
	
	def get_compounds_by_colname(self, col_name:str, values_list:list[str|int|float], verbose:bool=False):
		"""_summary_

		Args:
			smiles_list (List[str]): _description_

		Returns:
			_type_: return -1 for not found
		"""
		# TODO use fetch and yelid if there is too many returns
		if col_name in ['id','exact_mass']:
			values_query = ','.join(["{}".format(i) for i in values_list])
		elif col_name in ['inchikey','smiles','formula']:
			values_query = ','.join(["'{}'".format(i) for i in values_list])
		else:
			raise AttributeError(f"col_name need to be one of 'id','exact_mass','inchikey','smiles','formula' not {col_name} ")
		
		selection_query = f"SELECT COALESCE(id,-1),inchikey,smiles,formula,exact_mass FROM compound WHERE {col_name} in ({values_query})"
		if verbose:
			print(f">get_compounds_by_colname {selection_query}")
		return self.execute_query(selection_query, fetch=True)
		
	@classmethod
	def _list_to_sqlite(cls, data: list, quote = True):
		if type(data) is list:
			if quote:
				return "('" + "','".join([str(d) for d in data]) + "')"
			else:
				return f"({','.join([str(d) for d in data])})"
		else:
			if quote:
				return f"('{data}')"
			else:
				return f"({data})"
	
	def add_compounds_from_df(self, df, chunk_size = 10000):
		compound_query = """INSERT OR IGNORE INTO compound(id, inchikey, exact_mass, formula, smiles) VALUES (?,?,?,?,?); """
		for i in tqdm(range(0,df.shape[0],chunk_size), leave = False):
			compound_insert_data = df[i:i+chunk_size].values.tolist()
			self.execute_many_query(compound_query, compound_insert_data)

#
model_type_to_model_cls = {
	"neims": NeimsPL,
	"gnn": GNNPL,
	"frag_gnn": FragGNNPL,
	"massformer": MassFormerPL,
	"iceberg_inten": IcebergIntenPL
}

def load_model_and_init_config(
		ckpt_fp:str, 
		device:str = None, 
		batch_size:int = 32, 
		auxiliary_scores:list = None, 
		eval_mz_bin_res:list[float] = None, 
		custom_fp:str = None,
		template_fp:str = None,
		num_workers:int = None):
	"""load model and reset config for inference

	Args:
		ckpt_fp (str): check point path
		device (str, optional): device eg cpu or cuda:0. Defaults to None.
		batch_size (int, optional): batch_size. Defaults to 32.
		auxiliary_scores (str, optional): list of auxiliary_scores. Defaults to None, only used on split evalution ['cos_sim']
		eval_mz_bin_res (str, optional): bin size of auxiliary scores. Defaults to None, only used on split evalution, eg [03.01]

	Returns:
		_type_: _description_
	"""
	
	# load ckpt
	ckpt = th.load(ckpt_fp, map_location=device)
	
	print(">> Loading configs")
	config_d = get_pl_hparams(ckpt)
	if custom_fp is not None:
		assert template_fp is not None
		config_d_2 = load_config(template_fp, custom_fp)
		for k, v in config_d_2.items():
			if k in config_d and v != config_d[k]:
				print(f"> overwrite config: {k} -- {config_d[k]} vs {v}")
			elif k not in config_d:
				print(f"> add config: {k} -- {v}")
			config_d[k] = v

	# overwrite config values
	# set eval batch size
	config_d['eval_batch_size'] = batch_size
	# disable metrics
	config_d['track_datapoint_metrics'] = False 
	config_d['auxiliary_scores'] = auxiliary_scores if auxiliary_scores is not None else []
	config_d['eval_mz_bin_res'] = eval_mz_bin_res if eval_mz_bin_res is not None else []
	# let us see if ram will be an issue
	config_d['frag_params']['preload'] = False
	config_d['frag_params']['preprocess'] = False
	config_d['magma_params']['preprocess'] = False
	# disable sampler
	config_d["dynamic_batch_sampler"] = False
	config_d["group_sampler"] = False 
	config_d["simple_group_sampler"] = False
	# additional spec_params
	config_d['spec_params']['unique_id'] = True
	config_d['spec_params']["counts"] = True
	# set num works and cpu stuff
	if num_workers is not None:
		config_d['num_workers'] = num_workers

    # config cuda and cpus
	if device is None:
		device = th.device("cuda:0") if config_d["accelerator"]=="gpu" else th.device("cpu")
	elif not th.cuda.is_available() and device != "cpu":
		device = "cpu"
	print(f"> using {device}")
	# cpu specifc setup
	if device == "cpu":
		th.set_num_threads(multiprocessing.cpu_count())
		th.multiprocessing.set_sharing_strategy('file_system')
		th.use_deterministic_algorithms(True)
  
	# check class
	print(f">> model type: {config_d['model_type']}")
	model_type = config_d['model_type']
	model_cls = model_type_to_model_cls[model_type]

 	# load models
	print(">> Loading models")
	print(f"> ckpt was compiled: {config_d['compile']}")
	if config_d['compile']:
		ckpt = decompile_jit_ckpt(ckpt)
		config_d['compile'] = False

	model = model_cls(**config_d)
	model.load_state_dict(ckpt["state_dict"])
	model.to(device)
	return model, config_d, device


def run_spectra_prediction(model,
				 config_d:dict,
				 mol_data_ptr:str|pd.DataFrame,
				 spec_data_ptr:str|pd.DataFrame,
				 device:str,
				 frag_dp:str = None,
				 magma_dp:str = None,
				 validate:bool = False,
				 ):
	"""_summary_

	Args:
		model (_type_): _description_
		config_d (dict): _description_
		mol_data_ptr (str | pd.DataFrame): _description_
		spec_data_ptr (str | pd.DataFrame): _description_
		frag_dp (str): _description_
		device (str): _description_

	Returns:
		_type_: _description_
	"""
	local_config_d = copy.deepcopy(config_d)
	# config proc and check if frags are avaibale
	if isinstance(mol_data_ptr, str):
		print(f"> mol_fp {mol_data_ptr}")
		mol_df = pd.read_pickle(mol_data_ptr)
	elif isinstance(mol_data_ptr, pd.DataFrame):
		mol_df = mol_data_ptr
		print(f"> mol_df {mol_df.shape}")
	else:
		raise ValueError("mol_data_ptr should be str or pd.DataFrame")

	if isinstance(spec_data_ptr, str):
		print(f"> spec_fp {spec_data_ptr}")
		spec_df = pd.read_pickle(spec_data_ptr)
	elif isinstance(spec_data_ptr, pd.DataFrame):
		spec_df = spec_data_ptr
		print(f"> spec_df {spec_df.shape}")
	else:
		raise ValueError("mol_data_ptr should be str or pd.DataFrame")


	# check model class and ds class
	model_ds_cls = None
 
	# only check for FragGNNet
	if isinstance(model,FragGNNPL):
		assert os.path.isdir(frag_dp)
		print(f">> frag_dp {frag_dp}")
		print(">> filtering input without frags")
		all_frags = set([f.split('.')[0] for f in os.listdir(frag_dp)])
		print(">> num of frags: ", len(all_frags))
		mol_df = mol_df[mol_df['mol_id'].astype(str).isin(all_frags)]
		local_config_d['frag_dp'] = frag_dp
		model_ds_cls = SpecMolFragDataset
		print("> num of spec before filtering by dag", len(spec_df))
		spec_df = spec_df[spec_df['mol_id'].isin(mol_df['mol_id'].to_list())]
		print("> num of spec after filtering by dag", len(spec_df))
	elif isinstance(model,NeimsPL) or isinstance(model,MassFormerPL) or isinstance(model, GNNPL):
		mz_max = local_config_d['mz_max']
		spec_df = spec_df[spec_df['prec_mz'] <= mz_max]
		#print("> [NeimsPL] num of spec after prec_mz filtering", len(spec_df))
		model_ds_cls = SpecMolDataset
	elif isinstance(model,IcebergIntenPL):
		assert os.path.isdir(magma_dp)
		model_ds_cls = SpecMolMagmaIntenDataset
		print(f">> filtering input without magma. magma_dp: {magma_dp}")
		magma_jsons = set([f.split('.')[0] for f in os.listdir(os.path.join(magma_dp,'magma_tree')) if f.endswith('.json')])
		print(">> num of magma jsons: ", len(magma_jsons))
		print("> num of spec before filtering by magma", len(spec_df))
		spec_df = spec_df[spec_df['group_id'].astype(str).isin(magma_jsons)]
		print("> num of spec after filtering by magma", len(spec_df))		
		local_config_d['magma_dp'] = magma_dp
	else:
		print(f"Not supported model: {model}")
  

	if len(spec_df) == 0:
		columns = ['spec_id','mol_id','pred_mzs','pred_ints','pred_oos']
		return pd.DataFrame([], columns=columns)
	
	# config mol df, spec df and frag_dp
	local_config_d['mol_fp'] = mol_df
	local_config_d['spec_fp'] = spec_df

	# load datasets
	print(">> Loading datasets")
	#if validate:
	#	ds = model_ds_cls(split='val', **config_d)
	#else:
	ds = model_ds_cls(split='predict_only', **local_config_d)
	print(">> init dataloader")
	dl = init_dataloader(ds, local_config_d)
	
	eval_score_types = []
	if validate:
		for score_type in local_config_d['auxiliary_scores']:
			if "_hun" in score_type:
				eval_score_types.append(score_type)
			else:
				for bin_res in local_config_d['eval_mz_bin_res']:
					eval_score_types.append(f"{score_type}_{bin_res}")
	preds_l = []
 	#print(">> set model to this to eval")
	model.eval()
	with th.inference_mode():
		for _, batch_input in tqdm(enumerate(dl),total=len(dl), desc=" Running inference"):
			spec_ids = batch_input['spec_unique_id']
			mol_ids = batch_input['mol_id']

			batch_input = to_device(batch_input, device)
			# use existing validate to compute cos sim and etc
			if validate:
				batch_result = model._common_step(batch_input, split="test", log=False)
			else:
				batch_result = model.predict_step(**batch_input)
    
			batch_size = batch_input['batch_size'].detach().cpu()

			pred_mzs = batch_result["pred_mzs"].detach().cpu()
			pred_batch_idxs = batch_result["pred_batch_idxs"].detach().cpu()
			
			has_oos_b = "pred_oos_logprobs" in batch_result
			if has_oos_b:
				pred_oos_logprobs = batch_result["pred_oos_logprobs"].detach().cpu()
	
			pred_ints = th.exp(batch_result["pred_logprobs"]).detach().cpu()
			pred_ints = model.ints_untransform_func(pred_ints, pred_batch_idxs)

			eval_scores_d = {}
			for eval_score_type in eval_score_types:
				eval_scores_d[eval_score_type] = batch_result[eval_score_type].detach().cpu()
				#print(eval_score_type, eval_scores_d[eval_score_type])
			for b_idx in range(batch_size):
				b_peak_mask = pred_batch_idxs == b_idx
	
				b_pred_oos_logprob = pred_oos_logprobs[b_idx].numpy().item() if has_oos_b else np.nan
				b_spec_id = spec_ids[b_idx]
				b_mol_id = mol_ids[b_idx]

				b_spec_mzs = pred_mzs[b_peak_mask]
				b_spec_ints = pred_ints[b_peak_mask]

				data_row = [b_spec_id.item(), b_mol_id, b_spec_mzs.tolist(),b_spec_ints.tolist(), b_pred_oos_logprob]
				for eval_score_type in eval_score_types:
					data_row.append(eval_scores_d[eval_score_type][b_idx].item())
				preds_l.append(data_row)

	columns = ['spec_id','mol_id','pred_mzs','pred_ints','pred_oos'] + eval_score_types
	df = pd.DataFrame(preds_l, columns=columns)
	return df