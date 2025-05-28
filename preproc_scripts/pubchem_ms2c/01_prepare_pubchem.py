from io import StringIO
import urllib.request
import os
from datetime import date
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from tqdm import tqdm
from rdkit import Chem

import gzip
import re
import pandas as pd
import multiprocessing
import concurrent.futures
import signal
import hashlib
from threading import Event
from fragnnet.utils.ms2c_utils import MolCandidateDB
import numpy as np
import argparse

def tqdm_hook(t):
	"""
	https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5
	https://pypi.org/project/tqdm/3.4.0/
	"""
	last_b = [0]

	def inner(b=1, bsize=1, tsize=None):
		"""
		b  : int, optional
			Number of blocks just transferred [default: 1].
		bsize  : int, optional
			Size of each block (in tqdm units) [default: 1].
		tsize  : int, optional
			Total size (in tqdm units). If [default: None] remains unchanged.
		"""
		if tsize is not None:
			t.total = tsize
		t.update((b - last_b[0]) * bsize)
		last_b[0] = b
	return inner

def download_cid_smile(data_dir):
	current_datetime = datetime.now(ZoneInfo("Canada/Mountain"))
	date_stamp = current_datetime.date()

	pubchem_cid_smiles_fp = f"{data_dir}/CID-SMILES_{date_stamp}.gz"
	#print(f">pubchem_cid_smiles_fp: {pubchem_cid_smiles_fp}")
	if not os.path.isfile(pubchem_cid_smiles_fp):
		pubchem_cid_smiles_url = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"
		with tqdm(unit = 'B', unit_scale = True, unit_divisor = 1024, miniters = 1, desc = pubchem_cid_smiles_url) as t:
			urllib.request.urlretrieve(pubchem_cid_smiles_url, pubchem_cid_smiles_fp, reporthook = tqdm_hook(t), data = None)

def process_sdf_imap(args):
	#print(*args)
	process_sdf(*args)

def read_pickle_imap(args):
	return pd.read_pickle(*args)

def process_sdf(sdf_url,  
				sdf_fp, 
				pickle_fp, 
				force_redownload=False,
				force_reprocess=False):

	current_process = multiprocessing .current_process()
	bar_pos = current_process._identity[0] if len(current_process._identity) > 0 else 1

	need_download = not os.path.isfile(sdf_fp)
	if need_download or force_redownload:
		sdf_md5_fp = sdf_fp + '.md5'
		sdf_md5_url = sdf_url + '.md5'

		if not os.path.isfile(sdf_md5_fp) or force_redownload:
			with tqdm(unit = 'B', unit_scale = True, unit_divisor = 1024, miniters = 1, desc = sdf_md5_url, leave=False, position = bar_pos) as t:
				urllib.request.urlretrieve(sdf_md5_url, sdf_md5_fp, reporthook = tqdm_hook(t), data = None)
			md5_hash = None
		with open(sdf_md5_fp) as md5_in:
			md5_hash = md5_in.readlines()[0].split()[0]

		if not need_download:
			sdf_fp_md5 = hashlib.md5(open(sdf_fp,'rb').read()).hexdigest()
			need_download = (md5_hash != sdf_fp_md5)
		
		if need_download:
			with tqdm(unit = 'B', unit_scale = True, unit_divisor = 1024, miniters = 1, desc = sdf_url, leave=False, position = bar_pos) as t:
				urllib.request.urlretrieve(sdf_url, sdf_fp, reporthook = tqdm_hook(t), data = None)

			sdf_fp_md5 = hashlib.md5(open(sdf_fp,'rb').read()).hexdigest()
			if md5_hash != sdf_fp_md5:
				UserWarning(f"md5 for {sdf_url} is matching expecting {md5_hash} getting {sdf_fp_md5}")

	if not os.path.isfile(pickle_fp) or force_reprocess:
		data_l = []
		try:
			with gzip.open(sdf_fp, "rt") as gzip_hnd:
				next_line_title = None
				data_row = [""] * 5 
				title_idx = {
					'cid':0,
					'inchikey':1,
					'exact_mass':2,
					'formula':3,
					'smiles':4
				}
				for row in tqdm(gzip_hnd, desc = sdf_fp.split('/')[-1], position = bar_pos, leave = True):
					row = row.strip()
					
					if row == "> <PUBCHEM_COMPOUND_CID>":
						next_line_title = 'cid'
					elif row =="> <PUBCHEM_IUPAC_INCHIKEY>":
						next_line_title = 'inchikey'
					elif row =="> <PUBCHEM_EXACT_MASS>":
						next_line_title = 'exact_mass'
					elif row =="> <PUBCHEM_MOLECULAR_FORMULA>":
						next_line_title = 'formula'
					elif row =="> <PUBCHEM_OPENEYE_CAN_SMILES>":
						next_line_title = 'smiles'
					elif not row.startswith("> <") and next_line_title is not None:
						#data_row.append(row.strip())
						data_row[title_idx[next_line_title]] = row
						next_line_title = None
					elif row == ("$$$$"):
						data_l.append(data_row)
						data_row = [""] * 5 
		except:
			print(f"Cannot read {sdf_fp}")
			del data_l
		else:
			df = pd.DataFrame(data_l, columns=['cid','inchikey','exact_mass', 'formula','smiles'])
			df['cid'].astype(int)
			df['exact_mass'].astype(np.float64)
			df.to_pickle(pickle_fp, compression='gzip')
			del data_l
			del df

def fix_type(pickle_fp):
	df = pd.read_pickle(pickle_fp, "gzip")
	df['cid'].astype(int)
	df['exact_mass'].astype(np.float64)
	df.to_pickle(pickle_fp, compression='gzip')
	del df

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dp',  type=str, default="./data/ms2c/pubchem_v1")
	args = parser.parse_args()
	data_dp = args.data_dp
 
	print("> Data dir: {data_dir}")
	os.makedirs(data_dp,exist_ok=True)
	os.makedirs(f"{data_dp}/pubchem_raw",exist_ok=True)
	os.makedirs(f"{data_dp}/pubchem_pickle",exist_ok=True)

	data_l = []
	print("> Get SDF data from Pubchem")
	pubchem_sdf_basurl = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/"
	fp = urllib.request.urlopen(pubchem_sdf_basurl)
	mybytes = fp.read()
	pubchem_sdf_html = mybytes.decode("utf8")
	fp.close()

	force_redownload=False
	force_reprocess=False

	re_pattern = r'<a\shref="(.*).gz">'
	sdf_names = [f for f in re.findall(re_pattern, pubchem_sdf_html)]

	num_process = os.cpu_count()
	og_sigint_handler = signal.getsignal(signal.SIGINT)
	og_sigterm_handler = signal.getsignal(signal.SIGTERM)
	with multiprocessing.Pool(num_process) as mp_pool:
		def pool_term_signal_hander(sig, frame):
			#pool_executor.shutdown(cancel_futures=True)
			mp_pool.terminate()
			exit()

		signal.signal(signal.SIGINT, pool_term_signal_hander)
		signal.signal(signal.SIGTERM, pool_term_signal_hander)

		starmap_arg = []
		for sdf_filename in sdf_names:
			sdf_url = f"{pubchem_sdf_basurl}{sdf_filename}.gz" 
			sdf_fp = f"{data_dp}/raw/{sdf_filename}.gz"
			pickle_fp = f"{data_dp}/pickle/{sdf_filename}.pickle.gz"
			starmap_arg.append((sdf_url, sdf_fp, pickle_fp, force_redownload, force_reprocess))

		list(tqdm(mp_pool.imap(process_sdf_imap,starmap_arg), total = len(starmap_arg), desc="Process Pubchem SDFs", position=0))

	signal.signal(signal.SIGINT, og_sigint_handler)
	signal.signal(signal.SIGTERM, og_sigterm_handler)

	db_file  = f"{data_dp}/pubchem.sqlite"
	with MolCandidateDB(db_file) as db:
		db._go_fast_at_all_cost()
		db.create_tables()
		for sdf_filename in tqdm(sdf_names, desc = "Build DB"):
			pickle_fp = f"{data_dp}/pickle/{sdf_filename}.pickle.gz"
			df = pd.read_pickle(pickle_fp, "gzip")
			df = df.astype({'cid':int, 'inchikey':'string', 'exact_mass':float, 'formula':'string','smiles':'string'})
			db.add_compounds_from_df(df)