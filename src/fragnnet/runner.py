import torch as th
import torch_geometric as pyg

try:
	import lightning.pytorch as pl	
	from lightning.pytorch.callbacks import DeviceStatsMonitor
	from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
	from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler, PyTorchProfiler
	from lightning.fabric.utilities.seed import seed_everything
except ModuleNotFoundError:
	import pytorch_lightning as pl
	from pytorch_lightning.callbacks import DeviceStatsMonitor
	from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
	from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler, PyTorchProfiler
	from pytorch_lightning import seed_everything

import logging
import yaml
import os

import glob
import shutil
import tempfile
from multiprocessing import Manager
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from fragnnet.pl_model import FragGNNPL, NeimsPL, PrecursorPL, GNNPL
from fragnnet.iceberg.pl_model import IcebergGenPL, IcebergIntenPL
from fragnnet.massformer.pl_model import MassFormerPL
from fragnnet.graff.pl_model import GrAFFPL
from fragnnet.utils.nn_utils import nan_forward_hook, nan_backward_hook
from fragnnet.utils.pl_utils import ConsoleLogger
from fragnnet.dataset import SpecMolDataset, SpecMolFragDataset, GroupSampler, get_group_sampler, SpecMolFragDynamicBatchSampler
from fragnnet.iceberg.dataset import SpecMolMagmaGenDataset, SpecMolMagmaIntenDataset
from fragnnet.graff.dataset import SpecMolAnnDataset
import fragnnet.utils.misc_utils as misc_utils
from fragnnet.utils.misc_utils import deep_update
from fragnnet.utils.profile_utils import MyPyTorchProfiler


def load_config(template_fp, custom_fp):

	assert os.path.isfile(template_fp), template_fp
	if custom_fp:
		assert os.path.isfile(custom_fp), custom_fp
	with open(template_fp, "r") as template_file:
		config_d = yaml.load(template_file, Loader=yaml.FullLoader)
	# overwrite parts of the config
	if custom_fp:
		with open(custom_fp, "r") as custom_file:
			custom_d = yaml.load(custom_file, Loader=yaml.FullLoader)
		assert all([k in config_d for k in custom_d]), set(custom_d.keys()) - set(config_d.keys())
		config_d = deep_update(config_d, custom_d)
	return config_d


def load_wandb_config(wandb_config_dp) -> dict:
	"""Load wandb config file if exists, else return empty dict

	Args:
		wandb_config_dp (_type_): _description_

	Returns:
		dict: wandb config
	"""
	wandb_config_dp = os.path.join(wandb_config_dp,"config.yaml")

	if os.path.isfile(wandb_config_dp):
		with open(wandb_config_dp,"r") as wandb_config_file:	
			_config_d = yaml.load(
				wandb_config_file,
				Loader=yaml.FullLoader
			)
		del _config_d["wandb_version"]
		del _config_d["_wandb"]
		config_d = {}
		for k in _config_d.keys():
			config_d[k] = _config_d[k]["value"]
	else:
		config_d = {}
	return config_d


def init_dataset(config_d, splits=("train","val")):

	if config_d["model_type"] == "frag_gnn":
		dataset_cls = SpecMolFragDataset
	elif config_d["model_type"] == "iceberg_gen":
		dataset_cls = SpecMolMagmaGenDataset
	elif config_d["model_type"] == "iceberg_inten":
		dataset_cls = SpecMolMagmaIntenDataset
	elif config_d["model_type"] == "graff":
		dataset_cls = SpecMolAnnDataset
	else:
		assert config_d["model_type"] in ["neims","massformer","precursor", "gnn"], config_d["model_type"]
		dataset_cls = SpecMolDataset
	
	data_dict_types = dataset_cls.get_data_dict_types()
	
	dses = []
	for split in splits:
		if config_d["num_workers"] > 0 and config_d["share_memory"]:
			manager = Manager()
			data_sds = {k: manager.dict() for k in data_dict_types}
		else:
			data_sds = {k: dict() for k in data_dict_types}
		ds = dataset_cls(
			split=split,
			**{**data_sds,**config_d})
		dses.append(ds)

	return tuple(dses)


def init_dataloader(ds, config_d):

	split = ds.split
	assert not (config_d["group_sampler"] and config_d["simple_group_sampler"]), "Cannot use both group_sampler and simple_group_sampler"
	print(f"> init_dataloader for split {split}")
	dl_param_d = {
		"dataset": ds,
		"num_workers": config_d["num_workers"],
		"collate_fn": ds.get_collate_fn(),
		"pin_memory": config_d["pin_memory"] and (config_d["accelerator"] != "cpu")
	}

	# note: this generator will get overwritten at the beginning/ending of each training epoch
	generator = th.Generator() 
	if split == "train":
		if config_d["group_sampler"]:
			sampler = GroupSampler(
				ds,
				sample_k=config_d['group_sampler_max_per_group'],
				generator=generator)
		elif config_d["simple_group_sampler"]:
			sampler = get_group_sampler(
				ds,
				config_d["simple_group_sampler_type"],
				config_d["simple_group_sampler_avg_per_group"],
				generator)
		else:
			sampler = RandomSampler(ds, False, generator=generator)
	else:
		# split in ["val", "test", "predict_only"]
		sampler = SequentialSampler(ds)
  
	# for batch sampler 
	if config_d["dynamic_batch_sampler"]:
		max_batch_size = config_d["train_batch_size"] * config_d["accumulate_grad_batches"]
		if split == "train":
			return_batch_at = max_batch_size
		else:
			return_batch_at = 0
		batch_sampler = SpecMolFragDynamicBatchSampler(
							ds, 
							max_num = config_d["dynamic_batch_sampler_max"], 
							limited_by = config_d["dynamic_batch_sampler_mode"], 
							skip_too_big = True,
							return_batch_at = return_batch_at,
							sampler = sampler)
		dl_param_d["batch_sampler"] = batch_sampler
	else:
		dl_param_d["sampler"] = sampler
		if split == "train":
			dl_param_d["batch_size"] = config_d["train_batch_size"]	
			dl_param_d["drop_last"] = config_d["drop_last"]
		else:
			dl_param_d["batch_size"] = config_d["eval_batch_size"]
			dl_param_d["drop_last"] = False
	dl = DataLoader(**dl_param_d)

	return dl


def init_run(template_fp, custom_fp, wandb_mode, job_id):

	# setup logger
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s [%(levelname)s] %(message)s",
		handlers=[logging.StreamHandler()]
	)

	# load config
	config_d = load_config(template_fp, custom_fp)

	# set random seeds
	seed_everything(config_d["seed"],workers=True)

	# set torch multiprocessing strategy
	th.multiprocessing.set_sharing_strategy(config_d["mp_sharing_strategy"])

	# setup logging and wandb
	logging.info("setup loggers")
	loggers = []
	console_logger = ConsoleLogger()
	loggers.append(console_logger)

	if wandb_mode != "disabled":
		import wandb
		wandb_d = dict(
			project=config_d["wandb_project"],
			name=config_d["wandb_name"],
			group=config_d["wandb_group"],
			mode=wandb_mode,
			entity=config_d["wandb_entity"],
			tags=config_d["wandb_tags"],
			resume="allow",
		)

		if job_id and not config_d["disable_checkpoints"]:
			job_id_fp = os.path.join("job_id",f"{job_id}.id")
			if os.path.isfile(job_id_fp):
				is_resume = True
				with open(job_id_fp,"r") as job_id_file:
					text = job_id_file.read().strip()
				run_id, old_wandb_dp = text.split(";")
				wandb_config = load_wandb_config(old_wandb_dp)
				
				# for offline mode, if there is no configure file, just skip it
				# wandb config only exists after wandb sync
				if len(wandb_config) == 0:
					wandb_config = config_d
			else:
				is_resume = False
				run_id = old_wandb_dp = None
				wandb_config = config_d
			wandb_d["id"] = run_id
			wandb_d["config"] = wandb_config
			if config_d["wandb_root_dir"] is not None:
				wandb_dir = os.path.join(config_d["wandb_root_dir"],str(job_id))
				if os.path.isdir(wandb_dir):
					wandb_d["dir"] = wandb_dir
				else:
					print(f"> wandb dir does not exist: {wandb_dir}, using default instead")
		else:
			is_resume = False
			job_id_fp = None
			wandb_d["config"] = config_d
		wandb.init(**wandb_d)
		assert wandb.run is not None, wandb.run
		wandb_d["offline"] = wandb_mode == "offline"
		wandb_logger = pl.loggers.WandbLogger(**wandb_d)
		if job_id_fp:
			os.makedirs("job_id", exist_ok=True)
			with open(job_id_fp,"w+") as job_id_file:
				text = f"{wandb.run.id};{os.path.abspath(wandb.run.dir)}"
				job_id_file.write(text)
		# update config results (primarily for wandb)
		# configs can be nested up to 2 levels
		for k in list(config_d.keys()):
			if k in wandb.config:
				if isinstance(config_d[k],dict):
					# nested
					assert isinstance(wandb.config[k],dict)
					for kk in list(config_d[k].keys()):
						if kk in wandb.config[k]:
							if config_d[k][kk] != wandb.config[k][kk]:
								print(f"> config diff -- {k}->{kk}: {config_d[k][kk]} vs {wandb.config[k][kk]}")
							config_d[k][kk] = wandb.config[k][kk]
				else:
					if config_d[k] != wandb.config[k]:
						print(f"> config diff -- {k}: {config_d[k]} vs {wandb.config[k]}")
					config_d[k] = wandb.config[k]
		loggers.append(wandb_logger)
	else:
		is_resume = False
		job_id_fp = None

	# CUDA and TF32 setup
	if config_d['use_tensor_float32'] == True and config_d['accelerator'] == 'gpu':
		# The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
		th.backends.cuda.matmul.allow_tf32 = True
		# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
		th.backends.cudnn.allow_tf32 = True

	# LOG_ZERO setup
	if config_d["log_zero_fp32"] is not None:
		misc_utils.LOG_ZERO_FP32 = float(config_d["log_zero_fp32"])
	if config_d["log_zero_fp16"] is not None:
		misc_utils.LOG_ZERO_FP16 = float(config_d["log_zero_fp16"])

	# setup model
	logging.info("setup model")
	if config_d["model_type"] == "frag_gnn":
		model_cls = FragGNNPL
	elif config_d["model_type"] == "neims":
		model_cls = NeimsPL
	elif config_d["model_type"] == "iceberg_gen":
		model_cls = IcebergGenPL
	elif config_d["model_type"] == "iceberg_inten":
		model_cls = IcebergIntenPL
	elif config_d["model_type"] == "massformer":
		model_cls = MassFormerPL
	elif config_d["model_type"] == "graff":
		model_cls = GrAFFPL
	elif config_d["model_type"] == "precursor":
		model_cls = PrecursorPL
	elif config_d["model_type"] == "gnn":
		model_cls = GNNPL
	else:
		raise ValueError(config_d["model_type"])
	model = model_cls(**config_d)
	model.train()

	# setup callbacks
	callbacks = []
	if wandb_mode != "disabled":
		ckpt_dp = os.path.join(wandb.run.dir,"ckpt")
	else:
		ckpt_dp = "tmp_ckpt"
	os.makedirs(ckpt_dp, exist_ok=True)
	if is_resume:
		assert not config_d["disable_checkpoints"]
		# copy the checkpoint files
		old_ckpt_fps = glob.glob(os.path.join(old_wandb_dp,"ckpt","*.ckpt"))
		for old_ckpt_fp in old_ckpt_fps:
			new_ckpt_fp = os.path.join(ckpt_dp,os.path.basename(old_ckpt_fp))
			# modify checkpoint metadata (hacky)
			new_ckpt_data = th.load(old_ckpt_fp)
			ckpt_callback_data = None
			for k,v in new_ckpt_data["callbacks"].items():
				if k.startswith("ModelCheckpoint"):
					ckpt_callback_data = v
					break
			assert ckpt_callback_data is not None
			ckpt_keys = ["best_model_path","last_model_path","dirpath","kth_best_model_path"]
			for k in ckpt_keys:
				if k in ckpt_callback_data:
					ckpt_callback_data[k] = ckpt_callback_data[k].replace(
						os.path.abspath(old_wandb_dp),
						os.path.abspath(wandb.run.dir)
					)
			if "best_k_models" in ckpt_callback_data:
				for k in list(ckpt_callback_data["best_k_models"].keys()):
					new_k = k.replace(
						os.path.abspath(old_wandb_dp),
						os.path.abspath(wandb.run.dir)
					)
					ckpt_callback_data["best_k_models"][new_k] = ckpt_callback_data["best_k_models"].pop(k)
			# save modified checkpoint in new wandb dir
			th.save(new_ckpt_data,new_ckpt_fp)
	if not config_d["disable_checkpoints"]:
		checkpoint_callback = ModelCheckpoint(
			dirpath=ckpt_dp,
			filename="model-{epoch:03d}",
			monitor=config_d["checkpoint_metric"],
			mode=config_d["checkpoint_metric_mode"],
			save_last=config_d["checkpoint_save_last"],
		)
		callbacks.append(checkpoint_callback)

	# setup profiler
	logging.info("setup profiler")
	if wandb_mode != "disabled":
		profile_dp = os.path.join(wandb.run.dir,"profile")
	else:
		profile_dp = "tmp_profile"
	os.makedirs(profile_dp, exist_ok=True)
	if config_d["profiler"] == "simple":
		profiler = SimpleProfiler(
			dirpath=profile_dp,
			filename="profile"
		)
	elif config_d["profiler"] == "advanced":
		profiler = AdvancedProfiler(
			dirpath=profile_dp,
			filename="profile"
		)
	elif config_d["profiler"] == "pytorch":
		th.profiler._utils._init_for_cuda_graphs()
		profiler = MyPyTorchProfiler(
			dirpath=profile_dp,
			filename="profile",
			activities=[
				th.profiler.ProfilerActivity.CPU,
				th.profiler.ProfilerActivity.CUDA
			],
			# on_trace_ready=th.profiler.tensorboard_trace_handler(profile_dp),
			profile_memory=True,
			record_shapes=False, #True,
			with_flops=False, #True,
			with_stack=True,
			with_modules=False,
			export_to_chrome=False,
			export_to_flame_graph=True,
			experimental_config=th._C._profiler._ExperimentalConfig(verbose=True),
			schedule=th.profiler.schedule(wait=1, warmup=1, active=6, repeat=0, skip_first=0)
		)
	else:
		assert config_d["profiler"] == "none", config_d["profiler"]
		# no profiler
		profiler = None

	# setup datasets
	logging.info("setup dataset")
	splits = ["train", "val"]
	if config_d["eval_test_split"]:
		splits.append("test")
	dses = init_dataset(config_d, splits=splits)
	train_ds = dses[0]
	val_ds = dses[1]
	if config_d["eval_test_split"]:
		test_ds = dses[2]

	# check config for sampler
	if config_d["dynamic_batch_sampler"]:
		assert config_d["model_type"] == "frag_gnn",\
			f"Dynamic batch sampler can only be use with frag_gnn model"
		assert config_d['automatic_optimization'] == False,\
			f"Dynamic batch sampler can not use in junction of automatic optimization {config_d['automatic_optimization']}"
		assert config_d['dynamic_batch_sampler_mode'] in ['frag_node','frag_edge'],\
			f"Dynamic batch sampler only support frag_node or frag_edge mode (given {config_d['dynamic_batch_sampler_mode']}) "
		assert config_d['dynamic_batch_sampler_max'] is not None,\
			"Dynamic batch sampler needs a max num"
	if  config_d["frag_gnn_type"] == "NodeMLP" and config_d["dynamic_batch_sampler"]:
		assert config_d["dynamic_batch_sampler_mode"] == "frag_node",\
      		f"Dynamic batch sampler can only be use with dynamic_batch_sampler_mode frag_node, not {config_d['dynamic_batch_sampler_mode']}"		
	# setup dataloaders
	logging.info("setup dataloader")
	train_dl = init_dataloader(train_ds, config_d)
	val_dl = init_dataloader(val_ds, config_d)
	if config_d["eval_test_split"]:
		test_dl = init_dataloader(test_ds, config_d)

	# setup trainer
	logging.info("setup trainer")
	if config_d["debug_overfit"]:
		overfit_batches = config_d["debug_overfit_batches"]
	else:
		overfit_batches = 0
	log_every_n_steps = min(len(train_dl),config_d["log_every_n_steps"])

	trainer_param_d  = {
		'logger': loggers,
		'callbacks': callbacks,
		'accelerator': config_d["accelerator"],
		'devices': config_d["devices"],
		'min_epochs': config_d["min_epochs"],
		'max_epochs': config_d["max_epochs"],
		'precision': config_d["precision"],
		'log_every_n_steps': log_every_n_steps,
		'detect_anomaly': config_d["detect_anomaly"],
		'overfit_batches': overfit_batches,
		'profiler': profiler,
		'num_sanity_val_steps': config_d["num_sanity_val_steps"],
		'enable_progress_bar': config_d["pl_enable_progress_bar"],
		'enable_checkpointing': not config_d["disable_checkpoints"],
	}

	# this things can only set if automatic_optimization 
	if config_d['automatic_optimization']:
		trainer_param_d['accumulate_grad_batches'] = config_d["accumulate_grad_batches"]
		trainer_param_d['gradient_clip_val'] = config_d["gradient_clip_val"]
		trainer_param_d['gradient_clip_algorithm'] = config_d["gradient_clip_algorithm"]
		if config_d["num_workers"] > 0:
			# little hack to prevent memory explosion
			# https://discuss.pytorch.org/t/how-to-share-data-among-dataloader-processes-to-save-memory/108772
			# https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
			trainer_param_d['reload_dataloaders_every_n_epochs'] = 1
	elif config_d['dynamic_batch_sampler']:
		# with out this progress will not track dataset length change
		# this will call train_dataloader and val_dataloader before every epoch
		# not sure why this fixed our problem but it works 
		trainer_param_d['reload_dataloaders_every_n_epochs'] = 1

	trainer = pl.Trainer(**trainer_param_d)

	# set determinism
	th.use_deterministic_algorithms(config_d["deterministic"],warn_only=True)

	# register nan hook
	if config_d["nan_module_hook"]:
		th.nn.modules.module.register_module_forward_hook(nan_forward_hook)
		th.nn.modules.module.register_module_full_backward_hook(nan_backward_hook)
	
	# fit
	if config_d["debug_overfit"]:
		assert not is_resume
		logging.info("debug overfit model")
		trainer.fit(model, train_dl)
	else:
		logging.info("fit model")
		if is_resume:
			ckpt_fp = os.path.join(ckpt_dp,"last.ckpt")
			if os.path.isfile(ckpt_fp):
				logging.info(f"resuming from checkpoint: {ckpt_fp}")
			else:
				ckpt_fp = None
		else:
			ckpt_fp = None
		trainer.fit(
			model, 
			train_dl, 
			val_dl,
			ckpt_path=ckpt_fp
		)
	logging.info("callback metrics")

	if not trainer.interrupted and config_d["eval_test_split"]:
		logging.info("test model")
		trainer.test(
			model=model,
			ckpt_path="best" if config_d["min_epochs"] > 0 else None,
			dataloaders=test_dl
		)

	if config_d["compile"]:
		print(model.dynamo_prof.report())

	if not trainer.interrupted:
		# find checkpoints
		ckpt_fps = glob.glob(os.path.join(ckpt_dp,"*.ckpt"))
		delete_before_ckpt_flag = not config_d["upload_checkpoints"] and \
			not config_d["disable_checkpoints"]
		transfer_ckpt_flag = not config_d["delete_checkpoints"] and \
			delete_before_ckpt_flag
		if transfer_ckpt_flag:
			temp_ckpt_dir = tempfile.TemporaryDirectory()
			for ckpt_fp in ckpt_fps:
				shutil.move(ckpt_fp,os.path.join(temp_ckpt_dir.name,os.path.basename(ckpt_fp)))
		elif delete_before_ckpt_flag:
			for ckpt_fp in ckpt_fps:
				os.remove(ckpt_fp)

	if wandb_mode != "disabled":
		wandb.finish()

	if not trainer.interrupted:
		ckpt_fps = glob.glob(os.path.join(ckpt_dp,"*.ckpt"))
		delete_after_ckpt_flag = config_d["delete_checkpoints"] and \
			not config_d["disable_checkpoints"]
		assert not (delete_after_ckpt_flag and transfer_ckpt_flag)
		if transfer_ckpt_flag:
			# transfer checkpoints
			for ckpt_fp in glob.glob(os.path.join(temp_ckpt_dir.name,"*.ckpt")):
				shutil.move(ckpt_fp,os.path.join(ckpt_dp,os.path.basename(ckpt_fp)))
			temp_ckpt_dir.cleanup()
		elif delete_after_ckpt_flag:
			for ckpt_fp in ckpt_fps:
				os.remove(ckpt_fp)
		# cleanup (post-wandb)
		if job_id_fp:
			os.remove(job_id_fp)

	return model
