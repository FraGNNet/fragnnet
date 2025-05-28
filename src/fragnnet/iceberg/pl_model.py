import torch as th
import torch.nn as nn
import torch._dynamo as th_dynamo
import numpy as np
import dgl

from fragnnet.pl_model import SpectrumPL
from fragnnet.utils.misc_utils import to_cpu, scatter_reduce, LOG_ZERO
from fragnnet.loss import sparse_cosine_distance_hungarian, sparse_cosine_distance_binned
import fragnnet.iceberg.common as common
from fragnnet.iceberg.model import IcebergGenModel, IcebergIntenModel
from fragnnet.iceberg.dataset import TreeProcessor
from fragnnet.iceberg import fragmentation
from fragnnet.utils.spec_utils import calculate_cosine_similarity, binned_cosine_similarity


class IcebergGenPL(SpectrumPL):

	def _setup_model(self):

		self.model = IcebergGenModel(
			hidden_size=self.hparams.iceberg_hidden_size,
			layers=self.hparams.iceberg_num_layers,
			set_layers=self.hparams.iceberg_num_set_layers,
			dropout=self.hparams.iceberg_dropout,
			mpnn_type=self.hparams.iceberg_gnn_type,
			pool_op=self.hparams.iceberg_pool_type,
			inject_early=self.hparams.iceberg_inject_early,
			embed_adduct=self.hparams.iceberg_embed_adduct,
			encode_forms=self.hparams.iceberg_encode_forms,
			root_encode=self.hparams.magma_params["root_encode"],
			pe_embed_k=self.hparams.magma_params["pe_embed_k"],
			add_hs=self.hparams.magma_params["add_hs"],
		)
		assert self.hparams.spec_params["merge"]

		# compile
		if self.hparams.compile:
			th_dynamo.reset()
			self.dynamo_prof = th_dynamo.utils.CompileProfiler()

	def _setup_loss_fn(self):

		assert self.hparams.loss_type == "iceberg_gen_cross_entropy", self.hparams.loss_type
		def _loss_fn(
			outputs,
			targets,
			natoms,
			batch_idxs,
			**kwargs):
			bce_loss_fn = nn.BCELoss(reduction="none")
			loss = bce_loss_fn(outputs, targets.float())
			is_valid = (
				th.arange(loss.shape[1], device=loss.device)[None, :] < natoms[:, None]
			)
			assert (is_valid.sum(1) == natoms).all()
			loss = th.sum(loss * is_valid, dim=1) / natoms
			loss = scatter_reduce(
				loss, 
				batch_idxs,
				"sum",
				dim_size=th.max(batch_idxs)+1
			)
			loss_d = {
				"loss": loss
			}
			return loss_d
		self.loss_fn = _loss_fn
		self.metric_names.add("loss")

	def forward(self,**kwargs):

		outputs = self.model(
			graphs=kwargs["magma_frag_graphs"],
			root_repr=kwargs["magma_root_reprs"],
			ind_maps=kwargs["magma_inds"],
			broken=kwargs["magma_broken_bonds"],
			adducts=kwargs["magma_adducts"],
			root_forms=kwargs["magma_root_form_vecs"],
			frag_forms=kwargs["magma_frag_form_vecs"]
		)
		return outputs

	def _common_step(self, batch, split="train", log=True):

		batch_size = batch["batch_size"]
		unique_id = batch["spec_unique_id"]
		smiles = batch["mol_smiles"]
		targets = batch["magma_targ_atoms"]
		natoms = batch["magma_frag_atoms"]
		batch_idxs = batch["magma_inds"]
		outputs = self.forward(**batch)
		both = {
			"targets": targets,
			"natoms": natoms,
			"outputs": outputs,
			"batch_idxs": batch_idxs
		}
		loss_d = self.loss_fn(**both)
		loss = loss_d["loss"]
		mean_loss = th.mean(loss,dim=0)
		# just log loss
		if log:
			self.log(
				f"{split}_batch_loss",
				mean_loss,
				batch_size=batch_size,
				on_epoch=True
			)
		results = {
			"unique_id": unique_id,
			"smiles": smiles,
			"loss": loss,
			"mean_loss": mean_loss,
			**both,
			**loss_d
		}
		return results

	def _update_results(self, batch_results, split):

		if split == "train":
			num_log_images = self.hparams.num_log_train_images
			results_attr = "train_results"
			counter_attr = "train_counter"
		elif split == "val":
			num_log_images = self.hparams.num_log_val_images
			results_attr = "val_results"
			counter_attr = "val_counter"
		else:
			raise NotImplementedError(split)
		# filter keys (filtering first to save time/memory)
		keys = [
			"unique_ids",
			"smiles",
		]
		keys.extend(list(self.metric_names))
		unique_ids = batch_results.pop("unique_id")
		batch_results = {k:v for k,v in batch_results.items() if k in keys}
		batch_results["unique_ids"] = unique_ids
		# check all metrics
		assert all([metric_name in batch_results for metric_name in self.metric_names])
		# transfer to cpu
		batch_results = to_cpu(batch_results,detach=True)
		# setup results dict
		if getattr(self,results_attr) is None:
			setattr(self,results_attr,{k:list() for k in keys})
		else:
			assert set(keys) == set(getattr(self,results_attr).keys())
		results_dict = getattr(self,results_attr)
		# add to results dict
		for k,v in batch_results.items():
			results_dict[k].append(v)
		# increment counter
		setattr(self,counter_attr,getattr(self,counter_attr)+unique_ids.shape[0])

	def _reduce_metrics(self, results, split):

		mean_metrics, std_metrics = {}, {}
		for k, v in results.items():
			if k in self.metric_names:
				mean_metrics[k] = th.mean(v, dim=0)
				std_metrics[k] = th.std(v, dim=0)
		return mean_metrics, std_metrics

	def _log_images(self,split):

		if split == "train":
			num_log_images = self.hparams.num_log_train_images
		elif split == "val":
			num_log_images = self.hparams.num_log_val_images
		if num_log_images > 0:
			raise NotImplementedError("Logging images is not supported for IcebergGenPL")

	def predict_mol(
		self,
		root_smi: str,
		adduct,
		threshold=0,
		device: str = "cpu",
		max_nodes: int = None,
		tree_processor: TreeProcessor = None):
		"""predict_mol.

		Predict a new fragmentation tree from a starting root molecule
		autoregressively. First a new fragment is added to the
		frag_hash_to_entry dict and also put on the stack. Then it is
		fragmented and its "atoms_pulled" and "left_pred" are updated
		accordingly. The resulting new fragments are added to the hash.

		Args:
			root_smi (smi)
			threshold: Leaving probability
			device: Device
			max_nodes (int): Max number to include

		Return:
			Dictionary containing results
		"""

		# Step 1: Get a fragmentation engine for root mol
		engine = fragmentation.FragmentEngine(root_smi)
		max_depth = engine.max_tree_depth
		root_frag = engine.get_root_frag()
		root_form = common.form_from_smi(root_smi)
		root_form_vec = th.tensor(common.formula_to_dense(root_form), dtype=th.float, device=device)
		root_form_vec = root_form_vec.reshape(1, -1)
		adducts = th.tensor([common.ion2onehot_pos[adduct]], dtype=th.long, device=device)

		# Step 2: Featurize the root molecule
		root_graph_dict = tree_processor.featurize_frag(
			frag=root_frag,
			engine=engine,
			add_random_walk=True,
		)

		root_repr = None
		if self.model.root_encode == "gnn":
			root_repr = root_graph_dict["graph"].to(device)
		elif self.model.root_encode == "fp":
			root_fp = th.from_numpy(np.array(common.get_morgan_fp_smi(root_smi)))
			root_repr = root_fp.float().to(device)[None, :]

		form_to_min_score = {}
		frag_hash_to_entry = {}
		frag_to_hash = {}
		stack = [root_frag]
		depth = 0
		root_hash = engine.wl_hash(root_frag)
		frag_to_hash[root_frag] = root_hash
		root_score = engine.score_fragment(root_frag)[1]
		id_ = 0
		# TODO: Compute as in fragment engine
		root_entry = {
			"frag": int(root_frag),
			"frag_hash": root_hash,
			"parents": [],
			"atoms_pulled": [],
			"left_pred": [],
			"max_broken": 0,
			"tree_depth": 0,
			"id": 0,
			"prob_gen": 1,
			"score": root_score,
		}
		id_ += 1
		root_entry.update(engine.atom_pass_stats(root_frag, depth=0))
		form_to_min_score[root_entry["form"]] = root_entry["score"]
		frag_hash_to_entry[root_hash] = root_entry

		# Step 3: Run the autoregressive gen loop
		with th.inference_mode():

			# Note: we don't fragment at the final depth
			while len(stack) > 0 and depth < max_depth:
				# Convert all new frags to graphs (stack is to run next)

				new_dgl_dicts = [
					tree_processor.featurize_frag(
						frag=i, engine=engine, add_random_walk=True
					)
					for i in stack
				]
				tuple_list = [
					(i, j)
					for i, j in zip(new_dgl_dicts, stack)
					if i["graph"].num_nodes() > 1
				]

				if len(tuple_list) == 0:
					break

				new_dgl_dicts, stack = zip(*tuple_list)
				mol_batch_graph = [i["graph"] for i in new_dgl_dicts]
				frag_forms = [i["form"] for i in new_dgl_dicts]
				frag_form_vecs = [common.formula_to_dense(i) for i in frag_forms]
				frag_form_vecs = th.tensor(np.array(frag_form_vecs), dtype=th.float, device=device)

				new_frag_hashes = [engine.wl_hash(i) for i in stack]

				frag_to_hash.update(dict(zip(stack, new_frag_hashes)))

				frag_batch = dgl.batch(mol_batch_graph).to(device)
				inds = th.zeros(frag_batch.batch_size).long().to(device)

				# Note: Can speed by reducing redundant root graph passes
				# TODO: Figure out how to include frag form vec and root form
				# vec

				# TODO: Compute broken for each of these...

				broken_nums_ar = np.array(
					[frag_hash_to_entry[i]["max_broken"] for i in new_frag_hashes]
				)
				broken_nums_tensor = th.tensor(broken_nums_ar, dtype=th.float, device=device)

				pred_leaving = self.model.forward(
					graphs=frag_batch,
					root_repr=root_repr,
					ind_maps=inds,
					broken=broken_nums_tensor,
					adducts=adducts,
					root_forms=root_form_vec,
					frag_forms=frag_form_vecs,
				)
				depth += 1

				# Rank order all the atom preds and predictions
				# Continuously add items to the stack as long as they maintain
				# the max node constraint ranked by prob

				# Get all frag probabilities and sort them
				cur_probs = sorted(
					[i["prob_gen"] for i in frag_hash_to_entry.values()]
				)[::-1]
				if max_nodes is None or len(cur_probs) < max_nodes:
					min_prob = threshold
				elif max_nodes is not None and len(cur_probs) >= max_nodes:
					min_prob = cur_probs[max_nodes - 1]
				else:
					raise NotImplementedError()

				new_items = list(
					zip(stack, new_frag_hashes, pred_leaving, new_dgl_dicts)
				)
				sorted_order = []
				for item_ind, item in enumerate(new_items):
					frag_hash = item[1]
					pred_vals_f = item[2]
					parent_prob = frag_hash_to_entry[frag_hash]["prob_gen"]
					for atom_ind, (atom_pred, prob_gen) in enumerate(
						zip(pred_vals_f, parent_prob * pred_vals_f)
					):

						sorted_order.append(
							dict(
								item_ind=item_ind,
								atom_ind=atom_ind,
								prob_gen=prob_gen.item(),
								atom_pred=atom_pred.item(),
							)
						)

				sorted_order = sorted(sorted_order, key=lambda x: -x["prob_gen"])
				new_stack = []

				# Process ordered list continuously
				for new_item in sorted_order:
					prob_gen = new_item["prob_gen"]
					atom_ind = new_item["atom_ind"]
					atom_pred = new_item["atom_pred"]
					item_ind = new_item["item_ind"]

					# Filter out on minimum prob
					if prob_gen <= min_prob:
						continue

					# Calc stack ind
					orig_entry = new_items[item_ind]
					frag_int = orig_entry[0]
					frag_hash = orig_entry[1]
					dgl_dict = orig_entry[3]

					# Get atom ind
					atom = dgl_dict["new_to_old"][atom_ind]

					# Calc remove dict
					out_dicts = engine.remove_atom(frag_int, int(atom))

					# Update atoms_pulled for parent
					frag_hash_to_entry[frag_hash]["atoms_pulled"].append(int(atom))
					frag_hash_to_entry[frag_hash]["left_pred"].append(float(atom_pred))
					parent_broken = frag_hash_to_entry[frag_hash]["max_broken"]

					for out_dict in out_dicts:
						out_hash = out_dict["new_hash"]
						out_frag = out_dict["new_frag"]
						rm_bond_t = out_dict["rm_bond_t"]
						frag_to_hash[out_frag] = out_hash
						current_entry = frag_hash_to_entry.get(out_hash)

						max_broken = parent_broken + rm_bond_t

						# Define probability of generating
						if current_entry is None:
							score = engine.score_fragment(int(out_frag))[1]

							new_stack.append(out_frag)
							new_entry = {
								"frag": int(out_frag),
								"frag_hash": out_hash,
								"score": score,
								"id": id_,
								"parents": [frag_hash],
								"atoms_pulled": [],
								"left_pred": [],
								"max_broken": max_broken,
								"tree_depth": depth,
								"prob_gen": prob_gen,
							}
							id_ += 1
							new_entry.update(
								engine.atom_pass_stats(out_frag, depth=max_broken)
							)

							# reset to best score
							temp_form = new_entry["form"]
							prev_best_score = form_to_min_score.get(
								temp_form, float("inf")
							)
							form_to_min_score[temp_form] = min(
								new_entry["score"], prev_best_score
							)
							frag_hash_to_entry[out_hash] = new_entry

						else:
							current_entry["parents"].append(frag_hash)
							current_entry["prob_gen"] = max(
								current_entry["prob_gen"], prob_gen
							)

						# Update cur probs
						# This is inefficeint and can be made smarter withotu
						# doing another minimum calculation
						cur_probs = sorted(
							[i["prob_gen"] for i in frag_hash_to_entry.values()]
						)[::-1]
						if max_nodes is None or len(cur_probs) < max_nodes:
							min_prob = threshold
						elif max_nodes is not None and len(cur_probs) >= max_nodes:
							min_prob = cur_probs[max_nodes - 1]
						else:
							raise NotImplementedError()

				# Truncate stack; this should be handled above by min prob
				# if max_nodes is not None:
				#    new_stack = sorted(new_stack,
				#                       key=lambda x:
				#                       -frag_hash_to_entry[frag_to_hash[x]]['prob_gen'])
				#    new_stack = new_stack[:max_nodes]
				stack = new_stack

		# Only get min score for ech formula
		frag_hash_to_entry = {
			k: v
			for k, v in frag_hash_to_entry.items()
			if form_to_min_score[v["form"]] == v["score"]
		}

		if max_nodes is not None:
			sorted_keys = sorted(
				list(frag_hash_to_entry.keys()),
				key=lambda x: -frag_hash_to_entry[x]["prob_gen"],
			)
			frag_hash_to_entry = {
				k: frag_hash_to_entry[k] for k in sorted_keys[:max_nodes]
			}
		return frag_hash_to_entry


class IcebergIntenPL(SpectrumPL):

	def _setup_model(self):

		self.model = IcebergIntenModel(
			hidden_size=self.hparams.iceberg_hidden_size,
			mlp_layers=self.hparams.iceberg_num_mlp_layers,
			gnn_layers=self.hparams.iceberg_num_gnn_layers,
			set_layers=self.hparams.iceberg_num_set_layers,
			frag_set_layers=self.hparams.iceberg_num_frag_set_layers,
			dropout=self.hparams.iceberg_dropout,
			mpnn_type=self.hparams.iceberg_gnn_type,
			pool_op=self.hparams.iceberg_pool_type,
			inject_early=self.hparams.iceberg_inject_early,
			embed_adduct=self.hparams.iceberg_embed_adduct,
			encode_forms=self.hparams.iceberg_encode_forms,
			root_encode=self.hparams.magma_params["root_encode"],
			pe_embed_k=self.hparams.magma_params["pe_embed_k"],
			add_hs=self.hparams.magma_params["add_hs"],
			binned_targs=self.hparams.magma_params["binned_targs"],
			mz_bin_res=self.hparams.magma_params["mz_bin_res"],
			mz_max=self.hparams.magma_params["mz_max"],
			sum_ints=self.hparams.magma_params["sum_ints"],
			output_formula_str=self.hparams.output_formula_str,
			fix_batch_bug=self.hparams.iceberg_fix_batch_bug,
		)
		if self.hparams.magma_params["binned_targs"]:
			assert self.hparams.mz_bin_res == self.hparams.magma_params["mz_bin_res"]
			assert self.hparams.mz_max == self.hparams.magma_params["mz_max"]
			assert self.hparams.sum_ints == self.hparams.magma_params["sum_ints"]
		assert self.hparams.spec_params["merge"]

		# compile
		if self.hparams.compile:
			th_dynamo.reset()
			self.dynamo_prof = th_dynamo.utils.CompileProfiler()

	def _setup_loss_fn(self):

		# binned cosine distance
		def cos_dist_fn(
			true_mzs,
			true_logprobs,
			true_batch_idxs,
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs):
			return sparse_cosine_distance_binned(
				true_mzs,
				true_logprobs,
				true_batch_idxs,
				pred_mzs,
				pred_logprobs,
				pred_batch_idxs,
				log_distance=(self.hparams.loss_type == "log_cosine_distance")
			)

		# cosine distance
		assert self.hparams.loss_type == "cosine_distance", self.hparams.loss_type
		assert self.hparams.sparse_cosine_similarity
		self.binned_loss = True
		
		def _loss_fn(
			true_mzs,
			true_logprobs,
			true_batch_idxs,
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs,
			**kwargs):
		
			spec_cd = cos_dist_fn(
				true_mzs,
				true_logprobs,
				true_batch_idxs,
				pred_mzs,
				pred_logprobs,
				pred_batch_idxs,
			)
			primary_loss = spec_cd

			loss = primary_loss
			loss_d = {
				"loss": loss,
				"primary_loss": primary_loss
			}
			return loss_d

		self.loss_fn = _loss_fn
		loss_names = [
			"loss",
			"primary_loss"
		]
		self.metric_names.update(loss_names)

	def forward(self,**kwargs):

		outputs = self.model(
			graphs=kwargs["magma_frag_graphs"],
			root_repr=kwargs["magma_root_reprs"],
			ind_maps=kwargs["magma_inds"],
			num_frags=kwargs["magma_num_frags"],
			broken=kwargs["magma_broken_bonds"],
			adducts=kwargs["magma_adducts"],
			max_add_hs=kwargs["magma_max_add_hs"],
			max_remove_hs=kwargs["magma_max_remove_hs"],
			masses=kwargs["magma_masses"],
			root_forms=kwargs["magma_root_form_vecs"],
			frag_forms=kwargs["magma_frag_form_vecs"],
			adduct_form_deltas=kwargs.get("magma_adduct_form_deltas",None),
		)
		return outputs
