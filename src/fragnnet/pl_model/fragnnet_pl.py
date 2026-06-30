import logging
from typing import Any

import torch as th
import torch._dynamo as th_dynamo
import torch.nn.functional as F

try:
    from torch._dynamo.utils import CompileProfiler

    profiler_available = True
except ImportError:
    profiler_available = False
import os

from fragnnet.model import FraGNNetModel
from fragnnet.model.loss import (
    get_edge_loss_fn,
    get_pairwise_cossim,
    get_pairwise_cross_entropy,
    get_pairwise_jss_sim,
    get_sparse_cross_entropy_fn,
    sparse_conditional_entropy_fn,
    sparse_cosine_distance_binned,
    sparse_cosine_distance_hungarian,
    sparse_entropy_fn,
    sparse_jensen_shannon_divergence,
    sparse_jensen_shannon_divergence_hungarian,
    sparse_jensen_shannon_divergence_hungarian_vec,
)
from fragnnet.pl_model import SpectrumPL
from fragnnet.utils.misc_utils import safelog, scatter_logsumexp
from fragnnet.utils.nn_utils import (
    build_lr_scheduler,
    decompile_jit_ckpt,
    get_optimizer_class,
)
from fragnnet.utils.spec_utils import scatter_reduce


class FraGNNetPL(SpectrumPL):
    hparams: Any

    def _setup_model(self):
        # frag GNN
        self.model = FraGNNetModel(
            num_depth=self.hparams.num_depth,
            num_hs=self.hparams.num_hs,
            num_elements=self.hparams.num_elements,
            int_embedder=self.hparams.int_embedder,
            int_embedder_tight=self.hparams.int_embedder_tight,
            mol_node_feats=self.hparams.mol_params["pyg_node_feats"],
            mol_edge_feats=self.hparams.mol_params["pyg_edge_feats"],
            mol_pe_embed_k=self.hparams.mol_params["pyg_pe_embed_k"],
            mol_hidden_size=self.hparams.mol_hidden_size,
            mol_num_layers=self.hparams.mol_num_layers,
            mol_gnn_type=self.hparams.mol_gnn_type,
            mol_dropout=self.hparams.mol_dropout,
            mol_normalization=self.hparams.mol_normalization,
            mol_num_heads=self.hparams["mol_num_heads"],
            mol_pool_type=self.hparams.mol_pool_type,
            frag_node_feats=self.hparams.frag_params["pyg_node_feats"],
            frag_edge_feats=self.hparams.frag_params["pyg_edge_feats"],
            frag_hidden_size=self.hparams.frag_hidden_size,
            frag_num_layers=self.hparams.frag_num_layers,
            frag_gnn_type=self.hparams.frag_gnn_type,
            frag_dropout=self.hparams.frag_dropout,
            frag_normalization=self.hparams.frag_normalization,
            frag_pool_type=self.hparams.frag_pool_type,
            frag_embed_combine=self.hparams.frag_embed_combine,
            frag_pool_combine=self.hparams.frag_pool_combine,
            mlp_output_format=self.hparams.mlp_output_format,
            mlp_hidden_size=self.hparams.mlp_hidden_size,
            mlp_normalization=self.hparams.mlp_normalization,
            mlp_dropout=self.hparams.mlp_dropout,
            mlp_num_layers=self.hparams.mlp_num_layers,
            mlp_use_residuals=self.hparams.mlp_use_residuals,
            cc_interstage_type=self.hparams.cc_interstage_type,
            cc_interstage_use_rest=self.hparams.cc_interstage_use_rest,
            nb_iso=self.hparams.nb_iso,
            skip_edge_loss=self.hparams.skip_edge_loss,
            mask_null_formula=self.hparams.mask_null_formula,
            predict_oos=self.hparams.predict_oos,
            bin_output=self.hparams.bin_output,
            mz_bin_res=self.hparams.mz_bin_res,
            mz_max=self.hparams.mz_max,
            ce_insert_type=self.hparams.ce_insert_type,
            ce_insert_location=self.hparams.ce_insert_location,
            ce_insert_merge=self.hparams.ce_insert_merge,
            ce_insert_size=self.hparams.ce_insert_size,
            nce_mean=self.hparams.nce_mean,
            nce_std=self.hparams.nce_std,
            nce_max=self.hparams.nce_max,
            use_nce=self.hparams.spec_params["nce"],
            use_ace=self.hparams.spec_params["ace"],
            ace_max=self.hparams.ace_max,
            ace_mean=self.hparams.ace_mean,
            ace_std=self.hparams.ace_std,
            prec_insert_location=self.hparams.prec_insert_location,
            prec_insert_size=self.hparams.prec_insert_size,
            prec_types=self.hparams.spec_params["prec_types"],
            inst_insert_location=self.hparams.inst_insert_location,
            inst_insert_size=self.hparams.inst_insert_size,
            inst_types=self.hparams.spec_params["inst_types"],
            frag_mode_insert_location=self.hparams.frag_mode_insert_location,
            frag_mode_insert_size=self.hparams.frag_mode_insert_size,
            frag_modes=self.hparams.spec_params["frag_modes"],
            frag_mode_scale=self.hparams.frag_mode_scale,
            ce_scale=self.hparams.ce_scale,
            ce_scaler_hidden_dim=self.hparams.ce_scaler_hidden_dim,
            output_formula_str=self.hparams.output_formula_str,
            cc_feature_dropout=self.hparams.cc_feature_dropout,
            debug_validate_outputs=self.hparams.debug_validate_outputs,
        )

        self._check_ce_params()
        self.pretrained_param_names = []
        # check edge loss params
        if not self.hparams.skip_edge_loss:
            assert "h_counts" in self.hparams.frag_params["pyg_node_feats"]
            assert "h_range" in self.hparams.frag_params["pyg_edge_feats"]
        else:
            if "h_counts" in self.hparams.frag_params["pyg_node_feats"]:
                logging.warning("h_counts in frag pyg_node_feats but edge_loss is disabled!")
            if "h_range" in self.hparams.frag_params["pyg_edge_feats"]:
                logging.warning("h_range in frag pyg_edge_feats but edge_loss is disabled!")

        # Load pre-trained weights if specified for finetuning
        if self.hparams.finetune:
            self._load_pretrained_weights(
                self.hparams.finetune["pretrained_ckpt"],
                strict=self.hparams.finetune["strict_load"],
                shared_components=self.hparams.finetune["finetune_shared_components"]
                if not self.hparams.finetune["shared_all"]
                else None,
            )

            # Handle freezing
            if self.hparams.finetune["freeze_all_shared"]:
                # If freeze_all_shared is True, we freeze whatever we loaded
                if self.pretrained_param_names:
                    self._freeze_components(self.pretrained_param_names, exact_match=True)
            elif self.hparams.finetune["freeze_components"]:
                freeze_components = self.hparams.finetune["freeze_components"]
                # Otherwise freeze specific components by prefix
                self._freeze_components(freeze_components, exact_match=False)

        # compile
        if self.hparams.compile:
            th_dynamo.reset()
            if profiler_available:
                self.dynamo_prof = CompileProfiler()
                self.model = self.model.get_compile(backend=self.dynamo_prof, dynamic=True)
            else:
                self.model = th.compile(self.model, dynamic=True)

    def _load_pretrained_weights(
        self,
        ckpt_path: str,
        strict: bool = False,
        shared_components: list[str] | None = None,
    ):
        """
        Load pre-trained weights from a checkpoint file.
        Supports both MCES pre-training (partial load) and generic fine-tuning.

        Args:
            ckpt_path: Path to checkpoint file
            strict: Whether to enforce strict state dict matching
            shared_components: List of component prefixes to load (if None, load all matching)
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        logging.info(f"Loading pre-trained weights from: {ckpt_path}")
        ckpt = th.load(ckpt_path, map_location=self.device, weights_only=False)
        # fix compiled ckpt state dict prefix
        ckpt = decompile_jit_ckpt(ckpt)
        # 1. Load ckpt
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        # Determine target components
        if shared_components is not None:
            target_components = shared_components
        else:
            target_components = None

        # 2. Check if shared_components are in the target and source
        target_keys = set(self.model.state_dict().keys())

        # Normalize source keys (remove 'model.' prefix if present)
        source_keys = list(state_dict.keys())
        source_starts_with_model = any(k.startswith("model.") for k in source_keys)
        normalized_source_dict = {}
        for k, v in state_dict.items():
            norm_k = k[6:] if (source_starts_with_model and k.startswith("model.")) else k
            normalized_source_dict[norm_k] = v

        if target_components:
            for comp in target_components:
                # Check target
                if not any(k.startswith(comp) for k in target_keys):
                    logging.warning(f"Component '{comp}' not found in current model (target).")
                # Check source
                if not any(k.startswith(comp) for k in normalized_source_dict):
                    logging.warning(f"Component '{comp}' not found in checkpoint (source).")
        else:
            logging.info(
                "No specific shared_components provided; attempting to load all compatible weights."
            )

        # 3. Load model weight into target components
        weights_to_load = {}
        for key in target_keys:
            # Filter by component
            if target_components is not None:
                if not any(key.startswith(comp) for comp in target_components):
                    continue

            # Check if key exists in source
            if key in normalized_source_dict:
                weights_to_load[key] = normalized_source_dict[key]

        if not weights_to_load:
            logging.warning("No compatible weights found to transfer")
            return

        # Load into self.model
        try:
            self.model.load_state_dict(weights_to_load, strict=False)
            logging.info(f"Successfully loaded {len(weights_to_load)} parameters.")

            # 4. Update dictionary for tracking
            self.pretrained_param_names = list(weights_to_load.keys())

        except RuntimeError as e:
            if strict:
                raise e
            else:
                logging.warning(f"Load failed with error: {e}")

    def _freeze_components(self, components: list[str], exact_match: bool = False):
        """
        Freeze parameters based on component names or prefixes.

        Args:
            components: List of component names (exact or prefixes)
            exact_match: If True, requires exact name match (for specific weights).
                         If False, matches as prefix (for whole submodules).
        """
        frozen_count = 0
        # Iterate over model parameters directly
        for name, param in self.model.named_parameters():
            should_freeze = False
            if exact_match:
                if name in components:
                    should_freeze = True
            else:
                if any(name.startswith(c) for c in components):
                    should_freeze = True

            if should_freeze:
                param.requires_grad = False
                frozen_count += 1

        logging.info(f"Froze {frozen_count} parameters (exact_match={exact_match})")

    def _setup_loss_names(self):
        """Setup the list of loss names to be tracked during training/validation.

        A term is included only when at least one of its associated weights is non-zero.
        Zero-weight terms are excluded from ``loss_names`` and never computed.
        """
        loss_names = [
            "loss",
            "primary_loss",
            "null_formula_prob",
            "oos_prob",
        ]

        if self.hparams.loss_type == "cross_entropy":
            loss_names.extend(["spec_ce", "oos_ce", "ios_ce"])
        elif "cross_entropy" in self.hparams.auxiliary_scores:
            loss_names.append("spec_ce")

        def _add_if_active(names: list[str], *weights: float) -> None:
            """Add names only when at least one weight is non-zero."""
            if any(w != 0.0 for w in weights):
                loss_names.extend(names)

        # Entropy terms
        _add_if_active(
            ["formula_e", "formula_ne"],
            self.hparams.formula_entropy_weight,
            self.hparams.formula_normalized_entropy_weight,
        )
        _add_if_active(
            ["node_e", "node_ne"],
            self.hparams.node_entropy_weight,
            self.hparams.node_normalized_entropy_weight,
        )
        _add_if_active(
            ["node_formula_e", "node_formula_ne"],
            self.hparams.node_formula_entropy_weight,
            self.hparams.node_formula_normalized_entropy_weight,
        )
        _add_if_active(
            ["formula_node_e", "formula_node_ne"],
            self.hparams.formula_node_entropy_weight,
            self.hparams.formula_node_normalized_entropy_weight,
        )
        _add_if_active(
            ["joint_e", "joint_ne"],
            self.hparams.joint_entropy_weight,
            self.hparams.joint_normalized_entropy_weight,
        )

        # Edge losses: only when edge computation is not globally disabled
        if not self.hparams.skip_edge_loss:
            _add_if_active(
                ["edge_h_range_loss", "edge_h_transfer_loss"],
                self.hparams.edge_h_range_loss_weight,
                self.hparams.edge_h_transfer_loss_weight,
            )
            _add_if_active(
                ["edge_e", "edge_ne"],
                self.hparams.edge_entropy_weight,
                self.hparams.edge_normalized_entropy_weight,
            )

        # Neighbor isolation losses
        if self.hparams.nb_iso:
            _add_if_active(
                ["nb_node_e", "nb_node_ne"],
                self.hparams.nb_node_entropy_weight,
                self.hparams.nb_node_normalized_entropy_weight,
            )
            _add_if_active(
                ["nb_node_formula_e", "nb_node_formula_ne"],
                self.hparams.nb_node_formula_entropy_weight,
                self.hparams.nb_node_formula_normalized_entropy_weight,
            )
            _add_if_active(
                ["nb_formula_node_e", "nb_formula_node_ne"],
                self.hparams.nb_formula_node_entropy_weight,
                self.hparams.nb_formula_node_normalized_entropy_weight,
            )
            _add_if_active(
                ["nb_joint_e", "nb_joint_ne"],
                self.hparams.nb_joint_entropy_weight,
                self.hparams.nb_joint_normalized_entropy_weight,
            )
            _add_if_active(
                ["nb_node_node_e", "nb_node_node_ne"],
                self.hparams.nb_node_node_entropy_weight,
                self.hparams.nb_node_node_normalized_entropy_weight,
            )

        _add_if_active(
            [f"pairwise_cossine_{self.hparams.mz_bin_res}"],
            self.hparams.pairwise_cosine_loss_weight,
        )
        _add_if_active(
            [f"pairwise_jss_{self.hparams.mz_bin_res}"],
            self.hparams.pairwise_jss_loss_weight,
        )
        _add_if_active(
            [f"pairwise_cross_entropy_{self.hparams.mz_bin_res}"],
            self.hparams.pairwise_cross_entropy_loss_weight,
        )

        self.loss_names = loss_names
        self.metric_names.update(loss_names)

    def _setup_loss_fn(self):
        # cross entropy
        sparse_ce_fn = get_sparse_cross_entropy_fn(
            dist=self.hparams.output_distribution,
            vectorized=self.hparams.loss_vectorized,
            tolerance=self.tolerance,
            relative=self.relative,
            tolerance_min_mz=self.tolerance_min_mz,
            oos_tolerance_multiple=self.hparams.oos_tolerance_multiple,
            gaussian_renormalize=self.hparams.gaussian_renormalize,
            pm_tolerance_multiple=self.hparams.pm_tolerance_multiple,
            loss_batch_size=self.hparams.loss_batch_size,
        )
        # binned cosine distance
        def cos_dist_fn(
            true_mzs,
            true_logprobs,
            true_batch_idxs,
            pred_mzs,
            pred_logprobs,
            pred_batch_idxs,
        ):
            return sparse_cosine_distance_binned(
                true_mzs,
                true_logprobs,
                true_batch_idxs,
                pred_mzs,
                pred_logprobs,
                pred_batch_idxs,
                log_distance=(self.hparams.loss_type == "log_cosine_distance"),
            )

        # hungarian cosine distance
        def cos_hun_fn(
            true_mzs,
            true_logprobs,
            true_batch_idxs,
            pred_mzs,
            pred_logprobs,
            pred_batch_idxs,
            tol_per_sample=None,
            min_mz_per_sample=None,
        ):
            return sparse_cosine_distance_hungarian(
                true_mzs,
                true_logprobs,
                true_batch_idxs,
                pred_mzs,
                pred_logprobs,
                pred_batch_idxs,
                tolerance=self.tolerance,
                relative=self.relative,
                log_distance=(self.hparams.loss_type == "log_cosine_distance_hungarian"),
                tol_per_sample=tol_per_sample,
                min_mz_per_sample=min_mz_per_sample,
            )

        # jensen-shannon distance
        def jss_dist_fn(
            true_mzs,
            true_logprobs,
            true_batch_idxs,
            pred_mzs,
            pred_logprobs,
            pred_batch_idxs,
        ):
            return sparse_jensen_shannon_divergence(
                true_mzs,
                true_logprobs,
                true_batch_idxs,
                pred_mzs,
                pred_logprobs,
                pred_batch_idxs,
                mz_max=self.hparams.mz_max,
                mz_bin_res=self.hparams.mz_bin_res,
            )

        # jensen-shannon distance hungarian
        def jss_dist_hun_fn(
            true_mzs,
            true_logprobs,
            true_batch_idxs,
            pred_mzs,
            pred_logprobs,
            pred_batch_idxs,
            tol_per_sample=None,
            min_mz_per_sample=None,
        ):
            kwargs = {
                "tolerance": self.tolerance,
                "relative": self.relative,
                "tol_per_sample": tol_per_sample,
                "min_mz_per_sample": min_mz_per_sample,
            }
            if self.tolerance_min_mz is not None:
                kwargs["tolerance_min_mz"] = self.tolerance_min_mz
            if self.hparams.loss_vectorized:
                kwargs["loss_batch_size"] = self.hparams.loss_batch_size
                return sparse_jensen_shannon_divergence_hungarian_vec(
                    true_mzs,
                    true_logprobs,
                    true_batch_idxs,
                    pred_mzs,
                    pred_logprobs,
                    pred_batch_idxs,
                    **kwargs,
                )
            return sparse_jensen_shannon_divergence_hungarian(
                true_mzs,
                true_logprobs,
                true_batch_idxs,
                pred_mzs,
                pred_logprobs,
                pred_batch_idxs,
                **kwargs,
            )

        assert self.hparams.sparse_cosine_similarity
        assert self.hparams.sum_ints
        if self.hparams.loss_type == "cross_entropy":
            self.binned_loss = False
        elif self.hparams.loss_type in ["cosine_distance", "log_cosine_distance"]:
            self.binned_loss = True
        elif self.hparams.loss_type in [
            "cosine_distance_hungarian",
            "log_cosine_distance_hungarian",
        ] or self.hparams.loss_type in ["jss_distance", "jss_distance_hungarian"]:
            self.binned_loss = False
        else:
            raise ValueError(f"Unknown loss type {self.hparams.loss_type}")

        if self.binned_loss:
            if self.hparams.ints_transform != "none" and not self.hparams.bin_output:
                print(
                    "> Warning: binned loss with ints transform; output is not binned; inverse transform will be biased for binned metrics"
                )
        else:
            assert not self.hparams.bin_output, "cannot train binned model with unbinned loss"

        if not self.hparams.skip_edge_loss:
            edge_h_range_loss_fn = get_edge_loss_fn(self.hparams.edge_h_range_loss_fn, 4.0)
            edge_h_transfer_loss_fn = get_edge_loss_fn(self.hparams.edge_h_transfer_loss_fn, 4.0)

            def edge_loss(
                pred_edge_logprobs,
                pred_edge_h_diffs,
                pred_edge_h_range_masks,
                pred_edge_h_logprobs,
                pred_edge_batch_idxs,
            ):
                edge_h_range_costs = edge_h_range_loss_fn(
                    pred_edge_h_diffs * pred_edge_h_range_masks.float()
                )
                edge_h_range_costs = edge_h_range_costs.reshape(edge_h_range_costs.shape[0], -1)
                edge_h_transfer_costs = edge_h_transfer_loss_fn(pred_edge_h_diffs)
                edge_h_transfer_costs = edge_h_transfer_costs.reshape(
                    edge_h_transfer_costs.shape[0], -1
                )
                edge_h_logprobs = pred_edge_h_logprobs.reshape(pred_edge_h_logprobs.shape[0], -1)
                edge_h_range_avg_cost = th.logsumexp(
                    edge_h_logprobs + safelog(edge_h_range_costs, eps=self.hparams.log_min),
                    dim=1,
                )
                edge_h_transfer_avg_cost = th.logsumexp(
                    edge_h_logprobs + safelog(edge_h_transfer_costs, eps=self.hparams.log_min),
                    dim=1,
                )
                h_range_loss = scatter_logsumexp(
                    pred_edge_logprobs + edge_h_range_avg_cost, pred_edge_batch_idxs
                )
                h_transfer_loss = scatter_logsumexp(
                    pred_edge_logprobs + edge_h_transfer_avg_cost, pred_edge_batch_idxs
                )
                return h_range_loss, h_transfer_loss

        self._setup_loss_names()

        def _loss_fn(
            true_mzs,
            true_logprobs,
            true_batch_idxs,
            pred_mzs,
            pred_logprobs,
            pred_batch_idxs,
            pred_formula_logprobs,
            pred_formula_batch_idxs,
            pred_node_logprobs,
            pred_node_batch_idxs,
            pred_node_formula_logprobs,
            pred_formula_node_logprobs,
            pred_joint_logprobs,
            pred_joint_node_idxs,
            pred_joint_formula_idxs,
            pred_joint_batch_idxs,
            pred_null_formula_logprob,
            pred_edge_logprobs,
            pred_edge_h_diffs,
            pred_edge_h_range_masks,
            pred_edge_h_logprobs,
            pred_edge_batch_idxs,
            pred_oos_logprobs,
            pred_h_logprobs,
            pred_h_counts,
            pred_h_batch_idxs,
            pred_nb_node_logprobs,
            pred_nb_node_formula_logprobs,
            pred_nb_formula_node_logprobs,
            pred_nb_joint_logprobs,
            pred_nb_node_node_logprobs,
            pred_nb_node_node_idxs,
            pred_nb_node_batch_idxs,
            pred_nb_joint_node_idxs,
            pred_nb_joint_formula_idxs,
            pred_nb_joint_batch_idxs,
            pred_nb_node_node_node_idxs,
            pred_nb_node_node_batch_idxs,
            pred_formula_formula_idxs=None,
            pred_node_node_idxs=None,
            **kwargs,
        ):
            loss_d = {}
            tol_per_sample = kwargs.get("tol_per_sample")
            min_mz_per_sample = kwargs.get("min_mz_per_sample")

            if self.hparams.loss_type == "cross_entropy":
                ios_ce, oos_ce, _, _ = sparse_ce_fn(
                    true_mzs,
                    true_logprobs,
                    true_batch_idxs,
                    pred_mzs,
                    pred_logprobs,
                    pred_batch_idxs,
                    pred_oos_logprobs,
                    tol_per_sample=tol_per_sample,
                    min_mz_per_sample=min_mz_per_sample,
                )
                spec_ce = ios_ce + oos_ce
                if self.hparams.oos_loss:
                    primary_loss = spec_ce
                else:
                    primary_loss = ios_ce
                loss_d["ios_ce"] = ios_ce
                loss_d["oos_ce"] = oos_ce
                loss_d["spec_ce"] = spec_ce
            elif self.hparams.loss_type in ["cosine_distance", "log_cosine_distance"]:
                primary_loss = cos_dist_fn(
                    true_mzs,
                    true_logprobs,
                    true_batch_idxs,
                    pred_mzs,
                    pred_logprobs,
                    pred_batch_idxs,
                )
            elif self.hparams.loss_type in [
                "cosine_distance_hungarian",
                "log_cosine_distance_hungarian",
            ]:
                primary_loss = cos_hun_fn(
                    true_mzs,
                    true_logprobs,
                    true_batch_idxs,
                    pred_mzs,
                    pred_logprobs,
                    pred_batch_idxs,
                    tol_per_sample=tol_per_sample,
                    min_mz_per_sample=min_mz_per_sample,
                )
            elif self.hparams.loss_type == "jss_distance":
                primary_loss = jss_dist_fn(
                    true_mzs,
                    true_logprobs,
                    true_batch_idxs,
                    pred_mzs,
                    pred_logprobs,
                    pred_batch_idxs,
                )
            elif self.hparams.loss_type == "jss_distance_hungarian":
                primary_loss = jss_dist_hun_fn(
                    true_mzs,
                    true_logprobs,
                    true_batch_idxs,
                    pred_mzs,
                    pred_logprobs,
                    pred_batch_idxs,
                    tol_per_sample=tol_per_sample,
                    min_mz_per_sample=min_mz_per_sample,
                )

            loss = self.hparams.primary_loss_weight * primary_loss
            loss_d["primary_loss"] = primary_loss

            if self.hparams.loss_type != "cross_entropy" and "spec_ce" in self.loss_names:
                ios_ce, oos_ce, _, _ = sparse_ce_fn(
                    true_mzs,
                    true_logprobs,
                    true_batch_idxs,
                    pred_mzs,
                    pred_logprobs,
                    pred_batch_idxs,
                    pred_oos_logprobs,
                    tol_per_sample=tol_per_sample,
                    min_mz_per_sample=min_mz_per_sample,
                )
                loss_d["spec_ce"] = ios_ce + oos_ce

            if self.hparams.null_loss:
                assert self.hparams.loss_type != "cross_entropy", self.hparams.loss_type
                if self.hparams.loss_type in [
                    "cosine_distance",
                    "cosine_distance_hungarian",
                    "jss_distance",
                    "jss_distance_hungarian",
                ]:
                    loss = loss + self.hparams.null_loss_weight * th.exp(pred_null_formula_logprob)
                elif self.hparams.loss_type in [
                    "log_cosine_distance",
                    "log_cosine_distance_hungarian",
                ]:
                    loss = loss + self.hparams.null_loss_weight * pred_null_formula_logprob

            if "spec_e" in self.loss_names or "spec_ne" in self.loss_names:
                spec_e, spec_ne = sparse_entropy_fn(
                    pred_logprobs,
                    pred_batch_idxs,
                    oos_logprobs=pred_oos_logprobs,
                    renormalize=True,
                )
                loss_d["spec_e"] = spec_e
                loss_d["spec_ne"] = spec_ne

            if "formula_e" in self.loss_names or "formula_ne" in self.loss_names:
                # formula logprobs do not consider NULL, OOS
                formula_e, formula_ne = sparse_entropy_fn(
                    pred_formula_logprobs,
                    pred_formula_batch_idxs,
                    support_size_delta=-1.0,  # for NULL formula
                )
                loss = (
                    loss
                    + self.hparams.formula_entropy_weight * formula_e
                    + self.hparams.formula_normalized_entropy_weight * formula_ne
                )
                loss_d["formula_e"] = formula_e
                loss_d["formula_ne"] = formula_ne

            if "node_e" in self.loss_names or "node_ne" in self.loss_names:
                node_e, node_ne = sparse_entropy_fn(pred_node_logprobs, pred_node_batch_idxs)
                loss = (
                    loss
                    + self.hparams.node_entropy_weight * node_e
                    + self.hparams.node_normalized_entropy_weight * node_ne
                )
                loss_d["node_e"] = node_e
                loss_d["node_ne"] = node_ne

            if "node_formula_e" in self.loss_names or "node_formula_ne" in self.loss_names:
                # node_formula logprobs do not consider NULL, OOS
                node_formula_e, node_formula_ne = sparse_conditional_entropy_fn(
                    pred_node_logprobs,
                    pred_node_batch_idxs,
                    pred_node_formula_logprobs,
                    pred_joint_node_idxs,
                    pred_joint_batch_idxs,
                )
                loss = (
                    loss
                    + self.hparams.node_formula_entropy_weight * node_formula_e
                    + self.hparams.node_formula_normalized_entropy_weight * node_formula_ne
                )
                loss_d["node_formula_e"] = node_formula_e
                loss_d["node_formula_ne"] = node_formula_ne

            if "formula_node_e" in self.loss_names or "formula_node_ne" in self.loss_names:
                # formula_node logprobs do not consider NULL, OOS
                formula_node_e, formula_node_ne = sparse_conditional_entropy_fn(
                    pred_formula_logprobs,
                    pred_formula_batch_idxs,
                    pred_formula_node_logprobs,
                    pred_joint_formula_idxs,
                    pred_joint_batch_idxs,
                )
                loss = (
                    loss
                    + self.hparams.formula_node_entropy_weight * formula_node_e
                    + self.hparams.formula_node_normalized_entropy_weight * formula_node_ne
                )
                loss_d["formula_node_e"] = formula_node_e
                loss_d["formula_node_ne"] = formula_node_ne

            if "joint_e" in self.loss_names or "joint_ne" in self.loss_names:
                # joint
                joint_e, joint_ne = sparse_entropy_fn(pred_joint_logprobs, pred_joint_batch_idxs)
                # assert th.all(th.isclose(joint_e, node_formula_e+node_e,rtol=0.,atol=0.001))
                # assert th.all(th.isclose(joint_e, formula_node_e+formula_e,rtol=0.,atol=0.001))
                loss = (
                    loss
                    + self.hparams.joint_entropy_weight * joint_e
                    + self.hparams.joint_normalized_entropy_weight * joint_ne
                )
                loss_d["joint_e"] = joint_e
                loss_d["joint_ne"] = joint_ne

            if "node_formula_mi" in self.loss_names:
                # mi
                node_formula_mi = formula_e - node_formula_e
                loss_d["node_formula_mi"] = node_formula_mi

            if "formula_node_mi" in self.loss_names:
                formula_node_mi = node_e - formula_node_e
                loss_d["formula_node_mi"] = formula_node_mi

            # special probabilities
            null_formula_prob = th.exp(pred_null_formula_logprob)
            oos_prob = th.exp(pred_oos_logprobs)
            loss_d["null_formula_prob"] = null_formula_prob
            loss_d["oos_prob"] = oos_prob

            if "h_mean" in self.loss_names or "h_e" in self.loss_names or "h_ne" in self.loss_names:
                # hydrogens
                h_mean = scatter_reduce(
                    th.exp(pred_h_logprobs) * pred_h_counts,
                    pred_h_batch_idxs,
                    reduce="sum",
                )
                h_e, h_ne = sparse_entropy_fn(pred_h_logprobs, pred_h_batch_idxs)
                loss_d["h_mean"] = h_mean  # The mean predicted hydrogen count.
                # Shannon entropy of the predicted hydrogen count distribution
                loss_d["h_e"] = h_e
                loss_d["h_ne"] = h_ne  # The normalized entropy

            if not self.hparams.skip_edge_loss:
                if (
                    "edge_h_range_loss" in self.loss_names
                    or "edge_h_transfer_loss" in self.loss_names
                ):
                    r_el, t_el = edge_loss(
                        pred_edge_logprobs,
                        pred_edge_h_diffs,
                        pred_edge_h_range_masks,
                        pred_edge_h_logprobs,
                        pred_edge_batch_idxs,
                    )
                    loss = (
                        loss
                        + self.hparams.edge_h_range_loss_weight * r_el
                        + self.hparams.edge_h_transfer_loss_weight * t_el
                    )
                    loss_d["edge_h_range_loss"] = r_el
                    loss_d["edge_h_transfer_loss"] = t_el
                if "edge_e" in self.loss_names or "edge_ne" in self.loss_names:
                    edge_e, edge_ne = sparse_entropy_fn(pred_edge_logprobs, pred_edge_batch_idxs)
                    loss = (
                        loss
                        + self.hparams.edge_entropy_weight * edge_e
                        + self.hparams.edge_normalized_entropy_weight * edge_ne
                    )
                    loss_d["edge_e"] = edge_e
                    loss_d["edge_ne"] = edge_ne
            else:
                assert self.hparams.edge_h_range_loss_weight == 0.0
                assert self.hparams.edge_h_transfer_loss_weight == 0.0
                assert self.hparams.edge_entropy_weight == 0.0
                assert self.hparams.edge_normalized_entropy_weight == 0.0

            if self.hparams.nb_iso:
                if "nb_node_e" in self.loss_names or "nb_node_ne" in self.loss_names:
                    nb_node_e, nb_node_ne = sparse_entropy_fn(
                        pred_nb_node_logprobs, pred_nb_node_batch_idxs
                    )
                    loss = (
                        loss
                        + self.hparams.nb_node_entropy_weight * nb_node_e
                        + self.hparams.nb_node_normalized_entropy_weight * nb_node_ne
                    )
                    loss_d["nb_node_e"] = nb_node_e
                    loss_d["nb_node_ne"] = nb_node_ne
                if (
                    "nb_node_formula_e" in self.loss_names
                    or "nb_node_formula_ne" in self.loss_names
                ):
                    nb_node_formula_e, nb_node_formula_ne = sparse_conditional_entropy_fn(
                        pred_nb_node_logprobs,
                        pred_nb_node_batch_idxs,
                        pred_nb_node_formula_logprobs,
                        pred_nb_joint_node_idxs,
                        pred_nb_joint_batch_idxs,
                    )
                    loss = (
                        loss
                        + self.hparams.nb_node_formula_entropy_weight * nb_node_formula_e
                        + self.hparams.nb_node_formula_normalized_entropy_weight
                        * nb_node_formula_ne
                    )
                    loss_d["nb_node_formula_e"] = nb_node_formula_e
                    loss_d["nb_node_formula_ne"] = nb_node_formula_ne
                if (
                    "nb_formula_node_e" in self.loss_names
                    or "nb_formula_node_ne" in self.loss_names
                ):
                    nb_formula_node_e, nb_formula_node_ne = sparse_conditional_entropy_fn(
                        pred_formula_logprobs,
                        pred_formula_batch_idxs,
                        pred_nb_formula_node_logprobs,
                        pred_nb_joint_formula_idxs,
                        pred_nb_joint_batch_idxs,
                    )
                    assert not th.any(nb_formula_node_ne < 0.0), nb_formula_node_ne
                    assert not th.any(nb_formula_node_ne > 1.0), nb_formula_node_ne
                    loss = (
                        loss
                        + self.hparams.nb_formula_node_entropy_weight * nb_formula_node_e
                        + self.hparams.nb_formula_node_normalized_entropy_weight
                        * nb_formula_node_ne
                    )
                    loss_d["nb_formula_node_e"] = nb_formula_node_e
                    loss_d["nb_formula_node_ne"] = nb_formula_node_ne
                if "nb_node_node_e" in self.loss_names or "nb_node_node_ne" in self.loss_names:
                    nb_node_node_e, nb_node_node_ne = sparse_conditional_entropy_fn(
                        pred_nb_node_logprobs,
                        pred_nb_node_batch_idxs,
                        pred_nb_node_node_logprobs,
                        pred_nb_node_node_node_idxs,
                        pred_nb_node_node_batch_idxs,
                    )
                    loss = (
                        loss
                        + self.hparams.nb_node_node_entropy_weight * nb_node_node_e
                        + self.hparams.nb_node_node_normalized_entropy_weight * nb_node_node_ne
                    )
                    loss_d["nb_node_node_e"] = nb_node_node_e
                    loss_d["nb_node_node_ne"] = nb_node_node_ne
                if "nb_joint_e" in self.loss_names or "nb_joint_ne" in self.loss_names:
                    nb_joint_e, nb_joint_ne = sparse_entropy_fn(
                        pred_nb_joint_logprobs, pred_nb_joint_batch_idxs
                    )
                    # assert th.all(th.isclose(nb_joint_e, nb_node_formula_e+nb_node_e,rtol=0.,atol=0.001))
                    # assert th.all(th.isclose(nb_joint_e, nb_formula_node_e+formula_e,rtol=0.,atol=0.001))
                    loss = (
                        loss
                        + self.hparams.nb_joint_entropy_weight * nb_joint_e
                        + self.hparams.nb_joint_normalized_entropy_weight * nb_joint_ne
                    )
                    loss_d["nb_joint_e"] = nb_joint_e
                    loss_d["nb_joint_ne"] = nb_joint_ne
                if "nb_node_formula_mi" in self.loss_names:
                    nb_node_formula_mi = formula_e - nb_node_formula_e
                    loss_d["nb_node_formula_mi"] = nb_node_formula_mi
                if "nb_formula_node_mi" in self.loss_names:
                    nb_formula_node_mi = nb_node_e - nb_formula_node_e
                    loss_d["nb_formula_node_mi"] = nb_formula_node_mi
                if "nb_node_node_mi" in self.loss_names:
                    nb_node_node_mi = nb_node_e - nb_node_node_e
                    loss_d["nb_node_node_mi"] = nb_node_node_mi
            else:
                assert self.hparams.nb_node_entropy_weight == 0.0
                assert self.hparams.nb_node_normalized_entropy_weight == 0.0
                assert self.hparams.nb_node_formula_entropy_weight == 0.0
                assert self.hparams.nb_node_formula_normalized_entropy_weight == 0.0
                assert self.hparams.nb_formula_node_entropy_weight == 0.0
                assert self.hparams.nb_formula_node_normalized_entropy_weight == 0.0

            # Pairwise similarity MSE loss
            if f"pairwise_cossine_{self.hparams.mz_bin_res}" in self.loss_names:
                batch_size = kwargs.get("batch_size", th.max(true_batch_idxs) + 1)
                with th.no_grad():
                    true_sim = get_pairwise_cossim(
                        true_mzs,
                        true_logprobs,
                        true_batch_idxs,
                        batch_size,
                        self.hparams.mz_max,
                        self.hparams.mz_bin_res,
                        chunk_size=self.hparams.loss_batch_size,
                    )
                pred_sim = get_pairwise_cossim(
                    pred_mzs,
                    pred_logprobs,
                    pred_batch_idxs,
                    batch_size,
                    self.hparams.mz_max,
                    self.hparams.mz_bin_res,
                    chunk_size=self.hparams.loss_batch_size,
                )

                pairwise_sim_mse = F.mse_loss(pred_sim, true_sim)
                loss = loss + self.hparams.pairwise_cosine_loss_weight * pairwise_sim_mse
                # expand batch-level scalar to per-sample tensor for metric tracking
                loss_d[f"pairwise_cossine_{self.hparams.mz_bin_res}"] = pairwise_sim_mse.expand(
                    batch_size
                )

            # Pairwise JSS MSE loss
            if f"pairwise_jss_{self.hparams.mz_bin_res}" in self.loss_names:
                batch_size = kwargs.get("batch_size", th.max(true_batch_idxs) + 1)
                with th.no_grad():
                    true_sim = get_pairwise_jss_sim(
                        true_mzs,
                        true_logprobs,
                        true_batch_idxs,
                        batch_size,
                        self.hparams.mz_max,
                        self.hparams.mz_bin_res,
                        chunk_size=self.hparams.loss_batch_size,
                    )
                pred_sim = get_pairwise_jss_sim(
                    pred_mzs,
                    pred_logprobs,
                    pred_batch_idxs,
                    batch_size,
                    self.hparams.mz_max,
                    self.hparams.mz_bin_res,
                    chunk_size=self.hparams.loss_batch_size,
                )

                pairwise_jss_mse = F.mse_loss(pred_sim, true_sim)
                loss = loss + self.hparams.pairwise_jss_loss_weight * pairwise_jss_mse
                # expand batch-level scalar to per-sample tensor for metric tracking
                loss_d[f"pairwise_jss_{self.hparams.mz_bin_res}"] = pairwise_jss_mse.expand(
                    batch_size
                )

            # Pairwise cross-entropy MSE loss
            if f"pairwise_cross_entropy_{self.hparams.mz_bin_res}" in self.loss_names:
                batch_size = kwargs.get("batch_size", th.max(true_batch_idxs) + 1)
                with th.no_grad():
                    true_sim = get_pairwise_cross_entropy(
                        true_mzs,
                        true_logprobs,
                        true_batch_idxs,
                        batch_size,
                        self.hparams.mz_max,
                        self.hparams.mz_bin_res,
                        chunk_size=self.hparams.loss_batch_size,
                    )
                pred_sim = get_pairwise_cross_entropy(
                    pred_mzs,
                    pred_logprobs,
                    pred_batch_idxs,
                    batch_size,
                    self.hparams.mz_max,
                    self.hparams.mz_bin_res,
                    chunk_size=self.hparams.loss_batch_size,
                )

                pairwise_ce_mse = F.mse_loss(pred_sim, true_sim)
                loss = loss + self.hparams.pairwise_cross_entropy_loss_weight * pairwise_ce_mse
                # expand batch-level scalar to per-sample tensor for metric tracking
                loss_d[f"pairwise_cross_entropy_{self.hparams.mz_bin_res}"] = (
                    pairwise_ce_mse
                    * th.ones(
                        [batch_size], device=pairwise_ce_mse.device, dtype=pairwise_ce_mse.dtype
                    )
                )
            # finally, update the loss
            loss_d["loss"] = loss

            return loss_d

        self.loss_fn = _loss_fn

    def on_train_epoch_end(self):
        if (
            not self.hparams.automatic_optimization
            and self.hparams.dynamic_batch_sampler
            and self._cur_batch_size > 0
        ):
            # this is used for manul opt when lr_schedulers is not available from torch lightning
            self._manual_opt()

        super().on_train_epoch_end()

    def _manual_opt(self):
        opt = self.optimizers()

        if self.hparams.dynamic_batch_sampler:
            grads = [p.grad for pg in opt.param_groups for p in pg["params"] if p.grad is not None]
            if not grads:
                return
            scale = self._max_batch_size / self._cur_batch_weight
            th._foreach_mul_(grads, scale)

        self.on_before_optimizer_step(opt)
        self.clip_gradients(
            opt,
            gradient_clip_val=self.hparams.gradient_clip_val,
            gradient_clip_algorithm=self.hparams.gradient_clip_algorithm,
        )
        opt.step()
        # on_after_optimizer_step is not called in manual opt mode, so reset here
        self._cur_batch_size = 0
        self._cur_batch_weight = 0.0

        if self.hparams.lr_schedule:
            sch = self.lr_schedulers()
            if sch is not None:
                sch.step()

        opt.zero_grad()

    def training_step(self, batch, batch_idx):
        """training loop for FragGNNPL

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """

        if self.hparams.automatic_optimization:
            batch_results = self._common_step(batch, split="train")
            mean_loss = batch_results["mean_loss"]
            if self.hparams.debug_zero_loss:
                mean_loss = 0.0 * mean_loss
                self.print(mean_loss)
            self._cur_batch_size += self.hparams.train_batch_size
            self._update_results(batch_results, "train")
            return mean_loss

        elif self.hparams.dynamic_batch_sampler:
            batch_size = batch["batch_size"].item()
            assert batch_size > 0, batch_size
            batch_results = self._common_step(batch, split="train")
            total_loss = batch_results["total_loss"]
            total_weight = batch_results["total_weight"]
            mean_loss = total_loss / self._max_batch_size
            if not isinstance(mean_loss, th.Tensor):
                raise TypeError(f"Expected mean_loss to be a torch.Tensor, got {type(mean_loss)}")
            if mean_loss.ndim != 0:
                raise RuntimeError(
                    f"Expected scalar mean_loss for backward(), got shape={tuple(mean_loss.shape)}"
                )
            if not mean_loss.requires_grad:
                raise RuntimeError(
                    "mean_loss does not require grad (detached). "
                    f"total_loss.requires_grad={getattr(total_loss, 'requires_grad', None)}"
                )
            if not th.isfinite(mean_loss):
                raise RuntimeError(
                    f"Non-finite mean_loss detected: {mean_loss.detach().float().cpu().item()} "
                    f"(batch_idx={batch_idx}, batch_size={batch_size}, total_weight={float(total_weight) if not isinstance(total_weight, th.Tensor) else total_weight.detach().float().cpu().item()})"
                )
            if self.hparams.debug_zero_loss:
                mean_loss = 0.0 * mean_loss
                self.print(mean_loss)
            self.manual_backward(mean_loss)
            # accumulate gradients of N samples
            self._cur_batch_size += batch_size
            self._cur_batch_weight += total_weight
            if self._cur_batch_size >= self._max_batch_size:
                self._manual_opt()
            self._update_results(batch_results, "train")
            return mean_loss

        else:
            batch_results = self._common_step(batch, split="train")
            mean_loss = batch_results["mean_loss"] / self.hparams.accumulate_grad_batches
            total_weight = batch_results["total_weight"]
            if not isinstance(mean_loss, th.Tensor):
                raise TypeError(f"Expected mean_loss to be a torch.Tensor, got {type(mean_loss)}")
            if mean_loss.ndim != 0:
                raise RuntimeError(
                    f"Expected scalar mean_loss for backward(), got shape={tuple(mean_loss.shape)}"
                )
            if not mean_loss.requires_grad:
                raise RuntimeError("mean_loss does not require grad (detached).")
            if not th.isfinite(mean_loss):
                raise RuntimeError(
                    f"Non-finite mean_loss detected: {mean_loss.detach().float().cpu().item()} "
                    f"(batch_idx={batch_idx}, total_weight={float(total_weight) if not isinstance(total_weight, th.Tensor) else total_weight.detach().float().cpu().item()})"
                )
            if self.hparams.debug_zero_loss:
                mean_loss = 0.0 * mean_loss
                self.print(mean_loss)
            self.manual_backward(mean_loss)
            # accumulate gradients of N samples
            self._cur_batch_size += self.hparams.train_batch_size
            self._cur_batch_weight += total_weight
            if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0:
                self._manual_opt()
            self._update_results(batch_results, "train")
            return mean_loss

    def configure_optimizers(self):
        """
        Configure optimizers with different learning rates for pre-trained and other parameters.
        """
        # Get LR ratio from finetune config
        if self.hparams.finetune:
            lr_ratio = self.hparams.finetune["pretrained_weights_lr_ratio"]
        else:
            lr_ratio = self.hparams.finetune_weights_lr_ratio

        # Separate parameters into pre-trained and others
        pretrained_params = [
            p for n, p in self.model.named_parameters() if n in self.pretrained_param_names
        ]
        other_params = [
            p for n, p in self.model.named_parameters() if n not in self.pretrained_param_names
        ]
        optimizer_cls = get_optimizer_class(self.hparams.optimizer)

        # Define optimizer with parameter groups
        optimizer = optimizer_cls(
            [
                {"params": pretrained_params, "lr": self.hparams.lr * lr_ratio},
                {"params": other_params, "lr": self.hparams.lr},
            ],
            weight_decay=self.hparams.weight_decay,
        )

        ret = {
            "optimizer": optimizer,
        }

        # Optionally add a learning rate scheduler
        if self.hparams.lr_schedule:
            scheduler = build_lr_scheduler(
                optimizer=optimizer,
                decay_rate=self.hparams.lr_decay_rate,
                warmup_steps=self.hparams.lr_warmup_steps,
                decay_steps=self.hparams.lr_decay_steps,
                schedule_type=self.hparams.lr_schedule_type,
                total_steps=self.hparams.lr_total_steps,
                min_lr_ratio=self.hparams.lr_min_ratio,
            )
            ret["lr_scheduler"] = {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "step",
            }

        return ret

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        opt = self.optimizers()
        self.on_after_optimizer_step(opt)
