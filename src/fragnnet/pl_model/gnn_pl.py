import torch as th
import torch._dynamo as th_dynamo

try:
    from torch._dynamo.utils import CompileProfiler

    profiler_available = True
except ImportError:
    profiler_available = False

from fragnnet.model import GNNModel
from fragnnet.pl_model import BinnedPL


class GNNPL(BinnedPL):
    """PyTorch Lightning wrapper for GNNModel with binned spectrum loss."""

    def _setup_model(self) -> None:
        """Initialize the GNNModel based on hyperparameters."""
        # TODO: initialize model based on params. finalize params
        self.model = GNNModel(
            mol_node_feats=self.hparams.mol_params["pyg_node_feats"],  # mol feats
            mol_edge_feats=self.hparams.mol_params["pyg_edge_feats"],
            mol_pe_embed_k=self.hparams.mol_params["pyg_pe_embed_k"],
            mol_hidden_size=self.hparams.mol_hidden_size,
            mol_num_layers=self.hparams.mol_num_layers,
            mol_gnn_type=self.hparams.mol_gnn_type,
            mol_normalization=self.hparams.mol_normalization,
            mol_dropout=self.hparams.mol_dropout,
            mol_pool_type=self.hparams.mol_pool_type,
            mlp_hidden_size=self.hparams.mlp_hidden_size,  # FFN
            mlp_dropout=self.hparams.mlp_dropout,
            mlp_num_layers=self.hparams.mlp_num_layers,
            mlp_use_residuals=self.hparams.mlp_use_residuals,
            mz_max=self.hparams.mz_max,
            mz_bin_res=self.hparams.mz_bin_res,
            ff_prec_mz_offset=self.hparams.ff_prec_mz_offset,
            ff_bidirectional=self.hparams.ff_bidirectional,
            ff_output_map_size=self.hparams.ff_output_map_size,
            ff_output_activation=self.hparams.ff_output_activation,
            int_embedder=self.hparams.int_embedder,  # cross entropy
            ce_insert_type=self.hparams.ce_insert_type,
            ce_insert_location=self.hparams.ce_insert_location,
            ce_insert_merge=self.hparams.ce_insert_merge,
            ce_insert_size=self.hparams.ce_insert_size,
            nce_mean=self.hparams.nce_mean,
            nce_std=self.hparams.nce_std,
            nce_max=self.hparams.nce_max,
            prec_insert_location=self.hparams.prec_insert_location,  # precursor
            prec_insert_size=self.hparams.prec_insert_size,
            prec_types=self.hparams.spec_params["prec_types"],
            inst_insert_location=self.hparams.inst_insert_location,  # instrument
            inst_insert_size=self.hparams.inst_insert_size,
            inst_types=self.hparams.spec_params["inst_types"],
            log_min=self.hparams.log_min,
        )

        self._check_ce_params()

        # compile
        if self.hparams.compile:
            th_dynamo.reset()
            if profiler_available:
                self.dynamo_prof = CompileProfiler()
                self.model = self.model.get_compile(backend=self.dynamo_prof, dynamic=True)
            else:
                self.model = th.compile(self.model, dynamic=True)
