import torch as th
import torch._dynamo as th_dynamo

try:
    from torch._dynamo.utils import CompileProfiler

    profiler_available = True
except ImportError:
    profiler_available = False

from fragnnet.model import NeimsModel
from fragnnet.pl_model import BinnedPL


class NeimsPL(BinnedPL):
    """PyTorch Lightning wrapper for NEIMS model with binned spectrum loss."""

    def _setup_model(self) -> None:
        """Initialize the NeimsModel using hyperparameters."""
        self.model = NeimsModel(
            mol_fingerprint_morgan=self.hparams.mol_params["fingerprint_morgan"],
            mol_fingerprint_rdkit=self.hparams.mol_params["fingerprint_rdkit"],
            mol_fingerprint_maccs=self.hparams.mol_params["fingerprint_maccs"],
            morgan_radius=self.hparams.mol_params["morgan_radius"],
            morgan_nbits=self.hparams.mol_params["morgan_nbits"],
            rdkit_nbits=self.hparams.mol_params["rdkit_nbits"],
            mlp_hidden_size=self.hparams.mlp_hidden_size,
            mlp_dropout=self.hparams.mlp_dropout,
            mlp_num_layers=self.hparams.mlp_num_layers,
            mlp_use_residuals=self.hparams.mlp_use_residuals,
            neims_bottleneck_factor=self.hparams.neims_bottleneck_factor,
            mz_max=self.hparams.mz_max,
            mz_bin_res=self.hparams.mz_bin_res,
            ff_prec_mz_offset=self.hparams.ff_prec_mz_offset,
            ff_bidirectional=self.hparams.ff_bidirectional,
            ff_output_map_size=self.hparams.ff_output_map_size,
            ff_output_activation=self.hparams.ff_output_activation,
            int_embedder=self.hparams.int_embedder,
            ce_insert_type=self.hparams.ce_insert_type,
            ce_insert_location=self.hparams.ce_insert_location,
            ce_insert_merge=self.hparams.ce_insert_merge,
            ce_insert_size=self.hparams.ce_insert_size,
            use_nce=self.hparams.spec_params["nce"],
            nce_mean=self.hparams.nce_mean,
            nce_std=self.hparams.nce_std,
            nce_max=self.hparams.nce_max,
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
