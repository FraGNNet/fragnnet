"""`COLUMN_SPECS` is the single source of truth for the inference export schema.

The DataFrame header (`get_tabular_columns`), the per-spectrum row
(`process_batch_to_tabular_row`), the HDF5 ragged-family grouping (`run_inference_hdf5`)
and the `--output_subset` vocabulary all used to be written out independently, in four
places, coupled only by hand-matched ordering. These tests pin that they now agree by
construction, and pin the column order itself so a reorder cannot slip through unnoticed.
"""

import numpy as np
import pytest
import torch as th
import torch_geometric as pyg

from fragnnet.pl_model import FraGNNetPL
from fragnnet.utils import script_utils
from fragnnet.utils.script_utils import (
    COLUMN_SPECS,
    OUTPUT_SUBSET_OPTIONS,
    RAGGED_FAMILY_PREFIX,
    selected_column_specs,
)

# The emitted order, ahead of the annotation and eval-score columns. Frozen here because
# some readers index positionally; changing it is a breaking change to saved artifacts.
BASE_COLUMNS = [
    "spec_id",
    "mol_id",
    "pred_mzs",
    "pred_ints",
    "pred_oos_prob",
    "frag_formula_peak_idxs",
    "frag_formula_peak_mzs",
    "pred_formula_formula_idxs",
    "pred_formula_probs",
    "pred_joint_formula_idxs",
    "pred_formula_node_probs",
    "pred_joint_node_idxs",
    "pred_node_formula_probs",
    "pred_node_probs",
    "pred_node_node_idxs",
    "index_rebased",
]

HPARAMS = {
    "auxiliary_scores": [],
    "eval_mz_bin_res": [],
    "eval_hun_sqrt": False,
    "eval_bin_sqrt": False,
}


class TestTable:
    def test_column_order_is_frozen(self):
        assert [c.name for c in selected_column_specs(False, False)] == BASE_COLUMNS

    def test_annotation_columns_append_in_order(self):
        assert [c.name for c in selected_column_specs(True, True)] == BASE_COLUMNS + [
            "pred_formula_strs",
            "pred_formula_adducts",
            "pred_node_ccs",
        ]

    @pytest.mark.parametrize(
        ("formula_str", "cc_hash", "extra"),
        [
            (False, False, []),
            # `pred_formula_adducts` rides on EITHER annotation flag: it is what turns a
            # neutral formula string into the ion, and what tells a cc bitmask its adduct.
            (True, False, ["pred_formula_strs", "pred_formula_adducts"]),
            (False, True, ["pred_formula_adducts", "pred_node_ccs"]),
            (True, True, ["pred_formula_strs", "pred_formula_adducts", "pred_node_ccs"]),
        ],
    )
    def test_each_flag_combination(self, formula_str, cc_hash, extra):
        names = [c.name for c in selected_column_specs(formula_str, cc_hash)]
        assert names == BASE_COLUMNS + extra

    def test_names_are_unique(self):
        names = [c.name for c in COLUMN_SPECS]
        assert len(names) == len(set(names))

    def test_every_family_is_known(self):
        for c in COLUMN_SPECS:
            assert c.family == "" or c.family in RAGGED_FAMILY_PREFIX, c.name

    def test_tokens_are_column_names_and_unique(self):
        names = {c.name for c in COLUMN_SPECS}
        assert len(OUTPUT_SUBSET_OPTIONS) == len(set(OUTPUT_SUBSET_OPTIONS))
        assert set(OUTPUT_SUBSET_OPTIONS) <= names

    def test_a_column_is_never_both_token_and_flag_gated(self):
        for c in COLUMN_SPECS:
            assert not (c.token and c.flag), c.name

    def test_ragged_families_partition_the_ragged_columns(self):
        """Each ragged column belongs to exactly one family, and no family is empty."""
        seen = []
        for family in RAGGED_FAMILY_PREFIX:
            members = [c.name for c in COLUMN_SPECS if c.family == family]
            assert members, f"family {family!r} has no columns"
            seen.extend(members)
        assert sorted(seen) == sorted(c.name for c in COLUMN_SPECS if c.family)

    def test_header_matches_the_table(self):
        columns = script_utils.get_tabular_columns(
            HPARAMS, return_formula_str=True, return_cc_hash=True
        )
        assert columns == [c.name for c in selected_column_specs(True, True)]

    def test_output_subset_does_not_narrow_the_header(self):
        """Documented contract: out-of-subset columns are emitted as NaN, not dropped."""
        full = script_utils.get_tabular_columns(HPARAMS, output_subset=None)
        subset = script_utils.get_tabular_columns(
            HPARAMS, output_subset={"pred_mzs", "pred_ints"}
        )
        assert subset == full


class _StubInner:
    """Just the two attributes the exporter reads to place the CRF/CMF formula split.

    Canonical definition: `tests/test_inference_annotation_without_probs.py` imports this
    one rather than keeping a second copy, so a change to what the exporter reads off
    `model.model` cannot leave one suite passing against a stale stub shape.
    """

    def __init__(
        self,
        frag_node_feats: tuple[str, ...] = ("cc", "h_formulae_idx"),
        num_hs: int = 4,
    ):
        self.frag_node_feats = frag_node_feats
        self.num_hs = num_hs


class _StubFraGNNet(FraGNNetPL):
    """FraGNNetPL-typed stub replaying one canned model output, skipping __init__."""

    def __init__(self, batch_output: dict):
        self.training = False
        self._batch_output = batch_output
        self.model = _StubInner()

    @property
    def hparams(self):
        return HPARAMS

    def eval(self):
        return self

    def inference_step(self, batch_input, *, metrics=None):  # noqa: ARG002
        return self._batch_output


def _log(vals):
    return th.log(th.tensor(vals, dtype=th.float32))


@pytest.fixture
def one_spectrum():
    """One spectrum: 3 peaks over 4 formulas (index 0 is NULL) and 3 DAG nodes."""
    batch_result = {
        "pred_mzs": th.tensor([100.0, 200.0, 300.0]),
        "pred_ints": th.tensor([0.5, 0.3, 0.2]),
        "pred_batch_idxs": th.tensor([0, 0, 0]),
        "pred_formula_logprobs": _log([0.1, 0.5, 0.3, 0.1]),
        "pred_formula_batch_idxs": th.tensor([0, 0, 0, 0]),
        "pred_formula_formula_idxs": th.arange(4),
        "pred_node_logprobs": _log([0.2, 0.5, 0.3]),
        "pred_node_batch_idxs": th.tensor([0, 0, 0]),
        "pred_node_node_idxs": th.arange(3),
        "pred_joint_batch_idxs": th.tensor([0, 0, 0, 0]),
        "pred_joint_formula_idxs": th.tensor([1, 1, 2, 3]),
        "pred_joint_node_idxs": th.tensor([0, 1, 1, 2]),
        "pred_formula_node_logprobs": _log([0.6, 0.4, 1.0, 1.0]),
        "pred_node_formula_logprobs": _log([1.0, 0.5, 0.5, 1.0]),
    }
    frag_data = pyg.data.Data(
        x=th.tensor([[0.0, 11.0], [1.0, 22.0], [2.0, 33.0]]),
        node_feat_idxs=th.tensor([[0, 1, 2]]),
    )
    batch_input = {
        "spec_unique_id": th.tensor([7]),
        "mol_id": ["m0"],
        "batch_size": th.tensor(1),
        "frag_formula_cumsizes": th.tensor([0, 4]),
        "frag_num_nodes": th.tensor([0, 3]),
        "spec_prec_type_str": np.array(["[M+H]+"]),
        "frag_formula_str": np.array(["", "CH4", "C2H6", "C3H8"]),
        "frag_formula_peak_idxs": th.tensor([1, 2, 3]),
        "frag_formula_peak_mzs": th.tensor([100.0, 200.0, 300.0]),
        "frag_pyg": pyg.data.Batch.from_data_list([frag_data]),
    }
    return batch_input, batch_result


class TestRowMatchesHeader:
    """The row builder and the header must stay in step for every flag/subset combination."""

    @staticmethod
    def _emit(one_spectrum, output_set, formula_str, cc_hash, device=th.device("cpu")):
        batch_input, batch_result = one_spectrum
        model = _StubFraGNNet(batch_result)
        batch_idx_vars, node_idx_vars, formula_idx_vars, _ = script_utils.get_idx_vars(
            output_set,
            model,
            nb_iso=False,
            return_formula_str=formula_str,
            return_cc_hash=cc_hash,
        )
        columns = script_utils.get_tabular_columns(
            HPARAMS,
            return_formula_str=formula_str,
            return_cc_hash=cc_hash,
            output_subset=output_set,
        )
        batch_rows = script_utils.process_batch_to_tabular_row(
            batch_input=batch_input,
            model=model,
            device=device,
            evaluate=False,
            eval_score_types=[],
            BATCH_IDX_VARS=batch_idx_vars,
            NODE_IDX_VARS=node_idx_vars,
            FORMULA_IDX_VARS=formula_idx_vars,
            min_peak_int=0.0,
            rebase_indices=True,
            return_formula_str=formula_str,
            return_cc_hash=cc_hash,
            output_set=output_set,
        )
        return columns, batch_rows

    @pytest.mark.parametrize("formula_str", [False, True])
    @pytest.mark.parametrize("cc_hash", [False, True])
    @pytest.mark.parametrize(
        "output_set", [None, {"pred_mzs", "pred_ints"}, {"pred_mzs", "pred_node_probs"}]
    )
    def test_row_width_equals_header_width(self, one_spectrum, output_set, formula_str, cc_hash):
        columns, batch_rows = self._emit(one_spectrum, output_set, formula_str, cc_hash)
        assert len(batch_rows) == len(columns)

    def test_values_land_under_the_right_names(self, one_spectrum):
        """A misordered row would still be the right width, so check the values too."""
        columns, batch_rows = self._emit(one_spectrum, None, True, True)
        row = {k: v[0] for k, v in zip(columns, batch_rows)}
        assert row["spec_id"] == 7
        assert row["mol_id"] == "m0"
        assert row["index_rebased"] is True
        assert th.allclose(row["pred_mzs"], th.tensor([100.0, 200.0, 300.0]))
        # min_peak_int=0 filters nothing, so the NULL formula at index 0 survives too
        assert row["pred_formula_strs"] == ["", "CH4", "C2H6", "C3H8"]
        # cc is node-feature column 1, and the column is 2-D (one hash row per node)
        assert th.allclose(row["pred_node_ccs"], th.tensor([[11.0], [22.0], [33.0]]))
        assert len(row["pred_joint_formula_idxs"]) == 4

    def test_unrequested_columns_are_nan_not_missing(self, one_spectrum):
        """Peaks-only still emits every column; the unrequested ones hold NaN."""
        columns, batch_rows = self._emit(one_spectrum, {"pred_mzs", "pred_ints"}, False, False)
        row = {k: v[0] for k, v in zip(columns, batch_rows)}
        assert set(columns) == set(BASE_COLUMNS)
        assert isinstance(row["pred_node_probs"], float) and np.isnan(row["pred_node_probs"])
        # the exporter rescales each spectrum so its base peak is 1000
        assert th.allclose(row["pred_ints"], th.tensor([1000.0, 600.0, 400.0]))


class TestRowLeavesTheDevice:
    """Saved predictions must load on a host with no GPU.

    Only some of the arrays the row builder assembles were explicitly moved to the CPU
    (`pred_mzs`, `pred_ints`, `pred_node_ccs`); the index and probability arrays came
    straight off the model. A GPU run therefore pickled a mix of CPU and CUDA tensors,
    and unpickling it anywhere without that CUDA device raised.
    """

    @staticmethod
    def _to_device(value, device):
        if th.is_tensor(value):
            return value.to(device)
        if isinstance(value, pyg.data.Batch):
            return value.to(device)
        return value

    @pytest.mark.skipif(not th.cuda.is_available(), reason="needs a CUDA device")
    def test_every_emitted_tensor_is_on_the_cpu(self, one_spectrum):
        batch_input, batch_result = one_spectrum
        device = th.device("cuda")
        batch_input = {k: self._to_device(v, device) for k, v in batch_input.items()}
        batch_result = {k: self._to_device(v, device) for k, v in batch_result.items()}
        columns, batch_rows = TestRowMatchesHeader._emit(
            (batch_input, batch_result), None, True, True, device=device
        )
        row = {k: v[0] for k, v in zip(columns, batch_rows)}
        on_gpu = [
            name
            for name, value in row.items()
            if th.is_tensor(value) and value.device.type != "cpu"
        ]
        assert on_gpu == []
        # and the values still survive the move
        assert th.allclose(row["pred_mzs"], th.tensor([100.0, 200.0, 300.0]))
        assert th.equal(row["frag_formula_peak_idxs"], th.tensor([1, 2, 3]))
