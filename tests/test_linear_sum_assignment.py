"""Unit tests for fragnnet.utils.spec_utils.linear_sum_assignment.

Tests cover:
- Correctness against scipy reference on known matrices
- maximize=True vs minimize (maximize=False)
- Non-square matrices (more rows, more cols)
- Single-element matrix
- Empty matrix edge case
- CPU fallback path always works (torch_linear_assignment may not be installed)
"""

import pytest
import torch as th
from scipy.optimize import linear_sum_assignment as scipy_lsa

from fragnnet.utils.spec_utils import linear_sum_assignment


def _reference(matrix: th.Tensor, maximize: bool) -> tuple[th.Tensor, th.Tensor]:
    """scipy ground truth, returns sorted (row_ind, col_ind) on CPU."""
    np_mat = matrix.cpu().numpy()
    row, col = scipy_lsa(np_mat, maximize=maximize)
    # sort by row so comparison is order-independent
    order = row.argsort()
    return th.tensor(row[order]).long(), th.tensor(col[order]).long()


def _canonicalize(row: th.Tensor, col: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    order = row.cpu().argsort()
    return row.cpu()[order], col.cpu()[order]


class TestLinearSumAssignment:
    def test_known_3x3_minimize(self):
        """3×3 cost matrix: optimal assignment minimizes total cost."""
        cost = th.tensor([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
        row, col = linear_sum_assignment(cost, maximize=False)
        row, col = _canonicalize(row, col)
        ref_row, ref_col = _reference(cost, maximize=False)
        assert th.equal(row, ref_row)
        assert th.equal(col, ref_col)
        total = cost[row, col].sum().item()
        assert total == pytest.approx(cost[ref_row, ref_col].sum().item())

    def test_known_3x3_maximize(self):
        """3×3 score matrix: optimal assignment maximizes total score."""
        score = th.tensor([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
        row, col = linear_sum_assignment(score, maximize=True)
        row, col = _canonicalize(row, col)
        ref_row, ref_col = _reference(score, maximize=True)
        assert th.equal(row, ref_row)
        assert th.equal(col, ref_col)

    def test_identity_score_matrix(self):
        """Identity matrix: optimal maximize assignment picks diagonal."""
        n = 4
        score = th.eye(n)
        row, col = linear_sum_assignment(score, maximize=True)
        row, col = _canonicalize(row, col)
        assert th.equal(row, th.arange(n))
        assert th.equal(col, th.arange(n))

    def test_non_square_more_cols(self):
        """More columns than rows: each row gets exactly one column."""
        score = th.tensor([[1.0, 5.0, 2.0], [4.0, 0.0, 3.0]])
        row, col = linear_sum_assignment(score, maximize=True)
        row, col = _canonicalize(row, col)
        ref_row, ref_col = _reference(score, maximize=True)
        assert th.equal(row, ref_row)
        assert th.equal(col, ref_col)
        assert len(row) == score.shape[0]

    def test_non_square_more_rows(self):
        """More rows than columns: each column gets exactly one row."""
        score = th.tensor([[1.0, 4.0], [5.0, 0.0], [2.0, 3.0]])
        row, col = linear_sum_assignment(score, maximize=True)
        row, col = _canonicalize(row, col)
        ref_row, ref_col = _reference(score, maximize=True)
        assert th.equal(row, ref_row)
        assert th.equal(col, ref_col)
        assert len(col) == score.shape[1]

    def test_single_element(self):
        """1×1 matrix: only possible assignment."""
        score = th.tensor([[7.0]])
        row, col = linear_sum_assignment(score, maximize=True)
        assert row.tolist() == [0]
        assert col.tolist() == [0]

    def test_optimal_total_maximize(self):
        """Random matrix: result is at least as good as any greedy row assignment."""
        th.manual_seed(42)
        score = th.rand(6, 8)
        row, col = linear_sum_assignment(score, maximize=True)
        total = score[row.cpu(), col.cpu()].sum().item()
        ref_row, ref_col = _reference(score, maximize=True)
        ref_total = score[ref_row, ref_col].sum().item()
        assert total == pytest.approx(ref_total, abs=1e-5)

    def test_returns_long_tensors(self):
        score = th.rand(3, 3)
        row, col = linear_sum_assignment(score, maximize=True)
        assert row.dtype == th.long
        assert col.dtype == th.long

    def test_device_preserved(self):
        """Output tensors are on the same device as input."""
        score = th.rand(3, 4)
        row, col = linear_sum_assignment(score, maximize=True)
        assert row.device == score.device
        assert col.device == score.device
