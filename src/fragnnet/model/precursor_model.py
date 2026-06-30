import torch as th
import torch.nn as nn


class PrecursorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_params = nn.Parameter(th.zeros((1,), dtype=th.float32))

    def forward(self, spec_prec_mz: th.Tensor, **kwargs):
        pred_mzs = spec_prec_mz
        pred_logprobs = 0.0 * self.dummy_params + th.zeros_like(pred_mzs)
        pred_batch_idxs = th.arange(pred_mzs.shape[0], device=pred_mzs.device)

        out_d = {
            "pred_mzs": pred_mzs,
            "pred_logprobs": pred_logprobs,
            "pred_batch_idxs": pred_batch_idxs,
        }
        return out_d
