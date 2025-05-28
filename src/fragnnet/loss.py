import numpy as np
import torch as th
import torch.nn.functional as F
from typing import Callable

from fragnnet.utils.misc_utils import scatter_reduce, EPS, safelog, LOG_ZERO, scatter_logsumexp, scatter_logl2normalize, TOLERANCE_MIN_MZ
from fragnnet.utils.spec_utils import calculate_match_mzs, scipy_linear_sum_assignment, round_aggregate_peaks, batched_bin_func, jss_helper

# LOG_TRUNC_FACTOR = {
# 	i: float(th.log(th.erf(th.tensor(i)/np.sqrt(2)))) for i in range(0,6)
# }

def mog_log_prob(
		samples: th.Tensor,
		means: th.Tensor,
		variances: th.Tensor,
		log_weights: th.Tensor,
		log_trunc_factors: th.Tensor) -> th.Tensor:
	""" Mixture of Gaussian
		We are modeling each peaks are mixture of guassian for given set of fragments
	Args:
		samples (_type_): b_true_mz,
		means (_type_): b_pred_mzs
		variances (_type_): b_variances
		log_weights (_type_): b_pred_logprobs
		log_trunc_factors (_type_): b_log_trunc_factors

	Returns:
		_type_: _description_
	"""
	
	samples = samples.reshape(-1,1)
	means = means.reshape(1,-1)
	variances = variances.reshape(1,-1)
	log_weights = log_weights.reshape(1,-1)
	log_trunc_factors = log_trunc_factors.reshape(1,-1)
	# [B,K]
	normal_log_probs = -0.5*(samples-means)**2/variances - 0.5*th.log(2*th.pi*variances)
	normal_log_probs = normal_log_probs - log_trunc_factors
	# [B]
	log_probs = th.logsumexp(normal_log_probs + log_weights, dim=1)
	return log_probs

def mog_ce_fn(
		b_true_mzs: th.Tensor,
		b_pred_mzs: th.Tensor,
		b_pred_logprobs: th.Tensor,
		relative: bool,
		tolerance: float,
		tolerance_min_mz: float,
		tolerance_multiple: float,
		gaussian_renormalize: bool,
		**kwargs) -> th.Tensor:

	if relative:
		b_stds = th.clamp(b_pred_mzs,min=tolerance_min_mz)*tolerance
	else:
		b_stds = tolerance*th.ones_like(b_pred_mzs)
	b_vars = b_stds**2
	if gaussian_renormalize:
		assert tolerance_multiple > 0, tolerance_multiple
		# log_trunc_factor = LOG_TRUNC_FACTOR.get(tolerance_multiple,1.)
		# b_log_trunc_factors = log_trunc_factor*th.ones_like(b_pred_mzs)
		trunc_factor = th.erf(th.tensor(tolerance_multiple,device=b_pred_mzs.device)/float(np.sqrt(2)))
		b_log_trunc_factors = th.log(trunc_factor*th.ones_like(b_pred_mzs))
	else:
		b_log_trunc_factors = th.zeros_like(b_pred_mzs)
	b_logprobs = mog_log_prob(
		b_true_mzs,
		b_pred_mzs,
		b_vars,
		b_pred_logprobs,
		b_log_trunc_factors
	)
	
	return b_logprobs

def pm_ce_fn(
		b_pred_mzs: th.Tensor,
		b_true_mzs: th.Tensor,
		b_pred_logprobs: th.Tensor,
		relative: bool,
		tolerance: float,
		tolerance_multiple: float,
		tolerance_min_mz: float,
		**kwargs) -> th.Tensor:

	b_match_mask = calculate_match_mzs(
		b_true_mzs,
		b_pred_mzs,
		tolerance=tolerance*tolerance_multiple,
		relative=relative,
		tolerance_min_mz=tolerance_min_mz
	)
	b_logprobs = th.logsumexp(
		b_pred_logprobs.reshape(1,-1)*b_match_mask.float() + LOG_ZERO(b_pred_logprobs.dtype)*(~b_match_mask).float(),
		dim=1)
	return b_logprobs

def sparse_cross_entropy_seq(
		true_mzs: th.Tensor,
		true_logprobs: th.Tensor,
		true_batch_idxs: th.Tensor,
		pred_mzs: th.Tensor,
		pred_logprobs: th.Tensor,
		pred_batch_idxs: th.Tensor,
		pred_oos_logprobs: th.Tensor,
		ce_fn: Callable,
		tolerance: float,
		relative: bool,
		tolerance_min_mz: float,
		tolerance_multiple: int,
		gaussian_renormalize: bool) -> th.Tensor:
	"""compute sparse cross entropy in squence

	Args:
		true_mzs (_type_): groundtruth mzs
		true_logprobs (_type_): groundtruth log prob of each mz aka intensities
		true_batch_idxs (_type_): spec idx signed to each groundtruth
		pred_mzs (_type_): predicted mzs
		pred_logprobs (_type_): predicted log prob of each mz aka intensities
		pred_batch_idxs (_type_): pec idx signed to each prediction
		ce_fn (_type_, optional): _description_. Defaults to mog_log_prob.
		tolerance (_type_, optional): _description_. Defaults to 1e-3.
		relative (bool, optional): _description_. Defaults to False.
		oos_tolerance_multiple (int, optional): _description_.
		gaussian_renormalize (bool, optional): _description_.
		pm_tolerance_multiple (int, optional): _description_.
		
	Returns:
		_type_: _description_
	"""

	# print(pred_logprobs.shape[0]*true_logprobs.shape[0]*4 / 1e6)
	mask_value = LOG_ZERO(true_logprobs.dtype)
	batch_size = th.max(true_batch_idxs)+1
	# ces = th.zeros([batch_size],dtype=true_logprobs.dtype,device=true_logprobs.device)
	ios_ces = th.zeros([batch_size], dtype=true_logprobs.dtype,device=true_logprobs.device)
	oos_ces = th.zeros([batch_size], dtype=true_logprobs.dtype,device=true_logprobs.device)
	true_oos_logprobs = th.zeros([batch_size],dtype=true_logprobs.dtype,device=true_logprobs.device)
	true_oos_entropies = th.zeros_like(true_oos_logprobs)
	for batch_idx in th.arange(batch_size):
		# get groundtruth belongs to this batch idx
		b_true_mask = true_batch_idxs==batch_idx
		# get predictions belongs to this batch idx
		b_pred_mask = pred_batch_idxs==batch_idx
		# get mzs and logprobs
		b_true_logprobs = true_logprobs[b_true_mask]
		b_true_mzs = true_mzs[b_true_mask]
		b_pred_logprobs = pred_logprobs[b_pred_mask]
		b_pred_mzs = pred_mzs[b_pred_mask]
		# get matches
		b_match_mask = calculate_match_mzs(
			b_true_mzs,
			b_pred_mzs,
			tolerance=tolerance_multiple*tolerance,
			relative=relative,
			tolerance_min_mz=tolerance_min_mz
		)
		if tolerance_multiple == -1:
			b_true_match_mask = th.ones_like(b_true_mzs,dtype=th.bool)
		else:
			assert tolerance_multiple > 0, tolerance_multiple	
			b_true_match_mask = th.any(b_match_mask,dim=1)
		if not th.any(b_true_match_mask):
			print("> Warning: Everything is OOS!")
		# calculate ios logprobs
		b_ios_logprobs = ce_fn(
			b_true_mzs=b_true_mzs,
			b_pred_mzs=b_pred_mzs,
			b_pred_logprobs=b_pred_logprobs,
			relative=relative,
			tolerance=tolerance,
			tolerance_min_mz=tolerance_min_mz,
			tolerance_multiple=tolerance_multiple,
			gaussian_renormalize=gaussian_renormalize)
		b_ios_ce = th.sum(b_true_match_mask.float() * -th.exp(b_true_logprobs) * b_ios_logprobs)
		# calculate oos logprobs
		b_true_oos_logprobs = th.logsumexp( mask_value*b_true_match_mask.float() + b_true_logprobs*(~b_true_match_mask).float(), dim=0)
		b_true_oos_entropy = - th.sum( th.exp(b_true_logprobs[~b_true_match_mask]-b_true_oos_logprobs) * (b_true_logprobs[~b_true_match_mask]-b_true_oos_logprobs), dim=0)
		b_pred_oos_logprobs = pred_oos_logprobs[batch_idx]
		b_oos_ce = -th.exp(b_true_oos_logprobs) * b_pred_oos_logprobs
		# combine
		ios_ces[batch_idx] = b_ios_ce
		oos_ces[batch_idx] = b_oos_ce
		true_oos_logprobs[batch_idx] = b_true_oos_logprobs
		true_oos_entropies[batch_idx] = b_true_oos_entropy
	return ios_ces, oos_ces, true_oos_logprobs, true_oos_entropies

def sparse_cross_entropy_vec(
		true_mzs: th.Tensor,
		true_logprobs: th.Tensor,
		true_batch_idxs: th.Tensor,
		pred_mzs: th.Tensor,
		pred_logprobs: th.Tensor,
		pred_batch_idxs: th.Tensor,
		pred_oos_logprobs: th.Tensor,
		tolerance: float,
		relative: bool,
		tolerance_min_mz: float,
		oos_tolerance_multiple: int,
		gaussian_renormalize: bool,
		loss_batch_size: int) -> th.Tensor:
	
	# print(pred_logprobs.shape[0]*true_logprobs.shape[0]*4 / 1e6)
	float_dtype = true_logprobs.dtype
	int_type = true_batch_idxs.dtype
	device = true_logprobs.device
	mask_value = LOG_ZERO(float_dtype)
	batch_size = th.max(true_batch_idxs)+1
	loss_num_batches = (batch_size // loss_batch_size) + int(batch_size % loss_batch_size > 0)
	true_batch_cumsum = th.cumsum(F.pad(th.unique(true_batch_idxs,return_counts=True)[1],(1,0),"constant",0),dim=0)
	pred_batch_cumsum = th.cumsum(F.pad(th.unique(pred_batch_idxs,return_counts=True)[1],(1,0),"constant",0),dim=0)

	ios_ce = th.zeros([batch_size],dtype=float_dtype,device=device)
	oos_ce = th.zeros([batch_size],dtype=float_dtype,device=device)

	for bl in range(loss_num_batches):

		bl_lower = bl*loss_batch_size
		bl_upper = min((bl+1)*loss_batch_size,batch_size)
		bl_true_lower = true_batch_cumsum[bl_lower]
		bl_true_upper = true_batch_cumsum[bl_upper]
		bl_pred_lower = pred_batch_cumsum[bl_lower]
		bl_pred_upper = pred_batch_cumsum[bl_upper]
		bl_batch_size = bl_upper-bl_lower
		# print((bl_true_upper-bl_true_lower)*(bl_pred_upper-bl_pred_lower)*4)

		bl_true_batch_idxs = true_batch_idxs[bl_true_lower:bl_true_upper] - bl_lower
		bl_pred_batch_idxs = pred_batch_idxs[bl_pred_lower:bl_pred_upper] - bl_lower
		bl_true_logprobs = true_logprobs[bl_true_lower:bl_true_upper]
		bl_pred_logprobs = pred_logprobs[bl_pred_lower:bl_pred_upper]
		bl_true_mzs = true_mzs[bl_true_lower:bl_true_upper]
		bl_pred_mzs = pred_mzs[bl_pred_lower:bl_pred_upper]
		bl_pred_oos_logprobs = pred_oos_logprobs[bl_lower:bl_upper]

		# this should be optimized
		bl_batch_mask = bl_true_batch_idxs.reshape(-1,1) == bl_pred_batch_idxs.reshape(1,-1)
		bl_match_mask = calculate_match_mzs(
			bl_true_mzs,
			bl_pred_mzs,
			tolerance=oos_tolerance_multiple*tolerance,
			relative=relative,
			tolerance_min_mz=tolerance_min_mz
		)
		bl_both_mask = bl_batch_mask & bl_match_mask
		del bl_batch_mask, bl_match_mask

		bl_true_both_mask = th.any(bl_both_mask,dim=1)

		# TODO: move this before the loop?
		if relative:
			bl_stds = th.clamp(bl_pred_mzs,min=tolerance_min_mz)*tolerance
		else:
			bl_stds = tolerance*th.ones_like(bl_pred_mzs)
		bl_vars = bl_stds**2
		if gaussian_renormalize:
			assert oos_tolerance_multiple > 0, oos_tolerance_multiple
			bl_trunc_factor = th.erf(th.tensor(oos_tolerance_multiple,device=device)/float(np.sqrt(2)))
			bl_log_trunc_factors = th.log(bl_trunc_factor*th.ones_like(bl_pred_mzs))
		else:
			bl_log_trunc_factors = th.zeros_like(bl_pred_mzs)
		
		bl_ios_log_probs = -0.5*(bl_true_mzs.reshape(-1,1)-bl_pred_mzs.reshape(1,-1))**2/bl_vars.reshape(1,-1) - 0.5*th.log(2*th.pi*bl_vars.reshape(1,-1))
		bl_ios_log_probs = bl_ios_log_probs + bl_pred_logprobs.reshape(1,-1) - bl_log_trunc_factors.reshape(1,-1)
		bl_ios_log_probs = bl_ios_log_probs + (~bl_both_mask).float()*mask_value
		bl_ios_log_probs = th.logsumexp(bl_ios_log_probs,dim=1)

		if not th.any(bl_true_both_mask):
			# all zeros
			bl_ios_ce = th.full([bl_batch_size],LOG_ZERO(float_dtype),dtype=float_dtype,device=device)
		else:
			bl_ios_ce = scatter_reduce(
				(-th.exp(bl_true_logprobs)*bl_ios_log_probs)[bl_true_both_mask],
				bl_true_batch_idxs[bl_true_both_mask],
				"sum",
				dim=0,
				dim_size=bl_batch_size,
				default=0.)
			ios_ce[bl_lower:bl_upper] = bl_ios_ce

		if not th.all(bl_true_both_mask):
			bl_true_oos_logprobs = scatter_logsumexp(
				bl_true_logprobs[~bl_true_both_mask],
				bl_true_batch_idxs[~bl_true_both_mask],
				dim_size=bl_batch_size
			)
		else:
			# all zeros
			bl_true_oos_logprobs = th.full([bl_batch_size],LOG_ZERO(float_dtype),dtype=float_dtype,device=device)
		bl_oos_ce = -th.exp(bl_true_oos_logprobs)*bl_pred_oos_logprobs
		oos_ce[bl_lower:bl_upper] = bl_oos_ce
		
	return ios_ce, oos_ce, 0., 1.

def get_sparse_cross_entropy_fn(
		dist: str,
		vectorized: bool,
		tolerance: float,
		relative: bool,
		tolerance_min_mz: float,
		oos_tolerance_multiple: float,
		gaussian_renormalize: bool,
		pm_tolerance_multiple: float,
		loss_batch_size: int) -> Callable:

	if dist == "gaussian":
		ce_fn = mog_ce_fn
	elif dist == "peak_marginal":
		ce_fn = pm_ce_fn
	else:
		raise NotImplementedError

	if vectorized:
		assert dist == "gaussian", dist
		sce_fn = lambda *args: sparse_cross_entropy_vec(
			*args,
			tolerance=tolerance,
			relative=relative,
			tolerance_min_mz=tolerance_min_mz,
			oos_tolerance_multiple=oos_tolerance_multiple,
			gaussian_renormalize=gaussian_renormalize,
			loss_batch_size=loss_batch_size
		)
	else:
		if dist == "gaussian":
			tolerance_multiple = oos_tolerance_multiple
		else:
			tolerance_multiple = pm_tolerance_multiple
		sce_fn = lambda *args: sparse_cross_entropy_seq(
			*args,
			ce_fn=ce_fn,
			tolerance=tolerance,
			relative=relative,
			tolerance_min_mz=tolerance_min_mz,
			tolerance_multiple=tolerance_multiple,
			gaussian_renormalize=gaussian_renormalize,
		)
	return sce_fn

def sparse_entropy_fn(
		logprobs: th.Tensor,
		batch_idxs: th.Tensor,
		oos_logprobs: th.Tensor=None,
		renormalize: bool=False,
		support_size_delta: float=0.) -> th.Tensor:

	if renormalize:
		logpartition = scatter_logsumexp(
			logprobs,
			batch_idxs
		)
		if oos_logprobs is not None:
			logpartition = th.logsumexp(th.stack([logpartition,oos_logprobs],dim=1),dim=1)
			oos_logprobs = oos_logprobs - logpartition
		logprobs = logprobs - th.gather(
			input=logpartition,
			index=batch_idxs,
			dim=0
		)
	logprobs = th.clamp(logprobs,max=0.)
	probs = th.exp(logprobs)	
	plogp = probs*logprobs
	batch_size = th.max(batch_idxs)+1
	entropy = -scatter_reduce(
		src=plogp,
		index=batch_idxs,
		reduce="sum",
		dim_size=batch_size,
	)
	assert th.min(entropy) >= -0.001, th.min(entropy)
	support_size = scatter_reduce(
		src=th.ones_like(batch_idxs,dtype=logprobs.dtype),
		index=batch_idxs,
		reduce="sum",
		dim_size=batch_size,
	)
	support_size += support_size_delta
	if oos_logprobs is not None:
		assert oos_logprobs.shape[0] == batch_size, (oos_logprobs.shape,batch_size)
		entropy = entropy - th.exp(oos_logprobs)*oos_logprobs
		support_size = support_size + 1.
	max_entropy = safelog(support_size)
	# rescale entropy
	max_entropy = th.clamp(max_entropy,min=EPS)
	entropy = th.clamp(entropy,min=th.zeros_like(max_entropy),max=max_entropy)
	norm_entropy = entropy/max_entropy
	assert not th.any(norm_entropy < 0.) and not th.any(norm_entropy > 1.)
	return entropy, norm_entropy

def sparse_conditional_entropy_fn(
		marginal_logprobs: th.Tensor,
		marginal_batch_idxs: th.Tensor,
		conditional_logprobs: th.Tensor,
		conditional_idxs: th.Tensor,
		conditional_batch_idxs: th.Tensor,
		conditional_support_size_delta: float=0.) -> th.Tensor:

	batch_size = th.max(conditional_batch_idxs)+1
	marginal_support_size = marginal_logprobs.shape[0]
	marginal_logprobs = th.clamp(marginal_logprobs,max=0.)
	conditional_logprobs = th.clamp(conditional_logprobs,max=0.)
	marginal_probs = th.exp(marginal_logprobs)
	conditional_plogp = th.exp(conditional_logprobs)*conditional_logprobs
	conditional_entropy = -scatter_reduce(
		conditional_plogp,
		index=conditional_idxs,
		reduce="sum",
		dim_size=marginal_support_size
	)
	entropy = scatter_reduce(
		marginal_probs * conditional_entropy,
		index=marginal_batch_idxs,
		reduce="sum",
		dim_size=batch_size
	)
	assert th.min(entropy) >= -0.001, th.min(entropy)
	conditional_support_size = scatter_reduce(
		th.ones_like(conditional_logprobs,dtype=marginal_logprobs.dtype),
		index=conditional_idxs,
		reduce="sum",
		dim_size=marginal_support_size
	)
	conditional_support_size += conditional_support_size_delta
	max_entropy = scatter_reduce(
		marginal_probs * th.clamp(safelog(conditional_support_size),min=0.),
		index=marginal_batch_idxs,
		reduce="sum",
		dim_size=batch_size
	)
	if th.any(max_entropy < 0.):
		import pdb; pdb.set_trace()
	# rescale entropy
	max_entropy = th.clamp(max_entropy,min=EPS)
	entropy = th.clamp(entropy,min=th.zeros_like(max_entropy),max=max_entropy)
	norm_entropy = entropy/max_entropy
	assert not th.any(norm_entropy < 0.) and not th.any(norm_entropy > 1.)
	return entropy, norm_entropy

def get_edge_loss_fn(edge_loss_fn_type: str, constant: float) -> Callable:

	if edge_loss_fn_type == "quadratic":
		edge_loss_fn = lambda x: constant*x**2
	elif edge_loss_fn_type == "linear":
		edge_loss_fn = lambda x: constant*th.abs(x)
	elif edge_loss_fn_type == "exponential":
		edge_loss_fn = lambda x: constant*th.exp(th.abs(x))
	else:
		raise NotImplementedError
	return edge_loss_fn

def sparse_cosine_distance(
		true_mzs: th.Tensor, 
		true_logprobs: th.Tensor,
		true_batch_idxs: th.Tensor,
		pred_mzs: th.Tensor,
		pred_logprobs: th.Tensor,
		pred_batch_idxs: th.Tensor,
		mz_max: float=1500.,
		mz_bin_res: float=0.01,
		log_distance: bool=False) -> th.Tensor:

	# sparse bin
	true_bin_idxs, true_bin_logprobs, true_bin_batch_idxs = batched_bin_func(
		true_mzs,
		true_logprobs,
		true_batch_idxs,
		mz_max=mz_max,
		mz_bin_res=mz_bin_res,
		agg="lse",
		sparse=True
	)
	pred_bin_idxs, pred_bin_logprobs, pred_bin_batch_idxs = batched_bin_func(
		pred_mzs,
		pred_logprobs,
		pred_batch_idxs,
		mz_max=mz_max,
		mz_bin_res=mz_bin_res,
		agg="lse",
		sparse=True
	)
	return sparse_cosine_distance_binned(
		true_bin_idxs,
		true_bin_logprobs,
		true_bin_batch_idxs,
		pred_bin_idxs,
		pred_bin_logprobs,
		pred_bin_batch_idxs,
		log_distance=log_distance
	)

def sparse_cosine_distance_binned(
	true_bin_idxs: th.Tensor,
	true_bin_logprobs: th.Tensor,
	true_bin_batch_idxs: th.Tensor,
	pred_bin_idxs: th.Tensor,
	pred_bin_logprobs: th.Tensor,
	pred_bin_batch_idxs: th.Tensor,
	log_distance: bool=False) -> th.Tensor:

	# l2 normalize
	true_bin_logprobs = scatter_logl2normalize(
		true_bin_logprobs,
		true_bin_batch_idxs
	)
	pred_bin_logprobs = scatter_logl2normalize(
		pred_bin_logprobs,
		pred_bin_batch_idxs
	)
	# dot product
	pred_mask = th.isin(pred_bin_idxs, true_bin_idxs)
	true_mask = th.isin(true_bin_idxs, pred_bin_idxs)
	batch_size = th.max(true_bin_batch_idxs)+1
	if th.any(pred_mask):
		both_bin_logprobs = pred_bin_logprobs[pred_mask] + true_bin_logprobs[true_mask]
		assert th.all(pred_bin_batch_idxs[pred_mask] == true_bin_batch_idxs[true_mask])
		log_cos_sim = scatter_logsumexp(
			both_bin_logprobs,
			pred_bin_batch_idxs[pred_mask],
			dim_size=batch_size
		)
	else:
		# cosine similarities are all zero
		log_cos_sim = th.full(
			size=(batch_size,),
			fill_value=LOG_ZERO(pred_bin_logprobs.dtype),
			dtype=pred_bin_logprobs.dtype,
			device=pred_bin_logprobs.device
		)
		# involve pred_logprobs to keep gradient
		log_cos_sim = log_cos_sim + 0.*th.mean(pred_bin_logprobs,dim=0)
	if log_distance:
		cos_dist = th.log1p(-th.exp(log_cos_sim))
	else:
		cos_dist = 1.-th.exp(log_cos_sim)
	return cos_dist

def sparse_cosine_distance_hungarian(
		true_mzs: th.Tensor, 
		true_logprobs: th.Tensor,
		true_batch_idxs: th.Tensor,
		pred_mzs: th.Tensor,
		pred_logprobs: th.Tensor,
		pred_batch_idxs: th.Tensor,
		tolerance: float=1e-5,
		relative: bool=True,
		tolerance_min_mz: float=TOLERANCE_MIN_MZ,
		log_distance: bool=False) -> th.Tensor:

	# round peaks
	true_mzs, true_logprobs, true_batch_idxs = round_aggregate_peaks(
		true_mzs,
		true_logprobs,
		true_batch_idxs,
		agg="lse"
	)
	pred_mzs, pred_logprobs, pred_batch_idxs = round_aggregate_peaks(
		pred_mzs,
		pred_logprobs,
		pred_batch_idxs,
		agg="lse"
	)
	# calculate
	batch_size = th.max(true_batch_idxs)+1
	cos_dist_hun = th.zeros([batch_size],device=true_logprobs.device,dtype=true_logprobs.dtype)
	for batch_idx in range(batch_size):
		b_true_mask = (true_batch_idxs==batch_idx)
		b_pred_mask = (pred_batch_idxs==batch_idx)
		b_true_mzs = true_mzs[b_true_mask]
		b_pred_mzs = pred_mzs[b_pred_mask]
		b_true_logprobs = true_logprobs[b_true_mask]
		b_pred_logprobs = pred_logprobs[b_pred_mask]
		# l2 normalize
		b_true_logprobs = b_true_logprobs - 0.5*th.logsumexp(2*b_true_logprobs,dim=0)
		b_pred_logprobs = b_pred_logprobs - 0.5*th.logsumexp(2*b_pred_logprobs,dim=0)
		b_match_mzs = calculate_match_mzs(
			b_true_mzs,
			b_pred_mzs,
			tolerance=tolerance,
			relative=relative,
			tolerance_min_mz=tolerance_min_mz
		)
		b_true_match_mzs = th.any(b_match_mzs,dim=1)
		b_pred_match_mzs = th.any(b_match_mzs,dim=0)
		b_score = th.exp(
			b_true_logprobs[b_true_match_mzs].detach().unsqueeze(1) + \
			b_pred_logprobs[b_pred_match_mzs].detach().unsqueeze(0)
		)
		b_score[~b_match_mzs[b_true_match_mzs][:,b_pred_match_mzs]] = LOG_ZERO(b_score.dtype)
		b_true_idxs, b_pred_idxs = scipy_linear_sum_assignment(b_score, maximize=True)
		b_log_cos_hun = th.logsumexp(
			b_true_logprobs[b_true_match_mzs][b_true_idxs] + \
			b_pred_logprobs[b_pred_match_mzs][b_pred_idxs],
			dim=0
		)
		if log_distance:
			b_cos_dist_hun = th.log1p(-th.exp(b_log_cos_hun))
		else:
			b_cos_dist_hun = 1.-th.exp(b_log_cos_hun)
		cos_dist_hun[batch_idx] = b_cos_dist_hun
	return cos_dist_hun

def sparse_jensen_shannon_divergence(
		true_mzs: th.Tensor, 
		true_logprobs: th.Tensor,
		true_batch_idxs: th.Tensor,
		pred_mzs: th.Tensor,
		pred_logprobs: th.Tensor,
		pred_batch_idxs: th.Tensor,
		mz_max: float=1500.,
		mz_bin_res: float=0.01,
		log_min: float=EPS) -> th.Tensor:

	# sparse bin
	true_bin_idxs, true_bin_logprobs, true_bin_batch_idxs = batched_bin_func(
		true_mzs,
		true_logprobs,
		true_batch_idxs,
		mz_max=mz_max,
		mz_bin_res=mz_bin_res,
		agg="lse",
		sparse=True
	)
	pred_bin_idxs, pred_bin_logprobs, pred_bin_batch_idxs = batched_bin_func(
		pred_mzs,
		pred_logprobs,
		pred_batch_idxs,
		mz_max=mz_max,
		mz_bin_res=mz_bin_res,
		agg="lse",
		sparse=True
	)
	# calculate similarity
	jss = jss_helper(
        true_bin_idxs,
        true_bin_logprobs.exp(),
        true_bin_batch_idxs,
        pred_bin_idxs,
        pred_bin_logprobs.exp(),
        pred_bin_batch_idxs,
		log_min=log_min
    )
	return 1.-jss
