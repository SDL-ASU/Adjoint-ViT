from functools import partial
from torch import optim as optim

try:
	from apex.optimizers import FusedAdam, FusedLAMB
except:
	FusedAdam = None
	FusedLAMB = None
	print("To use FusedLAMB or FusedAdam, please install apex.")


def build_optimizer_search(config, model, simmim=False, is_pretrain=False):
	"""
	Build optimizer, set weight decay of normalization to 0 by default.
	"""
	skip = {}
	skip_keywords = {}
	if hasattr(model, 'no_weight_decay'):
		skip = model.no_weight_decay()
	if hasattr(model, 'no_weight_decay_keywords'):
		skip_keywords = model.no_weight_decay_keywords()
	if simmim:
		if is_pretrain:
			parameters, gumbel_list = get_pretrain_param_groups(model, skip, skip_keywords)
		else:
			depths = config.MODEL.SWIN.DEPTHS if config.MODEL.TYPE == 'swin' else config.MODEL.SWINV2.DEPTHS
			num_layers = sum(depths)
			get_layer_func = partial(get_swin_layer, num_layers=num_layers + 2, depths=depths)
			scales = list(config.TRAIN.LAYER_DECAY ** i for i in reversed(range(num_layers + 2)))
			parameters, gumbel_list = get_finetune_param_groups(model, config.TRAIN.BASE_LR, config.TRAIN.WEIGHT_DECAY, config.TRAIN.GUMBEL_WEIGHT_DECAY, get_layer_func, scales, skip, skip_keywords)
	else:
		parameters, gumbel_list = set_weight_decay(model, skip, skip_keywords)

	opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
	optimizer = None
	gumbel_optimizer = None
	if opt_lower == 'adamw':
		optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
								lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
		gumbel_optimizer = optim.AdamW(gumbel_list, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
								lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.GUMBEL_WEIGHT_DECAY)
	elif opt_lower == 'fused_adam':
		optimizer = FusedAdam(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
							  lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
		gumbel_optimizer = FusedAdam(gumbel_list, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
							  lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.GUMBEL_WEIGHT_DECAY)

	return optimizer, gumbel_optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
	has_gumbel_decay = []
	gumbel_no_decay = []

	has_decay = []
	no_decay = []

	for name, param in model.named_parameters():
		if 'gumbel' in name:
			if not param.requires_grad:
				continue
			if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
					check_keywords_in_name(name, skip_keywords):
				gumbel_no_decay.append(param)
			else:
				has_gumbel_decay.append(param)
		else:
			if not param.requires_grad:
				continue  # frozen weights
			if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
					check_keywords_in_name(name, skip_keywords):
				no_decay.append(param)
				# print(f"{name} has no weight decay")
			else:
				has_decay.append(param)
	return [
		{'params': has_decay},
		{'params': no_decay, 'weight_decay': 0.}], [
		{'params': has_gumbel_decay},
		{'params': gumbel_no_decay, 'weight_decay': 0.}]

def check_keywords_in_name(name, keywords=()):
	isin = False
	for keyword in keywords:
		if keyword in name:
			isin = True
	return isin

def get_pretrain_param_groups(model, skip_list=(), skip_keywords=()):
	has_decay = []
	no_decay = []
	has_decay_name = []
	no_decay_name = []
	
	has_gumbel_decay = []
	gumbel_no_decay = []
	has_gumbel_decay_name = []
	gumbel_no_decay_name = []
	
	for name, param in model.named_parameters():
		if 'gumbel' in name:
			if not param.requires_grad:
				continue
			if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
					check_keywords_in_name(name, skip_keywords):
				gumbel_no_decay.append(param)
				gumbel_no_decay_name.append(name)
			else:
				has_gumbel_decay.append(param)
				has_gumbel_decay_name.append(name)
		else:
			if not param.requires_grad:
				continue
			if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
					check_keywords_in_name(name, skip_keywords):
				no_decay.append(param)
				no_decay_name.append(name)
			else:
				has_decay.append(param)
				has_decay_name.append(name)
	return [
		{'params': has_decay},
		{'params': no_decay, 'weight_decay': 0.}], [
		{'params': has_gumbel_decay},
		{'params': gumbel_no_decay, 'weight_decay': 0.}]

def get_swin_layer(name, num_layers, depths):
	if name in ("mask_token"):
		return 0
	elif name.startswith("patch_embed"):
		return 0
	elif name.startswith("layers"):
		layer_id = int(name.split('.')[1])
		block_id = name.split('.')[3]
		if block_id == 'reduction' or block_id == 'norm':
			return sum(depths[:layer_id + 1])
		layer_id = sum(depths[:layer_id]) + int(block_id)
		return layer_id + 1
	else:
		return num_layers - 1

def get_finetune_param_groups(model, lr, weight_decay, gumbel_weight_decay, get_layer_func, scales, skip_list=(), skip_keywords=()):
	parameter_group_names = {}
	parameter_group_vars = {}

	gumbel_parameter_group_names = {}
	gumbel_parameter_group_vars = {}

	for name, param in model.named_parameters():
		if 'gumbel' in name:
			if not param.requires_grad:
				continue
			if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
					check_keywords_in_name(name, skip_keywords):
				gumbel_group_name = "no_decay"
				this_gumbel_weight_decay = 0.
			else:
				gumbel_group_name = "decay"
				this_gumbel_weight_decay = gumbel_weight_decay
			if get_layer_func is not None:
				layer_id = get_layer_func(name)
				gumbel_group_name = "layer_%d_%s" % (layer_id, gumbel_group_name)
			else:
				layer_id = None

			if gumbel_group_name not in gumbel_parameter_group_names:
				if scales is not None:
					scale = scales[layer_id]
				else:
					scale = 1.

				gumbel_parameter_group_names[gumbel_group_name] = {
					"group_name": gumbel_group_name,
					"weight_decay": this_gumbel_weight_decay,
					"params": [],
					"lr": lr * scale,
					"lr_scale": scale,
				}
				gumbel_parameter_group_vars[gumbel_group_name] = {
					"group_name": gumbel_group_name,
					"weight_decay": this_gumbel_weight_decay,
					"params": [],
					"lr": lr * scale,
					"lr_scale": scale
				}

			gumbel_parameter_group_vars[gumbel_group_name]["params"].append(param)
			gumbel_parameter_group_names[gumbel_group_name]["params"].append(name)
		else:
			if not param.requires_grad:
				continue
			if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
					check_keywords_in_name(name, skip_keywords):
				group_name = "no_decay"
				this_weight_decay = 0.
			else:
				group_name = "decay"
				this_weight_decay = weight_decay
			if get_layer_func is not None:
				layer_id = get_layer_func(name)
				group_name = "layer_%d_%s" % (layer_id, group_name)
			else:
				layer_id = None

			if group_name not in parameter_group_names:
				if scales is not None:
					scale = scales[layer_id]
				else:
					scale = 1.

				parameter_group_names[group_name] = {
					"group_name": group_name,
					"weight_decay": this_weight_decay,
					"params": [],
					"lr": lr * scale,
					"lr_scale": scale,
				}
				parameter_group_vars[group_name] = {
					"group_name": group_name,
					"weight_decay": this_weight_decay,
					"params": [],
					"lr": lr * scale,
					"lr_scale": scale
				}

			parameter_group_vars[group_name]["params"].append(param)
			parameter_group_names[group_name]["params"].append(name)
	return list(parameter_group_vars.values()), list(gumbel_parameter_group_vars.values())
