from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .adamw import AdamW

try:
    from apex.optimizers import FusedAdam
    has_apex = True
except ImportError:
    has_apex = False


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def optimizer_kwargs(cfg):
    """ cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = dict(
        optimizer_name=cfg.opt,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum)
    if getattr(cfg, 'opt_eps', None) is not None:
        kwargs['eps'] = cfg.opt_eps
    if getattr(cfg, 'opt_betas', None) is not None:
        kwargs['betas'] = cfg.opt_betas
    if getattr(cfg, 'opt_args', None) is not None:
        kwargs.update(cfg.opt_args)
    return kwargs


def create_optimizer(args, model, filter_bias_and_bn=True):
    """ Legacy optimizer factory for backwards compatibility.
    NOTE: Use create_optimizer_v2 for new code.
    """
    return create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        filter_bias_and_bn=filter_bias_and_bn,
    )

def create_optimizer_v2(
        model: nn.Module,
        optimizer_name: str = 'sgd',
        learning_rate: Optional[float] = None,
        weight_decay: float = 0.,
        momentum: float = 0.9,
        filter_bias_and_bn: bool = True,
        **kwargs):

    opt_lower = optimizer_name.lower()
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model.parameters()
    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=learning_rate, weight_decay=weight_decay, **kwargs)
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer

def add_weight_decay_gs(model, weight_decay=1e-5, gumbel_weight_decay=1e-5, skip_list=()):
    gumbel_decay = []
    gumbel_no_decay = []

    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if 'gumbel' in name:
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                gumbel_no_decay.append(param)
            else:
                gumbel_decay.append(param)
        else:
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}], [
        {'params': gumbel_no_decay, 'weight_decay': 0.},
        {'params': gumbel_decay, 'weight_decay': gumbel_weight_decay}]

def get_params(model, gumbel_weight_decay=1e-5, skip_list=()):
    gumbel_decay = []
    gumbel_no_decay = []

    no_decay = []
    for name, param in model.named_parameters():
        if 'gumbel' in name:
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                gumbel_no_decay.append(param)
            else:
                gumbel_decay.append(param)
        else:
            if not param.requires_grad:
                continue  # frozen weights
            no_decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.}], [
        {'params': gumbel_no_decay, 'weight_decay': 0.},
        {'params': gumbel_decay, 'weight_decay': gumbel_weight_decay}]

def create_optimizer_search(args, model, filter_bias_and_bn=True):
    return create_optimizer_search_v2(
        model,
        **optimizer_kwargs(cfg=args),
        filter_bias_and_bn=filter_bias_and_bn,
    )

def create_optimizer_search_v2(
        model: nn.Module,
        optimizer_name: str = 'sgd',
        learning_rate: Optional[float] = None,
        weight_decay: float = 0.,
        gumbel_weight_decay: float = 0.,
        momentum: float = 0.9,
        filter_bias_and_bn: bool = True,
        **kwargs):

    opt_lower = optimizer_name.lower()
    skip = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if weight_decay and filter_bias_and_bn:
        parameters, gumbel_list = add_weight_decay_gs(model, weight_decay, gumbel_weight_decay, skip)
        weight_decay = 0.
        gumbel_weight_decay = 0.
    else:
        parameters, gumbel_list = get_params(model, gumbel_weight_decay, skip)

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=learning_rate, weight_decay=weight_decay, **kwargs)
    gumbel_opt_args = dict(lr=learning_rate, weight_decay=gumbel_weight_decay, **kwargs)

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]

    if opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
        gumbel_optimizer = optim.AdamW(gumbel_list, **gumbel_opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
        gumbel_optimizer = FusedAdam(gumbel_list, adam_w_mode=True, **gumbel_opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)
            gumbel_optimizer = Lookahead(gumbel_optimizer)

    return optimizer, gumbel_optimizer