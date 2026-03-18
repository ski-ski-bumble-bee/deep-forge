"""
Model registry for pretrained and loadable models.

Supports:
- Loading .safetensors single-file models (CivitAI style)
- Loading HuggingFace-style models if available
- Reconstructing architecture from state dict shape analysis
- Freezing/unfreezing selected layers for fine-tuning
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def load_state_dict_from_file(path: str, device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """Load state dict from any supported format."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.safetensors':
        from safetensors.torch import load_file
        return load_file(path, device=device)
    elif ext in ('.ckpt', '.pt', '.pth', '.bin'):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            for key in ('state_dict', 'model', 'model_state_dict'):
                if key in ckpt:
                    return ckpt[key]
            return ckpt
        return ckpt
    raise ValueError(f"Unsupported format: {ext}")


def analyze_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Analyze a state dict structure for the frontend."""
    total_params = sum(t.numel() for t in state_dict.values())
    groups = {}
    linear_layers = []
    conv_layers = []
    all_layers = []

    for key, tensor in sorted(state_dict.items()):
        parts = key.split('.')
        group = '.'.join(parts[:2]) if len(parts) >= 2 else parts[0]
        if group not in groups:
            groups[group] = []
        groups[group].append({'key': key, 'shape': list(tensor.shape), 'params': tensor.numel()})

        if 'weight' in key:
            if len(tensor.shape) == 2:
                linear_layers.append(key)
            elif len(tensor.shape) == 4:
                conv_layers.append(key)

        all_layers.append({
            'key': key,
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'params': tensor.numel(),
        })

    return {
        'total_params': total_params,
        'total_params_human': _human_number(total_params),
        'num_keys': len(state_dict),
        'linear_layers': linear_layers,
        'conv_layers': conv_layers,
        'groups': {k: v[:10] for k, v in groups.items()},
        'group_names': sorted(groups.keys()),
        'all_layers': all_layers[:500],
    }


def reconstruct_model_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    """
    Attempt to reconstruct a sequential model from a flat state dict.
    Works for simple sequential models. Complex models need explicit architecture.
    """
    layers = []
    layer_map = {}

    # Group by layer prefix
    for key in sorted(state_dict.keys()):
        parts = key.rsplit('.', 1)
        if len(parts) == 2:
            prefix, param_name = parts
        else:
            prefix, param_name = key, 'weight'

        if prefix not in layer_map:
            layer_map[prefix] = {}
        layer_map[prefix][param_name] = state_dict[key]

    # Reconstruct each layer
    for prefix in sorted(layer_map.keys()):
        params = layer_map[prefix]
        if 'weight' in params:
            w = params['weight']
            has_bias = 'bias' in params

            if len(w.shape) == 2:
                layer = nn.Linear(w.shape[1], w.shape[0], bias=has_bias)
                layer.weight.data.copy_(w)
                if has_bias:
                    layer.bias.data.copy_(params['bias'])
                layers.append(layer)
            elif len(w.shape) == 4:
                layer = nn.Conv2d(w.shape[1], w.shape[0], w.shape[2], bias=has_bias)
                layer.weight.data.copy_(w)
                if has_bias:
                    layer.bias.data.copy_(params['bias'])
                layers.append(layer)
            elif len(w.shape) == 1:
                # BatchNorm running_mean/var or bias-only
                if 'running_mean' in params:
                    layer = nn.BatchNorm2d(w.shape[0]) if len(w.shape) == 1 else nn.BatchNorm1d(w.shape[0])
                    layer.weight.data.copy_(w)
                    if 'bias' in params:
                        layer.bias.data.copy_(params['bias'])
                    if 'running_mean' in params:
                        layer.running_mean.data.copy_(params['running_mean'])
                    if 'running_var' in params:
                        layer.running_var.data.copy_(params['running_var'])
                    layers.append(layer)

    model = nn.Sequential(*layers)
    return model


def freeze_model(model: nn.Module):
    """Freeze all parameters."""
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_layers(model: nn.Module, patterns: List[str]):
    """Unfreeze layers matching patterns."""
    compiled = [re.compile(p) for p in patterns]
    unfrozen = 0
    for name, param in model.named_parameters():
        for pattern in compiled:
            if pattern.search(name):
                param.requires_grad = True
                unfrozen += 1
                break
    return unfrozen


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Get a summary of a model's architecture."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    layer_info = []
    for name, module in model.named_modules():
        if name == '':
            continue
        n_params = sum(p.numel() for p in module.parameters(recurse=False))
        if n_params > 0:
            layer_info.append({
                'name': name,
                'type': type(module).__name__,
                'params': n_params,
                'trainable': any(p.requires_grad for p in module.parameters(recurse=False)),
            })

    return {
        'total_params': total,
        'trainable_params': trainable,
        'frozen_params': frozen,
        'total_human': _human_number(total),
        'trainable_human': _human_number(trainable),
        'layers': layer_info,
    }


LORA_TARGET_PRESETS = {
    'sdxl': {
        'unet_attention': [r'unet.*attn.*to_[qkvo]'],
        'unet_all_linear': [r'unet.*\.(linear|proj)'],
        'text_encoder_1': [r'text_encoder\..*self_attn\.[qkvo]_proj'],
        'text_encoder_2': [r'text_encoder_2\..*self_attn\.[qkvo]_proj'],
    },
    'flux': {
        'transformer_attention': [r'transformer.*attn.*to_[qkvo]'],
        'transformer_mlp': [r'transformer.*mlp.*'],
        'text_encoder_clip': [r'text_encoder\..*self_attn\.[qkvo]_proj'],
        'text_encoder_t5': [r'text_encoder_2\..*SelfAttention\.[qkvo]'],
    },
    'sd15': {
        'unet_attention': [r'unet.*attn.*to_[qkvo]'],
        'text_encoder': [r'text_encoder\..*self_attn\.[qkvo]_proj'],
    },
    'generic': {
        'all_attention': [r'.*attn.*to_[qkv]', r'.*[qkv]_proj'],
        'all_linear': [r'.*\.(linear|proj|fc)'],
    },
}


def _human_number(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)
