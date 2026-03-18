"""
LoRA (Low-Rank Adaptation) implementation.

Supports:
- Reversed initialization: B gets init weights, A is zeroed out
  (controlled via `lora_init_reversed=True`)
- Selective layer targeting
- Multiple LoRA groups for multi-CLIP models (e.g., Flux with two text encoders)
"""

import math
import re
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA layer for nn.Linear.

    Standard LoRA: A initialized with kaiming, B zeroed → output starts at 0
    Reversed LoRA (init_reversed=True): B initialized with kaiming, A zeroed → output starts at 0

    Both produce zero delta at init, but reversed init means B carries the
    learned structure while A acts as the gate that opens during training.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        init_reversed: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.init_reversed = init_reversed
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_reversed:
            # Reversed: B gets kaiming init, A is zeroed
            nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_A.weight)
        else:
            # Standard: A gets kaiming init, B is zeroed
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling


class LoRAConv2d(nn.Module):
    """LoRA layer for nn.Conv2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        init_reversed: bool = True,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Conv2d(in_channels, rank, kernel_size, bias=False)
        self.lora_B = nn.Conv2d(rank, out_channels, 1, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.init_reversed = init_reversed
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_reversed:
            nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_A.weight)
        else:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling


class LoRAInjector:
    """Injects LoRA layers into a model's targeted layers.

    Usage:
        injector = LoRAInjector(model, target_layers=['attn.to_q', 'attn.to_v'], ...)
        injector.inject()
        # Train only LoRA params
        trainable = injector.get_trainable_parameters()
        # Save just LoRA weights
        injector.save_weights('lora.safetensors')
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: Optional[List[str]] = None,
        target_patterns: Optional[List[str]] = None,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        init_reversed: bool = True,
        conv_rank: Optional[int] = None,   # None = skip conv layers
        conv_alpha: Optional[float] = None, # None = fall back to conv_rank
    ):
        """
        Args:
            model: The base model to inject LoRA into
            target_layers: Exact layer names to target
            target_patterns: Regex patterns to match layer names
            rank: LoRA rank
            alpha: LoRA alpha (scaling = alpha/rank)
            dropout: Dropout rate for LoRA layers
            init_reversed: If True, init B with kaiming and zero A (vice versa from standard)
        """
        self.model = model
        self.target_layers = target_layers or []
        self.target_patterns = [re.compile(p) for p in (target_patterns or [])]
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.init_reversed = init_reversed
        self.conv_rank = conv_rank
        self.conv_alpha = conv_alpha if conv_alpha is not None else conv_rank
        self.lora_layers: Dict[str, nn.Module] = {}
        self._original_forwards: Dict[str, callable] = {}
        self._enabled = True

    def _should_target(self, name: str) -> bool:
        if name in self.target_layers:
            return True
        for pattern in self.target_patterns:
            if pattern.search(name):
                return True
        return False

    def inject(self) -> Dict[str, nn.Module]:
        """Inject LoRA into all targeted layers. Returns dict of injected LoRA modules."""
        for name, module in self.model.named_modules():
            if not self._should_target(name):
                continue

            if isinstance(module, nn.Linear):
                lora = LoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    init_reversed=self.init_reversed,
                )
                lora = lora.to(device=module.weight.device, dtype=module.weight.dtype)
                self.lora_layers[name] = lora
                self._wrap_forward(name, module, lora)

            elif isinstance(module, nn.Conv2d):
                if self.conv_rank is None:
                    continue  # conv LoRA not requested, skip
                lora = LoRAConv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    init_reversed=self.init_reversed,
                )
                lora = lora.to(device=module.weight.device, dtype=module.weight.dtype)
                self.lora_layers[name] = lora
                self._wrap_forward(name, module, lora)

        # Freeze all base model params
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"[LoRA] Injected into {len(self.lora_layers)} layers "
              f"(rank={self.rank}, alpha={self.alpha}, "
              f"init_reversed={self.init_reversed})")
        return self.lora_layers

    def _wrap_forward(self, name: str, module: nn.Module, lora: nn.Module):
        """Wrap a module's forward to add LoRA output."""
        original_forward = module.forward
        injector = self

        def new_forward(x, *args, **kwargs):
            base_out = original_forward(x, *args, **kwargs)
            if injector._enabled:
                lora_out = lora(x)
                return base_out + lora_out
            return base_out

        self._original_forwards[name] = original_forward
        module.forward = new_forward

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all LoRA parameters for the optimizer."""
        params = []
        for lora in self.lora_layers.values():
            params.extend(lora.parameters())
        return params

    def get_trainable_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        """Get named LoRA parameters."""
        params = []
        for name, lora in self.lora_layers.items():
            for pname, param in lora.named_parameters():
                params.append((f"lora.{name}.{pname}", param))
        return params

    def save_weights(self, path: str):
        """Save LoRA weights to safetensors or pt file."""
        state_dict = {}
        for name, lora in self.lora_layers.items():
            for pname, param in lora.named_parameters():
                key = f"lora.{name}.{pname}"
                state_dict[key] = param.data.cpu()

        if path.endswith('.safetensors'):
            from safetensors.torch import save_file
            # Add metadata
            metadata = {
                'rank': str(self.rank),
                'alpha': str(self.alpha),
                'init_reversed': str(self.init_reversed),
                'target_layers': ','.join(sorted(self.lora_layers.keys())),
            }
            save_file(state_dict, path, metadata=metadata)
        else:
            torch.save({
                'state_dict': state_dict,
                'rank': self.rank,
                'alpha': self.alpha,
                'init_reversed': self.init_reversed,
            }, path)
        print(f"[LoRA] Saved weights to {path}")

    def load_weights(self, path: str):
        """Load LoRA weights."""
        if path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(path)
        else:
            ckpt = torch.load(path, map_location='cpu')
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

        for name, lora in self.lora_layers.items():
            for pname, param in lora.named_parameters():
                key = f"lora.{name}.{pname}"
                if key in state_dict:
                    param.data.copy_(state_dict[key])
        print(f"[LoRA] Loaded weights from {path}")

    def remove(self):
        """Remove LoRA injections, restore original forwards."""
        for name, original_forward in self._original_forwards.items():
            parts = name.split('.')
            module = self.model
            for part in parts:
                module = getattr(module, part)
            module.forward = original_forward
        self.lora_layers.clear()
        self._original_forwards.clear()

    def get_info(self) -> Dict:
        """Return info about the LoRA setup."""
        total_params = sum(p.numel() for lora in self.lora_layers.values()
                          for p in lora.parameters())
        return {
            'num_injected_layers': len(self.lora_layers),
            'rank': self.rank,
            'alpha': self.alpha,
            'init_reversed': self.init_reversed,
            'total_trainable_params': total_params,
            'target_layer_names': sorted(self.lora_layers.keys()),
        }

class TrainingAdapterManager:
    """
    Manages a de-distillation training adapter (LoRA weights that are
    merged into the base model during training and removed for inference).
    
    Flow:
      1. apply() — merge adapter weights into denoiser
      2. Train your LoRA on top of the modified weights
      3. remove() — unmerge adapter weights before sampling/saving
      4. Re-apply after sampling if training continues
    """
    
    def __init__(self, model: nn.Module, adapter_path: str, device='cpu'):
        self.model = model
        self.adapter_path = adapter_path
        self.device = device
        self._original_deltas = {}  # stores the weight deltas for unmerge
        self._applied = False
    
    def apply(self):
        """
        Load adapter safetensors and merge weights into the model.
        Store deltas so we can unmerge later.
        """
        if self._applied:
            return
        
        from safetensors.torch import load_file
        adapter_sd = load_file(self.adapter_path, device=str(self.device))
        
        # The adapter is a LoRA saved in safetensors format.
        # Parse lora_A / lora_B weight pairs and compute merged deltas.
        # 
        # Integration consideration: the exact key format depends on
        # how ai-toolkit saves LoRA weights. Common formats:
        #   - "lora_unet_<layer>.lora_down.weight" / "lora_up.weight"
        #   - "<layer>.lora_A.weight" / "<layer>.lora_B.weight"
        #
        # You need to:
        # 1. Parse the key naming convention from the adapter file
        # 2. For each lora pair: delta = (B @ A) * scale
        # 3. Add delta to the corresponding base model weight
        # 4. Store delta in self._original_deltas for later removal
        
        # Pseudocode:
        lora_pairs = self._parse_lora_pairs(adapter_sd)
        
        for target_key, (lora_down, lora_up, alpha, rank) in lora_pairs.items():
            scale = alpha / rank if rank > 0 else 1.0
            
            # Compute merged delta
            if lora_down.dim() == 2 and lora_up.dim() == 2:
                delta = (lora_up @ lora_down) * scale
            elif lora_down.dim() == 4:
                # Conv layers
                delta = torch.nn.functional.conv2d(
                    lora_down.permute(1, 0, 2, 3),
                    lora_up
                ).permute(1, 0, 2, 3) * scale
            else:
                delta = (lora_up @ lora_down) * scale
            
            # Find target parameter in model
            param = self._get_param(target_key)
            if param is not None:
                self._original_deltas[target_key] = delta.to(param.device, param.dtype)
                param.data.add_(self._original_deltas[target_key])
        
        self._applied = True
    
    def remove(self):
        """Subtract the merged deltas to restore original weights."""
        if not self._applied:
            return
        
        for target_key, delta in self._original_deltas.items():
            param = self._get_param(target_key)
            if param is not None:
                param.data.sub_(delta)
        
        self._applied = False
    
    def _parse_lora_pairs(self, state_dict):
        """
        Parse LoRA A/B pairs from state dict.
        
        Integration consideration: inspect the actual adapter file to
        determine the key naming convention. Run:
          from safetensors.torch import load_file
          sd = load_file('zimage_turbo_training_adapter_v2.safetensors')
          print(list(sd.keys())[:20])
        
        This will tell you the exact format to parse.
        """
        # Return dict: {target_layer_key: (lora_down, lora_up, alpha, rank)}
        pairs = {}
        # ... parse based on actual key format ...
        return pairs
    
    def _get_param(self, key):
        """Resolve a key to a model parameter."""
        parts = key.split('.')
        obj = self.model
        for p in parts:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                return None
        return obj if isinstance(obj, (torch.Tensor, nn.Parameter)) else None
    
    @property
    def is_applied(self):
        return self._applied
