"""
Background task: unified training across all modes.

ONE setup path. Per-component training.strategy controls everything.
Strategies: frozen, lora, finetune, full, adapter
"""

import traceback
from typing import Any, Dict, List, Optional

from backend.api.state.training_state import training_state, stop_event
from backend.api.dependencies import TB_LOG_DIR
from backend.api.tasks.helpers import APICallback, BundleWrapper


def run_unified_training(config: Dict[str, Any], mode: str,
                         resume_checkpoint: str = None, **kwargs):
    """Entry point called by BackgroundTasks."""
    try:
        stop_event.clear()
        import torch
        from backend.core.unified_trainer import UnifiedTrainer
        from backend.modules.optimizers import create_optimizer

        _seed(config)

        bundle, lora_injectors, loss_fn, training_adapter = _setup_unified(config, mode)
        train_dl, val_dl = _load_dataloaders(config)

        param_groups = _build_param_groups(bundle, lora_injectors, config)
        if not param_groups:
            raise ValueError("No trainable parameters. Check component training.strategy.")

        oc = config.get("optimizer", {})
        optimizer = create_optimizer(
            oc.get("name", "adamw"), param_groups,
            lr=oc.get("lr", 1e-4), weight_decay=oc.get("weight_decay", 0.01),
        )

        scheduler = _build_scheduler(config, optimizer, train_dl)
        callbacks = _build_callbacks(config)

        model = BundleWrapper(bundle)

        tc = config.get("training", {})
        tc["output_dir"] = config.get("output", {}).get("dir", "./outputs")
        lc = config.get("logging", {})

        # Pipeline for sampling
        pipeline = _setup_pipeline(
            config, bundle,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            dtype=config.get('model', {}).get('dtype', 'bfloat16'),
        )

        trainer = UnifiedTrainer(
            model=model, optimizer=optimizer,
            train_dataloader=train_dl, val_dataloader=val_dl,
            loss_fn=loss_fn, scheduler=scheduler, callbacks=callbacks,
            config=tc, full_config=config,
            lora_injectors=lora_injectors, component_bundle=bundle,
            mode=mode, run_name=lc.get("run_name"),
            pipeline=pipeline,
            training_adapter=training_adapter,
            stop_event=stop_event, 
        )

        training_state["_trainer_ref"] = trainer

        freeze_schedule = _build_freeze_schedule(bundle)
        if freeze_schedule:
            trainer._freeze_schedule = freeze_schedule

        if resume_checkpoint:
            trainer.load_checkpoint(resume_checkpoint)
            print(f"[Resume] Loaded checkpoint from {resume_checkpoint}")
            training_state["current_step"] = trainer.global_step
            training_state["current_epoch"] = trainer.current_epoch
        
        trainer.train()
        training_state["status"] = "completed"
        training_state["run_dir"] = trainer.run_dir

    except Exception as e:
        training_state["status"] = "error"
        training_state["error"] = str(e)
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════
# Unified setup
# ═══════════════════════════════════════════════════════════════

def _setup_unified(config: dict, mode: str):
    """Returns (bundle, lora_injectors_dict, loss_fn)."""
    from backend.core.component_loader import (
        load_component_bundle,
        normalize_single_model_to_bundle,
        normalize_model_spec_to_bundle,
    )
    from backend.configs.config_manager import normalize_component_config
    from backend.modules.model_registry import reconstruct_model_from_state_dict

    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    finetune_cfg = config.get("finetune", {})

    training_adapter = None

    # Step 1: Normalize to components list
    components_config = model_cfg.get("components", [])
    if not components_config:
        model_spec = config.get("model_spec")
        if model_spec:
            components_config = normalize_model_spec_to_bundle(model_spec)
        elif model_cfg.get("path"):
            components_config = normalize_single_model_to_bundle(model_cfg)
        else:
            raise ValueError("No model source.")

    # Step 2: Infer strategy defaults from mode
    _apply_mode_defaults(components_config, mode, lora_cfg, finetune_cfg)

    # Step 3: Load bundle
    bundle = load_component_bundle(components_config, base_dir=model_cfg.get("base_dir", ""))

    # Step 4: Build modules & apply strategies
    pipeline_cfg = config.get('pipeline', {})
    adapter_injectors = {}
    lora_injectors = {}

    for comp in bundle:
        normalize_component_config(comp.config)

        if comp.module is None and comp.state_dict:
            comp.module = reconstruct_model_from_state_dict(comp.state_dict)

        if comp.module is None:
            continue

        strategy = comp.config.get("training", {}).get("strategy", "frozen")
        # FIRST: inject training adapter if this is a denoiser with LoRA strategy
        if strategy == "lora" and comp.role == "denoiser":
            training_adapter = _maybe_load_training_adapter(
                comp.module, pipeline_cfg, model_cfg.get("base_dir", "")
            )

        # ── Now apply the training strategy (LoRA injection happens here) ──
        _apply_strategy(comp, strategy, lora_cfg, lora_injectors)

    # Step 5: Loss — skip if pipeline handles it
    has_pipeline = bool(pipeline_cfg.get('name'))
    loss_fn = None if has_pipeline else _build_loss_fn(config)

    return bundle, lora_injectors, loss_fn, training_adapter

def _setup_pipeline(config, bundle, device, dtype):
    """Create the diffusion pipeline if configured. Returns pipeline or None."""
    import torch
    pipe_cfg = config.get('pipeline', {})
    pipe_name = pipe_cfg.get('name')

    if not pipe_name:
        return None

    from backend.pipelines.registry import get_pipeline
    from backend.pipelines.base_pipeline import PipelineComponents

    # Import pipeline modules to trigger registration
    try:
        import backend.pipelines.zimage_turbo_pipeline
    except ImportError:
        pass
    try:
        import backend.pipelines.sdxl_pipeline
    except ImportError:
        pass

    pipeline_cls = get_pipeline(pipe_name)

    # Map bundle components to pipeline components
    components = PipelineComponents()
    for comp in bundle:
        if comp.role == 'denoiser':
            components.denoiser = comp.module
        elif comp.role == 'text_encoder':
            components.text_encoder = comp.module
        elif comp.role == 'vae':
            components.vae = comp.module
        else:
            components.extras[comp.name] = comp.module

    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    pipeline = pipeline_cls(
        components=components,
        device=device,
        dtype=torch_dtype,
        **pipe_cfg.get('params', {}),
    )

    return pipeline

def _apply_mode_defaults(components_config: list, mode: str, lora_cfg: dict, finetune_cfg: dict):
    """Infer strategy from mode for components that don't set one explicitly."""
    global_lora_targets = lora_cfg.get("target_components", [])
    global_unfreeze = finetune_cfg.get("unfreeze_patterns", [])

    for comp in components_config:
        training = comp.setdefault("training", {})
        if training.get("strategy"):
            continue

        if mode == "lora":
            has_lora = training.get("lora") is not None
            in_targets = comp.get("name", "") in global_lora_targets
            is_trainable = comp.get("trainable", False)

            if has_lora or in_targets:
                training["strategy"] = "lora"
                if not has_lora and in_targets:
                    training["lora"] = {
                        "rank": lora_cfg.get("rank", 16), "alpha": lora_cfg.get("alpha", 16),
                        "dropout": lora_cfg.get("dropout", 0.0),
                        "target_patterns": lora_cfg.get("target_patterns", []),
                        "target_layers": lora_cfg.get("target_layers", []),
                    }
            elif comp.get("forward", {}).get("no_grad"):
                training["strategy"] = "frozen"
            elif is_trainable:
                training["strategy"] = "full"
            else:
                training["strategy"] = "frozen"

        elif mode == "full_finetune":
            if comp.get("forward", {}).get("no_grad") or not comp.get("trainable", False):
                training["strategy"] = "frozen"
            else:
                training["strategy"] = "finetune"
                if not training.get("unfreeze_patterns") and global_unfreeze:
                    training["unfreeze_patterns"] = global_unfreeze

        elif mode in ("train_custom", "custom_adapter"):
            if comp.get("forward", {}).get("no_grad"):
                training["strategy"] = "frozen"
            else:
                training["strategy"] = "full"
        else:
            training["strategy"] = "frozen"


def _apply_strategy(comp, strategy: str, lora_cfg: dict, lora_injectors: dict):
    """Apply training strategy to one component."""

    if strategy == "frozen":
        comp.freeze()

    elif strategy == "lora":
        comp.freeze()
        comp_lora = comp.config.get("training", {}).get("lora", {})
        from backend.core.lora import LoRAInjector
        inj = LoRAInjector(
            model=comp.module,
            target_layers=comp_lora.get("target_layers", lora_cfg.get("target_layers", [])),
            target_patterns=comp_lora.get("target_patterns", lora_cfg.get("target_patterns", [])),
            rank=comp_lora.get("rank", lora_cfg.get("rank", 16)),
            alpha=comp_lora.get("alpha", lora_cfg.get("alpha", 16)),
            dropout=comp_lora.get("dropout", lora_cfg.get("dropout", 0.0)),
            init_reversed=lora_cfg.get("init_reversed", True),
        )
        inj.inject()
        lora_injectors[comp.name] = inj
        comp.trainable = True

    elif strategy == "finetune":
        comp.freeze()
        patterns = comp.config.get("training", {}).get("unfreeze_patterns", [])
        comp.unfreeze(patterns if patterns else None)

    elif strategy in ("full", "adapter"):
        # "adapter" is same as "full" — the component IS the adapter (built from spec).
        # It's trainable from scratch, all params unfrozen.
        comp.unfreeze()

    else:
        raise ValueError(f"Unknown strategy '{strategy}' for '{comp.name}'")

def _maybe_load_training_adapter(model, pipeline_cfg, base_dir):
    """
    Load training adapter LoRA if configured. Injects frozen LoRA hooks
    into the model so that the training LoRA stacks on top.
    
    Returns the LoRAInjector or None.
    """
    import os
    adapter_cfg = pipeline_cfg.get('training_adapter', {})
    adapter_path = adapter_cfg.get('path')

    if not adapter_path:
        return None

    full_path = os.path.join(base_dir, adapter_path) if base_dir else adapter_path

    if not os.path.exists(full_path):
        print(f"[Training Adapter] WARNING: File not found: {full_path}")
        return None

    from backend.core.lora import LoRAInjector
    from safetensors.torch import safe_open

    # Read metadata and keys to determine target layers and hyperparams
    with safe_open(full_path, framework="pt") as f:
        metadata = f.metadata() or {}
        adapter_keys = list(f.keys())

    # Parse target layers from saved key names
    target_layers = set()
    for key in adapter_keys:
        if key.startswith('lora.'):
            layer_name = key[5:]
            for suffix in ('.lora_A.weight', '.lora_B.weight',
                           '.lora_A.bias', '.lora_B.bias'):
                if layer_name.endswith(suffix):
                    layer_name = layer_name[:-len(suffix)]
                    break
            target_layers.add(layer_name)

    if not target_layers:
        print(f"[Training Adapter] WARNING: No LoRA layers found in {full_path}")
        return None

    injector = LoRAInjector(
        model=model,
        target_layers=list(target_layers),
        rank=int(metadata.get('rank', 16)),
        alpha=float(metadata.get('alpha', 16)),
        init_reversed=metadata.get('init_reversed', 'True') == 'True',
    )
    injector.inject()
    injector.load_weights(full_path)

    # Freeze — adapter weights are NOT trained
    for param in injector.get_trainable_parameters():
        param.requires_grad = False

    info = injector.get_info()
    print(f"[Training Adapter] Loaded from {full_path}")
    print(f"[Training Adapter] {info['num_injected_layers']} layers, "
          f"rank={info['rank']}, {info['total_trainable_params']} params (frozen)")

    return injector

# ═══════════════════════════════════════════════════════════════
# Param groups / freeze schedule
# ═══════════════════════════════════════════════════════════════

def _build_param_groups(bundle, lora_injectors: dict, config: dict) -> list:
    global_lr = config.get("optimizer", {}).get("lr", 1e-4)
    global_wd = config.get("optimizer", {}).get("weight_decay", 0.01)
    global_norm = config.get("training", {}).get("max_grad_norm", 1.0)

    param_groups = []
    for comp in bundle:
        ct = comp.config.get("training", {}) or {}
        strategy = ct.get("strategy", "frozen")
        if strategy == "frozen":
            continue
        if ct.get("freeze_epochs", 0) > 0:
            continue  # added later by trainer

        params = []
        if comp.name in lora_injectors:
            params = list(lora_injectors[comp.name].get_trainable_parameters())
        elif comp.module is not None:
            params = [p for p in comp.module.parameters() if p.requires_grad]

        if not params:
            continue

        param_groups.append({
            "params": params,
            "lr": ct.get("lr") or global_lr,
            "weight_decay": ct.get("weight_decay") or global_wd,
            "_component_name": comp.name,
            "_max_grad_norm": ct.get("max_grad_norm") or global_norm,
        })
    return param_groups


def _build_freeze_schedule(bundle) -> dict:
    schedule = {}
    for comp in bundle:
        ct = comp.config.get("training", {}) or {}
        fe = ct.get("freeze_epochs", 0)
        if fe > 0:
            schedule[comp.name] = {
                "freeze_epochs": fe,
                "unfreeze_patterns": ct.get("unfreeze_patterns"),
                "strategy": ct.get("strategy", "full"),
            }
    return schedule


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def _seed(config):
    import torch
    seed = config.get("training", {}).get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _load_dataloaders(config):
    ds = config.get("dataset", {})
    if ds.get("builtin"):
        from backend.datasets.builtin_datasets import create_builtin_dataloaders
        t, v, _ = create_builtin_dataloaders(
            ds["builtin"], batch_size=ds.get("batch_size", 64),
            num_workers=ds.get("num_workers", 2), val_split=ds.get("validation_split", 0.1))
        return t, v
    if ds.get("path"):
        from backend.datasets.image_caption import create_dataloader
        t, _ = create_dataloader(ds["path"], batch_size=ds.get("batch_size", 1))
        return t, None
    raise ValueError("No dataset. Set dataset.path or dataset.builtin.")

def _build_loss_fn(config):
    import torch
    from backend.modules.losses import create_loss
    lc = config.get("loss", {})
    name = lc.get("name", "mse")
    if name == "cross_entropy":
        class W:
            def __init__(self): self.fn = torch.nn.CrossEntropyLoss()
            def compute(self, p, t, **kw): return self.fn(p, t)
        return W()
    return create_loss(name, **lc.get("params", {}))

def _build_scheduler(config, optimizer, train_dl):
    from backend.modules.schedulers import create_scheduler
    sc = config.get("scheduler", {})
    if not sc.get("name"): return None
    total = sc.get("total_steps") or len(train_dl) * config.get("training", {}).get("epochs", 10)
    return create_scheduler(sc["name"], optimizer, warmup_steps=sc.get("warmup_steps", 100), total_steps=total)

def _build_callbacks(config):
    import os
    print(f"[CB DEBUG] TB_LOG_DIR = {TB_LOG_DIR}")
    print(f"[CB DEBUG] dir exists = {os.path.exists(TB_LOG_DIR)}")
    print(f"[CB DEBUG] tensorboard installed = ", end="")
    try:
        import tensorboard
        print(tensorboard.__version__)
    except ImportError:
        print("NOT INSTALLED")
    
    from backend.modules.callbacks import TensorBoardCallback, JSONLogCallback, ProgressCallback
    lc = config.get("logging", {})
    cbs = [ProgressCallback(print_every=lc.get("print_every", 10)), APICallback()]
    if lc.get("tensorboard", True):
        print(f"[CB DEBUG] Creating TensorBoard callback...")
        cbs.append(TensorBoardCallback(log_dir=lc.get("tensorboard_dir", TB_LOG_DIR), run_name=lc.get("run_name")))
    if lc.get("json_log", True):
        cbs.append(JSONLogCallback(log_path=lc.get("json_log_path", "./logs/training_log.json")))
    return cbs
