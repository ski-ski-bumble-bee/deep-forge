"""Shared helpers for background training tasks."""

from typing import Any, Dict, List, Optional

import torch
from torch import nn

from backend.api.state.training_state import training_state


class APICallback:
    """Pushes trainer progress into the shared training_state dict."""

    def on_train_start(self, state):
        training_state["run_dir"] = state.get("run_dir")

    def on_train_end(self, state):
        pass

    def on_epoch_start(self, epoch, state):
        pass

    def on_step_start(self, step, state):
        pass

    def on_step_end(self, step, loss, state):
        training_state["current_step"] = step
        training_state["loss"] = loss
        training_state["smoothed_loss"] = state.get("smoothed_loss", loss)
        training_state["lr"] = state.get("lr", 0)
        training_state["accuracy"] = state.get("running_accuracy")
        training_state["loss_history"].append({
            "step": step,
            "loss": loss,
            "smoothed": state.get("smoothed_loss", loss),
        })

    def on_epoch_end(self, epoch, state):
        training_state["current_epoch"] = epoch
        if "val_loss" in state:
            training_state["val_loss"] = state["val_loss"]
            training_state["val_loss_history"].append({
                "epoch": epoch,
                "val_loss": state["val_loss"],
            })
        val_metrics = state.get("val_metrics")
        if isinstance(val_metrics, dict):
            training_state["val_accuracy"] = val_metrics.get("accuracy")


class BundleWrapper(nn.Module):
    """
    Wraps a ComponentBundle as a single nn.Module.

    Handles:
      - .to(device), .train(), .eval(), .parameters() — via registered submodules
      - forward(*args) — simple call to first/primary component
      - forward_pass(batch) — orchestrated pipeline execution in component order

    Pipeline execution:
      Components run in execution_order. Each component can:
        - Read from batch keys (input_keys)
        - Write to batch keys (output_keys)
        - Run in no_grad context (forward.no_grad)
        - Cache its output (forward.cache_output)
        - Have an attached adapter (training.strategy: "adapter")

    For single-component models, forward_pass just calls forward normally.
    """

    def __init__(self, bundle):
        super().__init__()
        self._bundle = bundle

        # Register as submodules for .to() / .train() / .parameters()
        for comp in bundle:
            if comp.module is not None:
                attr = f"_comp_{comp.name.replace('.', '_').replace('-', '_')}"
                setattr(self, attr, comp.module)

    def forward(self, *args, **kwargs):
        """Simple forward — routes to primary component."""
        # Single component bundle — just call it
        if len(self._bundle) == 1:
            comp = list(self._bundle.components.values())[0]
            if comp.module:
                return comp.module(*args, **kwargs)

        # Multi-component: try first denoiser, then first available
        denoisers = self._bundle.get_by_role("denoiser")
        if denoisers and denoisers[0].module:
            return denoisers[0].module(*args, **kwargs)

        for comp in self._bundle:
            if comp.module:
                return comp.module(*args, **kwargs)

        raise RuntimeError("No component module available for forward pass")

    def forward_pass(self, batch: Dict[str, Any]) -> Any:
        """
        Pipeline-aware forward. Executes components in order.

        For single-component: extracts input, calls module, returns result.

        For multi-component: runs each component in execution_order.
        Each component reads its input from batch[input_key] and
        writes output to batch[output_key].

        The LAST trainable component's output is used for loss.

        Custom pipeline override:
          Set wrapper._custom_forward_pass = fn(bundle, batch) -> dict
          for full control over complex architectures.
        """
        # Custom pipeline override
        if hasattr(self, '_custom_forward_pass') and self._custom_forward_pass is not None:
            return self._custom_forward_pass(self._bundle, batch)

        # Single component — simple path
        if len(self._bundle) == 1:
            return self._single_component_forward(batch)

        # Multi-component pipeline
        return self._pipeline_forward(batch)

    def _single_component_forward(self, batch):
        """Forward for single-component bundles."""
        comp = list(self._bundle.components.values())[0]
        inp = batch.get('input', batch.get('pixel_values'))
        tgt = batch.get('target', batch.get('labels'))

        if inp is None:
            raise ValueError(f"No 'input' in batch. Keys: {list(batch.keys())}")

        output = comp.module(inp)

        if isinstance(output, dict) and 'loss' in output:
            return output
        if isinstance(output, tuple) and len(output) >= 1:
            # Some models return (loss, logits) or (output, hidden)
            return output
        return {'predictions': output}

    def _pipeline_forward(self, batch):
        """
        Multi-component pipeline execution.

        Each component config can specify:
          forward.input_key:  which batch key to read (default: "input" or previous output)
          forward.output_key: which batch key to write (default: component name)
          forward.no_grad:    run without gradients
          forward.cache_output: cache and reuse across steps

        The pipeline builds up the batch dict as it goes.
        Last trainable component's output is used as the return value.
        """
        pipeline_data = dict(batch)  # copy so we don't mutate original
        last_output = None
        last_trainable_output = None

        for comp in self._bundle:  # iterates in execution_order
            if comp.module is None:
                continue

            fwd_cfg = comp.config.get('forward', {}) or {}

            # Check cache
            if fwd_cfg.get('cache_output') and comp._cached_output is not None:
                pipeline_data[comp.name] = comp._cached_output
                continue

            # Determine input for this component
            input_key = fwd_cfg.get('input_key')
            if input_key:
                comp_input = pipeline_data.get(input_key)
            elif last_output is not None:
                comp_input = last_output
            else:
                comp_input = pipeline_data.get('input', pipeline_data.get('pixel_values'))

            if comp_input is None:
                continue  # skip components with no available input

            # Execute
            if fwd_cfg.get('no_grad'):
                with torch.no_grad():
                    output = comp.module(comp_input)
            else:
                output = comp.module(comp_input)

            # Unwrap common return types
            if isinstance(output, tuple):
                output = output[0]

            # Store in pipeline
            output_key = fwd_cfg.get('output_key', comp.name)
            pipeline_data[output_key] = output
            last_output = output

            # Cache if requested
            if fwd_cfg.get('cache_output'):
                comp._cached_output = output.detach() if isinstance(output, torch.Tensor) else output

            if comp.trainable:
                last_trainable_output = output

        # Return the last trainable component's output (or last output)
        result = last_trainable_output if last_trainable_output is not None else last_output

        if result is None:
            raise RuntimeError("Pipeline produced no output")

        if isinstance(result, dict) and 'loss' in result:
            return result

        return {'predictions': result, '_pipeline_data': pipeline_data}
