"""Routes for training lifecycle: start, stop, status, streaming, logs."""

import asyncio
import json
import os

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from datetime import datetime

from backend.api.models import TrainRequest, SampleRequest, SaveRequest
from backend.api.state.training_state import training_state, stop_event
from backend.api.dependencies import get_config_manager
from backend.api.tasks.train import run_unified_training

router = APIRouter(prefix="/api/training", tags=["training"])
config_manager = get_config_manager()


@router.get("/status")
async def get_training_status():
    return {k: v for k, v in training_state.items() if not k.startswith("_")}


@router.post("/start")
async def start_training(req: TrainRequest, bg: BackgroundTasks):
    if training_state["status"] == "training":
        raise HTTPException(409, "Training in progress")

    config = _resolve_config(req)

    stop_event.clear()
    training_state.update({
        "status": "training",
        "current_step": 0,
        "current_epoch": 0,
        "total_epochs": config.get("training", {}).get("epochs", 10),
        "loss": 0.0,
        "smoothed_loss": 0.0,
        "val_loss": None,
        "accuracy": None,
        "val_accuracy": None,
        "lr": config.get("optimizer", {}).get("lr", 1e-4),
        "start_time": datetime.now().isoformat(),
        "error": None,
        "config_name": req.config_name,
        "mode": req.mode,
        "run_dir": None,
        "loss_history": [],
        "val_loss_history": [],
        "lr_history": [],
    })
    bg.add_task(run_unified_training, config, req.mode)
    return {"status": "started", "mode": req.mode}

@router.post("/stop")
async def stop_training():
    if training_state["status"] != "training":
        raise HTTPException(400, "Not currently training")
    stop_event.set()
    training_state["status"] = "stopping"
    
    # Optionally also poke the trainer directly as a fallback
    trainer = training_state.get("_trainer_ref")
    if trainer and hasattr(trainer, '_save_requested'):
        pass  # stop_event is sufficient, trainer checks it each step
    
    return {"status": "stopping"}

@router.get("/stream")
async def training_stream():
    async def generate():
        last_step = -1
        while True:
            if training_state["current_step"] != last_step or training_state["status"] != "training":
                yield f"data: {json.dumps(training_state, default=str)}\n\n"
                last_step = training_state["current_step"]
            if training_state["status"] in ("completed", "error", "idle"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(generate(), media_type="text/event-stream")

@router.post("/sample")
async def request_sample(req: SampleRequest = None):
    """Request sample generation at next step boundary."""
    if training_state["status"] != "training":
        raise HTTPException(400, "Not currently training")

    trainer = training_state.get("_trainer_ref")
    if trainer is None:
        raise HTTPException(400, "Trainer not initialized")

    request_data = None
    if req:
        request_data = req.dict(exclude_none=True)

    trainer.request_sample(request_data)
    return {"status": "sample_requested", "message": "Will generate at next step boundary"}


@router.post("/save_now")
async def request_save():
    """Request checkpoint save at next step boundary."""
    if training_state["status"] != "training":
        raise HTTPException(400, "Not currently training")

    trainer = training_state.get("_trainer_ref")
    if trainer is None:
        raise HTTPException(400, "Trainer not initialized")

    trainer.request_save()
    return {"status": "save_requested", "message": "Will save at next step boundary"}


@router.post("/resume")
async def resume_training(req: TrainRequest, bg: BackgroundTasks):
    """Resume training from a checkpoint."""
    if training_state["status"] == "training":
        raise HTTPException(409, "Training in progress")

    config = _resolve_config(req)
    checkpoint_path = req.checkpoint_path  # Add this field to TrainRequest

    if not checkpoint_path:
        # Try to find latest checkpoint in the run dir
        run_dir = training_state.get("run_dir")
        if run_dir:
            ckpt_dir = os.path.join(run_dir, 'checkpoints')
            if os.path.exists(ckpt_dir):
                pts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
                if pts:
                    checkpoint_path = os.path.join(ckpt_dir, pts[-1])

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise HTTPException(400, "No checkpoint found to resume from")

    stop_event.clear()
    training_state.update({
        "status": "training",
        "error": None,
        "mode": req.mode,
    })

    bg.add_task(run_unified_training, config, req.mode,
                resume_checkpoint=checkpoint_path)
    return {"status": "resuming", "checkpoint": checkpoint_path}


@router.get("/samples")
async def get_samples():
    """Get list of generated samples."""
    trainer = training_state.get("_trainer_ref")
    if trainer is None or trainer.sampler is None:
        return {"samples": []}

    return {"samples": trainer.sampler.sample_log}


@router.get("/samples/latest")
async def get_latest_samples():
    """Get the most recent sample images."""
    trainer = training_state.get("_trainer_ref")
    if trainer is None or trainer.sampler is None:
        return {"sample": None}

    latest = trainer.sampler.get_latest_samples()
    return {"sample": latest}


@router.get("/pipelines")
async def list_available_pipelines():
    """List registered diffusion pipelines."""
    from backend.pipelines.registry import list_pipelines
    return {"pipelines": list_pipelines()}


@router.get("/pipelines/{name}/presets")
async def get_pipeline_presets(name: str):
    """Get LoRA presets for a pipeline."""
    if name == "zimage_turbo":
        from backend.pipelines.zimage_turbo_pipeline import ZIMAGE_TURBO_LORA_PRESETS
        return {"presets": ZIMAGE_TURBO_LORA_PRESETS}
    return {"presets": {}}


@router.get("/logs")
async def get_training_logs():
    path = "./logs/training_log.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"steps": [], "epochs": []}


# ── Helpers ──

def _resolve_config(req: TrainRequest) -> dict:
    if req.config_name:
        try:
            return config_manager.load(req.config_name)
        except FileNotFoundError:
            raise HTTPException(404)
    elif req.config:
        return req.config
    raise HTTPException(400, "Provide config_name or config")
