"""
Setup experiment tracking using trackio (W&B API-compatible, local-first).
"""

import os
import logging
from transformers import TrainerCallback, TrainerState, TrainerControl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    import trackio
    TRACKIO_AVAILABLE = True
except ImportError:
    logger.warning("trackio not found. Run `pip install trackio`. Tracking will be disabled.")
    trackio = None
    TRACKIO_AVAILABLE = False

class TrackioCallback(TrainerCallback):
    """
    HuggingFace TrainerCallback that logs metrics to trackio.
    Suitable for SFTTrainer during fine-tuning.
    """
    def __init__(self, project_name="sft-coding-agent"):
        self.project_name = project_name
        
    def on_init_end(self, args, state, control, **kwargs):
        """Log hyperparams at init."""
        if TRACKIO_AVAILABLE:
            trackio.init(project=self.project_name, config=args.to_dict())
            logger.info(f"Initialized trackio for experiment tracking in project '{self.project_name}'.")
            
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Log metrics on each log event."""
        if TRACKIO_AVAILABLE and logs is not None:
            metrics = {
                "step": state.global_step,
                "epoch": state.epoch,
            }
            if "loss" in logs:
                metrics["train_loss"] = logs["loss"]
            if "eval_loss" in logs:
                metrics["eval_loss"] = logs["eval_loss"]
            if "learning_rate" in logs:
                metrics["learning_rate"] = logs["learning_rate"]
                
            try:
                import torch
                if torch.cuda.is_available():
                    metrics["gpu_memory_allocated"] = torch.cuda.memory_allocated()
                    metrics["gpu_memory_reserved"] = torch.cuda.memory_reserved()
            except ImportError:
                pass
                
            trackio.log(metrics)

def init_tracking(config: dict):
    """
    Utility function to initialize tracking manually.
    
    Dashboard viewing instructions:
    To view the local trackio dashboard, run `trackio ui` in your terminal.
    For HuggingFace Spaces syncing, configure your trackio environment variables 
    (e.g., TRACKIO_SYNC_URL) pointing to your Space.
    """
    if TRACKIO_AVAILABLE:
        project = config.get("project_name", "sft-coding-agent")
        trackio.init(project=project, config=config)
        logger.info("Trackio initialized. Dashboard can be viewed locally via `trackio ui`.")
        
def finish_tracking():
    """Utility function to finish tracking session."""
    if TRACKIO_AVAILABLE:
        trackio.finish()
        logger.info("Trackio tracking finished.")
