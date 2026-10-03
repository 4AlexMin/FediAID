"""Runtime configuration utilities for GPU assignment and HuggingFace endpoint.


Design goals:
- Minimal external dependencies (uses subprocess and optionally torch if
  installed).
- Safe defaults: if no GPU is available or selection fails, falls back to CPU.
"""
from __future__ import annotations

import os

DEFAULT_ENCODER = "microsoft/deberta-v3-base"
DEFAULT_ENCODER_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"


def resolve_encoder_revision(model_name: str | None, revision: str | None = None) -> str | None:
    if revision:
        return revision
    if model_name == DEFAULT_ENCODER:
        return DEFAULT_ENCODER_REVISION
    return None



def set_hf_mirror(url: str = "https://hf-mirror.com") -> None:
    """Set the HuggingFace mirror endpoint for the current process.

    Example:
        set_hf_mirror("https://hf-mirror.com")
    """
    if not os.environ.get("HF_ENDPOINT"):
        if os.environ.get("GAI_HF_MIRROR"):
            os.environ["HF_ENDPOINT"] = os.environ["GAI_HF_MIRROR"]
        else:    
            os.environ["HF_ENDPOINT"] = url
    
    # Optional but strongly recommended for stability
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    
    print(f"Using HF endpoint: {os.environ['HF_ENDPOINT']}")
    
    


def get_default_device() -> str:
    """Return the default device string following the user's manual GPU-reservation workflow.

    This helper implements the simple rule:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    It's intentionally lightweight so callers can manually set
    ``CUDA_VISIBLE_DEVICES`` externally (for example: ``CUDA_VISIBLE_DEVICES=3 python ...``)
    and then call this helper in their scripts to get the device string.
    """
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


__all__ = [
    "DEFAULT_ENCODER",
    "DEFAULT_ENCODER_REVISION",
    "resolve_encoder_revision",
    "set_hf_mirror",
    "get_default_device",
]
