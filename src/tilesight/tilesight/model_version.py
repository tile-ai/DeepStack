"""Select corrected resource accounting or the frozen AE compatibility path."""

import os


MODEL_VERSIONS = ("paper_legacy", "current_corrected")


def get_model_version() -> str:
    """Default library callers to corrected cache and SMEM accounting."""
    version = os.environ.get("TILESIGHT_MODEL_VERSION", "current_corrected")
    if version not in MODEL_VERSIONS:
        raise ValueError(
            f"unknown TILESIGHT_MODEL_VERSION {version!r}; expected {MODEL_VERSIONS}"
        )
    return version
