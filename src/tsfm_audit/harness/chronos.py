"""Chronos behind the audit's one forecasting interface.

The checkpoint is always loaded at the SHA pinned in ``model_revisions.lock.json``.
Loading from ``main`` would let a checkpoint change underneath a result, which is
exactly the kind of silent drift this project exists to detect in other people's
work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

from .. import config

# The nine levels used by the published Chronos evaluation.
QUANTILE_LEVELS: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def pinned_revision(model_key: str) -> tuple[str, str]:
    """Return ``(repo_id, revision)`` for an audited model, or raise if unpinned."""
    lock = json.loads(config.MODEL_REVISION_LOCK.read_text(encoding="utf-8"))
    entry = lock["models"][model_key]
    revision = entry.get("revision")
    if not revision:
        raise RuntimeError(
            f"{model_key} has no pinned revision in {config.MODEL_REVISION_LOCK.name}; "
            "run scripts/pin_model_revisions.py before evaluating"
        )
    return entry["repo_id"], revision


@dataclass
class ChronosForecaster:
    """Wraps a pinned Chronos pipeline and produces quantile forecasts."""

    model_key: str = "chronos-base"
    device: str = "cpu"
    dtype: str = "float32"
    num_samples: int = 20
    batch_size: int = 32

    def __post_init__(self) -> None:
        from chronos import BaseChronosPipeline

        self.repo_id, self.revision = pinned_revision(self.model_key)
        self.pipeline = BaseChronosPipeline.from_pretrained(
            self.repo_id,
            revision=self.revision,
            device_map=self.device,
            torch_dtype=getattr(torch, self.dtype),
        )

    def predict_quantiles(
        self,
        histories: list[np.ndarray],
        prediction_length: int,
        quantile_levels: list[float] | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Quantile forecasts for a list of histories, shape ``(n, horizon, n_levels)``.

        ``seed`` makes the sampling reproducible. Chronos draws trajectories, so
        without a fixed seed the same input gives different scores on every run -
        which would make any reproduction tolerance meaningless.
        """
        levels = quantile_levels or QUANTILE_LEVELS
        if seed is not None:
            torch.manual_seed(seed)

        outputs: list[np.ndarray] = []
        for start in range(0, len(histories), self.batch_size):
            batch = histories[start : start + self.batch_size]
            context = [torch.tensor(np.asarray(h, dtype=np.float32)) for h in batch]
            quantiles, _ = self.pipeline.predict_quantiles(
                inputs=context,
                prediction_length=prediction_length,
                quantile_levels=levels,
                num_samples=self.num_samples,
            )
            outputs.append(np.asarray(quantiles, dtype=float))
        return np.concatenate(outputs, axis=0)
