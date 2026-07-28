"""GIFT-Eval: the one benchmark all four audited models report under one protocol.

The Chronos zero-shot benchmark (``published.py``) only has published numbers for
Chronos, so it cannot validate the other three harnesses. GIFT-Eval can: all 97
of its tasks carry published MASE and CRPS for Chronos, TimesFM, Moirai and
Lag-Llama alike.

Sizing rules transcribed from the reference implementation
(``src/gift_eval/data.py`` in SalesforceAIResearch/gift-eval). The windowing is
reimplemented here rather than taken from gluonts, for the same reason the
Chronos metrics were: running the reference validates the reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config

HF_REPO = "Salesforce/GiftEval"
LOCAL_DIR = config.DATA_DIR / "gift_eval"
RESULTS_URL = (
    "https://raw.githubusercontent.com/SalesforceAIResearch/gift-eval/"
    "main/results/{model}/all_results.csv"
)

# Published result directory name per audited model key.
RESULT_DIRS = {
    "chronos-base": "chronos_base",
    "timesfm-200m": "timesfm",
    "moirai-base": "Moirai_base",
    "lag-llama": "Lag-Llama",
}

TEST_SPLIT = 0.1
MAX_WINDOW = 20

TERM_MULTIPLIER = {"short": 1, "medium": 10, "long": 15}

PRED_LENGTH_MAP = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
M4_PRED_LENGTH_MAP = {"A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}

# The reference normalises modern pandas aliases back to the legacy ones its
# lookup tables are keyed on.
_LEGACY_FREQ = {
    "Y": "A",
    "YE": "A",
    "QE": "Q",
    "ME": "M",
    "h": "H",
    "min": "T",
    "s": "S",
    "us": "U",
}


def legacy_freq(freq: str) -> str:
    base = freq.split("-")[0]
    base = "".join(ch for ch in base if not ch.isdigit())
    return _LEGACY_FREQ.get(base, base)


@dataclass(frozen=True)
class GiftTask:
    """One GIFT-Eval task, e.g. ``ett1/H/short``."""

    dataset: str
    freq: str
    term: str

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.freq}/{self.term}"

    @property
    def path(self) -> str:
        return f"{self.dataset}/{self.freq}"


def parse_task(key: str) -> GiftTask:
    dataset, freq, term = key.split("/")
    return GiftTask(dataset=dataset, freq=freq, term=term)


def download_task(task: GiftTask) -> str:
    """Fetch one task's files into ``data/gift_eval``. Returns the local root.

    Downloads to a real directory rather than the HF cache: the cache uses
    symlinks, which need Developer Mode on Windows and fail with WinError 1314
    without it.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        allow_patterns=[f"{task.path}/*"],
        local_dir=str(LOCAL_DIR),
    )


def prediction_length(task: GiftTask, freq: str) -> int:
    """Horizon for a task, from its frequency and term."""
    base = legacy_freq(freq)
    table = M4_PRED_LENGTH_MAP if "m4" in task.dataset else PRED_LENGTH_MAP
    return TERM_MULTIPLIER[task.term] * table[base]


def n_windows(task: GiftTask, min_series_length: int, horizon: int) -> int:
    """Rolling evaluation windows per series. M4 tasks always use exactly one."""
    if "m4" in task.dataset:
        return 1
    w = math.ceil(TEST_SPLIT * min_series_length / horizon)
    return min(max(1, w), MAX_WINDOW)


@dataclass
class GiftWindow:
    past: np.ndarray
    target: np.ndarray
    freq: str
    series_index: int
    window_index: int


def rolling_windows(
    values: np.ndarray, horizon: int, windows: int, freq: str, series_index: int
) -> list[GiftWindow]:
    """Generate the rolling test instances for one univariate series.

    Window ``i`` forecasts the slice starting ``horizon * (windows - i)`` from the
    end. The final window ends at the series end, where the end index computes to
    zero - the same slicing trap that silently emptied every window in the Chronos
    loader, so it is handled explicitly here too.
    """
    values = np.asarray(values, dtype=float)
    total = horizon * windows
    out: list[GiftWindow] = []
    for i in range(windows):
        split_at = -total + i * horizon
        end = split_at + horizon
        past = values[:split_at]
        target = values[split_at:] if end >= 0 else values[split_at:end]
        target = target[:horizon]
        if len(past) == 0 or len(target) < horizon:
            continue
        out.append(
            GiftWindow(
                past=past,
                target=target,
                freq=freq,
                series_index=series_index,
                window_index=i,
            )
        )
    return out


def load_task(key: str) -> tuple[list[GiftWindow], int]:
    """Load one GIFT-Eval task and return ``(windows, horizon)``.

    Multivariate targets are split into independent univariate series, matching
    the reference's ``MultivariateToUnivariate``.
    """
    import datasets

    task = parse_task(key)
    root = download_task(task)
    ds = datasets.load_from_disk(f"{root}/{task.path}").with_format("numpy")

    freq = str(ds[0]["freq"])
    horizon = prediction_length(task, freq)

    series: list[np.ndarray] = []
    for row in ds:
        target = np.asarray(row["target"])
        if target.ndim == 1:
            series.append(target)
        else:
            series.extend(target[d] for d in range(target.shape[0]))

    min_length = min(len(s) for s in series)
    windows = n_windows(task, min_length, horizon)

    out: list[GiftWindow] = []
    for index, values in enumerate(series):
        out.extend(rolling_windows(values, horizon, windows, freq, index))
    return out, horizon


def load_published_results(model_key: str) -> pd.DataFrame:
    """Published GIFT-Eval results for an audited model, indexed by task key."""
    directory = RESULT_DIRS[model_key]
    frame = pd.read_csv(RESULTS_URL.format(model=directory))
    return frame.set_index("dataset")


def published_mase(model_key: str, task_key: str) -> float:
    return float(load_published_results(model_key).loc[task_key, "eval_metrics/MASE[0.5]"])
