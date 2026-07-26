"""The pre-registration is executable. These tests are what make it binding."""

import datetime as dt

from tsfm_audit import config


def test_cutoff_postdates_every_audited_model():
    """The fresh benchmark must start after the last checkpoint was released."""
    assert config.FRESH_BENCHMARK_START > config.latest_model_release()


def test_cutoff_has_a_real_buffer():
    """Release date is not data cutoff, so a token one-day gap is not enough."""
    gap = config.FRESH_BENCHMARK_START - config.latest_model_release()
    assert gap >= dt.timedelta(days=180)


def test_derived_seeds_are_deterministic_and_distinct():
    assert config.derive_seed("a", 1) == config.derive_seed("a", 1)
    assert config.derive_seed("a", 1) != config.derive_seed("a", 2)


def test_model_keys_are_unique():
    keys = [m.key for m in config.AUDITED_MODELS]
    assert len(keys) == len(set(keys))


def test_at_least_three_independently_trained_models():
    """The brief requires three; distinct organisations is the harder bar."""
    assert len({m.org for m in config.AUDITED_MODELS}) >= 3


def test_protocol_values_are_pre_registered():
    p = config.PROTOCOL
    assert p.n_surrogates == 100
    assert p.surrogate_families == ("iaaft", "block_bootstrap")
    assert p.fdr_q == 0.10
    assert p.firing_rate_threshold == 0.20
    assert p.negative_control_max_firing_rate < p.firing_rate_threshold


def test_detection_floor_is_unset_until_calibration():
    """Phase 4 fills this in. Setting it early would be picking the answer."""
    assert config.PROTOCOL.detection_floor is None


def test_fresh_window_end_respects_api_lag():
    today = dt.date(2026, 7, 26)
    assert config.fresh_benchmark_end(today) == dt.date(2026, 7, 19)
