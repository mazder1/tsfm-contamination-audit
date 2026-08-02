"""The near-duplicate matcher, held to its pre-registered verification gates."""

import numpy as np

from tsfm_audit.probes.duplicates import MATCH_RMS, find_matches, mass_distance


def _series(n: int = 2000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 10 + 3 * np.sin(2 * np.pi * t / 24) + np.cumsum(rng.normal(0, 0.3, n))


def test_planted_exact_copy_fires():
    corpus = _series(seed=1)
    query = corpus[700:1100].copy()
    hits = [m for m in find_matches(query, corpus) if m.kind == "match"]
    assert hits and hits[0].offset == 700 and hits[0].rms < 1e-6


def test_planted_rescaled_shifted_copy_fires():
    # A mirrored dataset most often differs by units and offset. Kilowatts to
    # megawatts plus a baseline shift must still be found.
    corpus = _series(seed=2)
    query = corpus[300:800] * 1000.0 - 47.5
    hits = [m for m in find_matches(query, corpus) if m.kind == "match"]
    assert hits and hits[0].offset == 300


def test_small_jitter_still_fires():
    corpus = _series(seed=3)
    rng = np.random.default_rng(0)
    query = corpus[500:900] + rng.normal(0, 0.001 * corpus.std(), 400)
    hits = [m for m in find_matches(query, corpus) if m.kind == "match"]
    assert hits and hits[0].offset == 500


def test_unrelated_series_does_not_fire():
    query = _series(seed=10)[:400]
    corpus = _series(seed=11)
    assert [m for m in find_matches(query, corpus) if m.kind == "match"] == []


def test_profile_is_shift_and_scale_invariant():
    corpus = _series(seed=4)
    a = mass_distance(corpus[100:300], corpus)
    b = mass_distance(corpus[100:300] * 5.0 + 12.0, corpus)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_constant_query_is_rejected():
    assert find_matches(np.ones(100), _series()) == []


def test_match_threshold_is_the_preregistered_one():
    assert MATCH_RMS == 0.05
