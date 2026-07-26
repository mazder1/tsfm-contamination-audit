"""The .env loader must never override real environment variables."""

import os

from tsfm_audit.env import load_dotenv, parse_dotenv


def test_parses_plain_pairs():
    assert parse_dotenv("A=1\nB=two\n") == {"A": "1", "B": "two"}


def test_ignores_comments_and_blanks():
    assert parse_dotenv("# note\n\nA=1\n") == {"A": "1"}


def test_strips_export_prefix_and_quotes():
    assert parse_dotenv("export A=\"x\"\nB='y'\n") == {"A": "x", "B": "y"}


def test_keeps_equals_signs_inside_values():
    assert parse_dotenv("TOKEN=ab==cd\n") == {"TOKEN": "ab==cd"}


def test_loads_into_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("TSFM_TEST_KEY=from_file\n", encoding="utf-8")
    monkeypatch.delenv("TSFM_TEST_KEY", raising=False)

    assert load_dotenv(env_file) == ["TSFM_TEST_KEY"]
    assert os.environ["TSFM_TEST_KEY"] == "from_file"


def test_real_environment_wins_over_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("TSFM_TEST_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("TSFM_TEST_KEY", "from_env")

    assert load_dotenv(env_file) == []
    assert os.environ["TSFM_TEST_KEY"] == "from_env"


def test_missing_file_is_not_an_error(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == []
