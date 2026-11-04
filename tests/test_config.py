"""Settings and install paths. These broke `pip install tth` in v0.1."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tth.core.config import Config, ConfigError, find_model, load_config, load_user_env, model_candidates

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_without_a_config_file(tmp_path: Path) -> None:
    assert load_config(tmp_path / "missing.toml", env={}) == Config()


def test_file_values_are_used(tmp_path: Path) -> None:
    f = tmp_path / "config.toml"
    f.write_text('backend = "openai"\nbudget = 4\n')
    c = load_config(f, env={})
    assert (c.backend, c.budget) == ("openai", 4)


def test_environment_beats_file(tmp_path: Path) -> None:
    f = tmp_path / "config.toml"
    f.write_text('backend = "openai"\n')
    assert load_config(f, env={"TTH_BACKEND": "mlx"}).backend == "mlx"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not toml =", "isn't valid TOML"),
        ('sandbox = "off"\n', "unknown settings: sandbox"),
        ("budget = 0\n", "positive whole number"),
        ('budget = "many"\n', "positive whole number"),
        ("backend = 3\n", "must be text"),
    ],
)
def test_bad_config_explains_itself(tmp_path: Path, text: str, message: str) -> None:
    f = tmp_path / "config.toml"
    f.write_text(text)
    with pytest.raises(ConfigError, match=message):
        load_config(f, env={})


def test_configured_model_path_is_the_only_candidate(tmp_path: Path) -> None:
    c = Config(model_path=str(tmp_path / "m"))
    assert model_candidates(c) == [tmp_path / "m"]
    assert find_model(c) is None
    (tmp_path / "m").mkdir()
    assert find_model(c) == tmp_path / "m"


def test_model_is_found_in_the_user_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    model = tmp_path / "tth" / "models" / "qwen2.5-coder-1.5b-4bit"
    model.mkdir(parents=True)
    assert find_model(Config(backend="mlx"), cwd=tmp_path / "elsewhere") == model


def test_a_projects_own_env_file_is_never_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text("PROJECT_SECRET_FOR_TEST=hunter2\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.delenv("PROJECT_SECRET_FOR_TEST", raising=False)
    load_user_env()
    assert "PROJECT_SECRET_FOR_TEST" not in __import__("os").environ


def test_no_module_locates_files_relative_to_its_own_source() -> None:
    """`Path(__file__).parents[n]` points into site-packages once installed from a wheel."""
    offenders = []
    for f in (ROOT / "packages" / "tth" / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Name) and node.id == "__file__":
                offenders.append(f"{f.name}:{node.lineno}")
    assert not offenders, offenders


def test_the_default_model_is_the_bake_off_winner() -> None:
    assert Config().backend == "llamacpp"
    assert "granite-4.1-3b" in model_candidates(Config())[0].name
