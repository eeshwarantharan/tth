"""One typed place for settings and file locations.

Precedence: command-line flags (applied by the CLI), then TTH_* environment variables, then
~/.config/tth/config.toml, then the defaults below. Safety settings aren't configurable here.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULT_MODEL_DIR = "qwen2.5-coder-1.5b-4bit"
DEFAULT_GGUF = "granite-4.1-3b-Q4_K_M.gguf"  # chosen by the bake-off: docs/analysis/2026-11-audit.md
_ENV_NAMES = {
    "backend": "TTH_BACKEND",
    "model_path": "TTH_MODEL",
    "adapter": "TTH_ADAPTER",
    "base_url": "TTH_BASE_URL",
    "expert_url": "TTH_EXPERT_URL",
    "expert_base_url": "TTH_EXPERT_BASE_URL",
    "expert_model": "TTH_EXPERT_MODEL",
}


class ConfigError(Exception):
    """The config file can't be used. The message says what to fix."""


def config_dir() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "tth"


def state_dir() -> Path:
    return Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state") / "tth"


def data_dir() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "tth"


@dataclass(frozen=True)
class Config:
    backend: str = "llamacpp"
    model_path: str | None = None
    adapter: str | None = None
    base_url: str = "http://127.0.0.1:8080/v1"
    expert_url: str = "http://127.0.0.1:8765"
    expert_base_url: str = "https://integrate.api.nvidia.com/v1"  # where the reference expert's model lives
    expert_model: str = "nvidia/nemotron-3.5-super-vl-120b-a12b"
    budget: int = 6


def load_config(path: Path | None = None, env: Mapping[str, str] | None = None) -> Config:
    env = os.environ if env is None else env
    path = path or config_dir() / "config.toml"
    values: dict[str, object] = {}
    if path.exists():
        try:
            values = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"{path} isn't valid TOML: {e}") from e
    known = {f.name: f for f in fields(Config)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ConfigError(f"{path} has unknown settings: {', '.join(unknown)}. Known: {', '.join(sorted(known))}")
    if "budget" in values and not (isinstance(values["budget"], int) and values["budget"] > 0):
        raise ConfigError(f"{path}: budget must be a positive whole number")
    for name in set(values) - {"budget"}:
        if not isinstance(values[name], str):
            raise ConfigError(f"{path}: {name} must be text")
    for name, variable in _ENV_NAMES.items():
        if env.get(variable):
            values[name] = env[variable]
    return Config(**values)  # type: ignore[arg-type]


def model_candidates(config: Config, cwd: Path | None = None) -> list[Path]:
    """Where the model may be, in order. A checkout's models/ folder is last, for development."""
    if config.model_path:
        return [Path(config.model_path).expanduser()]
    name = DEFAULT_GGUF if config.backend == "llamacpp" else DEFAULT_MODEL_DIR
    return [data_dir() / "models" / name, (cwd or Path.cwd()) / "models" / name]


def find_model(config: Config, cwd: Path | None = None) -> Path | None:
    return next((p for p in model_candidates(config, cwd) if p.exists()), None)


def load_env_file(path: str | Path) -> None:
    """Load KEY=VALUE lines into os.environ without overriding existing values."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_user_env() -> None:
    """API keys live in ~/.config/tth/.env. A project's own .env is the user's secret, so it's never read."""
    load_env_file(config_dir() / ".env")
