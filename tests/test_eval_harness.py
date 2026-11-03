"""Results must be traceable, and baselines must behave as described."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import manifest
from evals.baselines import AlwaysEscalatePolicy, RandomPolicy, SilentPolicy
from evals.report import find_arms
from tth.agent.loop import run_episode
from tth_gym.core.mcp_env import McpEnvironment
from tth_gym.envs.chess.env import ChessEnv, EnvConfig
from tth_gym.envs.chess.env import reward as chess_reward
from tth_gym.envs.chess.tasks import ChessTask
from tth_gym.envs.shell.env import ShellEnv, reward
from tth_gym.envs.shell.tasks import ShellTask

ROOT = Path(__file__).resolve().parents[1]
SHELL = ShellTask(id="t", seed=1, family="count_py", request="count .py files", decision="stage",
                  check={"type": "number", "value": 5, "tol": 0})  # fmt: skip
CHESS = ChessTask(id="t", kind="check", fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                  question="Is the side to move in check?", answer_type="bool", answer="no")  # fmt: skip


def _shell(policy: object) -> ShellEnv:
    env = ShellEnv(SHELL, seed=1, p_timeout=0)
    run_episode(env, policy, budget=6)  # type: ignore[arg-type]
    return env


def test_silence_scores_the_floor_on_shell() -> None:
    env = _shell(SilentPolicy())
    assert reward(env).total <= -1.0
    assert not reward(env).correct
    env.close()


def test_always_escalating_is_recognised() -> None:
    env = _shell(AlwaysEscalatePolicy())
    assert env.st.decision == "escalate"
    env.close()


def test_the_random_baseline_is_repeatable() -> None:
    def run() -> list:
        env = _shell(RandomPolicy(seed=3))
        tools = [r.tool for r in env.st.records]
        env.close()
        return tools

    assert run() == run()


def test_baselines_run_through_mcp_on_chess() -> None:
    env = McpEnvironment(ChessEnv(CHESS, seed=2, config=EnvConfig()))
    run_episode(env, RandomPolicy(seed=1), budget=6)
    env.close()
    assert chess_reward(CHESS, env.st).total <= 1.0


def _write_result(results: Path, arm: str) -> None:
    results.mkdir(exist_ok=True)
    (results / f"{arm}__shell.jsonl").write_text("{}\n")


def test_report_refuses_results_with_no_manifest(tmp_path: Path) -> None:
    _write_result(tmp_path, "mystery")
    with pytest.raises(SystemExit, match="Refusing to report mystery"):
        find_arms(tmp_path)


def test_report_refuses_an_incomplete_manifest(tmp_path: Path) -> None:
    _write_result(tmp_path, "partial")
    (tmp_path / "partial__manifest.json").write_text(json.dumps({"arm": "partial"}))
    with pytest.raises(SystemExit, match="manifest is missing"):
        find_arms(tmp_path)


def test_a_manifest_records_what_produced_the_results(tmp_path: Path) -> None:
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text("x\n")
    adapter = tmp_path / "a.safetensors"
    adapter.write_bytes(b"weights")
    m = manifest.build(
        "arm",
        "An arm",
        "mlx",
        6,
        {"shell": task_file},
        adapter=str(adapter),
        adapter_sha256=manifest.adapter_hash(str(adapter)),
    )
    manifest.save(tmp_path, m)
    loaded = manifest.load(tmp_path, "arm")
    assert len(loaded["git_sha"]) == 40
    assert loaded["suites"]["shell"] == manifest.sha256(task_file)
    assert loaded["adapter_sha256"]
    assert loaded["finished"]


def test_a_changed_adapter_changes_its_hash(tmp_path: Path) -> None:
    f = tmp_path / "a.safetensors"
    f.write_bytes(b"one")
    first = manifest.adapter_hash(str(f))
    f.write_bytes(b"two")
    assert manifest.adapter_hash(str(f)) != first


def test_committed_results_all_have_manifests() -> None:
    """Results from before manifests existed live in evals/results-v0.1, and are labelled as such."""
    results = ROOT / "evals" / "results"
    if results.exists():
        find_arms(results)
    assert (ROOT / "evals" / "results-v0.1" / "README.md").exists()


def test_committed_results_contain_no_local_username(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evals.run import scrub

    monkeypatch.setattr("getpass.getuser", lambda: "someperson")
    f = tmp_path / "t.jsonl"
    f.write_text(f"-rw-r--r-- 1 someperson staff 5 a.txt {Path.home()}/x\n")
    scrub(f)
    text = f.read_text()
    assert "someperson" not in text
    assert str(Path.home()) not in text
