"""How much memory does a model really take while it's doing the agent's job?

One number can flatter a model. Peak RSS swings with how many mapped weights the OS has evicted: we saw 634 MB and
1,280 MB for one file. Physical footprint leaves the mapped weights out, so alone it looks far too good. The model
needs its weights to run, so the server's number is the larger of peak RSS and file size plus footprint. The target
also covers the CLI process, which is alive for the whole request. Each setting gets its own server, and the model
answers a prompt as long as a real late-episode turn.

uv run python -m evals.diagnostics.memory <out.jsonl> model.gguf [model.gguf ...]
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from tth.agent.prompts import SHELL_SYSTEM
from tth.inference.llamacpp import server_args

PORT = 8799
TARGET_MB = 1000
FILLER = "-rw-r--r-- 1 user staff 18422 Nov 4 18:17 sales_2026_q3_final.csv\n"
SETTINGS = [
    {"ctx": 8192, "ubatch": 512, "checkpoints": 8},
    {"ctx": 4096, "ubatch": 512, "checkpoints": 8},
    {"ctx": 4096, "ubatch": 256, "checkpoints": 0},
    {"ctx": 3072, "ubatch": 256, "checkpoints": 0},
    {"ctx": 2048, "ubatch": 128, "checkpoints": 0},
]
_UNITS = {"K": 1 / 1024, "M": 1, "G": 1024}


def rss_mb(pid: int) -> float:
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])) / 1024


def footprint_mb(pid: int) -> float | None:
    """The peak physical footprint, from macOS's own accounting. None where that tool doesn't exist."""
    try:
        text = subprocess.check_output(["vmmap", "--summary", str(pid)], stderr=subprocess.DEVNULL, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    peak = re.search(r"Physical footprint \(peak\):\s*([\d.]+)\s*([KMG])", text)
    return round(float(peak.group(1)) * _UNITS[peak.group(2)], 1) if peak else None


_CLI = """
import resource, sys, tempfile
from pathlib import Path
import tth.cli.main
from tth.mcp.client import connect
with tempfile.TemporaryDirectory() as tmp:
    with connect(Path(tmp).resolve(), shell="zsh") as session:
        session.call("dry_run", {"command": "ls"})
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 if sys.platform == "darwin" else 1 / 1024) / 1048576)
"""


def cli_mb() -> float:
    """Peak memory of the CLI process after it has imported everything and made a real MCP call."""
    return round(float(subprocess.check_output([sys.executable, "-c", _CLI], text=True).strip()), 1)


def chat(content: str, max_tokens: int = 48) -> dict:
    body = {
        "messages": [{"role": "system", "content": SHELL_SYSTEM}, {"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=300) as r:  # noqa: S310 - fixed local URL
        return json.load(r)


def wait_until_healthy(process: subprocess.Popen) -> None:
    while process.poll() is None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1) as r:  # noqa: S310
                if r.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("llama-server exited before it was ready")


def measure(model: Path, setting: dict) -> dict:
    args = server_args(model, PORT, ctx=setting["ctx"], ubatch=setting["ubatch"], checkpoints=setting["checkpoints"])
    started = time.perf_counter()
    process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak = [0.0]
    done = threading.Event()

    def sample() -> None:
        while not done.wait(0.05):
            try:
                peak[0] = max(peak[0], rss_mb(process.pid))
            except (subprocess.CalledProcessError, ValueError):
                return

    sampler = threading.Thread(target=sample, daemon=True)
    try:
        wait_until_healthy(process)
        load = time.perf_counter() - started
        sampler.start()
        # About three quarters of the context, which is what a late turn with probe output looks like.
        filler = FILLER * max(1, (setting["ctx"] * 3 // 4 - 1000) // 30)
        first = chat(f"Files here:\n{filler}\nHow many are CSV?")
        second = chat(f"Files here:\n{filler}\nHow many are CSV? Answer briefly.")
        footprint = footprint_mb(process.pid)
        done.set()
        sampler.join()
        timing = first.get("timings", {})
        rss = round(peak[0], 1)
        file_mb = round(model.stat().st_size / 1e6)
        server = max(rss, file_mb + (footprint or 0))
        cli = cli_mb()
        total = round(server + cli, 1)
        return {
            "model": model.name, "file_mb": file_mb, **setting,
            "peak_rss_mb": rss, "peak_footprint_mb": footprint, "server_mb": server, "cli_mb": cli, "total_mb": total,
            "under_target": total < TARGET_MB,
            "load_s": round(load, 2), "prompt_tokens": timing.get("prompt_n"),
            "prompt_tps": round(timing.get("prompt_per_second", 0), 1),
            "gen_tps": round(timing.get("predicted_per_second", 0), 1),
            "cached_turn_prompt_tokens": second.get("timings", {}).get("prompt_n"),
            "machine": f"{platform.system()} {platform.machine()}",
        }  # fmt: skip
    finally:
        done.set()
        process.terminate()
        process.wait()
        time.sleep(1)


def main() -> None:
    out = Path(sys.argv[1])
    for name in sys.argv[2:]:
        for setting in SETTINGS:
            row = measure(Path(name), setting)
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
