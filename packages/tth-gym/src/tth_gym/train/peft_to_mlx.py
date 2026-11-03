"""Convert a PEFT LoRA adapter (Qwen2) into an mlx-lm adapter directory.

PEFT:  y = W x + (alpha/r) * B(A(x)),  A: (r, in), B: (out, r)
MLX:   y = W x + scale * (x @ lora_a) @ lora_b,  lora_a: (in, r), lora_b: (r, out)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from safetensors.numpy import load_file, save_file

_KEY = re.compile(r"(?:base_model\.model\.)?(model\.layers\.(\d+)\.(\w+)\.(\w+))\.lora_([AB])\.weight$")


def convert(peft_dir: str | Path, out_dir: str | Path) -> None:
    peft_dir, out_dir = Path(peft_dir), Path(out_dir)
    cfg = json.loads((peft_dir / "adapter_config.json").read_text())
    weights = load_file(str(peft_dir / "adapter_model.safetensors"))
    out: dict = {}
    layers: set[int] = set()
    keys: set[str] = set()
    for k, v in weights.items():
        m = _KEY.search(k)
        if not m:
            raise ValueError(f"unexpected adapter key: {k}")
        path, layer, block, proj, ab = m.groups()
        layers.add(int(layer))
        keys.add(f"{block}.{proj}")
        out[f"{path}.lora_{ab.lower()}"] = v.T.astype("float32").copy()
    r = int(cfg["r"])
    scale = float(cfg["lora_alpha"]) / r
    n_layers = max(layers) + 1
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "adapters.safetensors"))
    (out_dir / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora", "num_layers": n_layers,
        "lora_parameters": {"rank": r, "scale": scale, "dropout": 0.0, "keys": sorted(keys)},
        "source": "peft", "base_model": cfg.get("base_model_name_or_path"),
    }, indent=1))  # fmt: skip


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
