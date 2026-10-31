"""LoRA fine-tuning on trajectories (PyTorch + PEFT; runs on a Kaggle/Colab GPU).

Loss is applied only to assistant turns. Each row may carry a `weight` (default 1.0): SFT uses
weight 1; the GRPO step passes group-normalised advantages, which turns the same loss into a
policy gradient.

Speed: examples are batched by length, and the 152k-word output layer is applied only at
assistant positions (same loss, a fraction of the compute). Gradient checkpointing is off unless
the GPU runs out of memory, in which case it is switched on and the batch is retried.
Progress is printed as JSON lines (step, of, loss, tok_s, eta_min) for the notebook monitor.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tok: Any, row: dict, max_len: int) -> tuple[list[int], list[int]] | None:
    """Token ids + mask (1 on the model's own turns, including <tool_call> tags and the closing <|im_end|>).

    Spans are found directly in the token ids: everything after each `<|im_start|>assistant\\n` up to and
    including its `<|im_end|>`. This does not depend on how a tokenizer/template version renders prefixes
    or reports character offsets (both broke on one environment).
    """
    msgs, tools = row["messages"], row.get("tools") or None
    ids = tok(tok.apply_chat_template(msgs, tools=tools, tokenize=False), add_special_tokens=False)["input_ids"]
    start, end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
    head = tok("assistant\n", add_special_tokens=False)["input_ids"]
    mask, i = [0] * len(ids), 0
    while i < len(ids):
        if ids[i] == start and ids[i + 1 : i + 1 + len(head)] == head:
            j = i + 1 + len(head)
            while j < len(ids) and ids[j] != end:
                mask[j] = 1
                j += 1
            if j < len(ids):
                mask[j] = 1  # learn to end the turn after the call
            i = j
        i += 1
    if len(ids) > max_len or not any(mask):
        return None
    return ids, mask


def masked_preview(tok: Any, row: dict, n: int = 160) -> str:
    """The first trained tokens of an example, printed at startup so a broken mask is visible immediately."""
    e = encode(tok, row, 1 << 20)
    return "" if e is None else tok.decode([t for t, m in zip(*e, strict=True) if m])[:n]


def load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        rows += [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
    return rows


def build_model(base: str, init_adapter: str | None, rank: int, alpha: int) -> tuple[Any, Any, Any]:
    import torch  # pyright: ignore[reportMissingImports]  (GPU-only dependency)
    from peft import LoraConfig, PeftModel, get_peft_model  # pyright: ignore[reportMissingImports]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32  # CPU smoke tests
    tok = AutoTokenizer.from_pretrained(base)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype).to(dev)  # pyright: ignore[reportCallIssue,reportArgumentType]
    # Always on: a batch of 4 long episodes does not fit a 15 GB T4 without it.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if init_adapter:
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
    return model, tok, dtype


def _batch_loss(model: Any, batch: list[tuple[list[int], list[int], float]], pad: int, dtype: Any) -> tuple[Any, int]:
    import torch  # pyright: ignore[reportMissingImports]

    dev = next(model.parameters()).device
    n, L = len(batch), max(len(ids) for ids, _, _ in batch)
    x = torch.full((n, L), pad, dtype=torch.long, device=dev)
    am = torch.zeros((n, L), dtype=torch.long, device=dev)
    m = torch.zeros((n, L), dtype=torch.bool, device=dev)
    w = torch.tensor([b[2] for b in batch], device=dev, dtype=torch.float32)
    for i, (ids, mask, _) in enumerate(batch):
        x[i, : len(ids)] = torch.tensor(ids, device=dev)
        am[i, : len(ids)] = 1
        m[i, : len(ids)] = torch.tensor(mask, device=dev, dtype=torch.bool)
    base = model.get_base_model()
    with torch.autocast("cuda", dtype=dtype, enabled=dev.type == "cuda"):
        hidden = base.model(input_ids=x, attention_mask=am).last_hidden_state  # LoRA layers live inside
        sel = m[:, 1:]  # predict token t+1 from position t, only where t+1 is an assistant token
        logits = base.lm_head(hidden[:, :-1][sel]).float()
    nll = torch.nn.functional.cross_entropy(logits, x[:, 1:][sel], reduction="none")
    ex = torch.arange(n, device=dev).unsqueeze(1).expand(-1, L - 1)[sel]
    per_ex = torch.zeros(n, device=dev).index_add_(0, ex, nll)
    counts = torch.zeros(n, device=dev).index_add_(0, ex, torch.ones_like(nll)).clamp(min=1)
    return (w * per_ex / counts).sum() / n, int(am.sum())


def train(model: Any, tok: Any, dtype: Any, rows: list[dict], epochs: float, lr: float, batch_size: int, accum: int,
          max_len: int, seed: int, log_path: Path | None, label: str = "train") -> dict:  # fmt: skip
    import torch  # pyright: ignore[reportMissingImports]

    random.seed(seed)
    torch.manual_seed(seed)
    data, skipped = [], 0
    for r in rows:
        e = encode(tok, r, max_len)
        if e is None or sum(e[1]) == 0:
            skipped += 1
            continue
        data.append((e[0], e[1], float(r.get("weight", 1.0))))
    idx = sorted(range(len(data)), key=lambda i: len(data[i][0]))  # length buckets -> little padding
    batches_one = [idx[i : i + batch_size] for i in range(0, len(idx), batch_size)]
    n_batches = max(1, int(len(batches_one) * epochs))
    order: list[list[int]] = []
    while len(order) < n_batches:
        b = batches_one[:]
        random.shuffle(b)
        order += b
    order = order[:n_batches]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    total = max(1, math.ceil(n_batches / accum))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, total // 20)) * 0.5 * (1 + math.cos(math.pi * s / total))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16 and torch.cuda.is_available())
    model.train()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    log, t0, tokens, run_loss = [], time.time(), 0, 0.0
    for k, bi in enumerate(order):
        batch = [data[i] for i in bi]
        try:
            loss, ntok = _batch_loss(model, batch, pad, dtype)
            scaler.scale(loss / accum).backward()
        except torch.cuda.OutOfMemoryError:
            loss, ntok = None, 0  # retry below, after the failed batch's memory is released
        if loss is None:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(json.dumps({"job": label, "note": "out of memory: retrying batch one example at a time"}), flush=True)
            ntok = 0
            for one in batch:
                lo, nt = _batch_loss(model, [one], pad, dtype)
                scaler.scale(lo / accum / len(batch)).backward()
                loss, ntok = lo.detach(), ntok + nt
        tokens += ntok
        run_loss += float(loss.detach()) if loss is not None else 0.0
        if (k + 1) % accum == 0 or k == len(order) - 1:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step = math.ceil((k + 1) / accum)
            el = time.time() - t0
            entry = {"job": label, "step": step, "of": total, "loss": round(run_loss / accum, 4),
                     "tok_s": round(tokens / el), "eta_min": round(el / step * (total - step) / 60, 1)}  # fmt: skip
            print(json.dumps(entry), flush=True)
            log.append(entry)
            run_loss = 0.0
    summary = {"examples": len(data), "skipped": skipped, "steps": total, "seconds": round(time.time() - t0),
               "tokens": tokens, "log": log}  # fmt: skip
    if log_path:
        log_path.write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--init-adapter", default=None, help="continue training this PEFT adapter")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)  # effective batch: 8 examples
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from train.sft.peft_to_mlx import convert

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok, dtype = build_model(args.base, args.init_adapter, args.rank, args.alpha)
    rows = load_rows(args.data)
    print(json.dumps({"job": out.name, "note": f"training on {len(rows)} rows"}), flush=True)
    preview = masked_preview(tok, rows[0])
    print(json.dumps({"job": out.name, "note": f"trained tokens start: {preview[:90]!r}"}), flush=True)
    if '"name"' not in preview:
        raise SystemExit(f"loss mask does not cover the tool call: {preview!r}")
    summary = train(model, tok, dtype, rows, args.epochs, args.lr, args.batch_size, args.accum, args.max_len, args.seed,
                    out / "train_log.json", label=out.name)  # fmt: skip
    model.save_pretrained(out / "peft")
    convert(out / "peft", out / "mlx")
    (out / "run.json").write_text(json.dumps({"args": vars(args), **{k: v for k, v in summary.items() if k != "log"}}))
    print(json.dumps({"job": out.name, "note": "saved"}), flush=True)


if __name__ == "__main__":
    main()
