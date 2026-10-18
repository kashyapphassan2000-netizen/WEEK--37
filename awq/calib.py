"""
calib.py — text in, exact-length token windows out.

Plain words:
  AWQ needs to *watch* the model work on real text to learn which input channels matter.
  We take documents, tokenize them, glue them together with an end-of-text token, and cut
  the stream into `n_samples` windows of exactly `seq_len` tokens.

  NO padding is used. (The original Week-37 guide used padding="max_length", which crashes on
  Llama tokenizers — they have no pad token — and pollutes the statistics with pad tokens.)

Datasets:
  calibration : pileval (official AWQ), c4, wikitext2, or your own .jsonl with {"text": ...}
  evaluation  : wikitext2 (test), wikitext2_val (validation, used only for tuning), c4 (validation)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch


def _iter_texts(name: str, split_hint: str = "train"):
    from datasets import load_dataset

    if name == "pileval":
        ds = load_dataset("mit-han-lab/pile-val-backup", split="validation", streaming=True)
        for ex in ds:
            yield ex["text"]
    elif name == "c4":
        f = "en/c4-train.00000-of-01024.json.gz" if split_hint == "train" else "en/c4-validation.00000-of-00008.json.gz"
        ds = load_dataset("allenai/c4", data_files={"x": f}, split="x", streaming=True)
        for ex in ds:
            yield ex["text"]
    elif name in ("wikitext2", "wikitext2_val", "wikitext2_train"):
        split = {"wikitext2": "test", "wikitext2_val": "validation", "wikitext2_train": "train"}[name]
        if name == "wikitext2" and split_hint == "train":
            split = "train"
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        for ex in ds:
            yield ex["text"]
    elif Path(name).exists():
        with open(name) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)["text"]
    else:
        raise ValueError(f"Unknown dataset {name!r}. Use pileval | c4 | wikitext2 | path/to/file.jsonl")


def get_calib_tokens(tokenizer, dataset: str, n_samples: int, seq_len: int, seed: int = 42) -> torch.Tensor:
    """Return LongTensor (n_samples, seq_len) of calibration tokens."""
    rng = random.Random(seed)
    need = n_samples * seq_len
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    # Take a pool of documents, shuffle it with the seed, then fill the token budget.
    pool = []
    for i, text in enumerate(_iter_texts(dataset, "train")):
        if text and len(text.strip()) > 20:
            pool.append(text)
        if len(pool) >= max(4 * n_samples, 512) and sum(len(t) for t in pool) > need * 6:
            break
    rng.shuffle(pool)
    ids: list[int] = []
    for text in pool:
        ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        ids.append(eos)
        if len(ids) >= need:
            break
    if len(ids) < need:
        raise RuntimeError(f"Calibration data too small: got {len(ids)} tokens, need {need}.")
    return torch.tensor(ids[:need], dtype=torch.long).view(n_samples, seq_len)


def get_eval_tokens(tokenizer, dataset: str, seq_len: int, max_windows: int = 0) -> torch.Tensor:
    """Return LongTensor (n_windows, seq_len) of non-overlapping evaluation windows."""
    if dataset in ("wikitext2", "wikitext2_val"):
        texts = list(_iter_texts(dataset, "eval"))
        text = "\n\n".join(texts)
    elif dataset == "c4":
        texts = []
        for i, t in enumerate(_iter_texts("c4", "validation")):
            texts.append(t)
            if i >= 1100:            # same budget as the GPTQ/AWQ papers' c4 eval slice
                break
        text = "\n\n".join(texts)
    elif Path(dataset).exists():
        text = "\n\n".join(_iter_texts(dataset))
    else:
        raise ValueError(f"Unknown eval dataset {dataset!r}")
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n = ids.numel() // seq_len
    if dataset == "c4":
        n = min(n, 256)
    if max_windows and max_windows > 0:
        n = min(n, max_windows)
    if n == 0:
        raise RuntimeError(f"{dataset}: fewer than seq_len={seq_len} tokens")
    return ids[: n * seq_len].view(n, seq_len)
