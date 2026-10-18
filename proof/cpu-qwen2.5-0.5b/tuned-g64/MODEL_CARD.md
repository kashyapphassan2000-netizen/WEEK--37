---
base_model: Qwen/Qwen2.5-0.5B
tags: [awq, quantization, int4, awq-from-scratch]
---

# Qwen2.5-0.5B — AWQ 4-bit (group 64, zero-point)

4-bit AWQ (Lin et al. 2023, arXiv:2306.00978) quantization of [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B),
produced with the from-scratch, block-streaming implementation at
https://github.com/kashyapphassan2000-netizen/WEEK--37 . Every number below was measured, not copied.

## Results (measured)

| metric | original 16-bit | naive RTN 4-bit | **AWQ 4-bit (this)** |
|---|---|---|---|
| wikitext2 perplexity (lower is better) | 12.7007 | 15.0808 | 14.0573 |
| c4 perplexity (lower is better) | 19.1558 | 22.9873 | 21.1493 |
| weight memory (GB) | 1.976 | — | 0.752 |

Next-token agreement with the original on probe prompts: 0.830.
Eval: 2048-token windows; windows per dataset: {'wikitext2': 20, 'c4': 20}.

Gates:
- FAIL: wikitext2: AWQ ppl within 10% of original
- PASS: wikitext2: AWQ ppl < RTN ppl
- FAIL: c4: AWQ ppl within 10% of original
- PASS: c4: AWQ ppl < RTN ppl
- PASS: lm-eval avg drop <= 3.0 points
- PASS: no empty generations

## Method
- Calibration: pileval, 64 x 512 tokens.
- Per-group activation-aware scale search (20 grid points, duo scaling=True),
  weight-clip search=True, scales folded into preceding norms/linears where possible.
- Embeddings and output head kept in 16-bit (standard AWQ practice).

## Use
```bash
vllm serve <this-repo> --quantization awq_marlin --dtype float16
```

## License
Same license and use policy as the base model `Qwen/Qwen2.5-0.5B` (see LICENSE / USE_POLICY files in this repo).
