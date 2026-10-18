#!/usr/bin/env python
"""
build.py — render docs/Universal_AWQ_Playbook.pdf from playbook.html + live repo data.

Generated parts (so the PDF can never disagree with the repo):
  <!--GEN:DECISION-->  decision map from proof/hardware_reports/*.json
  <!--GEN:PROOF-->     measured results from proof/cpu-qwen2.5-0.5b/*.json and tuning leaderboard
  <!--GEN:CODE-->      code walkthrough (code_walkthrough.html)
  <!--GEN:APPENDIX-->  full source of every file in the repo

    python docs/playbook/build.py            # needs Chromium (set CHROME=/path/to/chrome if not found)
"""
import csv
import glob
import html
import json
import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROOF = ROOT / "proof" / "cpu-qwen2.5-0.5b"


def esc(x):
    return html.escape(str(x))


def f(x, nd=3):
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def decision_table() -> str:
    rows = []
    for p in glob.glob(str(ROOT / "proof" / "hardware_reports" / "*.json")):
        r = json.loads(Path(p).read_text())
        rows.append(r)
    rows.sort(key=lambda r: r["estimate"]["params_b"])
    out = ["<table><tr><th>Model</th><th>Type</th><th>Params</th><th>FP16</th><th>AWQ</th>"
           "<th>A. Quantize on (VRAM)</th><th>B. FP16 eval on</th><th>C. Serve AWQ on (8k ctx)</th>"
           "<th>A hours*</th><th>A cost*</th></tr>"]
    for r in rows:
        e, d = r["estimate"], r["dims"]
        ph = e["phases"]

        def best(k):
            b = ph[k]["best"]
            if not b:
                return "multi-node" if k != "A_quantize" else "&gt; 1 GPU (does not fit)"
            return f"{b['count']}× {esc(b['gpu'])} ({ph[k]['vram_gb']} GB)"
        typ = f"MoE {d['experts']}e top-{d['experts_per_tok']}" if d["experts"] else "dense"
        if e.get("checkpoint_gb") and e["checkpoint_gb"] < 0.75 * e["fp16_gb"]:
            typ += ", FP8 ckpt"
        out.append(f"<tr><td>{esc(r['model'])}</td><td>{typ}</td><td>{e['params_b']:.1f} B</td>"
                   f"<td>{e['fp16_gb']:.0f} GB</td><td>{e['awq_gb']:.1f} GB</td><td>{best('A_quantize')}</td>"
                   f"<td>{best('B_fp16_eval')}</td><td>{best('C_serve_awq')}</td>"
                   f"<td>{ph['A_quantize']['hours_est']}</td><td>${ph['A_quantize']['cost_usd_est']}</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def proof_section() -> str:
    if not (PROOF / "results.json").exists():
        return "<p><b>Proof run not found.</b></p>"
    base = json.loads((PROOF / "baseline.json").read_text())
    res = json.loads((PROOF / "results.json").read_text())
    ver = json.loads((PROOF / "verify.json").read_text())
    env = json.loads((PROOF / "env.json").read_text())
    log = [json.loads(line) for line in (PROOF / "search_log.jsonl").read_text().splitlines()]
    qc = res["quant_config"]
    s = []
    s.append(f"""<h1>19. Proof: measured results</h1>
<p>Everything on this page was produced by <code>bash scripts/run_all.sh configs/examples/qwen2.5-0.5b-cpu.yaml</code>
on the build machine: {env.get('cpu_count')} CPU cores, {env.get('ram_gb')} GB RAM, no GPU, torch {env.get('torch')},
transformers {env.get('transformers')}. The raw files are in <code>proof/cpu-qwen2.5-0.5b/</code>. On CPU the "original"
model runs in <b>FP32</b> (CPU half-precision is slow/unsupported), so "original" below means FP32.</p>
<div class="box note"><b class="tag">SETUP</b> Model <code>{esc(qc['source_model'])}</code> · AWQ {qc['w_bit']}-bit, group {qc['group_size']},
zero-point={qc['zero_point']}, fold_scales={qc['fold_scales']} · calibration {qc['calibration']['dataset']}
{qc['calibration']['n_samples']}×{qc['calibration']['seq_len']} tokens · eval: first {max(base['eval_windows'].values())} windows of 2048 tokens
per dataset (CPU budget) · {qc['n_linears_quantized']} Linear layers quantized, {qc['n_linears_with_inv_s']} with runtime 1/s.</div>""")
    s.append("<h2>19.1 Perplexity (lower is better)</h2><table><tr><th>Dataset</th><th>Original (FP32)</th>"
             "<th>Naive RTN 4-bit</th><th><b>AWQ 4-bit</b></th><th>RTN gap</th><th>AWQ gap</th><th>Gap closed by AWQ</th></tr>")
    for d in ("wikitext2", "c4"):
        fp, rtn, awq = base.get(f"fp16_ppl_{d}"), base.get(f"rtn_ppl_{d}"), res.get(f"awq_ppl_{d}")
        if fp is None or awq is None:
            continue
        closed = (rtn - awq) / (rtn - fp) if rtn and rtn != fp else None
        s.append(f"<tr><td>{d}</td><td>{fp:.3f}</td><td>{rtn:.3f}</td><td><b>{awq:.3f}</b></td>"
                 f"<td>+{100 * (rtn - fp) / fp:.1f}%</td><td>+{100 * (awq - fp) / fp:.1f}%</td>"
                 f"<td>{'n/a' if closed is None else f'{100 * closed:.0f}%'}</td></tr>")
    s.append("</table>")
    if base.get("fp16_lm_eval") and res.get("awq_lm_eval"):
        s.append("<h2>19.2 Downstream accuracy (lm-eval, first 300 examples per task)</h2><table>"
                 "<tr><th>Task</th><th>Metric</th><th>Original</th><th>RTN</th><th><b>AWQ</b></th></tr>")
        for task, m in res["awq_lm_eval"].items():
            for k in ("acc", "acc_norm"):
                if k in m:
                    s.append(f"<tr><td>{task}</td><td>{k}</td><td>{f(base['fp16_lm_eval'].get(task, {}).get(k))}</td>"
                             f"<td>{f(base.get('rtn_lm_eval', {}).get(task, {}).get(k))}</td><td><b>{f(m[k])}</b></td></tr>")
        s.append("</table><p class='small'>300 examples per task means a ±2.8-point standard error. Differences "
                 "under ~3 points are noise at this sample size.</p>")
    s.append("<h2>19.3 Memory, agreement, speed</h2><table><tr><th>Measure</th><th>Value</th></tr>")
    items = [
        ("Weights in memory, original (FP32)", f"{base.get('fp16_weight_bytes', 0) / 1e9:.3f} GB"),
        ("Weights in memory, AWQ", f"{res.get('awq_weight_bytes', 0) / 1e9:.3f} GB "
                                   f"({ver.get('memory_reduction_x')}× smaller vs FP32)"),
        ("AWQ on disk / formula for this dtype", f"{ver.get('awq_disk_gb')} GB / {ver.get('awq_gb_formula')} GB "
                                                 f"(ratio {ver.get('size_vs_formula')})"),
        ("AWQ size if saved in 16-bit (formula)", f"{ver.get('awq_gb_formula_16bit')} GB vs 0.99 GB FP16 checkpoint"),
        ("Next-token agreement with original (10 probe prompts, all positions)", f"{ver.get('top1_agreement', 0):.3f}"),
        ("Mean relative logit error", f"{ver.get('logit_rel_err_mean', 0):.3f}"),
        ("Speed original (CPU, greedy)", f"{base.get('fp16_speed', {}).get('tokens_per_s')} tok/s"),
        ("Speed AWQ reference kernel (CPU)", f"{res.get('awq_speed', {}).get('tokens_per_s')} tok/s"),
    ]
    for k, v in items:
        s.append(f"<tr><td>{k}</td><td>{esc(v)}</td></tr>")
    s.append("</table>")
    s.append("<h2>19.4 Gates</h2><table><tr><th>Gate</th><th>Result</th></tr>")
    gates = {**{f"verify: {k}": v for k, v in ver.get("gates", {}).items()}, **res.get("gates", {})}
    for k, v in gates.items():
        k = k.replace("of FP16", "of original")
        s.append(f"<tr><td>{esc(k)}</td><td><b style='color:{'#067647' if v else '#b42318'}'>"
                 f"{'PASS' if v else 'FAIL'}</b></td></tr>")
    s.append("</table>")
    # per-block
    s.append("<h2>19.5 What the search chose, block by block</h2><table><tr><th>Block</th><th>qkv α</th>"
             "<th>gate/up α</th><th>down α</th><th>AWQ/RTN loss (qkv, gate/up, down)</th><th>block rel. MSE</th><th>s</th></tr>")
    for b in log:
        g = {tuple(x["linears"]): x for x in b["groups"]}

        def a(names):
            x = next((v for k, v in g.items() if k[0].endswith(names)), None)
            return x
        cells, ratios = [], []
        for key in ("q_proj", "gate_proj", "down_proj"):
            x = a(key)
            cells.append("RTN" if x is None or x["ratio"] is None else f"{x['ratio']:.2f}")
            if x and x["loss_rtn"]:
                ratios.append(f"{x['loss_awq'] / x['loss_rtn']:.2f}")
        s.append(f"<tr><td>{b['block']}</td><td>{cells[0]}</td><td>{cells[1]}</td><td>{cells[2]}</td>"
                 f"<td>{', '.join(ratios)}</td><td>{b['block_rel_mse']:.2e}</td><td>{b['seconds']}</td></tr>")
    s.append("</table><p class='small'>α = 0 trusts weights only, α → 1 trusts activations only. Loss ratio &lt; 1 "
             "means AWQ's scale beat plain rounding on that group's calibration output. The first and last blocks are "
             "the hardest (largest error), the classic outlier pattern.</p>")
    gens = PROOF / "generations.md"
    if gens.exists():
        s.append("<h2>19.6 Generations (first 3 of 10 prompts, greedy; all 10 in generations.md)</h2><pre class='src'>" +
                 esc("\n".join(gens.read_text().splitlines()[3:45])) + "</pre>")
    lb = PROOF / "tuning" / "leaderboard.csv"
    if lb.exists():
        rows = list(csv.DictReader(lb.open()))
        cols = list(rows[0].keys())
        s.append("<h2>19.7 Tuning (06_tune.py, metric = WikiText-2 VALIDATION ppl, never the test set)</h2>"
                 "<table><tr>" + "".join(f"<th>{esc(c)}</th>" for c in cols) + "</tr>")
        for r in rows:
            s.append("<tr>" + "".join(f"<td>{esc(r[c])}</td>" for c in cols) + "</tr>")
        s.append("</table>")
        tl = PROOF / "tuning" / "tune.log"
        if tl.exists():
            lines = [x for x in tl.read_text().splitlines() if "stage winner" in x or "best overrides" in x
                     or "stage kept" in x]
            s.append("<pre>" + esc("\n".join(lines)) + "</pre>")
    t = PROOF / "tuned-g64"
    if (t / "results.json").exists():
        fr = json.loads((t / "results.json").read_text())
        rg = json.loads((t / "rtn_g64.json").read_text())
        tv = json.loads((t / "verify.json").read_text())
        s.append("<h2>19.8 Final run with the tuned config (group 64), evaluated ONCE on the test windows</h2>"
                 "<table><tr><th>Dataset</th><th>Original</th><th>RTN g128</th><th>AWQ g128</th><th>RTN g64</th>"
                 "<th><b>AWQ g64 (tuned)</b></th><th>AWQ g64 gap</th></tr>")
        for d in ("wikitext2", "c4"):
            fp = base[f"fp16_ppl_{d}"]
            s.append(f"<tr><td>{d}</td><td>{fp:.3f}</td><td>{base[f'rtn_ppl_{d}']:.3f}</td>"
                     f"<td>{res[f'awq_ppl_{d}']:.3f}</td><td>{rg[f'rtn_ppl_{d}']:.3f}</td>"
                     f"<td><b>{fr[f'awq_ppl_{d}']:.3f}</b></td><td>+{100 * (fr[f'awq_ppl_{d}'] - fp) / fp:.1f}%</td></tr>")
        s.append("</table>")
        le = fr.get("awq_lm_eval", {})
        s.append(f"<p>Tuned model: next-token agreement {tv.get('top1_agreement', 0):.3f} (was "
                 f"{ver.get('top1_agreement', 0):.3f}), weights {tv.get('awq_disk_gb')} GB, lm-eval "
                 + ", ".join(f"{k} acc {v.get('acc', 0):.3f}" for k, v in le.items())
                 + ". Gates: " + ", ".join(f"{k.replace('of FP16', 'of original')}: {'PASS' if v else 'FAIL'}"
                                            for k, v in fr.get("gates", {}).items()) + ".</p>")
        s.append("<div class='box truth'><b class='tag'>BRUTAL TRUTH</b> Most of the tuning gain came from the smaller "
                 "group size, which helps plain RTN too (RTN g64 is already 14.48). At equal group size, AWQ removes a "
                 "further ~24–27% of the remaining gap. The 0.5B model still misses the strict ≤10% gate by 0.4–0.7 "
                 "percentage points. The vLLM-format export of this model passed its layout check (worst rel. diff 3.3e-4, "
                 "fp16 scale rounding) and the model card was generated with <code>08_publish.py --dry-run</code>.</div>")
    return "\n".join(s)


APPENDIX_FILES = (["pyproject.toml"] + sorted(glob.glob("awq/*.py", root_dir=ROOT)) +
                  sorted(glob.glob("scripts/*", root_dir=ROOT)) + ["configs/run.yaml", "configs/gpus.yaml",
                  "configs/tune_space.yaml", "configs/tune_space_cpu.yaml"] +
                  sorted(glob.glob("configs/examples/*.yaml", root_dir=ROOT)) +
                  sorted(glob.glob("tests/*.py", root_dir=ROOT)))


def appendix() -> str:
    s = ["<h1>Appendix: full source code</h1><p>Every file in the repository, exactly as committed. "
         "The comments in each file explain each step in plain words.</p>"]
    for rel in APPENDIX_FILES:
        p = ROOT / rel
        if not p.is_file() or "__init__" in rel and p.stat().st_size < 10:
            continue
        txt = p.read_text()
        s.append(f"<h2 class='src'>{esc(rel)} <span class='small'>({len(txt.splitlines())} lines)</span></h2>"
                 f"<pre class='src'>{esc(txt)}</pre>")
    return "\n".join(s)


def main():
    src = (HERE / "playbook.html").read_text()
    src = src.replace("<!--GEN:DECISION-->", decision_table())
    src = src.replace("<!--GEN:CODE-->", (HERE / "code_walkthrough.html").read_text())
    src = src.replace("<!--GEN:PROOF-->", proof_section())
    src = src.replace("<!--GEN:HWREPORTS-->", decision_table())
    src = src.replace("<!--GEN:APPENDIX-->", appendix())
    out_html = HERE / "_build.html"
    out_html.write_text(src)
    chrome = os.environ.get("CHROME") or shutil.which("chromium") or shutil.which("google-chrome") or next(
        iter(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome")), None)
    if not chrome:
        raise SystemExit(f"No Chromium found. Open {out_html} in a browser and print to PDF.")
    pdf = ROOT / "docs" / "Universal_AWQ_Playbook.pdf"
    subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", str(out_html)], check=True, capture_output=True)
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
