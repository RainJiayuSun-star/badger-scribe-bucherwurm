"""Score a run with official metric.py plus diagnostics."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import data_root, results_dir, runs_dir, splits_dir


def load_metric():
    path = data_root() / "metric.py"
    spec = importlib.util.spec_from_file_location("badger_scribe_metric", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def solution_csv(split: str) -> Path:
    if split == "smoke":
        return splits_dir() / "smoke_solution.csv"
    return splits_dir() / "holdout_solution.csv"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    return float(statistics.median(xs))


def mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--split", default="smoke", choices=["smoke", "val"])
    p.add_argument("--append-results", action="store_true")
    p.add_argument("--write-table", action="store_true",
                   help="rebuild results/wave1.md from results/wave1.csv")
    p.add_argument("--record-skip", default="",
                   help="record a skipped/OOM run into results without scoring")
    p.add_argument("--model", default="")
    p.add_argument("--model-id", default="")
    p.add_argument("--device", default="")
    p.add_argument("--dtype", default="")
    return p.parse_args()


def qualitative_examples(sol_rows: list[dict], pred_map: dict[str, str], metric, k: int = 2):
    by_cat: dict[str, list[tuple[float, str, str, str]]] = {}
    for row in sol_rows:
        pid = row["page_id"]
        pred = pred_map.get(pid, "")
        cer = metric.page_cer(pred, row["text"])
        by_cat.setdefault(row["category"], []).append((cer, pid, row["text"][:240], pred[:240]))
    out = {}
    for cat, items in by_cat.items():
        items.sort(reverse=True)
        out[cat] = [
            {"page_id": pid, "cer": round(cer, 4), "ref_head": ref, "pred_head": pred}
            for cer, pid, ref, pred in items[:k]
        ]
    return out


def eval_run(run_id: str, split: str) -> dict:
    metric = load_metric()
    run_dir = runs_dir() / run_id
    pred_path = run_dir / "preds.csv"
    if not pred_path.exists():
        raise SystemExit(f"missing {pred_path}")

    sol_path = solution_csv(split)
    sol_rows = read_csv(sol_path)
    pred_rows = read_csv(pred_path)
    pred_map = {r["page_id"]: r.get("text", "") for r in pred_rows}

    import pandas as pd

    solution = pd.read_csv(sol_path, keep_default_na=False)
    submission = pd.read_csv(pred_path, keep_default_na=False)
    by_cat = metric.per_category_cer(solution, submission)
    overall = metric.score(solution, submission, "page_id")

    merged = solution.merge(
        submission.drop_duplicates("page_id", keep="last"),
        on="page_id", how="left", suffixes=("_ref", "_pred"),
    )
    merged["text_pred"] = merged["text_pred"].fillna("")
    cers = [metric.page_cer(p, r) for p, r in zip(merged["text_pred"], merged["text_ref"])]
    wers = [metric.page_wer(p, r) for p, r in zip(merged["text_pred"], merged["text_ref"])]
    merged["cer"] = cers
    merged["wer"] = wers

    human = merged[merged.get("label_source", "") == "human"] if "label_source" in merged.columns else merged.iloc[0:0]
    kade = merged[merged["category"] == "kade_letters"]
    by_source = {}
    if "label_source" in merged.columns:
        for src, g in merged.groupby("label_source"):
            by_source[src] = {
                "n": int(len(g)),
                "cer_mean": mean(list(g["cer"])),
                "wer_mean": mean(list(g["wer"])),
            }

    infer = {}
    infer_path = run_dir / "infer_metrics.json"
    if infer_path.exists():
        infer = json.loads(infer_path.read_text(encoding="utf-8"))
    config = {}
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        config = json.loads(cfg_path.read_text(encoding="utf-8"))

    payload = {
        "run_id": run_id,
        "split": split,
        "n_solution": int(len(solution)),
        "n_submission": int(len(submission)),
        "macro_cer": float(overall),
        "cer_by_category": {k: float(v) for k, v in by_cat.items()},
        "wer_by_category": {
            cat: mean(list(g["wer"])) for cat, g in merged.groupby("category")
        },
        "cer_mean": mean(cers),
        "cer_median": median(cers),
        "wer_mean": mean(wers),
        "wer_median": median(wers),
        "human_n": int(len(human)),
        "human_cer_mean": mean(list(human["cer"])) if len(human) else None,
        "human_wer_mean": mean(list(human["wer"])) if len(human) else None,
        "kade_n": int(len(kade)),
        "kade_cer_mean": mean(list(kade["cer"])) if len(kade) else None,
        "by_label_source": by_source,
        "device": infer.get("device") or config.get("device"),
        "dtype": infer.get("dtype") or config.get("dtype"),
        "model": infer.get("model") or config.get("model"),
        "model_id": infer.get("model_id") or config.get("model_id"),
        "with_metadata": infer.get("with_metadata", config.get("with_metadata")),
        "sec_per_page_mean": infer.get("sec_per_page_mean"),
        "sec_per_page_p50": infer.get("sec_per_page_p50"),
        "sec_per_page_p95": infer.get("sec_per_page_p95"),
        "peak_mem_mb": infer.get("peak_mem_mb"),
        "status": infer.get("status", "scored"),
        "qualitative_worst": qualitative_examples(sol_rows, pred_map, metric),
        "official_cmd": (
            f"python {data_root() / 'metric.py'} --solution {sol_path} --submission {pred_path}"
        ),
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"macro CER: {overall:.4f}")
    for cat, v in sorted(payload["cer_by_category"].items()):
        print(f"  {cat:<24} {v:.4f}")
    print(f"human CER: {payload['human_cer_mean']}")
    print(f"kade CER:  {payload['kade_cer_mean']}")
    print(f"median CER:{payload['cer_median']}")
    print(f"wrote {run_dir / 'metrics.json'}")
    return payload


CSV_FIELDS = [
    "run_id", "split", "model", "model_id", "device", "dtype", "with_metadata",
    "macro_cer", "cer_kade_letters", "cer_dominy_accounts", "cer_survey_notes",
    "human_cer_mean", "kade_cer_mean", "cer_median", "wer_mean",
    "sec_per_page_mean", "sec_per_page_p50", "sec_per_page_p95", "peak_mem_mb", "status",
]


def row_from_metrics(m: dict) -> dict:
    cats = m.get("cer_by_category") or {}
    return {
        "run_id": m.get("run_id"),
        "split": m.get("split"),
        "model": m.get("model"),
        "model_id": m.get("model_id"),
        "device": m.get("device"),
        "dtype": m.get("dtype"),
        "with_metadata": m.get("with_metadata"),
        "macro_cer": m.get("macro_cer"),
        "cer_kade_letters": cats.get("kade_letters"),
        "cer_dominy_accounts": cats.get("dominy_accounts"),
        "cer_survey_notes": cats.get("survey_notes"),
        "human_cer_mean": m.get("human_cer_mean"),
        "kade_cer_mean": m.get("kade_cer_mean"),
        "cer_median": m.get("cer_median"),
        "wer_mean": m.get("wer_mean"),
        "sec_per_page_mean": m.get("sec_per_page_mean"),
        "sec_per_page_p50": m.get("sec_per_page_p50"),
        "sec_per_page_p95": m.get("sec_per_page_p95"),
        "peak_mem_mb": m.get("peak_mem_mb"),
        "status": m.get("status"),
    }


def append_results(m: dict) -> Path:
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "wave1.csv"
    rows = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    new = row_from_metrics(m)
    rows = [r for r in rows if r.get("run_id") != new["run_id"]]
    rows.append({k: "" if new.get(k) is None else new[k] for k in CSV_FIELDS})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return csv_path


def fmt(v, digits=4) -> str:
    if v is None or v == "":
        return "—"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def write_wave1_md(csv_path: Path | None = None) -> Path:
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    csv_path = csv_path or (out / "wave1.csv")
    md_path = out / "wave1.md"
    if not csv_path.exists():
        md_path.write_text("# Wave 1 results\n\nNo runs yet.\n", encoding="utf-8")
        return md_path
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    val_rows = [r for r in rows if r.get("split") == "val"]
    smoke_rows = [r for r in rows if r.get("split") == "smoke"]
    ranked = sorted(
        [r for r in val_rows if r.get("human_cer_mean") not in (None, "")],
        key=lambda r: float(r["human_cer_mean"]),
    )
    pick = ranked[0] if ranked else None

    lines = [
        "# Wave 1 results",
        "",
        "Official score is category-macro CER from `badger-scribe-data/metric.py`.",
        "Lower is better. Wall-clock is device-specific; do not mix Mac and 3060 times in one ranking.",
        "",
        "## Holdout (val)",
        "",
        "| run_id | model | device | dtype | meta | macro CER | Kade CER | Dominy CER | Survey CER | human CER | median CER | WER | sec/page | peak MB |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in val_rows:
        lines.append(
            "| {run_id} | {model} | {device} | {dtype} | {meta} | {macro} | {kade} | {dom} | {surv} | {hum} | {med} | {wer} | {sec} | {mem} |".format(
                run_id=r.get("run_id"),
                model=r.get("model"),
                device=r.get("device"),
                dtype=r.get("dtype"),
                meta=r.get("with_metadata"),
                macro=fmt(r.get("macro_cer")),
                kade=fmt(r.get("cer_kade_letters")),
                dom=fmt(r.get("cer_dominy_accounts")),
                surv=fmt(r.get("cer_survey_notes")),
                hum=fmt(r.get("human_cer_mean")),
                med=fmt(r.get("cer_median")),
                wer=fmt(r.get("wer_mean")),
                sec=fmt(r.get("sec_per_page_mean"), 2),
                mem=fmt(r.get("peak_mem_mb"), 0),
            )
        )
    lines += [
        "",
        "## Smoke (12 val pages)",
        "",
        "| run_id | model | device | dtype | macro CER | Kade CER | human CER | status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in smoke_rows:
        lines.append(
            "| {run_id} | {model} | {device} | {dtype} | {macro} | {kade} | {hum} | {status} |".format(
                run_id=r.get("run_id"),
                model=r.get("model"),
                device=r.get("device"),
                dtype=r.get("dtype"),
                macro=fmt(r.get("macro_cer")),
                kade=fmt(r.get("cer_kade_letters")),
                hum=fmt(r.get("human_cer_mean")),
                status=r.get("status") or "",
            )
        )

    lines += ["", "## LoRA base", ""]
    if pick:
        lines.append(
            f"**Pick: `{pick['model']}`** (`{pick.get('model_id')}`), run `{pick['run_id']}`. "
            f"Best human-label mean CER on the frozen holdout "
            f"({fmt(pick.get('human_cer_mean'))}); Kade CER {fmt(pick.get('kade_cer_mean') or pick.get('cer_kade_letters'))}. "
            "Fits the intended 12 GB / 24 GB deploy path if the winning dtype did."
        )
    else:
        lines.append(
            "No val run with a human-label CER yet. Default pick remains **Churro 3B** "
            "(`stanford-oval/churro-3B`) as the historical-page specialist baseline."
        )

    # Qualitative from the picked (or last) val run.
    example_id = (pick or (val_rows[-1] if val_rows else None) or {}).get("run_id") if (pick or val_rows) else None
    if example_id:
        qpath = runs_dir() / example_id / "metrics.json"
        if qpath.exists():
            q = json.loads(qpath.read_text(encoding="utf-8")).get("qualitative_worst") or {}
            lines += ["", f"## Qualitative (worst pages from `{example_id}`)", ""]
            for cat, items in q.items():
                lines.append(f"### {cat}")
                for it in items:
                    lines.append(
                        f"- `{it['page_id']}` CER {it['cer']}: pred `{it['pred_head'][:160].replace('|', '/')}`"
                    )
                lines.append("")

    lines += [
        "",
        "Go/no-go: LoRA the picked base on human German (Kade train docs). "
        "Do not submit to Kaggle until this ranking is stable.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def record_skip(args: argparse.Namespace) -> dict:
    run_dir = runs_dir() / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": args.run_id,
        "split": args.split,
        "model": args.model,
        "model_id": args.model_id,
        "device": args.device or "mps",
        "dtype": args.dtype,
        "with_metadata": False,
        "macro_cer": None,
        "status": args.record_skip,
        "n_solution": 0,
        "n_submission": 0,
        "cer_by_category": {},
    }
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        for k in ("model", "model_id", "device", "dtype", "with_metadata"):
            if not payload.get(k) and k in cfg:
                payload[k] = cfg[k]
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    if args.record_skip:
        m = record_skip(args)
        csv_path = append_results(m)
        md_path = write_wave1_md(csv_path)
        print(f"recorded skip: {args.record_skip}")
        print(f"appended {csv_path}")
        print(f"wrote {md_path}")
        return
    if args.write_table and not (runs_dir() / args.run_id / "preds.csv").exists():
        path = write_wave1_md()
        print(f"wrote {path}")
        return
    m = eval_run(args.run_id, args.split)
    if args.append_results:
        csv_path = append_results(m)
        md_path = write_wave1_md(csv_path)
        print(f"appended {csv_path}")
        print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
