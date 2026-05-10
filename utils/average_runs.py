from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT_DIR / "output"
AGGREGATED_ROOT = OUTPUT_ROOT / "averaged_runs"
DEFAULT_RUN_COUNT = 10


@dataclass(frozen=True)
class RunRecord:
    key: tuple[str, str, str]
    timestamp: str
    run_dir: Path
    training_path: Path
    evaluation_path: Path
    evaluation_data: dict
    training_data: dict

    @property
    def sort_key(self) -> tuple[str, float]:
        return self.timestamp, max(self.training_path.stat().st_mtime, self.evaluation_path.stat().st_mtime)


def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def mean_std(v: list[float]) -> dict:
    if not v:
        return {"runs": 0}
    a = np.asarray(v, float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "runs": a.size,
    }


def collect_grouped_runs(root: Path) -> dict[tuple[str, str, str], list[RunRecord]]:
    grouped = defaultdict(list)
    for ep in root.rglob("*_evaluation_metrics.json"):
        tp = ep.parent / ep.name.replace("evaluation", "training")
        if not tp.exists() or len(ep.parts) < 5:
            continue
        try:
            with ep.open("r", encoding="utf-8") as f1, tp.open("r", encoding="utf-8") as f2:
                rec = RunRecord(
                    (ep.parts[-5], ep.parts[-4], ep.parts[-3]),
                    ep.parts[-2],
                    ep.parent,
                    tp,
                    ep,
                    json.load(f1),
                    json.load(f2),
                )
                grouped[rec.key].append(rec)
        except (OSError, ValueError):
            continue
    for k in grouped:
        grouped[k].sort(key=lambda r: r.sort_key, reverse=True)
    return dict(sorted(grouped.items(), key=lambda i: max(r.sort_key[1] for r in i[1]), reverse=True))


def parse_selection(raw: str, count: int) -> list[int]:
    raw = raw.strip().lower()
    # Empty input or explicit 'all' means select every run
    if raw == "":
        return list(range(count))
    if raw in {"default", "all"}:
        return list(range(count))
    sel = set()
    for t in raw.replace(" ", "").split(","):
        if not t:
            continue
        if "-" in t:
            l, r = map(int, t.split("-", 1))
            sel.update(range(min(l, r), max(l, r) + 1))
        else:
            sel.add(int(t))
    if not (res := sorted(i for i in sel if 0 <= i < count)):
        raise ValueError
    return res


def pick_runs(groups: dict) -> list[RunRecord]:
    if not groups:
        return []
    groups_list = list(groups.items())
    if len(groups_list) == 1:
        print(f"Selected only group: {'/'.join(groups_list[0][0])}")
        sel_idx = 0
    else:
        print("Available groups:")
        for i, (k, runs) in enumerate(groups_list):
            print(f"  [{i}] {'/'.join(k)} | runs={len(runs)} | latest={runs[0].timestamp if runs else 'unknown'}")
        while True:
            raw = input("Select experiment group [0]: ").strip()
            if not raw:
                sel_idx = 0
                break
            if raw.isdigit() and 0 <= (sel_idx := int(raw)) < len(groups_list):
                break

    k, runs = groups_list[sel_idx]
    print(f"\nSelected runs for {'/'.join(k)} (newest first):")
    for i, r in enumerate(runs):
        v = r.evaluation_data.get("validation", {})
        acc, f1, b_acc = v.get("accuracy"), v.get("macro_f1"), v.get("balanced_accuracy")
        if is_num(acc) and is_num(f1) and is_num(b_acc):
            print(f"  [{i}] {r.timestamp} | acc={acc:.4f} | macro_f1={f1:.4f} | balanced_acc={b_acc:.4f}")
        else:
            print(f"  [{i}] {r.timestamp}")

    while True:
        try:
            default_selection = "all"
            prompt_hint = f"Run selection [{default_selection}] (Enter/all/0/1-3/0,2): "
            raw_selection = input(prompt_hint).strip()
            return [runs[i] for i in parse_selection(raw_selection, len(runs))]
        except ValueError:
            print("Invalid selection. Examples: Enter, all, 0, 1-3, 0,2,4-6.")


def summarize(runs: list[RunRecord]):
    v_mets, p_mets, cms, labels = defaultdict(list), defaultdict(lambda: defaultdict(list)), [], []
    t_curves, t_final = defaultdict(list), defaultdict(list)

    for r in runs:
        v = r.evaluation_data.get("validation", {})
        for m in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"):
            if is_num(val := v.get(m)):
                v_mets[m].append(float(val))
        for c, cmets in v.get("per_class", {}).items():
            if isinstance(cmets, dict):
                for m, val in cmets.items():
                    if is_num(val):
                        p_mets[c][m].append(float(val))
        if (cm := v.get("confusion_matrix")) is not None:
            cms.append(np.asarray(cm, float))
            labels = v.get("confusion_matrix_labels") or v.get("labels") or labels

        for m in ("train_loss", "val_loss", "train_accuracy", "val_accuracy"):
            if s := r.training_data.get(m):
                arr = np.asarray(s, float)
                t_curves[m].append(arr)
                t_final[m].append(float(arr[-1]))

    v_sum = {m: {**mean_std(vs), "raw": vs} for m, vs in v_mets.items()}
    v_sum["per_class"] = {c: {m: mean_std(vs) for m, vs in cm.items()} for c, cm in p_mets.items()}

    c_sum = {}
    for m, lst in t_curves.items():
        arr = np.stack([a[: (ml := min(len(a) for a in lst))] for a in lst])
        c_sum[m] = {
            "mean": arr.mean(0).tolist(),
            "std": arr.std(0).tolist(),
            "min": arr.min(0).tolist(),
            "max": arr.max(0).tolist(),
            "epochs": ml,
            "raw": arr.tolist(),
        }

    return (
        v_sum,
        labels or [],
        np.mean(cms, axis=0) if cms else np.zeros((0, 0)),
        {"final_metrics": {m: mean_std(vs) for m, vs in t_final.items()}, "curves": c_sum},
    )


def plot_cm(cm: np.ndarray, labels: list[str], path: Path, title: str):
    if not cm.size:
        return
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(s := max(6, min(16, 1.2 * cm.shape[0] + 4)), s))

    lbls = labels if len(labels) == cm.shape[0] else [str(i) for i in range(cm.shape[0])]
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=lbls,
        yticklabels=lbls,
        annot_kws={"size": 15},
        cbar_kws={"fraction": 0.046, "pad": 0.04},
        ax=ax,
    )
    ax.set(title=title, xlabel="Predicted label", ylabel="True label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_all_runs_curves(curves: dict, path: Path, metric: str, title: str, ylabel: str):
    if metric not in curves or "raw" not in curves[metric]:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    c = curves[metric]
    eps = np.arange(1, c["epochs"] + 1)
    raw_runs = np.array(c["raw"])
    for i, run_data in enumerate(raw_runs):
        ax.plot(eps, run_data, alpha=0.5, label=f"Run {i+1}")
    ax.plot(eps, np.array(c["mean"]), color="black", linewidth=2.5, label="Mean")
    ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_curves(curves: dict, path: Path, metrics: list[str], title: str, ylabel: str):
    if not (mets := [m for m in metrics if m in curves]):
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in mets:
        c = curves[m]
        eps, mean, std = np.arange(1, c["epochs"] + 1), np.array(c["mean"]), np.array(c["std"])
        ax.plot(eps, mean, label=m.replace("_", " ").title())
        ax.fill_between(eps, mean - std, mean + std, alpha=0.18)
    ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_evaluation_runs(v_sum: dict, path: Path):
    metrics = ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"]
    mets = [m for m in metrics if m in v_sum]
    if not mets:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(mets))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, m in enumerate(mets):
        data = v_sum[m]
        mean_val = data["mean"]
        min_val = data["min"]
        max_val = data["max"]
        color = colors[i % len(colors)]

        # Min-Max slider bar
        ax.plot([i, i], [min_val, max_val], color=color, linewidth=2)
        # Cap bars
        cap_width = 0.1
        ax.plot([i - cap_width, i + cap_width], [min_val, min_val], color=color, linewidth=2)
        ax.plot([i - cap_width, i + cap_width], [max_val, max_val], color=color, linewidth=2)
        # Mean point
        ax.plot([i], [mean_val], marker="o", markersize=10, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in mets])
    ax.set(title="Evaluation Metrics (Mean with Min-Max Range)", ylabel="Score")
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactively average output runs and save summary plots.")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT), help="Root output directory")
    parser.add_argument("-n", type=int, help="Auto-process all groups with exactly N runs", default=None)
    parser.add_argument("-a", action="store_true", help="Generate an overall summary from averaged_runs directories")
    args = parser.parse_args()
    out_root = Path(args.output_root).resolve()

    if args.a:
        summary = {m: {} for m in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy")}
        for p in AGGREGATED_ROOT.rglob("averaged_metrics.json"):
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    sg = data.get("selected_group", {})
                    cache_version = sg.get("cache_version", "unknown")
                    model = sg.get("model", "unknown")
                    dataset = sg.get("dataset", "unknown")
                    val = data.get("validation", {})

                    for m in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"):
                        if m in val:
                            if dataset not in summary[m]:
                                summary[m][dataset] = {}
                            if model not in summary[m][dataset]:
                                summary[m][dataset][model] = {}

                            summary[m][dataset][model][cache_version] = {
                                k: val[m][k] for k in ["mean", "std", "min", "max", "runs"] if k in val[m]
                            }
            except Exception as e:
                print(f"Failed to read {p}: {e}")

        for m in summary:
            # Sort datasets (devemo before devemo+)
            summary[m] = dict(sorted(summary[m].items()))
            for dataset in summary[m]:
                # Sort models
                summary[m][dataset] = dict(sorted(summary[m][dataset].items()))
                for model in summary[m][dataset]:
                    # Sort cache versions
                    summary[m][dataset][model] = dict(sorted(summary[m][dataset][model].items()))

        out_file = AGGREGATED_ROOT / "overall_summary.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved overall summary to {out_file}")
        return 0

    groups = collect_grouped_runs(out_root)
    if not groups:
        return print(f"No runs found under {out_root}") or 1

    if args.n is not None:
        runs_to_process = [r[: args.n] for r in groups.values() if len(r) >= args.n]
        if not runs_to_process:
            return print(f"No groups found with >= {args.n} runs.") or 1
        print(f"Auto-processing {len(runs_to_process)} group(s) with {args.n} runs...")
    else:
        runs = pick_runs(groups)
        if not runs:
            return 1
        runs_to_process = [runs]

    for runs in runs_to_process:
        k = runs[0].key
        out_dir = AGGREGATED_ROOT.joinpath(*k)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing group: {'/'.join(k)}")

        v_sum, labels, cm_mean, t_sum = summarize(runs)
        cm_norm = (
            np.divide(cm_mean, sums := cm_mean.sum(1, keepdims=True), out=np.zeros_like(cm_mean), where=sums != 0)
            if cm_mean.size
            else cm_mean
        )

        with (out_dir / "averaged_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "validation": v_sum,
                    "training": t_sum,
                    "confusion_matrix_mean": cm_mean.tolist(),
                    "confusion_matrix_row_normalized": cm_norm.tolist(),
                    "confusion_matrix_labels": labels or [str(i) for i in range(len(cm_mean))],
                    "output_root": str(out_root),
                    "selected_group": dict(zip(("cache_version", "model", "dataset"), k)),
                    "selected_runs": [
                        {
                            "timestamp": r.timestamp,
                            "run_dir": str(r.run_dir),
                            "evaluation_path": str(r.evaluation_path),
                            "training_path": str(r.training_path),
                        }
                        for r in runs
                    ],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        plot_cm(cm_mean, labels, out_dir / "confusion_matrix_mean.png", "Mean Confusion Matrix")
        plot_cm(
            cm_norm, labels, out_dir / "confusion_matrix_mean_normalized.png", "Mean Confusion Matrix, Row Normalized"
        )
        plot_curves(
            t_sum.get("curves", {}),
            out_dir / "loss_mean.png",
            ["train_loss", "val_loss"],
            "Mean Training and Validation Loss",
            "Loss",
        )
        plot_curves(
            t_sum.get("curves", {}),
            out_dir / "accuracy_mean.png",
            ["train_accuracy", "val_accuracy"],
            "Mean Training and Validation Accuracy",
            "Accuracy",
        )

        plot_evaluation_runs(v_sum, out_dir / "evaluation_metrics_all_runs.png")

        print(f"Saved averaged results to {out_dir}\nSummary JSON: {out_dir / 'averaged_metrics.json'}")
        for m in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"):
            if (met := v_sum.get(m)) and met.get("runs"):
                print(f"  {m}: {met['mean']:.4f} ± {met['std']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
