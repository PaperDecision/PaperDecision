import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np

# Define all possible labels
LABELS = ["oral", "spotlight", "poster", "reject"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def compute_confusion_matrix(data):
    """
    Build confusion matrix:
    Rows = ground truth labels
    Columns = predicted labels
    """
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    label_counts = defaultdict(int)

    for item in data:
        pred = item["prediction"].lower()
        gt = item["ground_truth"].lower()

        # Normalize labels
        if pred not in LABEL_TO_IDX:
            pred = "unknown"
        if gt not in LABEL_TO_IDX:
            gt = "unknown"
        if pred == "unknown" or gt == "unknown":
            continue

        gt_idx = LABEL_TO_IDX[gt]
        pred_idx = LABEL_TO_IDX[pred]

        cm[gt_idx, pred_idx] += 1
        label_counts[gt] += 1

    return cm, label_counts

def print_confusion_matrix(cm, label_counts, title_suffix):
    print(f"\n{'='*80}")
    print(f"CONFUSION MATRIX - {title_suffix}")
    print(f"{'='*80}")

    header = "Ground\\Pred".ljust(12) + "".join(f"{lbl:>10}" for lbl in LABELS) + " | Total"
    print(header)
    print("-" * len(header))

    for i, gt_label in enumerate(LABELS):
        row = f"{gt_label:<12}"
        total = 0
        for j in range(len(LABELS)):
            count = int(cm[i, j])
            row += f"{count:>10}"
            total += count
        row += f" | {total:>5}"
        print(row)

    print("-" * len(header))
    footer = "Total".ljust(12)
    col_totals = cm.sum(axis=0)
    for j in range(len(LABELS)):
        footer += f"{int(col_totals[j]):>10}"
    footer += f" | {int(cm.sum()):>5}"
    print(footer)

def compute_binary_metrics(cm, pos_labels, neg_labels, task_name):
    """
    Generic binary classification metrics calculator

    pos_labels: labels treated as positive class (e.g. ['oral', 'spotlight'])
    neg_labels: labels treated as negative class (e.g. ['poster'])

    Notes:
    - Labels outside pos/neg sets (e.g. 'reject') are:
      * Counted as FN if they appear as predictions for a positive GT
      * Ignored in precision denominator if predicted outside scope
    """
    pos_indices = [LABEL_TO_IDX[l] for l in pos_labels]
    neg_indices = [LABEL_TO_IDX[l] for l in neg_labels]

    # True Positive: GT is Pos, prediction is Pos
    tp = sum(cm[r, c] for r in pos_indices for c in pos_indices)

    # False Positive: GT is Neg, prediction is Pos
    fp = sum(cm[r, c] for r in neg_indices for c in pos_indices)

    # False Negative: GT is Pos, prediction is not Pos
    fn = sum(cm[r, :].sum() for r in pos_indices) - tp

    # True Negative: GT is Neg, prediction is Neg
    tn = sum(cm[r, c] for r in neg_indices for c in neg_indices)

    # Accuracy denominator: only samples whose GT is in Pos or Neg
    total_relevant_gt = sum(cm[r, :].sum() for r in pos_indices + neg_indices)
    accuracy = (tp + tn) / total_relevant_gt if total_relevant_gt > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "task": task_name,
        "acc": accuracy,
        "prec": precision,
        "rec": recall,
        "f1": f1,
        "support_pos": sum(cm[r, :].sum() for r in pos_indices),
        "support_neg": sum(cm[r, :].sum() for r in neg_indices)
    }

def print_hierarchical_metrics(cm, title_suffix):
    """
    Compute and print three hierarchical evaluation metrics:
    1. Accept vs Reject
    2. Higher Tier (Oral + Spotlight) vs Poster
    3. Oral vs Spotlight
    """
    print(f"\n{'='*60}")
    print(f"HIERARCHICAL METRICS - {title_suffix}")
    print(f"{'='*60}")

    # 1. Accept vs Reject
    m1 = compute_binary_metrics(
        cm,
        pos_labels=["oral", "spotlight", "poster"],
        neg_labels=["reject"],
        task_name="Accept(Pos) vs Reject(Neg)"
    )

    # 2. Higher vs Poster
    m2 = compute_binary_metrics(
        cm,
        pos_labels=["oral", "spotlight"],
        neg_labels=["poster"],
        task_name="Higher(Pos) vs Poster(Neg)"
    )

    # 3. Oral vs Spotlight
    m3 = compute_binary_metrics(
        cm,
        pos_labels=["oral"],
        neg_labels=["spotlight"],
        task_name="Oral(Pos) vs Spotlight(Neg)"
    )

    print(f"{'Task':<30} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Supp(Pos/Neg)':>15}")
    print("-" * 80)

    for m in [m1, m2, m3]:
        print(f"{m['task']:<30} {m['acc']:>8.4f} {m['prec']:>8.4f} "
              f"{m['rec']:>8.4f} {m['f1']:>8.4f} "
              f"{m['support_pos']:>5}/{m['support_neg']:<5}")

def compute_detailed_metrics(cm, label_counts):
    """Compute per-class metrics and overall statistics"""
    metrics = {}
    total_samples = cm.sum()
    if total_samples == 0:
        return {
            "overall_accuracy": 0,
            "macro_f1": 0,
            "weighted_f1": 0,
            "per_class": {}
        }

    overall_acc = np.trace(cm) / total_samples

    for i, label in enumerate(LABELS):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(label_counts[label])
        }

    macro_f1 = np.mean([metrics[l]["f1"] for l in LABELS])
    weighted_f1 = np.sum(
        [metrics[l]["f1"] * metrics[l]["support"] for l in LABELS]
    ) / total_samples

    return {
        "overall_accuracy": overall_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": metrics
    }

def print_detailed_metrics(metrics_dict, title_suffix):
    print(f"\n{'='*60}")
    print(f"DETAILED CLASS METRICS - {title_suffix}")
    print(f"{'='*60}")
    if not metrics_dict["per_class"]:
        return

    print(f"Overall Accuracy: {metrics_dict['overall_accuracy']:.4f}")
    print(f"Macro F1:         {metrics_dict['macro_f1']:.4f}")
    print(f"Weighted F1:      {metrics_dict['weighted_f1']:.4f}\n")

    print(f"{'Label':<10} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print("-" * 50)
    for label in LABELS:
        m = metrics_dict["per_class"][label]
        print(f"{label:<10} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['support']:>8}")

def main():
    base_dir = Path("iclr2025_bench/iclr2025_bench/results_prompt_gemini3")
    files = [
        "result_3d.jsonl",
        "result_mllm.jsonl",
        "result_rl.jsonl"
    ]

    # Global aggregation
    total_cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    total_label_counts = defaultdict(int)

    for fname in files:
        path = base_dir / fname
        if not path.exists():
            print(f"⚠️ File not found: {path}")
            continue

        print(f"\n\n{'#' * 80}")
        print(f"Processing: {fname}")
        print(f"{'#' * 80}")

        data = load_jsonl(path)
        cm, label_counts = compute_confusion_matrix(data)

        print_hierarchical_metrics(cm, fname)
        print_confusion_matrix(cm, label_counts, fname)

        metrics = compute_detailed_metrics(cm, label_counts)
        print_detailed_metrics(metrics, fname)

        total_cm += cm
        for label, count in label_counts.items():
            total_label_counts[label] += count

    aggregate_name = "ALL FILES (AGGREGATE)"
    print(f"\n\n{'#' * 80}")
    print(f"SUMMARY: {aggregate_name}")
    print(f"{'#' * 80}")

    if total_cm.sum() > 0:
        print_hierarchical_metrics(total_cm, aggregate_name)
        print_confusion_matrix(total_cm, total_label_counts, aggregate_name)
        total_metrics = compute_detailed_metrics(total_cm, total_label_counts)
        print_detailed_metrics(total_metrics, aggregate_name)
    else:
        print("No data was processed.")

if __name__ == "__main__":
    main()
