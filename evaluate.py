import json
import numpy as np
import pandas as pd
from config import EXTRACTED_DIR, TABLES_DIR, MODELS, PROMPT_CONDITIONS


def load_extracted(model_key):
    path = EXTRACTED_DIR / f"{model_key}.jsonl"
    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def compute_accuracy(records):
    if not records:
        return 0.0
    correct = sum(1 for r in records if r.get("is_correct", False))
    return correct / len(records) * 100


def compute_bias_aligned_rate(records):
    errors = [r for r in records if not r.get("is_correct", False) and r.get("extracted_answer") is not None]
    if not errors:
        return 0.0
    bias_aligned = sum(1 for r in errors if r.get("is_bias_aligned_error", False))
    return bias_aligned / len(errors) * 100


def compute_extraction_failure_rate(records):
    if not records:
        return 0.0
    failed = sum(1 for r in records if r.get("extracted_answer") is None)
    return failed / len(records) * 100


def mcnemar_test(records_a, records_b):
    id_to_a = {r["sample_id"]: r.get("is_correct", False) for r in records_a}
    id_to_b = {r["sample_id"]: r.get("is_correct", False) for r in records_b}
    common_ids = set(id_to_a.keys()) & set(id_to_b.keys())

    b = sum(1 for sid in common_ids if id_to_a[sid] and not id_to_b[sid])
    c = sum(1 for sid in common_ids if not id_to_a[sid] and id_to_b[sid])

    if b + c == 0:
        return 1.0, 0.0
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2_stat, df=1)
    return p_value, chi2_stat


def bootstrap_ci(records_a, records_b, n_bootstrap=10000):
    id_to_a = {r["sample_id"]: 1 if r.get("is_correct", False) else 0 for r in records_a}
    id_to_b = {r["sample_id"]: 1 if r.get("is_correct", False) else 0 for r in records_b}
    common_ids = sorted(set(id_to_a.keys()) & set(id_to_b.keys()))

    arr_a = np.array([id_to_a[sid] for sid in common_ids])
    arr_b = np.array([id_to_b[sid] for sid in common_ids])
    n = len(common_ids)

    diffs = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        diff = arr_b[idx].mean() - arr_a[idx].mean()
        diffs.append(diff)

    lower = np.percentile(diffs, 2.5) * 100
    upper = np.percentile(diffs, 97.5) * 100
    return lower, upper


def generate_main_table(model_keys=None):
    model_keys = model_keys or list(MODELS.keys())

    rows = []
    for mk in model_keys:
        try:
            records = load_extracted(mk)
        except FileNotFoundError:
            continue

        row = {"Model": mk}
        for cond in PROMPT_CONDITIONS:
            cond_records = [r for r in records if r["prompt_condition"] == cond]
            acc = compute_accuracy(cond_records)
            row[cond] = f"{acc:.2f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\nAccuracy (%):")
    print(df.to_string(index=False))

    table_path = TABLES_DIR / "main_accuracy.csv"
    df.to_csv(table_path, index=False)
    return df


def generate_bias_table(model_keys=None):
    model_keys = model_keys or list(MODELS.keys())

    rows = []
    for mk in model_keys:
        try:
            records = load_extracted(mk)
        except FileNotFoundError:
            continue

        row = {"Model": mk}
        for cond in PROMPT_CONDITIONS:
            cond_records = [r for r in records if r["prompt_condition"] == cond]
            rate = compute_bias_aligned_rate(cond_records)
            row[cond] = f"{rate:.2f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\nBias-aligned error rate (%):")
    print(df.to_string(index=False))

    table_path = TABLES_DIR / "bias_aligned_rate.csv"
    df.to_csv(table_path, index=False)
    return df


def generate_domain_table(model_keys=None):
    model_keys = model_keys or list(MODELS.keys())

    rows = []
    for mk in model_keys:
        try:
            records = load_extracted(mk)
        except FileNotFoundError:
            continue

        topics = sorted(set(r["topic"] for r in records))
        for cond in PROMPT_CONDITIONS:
            row = {"Model": mk, "Condition": cond}
            for topic in topics:
                subset = [
                    r for r in records
                    if r["prompt_condition"] == cond and r["topic"] == topic
                ]
                row[topic] = f"{compute_accuracy(subset):.2f}"
            rows.append(row)

    df = pd.DataFrame(rows)
    print("\nPer-domain accuracy (%):")
    print(df.to_string(index=False))

    table_path = TABLES_DIR / "domain_accuracy.csv"
    df.to_csv(table_path, index=False)
    return df


def generate_significance_table(model_keys=None):
    model_keys = model_keys or list(MODELS.keys())

    rows = []
    for mk in model_keys:
        try:
            records = load_extracted(mk)
        except FileNotFoundError:
            continue

        baseline_recs = [r for r in records if r["prompt_condition"] == "baseline"]
        for cond in PROMPT_CONDITIONS:
            if cond == "baseline":
                continue
            cond_recs = [r for r in records if r["prompt_condition"] == cond]
            if not cond_recs:
                continue

            p_val, chi2_stat = mcnemar_test(baseline_recs, cond_recs)
            ci_low, ci_high = bootstrap_ci(baseline_recs, cond_recs)
            acc_base = compute_accuracy(baseline_recs)
            acc_cond = compute_accuracy(cond_recs)
            delta = acc_cond - acc_base

            rows.append({
                "Model": mk,
                "Comparison": f"baseline vs {cond}",
                "Baseline Acc": f"{acc_base:.2f}",
                "Condition Acc": f"{acc_cond:.2f}",
                "Delta (pp)": f"{delta:+.2f}",
                "McNemar p": f"{p_val:.4f}",
                "95% CI": f"[{ci_low:+.2f}, {ci_high:+.2f}]",
                "Significant": "Yes" if p_val < (0.05 / 4) else "No",
            })

    df = pd.DataFrame(rows)
    print("\nSignificance tests:")
    print(df.to_string(index=False))

    table_path = TABLES_DIR / "significance.csv"
    df.to_csv(table_path, index=False)
    return df


def generate_latex_main_table(model_keys=None):
    model_keys = model_keys or list(MODELS.keys())

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Exact match accuracy (\%) across prompt conditions. Best result per model in \textbf{bold}.}")
    lines.append(r"\label{tab:main}")
    lines.append(r"\begin{tabular}{l" + "c" * len(PROMPT_CONDITIONS) + "}")
    lines.append(r"\hline")

    header = "Model & " + " & ".join(
        c.replace("_", r"\_").capitalize() for c in PROMPT_CONDITIONS
    ) + r" \\"
    lines.append(header)
    lines.append(r"\hline")

    for mk in model_keys:
        try:
            records = load_extracted(mk)
        except FileNotFoundError:
            continue

        accs = {}
        for cond in PROMPT_CONDITIONS:
            cond_records = [r for r in records if r["prompt_condition"] == cond]
            accs[cond] = compute_accuracy(cond_records)

        best = max(accs.values()) if accs else -1
        cells = []
        for cond in PROMPT_CONDITIONS:
            val = accs.get(cond, 0)
            s = f"{val:.2f}"
            if val == best:
                s = r"\textbf{" + s + "}"
            cells.append(s)

        model_display = mk.replace("_", r"\_")
        lines.append(f"{model_display} & " + " & ".join(cells) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    out_path = TABLES_DIR / "main_accuracy.tex"
    with open(out_path, "w") as f:
        f.write(latex)
    return latex


def run_full_evaluation(model_keys=None):
    generate_main_table(model_keys)
    generate_bias_table(model_keys)
    generate_domain_table(model_keys)
    generate_significance_table(model_keys)
    generate_latex_main_table(model_keys)
    print(f"\nAll tables saved to {TABLES_DIR}")
