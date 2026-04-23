from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, kendalltau, spearmanr
from statsmodels.stats.contingency_tables import mcnemar


EVALUATOR_ORDER = ["model", "LL", "MG", "RPM"]
EXPERT_FILES = {
    "LL": "LL_combined_summary.xlsx",
    "MG": "MG_combined_summary.xlsx",
    "RPM": "RPM_combined_summary.xlsx",
}
MODEL_FILE = "model_evaluation_combined.xlsx"
LABEL_MAP = {
    "H": "H",
    "h": "H",
    "BS": "BS",
    "bs": "BS",
}


def normalize_label(value) -> str | float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    return LABEL_MAP.get(text, text.upper())


def confidence_to_rating(value) -> float:
    if pd.isna(value):
        return np.nan
    value = float(value)
    value = min(max(value, 0.0), 1.0)
    if value <= 0.20:
        return 1
    if value <= 0.40:
        return 2
    if value <= 0.60:
        return 3
    if value <= 0.80:
        return 4
    return 5


def load_model_df(evaluations_dir: Path) -> pd.DataFrame:
    path = evaluations_dir / MODEL_FILE
    df = pd.read_excel(path).copy()
    df["set"] = df["set"].astype(int)
    df["sample_id"] = df["sample_id"].astype(int)
    df["diagnosis_true"] = df["diagnosis_true"].map(normalize_label)
    df["diagnosis_predicted"] = df["diagnosis_predicted"].map(normalize_label)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["uncertainty"] = pd.to_numeric(df["uncertainty"], errors="coerce")
    df["model_correct"] = df["diagnosis_predicted"] == df["diagnosis_true"]
    df["model_conf_norm"] = df["confidence"]
    df["model_conf_rating"] = df["model_conf_norm"].map(confidence_to_rating)
    return df


def load_expert_df(evaluations_dir: Path, expert_name: str, filename: str) -> pd.DataFrame:
    path = evaluations_dir / filename
    df = pd.read_excel(path).copy()
    df["set"] = df["set"].astype(int)
    df["sample_id"] = df["sample_id"].astype(int)
    df["diagnosis"] = df["diagnosis"].map(normalize_label)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    out = df[["set", "sample_id", "diagnosis", "confidence"]].rename(
        columns={
            "diagnosis": f"{expert_name}_diagnosis",
            "confidence": f"{expert_name}_confidence",
        }
    )
    out[f"{expert_name}_conf_norm"] = (out[f"{expert_name}_confidence"] - 1.0) / 4.0
    out[f"{expert_name}_conf_rating"] = out[f"{expert_name}_confidence"]
    return out


def build_merged_table(evaluations_dir: Path) -> pd.DataFrame:
    merged = load_model_df(evaluations_dir)
    for expert_name, filename in EXPERT_FILES.items():
        merged = merged.merge(
            load_expert_df(evaluations_dir, expert_name, filename),
            on=["set", "sample_id"],
            how="left",
            validate="one_to_one",
        )
        merged[f"{expert_name}_correct"] = merged[f"{expert_name}_diagnosis"] == merged["diagnosis_true"]
    return merged.sort_values(["set", "sample_id"]).reset_index(drop=True)


def build_long_table(merged: pd.DataFrame) -> pd.DataFrame:
    parts = [
        pd.DataFrame({
            "set": merged["set"],
            "sample_id": merged["sample_id"],
            "evaluator": "model",
            "diagnosis_true": merged["diagnosis_true"],
            "diagnosis": merged["diagnosis_predicted"],
            "confidence_raw": merged["confidence"],
            "confidence_norm": merged["model_conf_norm"],
            "confidence_rating": merged["model_conf_rating"],
            "correct": merged["model_correct"],
            "uncertainty": merged["uncertainty"],
        })
    ]

    for expert_name in EXPERT_FILES:
        parts.append(pd.DataFrame({
            "set": merged["set"],
            "sample_id": merged["sample_id"],
            "evaluator": expert_name,
            "diagnosis_true": merged["diagnosis_true"],
            "diagnosis": merged[f"{expert_name}_diagnosis"],
            "confidence_raw": merged[f"{expert_name}_confidence"],
            "confidence_norm": merged[f"{expert_name}_conf_norm"],
            "confidence_rating": merged[f"{expert_name}_conf_rating"],
            "correct": merged[f"{expert_name}_correct"],
            "uncertainty": np.nan,
        }))

    long_df = pd.concat(parts, ignore_index=True)
    long_df["evaluator"] = pd.Categorical(long_df["evaluator"], categories=EVALUATOR_ORDER, ordered=True)
    return long_df.sort_values(["evaluator", "set", "sample_id"]).reset_index(drop=True)


def accuracy_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for evaluator in EVALUATOR_ORDER:
        sub = long_df[long_df["evaluator"] == evaluator]
        correct = int(sub["correct"].sum())
        total = int(sub["correct"].notna().sum())
        result = binomtest(correct, total, p=0.5, alternative="greater")
        rows.append({
            "evaluator": evaluator,
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
            "binomtest_p_gt_0.5": result.pvalue,
        })
    return pd.DataFrame(rows)


def cohen_kappa(labels_a: pd.Series, labels_b: pd.Series, classes: tuple[str, str] = ("H", "BS")) -> float:
    pairs = pd.DataFrame({"a": labels_a, "b": labels_b}).dropna()
    if pairs.empty:
        return np.nan

    conf = pd.crosstab(pairs["a"], pairs["b"]).reindex(index=classes, columns=classes, fill_value=0)
    n = conf.to_numpy().sum()
    if n == 0:
        return np.nan

    po = np.trace(conf.to_numpy()) / n
    row_marg = conf.sum(axis=1).to_numpy() / n
    col_marg = conf.sum(axis=0).to_numpy() / n
    pe = np.sum(row_marg * col_marg)
    if np.isclose(1.0 - pe, 0.0):
        return np.nan
    return (po - pe) / (1.0 - pe)


def pairwise_agreement_tests(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_rows = []
    mcnemar_rows = []
    label_cols = {
        "model": "diagnosis_predicted",
        "LL": "LL_diagnosis",
        "MG": "MG_diagnosis",
        "RPM": "RPM_diagnosis",
    }
    correct_cols = {
        "model": "model_correct",
        "LL": "LL_correct",
        "MG": "MG_correct",
        "RPM": "RPM_correct",
    }

    for left, right in combinations(EVALUATOR_ORDER, 2):
        same_label = (merged[label_cols[left]] == merged[label_cols[right]])
        label_rows.append({
            "evaluator_a": left,
            "evaluator_b": right,
            "label_agreement_rate": float(same_label.mean()),
            "cohen_kappa": cohen_kappa(merged[label_cols[left]], merged[label_cols[right]]),
        })

        a_correct = merged[correct_cols[left]].astype(bool)
        b_correct = merged[correct_cols[right]].astype(bool)
        table = [
            [int(((a_correct) & (b_correct)).sum()), int(((a_correct) & (~b_correct)).sum())],
            [int(((~a_correct) & (b_correct)).sum()), int(((~a_correct) & (~b_correct)).sum())],
        ]
        result = mcnemar(table, exact=True)
        mcnemar_rows.append({
            "evaluator_a": left,
            "evaluator_b": right,
            "table_00_both_correct": table[0][0],
            "table_01_a_only": table[0][1],
            "table_10_b_only": table[1][0],
            "table_11_both_wrong": table[1][1],
            "mcnemar_pvalue": result.pvalue,
        })

    return pd.DataFrame(label_rows), pd.DataFrame(mcnemar_rows)


def overlap_summary(merged: pd.DataFrame) -> pd.DataFrame:
    expert_diag_cols = [f"{name}_diagnosis" for name in EXPERT_FILES]
    expert_conf_cols = [f"{name}_conf_norm" for name in EXPERT_FILES]

    rows = []
    for _, row in merged.iterrows():
        expert_labels = [row[col] for col in expert_diag_cols]
        label_counts = pd.Series(expert_labels).value_counts()
        expert_majority = label_counts.index[0]
        expert_unanimous = int(label_counts.max()) == len(expert_labels)

        expert_mean_conf = float(np.nanmean([row[col] for col in expert_conf_cols]))
        model_conf = float(row["model_conf_norm"])
        conf_gap = abs(model_conf - expert_mean_conf)

        rows.append({
            "set": int(row["set"]),
            "sample_id": int(row["sample_id"]),
            "diagnosis_true": row["diagnosis_true"],
            "model_prediction": row["diagnosis_predicted"],
            "expert_majority_prediction": expert_majority,
            "expert_unanimous": expert_unanimous,
            "model_matches_expert_majority": row["diagnosis_predicted"] == expert_majority,
            "all_experts_match_model": all(label == row["diagnosis_predicted"] for label in expert_labels),
            "all_evaluators_match_truth": (
                row["model_correct"] and row["LL_correct"] and row["MG_correct"] and row["RPM_correct"]
            ),
            "all_evaluators_wrong": (
                (not row["model_correct"]) and (not row["LL_correct"]) and (not row["MG_correct"]) and (not row["RPM_correct"])
            ),
            "model_conf_norm": model_conf,
            "expert_mean_conf_norm": expert_mean_conf,
            "confidence_gap_abs": conf_gap,
            "classification_and_confidence_overlap": (
                (row["diagnosis_predicted"] == expert_majority) and (conf_gap <= 0.15)
            ),
        })

    return pd.DataFrame(rows)


def confidence_relationships(merged: pd.DataFrame) -> pd.DataFrame:
    expert_means = merged[[f"{name}_conf_norm" for name in EXPERT_FILES]].mean(axis=1)
    rows = []
    for expert_name in EXPERT_FILES:
        spearman_corr = spearmanr(merged["model_conf_norm"], merged[f"{expert_name}_conf_norm"], nan_policy="omit")
        kendall_corr = kendalltau(merged["model_conf_norm"], merged[f"{expert_name}_conf_norm"], nan_policy="omit")
        rows.append({
            "comparison": f"model_vs_{expert_name}",
            "spearman_rho": spearman_corr.statistic,
            "spearman_pvalue": spearman_corr.pvalue,
            "kendall_tau": kendall_corr.statistic,
            "kendall_pvalue": kendall_corr.pvalue,
        })

    spearman_corr = spearmanr(merged["model_conf_norm"], expert_means, nan_policy="omit")
    kendall_corr = kendalltau(merged["model_conf_norm"], expert_means, nan_policy="omit")
    rows.append({
        "comparison": "model_vs_mean_expert_confidence",
        "spearman_rho": spearman_corr.statistic,
        "spearman_pvalue": spearman_corr.pvalue,
        "kendall_tau": kendall_corr.statistic,
        "kendall_pvalue": kendall_corr.pvalue,
    })

    spearman_corr = spearmanr(merged["uncertainty"], expert_means, nan_policy="omit")
    kendall_corr = kendalltau(merged["uncertainty"], expert_means, nan_policy="omit")
    rows.append({
        "comparison": "model_uncertainty_vs_mean_expert_confidence",
        "spearman_rho": spearman_corr.statistic,
        "spearman_pvalue": spearman_corr.pvalue,
        "kendall_tau": kendall_corr.statistic,
        "kendall_pvalue": kendall_corr.pvalue,
    })
    return pd.DataFrame(rows)


def confidence_rating_agreement(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for expert_name in EXPERT_FILES:
        exact = merged["model_conf_rating"] == merged[f"{expert_name}_conf_rating"]
        within_one = (merged["model_conf_rating"] - merged[f"{expert_name}_conf_rating"]).abs() <= 1
        spearman_corr = spearmanr(merged["model_conf_rating"], merged[f"{expert_name}_conf_rating"], nan_policy="omit")
        kendall_corr = kendalltau(merged["model_conf_rating"], merged[f"{expert_name}_conf_rating"], nan_policy="omit")
        rows.append({
            "comparison": f"model_vs_{expert_name}",
            "exact_rating_agreement": float(exact.mean()),
            "within_one_rating": float(within_one.mean()),
            "spearman_rho": spearman_corr.statistic,
            "spearman_pvalue": spearman_corr.pvalue,
            "kendall_tau": kendall_corr.statistic,
            "kendall_pvalue": kendall_corr.pvalue,
        })

    expert_mean_rating = merged[[f"{name}_conf_rating" for name in EXPERT_FILES]].mean(axis=1)
    spearman_corr = spearmanr(merged["model_conf_rating"], expert_mean_rating, nan_policy="omit")
    kendall_corr = kendalltau(merged["model_conf_rating"], expert_mean_rating, nan_policy="omit")
    rows.append({
        "comparison": "model_vs_mean_expert_rating",
        "exact_rating_agreement": float((merged["model_conf_rating"] == expert_mean_rating.round()).mean()),
        "within_one_rating": float((merged["model_conf_rating"] - expert_mean_rating).abs().le(1).mean()),
        "spearman_rho": spearman_corr.statistic,
        "spearman_pvalue": spearman_corr.pvalue,
        "kendall_tau": kendall_corr.statistic,
        "kendall_pvalue": kendall_corr.pvalue,
    })
    return pd.DataFrame(rows)


def plot_accuracy_by_evaluator(long_df: pd.DataFrame, out_dir: Path) -> None:
    summary = long_df.groupby("evaluator", observed=True)["correct"].mean().reindex(EVALUATOR_ORDER)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary.index, summary.values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by evaluator")
    for idx, value in enumerate(summary.values):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_evaluator.png", dpi=200)
    plt.close(fig)


def plot_confidence_histograms(long_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    bins = np.arange(0.5, 6.6, 1.0)

    for ax, evaluator in zip(axes.flat, EVALUATOR_ORDER):
        sub = long_df[long_df["evaluator"] == evaluator]
        ax.hist(sub.loc[sub["correct"], "confidence_rating"], bins=bins, alpha=0.7, label="correct")
        ax.hist(sub.loc[~sub["correct"], "confidence_rating"], bins=bins, alpha=0.7, label="wrong")
        ax.set_title(evaluator)
        ax.set_xlabel("Confidence rating (1-5)")
        ax.set_ylabel("Count")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.legend()

    fig.suptitle("Confidence Rating Histograms by evaluator and Correctness")
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_histograms_by_evaluator.png", dpi=200)
    plt.close(fig)


def plot_confidence_rating_histograms(long_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    bins = np.arange(0.5, 6.6, 1.0)

    for ax, evaluator in zip(axes.flat, EVALUATOR_ORDER):
        sub = long_df[long_df["evaluator"] == evaluator]
        ax.hist(sub.loc[sub["correct"], "confidence_rating"], bins=bins, alpha=0.7, label="correct")
        ax.hist(sub.loc[~sub["correct"], "confidence_rating"], bins=bins, alpha=0.7, label="wrong")
        ax.set_title(evaluator)
        ax.set_xlabel("Confidence rating (1-5)")
        ax.set_ylabel("Count")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.legend()

    fig.suptitle("Confidence Rating Histograms by evaluator and Correctness")
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_rating_histograms_by_evaluator.png", dpi=200)
    plt.close(fig)


def plot_model_diagnostics(merged: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bins = np.linspace(0, 1, 16)

    axes[0].hist(merged.loc[merged["model_correct"], "confidence"], bins=bins, alpha=0.75, label="correct")
    axes[0].hist(merged.loc[~merged["model_correct"], "confidence"], bins=bins, alpha=0.75, label="wrong")
    axes[0].set_title("Model Confidence")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(merged.loc[merged["model_correct"], "uncertainty"], bins=bins, alpha=0.75, label="correct")
    axes[1].hist(merged.loc[~merged["model_correct"], "uncertainty"], bins=bins, alpha=0.75, label="wrong")
    axes[1].set_title("Model Uncertainty")
    axes[1].set_xlabel("Uncertainty")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "model_confidence_uncertainty_histograms.png", dpi=200)
    plt.close(fig)


def plot_pairwise_agreement_heatmap(merged: pd.DataFrame, out_dir: Path) -> None:
    label_cols = {
        "model": "diagnosis_predicted",
        "LL": "LL_diagnosis",
        "MG": "MG_diagnosis",
        "RPM": "RPM_diagnosis",
    }
    matrix = np.zeros((len(EVALUATOR_ORDER), len(EVALUATOR_ORDER)))
    for i, left in enumerate(EVALUATOR_ORDER):
        for j, right in enumerate(EVALUATOR_ORDER):
            matrix[i, j] = (merged[label_cols[left]] == merged[label_cols[right]]).mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(EVALUATOR_ORDER)), EVALUATOR_ORDER)
    ax.set_yticks(range(len(EVALUATOR_ORDER)), EVALUATOR_ORDER)
    ax.set_title("Pairwise Label Agreement")
    for i in range(len(EVALUATOR_ORDER)):
        for j in range(len(EVALUATOR_ORDER)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "pairwise_label_agreement_heatmap.png", dpi=200)
    plt.close(fig)


def plot_confidence_scatter(merged: pd.DataFrame, out_dir: Path) -> None:
    expert_mean_conf = merged[[f"{name}_conf_norm" for name in EXPERT_FILES]].mean(axis=1)
    agree = merged["diagnosis_predicted"] == pd.Series(
        [
            pd.Series([row["LL_diagnosis"], row["MG_diagnosis"], row["RPM_diagnosis"]]).value_counts().index[0]
            for _, row in merged.iterrows()
        ]
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        expert_mean_conf,
        merged["model_conf_norm"],
        c=agree.map({True: 1, False: 0}),
        cmap="coolwarm",
        s=70,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.4,
    )
    for _, row in merged.iterrows():
        ax.text(
            expert_mean_conf.loc[row.name] + 0.008,
            row["model_conf_norm"] + 0.008,
            f"S{int(row['set'])}-{int(row['sample_id'])}",
            fontsize=7,
            alpha=0.7,
        )
    ax.set_xlabel("Mean expert confidence (normalized)")
    ax.set_ylabel("Model confidence")
    ax.set_title("Model vs Expert Confidence")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    cbar = fig.colorbar(scatter, ax=ax, ticks=[0, 1], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(["disagree", "agree"])
    fig.tight_layout()
    fig.savefig(out_dir / "model_vs_expert_confidence_scatter.png", dpi=220)
    plt.close(fig)


def plot_confidence_rating_scatter(merged: pd.DataFrame, out_dir: Path) -> None:
    expert_mean_rating = merged[[f"{name}_conf_rating" for name in EXPERT_FILES]].mean(axis=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        expert_mean_rating,
        merged["model_conf_rating"],
        s=70,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.4,
        color="#4c78a8",
    )
    for _, row in merged.iterrows():
        ax.text(
            expert_mean_rating.loc[row.name] + 0.03,
            row["model_conf_rating"] + 0.03,
            f"S{int(row['set'])}-{int(row['sample_id'])}",
            fontsize=7,
            alpha=0.7,
        )
    ax.set_xlabel("Mean expert confidence rating")
    ax.set_ylabel("Model confidence rating")
    ax.set_title("Model vs Expert Confidence Rating")
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0.8, 5.2)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.plot([1, 5], [1, 5], linestyle="--", color="gray", linewidth=1)
    fig.tight_layout()
    fig.savefig(out_dir / "model_vs_expert_confidence_rating_scatter.png", dpi=220)
    plt.close(fig)


def plot_overlap_summary(overlap_df: pd.DataFrame, out_dir: Path) -> None:
    metrics = {
        "All evaluators correct": overlap_df["all_evaluators_match_truth"].mean(),
        "All evaluators wrong": overlap_df["all_evaluators_wrong"].mean(),
        "Model matches expert majority": overlap_df["model_matches_expert_majority"].mean(),
        "All experts match model": overlap_df["all_experts_match_model"].mean(),
        "Class + confidence overlap": overlap_df["classification_and_confidence_overlap"].mean(),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(metrics.keys())
    values = list(metrics.values())
    ax.barh(labels, values, color="#4c78a8")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of samples")
    ax.set_title("Overlap Summary")
    for idx, value in enumerate(values):
        ax.text(value + 0.01, idx, f"{value:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(out_dir / "overlap_summary.png", dpi=200)
    plt.close(fig)


def write_report(
    out_dir: Path,
    merged: pd.DataFrame,
    accuracy_df: pd.DataFrame,
    pairwise_label_df: pd.DataFrame,
    pairwise_mcnemar_df: pd.DataFrame,
    confidence_df: pd.DataFrame,
    confidence_rating_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
) -> None:
    unanimous_correct = int(overlap_df["all_evaluators_match_truth"].sum())
    unanimous_wrong = int(overlap_df["all_evaluators_wrong"].sum())
    class_conf_overlap = overlap_df.loc[overlap_df["classification_and_confidence_overlap"], ["set", "sample_id"]]
    closest_conf = overlap_df.nsmallest(10, "confidence_gap_abs")[["set", "sample_id", "model_prediction", "expert_majority_prediction", "confidence_gap_abs"]]

    lines = [
        "Evaluation Analysis Report",
        "==========================",
        "",
        f"Total samples: {len(merged)}",
        "",
        "Accuracy",
        "--------",
    ]
    for _, row in accuracy_df.iterrows():
        lines.append(
            f"{row['evaluator']}: {int(row['correct'])}/{int(row['total'])} = {row['accuracy']:.3f} "
            f"(binomial p > 0.5: {row['binomtest_p_gt_0.5']:.4g})"
        )

    lines.extend([
        "",
        "Agreement",
        "---------",
    ])
    for _, row in pairwise_label_df.iterrows():
        lines.append(
            f"{row['evaluator_a']} vs {row['evaluator_b']}: agreement={row['label_agreement_rate']:.3f}, "
            f"kappa={row['cohen_kappa']:.3f}"
        )

    lines.extend([
        "",
        "McNemar Tests on Correctness",
        "----------------------------",
    ])
    for _, row in pairwise_mcnemar_df.iterrows():
        lines.append(
            f"{row['evaluator_a']} vs {row['evaluator_b']}: p={row['mcnemar_pvalue']:.4g} "
            f"[a_only={int(row['table_01_a_only'])}, b_only={int(row['table_10_b_only'])}]"
        )

    lines.extend([
        "",
        "Confidence Relationships",
        "------------------------",
    ])
    for _, row in confidence_df.iterrows():
        lines.append(
            f"{row['comparison']}: "
            f"Spearman rho={row['spearman_rho']:.3f}, p={row['spearman_pvalue']:.4g}; "
            f"Kendall tau={row['kendall_tau']:.3f}, p={row['kendall_pvalue']:.4g}"
        )

    lines.extend([
        "",
        "Confidence Rating Relationships (1-5)",
        "------------------------------------",
    ])
    for _, row in confidence_rating_df.iterrows():
        lines.append(
            f"{row['comparison']}: exact={row['exact_rating_agreement']:.3f}, "
            f"within_one={row['within_one_rating']:.3f}, "
            f"Spearman rho={row['spearman_rho']:.3f}, p={row['spearman_pvalue']:.4g}; "
            f"Kendall tau={row['kendall_tau']:.3f}, p={row['kendall_pvalue']:.4g}"
        )

    lines.extend([
        "",
        "Overlap Highlights",
        "------------------",
        f"All evaluators correct on {unanimous_correct} samples.",
        f"All evaluators wrong on {unanimous_wrong} samples.",
        f"Model matches expert majority on {int(overlap_df['model_matches_expert_majority'].sum())} samples.",
        f"Model and expert majority also align in confidence (gap <= 0.15) on {int(overlap_df['classification_and_confidence_overlap'].sum())} samples.",
        "",
        "Samples with classification + confidence overlap:",
    ])
    for _, row in class_conf_overlap.iterrows():
        lines.append(f"set_{int(row['set'])}/sample_{int(row['sample_id'])}")

    lines.extend([
        "",
        "Samples with the smallest model-expert confidence gap:",
    ])
    for _, row in closest_conf.iterrows():
        lines.append(
            f"set_{int(row['set'])}/sample_{int(row['sample_id'])}: "
            f"model={row['model_prediction']}, experts={row['expert_majority_prediction']}, gap={row['confidence_gap_abs']:.3f}"
        )

    (out_dir / "analysis_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    source_dir = Path(__file__).resolve().parent
    project_dir = source_dir.parent
    evaluations_dir = project_dir / "data" / "evaluations"
    out_dir = evaluations_dir / "analysis_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = build_merged_table(evaluations_dir)
    long_df = build_long_table(merged)
    accuracy_df = accuracy_summary(long_df)
    pairwise_label_df, pairwise_mcnemar_df = pairwise_agreement_tests(merged)
    overlap_df = overlap_summary(merged)
    confidence_df = confidence_relationships(merged)
    confidence_rating_df = confidence_rating_agreement(merged)

    merged.to_csv(out_dir / "merged_evaluation_table.csv", index=False)
    long_df.to_csv(out_dir / "long_evaluation_table.csv", index=False)
    accuracy_df.to_csv(out_dir / "accuracy_summary.csv", index=False)
    pairwise_label_df.to_csv(out_dir / "pairwise_label_agreement.csv", index=False)
    pairwise_mcnemar_df.to_csv(out_dir / "pairwise_mcnemar_tests.csv", index=False)
    overlap_df.to_csv(out_dir / "overlap_summary_by_sample.csv", index=False)
    confidence_df.to_csv(out_dir / "confidence_relationships.csv", index=False)
    confidence_rating_df.to_csv(out_dir / "confidence_rating_relationships.csv", index=False)

    plot_accuracy_by_evaluator(long_df, out_dir)
    plot_confidence_histograms(long_df, out_dir)
    plot_confidence_rating_histograms(long_df, out_dir)
    plot_model_diagnostics(merged, out_dir)
    plot_pairwise_agreement_heatmap(merged, out_dir)
    plot_confidence_scatter(merged, out_dir)
    plot_confidence_rating_scatter(merged, out_dir)
    plot_overlap_summary(overlap_df, out_dir)
    write_report(
        out_dir,
        merged,
        accuracy_df,
        pairwise_label_df,
        pairwise_mcnemar_df,
        confidence_df,
        confidence_rating_df,
        overlap_df,
    )

    print(f"Saved analysis outputs to: {out_dir}")


if __name__ == "__main__":
    main()
