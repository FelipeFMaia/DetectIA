"""Visualizações para avaliação."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve


def plot_confusion_matrix(labels, preds, output_path: Path,
                          class_names=("IA (0)", "Real (1)")):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, annot_kws={"size": 16})
    ax.set_xlabel("Predito", fontsize=12)
    ax.set_ylabel("Verdadeiro", fontsize=12)
    ax.set_title("Matriz de Confusão", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_roc_curve(labels, probs, output_path: Path):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_pr_curve(labels, probs, output_path: Path):
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, lw=2, label=f"PR (AUC = {pr_auc:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_probability_histogram(labels, probs, output_path: Path):
    """Distribuição de probabilidades preditas separada por classe verdadeira."""
    fig, ax = plt.subplots(figsize=(9, 6))
    probs_ai = probs[labels == 0].numpy()
    probs_real = probs[labels == 1].numpy()
    bins = np.linspace(0, 1, 50)
    ax.hist(probs_ai, bins=bins, alpha=0.6, color="red",
            label=f"Verdadeiro: IA (n={len(probs_ai)})")
    ax.hist(probs_real, bins=bins, alpha=0.6, color="blue",
            label=f"Verdadeiro: Real (n={len(probs_real)})")
    ax.axvline(0.5, color="black", linestyle="--", label="Threshold 0.5")
    ax.set_xlabel("Probabilidade predita de ser Real")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição de probabilidades por classe verdadeira")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()