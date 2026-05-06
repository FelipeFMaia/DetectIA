"""Módulo de avaliação."""

from .evaluator import evaluate_model, load_checkpoint
from .gradcam import GradCAM
from .plots import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_probability_histogram,
    plot_roc_curve,
)

__all__ = [
    "evaluate_model",
    "load_checkpoint",
    "GradCAM",
    "plot_confusion_matrix",
    "plot_pr_curve",
    "plot_probability_histogram",
    "plot_roc_curve",
]