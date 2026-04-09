from vad.training.metrics import BinaryClassificationMetrics


def format_metrics(prefix: str, metrics: BinaryClassificationMetrics) -> str:
    """
    Format aggregated metrics into a compact, human-readable string.

    This is typically used for logging training or validation results
    in the console.

    Args:
        prefix: Short label identifying the split or phase
            (e.g., "train", "val", "test").
        metrics: Aggregated metrics object.

    Returns:
        A formatted string summarizing key metrics.
    """
    return (
        f"{prefix:<5} | "
        f"loss={metrics.loss:.4f} | "
        f"f1={metrics.f1:.4f} | "
        f"prec={metrics.precision:.4f} | "
        f"rec={metrics.recall:.4f} | "
        f"acc={metrics.accuracy:.4f} | "
        f"fp_rate={metrics.false_positive_rate:.4f} | "
        f"fn_rate={metrics.false_negative_rate:.4f}"
    )
