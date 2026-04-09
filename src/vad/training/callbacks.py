from __future__ import annotations


class MetricTracker:
    """
    Track the best value of a scalar metric.

    The tracker supports both minimization and maximization objectives.
    A new value is considered an improvement only if it exceeds the
    current best by at least ``min_delta`` in the desired direction.

    Examples:
        - For validation loss, use ``mode="min"``.
        - For accuracy or F1 score, use ``mode="max"``.
    """

    def __init__(self, mode: str = "min", min_delta: float = 0.0) -> None:
        """
        Initialize the metric tracker.

        Args:
            mode: Optimization direction:
                - ``"min"`` means lower values are better.
                - ``"max"`` means higher values are better.
            min_delta: Minimum absolute improvement required to update the
                best value. For example, in ``"min"`` mode, a new value must
                be smaller than ``best_value - min_delta``.

        Raises:
            ValueError: If ``mode`` is not ``"min"`` or ``"max"``.
            ValueError: If ``min_delta`` is negative.
        """
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}")

        self.mode = mode
        self.min_delta = min_delta
        self.best_value: float | None = None

    def is_improvement(self, value: float) -> bool:
        """
        Return whether ``value`` improves on the current best.

        The first observed value is always treated as an improvement,
        because no best value exists yet.

        Args:
            value: Current metric value.

        Returns:
            True if ``value`` is better than the current best according to
            ``mode`` and ``min_delta``, otherwise False.
        """
        if self.best_value is None:
            return True

        if self.mode == "min":
            return value < (self.best_value - self.min_delta)

        return value > (self.best_value + self.min_delta)

    def update(self, value: float) -> bool:
        """
        Update the tracked best value if ``value`` is an improvement.

        Args:
            value: Current metric value.

        Returns:
            True if ``best_value`` was updated, otherwise False.
        """
        if self.is_improvement(value):
            self.best_value = value
            return True
        return False


class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    This helper counts consecutive epochs without improvement. Training
    should stop once the number of consecutive non-improving epochs
    reaches ``patience``.

    Example:
        If ``patience=3``, training stops after 3 consecutive epochs
        without improvement.
    """

    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        """
        Initialize the early stopping helper.

        Args:
            patience: Number of consecutive non-improving epochs allowed
                before stopping.
            min_delta: Minimum improvement required to reset the bad-epoch
                counter.
            mode: Optimization direction:
                - ``"min"`` means lower values are better.
                - ``"max"`` means higher values are better.

        Raises:
            ValueError: If ``patience`` is negative.
        """
        if patience < 0:
            raise ValueError(f"patience must be non-negative, got {patience}")

        self.patience = patience
        self.tracker = MetricTracker(mode=mode, min_delta=min_delta)
        self.num_bad_epochs = 0

    @property
    def best_value(self) -> float | None:
        """
        Return the best monitored value seen so far.

        Returns:
            The current best metric value, or None if no value has been
            observed yet.
        """
        return self.tracker.best_value

    def step(self, value: float) -> bool:
        """
        Process a new metric value and decide whether training should stop.

        If the metric improves, the internal bad-epoch counter is reset.
        Otherwise, the counter is incremented.

        Args:
            value: Current metric value.

        Returns:
            True if training should stop, otherwise False.
        """
        if self.tracker.update(value):
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience
