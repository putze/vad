from __future__ import annotations


class MetricTracker:
    """
    Track the best value of a metric according to an optimization mode.
    """

    def __init__(self, mode: str = "min", min_delta: float = 0.0) -> None:
        """
        Initialize the metric tracker.

        Args:
            mode (str): Optimization direction, either "min" or "max".
            min_delta (float): Minimum change required to count as an improvement.

        Raises:
            ValueError: If mode is not "min" or "max".
        """
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.mode = mode
        self.min_delta = min_delta
        self.best_value: float | None = None

    def is_improvement(self, value: float) -> bool:
        """
        Check whether a value improves on the current best.

        Args:
            value (float): Current metric value.

        Returns:
            bool: True if the value is better than the current best, else False.
        """
        if self.best_value is None:
            return True

        if self.mode == "min":
            return value < (self.best_value - self.min_delta)

        return value > (self.best_value + self.min_delta)

    def update(self, value: float) -> bool:
        """
        Update the best value if the new value is an improvement.

        Args:
            value (float): Current metric value.

        Returns:
            bool: True if best_value was updated, else False.
        """
        if self.is_improvement(value):
            self.best_value = value
            return True
        return False


class EarlyStopping:
    """
    Early stopping helper for a monitored validation metric.
    """

    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        """
        Initialize early stopping.

        Args:
            patience (int): Number of consecutive non-improving epochs allowed.
            min_delta (float): Minimum change required to count as an improvement.
            mode (str): Optimization direction, either "min" or "max".
        """
        self.patience = patience
        self.tracker = MetricTracker(mode=mode, min_delta=min_delta)
        self.num_bad_epochs = 0

    @property
    def best_value(self) -> float | None:
        """Return the best monitored value seen so far."""
        return self.tracker.best_value

    def step(self, value: float) -> bool:
        """
        Update the state with a new metric value.

        Args:
            value (float): Current metric value.

        Returns:
            bool: True if training should stop, else False.
        """
        if self.tracker.update(value):
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience
