from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class ExperimentPaths:
    """
    Container for filesystem paths associated with a single experiment run.
    """

    experiment_name: str
    run_name: str
    log_dir: Path
    checkpoint_dir: Path

    @classmethod
    def create(
        cls,
        experiment_name: str,
        log_root: str | Path = "runs",
        checkpoint_root: str | Path = "checkpoints",
        run_name: str | None = None,
    ) -> "ExperimentPaths":
        """
        Create directories and return an ExperimentPaths instance.

        Directory structure:
            log_dir        = log_root / experiment_name / run_name
            checkpoint_dir = checkpoint_root / experiment_name / run_name

        If ``run_name`` is not provided, a timestamp is used to ensure
        uniqueness.

        Args:
            experiment_name: Name of the experiment.
            log_root: Root directory for logs.
            checkpoint_root: Root directory for checkpoints.
            run_name: Optional custom run name. If None, a timestamp
                (YYYYMMDD_HHMMSS) is used.

        Returns:
            An initialized ExperimentPaths object with created directories.
        """
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_dir = Path(log_root) / experiment_name / run_name
        checkpoint_dir = Path(checkpoint_root) / experiment_name / run_name

        log_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            experiment_name=experiment_name,
            run_name=run_name,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
        )
