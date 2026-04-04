def set_plot_style(
    use_seaborn: bool = True,
    style: str = "whitegrid",
    context: str = "talk",
) -> None:
    """
    Configure plotting style for visualizations.

    Args:
        use_seaborn (bool): Whether to apply a Seaborn theme.
        style (str): Seaborn style (e.g., "whitegrid").
        context (str): Seaborn context (e.g., "talk", "paper").

    Notes:
        Falls back silently if Seaborn is not installed.
    """
    if not use_seaborn:
        return

    try:
        import seaborn as sns

        sns.set_theme(style=style, context=context)
    except ImportError:
        pass
