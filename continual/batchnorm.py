def normalise_momentum(num_steps: int, base_mom=0.1):
    """Normalise momentum in BatchNorm for frame-by-frame updates

    Args:
        num_steps (int): Number of steps considered in a time-series
        base_mom (float, optional): Momentum used if the whole time-series was processed at once. Defaults to 0.1.

    Returns:
        float: Momentum normalised to `num_steps`-fold update-steps
    """
    return 2 / (num_steps * (2 / base_mom - 1) + 1)
