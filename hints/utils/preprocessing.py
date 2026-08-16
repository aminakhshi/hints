import numpy as np
import logging

# Create a logger object for this module
logger = logging.getLogger(__name__)

def rolling_window(data, window_size, step=None, overlap=None, apply_func=None, padded=False, axis=-1, copy=True):
    """
    Calculate a rolling window over data with specified window length and step size or overlap,
    and optionally apply a function to each window.

    Parameters
    ----------
    data : array_like
        The array to slide over.
    window_size : int
        The size of each window.
    step : int, optional
        The step size between windows. Defaults to 1.
    overlap : float, optional
        The proportion of overlap between consecutive windows (0 <= overlap < 1).
    apply_func : callable, optional
        A function to apply to each rolling window. The function should accept an array and return
        an array or scalar. Defaults to None.
    padded : bool, optional
        If True, pad the array so that the number of windows fits the data length.
    axis : int, optional
        The axis to perform the sliding window calculation on. Defaults to the last axis.
    copy : bool, optional
        Return a copy of the array to avoid side effects. Defaults to True.

    Returns
    -------
    numpy.ndarray
        An array where each slice along the specified axis is a window of data, possibly transformed
        by `apply_func`.

    Notes
    -----
    If both `step` and `overlap` are specified, a ValueError is raised.
    This function uses numpy.lib.stride_tricks.sliding_window_view, available in NumPy 1.20 and later.
    """
    logger.debug("Starting rolling_window function.")
    data = np.asarray(data)
    logger.debug(f"Data converted to array with shape {data.shape} and dtype {data.dtype}.")
    axis = axis % data.ndim  # Ensure positive axis index
    logger.debug(f"Axis set to {axis}.")

    if window_size > data.shape[axis]:
        logger.error("Window length cannot exceed data length along the specified axis.")
        raise ValueError("Window length cannot exceed data length along the specified axis.")

    if overlap is not None:
        if step is not None:
            logger.error("Both 'step' and 'overlap' parameters were provided.")
            raise ValueError("Specify only one of 'step' or 'overlap', not both.")
        if not (0 <= overlap < 1):
            logger.error("Invalid overlap value: %s. Must be between 0 and 1.", overlap)
            raise ValueError("Overlap must be between 0 and 1 (non-inclusive of 1).")
        # Calculate step size based on overlap
        step = max(int(round(window_size * (1 - overlap))), 1)
        logger.info(f"Step size calculated from overlap: {step}")
    else:
        if step is None:
            step = 1
            logger.info("Step size not provided. Defaulting to 1.")

    if step < 1:
        logger.error("Invalid step size: %s. Must be at least 1.", step)
        raise ValueError("Step size must be at least 1.")

    logger.debug("Creating sliding windows using numpy.lib.stride_tricks.sliding_window_view.")
    try:
        windows = np.lib.stride_tricks.sliding_window_view(data, window_size, axis=axis)
    except Exception as e:
        logger.critical("Failed to create sliding windows: %s", e)
        raise

    logger.debug(f"Sliding windows created with shape {windows.shape}.")

    # Adjust for the step size
    if step > 1:
        slicer = [slice(None)] * windows.ndim
        slicer[axis] = slice(0, None, step)
        windows = windows[tuple(slicer)]
        logger.debug(f"Windows adjusted for step size. New shape: {windows.shape}")

    # Handle padding if required
    if padded:
        total_windows = ((data.shape[axis] - window_size) // step) + 1
        expected_length = ((total_windows - 1) * step) + window_size
        padding_needed = expected_length - data.shape[axis]
        if padding_needed > 0:
            pad_width = [(0, 0)] * data.ndim
            pad_width[axis] = (0, padding_needed)
            data = np.pad(data, pad_width, mode='constant', constant_values=0)
            logger.info(f"Data padded along axis {axis} with {padding_needed} zeros.")
            windows = np.lib.stride_tricks.sliding_window_view(data, window_size, axis=axis)
            if step > 1:
                windows = windows[tuple(slicer)]
            logger.debug(f"Sliding windows recalculated after padding. New shape: {windows.shape}")

    # Apply the function to each window if provided
    if apply_func is not None:
        logger.debug("Applying function to each window.")
        # Move the window axis to the first position for easy iteration
        windows = np.moveaxis(windows, axis, 0)
        # Prepare to collect the results
        results = []
        for idx, window in enumerate(windows):
            try:
                result = apply_func(window)
                results.append(result)
                logger.debug(f"Function applied to window {idx}.")
            except Exception as e:
                logger.error(f"Error applying function to window {idx}: {e}")
                raise
        # Stack the results into an array
        try:
            result_array = np.stack(results, axis=0)
            logger.debug("All windows processed and results stacked.")
        except Exception as e:
            logger.critical("Failed to stack results: %s", e)
            raise
        return result_array.copy() if copy else result_array
    else:
        logger.debug("No function applied to windows.")
        return windows.copy() if copy else windows


def center_diff(X: np.ndarray, dt: float) -> np.ndarray:
    """
    Differentiates the input matrix X using a second-order finite difference method.

    Parameters
    ----------
    X : np.ndarray
        The input data matrix. Must be 1D or 2D.
    dt : float
        The time step between each snapshot.

    Returns
    -------
    np.ndarray
        The differentiated data matrix.

    Raises
    ------
    ValueError
        If the input matrix X is not 1D or 2D.

    Notes
    -----
        This method assumes that the snapshots in X are uniformly sampled in time.

    Examples
    --------
    >>> X = np.array([1, 2, 3, 4, 5])
    >>> center_diff(X, dt=1)
    array([ 1.,  1.,  1.,  1.,  1.])
    """
    logger.info("Starting differentiation with dt=%.4f", dt)

    if not isinstance(X, np.ndarray) or X.ndim > 2:
        logger.error("Input data must be a 1D or 2D numpy array. Received ndim=%d", X.ndim if isinstance(X, np.ndarray) else "Non-ndarray")
        raise ValueError("Please ensure that input data is a 1D or 2D array.")

    if X.ndim == 1:
        X = X[None]
        logger.debug("Reshaped 1D input to 2D with shape %s", X.shape)

    X_prime = np.empty_like(X, dtype=np.float64)
    X_prime[:, 1:-1] = (X[:, 2:] - X[:, :-2]) / (2 * dt)
    X_prime[:, 0] = (-3 * X[:, 0] + 4 * X[:, 1] - X[:, 2]) / (2 * dt)
    X_prime[:, -1] = (3 * X[:, -1] - 4 * X[:, -2] + X[:, -3]) / (2 * dt)
    logger.debug("Completed differentiation. Output shape: %s", X_prime.shape)

    return np.squeeze(X_prime)

def differentiate(X: np.ndarray, dt: float, axis: int = -1, method: str = 'center') -> np.ndarray:
    """
    Differentiates the input array X along the specified axis using finite difference methods.

    Parameters
    ----------
    X : np.ndarray
        The input data array. Can be of any dimension.
    dt : float
        The time step between each snapshot.
    axis : int, optional
        The axis along which to perform differentiation. Default is the last axis (-1).
    method : str, optional
        The finite difference method to use. Options are:
        - 'center': Central difference (default)
        - 'forward': Forward difference
        - 'backward': Backward difference

    Returns
    -------
    np.ndarray
        The differentiated data array.

    Raises
    ------
    ValueError
        If the input array X is not a numpy array, if the specified axis is invalid,
        or if the method is not one of the allowed options.

    Notes
    -----
    This method assumes that the data in X are uniformly sampled along the specified axis.

    Examples
    --------
    >>> X = np.array([1, 2, 3, 4, 5])
    >>> differentiate(X, dt=1, method='center')
    array([ 1.,  1.,  1.,  1.,  1.])
    >>> differentiate(X, dt=1, method='forward')
    array([ 1.,  1.,  1.,  1.,  0.])
    >>> differentiate(X, dt=1, method='backward')
    array([ 0.,  1.,  1.,  1.,  1.])
    """
    logger.info("Starting differentiation with dt=%.4f along axis %d using method '%s'", dt, axis, method)

    # Validate input array
    if not isinstance(X, np.ndarray):
        logger.error("Input data must be a numpy array.")
        raise ValueError("Input data must be a numpy array.")

    # Validate axis
    if not isinstance(axis, int):
        logger.error("Axis must be an integer.")
        raise ValueError("Axis must be an integer.")

    axis = axis % X.ndim  # Handle negative axes
    N = X.shape[axis]
    if N < 2:
        logger.error("Axis %d must have at least 2 elements for differentiation. Got %d elements.", axis, N)
        raise ValueError("Axis must have at least 2 elements for differentiation.")

    # Validate method
    allowed_methods = {'center', 'forward', 'backward'}
    if method not in allowed_methods:
        logger.error("Method '%s' is not supported. Choose from %s.", method, allowed_methods)
        raise ValueError(f"Method must be one of {allowed_methods}.")

    X_prime = np.empty_like(X, dtype=np.float64)
    slc_all = [slice(None)] * X.ndim

    if method == 'center':
        if N < 3:
            logger.error("Axis %d must have at least 3 elements for central differentiation. Got %d elements.", axis, N)
            raise ValueError("Axis must have at least 3 elements for central differentiation.")

        # Central differences for internal points
        slc_center = slc_all.copy()
        slc_center[axis] = slice(1, -1)
        slc_plus = slc_all.copy()
        slc_plus[axis] = slice(2, None)
        slc_minus = slc_all.copy()
        slc_minus[axis] = slice(None, -2)

        X_prime[tuple(slc_center)] = (X[tuple(slc_plus)] - X[tuple(slc_minus)]) / (2 * dt)

        # Forward difference at the first point
        slc_first = slc_all.copy()
        slc_first[axis] = 0
        slc_first_p1 = slc_all.copy()
        slc_first_p1[axis] = 1
        slc_first_p2 = slc_all.copy()
        slc_first_p2[axis] = 2

        X_prime[tuple(slc_first)] = (-3 * X[tuple(slc_first)] + 4 * X[tuple(slc_first_p1)] - X[tuple(slc_first_p2)]) / (2 * dt)

        # Backward difference at the last point
        slc_last = slc_all.copy()
        slc_last[axis] = -1
        slc_last_m1 = slc_all.copy()
        slc_last_m1[axis] = -2
        slc_last_m2 = slc_all.copy()
        slc_last_m2[axis] = -3

        X_prime[tuple(slc_last)] = (3 * X[tuple(slc_last)] - 4 * X[tuple(slc_last_m1)] + X[tuple(slc_last_m2)]) / (2 * dt)

    elif method == 'forward':
        if N < 2:
            logger.error("Axis %d must have at least 2 elements for forward differentiation. Got %d elements.", axis, N)
            raise ValueError("Axis must have at least 2 elements for forward differentiation.")

        # Forward differences for all but the last point
        slc_forward = slc_all.copy()
        slc_forward[axis] = slice(0, -1)
        slc_next = slc_all.copy()
        slc_next[axis] = slice(1, None)

        X_prime[tuple(slc_forward)] = (X[tuple(slc_next)] - X[tuple(slc_forward)]) / dt

        # Backward difference at the last point
        slc_last = slc_all.copy()
        slc_last[axis] = -1
        slc_last_m1 = slc_all.copy()
        slc_last_m1[axis] = -2

        X_prime[tuple(slc_last)] = (X[tuple(slc_last)] - X[tuple(slc_last_m1)]) / dt

    elif method == 'backward':
        if N < 2:
            logger.error("Axis %d must have at least 2 elements for backward differentiation. Got %d elements.", axis, N)
            raise ValueError("Axis must have at least 2 elements for backward differentiation.")

        # Backward differences for all but the first point
        slc_backward = slc_all.copy()
        slc_backward[axis] = slice(1, None)
        slc_prev = slc_all.copy()
        slc_prev[axis] = slice(0, -1)

        X_prime[tuple(slc_backward)] = (X[tuple(slc_backward)] - X[tuple(slc_prev)]) / dt

        # Forward difference at the first point
        slc_first = slc_all.copy()
        slc_first[axis] = 0
        slc_first_p1 = slc_all.copy()
        slc_first_p1[axis] = 1

        X_prime[tuple(slc_first)] = (X[tuple(slc_first_p1)] - X[tuple(slc_first)]) / dt

    logger.debug("Completed differentiation. Output shape: %s", X_prime.shape)

    return X_prime
