"""
HiNTS: Kramers-Moyal coefficient estimation for N-dimensional time series.

This module implements the moment-based estimator described in Tabar et al.
(2024), PRX 14, 011050: drift and diffusion Kramers-Moyal coefficients are
expanded in a multivariate polynomial ("interaction") basis, and the expansion
coefficients are obtained by solving the linear system ``M c = Y`` built from
statistical moments of the data.
"""

import os
import pickle
import warnings
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

#: Condition number of the moment matrix above which a warning is emitted.
ILL_CONDITIONED_THRESHOLD = 1e10

#: File extensions understood by :meth:`kmcc._load_data`.
SUPPORTED_FORMATS = ('.csv', '.txt', '.npy', '.npz', '.pkl', '.pickle', '.mat')

#: Floating point precisions accepted by the ``dtype`` keyword.
DTYPES = ('float64', 'float32')


def _load_torch():
    """
    Imports PyTorch on demand.

    The import is deferred so that ``import hints`` stays fast for users who
    never touch the GPU backend.

    Returns
    ----------
    module or None: The imported ``torch`` module, or None when it is not installed.
    """
    try:
        import torch
    except ImportError:
        return None
    return torch


class kmcc:
    """
    The Kramers-Moyal Coefficients (KMC) Calculator class for analyzing time series data.

    This class analyzes N-dimensional time series data to estimate the interactions in deterministic and stochastic parts
    of a given N-dimensional time series and reconstructs a stochastic differential equation (SDE) within the Kramers-Moyal
    framework. The SDE represents and approximates the dynamics of the underlying system.

    References
    ----------
    Please cite the following paper when using this code:

    - Akhshi, A., et al., 2024. HiNTS: Higher-Order Interactions in N-Dimensional Time Series. arXive.
    - Tabar, M.R.R, et al., 2024. Revealing Higher-Order Interactions in High-Dimensional Complex Systems: A Data-Driven Approach. PRX.
    - Nikakhtar, F., et al. 2023. Data-driven reconstruction of stochastic dynamical equations based on statistical moments. New Journal of Physics.
    """

    #: Linear solvers accepted by the ``solver`` keyword.
    SOLVERS = ('solve', 'lstsq', 'pinv')

    #: Computational backends accepted by the ``backend`` keyword.
    BACKENDS = ('numpy', 'torch')

    def __init__(self, filepath=None, ts_array=None, **kwargs):
        r"""
        Initialize the KMC Calculator with provided parameters.

        Args
        ----------
        filepath (str):
            Path to the file containing the time series data.
        ts_array (numpy.ndarray, pandas.DataFrame or array-like):
            Time series data shaped (n_samples, dimensions).
        dt (float):
            Time interval between data points.
        interaction_order (int or list or tuple):
            Order of the polynomial to be calculated. If a tuple of two values is provided, the first and second elements represent the lower and upper bounds of the order, respectively.
        estimation_mode (str):
            Mode of calculation ('drift' or 'diffusion').
        window_exp_order (int):
            Exponential order for window size calculation. This only controls
            how the data is chunked to bound peak memory use; it does not
            change the estimated coefficients.
        solver (str):
            Linear solver used for ``M c = Y`` ('solve', 'lstsq' or 'pinv').
            Defaults to 'solve', which falls back to a least-squares solution
            if the moment matrix turns out to be singular.
        backend (str):
            Where the statistical moments are accumulated: 'numpy' (default) or
            'torch'. The torch backend is optional and only worth enabling for
            large datasets on a GPU; it produces the same coefficients as numpy.
        device (str):
            Device used by the torch backend ('cpu', 'cuda', 'cuda:0', 'mps').
            Defaults to 'cpu'. Falls back to 'cpu' with a warning when the
            requested device is unavailable.
        dtype (str):
            Floating point precision used by the torch backend to accumulate the
            moments, 'float64' (default) or 'float32'. The numpy backend always
            accumulates in double precision.

        Notes
        ----------
        * Provide either ``filepath`` or ``ts_array``. If both are given, ``filepath`` takes priority.

        * The KMC Calculator requires a time series of N-dimensional data, where N is the number of state variables.

        * If the time series data is zero-mean for the estimation of the drift coefficients, exclude 0 from the order list.

        * To determine the upper limit of the interaction order, refer to Appendix J: "Estimating the Highest Order Z of Expansion from Data".

        * The diffusion coefficients follow the convention
          :math:`D^{(2)}_{ij} = \langle \Delta x_i \Delta x_j \rangle / \Delta t`
          i.e. **without** a factor of 1/2. For additive noise this returns
          :math:`(G G^{T})_{ij}` directly.

        * Only the accumulation of the moment matrices runs on the selected
          backend. The linear system itself is always solved in double precision
          on the CPU, because it is small compared with the data and because the
          monomial moment matrix is often ill conditioned.

        * ``dtype='float32'`` roughly halves memory traffic but carries about
          seven significant digits, which is not enough once the moment matrix
          is ill conditioned. Prefer the default when accuracy matters.

        * The torch backend only pays off for large problems, because moving the
          data to the device costs more than the moment accumulation saves on
          small ones. Measured on one machine, with an expansion of order 2:
          at 4x10^5 samples and 2 state variables it is about 8 times slower
          than numpy, while at 5x10^6 samples and 12 state variables it is about
          8 times faster on a GPU. Benchmark your own workload before switching.

        Hints
        ----------
        For time series data exhibiting second-order stationarity, the typical number of data points required to estimate interaction strengths up to order Z = 3 is ~10^4 - 10^6 data points. For smaller datasets, it is advisable to choose a lower order of expansion, such as Z = 2 or Z = 1.
        """
        self.filepath = filepath
        self.dt = kwargs.get('dt', 1)
        self.order = kwargs.get('interaction_order', [0, 1])
        self.mode = kwargs.get('estimation_mode', 'drift')
        self.window_order = kwargs.get('window_exp_order', 6)
        self.solver = kwargs.get('solver', 'solve')
        self.backend = kwargs.get('backend', 'numpy')
        self.device = kwargs.get('device', 'cpu')
        self.dtype = kwargs.get('dtype', 'float64')
        self._torch = None

        if filepath is not None:
            self.time_series = self._load_data(filepath)
        elif ts_array is not None:
            self.time_series = self._as_array(ts_array)
        else:
            raise ValueError(
                "No input data. Provide either 'filepath' or 'ts_array'."
            )

        self._check_inputs()
        self._resolve_backend()
        self._prepare_data()

    def _resolve_backend(self):
        """
        Imports PyTorch and selects a device, degrading to numpy or CPU when needed.

        A missing PyTorch installation or an unavailable device is reported as a
        warning rather than an error, so that code written for a GPU machine
        still runs elsewhere.
        """
        if self.backend != 'torch':
            return

        self._torch = _load_torch()
        if self._torch is None:
            warnings.warn(
                'PyTorch is not installed; falling back to the numpy backend. '
                "Install it with 'pip install hints-kmcs[torch]'.",
                stacklevel=3,
            )
            self.backend = 'numpy'
            self.device = 'cpu'
            return

        family = self.device.split(':')[0]
        available = {
            'cpu': True,
            'cuda': self._torch.cuda.is_available(),
            'mps': (hasattr(self._torch.backends, 'mps')
                    and self._torch.backends.mps.is_available()),
        }
        if family not in available:
            warnings.warn(
                f"Unknown device '{self.device}'; falling back to 'cpu'.",
                stacklevel=3,
            )
            self.device = 'cpu'
        elif not available[family]:
            warnings.warn(
                f"Device '{self.device}' is not available; falling back to 'cpu'.",
                stacklevel=3,
            )
            self.device = 'cpu'

        if self.device.split(':')[0] == 'mps' and self.dtype == 'float64':
            warnings.warn(
                "The 'mps' backend does not support float64; using float32. "
                'Check the reported condition number before trusting the result.',
                stacklevel=3,
            )
            self.dtype = 'float32'

    @property
    def _torch_dtype(self):
        """The torch dtype matching the configured precision."""
        return getattr(self._torch, self.dtype)

    def _zeros(self, shape):
        """Allocates a zero-filled array on the configured backend."""
        if self.backend == 'torch':
            return self._torch.zeros(shape, dtype=self._torch_dtype, device=self.device)
        return np.zeros(shape)

    def _stack_columns(self, columns):
        """Stacks 1D arrays as the columns of a 2D array on the configured backend."""
        if self.backend == 'torch':
            return self._torch.stack(columns, dim=1)
        return np.column_stack(columns)

    def _to_numpy(self, array):
        """Returns a float64 numpy copy of a backend array."""
        if self.backend == 'torch':
            return array.detach().to('cpu', dtype=self._torch.float64).numpy()
        return np.asarray(array, dtype=float)

    @staticmethod
    def _as_array(data):
        """
        Converts array-like input into a 2D float numpy array.

        Args
        ----------
        data (array-like):
            Input time series as a numpy array, pandas DataFrame/Series or nested sequence.

        Returns
        ----------
        numpy.ndarray: The time series as a 2D array of shape (n_samples, dimensions).
        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.to_numpy()

        array = np.asarray(data, dtype=float)

        if array.ndim == 1:
            array = array[:, np.newaxis]
        if array.ndim != 2:
            raise ValueError(
                f"Time series must be 2D with shape (n_samples, dimensions); got {array.ndim}D."
            )
        return array

    def _load_data(self, filepath):
        """
        Loads data from a file into a numpy array. Supports CSV, TXT, NPY, NPZ, pickle and MAT formats.

        Args
        ----------
        filepath (str):
            Path to the file containing the time series data.

        Returns
        ----------
        timeseries (numpy.ndarray): The loaded timeseries from the file as a numpy array
        """
        if not isinstance(filepath, str):
            raise TypeError("The filepath must be a string.")
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"The file {filepath} does not exist.")

        extension = os.path.splitext(filepath)[1].lower()

        if extension in ('.csv', '.txt'):
            data = self._read_table(filepath)
        elif extension == '.npy':
            data = np.load(filepath)
        elif extension == '.npz':
            with np.load(filepath) as handle:
                data = handle[handle.files[0]]
        elif extension in ('.pkl', '.pickle'):
            with open(filepath, 'rb') as handle:
                data = pickle.load(handle)
        elif extension == '.mat':
            from scipy.io import loadmat
            contents = loadmat(filepath)
            arrays = [value for key, value in contents.items() if not key.startswith('__')]
            if not arrays:
                raise ValueError(f"No data variables found in MAT file {filepath}.")
            data = arrays[0]
        else:
            raise ValueError(
                f"Unsupported file format '{extension}'. Supported formats are: {SUPPORTED_FORMATS}."
            )

        return self._as_array(data)

    @staticmethod
    def _read_table(filepath):
        """
        Reads a delimited text file, detecting whether the first row is a header.

        Args
        ----------
        filepath (str):
            Path to a CSV/TXT file.

        Returns
        ----------
        numpy.ndarray: The parsed numeric contents of the file.
        """
        frame = pd.read_csv(filepath, sep=None, engine='python', header=None)
        first_row_is_numeric = frame.iloc[0].map(
            lambda value: isinstance(value, (int, float, np.number))
            and not isinstance(value, bool)
        ).all()

        if not first_row_is_numeric:
            frame = frame.iloc[1:].reset_index(drop=True).astype(float)

        return frame.to_numpy(dtype=float)

    def _check_inputs(self):
        """
        Validates essential inputs for the KMC Calculator.

        Raises
        ----------
        ValueError: If the input data or parameters are invalid.
        """
        if self.time_series.ndim != 2:
            raise ValueError('Time series must have (n_samples, dimensions) shape')
        if self.time_series.shape[0] < 2:
            raise ValueError('Time series must contain at least two samples')
        if self.time_series.shape[1] < 1:
            raise ValueError('Time series must contain at least one state variable')
        if not np.isfinite(self.time_series).all():
            raise ValueError('Time series contains NaN or infinite values')

        if not isinstance(self.dt, (int, float, np.number)) or self.dt <= 0:
            raise ValueError(f'dt must be a positive number; got {self.dt}')

        if isinstance(self.order, (int, np.integer)):
            orders = [int(self.order)]
        else:
            orders = list(np.asarray(self.order).ravel())
        if not orders:
            raise ValueError('interaction_order must not be empty')
        if not all(float(value).is_integer() for value in orders):
            raise ValueError('interaction_order values must be integers')
        if not all(value >= 0 for value in orders):
            raise ValueError('Negative order is not permitted')

        if self.mode not in ('drift', 'diffusion'):
            raise ValueError(
                f'Mode "{self.mode}" is not valid. Choose "drift" or "diffusion".'
            )

        if self.solver not in self.SOLVERS:
            raise ValueError(
                f'Solver "{self.solver}" is not valid. Choose one of {self.SOLVERS}.'
            )

        if not isinstance(self.window_order, (int, np.integer)) or self.window_order < 1:
            raise ValueError('window_exp_order must be a positive integer')

        if self.backend not in self.BACKENDS:
            raise ValueError(
                f'Backend "{self.backend}" is not valid. Choose one of {self.BACKENDS}.'
            )

        if self.dtype not in DTYPES:
            raise ValueError(
                f'dtype "{self.dtype}" is not valid. Choose one of {DTYPES}.'
            )

        if not isinstance(self.device, str):
            raise TypeError('device must be a string, for example "cpu" or "cuda".')

        n_samples, dimensions = self.time_series.shape
        if dimensions > n_samples:
            warnings.warn(
                f'Time series has more dimensions ({dimensions}) than samples ({n_samples}). '
                'The expected shape is (n_samples, dimensions); consider transposing the input.',
                stacklevel=3,
            )

    def _prepare_data(self):
        """
        Preprocesses input data for model calculations.

        Calculates differences (increments) between consecutive time points, extracts the
        underlying values, and generates all possible index combinations based on the
        specified interaction order.
        """
        if self.backend == 'torch':
            series = self._torch.as_tensor(self.time_series,
                                           dtype=self._torch_dtype,
                                           device=self.device)
            self.differences = self._torch.diff(series, dim=0)
            self.values = series[:-1, :]
        else:
            self.differences = np.diff(self.time_series, axis=0)
            self.values = self.time_series[:-1, :]

        self.n_samples, self.dimensions = self.values.shape
        self.index_combinations = self._generate_index_combinations()
        self.diffusion_indices = list(combinations_with_replacement(range(self.dimensions), 2))

        if len(self.index_combinations) > self.n_samples:
            warnings.warn(
                f'The expansion has more terms ({len(self.index_combinations)}) than samples '
                f'({self.n_samples}); the moment matrix is rank deficient. '
                'Reduce interaction_order or supply more data.',
                stacklevel=3,
            )

    def _generate_index_combinations(self):
        """
        Creates combinations of indices representing interactions between variables.

        Returns
        ----------
        list: A list of index combinations, where each combination is a tuple.
        """
        if isinstance(self.order, (int, np.integer)):
            comb_lengths = np.arange(int(self.order) + 1)
        elif isinstance(self.order, tuple) and len(self.order) == 2:
            comb_lengths = np.arange(self.order[0], self.order[1] + 1)
        else:
            comb_lengths = np.unique(np.asarray(self.order, dtype=int))

        return [comb for length in comb_lengths for comb in
                combinations_with_replacement(range(self.dimensions), int(length))]

    def _segment_data(self):
        """
        Divides the data into segments for windowed analysis.

        Segmentation only bounds the peak memory used while accumulating the
        moment matrices; the accumulated results are identical to processing
        the full array at once.

        Returns
        ----------
        tuple:
            * Segmented values as a NumPy array.
            * Remaining values not included in segmentation.
            * Segmented differences as a NumPy array.
            * Remaining differences not included in segmentation.
        """
        window_size = 10 ** self.window_order - 1
        num_windows = self.n_samples // window_size
        split = num_windows * window_size

        segmented_values = self.values[:split].reshape(num_windows, window_size, self.dimensions)
        segmented_diffs = self.differences[:split].reshape(num_windows, window_size, self.dimensions)

        return (segmented_values, self.values[split:],
                segmented_diffs, self.differences[split:])

    def _compute_ts_matrix(self, segment):
        """
        Computes the time series matrix for a given data segment.

        Args
        ----------
        segment (numpy.ndarray):
            A segment of the time series data.

        Returns
        ----------
        numpy.ndarray: The calculated time series matrix.
        """
        if self.backend == 'torch':
            ones = self._torch.ones(len(segment), dtype=self._torch_dtype, device=self.device)
            return self._stack_columns([
                segment[:, list(comb)].prod(dim=1) if comb else ones
                for comb in self.index_combinations
            ])

        return np.column_stack([
            np.prod(segment[:, comb], axis=1) if comb else np.ones(len(segment))
            for comb in self.index_combinations
        ])

    def _compute_M_matrix(self, ts_matrix):
        """
        Computes the M matrix (statistical moment matrix) to solve the set of linear equations to obtain the interaction strengths.

        Args
        ----------
        ts_matrix (numpy.ndarray):
            The time series matrix.

        Returns
        ----------
        numpy.ndarray:
            The calculated M matrix.

        Notes
        ----------
        * For reliable estimation of interaction coefficients, ensure the tails of the joint probability distribution functions (PDFs) are sufficiently resolved. This can be assessed by plotting products like x_i^m * p(x_i) for relevant powers 'm' and examining their convergence (refer to Fig. 4 in the appendix of Tabar et al. (2024)).

        * Statistical moments may require longer integration times (T) for proper convergence. Monitor the stability of moments like <x_i^(2k)> as T increases (refer to Fig. 5 in Tabar et al. (2024)).

        * Errors in moment calculations typically decrease as 1/(N*dt)^gamma with gamma ~ 0.5 (refer to Fig. 6 in Tabar et al. (2024)).

        See Also
        ----------
        * Appendix J of Tabar, M.R.R, et al., 2024. Revealing Higher-Order Interactions in High-Dimensional Complex Systems: A Data-Driven Approach. PRX, for in-depth discussions and guidelines.
        """
        return ts_matrix.T @ ts_matrix

    def _compute_Y_matrix(self, ts_matrix, segment_diff):
        """
        Constructs the Y matrix, representing statistical increments matrix from empirical N-dimensional timeseries

        Args
        ----------
        ts_matrix (numpy.ndarray):
            The time series matrix.
        segment_diff (numpy.ndarray):
            Differences within the data segment.

        Returns
        ----------
        numpy.ndarray:
            The calculated Y matrix.

        Notes
        ----------
        * Considerations outlined for the M matrix calculation in Appendix J also apply to the Y matrix computations.
        """
        if self.mode == 'drift':
            return ts_matrix.T @ segment_diff

        product_diff = self._stack_columns([
            segment_diff[:, i] * segment_diff[:, j] for i, j in self.diffusion_indices
        ])
        return ts_matrix.T @ product_diff

    def _construct_keys(self):
        """
        Generates descriptive keys for representing coefficients.

        Returns
        ----------
        list: A list of strings representing interaction terms (e.g., 'x1', 'x2x3').
        """
        var_keys = [f'x{i + 1}' for i in range(self.dimensions)]
        return [''.join(var_keys[i] for i in comb) or '1' for comb in self.index_combinations]

    def _construct_columns(self):
        """
        Generates the column labels of the coefficient table.

        Returns
        ----------
        list: Labels for the estimated drift ('F_x1', ...) or diffusion ('D_x1x1', ...) components.
        """
        if self.mode == 'drift':
            return [f'F_x{i + 1}' for i in range(self.dimensions)]
        return [f'D_x{i + 1}x{j + 1}' for i, j in self.diffusion_indices]

    def _solve(self, M_matrix, Y_matrix):
        """
        Solves the linear system ``M c = Y`` with the configured solver.

        Args
        ----------
        M_matrix (numpy.ndarray):
            The statistical moment matrix.
        Y_matrix (numpy.ndarray):
            The statistical increments matrix.

        Returns
        ----------
        numpy.ndarray: The estimated expansion coefficients.
        """
        condition = np.linalg.cond(M_matrix)
        if not np.isfinite(condition) or condition > ILL_CONDITIONED_THRESHOLD:
            warnings.warn(
                f'The moment matrix is ill conditioned (condition number {condition:.3e}). '
                'Coefficients may be unreliable; consider lowering interaction_order, '
                'rescaling the state variables, or using solver="lstsq".',
                stacklevel=3,
            )
        self.condition_number = condition

        if self.solver == 'pinv':
            return np.linalg.pinv(M_matrix) @ Y_matrix
        if self.solver == 'lstsq':
            return np.linalg.lstsq(M_matrix, Y_matrix, rcond=None)[0]

        try:
            return np.linalg.solve(M_matrix, Y_matrix)
        except np.linalg.LinAlgError:
            warnings.warn(
                'The moment matrix is singular; falling back to a least-squares solution.',
                stacklevel=3,
            )
            return np.linalg.lstsq(M_matrix, Y_matrix, rcond=None)[0]

    def get_coefficients(self):
        r"""
        Calculates the coefficients of the Langevin equation from the input time series data.
        This involves computing the M and Y matrices and solving the linear system to estimate
        the coefficients for both the deterministic and stochastic parts of the equation.

        Returns
        ----------
        coefficients (pandas.DataFrame):
            A DataFrame containing the estimated coefficients for each term in the polynomial expansion of the interactions. The coefficients are indexed by the corresponding
            terms, representing the interactions between the state variables. Columns are
            the drift components ('F_x1', ...) or the diffusion components ('D_x1x1', ...).

        Notes
        ----------
        If the time series data has a zero mean, exclude 0 from the list of orders. Conversely, to estimate :math:`\alpha`, set the order to 0 if the data does not have a zero mean.
        """
        n_terms = len(self.index_combinations)
        Y_matrix_dim = len(self.diffusion_indices) if self.mode == 'diffusion' else self.dimensions

        M_matrix = self._zeros((n_terms, n_terms))
        Y_matrix = self._zeros((n_terms, Y_matrix_dim))

        segmented_values, values_remainder, segmented_diffs, diffs_remainder = self._segment_data()

        for values, diffs in zip(segmented_values, segmented_diffs):
            ts_matrix = self._compute_ts_matrix(values)
            M_matrix += self._compute_M_matrix(ts_matrix)
            Y_matrix += self._compute_Y_matrix(ts_matrix, diffs)

        if len(values_remainder) > 0:
            ts_matrix = self._compute_ts_matrix(values_remainder)
            M_matrix += self._compute_M_matrix(ts_matrix)
            Y_matrix += self._compute_Y_matrix(ts_matrix, diffs_remainder)

        M_matrix = self._to_numpy(M_matrix) / self.n_samples
        Y_matrix = self._to_numpy(Y_matrix) / self.n_samples

        coefficients = self._solve(M_matrix, Y_matrix) / self.dt
        return pd.DataFrame(coefficients,
                            index=self._construct_keys(),
                            columns=self._construct_columns())
