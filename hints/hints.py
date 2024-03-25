import numpy as np
import pandas as pd
import warnings
from skimage.util.shape import view_as_blocks
from itertools import combinations_with_replacement


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

    def __init__(self, filepath=None, ts_array=None, **kwargs):
        """
        Initialize the KMC Calculator with provided parameters.

        Args
        ----------
        filepath (str):
            Path to the file containing the time series data.
        ts_array (numpy.ndarray):
            Time series data as a 2D numpy array.
        dt (float):
            Time interval between data points.
        interaction_order (int or list or tuple):
            Order of the polynomial to be calculated. If a tuple is provided, the first and second elements represent the lower and upper bounds of the order, respectively.
        estimation_mode (str):
            Mode of calculation ('drift' or 'diffusion').
        window_exp_order (int):
            Exponential order for window size calculation.

        Notes
        ----------
        ## TODO: the choice of options between the filepath and ts_array should be automatically handled by the class.

        * If filepath is provided the priority is given to the filepath and the ts_array is ignored. It's recommended to provide either filepath or ts_array.
        
        * The KMC Calculator requires a time series of N-dimensional data, where N is the number of state variables.

        * If the time series data is zero-mean for the estimation of the drift coefficients, exclude 0 from the order list.

        * To determine the upper limit of the interaction order, refer to Appendix J: "Estimating the Highest Order Z of Expansion from Data.

        Hints
        ----------
        For time series data exhibiting second-order stationarity, the typical number of data points required to estimate interaction strengths up to order Z = 3 is ~10^4 - 10^6 data points. For smaller datasets, it is advisable to choose a lower order of expansion, such as Z = 2 or Z = 1.
        """
        self.filepath = filepath
        self.time_series = ts_array if filepath is None else self._load_data(filepath)
        self.dt = kwargs.get('dt', 1)
        self.order = kwargs.get('interaction_order', [0, 1])
        self.mode = kwargs.get('estimation_mode', 'drift')
        self.window_order = kwargs.get('window_exp_order', 6)
        self._check_inputs()
        self._prepare_data()

    def _load_data(self):
        """
        Loads data from a file into a numpy array. Supports CSV, TXT, NPY, and pickle formats.

        Returns
        ----------
        timeseries (numpy.ndarray): The loaded timeseries from the file as a numpy array
        """
        if not isinstance(self.filepath, str):
            raise ValueError("The filepath must be a string.")

        # Determine the file format
        if self.filepath.endswith('.csv') or self.filepath.endswith('.txt'):
            self.time_series = pd.read_csv(self.filepath).values
        elif self.filepath.endswith('.npy'):
            self.time_series = np.load(self.filepath)
        elif self.filepath.endswith('.pkl') or self.filepath.endswith('.pickle'):
            with open(self.filepath, 'rb') as f:
                self.time_series = pickle.load(f)
            # Ensure the loaded data is a numpy array
            if not isinstance( self.time_series, np.ndarray):
                self.time_series = np.array(self.time_series)
        else:
            raise ValueError("Unsupported file format. Please use CSV, TXT, NPY, or pickle.")

        return self.time_series
    def _check_inputs(self):
        """
        Validates essential inputs for the KMC Calculator.

        Raises
        ----------
        AssertionError: If input data or parameters are invalid.
        """
        assert len(self.time_series.shape) == 2, 'Time series must have (n_samples, dimensions) shape'
        assert self.time_series.shape[0] > 0, 'No data in time series'
        assert (np.array(self.order) >= 0).all(), 'Negative order is not permitted'
        assert self.mode in ['drift', 'diffusion'], f'Mode "{self.mode}" is not valid. Choose "drift" or "diffusion".'

    def _prepare_data(self):
        """
        Preprocesses input data for model calculations.

        Calculates differences (increments) between consecutive time points, extracts the
        underlying values, and generates all possible index combinations based on the
        specified interaction order.
        """
        self.differences = np.diff(self.time_series, axis=0)
        self.values = self.time_series[:-1, :]
        self.n_samples, self.dimensions = self.values.shape
        self.index_combinations = self._generate_index_combinations()

    def _generate_index_combinations(self):
        """
        Creates combinations of indices representing interactions between variables.

        Returns
        ----------
        list: A list of index combinations, where each combination is a tuple.
        """
        if isinstance(self.order, int):
            comb_lengths = np.arange(self.order + 1)
        elif isinstance(self.order, tuple) and len(self.order) == 2:
            comb_lengths = np.arange(self.order[0], self.order[1] + 1)
        else:
            comb_lengths = np.sort(np.array(self.order))

        return [comb for length in comb_lengths for comb in
                combinations_with_replacement(range(self.dimensions), length)]

    def _segment_data(self):
        """
        Divides the data into segments for windowed analysis.

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
        remainder = self.n_samples % window_size

        segmented_values = view_as_blocks(self.values[:num_windows * window_size], (window_size, self.dimensions))
        segmented_diffs = view_as_blocks(self.differences[:num_windows * window_size], (window_size, self.dimensions))

        segmented_values = np.squeeze(segmented_values, axis=1)
        segmented_diffs = np.squeeze(segmented_diffs, axis=1)

        return segmented_values, self.values[-remainder:], segmented_diffs, self.differences[-remainder:]

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
        return np.array([np.prod(segment[:, comb], axis=1) for comb in self.index_combinations]).T

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
        * For reliable estimation of interaction coefficients, ensure the tails of the joint probability distribution functions (PDFs) are sufficiently resolved. This can be assessed by plotting products like x_i^m * p(x_i) for relevant powers 'm' and examining their convergence (refer to Fig. 4 in the appendix of Tabar et al. (2024)[1]).

        * Statistical moments may require longer integration times (T) for proper convergence. Monitor the stability of moments like <x_i^(2k)> as T increases (refer to Fig. 5 in Tabar et al. (2024)[1]).

        * Errors in moment calculations typically decrease as 1/(N*dt)^gamma with gamma ~ 0.5 (refer to Fig. 6 in Tabar et al. (2024)[1]).

        See Also
        ----------
        * Appendix J of the Tabar et al. (2024)[1], PRX for in-depth discussions and guidelines.

        .. [1] Tabar, M.R.R, et al., 2024. Revealing Higher-Order Interactions in High-Dimensional Complex Systems: A Data-Driven Approach. PRX.
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

        if self.mode == 'diffusion':
            diffusion_indices = list(combinations_with_replacement(range(self.dimensions), 2))
            product_diff = np.array([np.prod(segment_diff[:, idx], axis=1) for idx in diffusion_indices]).T
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

    def get_coefficients(self):
        """
        Calculates the coefficients of the Langevin equation from the input time series data.
        This involves computing the M and Y matrices and solving the linear system to estimate
        the coefficients for both the deterministic and stochastic parts of the equation.

        Returns
        ----------
        coefficients (pandas.DataFrame):
            A DataFrame containing the estimated coefficients for each term in the polynomial expansion of the interactions. The coefficients are indexed by the corresponding
            terms, representing the interactions between the state variables.

        Notes
        ----------
        If the time series data has a zero mean, exclude 0 from the list of orders. Conversely, to estimate \(\alpha\), set the order to 0 if the data does not have a zero mean.
        """

        M_matrix = np.zeros((len(self.index_combinations), len(self.index_combinations)))
        Y_matrix_dim = len(combinations_with_replacement(range(self.dimensions), 2)
                           ) if self.mode == 'diffusion' else self.dimensions
        Y_matrix = np.zeros((len(self.index_combinations), Y_matrix_dim))

        segmented_values, values_remainder, segmented_diffs, diffs_remainder = self._segment_data()
        
        for values, diffs in zip(segmented_values, segmented_diffs):
            ts_matrix = self._compute_ts_matrix(values)
            M_matrix += self._compute_M_matrix(ts_matrix)
            Y_matrix += self._compute_Y_matrix(ts_matrix, diffs)
        
        if len(values_remainder) > 0:
            ts_matrix = self._compute_ts_matrix(values_remainder)
            M_matrix += self._compute_M_matrix(ts_matrix)
            Y_matrix += self._compute_Y_matrix(ts_matrix, diffs_remainder)

        M_matrix /= self.n_samples
        Y_matrix /= self.n_samples
        coefficients = np.linalg.solve(M_matrix, Y_matrix) / self.dt
        return pd.DataFrame(coefficients, index=self._construct_keys())

