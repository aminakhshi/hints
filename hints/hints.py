import numpy as np
import pandas as pd
import warnings
from skimage.util.shape import view_as_blocks
from itertools import combinations_with_replacement

# Optional PyTorch support for HPC and GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


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
        backend (str):
            Computational backend ('numpy' or 'torch'). Default is 'numpy'.
        device (str):
            Device for torch backend ('cpu', 'cuda', 'mps'). Default is 'cpu'.

        Notes
        ----------
        ## TODO: the choice of options between the filepath and ts_array should be automatically handled by the class.

        * If filepath is provided the priority is given to the filepath and the ts_array is ignored. It's recommended to provide either filepath or ts_array.
        
        * The KMC Calculator requires a time series of N-dimensional data, where N is the number of state variables.

        * If the time series data is zero-mean for the estimation of the drift coefficients, exclude 0 from the order list.

        * To determine the upper limit of the interaction order, refer to Appendix J: "Estimating the Highest Order Z of Expansion from Data.

        * For HPC and GPU acceleration, use backend='torch' with appropriate device.

        Hints
        ----------
        For time series data exhibiting second-order stationarity, the typical number of data points required to estimate interaction strengths up to order Z = 3 is ~10^4 - 10^6 data points. For smaller datasets, it is advisable to choose a lower order of expansion, such as Z = 2 or Z = 1.
        
        For best performance with large datasets, consider using PyTorch backend with GPU acceleration.
        """
        self.filepath = filepath
        self.time_series = ts_array if filepath is None else self._load_data(filepath)
        self.dt = kwargs.get('dt', 1)
        self.order = kwargs.get('interaction_order', [0, 1])
        self.mode = kwargs.get('estimation_mode', 'drift')
        self.window_order = kwargs.get('window_exp_order', 6)
        
        # HPC and PyTorch options
        self.backend = kwargs.get('backend', 'numpy')
        self.device = kwargs.get('device', 'cpu')
        
        # Validate backend choice
        if self.backend == 'torch' and not TORCH_AVAILABLE:
            warnings.warn("PyTorch not available, falling back to numpy backend")
            self.backend = 'numpy'
        
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
        assert self.backend in ['numpy', 'torch'], f'Backend "{self.backend}" is not valid. Choose "numpy" or "torch".'
        
        if self.backend == 'torch' and self.device not in ['cpu', 'cuda', 'mps']:
            warnings.warn(f"Device '{self.device}' may not be supported, using 'cpu'")
            self.device = 'cpu'

    def _prepare_data(self):
        """
        Preprocesses input data for model calculations.

        Calculates differences (increments) between consecutive time points, extracts the
        underlying values, and generates all possible index combinations based on the
        specified interaction order.
        """
        if self.backend == 'torch':
            # Convert to torch tensors for GPU acceleration
            self.time_series_tensor = torch.tensor(self.time_series, dtype=torch.float32, device=self.device)
            self.differences = torch.diff(self.time_series_tensor, dim=0)
            self.values = self.time_series_tensor[:-1, :]
        else:
            # Use numpy arrays
            self.differences = np.diff(self.time_series, axis=0)
            self.values = self.time_series[:-1, :]
            
        self.n_samples, self.dimensions = self.values.shape
        self.index_combinations = self._generate_index_combinations()

    def _to_backend(self, array):
        """
        Convert array to the appropriate backend format.
        
        Args
        ----------
        array : numpy.ndarray or torch.Tensor
            Input array to convert
            
        Returns
        ----------
        Backend-appropriate array
        """
        if self.backend == 'torch':
            if isinstance(array, np.ndarray):
                return torch.tensor(array, dtype=torch.float32, device=self.device)
            return array.to(device=self.device, dtype=torch.float32)
        else:
            if torch is not None and isinstance(array, torch.Tensor):
                return array.cpu().numpy()
            return np.array(array)

    def _from_backend(self, array):
        """
        Convert array from backend format to numpy for compatibility.
        
        Args
        ----------
        array : Backend-specific array
            
        Returns
        ----------
        numpy.ndarray
        """
        if self.backend == 'torch' and torch is not None and isinstance(array, torch.Tensor):
            return array.cpu().numpy()
        return array

    def _matmul(self, a, b):
        """
        Backend-agnostic matrix multiplication.
        
        Args
        ----------
        a, b : arrays
            Matrices to multiply
            
        Returns
        ----------
        Matrix product using appropriate backend
        """
        if self.backend == 'torch':
            return torch.matmul(a, b)
        else:
            return np.matmul(a, b)
    
    def _transpose(self, array):
        """
        Backend-agnostic matrix transpose.
        
        Args
        ----------
        array : Backend-specific array
            
        Returns
        ----------
        Transposed array
        """
        if self.backend == 'torch':
            return array.T
        else:
            return array.T
    
    def _zeros(self, shape):
        """
        Backend-agnostic zeros array creation.
        
        Args
        ----------
        shape : tuple
            Shape of the array
            
        Returns
        ----------
        Zero array using appropriate backend
        """
        if self.backend == 'torch':
            return torch.zeros(shape, dtype=torch.float32, device=self.device)
        else:
            return np.zeros(shape)

    def _prod(self, array, axis=None):
        """
        Backend-agnostic product operation.
        
        Args
        ----------
        array : Backend-specific array
        axis : int or None
            Axis along which to compute product
            
        Returns
        ----------
        Product using appropriate backend
        """
        if self.backend == 'torch':
            if axis is None:
                return torch.prod(array)
            return torch.prod(array, dim=axis)
        else:
            return np.prod(array, axis=axis)

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
        Works with both numpy arrays and PyTorch tensors.

        Returns
        ----------
        tuple:
            * Segmented values as a backend-appropriate array.
            * Remaining values not included in segmentation.
            * Segmented differences as a backend-appropriate array.
            * Remaining differences not included in segmentation.
        """
        window_size = 10 ** self.window_order - 1
        num_windows = self.n_samples // window_size
        remainder = self.n_samples % window_size

        if self.backend == 'torch':
            # PyTorch tensor segmentation
            values_to_segment = self.values[:num_windows * window_size]
            diffs_to_segment = self.differences[:num_windows * window_size]
            
            # Reshape for windowing
            segmented_values = values_to_segment.view(num_windows, window_size, self.dimensions)
            segmented_diffs = diffs_to_segment.view(num_windows, window_size, self.dimensions)
            
            values_remainder = self.values[-remainder:] if remainder > 0 else torch.empty((0, self.dimensions), device=self.device)
            diffs_remainder = self.differences[-remainder:] if remainder > 0 else torch.empty((0, self.dimensions), device=self.device)
        else:
            # Numpy array segmentation using view_as_blocks
            segmented_values = view_as_blocks(self.values[:num_windows * window_size], (window_size, self.dimensions))
            segmented_diffs = view_as_blocks(self.differences[:num_windows * window_size], (window_size, self.dimensions))

            segmented_values = np.squeeze(segmented_values, axis=1)
            segmented_diffs = np.squeeze(segmented_diffs, axis=1)
            
            values_remainder = self.values[-remainder:] if remainder > 0 else np.empty((0, self.dimensions))
            diffs_remainder = self.differences[-remainder:] if remainder > 0 else np.empty((0, self.dimensions))

        return segmented_values, values_remainder, segmented_diffs, diffs_remainder

    def _compute_ts_matrix(self, segment):
        """
        Computes the time series matrix for a given data segment.
        Optimized for both numpy and PyTorch backends.

        Args
        ----------
        segment (numpy.ndarray or torch.Tensor):
            A segment of the time series data.

        Returns
        ----------
        Backend-appropriate array: The calculated time series matrix.
        """
        if self.backend == 'torch':
            # Optimized PyTorch implementation for GPU acceleration
            ts_list = []
            for comb in self.index_combinations:
                if len(comb) == 0:  # Empty combination for constant term
                    ts_list.append(torch.ones(segment.shape[0], device=self.device))
                else:
                    # Use torch.prod for better GPU performance
                    ts_list.append(torch.prod(segment[:, comb], dim=1))
            return torch.stack(ts_list, dim=1)
        else:
            # Original numpy implementation with slight optimization
            return np.array([np.prod(segment[:, comb], axis=1) if len(comb) > 0 
                           else np.ones(segment.shape[0]) 
                           for comb in self.index_combinations]).T

    def _compute_M_matrix(self, ts_matrix):
        """
        Computes the M matrix (statistical moment matrix) to solve the set of linear equations to obtain the interaction strengths.
        Optimized for both numpy and PyTorch backends with potential GPU acceleration.

        Args
        ----------
        ts_matrix (numpy.ndarray or torch.Tensor):
            The time series matrix.

        Returns
        ----------
        Backend-appropriate array:
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
        return self._matmul(self._transpose(ts_matrix), ts_matrix)

    def _compute_Y_matrix(self, ts_matrix, segment_diff):
        """
        Constructs the Y matrix, representing statistical increments matrix from empirical N-dimensional timeseries.
        Optimized for both numpy and PyTorch backends with potential GPU acceleration.

        Args
        ----------
        ts_matrix (numpy.ndarray or torch.Tensor):
            The time series matrix.
        segment_diff (numpy.ndarray or torch.Tensor):
            Differences within the data segment.

        Returns
        ----------
        Backend-appropriate array:
            The calculated Y matrix.

        Notes
        ----------
        * Considerations outlined for the M matrix calculation in Appendix J also apply to the Y matrix computations.
        """

        if self.mode == 'drift':
            return self._matmul(self._transpose(ts_matrix), segment_diff)

        if self.mode == 'diffusion':
            diffusion_indices = list(combinations_with_replacement(range(self.dimensions), 2))
            
            if self.backend == 'torch':
                # Optimized PyTorch implementation
                product_diff_list = []
                for idx in diffusion_indices:
                    if len(idx) == 1:
                        product_diff_list.append(segment_diff[:, idx[0]])
                    else:
                        product_diff_list.append(torch.prod(segment_diff[:, idx], dim=1))
                product_diff = torch.stack(product_diff_list, dim=1)
            else:
                # Original numpy implementation with slight optimization
                product_diff = np.array([np.prod(segment_diff[:, idx], axis=1) 
                                       for idx in diffusion_indices]).T
            
            return self._matmul(self._transpose(ts_matrix), product_diff)

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
        Optimized for both numpy and PyTorch backends with GPU acceleration support.

        Returns
        ----------
        coefficients (pandas.DataFrame):
            A DataFrame containing the estimated coefficients for each term in the polynomial expansion of the interactions. The coefficients are indexed by the corresponding
            terms, representing the interactions between the state variables.

        Notes
        ----------
        If the time series data has a zero mean, exclude 0 from the list of orders. Conversely, to estimate \\(\\alpha\\), set the order to 0 if the data does not have a zero mean.
        
        For large datasets, PyTorch backend with GPU acceleration can provide significant speedup.
        """

        # Initialize matrices using backend-agnostic methods
        M_matrix = self._zeros((len(self.index_combinations), len(self.index_combinations)))
        Y_matrix_dim = len(list(combinations_with_replacement(range(self.dimensions), 2))
                           ) if self.mode == 'diffusion' else self.dimensions
        Y_matrix = self._zeros((len(self.index_combinations), Y_matrix_dim))

        segmented_values, values_remainder, segmented_diffs, diffs_remainder = self._segment_data()
        
        # Process segmented data
        for values, diffs in zip(segmented_values, segmented_diffs):
            # No need to convert - already in appropriate backend format
            ts_matrix = self._compute_ts_matrix(values)
            M_matrix += self._compute_M_matrix(ts_matrix)
            Y_matrix += self._compute_Y_matrix(ts_matrix, diffs)
        
        # Process remaining data if any
        if self.backend == 'torch':
            remaining_check = values_remainder.shape[0] > 0
        else:
            remaining_check = len(values_remainder) > 0
            
        if remaining_check:
            ts_matrix = self._compute_ts_matrix(values_remainder)
            M_matrix += self._compute_M_matrix(ts_matrix)
            Y_matrix += self._compute_Y_matrix(ts_matrix, diffs_remainder)

        # Normalize by number of samples
        M_matrix /= self.n_samples
        Y_matrix /= self.n_samples
        
        # Solve linear system using appropriate backend
        if self.backend == 'torch':
            # Use PyTorch's linear solver for GPU acceleration
            coefficients_tensor = torch.linalg.solve(M_matrix, Y_matrix) / self.dt
            coefficients_array = self._from_backend(coefficients_tensor)
        else:
            # Use numpy's linear solver
            coefficients_array = np.linalg.solve(M_matrix, Y_matrix) / self.dt
        
        # Convert to pandas DataFrame for compatibility
        coefficients = pd.DataFrame(coefficients_array, index=self._construct_keys())
        
        if self.mode == 'diffusion':
            coefficients.columns = np.array([''.join([str(comb[0]), str(comb[1])]) 
                                           for comb in list(combinations_with_replacement(range(self.dimensions), 2))])
        return coefficients

