"""
Regression tests for the Kramers-Moyal coefficient calculator.

The tests estimate coefficients for linear stochastic differential equations
whose drift and diffusion are known analytically, and check the bookkeeping
that surrounds the estimator (input handling, segmentation, labeling).

Run with ``pytest`` or directly with ``python tests/test_kmcc.py``.
"""

import os
import pickle
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hints.hints import kmcc

try:  # The torch backend is optional, so its tests are skipped when absent.
    import torch
except ImportError:
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason='PyTorch is not installed')
requires_cuda = pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason='No CUDA device available'
)

DT = 0.01
DRIFT_MATRIX = np.array([[-1.0, 0.5], [-0.5, -2.0]])
NOISE_AMPLITUDE = np.array([0.5, 0.8])

# Standard error of a drift coefficient is sqrt(D_ii / (T * Sigma_jj)), which is
# about 0.02 for the trajectory length below. Tolerances are set at roughly four
# standard errors so the tests are sensitive to real regressions without being
# flaky, and the seeds are fixed so the outcome is deterministic.
DRIFT_TOLERANCE = 0.08
DIFFUSION_TOLERANCE = 0.02


def simulate_ornstein_uhlenbeck(n_samples=1200000, seed=0):
    """Euler-Maruyama realization of dx = A x dt + sigma dW."""
    rng = np.random.default_rng(seed)
    increments = rng.standard_normal((n_samples, 2)) * (NOISE_AMPLITUDE * np.sqrt(DT))
    series = np.zeros((n_samples, 2))
    state = np.zeros(2)
    for index in range(n_samples):
        state = state + state @ DRIFT_MATRIX.T * DT + increments[index]
        series[index] = state
    return series


@pytest.fixture(scope='module')
def ou_series():
    return simulate_ornstein_uhlenbeck()


def test_drift_recovers_known_matrix(ou_series):
    """The linear drift coefficients must reproduce A^T within sampling error."""
    coefficients = kmcc(ts_array=ou_series, dt=DT, interaction_order=[1],
                        estimation_mode='drift').get_coefficients()

    assert list(coefficients.index) == ['x1', 'x2']
    assert list(coefficients.columns) == ['F_x1', 'F_x2']
    assert np.allclose(coefficients.to_numpy(), DRIFT_MATRIX.T, atol=DRIFT_TOLERANCE)


def test_diffusion_recovers_known_amplitudes(ou_series):
    """Diffusion follows the <dx_i dx_j>/dt convention, i.e. (G G^T)_ij.

    The estimate carries a small positive bias because the second moment of the
    increments also contains the squared drift, which contributes at order dt.
    The tolerance accommodates that known bias at this sampling interval.
    """
    coefficients = kmcc(ts_array=ou_series, dt=DT, interaction_order=[0],
                        estimation_mode='diffusion').get_coefficients()

    assert list(coefficients.columns) == ['D_x1x1', 'D_x1x2', 'D_x2x2']
    estimated = coefficients.to_numpy().ravel()
    expected = np.array([NOISE_AMPLITUDE[0] ** 2, 0.0, NOISE_AMPLITUDE[1] ** 2])
    assert np.allclose(estimated, expected, atol=DIFFUSION_TOLERANCE)


def test_segmentation_does_not_change_results(ou_series):
    """Window size only bounds memory use, so coefficients must be invariant.

    This also covers the case where the sample count is an exact multiple of
    the window size, which previously double counted the trailing segment.
    """
    reference = kmcc(ts_array=ou_series, dt=DT, interaction_order=[1],
                     window_exp_order=6).get_coefficients()

    for window_exp_order in (2, 3, 4):
        chunked = kmcc(ts_array=ou_series, dt=DT, interaction_order=[1],
                       window_exp_order=window_exp_order).get_coefficients()
        assert np.allclose(chunked.to_numpy(), reference.to_numpy(), atol=1e-10)


def test_exact_multiple_of_window_size():
    """A sample count that divides evenly by the window size must not be reused."""
    rng = np.random.default_rng(1)
    # 999 windows of 99 samples, plus one row consumed by np.diff.
    series = rng.standard_normal((99 * 999 + 1, 2))

    calculator = kmcc(ts_array=series, dt=DT, interaction_order=[1], window_exp_order=2)
    values, values_remainder, diffs, diffs_remainder = calculator._segment_data()

    assert values.shape == (999, 99, 2)
    assert values_remainder.shape == (0, 2)
    assert diffs.shape == (999, 99, 2)
    assert diffs_remainder.shape == (0, 2)


def test_constant_term_included_for_nonzero_mean():
    """Order 0 estimates the constant drift alpha of dx = alpha dt + sigma dW."""
    rng = np.random.default_rng(2)
    alpha = np.array([1.5, -0.75])
    steps = alpha * DT + rng.standard_normal((200000, 2)) * (0.3 * np.sqrt(DT))
    series = np.cumsum(steps, axis=0)

    coefficients = kmcc(ts_array=series, dt=DT, interaction_order=[0]).get_coefficients()

    assert list(coefficients.index) == ['1']
    assert np.allclose(coefficients.to_numpy().ravel(), alpha, atol=0.05)


def test_index_combinations_cover_requested_orders():
    """Term bookkeeping must match the requested interaction orders."""
    series = np.random.default_rng(3).standard_normal((1000, 3))

    keys = kmcc(ts_array=series, dt=DT, interaction_order=[0, 1])._construct_keys()
    assert keys == ['1', 'x1', 'x2', 'x3']

    keys = kmcc(ts_array=series, dt=DT, interaction_order=[2])._construct_keys()
    assert keys == ['x1x1', 'x1x2', 'x1x3', 'x2x2', 'x2x3', 'x3x3']

    # A two-element tuple is interpreted as an inclusive range of orders.
    keys = kmcc(ts_array=series, dt=DT, interaction_order=(0, 1))._construct_keys()
    assert keys == ['1', 'x1', 'x2', 'x3']


@pytest.mark.parametrize('extension', ['.csv', '.npy', '.pkl'])
def test_file_loading_round_trip(extension):
    """Loading from disk must yield the same array as passing it directly."""
    series = np.random.default_rng(4).standard_normal((500, 2))

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, f'series{extension}')
        if extension == '.csv':
            pd.DataFrame(series, columns=['a', 'b']).to_csv(path, index=False)
        elif extension == '.npy':
            np.save(path, series)
        else:
            with open(path, 'wb') as handle:
                pickle.dump(series, handle)

        loaded = kmcc(path, dt=DT, interaction_order=[1]).time_series

    assert loaded.shape == series.shape
    assert np.allclose(loaded, series)


def test_dataframe_input_matches_array_input():
    """A DataFrame and its underlying array must produce identical results."""
    series = np.random.default_rng(5).standard_normal((2000, 2))
    frame = pd.DataFrame(series, columns=['first', 'second'])

    from_array = kmcc(ts_array=series, dt=DT, interaction_order=[1]).get_coefficients()
    from_frame = kmcc(ts_array=frame, dt=DT, interaction_order=[1]).get_coefficients()

    assert np.allclose(from_array.to_numpy(), from_frame.to_numpy())


def test_missing_input_raises():
    with pytest.raises(ValueError, match='No input data'):
        kmcc(dt=DT)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        kmcc('this_file_does_not_exist.csv', dt=DT)


def test_unsupported_extension_raises():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'series.xyz')
        with open(path, 'w') as handle:
            handle.write('0 0\n')
        with pytest.raises(ValueError, match='Unsupported file format'):
            kmcc(path, dt=DT)


@pytest.mark.parametrize('kwargs, message', [
    ({'dt': 0}, 'dt must be a positive number'),
    ({'dt': -1.0}, 'dt must be a positive number'),
    ({'estimation_mode': 'jump'}, 'is not valid'),
    ({'solver': 'newton'}, 'is not valid'),
    ({'interaction_order': [-1]}, 'Negative order'),
    ({'window_exp_order': 0}, 'window_exp_order must be a positive integer'),
])
def test_invalid_parameters_raise(kwargs, message):
    series = np.random.default_rng(6).standard_normal((100, 2))
    parameters = {'dt': DT, **kwargs}
    with pytest.raises(ValueError, match=message):
        kmcc(ts_array=series, **parameters)


def test_non_finite_values_raise():
    series = np.random.default_rng(7).standard_normal((100, 2))
    series[10, 1] = np.nan
    with pytest.raises(ValueError, match='NaN or infinite'):
        kmcc(ts_array=series, dt=DT)


def test_transposed_input_warns():
    series = np.random.default_rng(8).standard_normal((3, 500))
    with pytest.warns(UserWarning, match='consider transposing'):
        kmcc(ts_array=series, dt=DT, interaction_order=[1])


def test_solvers_agree_on_well_conditioned_system(ou_series):
    """All solvers must coincide when the moment matrix is well conditioned."""
    results = [
        kmcc(ts_array=ou_series, dt=DT, interaction_order=[1],
             solver=solver).get_coefficients().to_numpy()
        for solver in kmcc.SOLVERS
    ]
    for candidate in results[1:]:
        assert np.allclose(candidate, results[0], atol=1e-8)


def test_rank_deficient_expansion_warns():
    """More expansion terms than samples must be reported, not silently solved."""
    series = np.random.default_rng(9).standard_normal((12, 4))
    with pytest.warns(UserWarning, match='rank deficient'):
        kmcc(ts_array=series, dt=DT, interaction_order=[0, 1, 2, 3])


def test_noise_amplitude_recovers_additive_noise(ou_series):
    """G(x) must satisfy G G^T = D and recover the simulated noise amplitudes."""
    calculator = kmcc(ts_array=ou_series, dt=DT, interaction_order=[0],
                      estimation_mode='diffusion')
    origin = np.zeros((1, 2))

    diffusion = calculator.get_diffusion_matrix(states=origin)
    amplitude = calculator.get_noise_amplitude(states=origin)

    assert diffusion.shape == (1, 2, 2)
    assert np.allclose(diffusion[0], diffusion[0].T)
    assert np.allclose(amplitude[0] @ amplitude[0].T, diffusion[0])
    assert np.allclose(np.diag(amplitude[0]), NOISE_AMPLITUDE, atol=DIFFUSION_TOLERANCE)
    # The default factorization is the lower triangular one used in the papers.
    assert np.isclose(amplitude[0][0, 1], 0.0)


def test_noise_amplitude_defaults_to_observed_states(ou_series):
    """Without explicit states, G is evaluated at every observed sample."""
    calculator = kmcc(ts_array=ou_series, dt=DT, interaction_order=[0],
                      estimation_mode='diffusion')
    amplitude = calculator.get_noise_amplitude()

    assert amplitude.shape == (len(ou_series) - 1, 2, 2)
    # Additive noise, so G does not vary across the state space.
    assert np.allclose(amplitude, amplitude[0])


def test_noise_amplitude_tracks_state_dependent_diffusion():
    """A multiplicative-noise system must give the right G at several states."""
    rng = np.random.default_rng(20)
    dt, n_samples = 0.005, 400000
    series = np.zeros((n_samples, 1))
    state = np.zeros(1)
    for index in range(n_samples):
        amplitude = np.sqrt(0.1 + 0.4 * state ** 2)
        state = state + (state - state ** 3) * dt + amplitude * np.sqrt(dt) * rng.standard_normal(1)
        series[index] = state

    calculator = kmcc(ts_array=series, dt=dt, interaction_order=[0, 1, 2],
                      estimation_mode='diffusion')
    query = np.array([[-1.0], [0.0], [1.0]])
    estimated = calculator.get_noise_amplitude(states=query)[:, 0, 0]
    expected = np.sqrt(0.1 + 0.4 * query.ravel() ** 2)

    assert np.allclose(estimated, expected, atol=0.05)


def test_noise_amplitude_methods_agree(ou_series):
    """Both factorizations differ by an orthogonal transform, so G G^T matches."""
    calculator = kmcc(ts_array=ou_series, dt=DT, interaction_order=[0],
                      estimation_mode='diffusion')
    states = np.zeros((1, 2))

    products = []
    for method in ('cholesky', 'sqrt'):
        amplitude = calculator.get_noise_amplitude(states=states, method=method)
        products.append(amplitude[0] @ amplitude[0].T)

    assert np.allclose(products[0], products[1])


def test_diffusion_helpers_reject_drift_mode(ou_series):
    """The helpers are meaningless for a drift fit and must say so."""
    calculator = kmcc(ts_array=ou_series, dt=DT, interaction_order=[0])
    with pytest.raises(ValueError, match="estimation_mode='diffusion'"):
        calculator.get_diffusion_matrix()


def test_noise_amplitude_rejects_unknown_method(ou_series):
    calculator = kmcc(ts_array=ou_series, dt=DT, interaction_order=[0],
                      estimation_mode='diffusion')
    with pytest.raises(ValueError, match="method must be"):
        calculator.get_noise_amplitude(method='lu')


def test_diffusion_matrix_validates_state_shape(ou_series):
    calculator = kmcc(ts_array=ou_series, dt=DT, interaction_order=[0],
                      estimation_mode='diffusion')
    with pytest.raises(ValueError, match='to match the fitted model'):
        calculator.get_diffusion_matrix(states=np.zeros((4, 3)))


def test_non_psd_diffusion_is_clipped_with_warning():
    """An indefinite estimate must warn rather than produce a silent NaN."""
    rng = np.random.default_rng(21)
    series = np.cumsum(rng.standard_normal((5000, 1)) * 0.01, axis=0)
    calculator = kmcc(ts_array=series, dt=0.01, interaction_order=[0, 1],
                      estimation_mode='diffusion')

    # Force an indefinite diffusion by supplying a negative constant term.
    coefficients = calculator.get_coefficients()
    coefficients.iloc[:, :] = 0.0
    coefficients.iloc[0, 0] = -1.0

    with pytest.warns(UserWarning, match='not positive semidefinite'):
        amplitude = calculator.get_noise_amplitude(states=np.zeros((1, 1)),
                                                   coefficients=coefficients)
    assert np.all(np.isfinite(amplitude))
    assert np.allclose(amplitude, 0.0)


@requires_torch
def test_torch_backend_matches_numpy(ou_series):
    """The optional backend must not change the estimated coefficients."""
    for mode, order in (('drift', [0, 1, 2]), ('diffusion', [0])):
        reference = kmcc(ts_array=ou_series, dt=DT, interaction_order=order,
                         estimation_mode=mode).get_coefficients()
        accelerated = kmcc(ts_array=ou_series, dt=DT, interaction_order=order,
                           estimation_mode=mode, backend='torch',
                           device='cpu').get_coefficients()

        assert list(accelerated.index) == list(reference.index)
        assert list(accelerated.columns) == list(reference.columns)
        assert np.allclose(accelerated.to_numpy(), reference.to_numpy(), rtol=1e-9, atol=1e-11)


@requires_torch
def test_torch_single_precision_stays_close(ou_series):
    """Single precision trades accuracy for speed but must stay usable."""
    reference = kmcc(ts_array=ou_series, dt=DT, interaction_order=[1]).get_coefficients()
    reduced = kmcc(ts_array=ou_series, dt=DT, interaction_order=[1], backend='torch',
                   device='cpu', dtype='float32').get_coefficients()

    assert np.allclose(reduced.to_numpy(), reference.to_numpy(), atol=1e-3)


@requires_cuda
def test_cuda_backend_matches_numpy(ou_series):
    """Results computed on the GPU must match the CPU reference."""
    reference = kmcc(ts_array=ou_series, dt=DT, interaction_order=[0, 1, 2]).get_coefficients()
    on_gpu = kmcc(ts_array=ou_series, dt=DT, interaction_order=[0, 1, 2],
                  backend='torch', device='cuda').get_coefficients()

    assert np.allclose(on_gpu.to_numpy(), reference.to_numpy(), rtol=1e-8, atol=1e-10)


@requires_torch
def test_unavailable_device_falls_back_to_cpu():
    """An unknown device is a warning and a fallback, never a crash."""
    series = np.random.default_rng(10).standard_normal((500, 2))
    with pytest.warns(UserWarning, match="Unknown device"):
        calculator = kmcc(ts_array=series, dt=DT, interaction_order=[1],
                          backend='torch', device='definitely_not_a_device')
    assert calculator.device == 'cpu'
    assert calculator.backend == 'torch'
    assert calculator.get_coefficients().shape == (2, 2)


def test_torch_backend_falls_back_when_unavailable(monkeypatch):
    """Without PyTorch installed the backend degrades to numpy with a warning."""
    monkeypatch.setattr('hints.hints._load_torch', lambda: None)
    series = np.random.default_rng(11).standard_normal((500, 2))

    with pytest.warns(UserWarning, match='PyTorch is not installed'):
        calculator = kmcc(ts_array=series, dt=DT, interaction_order=[1], backend='torch')

    assert calculator.backend == 'numpy'
    assert calculator.get_coefficients().shape == (2, 2)


@pytest.mark.parametrize('kwargs, message', [
    ({'backend': 'jax'}, 'is not valid'),
    ({'dtype': 'float16'}, 'is not valid'),
])
def test_invalid_backend_options_raise(kwargs, message):
    series = np.random.default_rng(12).standard_normal((100, 2))
    with pytest.raises(ValueError, match=message):
        kmcc(ts_array=series, dt=DT, **kwargs)


def test_numpy_backend_does_not_import_torch():
    """The default path must not pay the cost of importing PyTorch."""
    import subprocess
    code = (
        'import sys; import hints; import numpy as np; '
        'hints.kmcc(ts_array=np.random.rand(200, 2), dt=0.1).get_coefficients(); '
        'print("torch" in sys.modules)'
    )
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert result.stdout.strip().endswith('False'), result.stdout + result.stderr


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
