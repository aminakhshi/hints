Usage
=====

Quick start
-----------

``kmcc`` estimates the drift and diffusion Kramers-Moyal coefficients of an
N-dimensional time series by expanding them in a polynomial interaction basis
and solving the resulting linear system of statistical moments.

.. code-block:: python

   import hints

   # Time series shaped (n_samples, n_state_variables), or a path to a file.
   calculator = hints.kmcc(
       ts_array=data,
       dt=0.01,                     # sampling interval, i.e. 1 / sampling rate
       interaction_order=[0, 1, 2], # 0: constant, 1: pairwise, 2: three-body
       estimation_mode='drift',     # or 'diffusion'
   )

   coefficients = calculator.get_coefficients()
   print(coefficients)

The result is a :class:`pandas.DataFrame`. Rows are the terms of the expansion
(``1``, ``x1``, ``x2``, ``x1x2``, ...) and columns are the estimated components:
``F_x1 ... F_xN`` for the drift, and ``D_x1x1``, ``D_x1x2``, ... for the
diffusion.

Choosing the expansion order
----------------------------

Estimating interactions up to order Z requires statistical moments up to order
2Z to be converged, so the order cannot be raised indefinitely. As a rule of
thumb, reaching Z = 3 for a second-order stationary series needs of the order of
10\ :sup:`4` to 10\ :sup:`6` samples; use Z = 1 or Z = 2 for shorter records.
See Appendix J of :cite:`revealing2024` for the criteria based on moment saturation
and resolution of the tails of the joint probability density.

If the state variables are zero mean, exclude ``0`` from ``interaction_order``
when estimating the drift. Include it to estimate the constant term.

Conventions
-----------

Diffusion coefficients are returned as

.. math::

   D^{(2)}_{ij} = \frac{\langle \Delta x_i \, \Delta x_j \rangle}{\Delta t}

that is, **without** a factor of one half, so for additive noise the estimate is
:math:`(G G^{T})_{ij}` directly.

Recovering the noise amplitude G(x)
-----------------------------------

``get_coefficients`` in diffusion mode returns the coefficients of the expansion
of :math:`D^{(2)}`. The Langevin equation
:math:`\dot{x} = F(x) + G(x)\eta(t)` contains :math:`G`, which is the factor
satisfying :math:`G G^{T} = D^{(2)}` rather than :math:`D^{(2)}` itself. Both are
available directly:

.. code-block:: python

   diffusion = hints.kmcc(ts_array=data, dt=0.01, interaction_order=[0],
                          estimation_mode='diffusion')

   D = diffusion.get_diffusion_matrix()    # (n_points, N, N)
   G = diffusion.get_noise_amplitude()     # (n_points, N, N)

By default both are evaluated at the observed samples; pass ``states=`` to
evaluate them anywhere in state space. :math:`G` is defined only up to an
orthogonal transformation, so ``method='cholesky'`` (the default) returns the
lower triangular factor used in :cite:`revealing2024`, and ``method='sqrt'``
returns the symmetric square root.

For strictly positive data, ``interaction_order=[1]`` is usually the appropriate
choice; for data containing both positive and negative values, start from
``interaction_order=[0]``.

Nothing in the linear system constrains the estimated :math:`D^{(2)}` to be
positive semidefinite. Negative eigenvalues are clipped to zero before
factorization and reported as a warning; a large negative eigenvalue indicates
that the diffusion estimate itself is unreliable at those states.

Numerical diagnostics
---------------------

The moment matrix built from a monomial basis becomes ill conditioned as the
expansion order grows, which is the dominant source of unreliable coefficients.
``kmcc`` records the condition number in ``calculator.condition_number`` after a
call to ``get_coefficients`` and warns when it becomes large. If that happens,
lower ``interaction_order``, rescale the state variables, or select a
least-squares solver:

.. code-block:: python

   calculator = hints.kmcc(ts_array=data, dt=0.01, solver='lstsq')

Optional GPU backend
--------------------

Accumulating the moments can optionally run on PyTorch, which is useful for long
multivariate records. It is an optional extra and the package runs on NumPy by
default:

.. code-block:: python

   calculator = hints.kmcc(ts_array=data, dt=0.01, backend='torch', device='cuda')

Both backends return the same coefficients; only the accumulation moves to the
device, while the linear system is always solved in double precision on the CPU.
Moving data to a GPU costs more than it saves on small problems, so this is
worth enabling only for large ones. If PyTorch or the requested device is not
available, the calculation falls back to the CPU with a warning rather than
failing.

API reference
-------------

.. automodule:: hints.hints
   :members:
   :private-members:
   :undoc-members:
   :show-inheritance:

References
----------

.. bibliography::
   :filter: docname in docnames
