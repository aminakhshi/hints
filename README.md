# HiNTS (Higher-Order Interactions in N-Dimensional Time Series)

This repository hosts the source code and tutorial notebooks for HiNTS, a Python package dedicated to the sophisticated analysis of complex systems via multidimensional time series data [Akhshi et al. (2024)](#). It offers a set of functions and tools for detecting and quantifying the directions and strengths of interactions in both deterministic and stochastic interactions within complex systems, encompassing pairwise to higher-order interactions \([Akhshi et al. (2024)](#) & [Tabar et al. (2024)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.14.011050)\).  The main function uses a data-driven approach for characterizing interactions of different orders based on solving a set of linear equations constructed from Kramers-Moyal coefficients derived from statistical moments of N-dimensional multivariate time series. It makes use of the method described by \([Akhshi et al. (2024)](#), [Tabar et al. (2024)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.14.011050), and [Nikakhtar et al. (2023)](https://iopscience.iop.org/article/10.1088/1367-2630/acec63/meta)\).

## Features

- **Universality and applications**: This package is designed to be user-friendly and accessible to a wide range of users, including researchers and practitioners from various fields such as physics, biology, economics, climatology, and engineering, etc.
- **Multidimensional Time Series Processing**: Capable of handling and analyzing data measured from subsystems of a complex system.
- **Higher-Order Interaction Detection**: Identifies the directions and strengths of interactions, encompassing both pairwise and higher-order interactions within a complex system.
- **Identification and Quantification of Directions and Strength of Interactions**: Quantifies both the strengths and directions of interactions in various orders within both the deterministic and stochastic components of a complex system dynamics.
- **Robust Mathematical Foundation**: Based on a solid theoretical framework involving estimations of Kramers-Moyal coefficients from N-dimensional time series.

## Installation

To install `HiNTS`, you can either use pip or install it directly from the source:

```bash
pip install hints-kmcs
```

If you prefer installing from the source, clone the repository and install using the setup script:

```bash
git clone https://github.com/aminakhshi/hints.git
cd hints
python setup.py install
```

## Usage

Below is a basic example of using `HiNTS`. For more detailed examples and tutorials, please refer to our documentation at [notebooks](/examples) and the corresponding papers [Akhshi et al. (2024)](#) and [Tabar et al. (2024)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.14.011050).

```python
import hints

# Setting initial parameters for the interaction coefficient calculator
dt = 1  # Time interval between data points, (1/sampling rate)
orders = [0, 1]  # Defines interaction orders: 0 for constant coefficients (alpha), 1 for pairwise interactions, etc.
mode = 'drift'  # Specifies the type of coefficients to be estimated (choices: 'drift', 'diffusion')

# Importing time series data as a NumPy array or a pandas DataFrame
# You need to 'example.csv' with the path to your data file. The function will automatically detect the format.
# You can also pass the data as a NumPy array.

calulator = hints.kmcc('example.csv', dt=dt, interaction_order=[0,1,2], estimation_mode='drift')

   
# Computing interaction coefficients
interaction_coefficients = calulator.get_coefficients()

# Display the results
print(interaction_coefficients)
```

## Documentation

For more detailed usage and API documentation, please refer to our [documentation](https://hints.readthedocs.io/en/latest/index.html).

## Contributing

We welcome contributions from the community. If you wish to contribute, please check out our [contribution guidelines](#).

## Authors

* Amin Akhshi (amin.akhshi@gmail.com)
* Fatemeh Nikpanjeh (f.nikp77@gmail.com)
* Farnik Nikakhtar (farnik.nikakhtar@yale.edu)
* Laya Parkavousi (laya.parkavousi@ds.mpg.de)
  
## Version History

* 0.1
    * Initial beta release

## Citation

If you use `HiNTS` in your research, please cite our work as follows:

```bibtex
@article{hints2024,
  title={HiNTS: Higher-Order Interactions in N-Dimensional Time Series},
  author={Akhshi et al.},
  journal={#},
  year={2024},
  publisher={#}
}



@article{revealing2024,
  title = {Revealing Higher-Order Interactions in High-Dimensional Complex Systems: A Data-Driven Approach},
  author = {Tabar, M. Reza Rahimi and Nikakhtar, Farnik and Parkavousi, Laya and Akhshi, Amin and Feudel, Ulrike and Lehnertz, Klaus},
  journal = {Phys. Rev. X},
  volume = {14},
  issue = {1},
  pages = {011050},
  numpages = {36},
  year = {2024},
  month = {Mar},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevX.14.011050},
  url = {https://link.aps.org/doi/10.1103/PhysRevX.14.011050}
}


@Article{reconstruction2023,
  author  = {Nikakhtar, Farnik and Parkavousi, Laya and Sahimi, Muhammad and Tabar, M Reza Rahimi and Feudel, Ulrike and Lehnertz, Klaus},
  journal = {New J. Phys.},
  title   = {Data-driven reconstruction of stochastic dynamical equations based on statistical moments},
  year    = {2023},
  number  = {8},
  pages   = {083025},
  volume  = {25},
  doi     = {10.1088/1367-2630/acec63},
}

```

## Contact

For any questions or feedback, please reach out to us at [amin.akhshi@gmail.com](mailto:amin.akhshi@gmail.com).

## License

`HiNTS` is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
