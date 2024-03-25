"""
Example case estimation of the Lorenz 96 model.

Created on 2024-03-10
Author: Amin Akhshi, amin.akhshi@gmail.com

References:
- [1] Akhshi, A., et al., 2024. HiNTS: Higher-Order Interactions in N-Dimensional Time Series.
- [2] Lorenz, E., 1998. Optimal sites for supplementary weather observations: Simulation with a small model. Journal of the Atmospheric Sciences.

See Also:
- [1] Tabar, M.R.R, et al., 2024. Revealing Higher-Order Interactions in High-Dimensional Complex Systems: A Data-Driven Approach. PRX.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from jitcsde import jitcsde, y, t

class L96SDE:
    """
    Simulates and visualizes the stochastic Lorenz 96 model, a simplified mathematical model for chaotic atmospheric behavior.
    
    Attributes:
        K (int): Number of variables in the model.
        F (float): Forcing constant.
        dt (float): Time step for the integration.
        sigma (float): Standard deviation of the noise.
        X (numpy.ndarray): Current state of the system.
        _history_X (list): History of system states.
        additive (bool): Indicates if the noise is additive.
        seed (int): Seed for RNG to ensure reproducibility.
        noprog (bool): If True, disables the progress bar.
        
    Parameters:
        K (int): Number of variables (default: 10).
        F (float): Forcing constant (default: 8).
        dt (float): Integration time step (default: 0.01).
        sigma (float): Noise standard deviation (default: 0.1).
        X_init (numpy.ndarray, optional): Initial state. Random if None (default: None).
        additive (bool): If True, noise is additive (default: True).
        seed (int): RNG seed for reproducibility (default: 0).
        show_plot (bool): If True, enables plotting (default: True).
        noprog (bool): If True, disables tqdm progress bar (default: False).
    """
    def __init__(self, K=10, F=8, dt=0.01, sigma=0.1, X_init=None, **kwargs):
        self.K = K
        self.F = F
        self.dt = dt
        self.sigma = sigma
        self.X0 = np.random.rand(K) if X_init is None else X_init.copy()
        self.X = [self.X0.copy()]
        self.additive = kwargs.get('additive', True)
        self.seed = kwargs.get('seed', 0)
        self.noprog = kwargs.get('noprog', False)
        self.show_plot  = kwargs.get('show_plot', True)
    
    def _define_f(self):
        """
        Defines the deterministic component of the Lorenz 96 model alongside its stochastic counterpart.
    
        The deterministic part is defined by a set of differential equations representing the model's dynamics,
        and the stochastic part introduces randomness to the system, simulating real-world unpredictability.
        
        Returns:
            list: A list of symbolic differential equations representing the Lorenz 96 model's dynamics.
        """
        f_sym = [
            (-y((j-1) % self.K) * (y((j-2) % self.K) - y((j+1) % self.K)) - y(j) + self.F)
            for j in range(self.K)
            ]
        return f_sym

    def _define_g(self):
        """
        Defines the stochastic component of the Lorenz 96 model.

        The stochastic component introduces randomness to the system, simulating real-world unpredictability.

        Returns:
            list: A list of symbolic equations representing the stochastic component of the Lorenz 96 model.
        """
        return [self.sigma for j in range(self.K)]
    def iterate(self, time):
        """
        Advances the model state over a specified period through numerical integration,
        recording the state at each step.
        
        This method uses the Just-In-Time Compilation Stochastic Differential Equation (JitCSDE) library
        to perform numerical integration, taking into account both the deterministic and stochastic
        components of the model.
        
        Parameters:
            time (float): The total time period over which to integrate the model, in model time units.
        """
        f_sym = self._define_f()
        g_sym = self._define_g()
        SDE = jitcsde(f_sym=f_sym, g_sym=g_sym, n=self.K, additive=self.additive)
        SDE.set_initial_value(initial_value=self.X0, time=0.0)
        SDE.set_seed(seed=self.seed)
        
        steps = int(time / self.dt)
        for _ in tqdm(range(steps), disable=self.noprog):
            self.X0 = SDE.integrate(SDE.t + self.dt)
            self.X.append(self.X0.copy())

    @property
    def _history(self):
        """
        This function help to provide access to the recorded history of the system states throughout the simulation.
        
        This property allows for analysis and visualization of the system's evolution over time.
        
        Returns:
            numpy.ndarray: A 2D array of the system's states over time, where each row represents a time step.
        """
        return np.array(self.X)

    def add_point(self, x):
        """
        Closes a loop in a plot by appending the first element of the input array to its end.
        
        This function is particularly useful for plotting cyclic structures, such as polar plots,
        ensuring a smooth and continuous appearance.
        
        Parameters:
            x (numpy.ndarray): The input array representing a series of points in a plot.
            
        Returns:
            numpy.ndarray: The modified array with the first element appended to the end.
        """
        return np.append(x, x[0])

    def static_plot(self):
        """
        Creates a static polar plot of the system's final state as recorded in the simulation history.
    
        This visualization method provides a snapshot of the system's state at the end of the simulation,
        offering insights into the system's dynamics and behavior.
        """
        x_theta = [2 * np.pi / self.K * i for i in range(self.K + 1)]
        fig, ax1 = plt.subplots(figsize=(5, 5), subplot_kw={'projection': 'polar'})
        ax1.plot(x_theta, self.add_point(self._history[-1]), lw=3, zorder=10, label='X')
        self._configure_plot(ax1)
    
    def animate_plot(self, total_frames=200):
        """
        Generates an animation representing the system's evolution over time, automatically calculating
        the number of points to skip based on the desired total number of frames.
        
        Parameters:
            total_frames (int): The total number of frames for the animation.
            
        Returns:
            matplotlib.animation.FuncAnimation: The animation object, which can be displayed in a Jupyter
            notebook or saved to a file.
        """
        fig, ax1 = plt.subplots(figsize=(5, 5), subplot_kw={'projection': 'polar'})
        linex1, = ax1.plot([], [], lw=3, zorder=10, label='X')
        self._configure_plot(ax1)
        
        x_theta = [2 * np.pi / self.K * i for i in range(self.K + 1)]
        
        # Calculate skip rate to fit the animation into the desired total number of frames
        history_length = len(self._history)
        skip = max(1, history_length // total_frames) 
        
        def init():
            linex1.set_data([], [])
            return linex1,
    
        def animate(i):
            index = i * skip
            if index < history_length:  # Check to avoid index error
                x = self.add_point(self._history[index])
                linex1.set_data(x_theta, x)
            return linex1,
    
        ani = animation.FuncAnimation(fig, animate, frames=total_frames,
                                      interval=40, blit=True, init_func=init)
        plt.close()
        return ani

    def _configure_plot(self, ax):
        """
        Applies a common set of configurations to polar plots created by this class, enhancing
        the visual consistency across static and animated visualizations.
        
        Parameters:
            ax (matplotlib.axes.Axes): The matplotlib axes object to configure.
        """
        ax.set_rmin(-14); ax.set_rmax(14)
        l = ax.set_rgrids([-7, 0, 7], labels=['', '', ''])[0][1]
        l.set_linewidth(2)
        ax.set_thetagrids([])
        ax.set_rorigin(-22)
        ax.legend(frameon=False, loc=1)
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)


# model = L96SDE()
# model.iterate(10000.0)
# model.static_plot()
# model.animate_plot(total_frames=50)
