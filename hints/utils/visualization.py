import os
import numpy as np

# Optional settings for plotting
import matplotlib
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

cm = 1/2.54

##TODO: heatmap plot of pairwise interactions
# adding heatmap plot of 9D example in the jupyter notebook

class ts_plot:
    def __init__(self, data, time=None, **kwargs):
        self.data = np.atleast_2d(data)
        if self.data.shape[0] > self.data.shape[1]:
            self.data = self.data.T
        self.kwargs = kwargs
        self.ndim = self.data.shape[0]
        self.time = time if time is not None else np.arange(self.data.shape[1])
        self.cm = cm
        self.save_plot = kwargs.pop('save_plot', False)
        if self.save_plot:
            self.save_path = kwargs.pop('save_path', os.getcwd())
        
        
        self.filename = kwargs.pop('filename', 'timeseries.pdf')
        self.style_path = kwargs.pop('style', 'seaborn-v0_8-talk')
        self.plot3d = kwargs.pop('plot3d', False)
        with plt.style.context(self.style_path):
            self.plot()
            if self.plot3d:
                self.filename_3d = 'timeseries3d.pdf'
                self.plot_3d(data, **kwargs)



    def plot(self):
        ncols = 1 if self.ndim < 5 else 2
        nrows = self.ndim if ncols == 1 else -(-self.ndim // ncols) 
        fig, axs = plt.subplots(nrows, ncols, figsize=(6*self.cm*ncols, 3*nrows*self.cm),
                                sharex=True, constrained_layout=True)
        
        # Ensure axs is 2D for uniform handling
        axs = np.atleast_2d(axs)

        for i, dat in enumerate(self.data):
            ax = axs.flat[i]
            ax.plot(self.time, dat, **self.kwargs)
            ax.set_ylabel(f'$x_{{{i+1}}}$')
            formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-100,100))  # Use scientific notation if exponent is greater than 1 or less than -1
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
            if i // ncols != nrows - 1:  
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('time')

            ax.autoscale(enable=True, axis='x', tight=True)
            ax.autoscale(enable=True, axis='y', tight=False)
        
        # Removing empty subplot if odd number of subplots
        if self.ndim % ncols != 0:  # Remove unused axes for odd number of plots
            for idx in range(self.ndim, nrows*ncols):
                fig.delaxes(axs.flat[idx])
            
        plt.tight_layout()

        # Optional settings to save the plot
        if self.save_plot:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            plt.savefig(os.path.join(self.save_path, self.filename), bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_3d(self, data, **kwargs):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(data[0], data[1], data[2], **kwargs)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$x_3$")
        
        plt.tight_layout()
        if self.save_plot:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            plt.savefig(os.path.join(self.save_path, self.filename_3d), bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        

class Plotter:
    def __init__(self, style=None):
        # Initialize with default settings
        # self.plot_settings()
        self.style_path = style

    def plot_examples(self, simulation_result, coeff_result, model_name, **kwargs):
        # Define your specific plot functions like plot_2d, plot_example2, etc.
        if model_name in ['example1', 'supp_example1', 'supp_example2']:
            self.figure_size = kwargs.get('figure_size', (17.5*cm,16*cm))
            if self.style_path:
                with plt.style.context(self.style_path):
                    self.plot_2d(simulation_result, coeff_result, model_name, **kwargs)
            else:
                self.plot_2d(simulation_result, coeff_result, model_name, **kwargs)

        elif model_name == 'example2':
            self.figure_size = kwargs.get('figure_size', (17.5*cm,17*cm))
            if self.style_path:
                with plt.style.context(self.style_path):
                    self.plot_example2(simulation_result, coeff_result, model_name, **kwargs)
            else:
                self.plot_example2(simulation_result, coeff_result, model_name, **kwargs)            
                
        elif model_name == 'example3':
            self.figure_size = kwargs.get('figure_size', (17.5*cm,12*cm))
            if self.style_path:
                with plt.style.context(self.style_path):
                    self.plot_example3(simulation_result, coeff_result, model_name, **kwargs)
            else:
                self.plot_example3(simulation_result, coeff_result, model_name, **kwargs)      
        elif model_name == 'supp_example3':
            self.figure_size = kwargs.get('figure_size', (17.5*cm,19*cm))
            if self.style_path:
                with plt.style.context(self.style_path):
                    self.plot_3d(simulation_result, coeff_result, model_name, **kwargs)
            else:
                self.plot_3d(simulation_result, coeff_result, model_name, **kwargs)    
        elif model_name == 'supp_example4':
            if self.style_path:
                with plt.style.context(self.style_path):
                    self.plot_supp_ex4(simulation_result, coeff_result, model_name, **kwargs)
            else:
                self.plot_supp_ex4(simulation_result, coeff_result, model_name, **kwargs)    
        else:
            raise ValueError('Model does not exist')

    # Example plot functions
    def plot_2d(self, simulation_result, coeff_result, model_name, **kwargs):
        cutoff = kwargs.get('cutoff', None)
        bins = kwargs.get('bins', 51)
        density = kwargs.get('density', True)
        cumulative = kwargs.get('cumulative', True)
        save_plot = kwargs.get('save_plot', True)
        
        ndim = simulation_result['ndim']
        model_name = simulation_result['model_name']
        subfig_hspace = 0.5 / self.figure_size[0]
        subfig_wspace = 0.5 / self.figure_size[1]
        colors = ['navy', 'darkorange', 'brown', '#217b7e', 'mediumslateblue', 'gray']
        markers = ['o', 's', '^', 'v', '<', '>']

        # plotting function for 2d examples
        fig = plt.figure(constrained_layout=True, figsize=self.figure_size)
        gs = GridSpec(3, 2, figure=fig, width_ratios=[2, 1])
        
        if cutoff:
            x_cutoff = simulation_result['true_sim'][:,cutoff:]
            time = simulation_result['time'][cutoff:] - simulation_result['time'][cutoff]
        else:
            x_cutoff = simulation_result['true_sim']
            time = simulation_result['time']
            
        sub_gs_a = GridSpecFromSubplotSpec(ndim, 1, subplot_spec=gs[0, 0])
        for i in range(ndim):
            ax = fig.add_subplot(sub_gs_a[i])
            ax.plot(time, x_cutoff[i], color=colors[i], lw=1, ls='-')
            ax.set_ylabel(f'$x_{i + 1}$')
            if i < 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel(r'time')
            ax.autoscale(enable=True, axis='x', tight=True)
            ax.autoscale(enable=True, axis='y', tight=False)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
        
        # Plotting CDF for state-variable timeseries    
        sub_gs_b = GridSpecFromSubplotSpec(ndim, 1, subplot_spec=gs[0, 1])
        labels = ["true", "estimated"]
        for i in range(ndim):
            ax = fig.add_subplot(sub_gs_b[i])
            ax.hist(simulation_result['true_sim'][i], bins=bins, density=density,
                    histtype="step", cumulative=cumulative, color=colors[i],
                    label=f"{labels[0]}")
            ax.hist(simulation_result['estimated_sim'][i], bins=bins, density=density,
                    histtype="step", cumulative=cumulative, color=colors[i],
                    ls='-.', label=f"{labels[1]}")
            
            ax.set_ylabel(f'CDF($x_{i + 1}$)')
            true_line = Line2D([0], [0], color=colors[i], lw=1.5, label=f"{labels[0]}")
            estimated_line = Line2D([0], [0], color=colors[i], lw=1.5, ls='-.',
                                    label=f"{labels[1]}")
            ax.legend(handles=[true_line, estimated_line], loc='upper left')
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3))
        
        sub_gs = GridSpecFromSubplotSpec(2, 5, subplot_spec=gs[1, :], hspace=subfig_hspace, wspace=subfig_wspace)
        labels = ['$1$', '$x_1$', '$x_2$', '$x_{1}^{2}$', '$x_{1}x_{2}$', '$x_{2}^{2}$',
                  '$x_{1}^{3}$', '$x_{1}^{2}x_{2}$', '$x_{1}x_{2}^{2}$', '$x_{2}^{3}$']
        xlabels = '$N$d$t$'
        axs = [fig.add_subplot(cell) for cell in sub_gs]
        for i, ax in zip(range(len(labels)), axs):
            ax.errorbar(coeff_result['Ndt'], coeff_result['mean_error'][:,0,i],
                          coeff_result['std_error'][:,0,i], marker=markers[0],
                          capsize=2, markersize=3, markeredgecolor=None, ls=':',
                          color=colors[0])
            ax.errorbar(coeff_result['Ndt'], coeff_result['mean_error'][:,1,i],
                          coeff_result['std_error'][:,1,i], marker=markers[1],
                          capsize=2, markersize=3, markeredgecolor=None, ls=':',
                          color=colors[1])
            # Set the x-label for the second row
            if 5 <= i < 10:
                ax.set_xlabel(f'{xlabels}')
            ax.text(0.90, 0.95, f'{labels[i]}', transform=ax.transAxes, 
                    va='top', ha='right', fontsize=12)
            ax.set_xscale('log')
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        for i in range(5):
            axs[i].set_xticks([])
        axs[0].set_ylabel(r'$\Delta$')
        axs[5].set_ylabel(r'$\Delta$')
        
        sub_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, :])
        ax6 = fig.add_subplot(sub_gs[0])
        diff_labels = ['$g_{11}$', '$g_{22}$', '$g_{12}$']
        for i, key in enumerate(diff_labels):
            ax6.errorbar(coeff_result['Ndt'], coeff_result['g_error_mean'][:,i],
                     coeff_result['g_error_std'][:,i], marker=markers[i], lw=1.5,
                     capsize=2, markersize=5, markeredgecolor=None, ls=':', color=colors[3+i],
                     label=f'{key}')
        ax6.set_ylabel(r'$\Delta$')
        ax6.set_xlabel(f'{xlabels}')
        ax6.set_xscale('log')
        ax6.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
        ax6.legend(loc='upper center', ncol=3, fontsize=12)
        ax6.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        
        ax7 = fig.add_subplot(sub_gs[1])
        for i in range(ndim):
            ax7.errorbar(coeff_result['Ndt'], coeff_result['mean_r2'][:,i],
                     coeff_result['std_r2'][:,i], marker=markers[i],
                     capsize=2, markersize=5, markeredgecolor=None, ls='-', color=colors[i],
                     label=f'$x_{i + 1}$')
        ax7.set_ylabel(r'$R^{2}-score$')
        ax7.set_xlabel(f'{xlabels}')
        ax7.set_xscale('log')
        ax7.legend(loc='upper center', ncol=ndim, fontsize=12)
        ax7.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
        ax7.ticklabel_format(style='sci', axis='y', scilimits=(2, 0))

        if model_name == 'supp_example1':
            axins = inset_axes(ax7, width="30%", height="30%", loc='lower right')
            for i in range(ndim):
                axins.errorbar(coeff_result['Ndt'][5:], coeff_result['mean_r2'][5:,i],
                               coeff_result['std_r2'][5:,i], marker=markers[i],
                               capsize=2, markersize=5, markeredgecolor=None, ls='-', color=colors[i],
                               )
            axins.set_xlim(5700, 10000)
            axins.set_ylim(-1.5, 2)
            axins.set_xscale('log')
            axins.axhline(y=1, color='black', linestyle='--')
            mark_inset(ax7, axins, loc1=3, loc2=2, fc="none", ec="0.5")
        else:
            ax7.axhline(y=1, color='black', linestyle='--')
        
        if save_plot:
            plt.savefig(os.path.join(os.getcwd(), 'result', model_name, f'{model_name}_figure.pdf'))
            plt.close()
        pass

    def plot_example2(self, simulation_result, coeff_result, model_name, **kwargs):
        cutoff = kwargs.get('cutoff', None)
        bins = kwargs.get('bins', 51)
        density = kwargs.get('density', True)
        cumulative = kwargs.get('cumulative', True)
        save_plot = kwargs.get('save_plot', True)
        
        ndim = simulation_result['ndim']
        model_name = simulation_result['model_name']
        subfig_hspace = 0.5 / self.figure_size[0]
        subfig_wspace = 0.5 / self.figure_size[1]
        colors = ['navy', 'darkorange', 'brown', '#217b7e', 'mediumslateblue', 'gray']
        # If you have defined a marker cycle in your style, use it. Otherwise define it manually
        markers = ['o', 's', '^', 'v', '<', '>']

        # plotting function for 3d example in example 2
        fig = plt.figure(constrained_layout=True, figsize=self.figure_size)
        gs = GridSpec(3, 2, figure=fig, width_ratios=[2, 1])
        
        if cutoff:
            x_cutoff = simulation_result['true_sim'][:,cutoff:]
            time = simulation_result['time'][cutoff:] - simulation_result['time'][cutoff]
        else:
            x_cutoff = simulation_result['true_sim']
            time = simulation_result['time']
            
        sub_gs_a = GridSpecFromSubplotSpec(ndim, 1, subplot_spec=gs[0, 0]) # height_ratios=[1.4,2,1.2]
        for i in range(ndim):
            ax = fig.add_subplot(sub_gs_a[i])
            ax.plot(time, x_cutoff[i], color=colors[i], lw=1, ls='-')
            ax.set_ylabel(f'$x_{i + 1}$')
            if i < 2:
                ax.set_xticks([])
            else:
                ax.set_xlabel(r'time')
            ax.autoscale(enable=True, axis='x', tight=True)
            ax.autoscale(enable=True, axis='y', tight=False)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3))
        
        # Plotting CDF for state-variable timeseries    
        sub_gs_b = GridSpecFromSubplotSpec(ndim, 1, subplot_spec=gs[0, 1])
        labels = ["true", "estimated"]
        for i in range(ndim):
            ax = fig.add_subplot(sub_gs_b[i])
            ax.hist(simulation_result['true_sim'][i], bins=bins, density=density,
                    histtype="step", cumulative=cumulative, color=colors[i],
                    label=f"{labels[0]}")
            ax.hist(simulation_result['estimated_sim'][i], bins=bins, density=density,
                    histtype="step", cumulative=cumulative, color=colors[i],
                    ls='-.', label=f"{labels[1]}")
            
            ax.set_ylabel(f'CDF($x_{i + 1}$)')
            true_line = Line2D([0], [0], color=colors[i], lw=1.5, label=f"{labels[0]}")
            estimated_line = Line2D([0], [0], color=colors[i], lw=1.5, ls='-.',
                                    label=f"{labels[1]}")
            ax.legend(handles=[true_line, estimated_line], loc='upper left')
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3))
        
        sub_gs = GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[1, :], hspace=subfig_hspace, wspace=subfig_wspace)
        labels = ['$a_{12}$', '$a_{13}$', '$r_{2}$', '$a_{21}$', '$r_{3}$', '$a_{31}$', '$k_{3}$', '$d_{3}$']
        # labels = ['a12', 'a13', 'r2', 'a21', 'r3', 'a31', 'k3', 'd3']
        temp_color = ['navy', 'navy', 'darkorange', 'darkorange', 'brown',
                      'brown', 'brown', 'brown']
        temp_markers = ['o', 'o', 's', 's', '^', '^', '^', '^']
        xlabels = '$N$d$t$'
        axs = [fig.add_subplot(cell) for cell in sub_gs]
        for i, cc, mm, ax in zip(range(len(labels)), temp_color, temp_markers, axs):
            ax.errorbar(coeff_result['Ndt'], coeff_result['mean_error'][:,i],
                          coeff_result['std_error'][:,i], marker=mm,
                          capsize=2, markersize=3, markeredgecolor=None, ls=':',
                          color=cc)
            # Set the x-label for the second row
            if 4 <= i < 8:
                ax.set_xlabel(f'{xlabels}')
            ax.text(0.90, 0.95, f'{labels[i]}', transform=ax.transAxes, 
                    va='top', ha='right', fontsize=12)
            ax.set_xscale('log')
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        for i in range(4):
            axs[i].set_xticks([])
        axs[0].set_ylabel(r'$\Delta$')
        axs[4].set_ylabel(r'$\Delta$')
        
        sub_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, :])
        ax6 = fig.add_subplot(sub_gs[0])
        diff_labels = ['$g_{11}$', '$g_{22}$', '$g_{12}$']
        for i, key in enumerate(diff_labels):
            ax6.errorbar(coeff_result['Ndt'], coeff_result['g_error_mean'][:,i],
                     coeff_result['g_error_std'][:,i], marker=markers[i], lw=1.5,
                     capsize=2, markersize=5, markeredgecolor=None, ls=':', color=colors[3+i],
                     label=f'{key}')
        ax6.set_ylabel(r'$\Delta$')
        ax6.set_xlabel(f'{xlabels}')
        ax6.set_xscale('log')
        ax6.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
        ax6.legend(loc='upper center', ncol=3, fontsize=12)
        ax6.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        ax7 = fig.add_subplot(sub_gs[1])
        for i in range(ndim):
            ax7.errorbar(coeff_result['Ndt'], coeff_result['mean_r2'][:,i],
                     coeff_result['std_r2'][:,i], marker=markers[i],
                     capsize=2, markersize=5, markeredgecolor=None, ls='-', color=colors[i],
                     label=f'$x_{i + 1}$')
        ax7.set_ylabel(r'$R^{2}-score$')
        ax7.set_xlabel(f'{xlabels}')
        ax7.set_xscale('log')
        ax7.legend(loc='upper center', ncol=ndim, fontsize=12)
        ax7.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
        ax7.ticklabel_format(style='sci', axis='y', scilimits=(2, 0))

        axins = inset_axes(ax7, width="30%", height="30%", loc='lower right')
        for i in range(ndim):
            axins.errorbar(coeff_result['Ndt'][2:], coeff_result['mean_r2'][2:,i],
                           coeff_result['std_r2'][2:,i], marker=markers[i],
                           capsize=2, markersize=5, markeredgecolor=None, ls='-', color=colors[i],
                           )
        axins.set_xlim(coeff_result['Ndt'][2]-10, coeff_result['Ndt'][-1]+50)
        axins.set_ylim(-1.5, 2)
        axins.set_xscale('log')
        axins.axhline(y=1, color='black', linestyle='--')
        axins.ticklabel_format(style='sci', axis='x', scilimits=(2, 0))
        # Add a rectangle indicating the zoomed region on the main plot
        mark_inset(ax7, axins, loc1=3, loc2=2, fc="none", ec="0.5")
        
        if save_plot:
            plt.savefig(os.path.join(os.getcwd(), 'result', model_name, f'{model_name}_figure.pdf'))
            plt.close()
        pass

    def plot_example3(self, simulation_result, coeff_result, model_name, **kwargs):
        cutoff = kwargs.get('cutoff', None)
        bins = kwargs.get('bins', 51)
        density = kwargs.get('density', True)
        cumulative = kwargs.get('cumulative', True)
        save_plot = kwargs.get('save_plot', True)
        g = coeff_result['true_g']
        ndim = len(g)
        model_name = simulation_result['model_name']
        subfig_hspace = 0.5 / self.figure_size[0]
        subfig_wspace = 0.5 / self.figure_size[1]
        colors = ['navy', '#0892d0', 'brown', '#217b7e', 'mediumslateblue', 'gray']
        # If you have defined a marker cycle in your style, use it. Otherwise define it manually
        markers = ['o', 's', '^', 'v', '<', '>']

        # plotting function for 1d example in example 3
        fig = plt.figure(constrained_layout=True, figsize=self.figure_size)
        gs = GridSpec(3, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1.8,1,1.4])
        
        if cutoff:
            time = simulation_result['time'][cutoff:] - simulation_result['time'][cutoff]
        else:
            cutoff = 0
            time = simulation_result['time']
            
        sub_gs_a = GridSpecFromSubplotSpec(ndim, 1, subplot_spec=gs[0, 0]) 
        for i in range(ndim):
            ax = fig.add_subplot(sub_gs_a[i])
            x_cutoff = simulation_result[f'true_sim_{g[i]}'][cutoff:]
            ax.plot(time, x_cutoff, color=colors[i], lw=1, ls='-', label=f'$\Gamma={g[i]}$')
            ax.set_ylabel(f'$x$')
            if i < 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel(r'time')
            ax.autoscale(enable=True, axis='x', tight=True)
            ax.autoscale(enable=True, axis='y', tight=False)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
            ax.legend(loc='upper left', fontsize=12)

        # Plotting CDF for state-variable timeseries    
        sub_gs_b = GridSpecFromSubplotSpec(ndim, 1, subplot_spec=gs[0, 1])
        labels = ["true", "estimated"]
        for i in range(ndim):
            ax = fig.add_subplot(sub_gs_b[i])
            ax.hist(simulation_result[f'true_sim_{g[i]}'], bins=bins, density=density,
                    histtype="step", cumulative=cumulative, color=colors[i],
                    label=f"{labels[0]}")
            ax.hist(simulation_result[f'estimated_sim_{g[i]}'], bins=bins, density=density,
                    histtype="step", cumulative=cumulative, color=colors[i],
                    ls='-.', label=f"{labels[1]}")
            
            ax.set_ylabel(f'CDF($x$)')
            true_line = Line2D([0], [0], color=colors[i], lw=1.5, label=f"{labels[0]}")
            estimated_line = Line2D([0], [0], color=colors[i], lw=1.5, ls='-.',
                                    label=f"{labels[1]}")
            ax.legend(handles=[true_line, estimated_line], loc='upper left')
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3))
        
        sub_gs = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1, :], hspace=subfig_hspace, wspace=subfig_wspace)
        labels = ['$1$', '$x_1$', '$x_{1}^{2}$', '$x_{1}^{3}$']
        xlabels = '$N$d$t$'
        axs = [fig.add_subplot(cell) for cell in sub_gs]
        for i, ax in zip(range(len(labels)), axs):
            ax.errorbar(coeff_result['Ndt'], coeff_result['mean_error'][0,:,i],
                          coeff_result['std_error'][0,:,i], marker=markers[0],
                          capsize=2, markersize=3, markeredgecolor=None, ls=':',
                          color=colors[0])
            ax.errorbar(coeff_result['Ndt'], coeff_result['mean_error'][1,:,i],
                          coeff_result['std_error'][1,:,i], marker=markers[1],
                          capsize=2, markersize=3, markeredgecolor=None, ls=':',
                          color=colors[1])
            # Set the x-label for the second row
            ax.set_xlabel(f'{xlabels}')
            ax.text(0.90, 0.95, f'{labels[i]}', transform=ax.transAxes, 
                    va='top', ha='right', fontsize=12)
            ax.set_xscale('log')
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            axs[0].set_ylabel(r'$\Delta$')

        sub_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, :])
        ax6 = fig.add_subplot(sub_gs[0])
        diff_labels = ['$\Gamma=0.6$', '$\Gamma=1.1$']
        for i, key in enumerate(diff_labels):
            ax6.errorbar(coeff_result['Ndt'], coeff_result['g_error_mean'][i,:],
                     coeff_result['g_error_std'][i,:], marker=markers[i], lw=1.5,
                     capsize=2, markersize=5, markeredgecolor=None, ls=':', color=colors[i],
                     label=f'{key}')
        ax6.set_ylabel(r'$\Delta$')
        ax6.set_xlabel(f'{xlabels}')
        ax6.set_xscale('log')
        ax6.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
        ax6.legend(loc='upper center', ncol=ndim, fontsize=12)
        ax6.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        ax7 = fig.add_subplot(sub_gs[1])
        for i in range(ndim):
            ax7.errorbar(coeff_result['Ndt'], coeff_result['mean_r2'][i],
                     coeff_result['std_r2'][i], marker=markers[i],
                     capsize=2, markersize=5, markeredgecolor=None, ls='-', color=colors[i])
        ax7.set_ylabel(r'$R^{2}-score$')
        ax7.set_xlabel(f'{xlabels}')
        ax7.set_xscale('log')
        # ax7.legend(loc='upper center', ncol=ndim, fontsize=12)
        ax7.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
        ax7.ticklabel_format(style='sci', axis='y', scilimits=(2, 0))

        axins = inset_axes(ax7, width="30%", height="30%", loc='lower right')
        for i in range(ndim):
            axins.errorbar(coeff_result['Ndt'][5:], coeff_result['mean_r2'][i,5:],
                           coeff_result['std_r2'][i,5:], marker=markers[i],
                           capsize=2, markersize=5, markeredgecolor=None, ls='-', color=colors[i],
                           )
        axins.set_xlim(coeff_result['Ndt'][5]-10, coeff_result['Ndt'][-1]+20)
        axins.set_ylim(0, 1.5)
        axins.set_xscale('log')
        axins.axhline(y=1, color='black', linestyle='--')
        # Add a rectangle indicating the zoomed region on the main plot
        mark_inset(ax7, axins, loc1=3, loc2=2, fc="none", ec="0.5")
        
        if save_plot:
            plt.savefig(os.path.join(os.getcwd(), 'result', model_name, f'{model_name}_figure.pdf'))
            plt.close()
        pass

    def plot_3d(self, simulation_result, coeff_result, model_name, **kwargs):
        cutoff = kwargs.get('cutoff', None)
        bins = kwargs.get('bins', 51)
        density = kwargs.get('density', True)
        cumulative = kwargs.get('cumulative', True)
        save_plot = kwargs.get('save_plot', True)
        
        ndim = simulation_result['ndim']
        model_name = simulation_result['model_name']
        subfig_hspace = 0.5 / self.figure_size[0]
        subfig_wspace = 0.5 / self.figure_size[1]
        colors = ['navy', 'darkorange', 'brown', '#217b7e', 'mediumslateblue', 'gray']
        markers = ['o', 's', '^', 'v', '<', '>']

        # plotting function for 3d examples
        fig = plt.figure(constrained_layout=True, figsize=self.figure_size)
        gs = GridSpec(3, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1.4,2,1.2])
        
        if cutoff:
            x_cutoff = simulation_result['true_sim'][:,cutoff:]
            time = simulation_result['time'][cutoff:] - simulation_result['time'][cutoff]
        else:
            x_cutoff = simulation_result['true_sim']
            time = simulation_result['time']
            
        sub_gs_a = GridSpecFromSubplotSpec(ndim, 1, subplot_spec=gs[0, 0])
        for i in range(ndim):
            ax = fig.add_subplot(sub_gs_a[i])
            ax.plot(time, x_cutoff[i], color=colors[i], lw=1, ls='-')
            ax.set_ylabel(f'$x_{i + 1}$')
            if i < 2:
                ax.set_xticks([])
            else:
                ax.set_xlabel(r'time')
            ax.autoscale(enable=True, axis='x', tight=True)
            ax.autoscale(enable=True, axis='y', tight=False)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
        
        # Plotting CDF for state-variable timeseries    
        sub_gs_b = GridSpecFromSubplotSpec(ndim, 1, subplot_spec=gs[0, 1])
        labels = ["true", "estimated"]
        for i in range(ndim):
            ax = fig.add_subplot(sub_gs_b[i])
            ax.hist(simulation_result['true_sim'][i], bins=bins, density=density,
                    histtype="step", cumulative=cumulative, color=colors[i],
                    label=f"{labels[0]}")
            ax.hist(simulation_result['estimated_sim'][i], bins=bins, density=density,
                    histtype="step", cumulative=cumulative, color=colors[i],
                    ls='-.', label=f"{labels[1]}")
            
            ax.set_ylabel(f'CDF($x_{i + 1}$)')
            true_line = Line2D([0], [0], color=colors[i], lw=1.5, label=f"{labels[0]}")
            estimated_line = Line2D([0], [0], color=colors[i], lw=1.5, ls='-.',
                                    label=f"{labels[1]}")
            ax.legend(handles=[true_line, estimated_line], loc='upper left')
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3))
        
        sub_gs = GridSpecFromSubplotSpec(4, 5, subplot_spec=gs[1, :], hspace=subfig_hspace, wspace=subfig_wspace)
        labels = ['$1$', '$x_1$', '$x_2$', '$x_3$', 
                  '$x_{1}^{2}$', '$x_{1}x_{2}$', '$x_{1}x_{3}$', '$x_{2}^{2}$', '$x_{2}x_{3}$', '$x_{3}^{2}$',
                  '$x_{1}^{3}$', '$x_{1}^{2}x_{2}$', '$x_{1}x_{2}^{2}$', '$x_{2}^{3}$', '$x_{1}^{2}x_{3}$', '$x_{1}x_{2}x_{3}$',
                  '$x_{1}x_{3}^{2}$', '$x_{2}^{2}x_{3}$', '$x_{2}x_{3}^{2}$', '$x_{3}^{3}$'
                  ]   
        xlabels = '$N$d$t$'
        axs = [fig.add_subplot(cell) for cell in sub_gs]
        for i, ax in zip(range(len(labels)), axs):
            ax.errorbar(coeff_result['Ndt'], coeff_result['mean_error'][:,0,i],
                          coeff_result['std_error'][:,0,i], marker=markers[0],
                          capsize=2, markersize=3, markeredgecolor=None, ls=':',
                          color=colors[0])
            ax.errorbar(coeff_result['Ndt'], coeff_result['mean_error'][:,1,i],
                          coeff_result['std_error'][:,1,i], marker=markers[1],
                          capsize=2, markersize=3, markeredgecolor=None, ls=':',
                          color=colors[1])
            ax.errorbar(coeff_result['Ndt'], coeff_result['mean_error'][:,2,i],
                          coeff_result['std_error'][:,2,i], marker=markers[2],
                          capsize=2, markersize=3, markeredgecolor=None, ls=':',
                          color=colors[2])
            # Set the x-label for the second row
            if 15 <= i < 20:
                ax.set_xlabel(f'{xlabels}')
            ax.text(0.90, 0.95, f'{labels[i]}', transform=ax.transAxes, 
                    va='top', ha='right', fontsize=12)
            ax.set_xscale('log')
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        for i in range(15):
            axs[i].set_xticks([])
        axs[0].set_ylabel(r'$\Delta$')
        axs[5].set_ylabel(r'$\Delta$')
        axs[10].set_ylabel(r'$\Delta$')
        axs[15].set_ylabel(r'$\Delta$')
        
        sub_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, :])
        ax6 = fig.add_subplot(sub_gs[0])
        diff_labels = ['$g_{11}$', '$g_{22}$', '$g_{12}$']
        for i, key in enumerate(diff_labels):
            ax6.errorbar(coeff_result['Ndt'], coeff_result['g_error_mean'][:,i],
                     coeff_result['g_error_std'][:,i], marker=markers[i], lw=1.5,
                     capsize=2, markersize=5, markeredgecolor=None, ls=':', color=colors[3+i],
                     label=f'{key}')
        ax6.set_ylabel(r'$\Delta$')
        ax6.set_xlabel(f'{xlabels}')
        ax6.set_xscale('log')
        ax6.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
        ax6.legend(loc='upper center', ncol=3, fontsize=12)
        ax6.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        
        ax7 = fig.add_subplot(sub_gs[1])
        for i in range(ndim):
            ax7.errorbar(coeff_result['Ndt'], coeff_result['mean_r2'][:,i],
                     coeff_result['std_r2'][:,i], marker=markers[i],
                     capsize=2, markersize=5, markeredgecolor=None, ls='-', color=colors[i],
                     label=f'$x_{i + 1}$')
        ax7.set_ylabel(r'$R^{2}-score$')
        ax7.set_xlabel(f'{xlabels}')
        ax7.set_xscale('log')
        ax7.legend(loc='upper center', ncol=ndim, fontsize=12)
        ax7.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
        ax7.ticklabel_format(style='sci', axis='y', scilimits=(2, 0))

        axins = inset_axes(ax7, width="30%", height="30%", loc='lower right')
        for i in range(ndim):
            axins.errorbar(coeff_result['Ndt'][2:], coeff_result['mean_r2'][2:,i],
                           coeff_result['std_r2'][2:,i], marker=markers[i],
                           capsize=2, markersize=5, markeredgecolor=None, ls='-', color=colors[i],
                           )
        axins.set_xlim(1000, 10000)
        axins.set_ylim(-0.2, 1.5)
        axins.set_xscale('log')
        axins.axhline(y=1, color='black', linestyle='--')
        mark_inset(ax7, axins, loc1=3, loc2=2, fc="none", ec="0.5")

        if save_plot:
            plt.savefig(os.path.join(os.getcwd(), 'result', model_name, f'{model_name}_figure.pdf'))
            plt.close()
        pass

    def plot_supp_ex4(self, simulation_result, coeff_result, model_name, **kwargs):
        
        cutoff = kwargs.get('cutoff', None)
        bins = kwargs.get('bins', 51)
        density = kwargs.get('density', True)
        cumulative = kwargs.get('cumulative', True)
        save_plot = kwargs.get('save_plot', True)
        
        ndim = simulation_result['ndim']
        model_name = simulation_result['model_name']
        colors = ['navy', 'darkorange', 'brown', '#217b7e', 'mediumslateblue', 'gray']
        markers = ['o', 's', '^', 'v', '<', '>', '+', 'D', "1"]
        
        self.figure_size = kwargs.get('figure_size', (17.5*cm,13*cm))
        # plotting function for 9d example
        fig = plt.figure(constrained_layout=True, figsize=self.figure_size)
        gs = GridSpec(1, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1])
        
        if cutoff:
            x_cutoff = simulation_result['true_sim'][:,cutoff:]
            time = simulation_result['time'][cutoff:] - simulation_result['time'][cutoff]
        else:
            x_cutoff = simulation_result['true_sim']
            time = simulation_result['time']
            
        sub_gs_a = GridSpecFromSubplotSpec(ndim, 1, subplot_spec=gs[0, 0])
        for i in range(ndim):
            ax = fig.add_subplot(sub_gs_a[i])
            ax.plot(time, x_cutoff[i], color=colors[0], lw=1, ls='-')
            ax.set_ylabel(f'$x_{i + 1}$')
            if i < 8:
                ax.set_xticks([])
            else:
                ax.set_xlabel(r'time')
            ax.autoscale(enable=True, axis='x', tight=True)
            ax.autoscale(enable=True, axis='y', tight=False)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
        
        # Plotting CDF for state-variable timeseries    
        sub_gs_b = GridSpecFromSubplotSpec(ndim, 1, subplot_spec=gs[0, 1])
        labels = ["true", "estimated"]
        for i in range(ndim):
            ax = fig.add_subplot(sub_gs_b[i])
            ax.hist(simulation_result['true_sim'][i], bins=bins, density=density,
                    histtype="step", cumulative=cumulative, color=colors[0],
                    label=f"{labels[0]}")
            ax.hist(simulation_result['estimated_sim'][i], bins=bins, density=density,
                    histtype="step", cumulative=cumulative, color=colors[0],
                    ls='-.', label=f"{labels[1]}")
            
            ax.set_ylabel(f'CDF($x_{i + 1}$)')
            true_line = Line2D([0], [0], color=colors[0], lw=1.5, label=f"{labels[0]}")
            estimated_line = Line2D([0], [0], color=colors[0], lw=1.5, ls='-.',
                                    label=f"{labels[1]}")
            ax.legend(handles=[true_line, estimated_line], loc='upper left')
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3))
        if save_plot:
            plt.savefig(os.path.join(os.getcwd(), 'result', model_name, f'{model_name}_timeseries.pdf'))
            plt.close()
        
        self.figure_size = kwargs.get('figure_size', (17.5*cm,6*cm))
        # plotting function for 9d example
        subfig_hspace = 0.5 / self.figure_size[0]
        subfig_wspace = 0.5 / self.figure_size[1]
        fig = plt.figure(constrained_layout=True, figsize=self.figure_size)
        gs = GridSpec(1, 1, figure=fig, width_ratios=[1], height_ratios=[1])
        
        sub_gs = GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0], hspace=subfig_hspace, wspace=subfig_wspace)
        labels = ['$a$', '$r$', '$b_1$', '$b_2$', '$b_{3}$', '$b_{4}$', '$b_{5}$', '$b_{6}$']
        xlabels = '$N$d$t$'
        axs = [fig.add_subplot(cell) for cell in sub_gs]
        for i, ax in zip(range(len(labels)), axs):
            ax.errorbar(coeff_result['Ndt'], coeff_result['pred_coeff_mean'][:,i],
                          coeff_result['pred_coeff_std'][:,i], marker=markers[i],
                          capsize=2, markersize=3, markeredgecolor=None, ls=':',
                          color=colors[0])
            # Set the x-label for the second row
            if 4 <= i < 8:
                ax.set_xlabel(f'{xlabels}')
            ax.text(0.90, 0.95, f'{labels[i]}', transform=ax.transAxes, 
                    va='top', ha='right', fontsize=12)
            ax.set_xscale('log')
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        for i in range(4):
            axs[i].set_xticks([])
        axs[0].set_ylabel(r'$\Delta$')
        axs[4].set_ylabel(r'$\Delta$')
        if save_plot:
            plt.savefig(os.path.join(os.getcwd(), 'result', model_name, f'{model_name}_drifts.pdf'))
            plt.close()
        
        self.figure_size = kwargs.get('figure_size', (17.5*cm,8*cm))
        subfig_hspace = 0.5 / self.figure_size[0]
        subfig_wspace = 0.5 / self.figure_size[1]
        # plotting function for 9d example
        fig = plt.figure(constrained_layout=True, figsize=self.figure_size)
        gs = GridSpec(1, 1, figure=fig, width_ratios=[1], height_ratios=[1])
        sub_gs = GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[0], hspace=subfig_hspace, wspace=subfig_wspace)
        labels = ['$g_{11}$', '$g_{22}$', '$g_{33}$', '$g_{44}$',
                  '$g_{55}$', '$g_{66}$', '$g_{77}$', '$g_{88}$', '$g_{99}$']
        xlabels = '$N$d$t$'
        axs = [fig.add_subplot(cell) for cell in sub_gs]
        for i, ax in zip(range(len(labels)), axs):
            ax.errorbar(coeff_result['Ndt'], coeff_result['g_error_mean'][:,i],
                          coeff_result['g_error_std'][:,i], marker=markers[i],
                          capsize=2, markersize=3, markeredgecolor=None, ls=':',
                          color=colors[0])
            # Set the x-label for the second row
            if 6 <= i < 9:
                ax.set_xlabel(f'{xlabels}')
            ax.text(0.90, 0.95, f'{labels[i]}', transform=ax.transAxes, 
                    va='top', ha='right', fontsize=12)
            ax.set_xscale('log')
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        for i in range(6):
            axs[i].set_xticks([])
        axs[0].set_ylabel(r'$\Delta$')
        axs[3].set_ylabel(r'$\Delta$')
        axs[6].set_ylabel(r'$\Delta$')

        if save_plot:
            plt.savefig(os.path.join(os.getcwd(), 'result', model_name, f'{model_name}_diffusion.pdf'))
            plt.close()
        
        self.figure_size = kwargs.get('figure_size', (17.5*cm,6*cm))
        subfig_hspace = 0.5 / self.figure_size[0]
        subfig_wspace = 0.5 / self.figure_size[1]
        # plotting function for 9d example
        fig = plt.figure(constrained_layout=True, figsize=self.figure_size)
        gs = GridSpec(1, 1, figure=fig, width_ratios=[1], height_ratios=[1])
        sub_gs = GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[0], hspace=subfig_hspace, wspace=subfig_wspace)
        labels = ['$x_{1}$', '$x_{2}$', '$x_{3}$', '$x_{4}$',
                  '$x_{5}$', '$x_{6}$', '$x_{7}$', '$x_{8}$', '$x_{9}$']
        xlabels = '$N$d$t$'
        axs = [fig.add_subplot(cell) for cell in sub_gs]
        for i, ax in zip(range(len(labels)), axs):
            ax.errorbar(coeff_result['Ndt'], coeff_result['mean_r2'][:,i],
                          coeff_result['std_r2'][:,i], marker=markers[i],
                          capsize=2, markersize=3, markeredgecolor=None, ls=':',
                          color=colors[0])
            # Set the x-label for the second row
            if 6 <= i < 9:
                ax.set_xlabel(f'{xlabels}')
            ax.text(0.90, 0.95, f'{labels[i]}', transform=ax.transAxes, 
                    va='top', ha='right', fontsize=12)
            ax.set_xscale('log')
            # ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.axhline(y=1, color='black', linestyle='--')

        for i in range(6):
            axs[i].set_xticks([])
        axs[0].set_ylabel(r'$R^{2}-score$')
        axs[3].set_ylabel(r'$R^{2}-score$')
        axs[6].set_ylabel(r'$R^{2}-score$')

        if save_plot:
            plt.savefig(os.path.join(os.getcwd(), 'result', model_name, f'{model_name}_rsquared.pdf'))
            plt.close()

def plot_ks_stats(results, **kwargs):
    figure_size = kwargs.get('figure_size', (17.5*cm,10*cm))
    style_path = kwargs.get('style', 'classic')
    save_plot = kwargs.get('save_plot', True)
    subfig_hspace = 0.5 / figure_size[0]
    subfig_wspace = 0.5 / figure_size[1]
    ndim = results['ndim']
    model_name = results['model_name']
    colors = ['navy', 'darkorange', 'brown', '#217b7e', 'mediumslateblue', 'gray']
    markers = ['o', 's', '^', 'v', '<', '>']
        

    with plt.style.context(style_path):
        fig = plt.figure(constrained_layout=True, figsize=figure_size)
        gs = GridSpec(2, 2, figure=fig, width_ratios=[1.5, 1])
        sub_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:,0], hspace=subfig_hspace, wspace=subfig_wspace)
        axs = [fig.add_subplot(cell) for cell in sub_gs]
        # Plotting for the two conditions
        for idx, ax in enumerate(axs):
            # Extracting the data for this subplot
            ks_data = results['ks_values'][:, :, idx]
            ks_data = ks_data.reshape(-1)
            group = np.repeat(results['Ndt'], results['n_iter'])  # Assuming 7 conditions
            # Plotting the distribution of KS-distances
            sns.violinplot(x=group, y=ks_data, ax=ax, color=colors[idx], label=f'$x_{idx+1}$')
            # ax.set_xticklabels([f"{Ndt}" for i in range(7)])
            ax.set_xlabel(r'$T$')
            ax.set_ylabel("KS distance")
            ax.legend()
        
        sub_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:,1], hspace=subfig_hspace, wspace=subfig_wspace)
        axs1 = [fig.add_subplot(cell) for cell in sub_gs]
        # Plotting for the two conditions
        for idx, ax in enumerate(axs1):
            pvals = results['p_values'][:, :, idx]
            ax.boxplot(pvals.T, showfliers=False) 
            ax.set_xticks(ticks=np.arange(1, 8), labels=results['Ndt'])
            ax.set_xlabel(r'$T$')
            ax.set_ylabel(r'p-values')
        
        if save_plot:
            plt.savefig(os.path.join(os.getcwd(), 'result', model_name, f'{model_name}_ks_stats.pdf'))
            plt.close()
