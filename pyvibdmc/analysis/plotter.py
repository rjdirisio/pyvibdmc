import matplotlib.pyplot as plt


class Plotter:
    """
    A very basic plotting class that will use matplotlib to generate various plots using results from PyVibDMC/AnalyzeWfn/SimInfo.
    """

    # Convenient parameters to make plots look better
    params = {'text.usetex': False,
              'mathtext.fontset': 'dejavusans',
              'font.size': 14}
    plt.rcParams.update(params)

    @staticmethod
    def plt_vref_vs_tau(vref_vs_tau, save_name="vref_vs_tau.png"):
        """
        Takes in the vref vs tau array from a DMC sim and plots it. Can also take in many vref_vs_tau arrays and plot them successively

        :param vref_vs_tau: The vref_vs_tau array from the *sim_info.hdf5 file. Can be a list of these as well.
        :type vref_vs_tau: str or list
        :param save_name: The name of the plot that will save as a .png
        :type save_name: str
        :return:
        """
        if not isinstance(vref_vs_tau, list):
            vref_vs_tau = [vref_vs_tau]
        if len(vref_vs_tau) == 1:
            plt.plot(vref_vs_tau[0][:, 0], vref_vs_tau[0][:, 1], 'k')
        else:
            for vref in vref_vs_tau:
                plt.plot(vref[:, 0], vref[:, 1])
        plt.xlabel("Time step")
        plt.ylabel(r"$\mathrm{E_{ref}}$ ($\mathrm{cm^{-1}}$)")
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        plt.close()

    @staticmethod
    def plt_pop_vs_tau(pop_vs_tau, save_name="pop_vs_tau.png"):
        """
        Takes in the pop_vs_tau array from a DMC sim and plots it. Can also take in many pop_vs_tau arrays and plot them successively

        :param pop_vs_tau: The vref_vs_tau array from the *sim_info.hdf5 file. Can be a list of these as well.
        :type pop_vs_tau: str or list
        :param save_name: The name of the plot that will save as a .png
        :type save_name: str
        :return:
        """
        if not isinstance(pop_vs_tau, list):
            pop_vs_tau = [pop_vs_tau]
        if len(pop_vs_tau) == 1:
            plt.plot(pop_vs_tau[0][:, 0], pop_vs_tau[0][:, 1], 'k')
        else:
            for pop in pop_vs_tau:
                plt.plot(pop[:, 0], pop[:, 1])
        plt.xlabel("Time step")
        plt.ylabel(r"Population, Weights")
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        plt.close()

    @staticmethod
    def plt_hist1d(hist, xlabel, ylabel=r'Probability Amplitude ($\mathrm{\Psi^{2}}$)', title='',
                   save_name="histogram.png"):
        """
        Plots the histogram generated from AnalyzeWfn.projection_1d.

        :param hist: Output from AnalyzeWfn.projection_1d ; array of shape (bin_num-1 x 2)
        :type hist: np.ndarray
        :param xlabel: What to label the x-axis
        :type xlabel: str
        :param ylabel: What to label the y-axis (units in Parenthesis is always good)
        :type ylabel: str
        :param title: Title of the plot
        :type title: str
        :param save_name: name of the .png file that this plot will be saved to
        :type save_name: str
        """
        plt.plot(hist[:, 0], hist[:, 1])  # bins,amps
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        plt.close()

    @staticmethod
    def plt_hist2d(binsx, binsy, hist_2d, xlabel, ylabel, title='',
                   save_name="histogram.png"):
        """
        Plots the 2D-histogram generated from AnalyzeWfn.projection_2d.

        :param bins: First output from AnalyzeWfn.projection_2d; the array containing the x and y bins
        :type bins: np.ndarray
        :param hist_2d: Second output from AnalyzeWfn.projection_2d; The matrix that contains the amplitude at each of the bins in the 2d histogram.
        :type hist_2d: np.ndarray
        :param xlabel: What to label the x-axis
        :type xlabel: str
        :param ylabel: What to label the y-axis (units in Parenthesis is always good)
        :type ylabel: str
        :param title: Title of the plot
        :type title: str
        :param save_name: name of the .png file that this plot will be saved to
        :type save_name: str
        """

        from matplotlib import cm
        plt.contour(binsx, binsy, hist_2d, colors='k')
        plt.contourf(binsx, binsy, hist_2d, cmap=cm.viridis)
        cb = plt.colorbar()
        cb.set_label(r'Probability Amplitude ($\rm{\Psi^{2}}$)', rotation=270, labelpad=20)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        plt.close()
