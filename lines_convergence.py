import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import numpy as np
import user_settings as flag
from lines_reading import check_list
matplotlib.rcParams['font.family'] = "sans-serif"
font_color = "black"
tick_color = "black"


# ==============================================================================
def plot_convergence(npy, file_name, file_npy_burnin,  lista_obs, linspace, param_to_latex):

    converged_idx = 0

    #if flag.model == 'befavor' or flag.model == 'aeri':
    #    if flag.box_W:
    #        Wmin = 0.8
    #    else:
    #        Wmin = 0.
    #
	#		
    #    if check_list(lista_obs, 'UV'):
    #        linspace = [np.linspace(3.4, 14.6, 5),
    #                    np.linspace(Wmin, 1.00, 5),
    #                    np.linspace(0.0, 1.00, 5),
    #                    np.linspace(0.0, 1.0, 5),
    #                    np.linspace(50, 130, 5),
    #                    np.linspace(0, 0.1, 5)]				
    #    else:
    #        linspace = [np.linspace(3.4, 14.6, 8),
    #                    np.linspace(Wmin, 1.00, 5),
    #                    np.linspace(0.0, 1.00, 5),
    #                    np.linspace(0.0, 1.0, 5)]
    #    else:            
    #        if flag.normal_spectra is False:
    #            linspace = [np.linspace(3.4, 14.6, 8),
    #                        np.linspace(Wmin, 1.00, 5),
    #                        np.linspace(0.0, 1.00, 5),
    #                        np.linspace(0.0, 1.0, 5),
    #                        np.linspace(30, 50, 4),
    #                        np.linspace(0, 0.1, 5)]
    #
    #                    
    #if flag.model == 'aara' or flag.model == 'acol' or flag.model == 'bcmi':
    #    linspace = [np.linspace(3.4, 14.6, 8),
    #                np.linspace(1.0, 1.45, 6),
    #                np.linspace(0.0, 1.00, 5),
    #                np.linspace(12., 14.0, 5),
    #                np.linspace(0., 30.0, 5),
    #                np.linspace(0.5, 2.0, 4),
    #                np.linspace(0.0, 1.0, 5),
    #                np.linspace(50, 130, 5),
    #                np.linspace(0, 0.1, 5)]
    #if flag.model == 'beatlas':
    #    linspace = [np.linspace(3.8, 14.6, 5),
    #                np.linspace(1.0, 1.45, 6),
    #                np.linspace(0.0, 1.00, 5),
    #                np.linspace(3.0, 4.5, 4),
    #                np.linspace(0.0, 1.0, 5),
    #                np.linspace(50, 130, 5),
    #                np.linspace(0, 0.1, 5)]
    #
    #if flag.include_rv:
    #    linspace.append(np.linspace() )
    #
    ## Map the codified parameter names to their sexy latex equivalents
    #if flag.model == 'aeri':
    #    
    #    if check_list(lista_obs, 'UV'):
    #        param_to_latex = dict(mass=r'$M\,[M_\odot]$',
    #                              W=r"$W$",
    #                              age=r"$t/t_{ms}$", inc=r'$\cos i$',
    #                              dis=r'$d\,[pc]$', ebv=r'$E(B-V)$')
    #        params = ["mass", "W", "age", "inc", "dis", "ebv"]
    #        fig = plt.figure(figsize=(16, 20.6))
    #    else:
    #        param_to_latex = dict(mass=r'$M\,[M_\odot]$',
    #                              W=r"$W$",
    #                              age=r"$t/t_{ms}$", inc=r'$\cos i$')
    #        params = ["mass", "W", "age", "inc"]
    #        fig = plt.figure(figsize=(16, 20.6))
    #    
    #    if flag.normal_spectra is False:
    #        param_to_latex = dict(mass=r'$M\,[M_\odot]$',
    #                              W=r"$W$",
    #                              age=r"$t/t_{ms}$", inc=r'$\cos i$',
    #                              dis=r'$d\,[pc]$', ebv=r'$E(B-V)$')
    #        params = ["mass", "W", "age", "inc", "dis", "ebv"]
    #        fig = plt.figure(figsize=(16, 20.6))
    #
    #if flag.model == 'befavor':
    #    param_to_latex = dict(mass=r'$M\,[M_\odot]$',
    #                          oblat=r"$R_\mathrm{eq} / R_\mathrm{pole}$",
    #                          age=r"$H_\mathrm{frac}$", inc=r'$\cos i$',
    #                          dis=r'$d\,[pc]$', ebv=r'$E(B-V)$')
    #    params = ["mass", "oblat", "age", "inc", "dis", "ebv"]
    #    fig = plt.figure(figsize=(16, 20.6))
    #if flag.model == 'aara' or flag.model == 'acol' or flag.model == 'bcmi':
    #    param_to_latex = dict(mass=r'$M\,[M_\odot]$',
    #                          oblat=r"$R_\mathrm{eq} / R_\mathrm{pole}$",
    #                          age=r"$H_\mathrm{frac}$",
    #                          logn0=r'$\log \, n_0 \, [\mathrm{cm}^{-3}]$',
    #                          rdk=r'$R_\mathrm{D}\, [R_\star]$',
    #                          inc=r'$\cos i$', nix=r'$m$',
    #                          dis=r'$d\,[\mathrm{pc}]$', ebv=r'$E(B-V)$')
    #    params = ["mass", "oblat", "age", "logn0", "rdk", "nix",
    #              "inc", "dis", "ebv"]
    #    fig = plt.figure(figsize=(16, 24.6))
    #if flag.model == 'beatlas':
    #    param_to_latex = dict(mass=r'$M\,[M_\odot]$',
    #                          oblat=r"$R_\mathrm{eq} / R_\mathrm{pole}$",
    #                          sig0=r'$\Sigma_0$',
    #                          nix=r'$n$',
    #                          inc=r'$\cos i$',
    #                          dis=r'$d\,[pc]$', ebv=r'$E(B-V)$')
    #    params = ["mass", "oblat", "sig0", "nix", "inc", "dis", "ebv"]
    #    fig = plt.figure(figsize=(16, 20.6))
    fig = plt.figure(figsize=(16, 24.6))
    # Load the main chain and the burn-in phase chain
    chain = np.load(npy)

    chain_burnin = np.load(file_npy_burnin)

    # chain = chain[(acceptance_fractions > 0.20) &
    #               (acceptance_fractions < 0.5)]

    gs = gridspec.GridSpec(len(param_to_latex), 3)
    # gs.update(hspace=0.10, wspace=0.025, top=0.85, bottom=0.44)
    gs.update(hspace=0.25)
    print(len(param_to_latex))

    for ii in range(len(param_to_latex)):
        these_chains = chain[:, :, ii]
        these_chains2 = chain_burnin[:, :, ii]
        
        max_var = max(np.var(these_chains[:, converged_idx:], axis=1))
        max_var2 = max(np.var(these_chains2[:, converged_idx:], axis=1))

        ax1 = plt.subplot(gs[ii, :2])

        ax1.axvline(0, color="#67A9CF", alpha=0.7, linewidth=2)

        # proposta: fazer um for antes igual com as these_chains2, com outra cor
        # dai comecar o these_chains com np.arange(len(walker)) + numero-steps-burnin
        # fazer teste no ipython de leitura da chain e o que significa these_chains
        if ii == 3 and flag.model == 'aeri':
            linspace[ii] = np.cos(linspace[ii] * (np.pi/180))
            param_to_latex[ii] = r'cosi'
        
        for walker2 in these_chains2:
            ax1.plot(np.arange(len(walker2)) + 1, walker2, drawstyle='steps',    #np.arange comeca em 0, mas len(walker2) eh numero de steps. somaria 1 se quisesse comecar no step1
                     color='r', alpha=0.5)

        for walker in these_chains:
            ax1.plot(np.arange(len(walker)) - converged_idx + len(chain_burnin[0,:]), walker,     # adicionando + len(chain_burnin[:,0]), que eh numero de steps
                     drawstyle="steps",
                     color='b',
                     alpha=0.5)

        ax1.set_ylabel(param_to_latex[ii], fontsize=30, labelpad=18,
                       rotation="vertical", color=font_color)

        # Don't show ticks on the y-axis
        ax1.yaxis.set_ticks([])

        # For the plot on the bottom, add an x-axis label. Hide all others
        if ii == len(param_to_latex) - 1:
            ax1.set_xlabel("step number", fontsize=24, labelpad=18,
                           color=font_color)
        else:
            ax1.xaxis.set_visible(False)

        ax2 = plt.subplot(gs[ii, 2])

        ax2.hist(np.ravel(these_chains[:, converged_idx:]),
                 bins=np.linspace(ax1.get_ylim()[0], ax1.get_ylim()[1], 20),
                 orientation='horizontal',
                 facecolor="#67A9CF",
                 edgecolor="none",
                 histtype='barstacked')

        ax2.xaxis.set_visible(False)
        ax2.yaxis.tick_right()

        # print(ii)
        # print(param)
        ax2.set_yticks(linspace[ii])
        ax1.set_ylim(ax2.get_ylim())

        if ii == 0:
            t = ax1.set_title("Walkers", fontsize=30, color=font_color)
            t.set_y(1.01)
            t = ax2.set_title("Posterior", fontsize=30, color=font_color)
            t.set_y(1.01)

        ax1.tick_params(axis='x', pad=2, direction='out',
                        colors=tick_color, labelsize=22)
        ax2.tick_params(axis='y', pad=2, direction='out',
                        colors=tick_color, labelsize=22)

        ax1.get_xaxis().tick_bottom()

    fig.subplots_adjust(hspace=0.0, wspace=0.0, bottom=0.075, top=0.9,
                        left=0.12, right=0.88)
    plt.savefig(file_name + '.png')
