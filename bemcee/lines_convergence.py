import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import numpy as np
import importlib
from __init__ import mod_name

flag = importlib.import_module(mod_name)


matplotlib.rcParams["font.family"] = "sans-serif"
font_color = "black"
tick_color = "black"

# ==============================================================================
def plot_convergence(npy, file_name, file_npy_burnin, linspace, param_to_latex):
    """
    Makes and saves the trace plot

    Usage:
    plot_convergence(npy, file_name, file_npy_burnin, linspace, param_to_latex)
    """

    converged_idx = 0

    fig = plt.figure(figsize=(16, 24.6))

    # Load the main chain and the burn-in phase chain
    chain = np.load(npy)

    chain_burnin = np.load(file_npy_burnin)

    gs = gridspec.GridSpec(len(param_to_latex), 3)

    gs.update(hspace=0.25)

    for ii in range(len(param_to_latex)):
        these_chains = chain[:, :, ii]
        these_chains2 = chain_burnin[:, :, ii]

        max_var = max(np.var(these_chains[:, converged_idx:], axis=1))
        max_var2 = max(np.var(these_chains2[:, converged_idx:], axis=1))

        ax1 = plt.subplot(gs[ii, :2])

        ax1.axvline(0, color="#67A9CF", alpha=0.7, linewidth=2)

        if ii == 3 and flag.model == "aeri":
            linspace[ii] = np.cos(linspace[ii] * (np.pi / 180))
            param_to_latex[ii] = r"cosi"
        elif ii == 6 and flag.model == "acol":
            linspace[ii] = np.cos(linspace[ii] * (np.pi / 180))
            param_to_latex[ii] = r"cosi"
        elif ii == 4 and flag.model == "beatlas":
            linspace[ii] = np.cos(linspace[ii] * (np.pi / 180))
            param_to_latex[ii] = r"cosi"

        for walker2 in these_chains2:
            ax1.plot(
                np.arange(len(walker2)) + 1,
                walker2,
                drawstyle="steps",  # np.arange comeca em 0, mas len(walker2) eh numero de steps. somaria 1 se quisesse comecar no step1
                color="r",
                alpha=0.5,
            )

        for walker in these_chains:
            ax1.plot(
                np.arange(len(walker)) - converged_idx + len(chain_burnin[0, :]),
                walker,  # adicionando + len(chain_burnin[:,0]), que eh numero de steps
                drawstyle="steps",
                color="b",
                alpha=0.5,
            )

        ax1.set_ylabel(
            param_to_latex[ii],
            fontsize=30,
            labelpad=18,
            rotation="vertical",
            color=font_color,
        )

        # Don't show ticks on the y-axis
        ax1.yaxis.set_ticks([])

        # For the plot on the bottom, add an x-axis label. Hide all others
        if ii == len(param_to_latex) - 1:
            ax1.set_xlabel("step number", fontsize=24, labelpad=18, color=font_color)
        else:
            ax1.xaxis.set_visible(False)

        ax2 = plt.subplot(gs[ii, 2])

        ax2.hist(
            np.ravel(these_chains[:, converged_idx:]),
            bins=np.linspace(ax1.get_ylim()[0], ax1.get_ylim()[1], 20),
            orientation="horizontal",
            facecolor="#67A9CF",
            edgecolor="none",
            histtype="barstacked",
        )

        ax2.xaxis.set_visible(False)
        ax2.yaxis.tick_right()

        # print(ii)
        # print(param)
        # ax2.set_yticks(linspace[ii])
        ax1.set_ylim(ax2.get_ylim())

        if ii == 0:
            t = ax1.set_title("Walkers", fontsize=30, color=font_color)
            t.set_y(1.01)
            t = ax2.set_title("Posterior", fontsize=30, color=font_color)
            t.set_y(1.01)

        ax1.tick_params(
            axis="x", pad=2, direction="out", colors=tick_color, labelsize=22
        )
        ax2.tick_params(
            axis="y", pad=2, direction="out", colors=tick_color, labelsize=22
        )

        ax1.get_xaxis().tick_bottom()

    fig.subplots_adjust(
        hspace=0.0, wspace=0.0, bottom=0.075, top=0.9, left=0.12, right=0.88
    )
    plt.savefig(file_name + ".png")
