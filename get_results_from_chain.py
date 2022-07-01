import numpy as np
import sys
from konoha import obl2W, hfrac2tms
from konoha.hpd import hpd_grid


def get_result(nchain, star):
    # date = nchain.split('Walkers')[0]
    # Nwalk = nchain.split('Walkers')[1].split('_')[1]
    # Nmcmc = nchain.split('Walkers')[1].split('_')[3]
    # af = nchain.split('Walkers')[1].split('_')[5]
    # apar = nchain.split('Walkers')[1].split('+')[0][-3:]
    # tag = "+" + nchain.split('Walkers')[1].split('+')[-1]
    model = nchain.split("Walkers")[1].split("+")[1].split("_")[0]

    # current_folder = '../figures/' + star + "/"
    # fig_name = (
    #     "Walkers_"
    #     + Nwalk
    #     + "_Nmcmc_"
    #     + Nmcmc
    #     + "_af_"
    #     + af
    #     + "_a_"
    #     + apar
    #     + tag
    # )
    file_npy = "../figures/" + star + "/" + nchain

    chain = np.load(file_npy)

    flatchain_1 = chain.reshape((-1, chain.shape[-1]))
    Ndim = chain.shape[-1]

    samples = np.copy(flatchain_1)  # [-300000:]

    for i in range(len(samples)):
        if model == "acol" or model == "aara":
            samples[i][1] = obl2W(samples[i][1])
            samples[i][2] = hfrac2tms(samples[i][2])
            samples[i][6] = (np.arccos(samples[i][6])) * (180.0 / np.pi)

        elif model == "aeri":
            samples[i][3] = (np.arccos(samples[i][3])) * (180.0 / np.pi)

        elif model == "beatlas":
            samples[i][1] = obl2W(samples[i][1])
            samples[i][4] = (np.arccos(samples[i][4])) * (180.0 / np.pi)

        if model == "aara":
            samples[i][5] = samples[i][5] + 1.5

    new_ranges = []
    for i in range(Ndim):
        new_ranges.append(np.array([np.min(samples[:, i]), np.max(samples[:, i])]))

    best_pars = []
    best_errs = []
    hpds = []

    for i in range(Ndim):
        # print("## " + labels[i])
        # print(samples[:,i])
        hpd_mu, x_mu, y_mu, modes_mu = hpd_grid(samples[:, i], alpha=0.32, roundto=3)
        # mode_val = mode1(np.round(samples[:,i], decimals=2))
        bpars = []
        epars = []
        hpds.append(hpd_mu)
        for (x0, x1) in hpd_mu:
            # qvalues = hpd(samples[:,i], alpha=0.32)
            cut = samples[samples[:, i] > x0, i]
            cut = cut[cut < x1]
            median_val = np.median(cut)

            bpars.append(median_val)
            epars.append([x1 - median_val, median_val - x0])
            print(
                "{0:.5f} + {1:.5f} - {2:.5f}".format(
                    median_val, x1 - median_val, median_val - x0
                )
            )

        best_errs.append(epars)
        best_pars.append(bpars)

    return best_pars, best_errs


if __name__ == "__main__":
    star = sys.argv[1]
    nchain = "22-06-24-183819Walkers_500_Nmcmc_5000_af_0.22_a_2.0+acol_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO+Ha.npy"
    bpars, berrs = get_result(nchain, star)
