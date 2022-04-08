import numpy as np
import matplotlib.pylab as plt
from bemcee.corner_HDR import corner
import sys
import importlib
import organizer as info
from bemcee.be_theory import obl2W, hfrac2tms
from bemcee.hpd import hpd_grid
from PyAstronomy import pyasl
from bemcee.utils import griddataBAtlas, griddataBA, linfit, lineProf
from bemcee.lines_reading import read_espadons

mod_name = sys.argv[1] + "_" + "user_settings"
flag = importlib.import_module(mod_name)

lines_dict = {"Ha": 0.6562801}

Nwalk = 500
nint_mcmc = 5000

if sys.argv[1] == "acol":
    af = "0.27"
    date = "22-03-13-233220"
    tag = "+acol_SigmaClipData_vsiniPrior_distPrior+votable+iue"
elif sys.argv[1] == "bcmi":
    af = "0.22"
    date = "22-03-17-050958"
    tag = "+acol_SigmaClipData_vsiniPrior_distPrior+votable+iue"

line = "Ha"

current_folder = str(flag.folder_fig) + str(flag.stars) + "/"
fig_name = (
    "Walkers_"
    + str(Nwalk)
    + "_Nmcmc_"
    + str(nint_mcmc)
    + "_af_"
    + str(af)
    + "_a_"
    + str(flag.a_parameter)
    + tag
)
file_npy = (
    flag.folder_fig
    + str(flag.stars)
    + "/"
    + date
    + "Walkers_"
    + str(Nwalk)
    + "_Nmcmc_"
    + str(nint_mcmc)
    + "_af_"
    + str(af)
    + "_a_"
    + str(flag.a_parameter)
    + tag
    + ".npy"
)

chain = np.load(file_npy)

flatchain_1 = chain.reshape((-1, chain.shape[-1]))


samples = np.copy(flatchain_1)[-100000:]


# for i in range(len(samples)):
#     if flag.model == "acol":
#         samples[i][1] = obl2W(samples[i][1])
#         samples[i][2] = hfrac2tms(samples[i][2])
#         samples[i][6] = (np.arccos(samples[i][6])) * (180.0 / np.pi)
#
#     if flag.model == "aeri":
#         samples[i][3] = (np.arccos(samples[i][3])) * (180.0 / np.pi)
#
#     if flag.model == "beatlas":
#         samples[i][1] = obl2W(samples[i][1])
#         samples[i][4] = (np.arccos(samples[i][4])) * (180.0 / np.pi)

# ==================
# READ SPECTRA
# ==================
lbd_central = lines_dict[line]

# table = flag.folder_data + str(flag.stars) + "/" + "spectra/" + "list_spectra.txt"
# specname = np.genfromtxt(table, comments="#", dtype="str")
# file_ha = str(flag.folder_data) + str(flag.stars) + "/spectra/" + str(specname)
# lbd_data, flux_data, MJD = read_espadons(file_ha)
# dflux_data = flux_data * 0.04
# vel_data, fx_data = lineProf(lbd_data, flux_data, hwidth=5000.0, lbc=lbd_central * 1e4)

# ==================
# READ MODELS
# ==================

u = np.where(info.lista_obs == line)
index = u[0][0]
# # Observations
# # logF_line = logF[index]
flux_data = info.logF[index]
dflux_data = info.dlogF[index]
# # dflux_line = dlogF[index] * flux_line

# keep = np.where(flux_line > 0)  # avoid plot zero flux

par_list = []
inds = np.random.randint(len(samples), size=300)
for ind in inds:
    params = samples[ind]
    par_list.append(params)

lbd_data = info.wave[index]
F_list = np.zeros([len(par_list), len(flux_data)])
F_list_unnorm = np.zeros([len(par_list), len(flux_data)])
# chi2 = np.zeros(len(F_list))
for i, params in enumerate(par_list):
    if flag.binary_star:
        F_mod_line_1_list = griddataBA(
            info.minfo,
            info.logF_grid[index],
            params[: -info.lim],
            info.listpar,
            info.dims,
        )
        F_mod_line_2_list = griddataBA(
            info.minfo,
            info.logF_grid[index],
            np.array([params[-1], 0.1, params[2], params[3]]),
            info.listpar,
            info.dims,
        )
        F_list[i] = linfit(info.wave[index], F_mod_line_1_list + F_mod_line_2_list)
        # logF_list[i] = np.log(norm_spectra(lbd[index], F_list))
    else:
        F_list_unnorm[i] = griddataBA(
            info.minfo,
            info.logF_grid[index],
            params[: -info.lim],
            info.listpar,
            info.dims,
        )
        F_list[i] = linfit(info.wave[index], F_list_unnorm[i])


fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})
# Plot models
for i in range(len(par_list)):
    vl, fx = lineProf(lbd_data, F_list[i], hwidth=5000.0, lbc=lbd_central)
    # ax1.plot(lbd_line, F_list[i], color='gray', alpha=0.1)
    ax1.plot(vl, fx, color="gray", alpha=0.1)
vel_data, fx_data = lineProf(lbd_data, flux_data, hwidth=5000.0, lbc=lbd_central)
ax1.errorbar(
    vel_data,
    fx_data,
    yerr=dflux_data,
    ls="",
    marker="o",
    alpha=0.5,
    ms=5,
    color="k",
    linewidth=1,
)

ax1.set_ylabel("Normalized Flux")
# ax1.set_xlim(min(vl), max(vl))
ax1.set_xlim(-700, +700)
# ax1.set_ylim(-0.05, 4)
# ax3.set_title(line)

ax2.plot(vl, (fx_data - fx) / dflux_data, marker="o", color="k", alpha=0.5)

ax2.set_ylabel(r"$(F-F_\mathrm{m})/\sigma$")
ax2.set_xlabel(r"Velocity [km/s]")
ax2.sharex(ax1)
ax1.tick_params(axis="both", which="major")
ax2.tick_params(axis="both", which="major")


plt.show()
