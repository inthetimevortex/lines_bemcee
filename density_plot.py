import numpy as np
from matplotlib import pyplot as plt

# library & dataset
import seaborn as sns
from matplotlib.lines import Line2D

sns.set_style("whitegrid")

# Ver https://seaborn.pydata.org/tutorial/distributions.html
#
# Para olhar um conjunto de cores:
# palette = sns.color_palette('bright', 10) #10 é o numero de cores
# sns.palplot(palette)
# Posso também passar uma lista personalizada como colormap: lista = ['#c7e9b4','#1d91c0','#081d58']
# my_cmap = ListedColormap(sns.color_palette(lista).as_hex())
#
# Outra opção (melhor): pafig, ax = plt.subplots()
# lette = sns.light_palette('#c7e9b4', 10, as_cmap = True)
# ou sns.dark_palette('', reverse=True, as_cmap = True) -- opção reverse é pra trocar ordem das cores.

star = "HD37795"

# Plot options ------------------------------
# Observable
Ha = True
SED = True
SED_Ha = True
SED_Pol = False


# Parameters
Mass = True
W = True
age = False
inclination = False

# Color options
colorHa = "xkcd:sunflower yellow"
colorSEDpol = "xkcd:grapefruit"
colorSED = "xkcd:sky"
colorSEDHa = "xkcd:pink"


# ------------------------------------------------
# The program starts below

cor_Ha = sns.light_palette(colorHa, 10, as_cmap=True)
cor_SED = sns.light_palette(colorSED, 10, as_cmap=True)
cor_SEDHa = sns.light_palette(colorSEDHa, 10, as_cmap=True)
cor_SEDpol = sns.light_palette(colorSEDpol, 10, as_cmap=True)


# Parameters
param = []
if Mass:
    param.append(0)
if W:
    param.append(1)
if age:
    param.append(2)
if inclination:
    param.append(3)

# Auxiliary variables
observable = []
filename = []
color = []
colorlabel = []

# Files
if Ha:
    chain_file = "22-03-23-102419Walkers_100_Nmcmc_100_af_0.08_a_2.0+acol_SigmaClipData_distPrior+Ha.npy"
    observable.append("H" + r"$\alpha$")
    filename.append(chain_file)
    color.append(cor_Ha)
    colorlabel.append(colorHa)
if SED:
    chain_file = "22-03-13-233220Walkers_500_Nmcmc_5000_af_0.27_a_2.0+acol_SigmaClipData_vsiniPrior_distPrior+votable+iue.npy"
    observable.append("SED")
    filename.append(chain_file)
    color.append(cor_SED)
    colorlabel.append(colorSED)
if SED_Ha:
    chain_file = "22-03-19-022726Walkers_600_Nmcmc_2000_af_0.07_a_1.2+acol_SigmaClipData_distPrior+votable+iue+Ha.npy"
    observable.append("SED+H" + r"$\alpha$")
    filename.append(chain_file)
    color.append(cor_SEDHa)
    colorlabel.append(colorSEDHa)
if SED_Pol:
    chain_file = "22-03-23-110348Walkers_100_Nmcmc_100_af_0.21_a_2.0+pol_SigmaClipData_distPrior.npy"
    observable.append("SED+Pol")
    filename.append(chain_file)
    color.append(cor_SEDpol)
    colorlabel.append(colorSEDpol)


# Density plot
for i in range(len(observable)):
    # Chain
    chain = np.load("../figures/" + star + "/" + filename[i])
    flatchain_M = chain[:, :, param[0]].flatten()[-5000:]  # Massa
    flatchain_W = chain[:, :, param[1]].flatten()[-5000:]  # W

    # 2D density plot
    ax = sns.kdeplot(
        flatchain_M,
        y=flatchain_W,
        color=colorlabel[i],
        # cmap=colorlabel[i],
        shade=True,
        alpha=0.7,
        label=observable[i],
        cbar=False,
        n_levels=5,
    )

    # plt.scatter(flatchain_M, flatchain_W, cmap=color[i], alpha=0.01, linewidth=0.1, s=5)

a = Line2D([], [], color=colorlabel[0], label=observable[0])
b = Line2D([], [], color=colorlabel[1], label=observable[1])
c = Line2D([], [], color=colorlabel[2], label=observable[2])
plt.legend(handles=[a, b, c], loc="lower right")  # , fontsize=14)

# ---- Escolher melhor os colormaps: ideal é ter um pra cada observável e uma cor única que signifique combinação.
plt.xlabel("Mass " + r"$(\rm M_\odot)$")  # , fontsize=16)
plt.ylabel("W")  # , fontsize=16)
# ax.set_facecolor(background)  # '#d3d5d4') # cor de fundo do gráfico
# plt.grid()
# plt.grid(color='#bebebf')
# ax.set_axisbelow(True)
# plt.tick_params(labelsize=12)
plt.show()
# Scatter plot --- não funcionou bem
# for i in range(len(observable)):
#    # Chain
#    chain = np.load(filename[i])
#    flatchain_M = chain[:,:,0].flatten() # Massa
#    flatchain_W = chain[:,:,1].flatten() # W

#    plt.scatter(flatchain_M, flatchain_W, cmap=color[i], label=observable[i], alpha=0.01, linewidth=0.1, s=5)
