#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  __init__.py
#
#  Copyright 2021 Amanda Rubio <amanda.rubio@usp.br>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#

from __future__ import division, print_function, absolute_import


from .utils import *

from .lines_radialv import *

from .be_theory import *

from .lines_reading import *

from .lines_convergence import *

from .lines_plot import *

from .hpd import *

from .lines_gauss import *

from .lines_emcee import *

from .corner_HDR import *


__version__ = "1.0"
__all__ = (
    "beta",
    "bin_data",
    "find_nearest",
    "find_neighbours",
    "geneva_interp_fast",
    "griddataBA",
    "griddataBA_new",
    "griddataBAtlas",
    "jy2cgs",
    "kde_scipy",
    "lineProf",
    "linfit",
    "quantile",
    "readBAsed",
    "readXDRsed",
    "readpck",
    "Ha_delta_v",
    "delta_v",
    "fwhm2sigma",
    "gauss",
    "W2oblat",
    "hfrac2tms",
    "obl2W",
    "oblat2w",
    "t_tms_from_Xc",
    "Sliding_Outlier_Removal",
    "check_list",
    "combine_sed",
    "create_list",
    "create_tag",
    "find_lim",
    "find_nearest2",
    "read_BAphot2_xdr",
    "read_aara_xdr",
    "read_acol_Ha_xdr",
    "read_beatlas_xdr",
    "read_befavor_xdr",
    "read_espadons",
    "read_iue",
    "read_line_spectra",
    "read_models",
    "read_observables",
    "read_star_info",
    "read_stellar_prior",
    "read_table",
    "read_votable",
    "select_xdr_part",
    "xdr_remove_lines",
    "plot_convergence",
    "par_errors",
    "plot_line",
    "plot_residuals_new",
    "print_output",
    "print_output_means",
    "print_to_latex",
    "hpd_grid",
    "lnlike",
    "lnprob",
    "lnprior",
    "emcee_inference",
    "run_emcee",
    "corner",
    "hist2d",
    "quantile",
)
