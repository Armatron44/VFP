# standard
from __future__ import annotations

# third party
import numpy as np
from refnx.reflect.interface import Step
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

from vfp.basevfp import BaseVFP

# FIXME: Colours for orientation = back seem to be broken.
def surfaces_for_display(vfp: BaseVFP, 
                         points: int = 50) -> np.ndarray:
    """
    Produces 2D array of RVs to describe each interface.
    
    The number of random variates is controlled by `points`.
    Used to create a graphical representation of the modelled interfaces.

    Parameters
    ----------
    vfp : BaseVFP
        Object which describes the interface.
    points : integer
        Number of points to simulate across the surfaces.

    Returns
    -------
    np.array
        2d array of shape = (Nlayers - 1, points)
    """
    interf_loc = np.cumsum(vfp.tup_thicks)
    roughs = vfp.tup_roughs

    if vfp.vfp_attrs.orientation == "back":
        interf_loc = np.fabs(interf_loc - interf_loc[-1])[::-1]
        roughs = roughs[::-1]

    interf_arr = np.ones(shape=(interf_loc.size, points))
    num_conform = np.sum(vfp.vfp_attrs.conformal)

    # return the non-conformal interfaces.
    for i in range(interf_arr.shape[0]):
        interf_arr[i] = stats.norm.rvs(
            loc=interf_loc[i], scale=float(roughs[i]), size=points
        )

    if vfp.vfp_attrs.orientation == "front":
        idx_where_conformal = (vfp.vfp_attrs.conformal == 1).nonzero()[0]
        # insert the conformal interfaces.
        if num_conform > 0:
            for i in idx_where_conformal:
                interf_arr[i] = np.max(interf_arr[:i].T, axis=1) + vfp.tup_thicks[i]

    elif vfp.vfp_attrs.orientation == "back":
        idx_where_conformal = (vfp.vfp_attrs.conformal == 1).nonzero()[0] - (
            vfp.vfp_attrs.conformal.size - 1
        )
        if num_conform > 0:
            for i in idx_where_conformal:
                interf_arr[i] = (
                    np.min(interf_arr[i + 1 :].T, axis=1) - vfp.tup_thicks[::-1][i]
                )

    return interf_arr

def model_plot(
    vfp: BaseVFP,
    points: int = 50,
    microslice: bool = True,
    total_sld: bool = False,
    total_vf: bool = True,
) -> tuple[matplotlib.figure.Figure, np.ndarray[plt.Axes]]:
    """
    Produces a three axis figure to visualise VFP model.
    
    Top plot = nsld / msld / isld
    Middle plot = volume fraction profiles
    Bottom plot = surface profiles

    Parameters
    ----------
    vfp : BaseVFP
        The VFP object which describes the interface.
    points : integer
        Number of points to simulate across the surfaces.
    microslice : boolean
        If True, will return the microsliced sld profiles.
        Otherwise, get original sld profiles before they are averaged and
        microsliced.
    total_sld : boolean
        If True, returns sldn +/- sldm profiles. If false, the sldn and sldm
        are plotted seperately.
    total_vf : boolean
        If True, will plot the sum of all layers' volume fractions.

    Returns
    -------
    tuple[matplotlib.figure.Figure, np.ndarray[plt.Axes]]
        matplotlib.pyplot figure and axes objects.
    """
    if points <= 0:
        raise ValueError("points must be > 0.")

    # add on two additional points to create fill effect on y axis of bottom plot
    points += 2  
    surfaces = surfaces_for_display(vfp, points=points)

    # define some colours to use for the surface plot.
    colours = matplotlib.colormaps["tab20"].colors

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 9))

    # ax[0] - nsld / msld / isld. 
    # Only plots msld and isld curves if they are not zero.
    if microslice:
        z, all_slds = _gen_sld_profile(vfp)
    else:
        z, all_slds = vfp.z_and_sld_scatter()
    
    ss_condition = vfp.vfp_attrs.spin_state if total_sld else "none"

    sld_to_plot, sld_label = _tot_sld(all_slds, ss_condition)

    ax[0].plot(z + vfp.sld_offset(),
               sld_to_plot,
               color="k",
               label=sld_label
               )
    
    # plot sldi if any are nonzero.
    if all_slds[:, 1].any():
                ax0_twinx = ax[0].twinx()
                ax0_twinx.plot(z + vfp.sld_offset(), all_slds[:, 1], color="tab:red")
    
    # if plotting slds separate & they are non zero.  
    if not total_sld and all_slds[:, 2].any(): 
        ax[0].plot(z + vfp.sld_offset(),
                   all_slds[:, 2],
                   color="tab:grey",
                   label=r"$\mathrm{SLD}_{\mathrm{m}}$"
                   )

    # ax[1] - Volume fractions
    vfs = vfp.vfs_for_display()[0]
    z = vfp.z_and_sld_scatter()[0]

    if vfp.vfp_attrs.orientation == "front":
        for i, j in enumerate(vfs):
            if i == 0:
                ax[1].plot(z, j.T, label="Fronting")

            elif i + 1 == len(vfs):
                ax[1].plot(z, j.T, label="Backing")

            else:
                ax[1].plot(z, j.T, label=f"Layer {i}")

    elif vfp.vfp_attrs.orientation == "back":
        for i, j in enumerate(vfs):
            if i == 0:
                ax[1].plot(
                    z,
                    j.T,
                    label="Backing",
                    color=colours[
                        (2 * len(vfp.tup_thicks) - 2 * i) % len(colours)
                    ],
                    zorder=len(vfp.tup_thicks) - i,
                )

            elif i + 1 == len(vfs):
                ax[1].plot(
                    z,
                    j.T,
                    label="Fronting",
                    color=colours[
                        (2 * len(vfp.tup_thicks) - 2 * i) % len(colours)
                    ],
                    zorder=len(vfp.tup_thicks) - i,
                )

            else:
                ax[1].plot(
                    z,
                    j.T,
                    label=f"Layer {len(vfs) - (i + 1)}",
                    color=colours[
                        (2 * len(vfp.tup_thicks) - 2 * i) % len(colours)
                    ],
                    zorder=len(vfp.tup_thicks) - i,
                )

    if total_vf:
        ax[1].plot(
            z, np.sum(vfs.T, axis=1), label=r"Total", linestyle="--", color="k"
        )

    # ax[2] - surface plots
    def_xlower_lim, def_xupper_lim = ax[
        1
    ].get_xlim()  # get the default x limits from ax1 before filling

    if vfp.vfp_attrs.orientation == "front":
        for i, j in enumerate(surfaces):  # do the surfaces
            ax[2].plot(
                j,
                range(0, points),
                marker=".",
                zorder=(2 * len(vfp.tup_thicks) + 1 - 2 * i),
            )

        for i in range(0, len(vfp.tup_thicks) + 1):  # then do the fills
            if i == 0:
                ax[2].fill_betweenx(
                    y=range(0, points),
                    x1=def_xlower_lim - 1,
                    x2=surfaces[i],
                    interpolate=True,
                    color=colours[(2 * i % len(colours)) + 1],
                    zorder=(2 * len(vfp.tup_thicks) - 2 * i),
                )

            elif i < len(vfp.tup_thicks):
                ax[2].fill_betweenx(
                    y=range(0, points),
                    x1=surfaces[i - 1],
                    x2=surfaces[i],
                    where=surfaces[i] > surfaces[i - 1],
                    interpolate=True,
                    zorder=(2 * len(vfp.tup_thicks) - 2 * i),
                    color=colours[(2 * i % len(colours)) + 1],
                )

            elif i == len(vfp.tup_thicks):
                ax[2].fill_betweenx(
                    y=range(0, points),
                    x1=surfaces[i - 1],
                    x2=def_xupper_lim + 1,
                    where=def_xupper_lim + 1 > surfaces[i - 1],
                    interpolate=True,
                    zorder=(2 * len(vfp.tup_thicks) - 2 * i),
                    color=colours[(2 * i % len(colours)) + 1],
                )

    elif vfp.vfp_attrs.orientation == "back":
        for i, j in enumerate(surfaces):  # do the surfaces
            ax[2].plot(
                j,
                range(0, points),
                marker=".",
                color=colours[(len(vfp.tup_thicks) - 2 * i) % len(colours)],
                zorder=(len(vfp.tup_thicks) + 1 + 2 * i),
            )

        for i in range(0, len(vfp.tup_thicks) + 1):  # then do the fills
            if i == 0:
                ax[2].fill_betweenx(
                    y=range(0, points),
                    x1=def_xlower_lim - 1,
                    x2=surfaces[i],
                    interpolate=True,
                    color=colours[
                        (2 * len(vfp.tup_thicks) + 1 - 2 * i) % len(colours)
                    ],
                    zorder=2 * i,
                )

            elif i < len(vfp.tup_thicks):
                ax[2].fill_betweenx(
                    y=range(0, points),
                    x1=surfaces[i - 1],
                    x2=surfaces[i],
                    where=surfaces[i] > surfaces[i - 1],
                    interpolate=True,
                    color=colours[
                        (2 * len(vfp.tup_thicks) + 1 - 2 * i) % len(colours)
                    ],
                    zorder=2 * i,
                )

            elif i == len(vfp.tup_thicks):
                ax[2].fill_betweenx(
                    y=range(0, points),
                    x1=surfaces[i - 1],
                    x2=def_xupper_lim + 1,
                    where=def_xupper_lim + 1 > surfaces[i - 1],
                    interpolate=True,
                    color=colours[
                        (2 * len(vfp.tup_thicks) + 1 - 2 * i) % len(colours)
                    ],
                    zorder=2 * i,
                )

    # formatting
    ax[0].legend(frameon=False)
    ax[0].set_ylabel(r"SLD / $\mathrm{\AA{}}^{-2} \times 10^{-6}$")
    ax[1].set_ylabel(r"Volume Fraction")
    ax[1].legend(frameon=False)

    # format the right-hand side y axis.
    if all_slds[:, 1].any():
        ax0_twinx.set_ylabel(
            r"$\mathrm{SLD}_{\mathrm{i}}$ / $\mathrm{\AA{}}^{-2} \times 10^{-6}$",
            color="tab:red",
        )
        ax0_twinx.tick_params(axis="y", colors="tab:red")
        ax[0].set_zorder(
            ax0_twinx.get_zorder() + 1
        )  # puts nsld and mslds above the isld.
        ax[0].patch.set_visible(
            False
        )  # make sure the isld isn't completely obscured by the first axis.

    # define some y limits for ax[2] that allow for points > 0.
    ylower = 0.5
    yupper = (points - 2) + ((points - 1) - (points - 2)) / 2

    ax[2].set_xlabel(r"Distance over Interface / $\mathrm{\AA{}}$")
    ax[2].set_yticks([])
    # set the x limits to the original x limits before plotting the fills.
    ax[2].set_xlim(def_xlower_lim, def_xupper_lim)
    ax[2].set_ylim(ylower, yupper)  # chop off the extra two points

    for border in ["top", "bottom", "left", "right"]:
        ax[2].spines[border].set_zorder(
            (len(vfp.tup_thicks) + 1) * 3
        )  # borders will be higher than surfaces and fills.

    return fig, ax

def _gen_sld_profile(
    vfp: BaseVFP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate sld profiles (nuclear, magnetic and imaginary) from the VFP.

    Parameters
    ----------
    vfp : BaseVFP
        VFP object which contains the description of the interface.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]:
        Contains the z distance (first index in tuple) over the interface and
        a 2D array of slds in order of sldn, sldi, sldm.
    """
    # grab the original nSLD and mSLDs
    slds = vfp.calc_slds()

    # get the average between each nuclear and magnetic SLD value.
    av_slds = [
        0.5 * np.diff(sld_row) + sld_row[:-1] for sld_row in slds
        ]

    # init arrays for final slds.
    fin_slds = np.ones_like(slds)

    if vfp.vfp_attrs.orientation == "front":
        for i, av_sld in enumerate(av_slds):
            # fill all but last with average slds.
            fin_slds[i, :-1] = fin_slds[i, :-1] * av_sld
            # now set the final sld value to those from the micro arrays.
            fin_slds[i, -1] = slds[i, -1]

    elif vfp.vfp_attrs.orientation == "back":
        for i, av_sld in enumerate(av_slds):
            # do the same but backwards for back orientations.
            fin_slds[i, 1:] = fin_slds[i, 1:] * av_sld[::-1]
            # now set the final sld value to those from the micro arrays.
            fin_slds[i, 0] = slds[i, -1]

    # init a 2D array (Nlayers, 5)
    microslices = np.zeros(shape=(vfp.dz.size, 4))

    # populate microslices with microslab thicknesses & slds.
    microslices[:, 0] = vfp.dz
    microslices[:, 1:4] = fin_slds.T

    # calc how many layers, total z distance, start and end points.
    nslices = np.size(microslices, axis=0)
    dist = np.cumsum(microslices[:, 0])
    zstart = -5
    zend = 5 + dist[-1]

    # workout how much space the sld profile should encompass
    # (z array not provided)
    # use twice as many points as the real sld profile
    max_delta_z = float(vfp.vfp_attrs.max_delta_z) / 2
    npnts = int(np.ceil((zend - zstart) / max_delta_z)) + 1
    zed = np.linspace(zstart, zend, num=npnts)

    # the output arrays - starting sld value at zero.
    all_slds = np.ones_like(zed, dtype=float)[:, None] * microslices[0, 1:4]

    # work out the step in sld at an interface
    # the delta arrays are shape (nlayers - 1)
    delta_all_slds = microslices[1:, 1:4] - microslices[:-1, 1:4]

    stepper = Step()
    # accumulate the sld of each step.
    for i in range(nslices - 1):
        all_slds += stepper(zed, scale=0, loc=dist[i])[:, None] * delta_all_slds[i]

    return zed, all_slds

def _tot_sld(all_slds: np.ndarray, ss: str) -> tuple[np.ndarray, str]:
    """
    Get sld profile for plotting given spin state.
    
    Parameters
    ----------
    all_slds : np.ndarray
        2D array containing sldn, sldi, sldm.
    ss : str
        Spin state for conditioning which sld is plotted.
        
    Returns
    -------
    tuple[np.ndarray, str]
        sld for plotting in first index and label for sld plot in second.
    """
    if ss == "none":
        tot_sld = all_slds[:, 0]
        sld_label = r"$\mathrm{SLD}_{\mathrm{n}}$"
    elif ss == "down":
        tot_sld = all_slds[:, 0] - all_slds[:, 2]
        sld_label = r"$\mathrm{SLD}_{\mathrm{n}} - \mathrm{SLD}_{\mathrm{m}}$"
    elif ss == "up":
        tot_sld = all_slds[:, 0] + all_slds[:, 2]
        sld_label = r"$\mathrm{SLD}_{\mathrm{n}} + \mathrm{SLD}_{\mathrm{m}}$"
    
    return tot_sld, sld_label