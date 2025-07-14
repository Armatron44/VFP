# standard
from __future__ import annotations
from functools import partial
import copy
from enum import IntEnum, StrEnum, auto
from typing import TYPE_CHECKING

# third party
import numpy as np
from refnx.reflect.interface import Step
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

if TYPE_CHECKING:
    from vfp.basevfp import BaseVFP


class PlotType(StrEnum):
    SLD = "sld"
    VFP = "vfp"
    SURFACES = "surfaces"
    
    def plot(self, *args, **kwargs) -> None:
        """
        Wraps specific plot functions depending on PlotType.
        """
        if self == PlotType.SLD:
            self._plot_sld(*args, **kwargs)
        elif self == PlotType.VFP:
            self._plot_vfp(*args, **kwargs)
        elif self == PlotType.SURFACES:
            self._plot_surfaces(*args, **kwargs)
    
    def _plot_sld(
        self, 
        ax: plt.Axes,
        vfp: BaseVFP,
        posterior: bool,
        microslice: bool = True,
        total_sld: bool = False
    ) -> None:
        """
        Plots sld profile.

        Parameters
        ----------
        ax : plt.Axes
            Which axes to plot vfp profile on.
        vfp : BaseVFP
            Concrete child instance of BaseVFP to plot.
        posterior : bool
            Flag to indicate if plotting posterior samples when calling function.
        microslice : bool, optional
            Flag to plot sld as microsliced slabs as fed into refnx / refl1d.
            If False, continuous sld is plotted as calculated from vfp. 
            By default True.
        total_sld : bool, optional
            If true, plots sldn (possibly) +/- sldm. 
            Else, plots sldn, sldm separately.
            By default, False.
        """
        # get slds to plot (1d z, 2d all_slds (z points, sld type))
        if microslice:
            z, all_slds = _gen_sld_profile(vfp)
            z = z + vfp.sld_offset()
        else:
            z, all_slds = vfp.z_and_sld()
        
        # the above return slightly different z lengths
        # we'll use z_and_sld to get appropriate lims.
        z_for_lim, _ = vfp.z_and_sld()
        z_for_lim = z_for_lim if vfp.vfp_attrs.orientation == 'front' else z_for_lim[::-1]
        def_xlower_lim, def_xupper_lim = self._calc_xlims(z_for_lim)
        
        ss_condition = vfp.vfp_attrs.spin_state if total_sld else "none"
        sld_to_plot, sld_label = _tot_sld(all_slds, ss_condition)
        alpha = 0.03 if posterior else 1
        sld_to_plot_kwargs = dict(
            alpha=alpha, 
            label=None if posterior else sld_label
        )
        #z = -z+np.sum(vfp.vfp_attrs.thicknesses) if vfp.vfp_attrs.orientation == 'back' else z
        ax.plot(z,
                sld_to_plot,
                color="k",
                **sld_to_plot_kwargs
                )
        
        # plot sldi if any are nonzero.
        if all_slds[:, 1].any():
            plot_sldi_kwargs = dict(
                alpha=alpha, 
                label=None if posterior else r"$\mathrm{SLD}_{\mathrm{i}}$"
            )
            ax_twinx = ax.twinx()
            ax_twinx.plot(
                z, 
                all_slds[:, 1], 
                color="tab:red",
                **plot_sldi_kwargs)
        
        # if plotting slds separate & they are non zero.  
        if not total_sld and all_slds[:, 2].any():
            plot_sldm_kwargs = dict(
                alpha=alpha, 
                label=None if posterior else r"$\mathrm{SLD}_{\mathrm{m}}$"
            )
            ax.plot(z,
                    all_slds[:, 2],
                    color="tab:grey",
                    **plot_sldm_kwargs
                    )
        
        if not posterior:
            # format 
            if all_slds[:, 1].any(): # format the right-hand side y axis if used.
                ax_twinx.set_ylabel(
                    r"$\mathrm{SLD}_{\mathrm{i}}$ / $\mathrm{\AA{}}^{-2} \times 10^{-6}$",
                    color="tab:red",
                )
                ax_twinx.tick_params(axis="y", colors="tab:red")
                ax.set_zorder(
                    ax_twinx.get_zorder() + 1
                )  # puts nsld and mslds above the isld.
                ax.patch.set_visible(
                    False
                )  # make sure the isld isn't obscured by the first axis.
            ax.legend(frameon=False)
            ax.set_ylabel(r"SLD / $\mathrm{\AA{}}^{-2} \times 10^{-6}$")
            ax.set_xlim(def_xlower_lim, def_xupper_lim)
            
    def _plot_vfp(
        self, 
        ax: plt.Axes, 
        vfp: BaseVFP, 
        posterior: bool, 
        colours: tuple[tuple[float, float, float], ...] | None = None, 
        total_vf: bool = True, 
        labels: list[str] | None = None
    ) -> None:
        """
        Plots the vfp profile on a given axis.
        
        Notes 
        -----
        When orientation = back, the return from 
        `vfp.vfs_for_display` values are reversed.
        Want to apply same colours to same material
        if one had two vfps with opposite orientations.

        Parameters
        ----------
        ax : plt.Axes
            Which axes to plot vfp profile on.
        vfp : BaseVFP
            Concrete child instance of BaseVFP to plot.
        posterior : bool
            Flag to indicate if plotting posterior samples when calling function.
        colours : tuple[tuple[float, float, float], ...] | None, optional
            Colours to plot vfp profile. Posterior samples are plotted in
            every second colour, while the representative profile of each layer
            is plotted in every odd colour. matplotlib's tab20 is default.
        total_vf : bool, optional.
            If true, plots the total_vf of the representative profiles by summing
            across all layers' volume fractions. Defaults to True.
        labels : list[str] | None , optional
            Labels to be applied to the legend of the volume fraction profile plot.
        """
        if labels is None:
            labels = [f"Layer {i}" for i in range(len(vfp.tup_thicks) + 1)]
            labels[0], labels[-1] = "Fronting", "Backing"
            labels = labels[::-1] if vfp.vfp_attrs.orientation == 'back' else labels
        colours = colours if colours is not None else matplotlib.colormaps["tab20"].colors
        
        vfs = vfp.vfs_for_display()[0]
        z = vfp.z_and_sld()[0]
        z = z if vfp.vfp_attrs.orientation == 'front' else z[::-1]
        def_xlower_lim, def_xupper_lim = self._calc_xlims(z)
        
        if posterior:
            if vfp.vfp_attrs.orientation == 'front':
                for i, lay_vfp in enumerate(vfs):
                    ax.plot(
                        z,
                        lay_vfp.T,
                        alpha=0.05,
                        color=colours[(1+(2*i)) % len(colours)],
                        zorder=i
                    )
            elif vfp.vfp_attrs.orientation == 'back':
                for i, lay_vfp in enumerate(vfs):
                    ax.plot(
                        z,
                        lay_vfp.T,
                        alpha=0.05,
                        color=colours[ # reverse colour order.
                            ((2 * len(vfp.tup_thicks) + 1) - 2 * i) % len(colours)
                        ],
                        zorder=len(vfp.tup_thicks) - i
                    )        
        else:
            if vfp.vfp_attrs.orientation == "front":
                for i, lay_vfp in enumerate(vfs):
                    ax.plot(z, lay_vfp.T, label=labels[i], zorder=len(vfp.tup_thicks) + i)

            elif vfp.vfp_attrs.orientation == "back":
                for i, lay_vfp in enumerate(vfs):
                    ax.plot(
                        z, 
                        lay_vfp.T,
                        label=labels[i],
                        color=colours[ # reverse colour order.
                            (2 * len(vfp.tup_thicks) - 2 * i) % len(colours)                        
                        ],
                        zorder=2*len(vfp.tup_thicks) - i
                    )

            if total_vf:
                ax.plot(
                    z, np.sum(vfs.T, axis=1), label=r"Total", linestyle="--", color="k"
                )
            ax.set_ylabel(r"Volume Fraction")
            ax.set_xlim(def_xlower_lim, def_xupper_lim)
            ax.legend(frameon=False)
    
    def _plot_surfaces(
        self, 
        ax: plt.Axes, 
        vfp: BaseVFP, 
        surfaces: np.ndarray, 
        points: int, 
        colours: tuple[tuple[float, float, float], ...] | None = None
    ) -> None:
        """
        Plots a stochastic simulation of layers.
        
        Parameters
        ----------
        ax : plt.Axes
            Which axes to plot vfp profile on.
        vfp : BaseVFP
            Concrete child instance of BaseVFP to plot.
        surfaces : np.ndarray
            RVs to plot.
        points : int
           Number of points to plot across the surfaces
        colours : tuple[tuple[float, float, float], ...] | None, optional
            Colours to plot. Defaults to tab20
        """
        n_interf = len(vfp.tup_thicks)
        # get default colours if non specified.
        colours = matplotlib.colormaps["tab20"].colors if colours is None else colours
        # reverse and select for fill + points.
        if vfp.vfp_attrs.orientation == 'front':
            points_colours = colours[:2*n_interf+1:2]
            fill_colours = colours[1:2*n_interf+2:2]
            # zorder should decrease away from fronting:
            points_zorder = np.arange(start=2*n_interf+1, stop=1, step=-2)
            fill_zorder = np.arange(start=2*n_interf, stop=-1, step=-2)
        else:
            points_colours = colours[:2*n_interf-1:2][::-1] # keep point colour as found in front.
            fill_colours = colours[1:2*n_interf+2:2][::-1]
            # zorder increase as progress toward fronting.
            points_zorder = np.arange(start=1, stop=2*n_interf+1, step=2) + 2
            fill_zorder = np.arange(start=0, stop=2*n_interf+1, step=2)
        
        # attempt to recreate margin that would be found in vfp plot.
        z = vfp.z_and_sld()[0]
        # flip z if back orientation so that we can define lower and upper lims
        # from first and last z position in array.
        z = z[::-1] if vfp.vfp_attrs.orientation == 'back' else z
        def_xlower_lim, def_xupper_lim = self._calc_xlims(z)
        
        # plot surfaces.
        # if vfp.vfp_attrs.orientation == "front":
        for i, j in enumerate(surfaces):  # do the surfaces
            ax.plot(
                j,
                range(0, points),
                marker=".",
                zorder=points_zorder[i],
                color=points_colours[
                    i % len(points_colours) # loop colours
                ]
            )

        for i in range(0, n_interf + 1):  # then do the fills
            if i == 0:
                ax.fill_betweenx(
                    y=range(0, points),
                    x1=def_xlower_lim,
                    x2=surfaces[i],
                    interpolate=True,
                    color=fill_colours[
                        i % len(fill_colours)
                    ],
                    zorder=fill_zorder[i],
                )

            elif i < len(vfp.tup_thicks):
                ax.fill_betweenx(
                    y=range(0, points),
                    x1=surfaces[i - 1],
                    x2=surfaces[i],
                    where=surfaces[i] > surfaces[i - 1],
                    interpolate=True,
                    color=fill_colours[
                        i % len(fill_colours)
                    ],
                    zorder=fill_zorder[i],
                )

            elif i == len(vfp.tup_thicks):
                ax.fill_betweenx(
                    y=range(0, points),
                    x1=surfaces[i - 1],
                    x2=def_xupper_lim,
                    where=def_xupper_lim > surfaces[i - 1],
                    interpolate=True,
                    color=fill_colours[
                        i % len(fill_colours)
                    ],
                    zorder=fill_zorder[i],
                )

        # define some y limits for ax[2] that allow for points > 0.
        ylower = 0.5
        yupper = (points - 2) + ((points - 1) - (points - 2)) / 2

        ax.set_xlabel(r"Distance over Interface / $\mathrm{\AA{}}$")
        ax.set_yticks([])
        # set the x limits to the original x limits before plotting the fills.
        ax.set_xlim(def_xlower_lim, def_xupper_lim)
        ax.set_ylim(ylower, yupper)  # chop off the extra two points

        for border in ["top", "bottom", "left", "right"]:
            ax.spines[border].set_zorder(
                (len(vfp.tup_thicks) + 1) * 3
            )  # borders will be higher than surfaces and fills.
            
    def _calc_xlims(
        self,
        z: np.ndarray
    ) -> tuple[float, float]:
        """
        Calculates horizontal limits for axes given `z`.

        Parameters
        ----------
        z : np.ndarray
            The z coordinate over the vfp structure.

        Returns
        -------
        tuple[float, float]
            Lower and upper x limits
        """
        margin = 0.05 * (z[-1] - z[0])
        return z[0] - margin, z[-1] + margin
    

class AxesIndex(IntEnum):
    FIRST = 0
    SECOND = auto()
    THIRD = auto()
    
    def __new__(cls, value):
        member = int.__new__(cls, value)
        member._value_ = value
        member._plot_type = None
        return member
    
    @property
    def plot_type(self) -> PlotType:
        return self._plot_type
    
    @plot_type.setter
    def plot_type(self, value: PlotType):
        self._plot_type = value
    
    @classmethod
    def from_requested_plots_list(
        cls, 
        requested_plots: list[str]
    ) -> list[AxesIndex]:
        plot_types_map = {pt.value: pt for pt in PlotType}
        requested_plot_types = [plot_types_map[plot_name] for plot_name in requested_plots]
        req_plots_and_axis = []
        for pt in requested_plot_types:
            axis = cls(requested_plot_types.index(pt))
            axis.plot_type = pt
            req_plots_and_axis.append(axis)
        return req_plots_and_axis


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
    posterior_samples: dict[str, np.ndarray] | None = None,
    plots_required: list[str] | None = None,
    fig: matplotlib.figure.Figure | None = None,
    sld_plot_kwargs: dict | None = None,
    vfp_plot_kwargs: dict | None = None,
    surface_plot_kwargs: dict | None = None,
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
    posterior_samples : dict[str, np.ndarray] | None, optional
        If supplied, will plot the posterior profiles.
    plots_required : list[str] | None, optional
        A list of "sld", "vfp" and "surfaces".
        Order in the list will affect the order of the plots.
    fig : matplotlib.figure.Figure | None = None,
    sld_plot_kwargs : dict | None = None, optional
    vfp_plot_kwargs : dict | None = None, optional
    surface_plot_kwargs : dict | None = None, optional
    
    Returns
    -------
    tuple[matplotlib.figure.Figure, np.ndarray[plt.Axes]]
        matplotlib.pyplot figure and axes objects.
    """
    if points <= 0:
        raise ValueError("points must be > 0.")
    
    # run check on unique vals in plots_required
    possible_plots = ["sld", "vfp", "surfaces"]
    if isinstance(plots_required, list):
        # remove duplicates, but preserve order.
        plots_required = list(dict.fromkeys(plots_required))
        if not all([
            ptype in possible_plots for ptype in plots_required
        ]):
            raise ValueError(f'Check plots_required only contains "sld", "vfp", "surfaces".')
    elif plots_required is None:
        plots_required = ["sld", "vfp", "surfaces"]
    else:
        raise TypeError(f'plots_required must be a list, got {type(required_plots)}.')
    
    # get axes index for required plots.
    axes_enum = AxesIndex.from_requested_plots_list(
        requested_plots=plots_required
    )
    
    # add on two additional points to create fill effect on y axis of bottom plot
    points += 2  
    surfaces = surfaces_for_display(vfp, points=points)
    
    # get original vfp varying_parameter values
    original_ps = copy.deepcopy(vfp.varying_parameters)

    # setup fig & axes.
    if fig is None:
        fig, _ = plt.subplots(
            nrows=len(plots_required), 
            ncols=1, 
            sharex=True, 
            figsize=(8, 3 * len(plots_required))
        )

    else:
        for i in range(len(plots_required)):
            fig.add_subplot(len(plots_required), 1, i)
    
    # get ax this way so that its a flat list for 1 or multiple axes.
    ax = fig.axes

    # plot posterior samples:
    if posterior_samples is not None:
        for i in range(list(posterior_samples.values())[0].size):
            vfp.varying_parameters = {
                key : values[i] for key, values in posterior_samples.items()
            }
            for axis in axes_enum:
                if axis.plot_type == PlotType.SLD:
                    plot_kwargs = sld_plot_kwargs
                elif axis.plot_type == PlotType.VFP:
                    plot_kwargs = vfp_plot_kwargs
                elif axis.plot_type == PlotType.SURFACES:
                    continue # not able to plot posterior for surface plot
                if plot_kwargs is not None:
                    axis.plot_type.plot(ax[axis], vfp, True, **plot_kwargs)
                else:
                    axis.plot_type.plot(ax[axis], vfp, True)
    
    # plot main profiles.    
    vfp.varying_parameters = original_ps # set to original values.
    for axis in axes_enum:
        # setup partially frozen function as all are common to these args.
        plot_fn = partial(axis.plot_type.plot, ax[axis], vfp, False)
        if axis.plot_type == PlotType.SLD:
            plot_kwargs = sld_plot_kwargs
        elif axis.plot_type == PlotType.VFP:
            plot_kwargs = vfp_plot_kwargs
        elif axis.plot_type == PlotType.SURFACES:
            plot_kwargs = surface_plot_kwargs
            # redefine plot_fn for surfaces as different args required.
            plot_fn = partial(axis.plot_type.plot, ax[axis], vfp, surfaces, points)
        if plot_kwargs is not None:
            plot_fn(**plot_kwargs)
        else:
            plot_fn()

    return fig, ax

def _gen_sld_profile(
    vfp: BaseVFP,
) -> tuple[np.ndarray, np.ndarray]:
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
    slds = vfp.get_slds()

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
    
    The returned value is the rows of sldn (possibly) +/- sldm.
    
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