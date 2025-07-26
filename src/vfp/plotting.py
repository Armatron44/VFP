# standard
from __future__ import annotations

import copy
from enum import IntEnum, StrEnum, auto
from typing import TYPE_CHECKING, Literal

# third party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from refnx.reflect.interface import Step
from scipy import stats

if TYPE_CHECKING:
    from vfp.basevfp import BaseVFP


class PlotType(StrEnum):
    SLD = "sld"
    VFP = "vfp"
    SURFACES = "surfaces"

    def _plot_sld(
        self,
        ax: Axes,
        vfp: BaseVFP,
        posterior: bool,
        microslice: bool = True,
        total_sld: bool = False,
    ) -> None:
        """
        Plots sld profile.

        Parameters
        ----------
        ax : Axes
            Which axes to plot vfp profile on.
        vfp : BaseVFP
            Concrete child instance of BaseVFP to plot.
        posterior : bool
            Flag to indicate if plotting posterior samples when calling
            function.
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
        # use z_and_sld to get lims that match vfp and surfaces.
        z_for_lim, _ = vfp.z_and_sld()
        def_xlower_lim, def_xupper_lim = self._calc_xlims(z_for_lim)

        ss_condition = vfp.vfp_attrs.spin_state if total_sld else "none"
        sld_to_plot, sld_label = _tot_sld(all_slds, ss_condition)
        alpha = 0.03 if posterior else 1
        sld_to_plot_kwargs = dict(
            alpha=alpha, label=None if posterior else sld_label
        )

        ax.plot(z, sld_to_plot, color="k", **sld_to_plot_kwargs)

        # plot sldi if any are nonzero.
        if all_slds[:, 1].any():
            plot_sldi_kwargs = dict(
                alpha=alpha,
                label=None if posterior else r"$\mathrm{SLD}_{\mathrm{i}}$",
            )
            ax_twinx = ax.twinx()
            ax_twinx.plot(
                z, all_slds[:, 1], color="tab:red", **plot_sldi_kwargs
            )

        # if plotting slds separate & they are non zero.
        if not total_sld and all_slds[:, 2].any():
            plot_sldm_kwargs = dict(
                alpha=alpha,
                label=None if posterior else r"$\mathrm{SLD}_{\mathrm{m}}$",
            )
            ax.plot(z, all_slds[:, 2], color="tab:grey", **plot_sldm_kwargs)

        if not posterior:
            # format
            if all_slds[
                :, 1
            ].any():  # format the right-hand side y axis if used.
                ax_twinx.set_ylabel(
                    (
                        r"$\mathrm{SLD}_{\mathrm{i}}$ / "
                        r"$\mathrm{\AA{}}^{-2} \times 10^{-6}$",
                    ),
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

    def _plot_vfp(  # noqa: PLR0913
        self,
        ax: Axes,
        vfp: BaseVFP,
        posterior: bool,
        colours: tuple[tuple[float, float, float], ...] | None = None,
        total_vf: bool = True,
        labels: list[str] | None = None,
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
        ax : Axes
            Which axes to plot vfp profile on.
        vfp : BaseVFP
            Concrete child instance of BaseVFP to plot.
        posterior : bool
            Flag to indicate if plotting posterior samples.
        colours : tuple[tuple[float, float, float], ...] | None, optional
            Colours to plot vfp profile. Posterior samples are plotted in
            every second colour, while the nominal profile of each layer
            is plotted in every odd colour. matplotlib's tab20 is default.
        total_vf : bool, optional.
            If true, plots the total_vf of the representative profiles by
            summing across all layers' volume fractions. Defaults to True.
        labels : list[str] | None , optional
            Labels to be applied to the legend of the volume fraction profile.
        """
        if labels is None:
            labels = [f"Layer {i}" for i in range(len(vfp.tup_thicks) + 1)]
            labels[0], labels[-1] = "Fronting", "Backing"
            labels = (
                labels[::-1]
                if vfp.vfp_attrs.orientation == "back"
                else labels
            )
        colours = (
            colours
            if colours is not None
            else matplotlib.colormaps["tab20"].colors
        )

        vfs = vfp.vfs_for_display()[0]
        z = vfp.z_and_sld()[0]
        xlower_lim, xupper_lim = self._calc_xlims(z)
        if posterior:
            if vfp.vfp_attrs.orientation == "front":
                for i, lay_vfp in enumerate(vfs):
                    ax.plot(
                        z,
                        lay_vfp,
                        alpha=0.05,
                        color=colours[(1 + (2 * i)) % len(colours)],
                        zorder=i,
                    )
            elif vfp.vfp_attrs.orientation == "back":
                for i, lay_vfp in enumerate(vfs):
                    ax.plot(
                        z,
                        lay_vfp,
                        alpha=0.05,
                        color=colours[  # reverse colour order.
                            ((2 * len(vfp.tup_thicks) + 1) - 2 * i)
                            % len(colours)
                        ],
                        zorder=len(vfp.tup_thicks) - i,
                    )
        else:
            if vfp.vfp_attrs.orientation == "front":
                for i, lay_vfp in enumerate(vfs):
                    ax.plot(
                        z,
                        lay_vfp,
                        label=labels[i],
                        zorder=len(vfp.tup_thicks) + i,
                    )

            elif vfp.vfp_attrs.orientation == "back":
                for i, lay_vfp in enumerate(vfs):
                    ax.plot(
                        z,
                        lay_vfp,
                        label=labels[i],
                        color=colours[  # reverse colour order.
                            (2 * len(vfp.tup_thicks) - 2 * i) % len(colours)
                        ],
                        zorder=2 * len(vfp.tup_thicks) - i,
                    )

            if total_vf:
                ax.plot(
                    z,
                    np.sum(vfs, axis=0),
                    label=r"Total",
                    linestyle="--",
                    color="k",
                )
            ax.set_ylabel(r"Volume Fraction")
            ax.set_xlim(xlower_lim, xupper_lim)
            ax.legend(frameon=False)

    def _plot_surfaces(
        self,
        ax: Axes,
        vfp: BaseVFP,
        surfaces: np.ndarray,
        points: int,
        colours: tuple[tuple[float, float, float], ...] | None = None,
    ) -> None:
        """
        Plots a stochastic simulation of layers.

        Parameters
        ----------
        ax : Axes
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
        colours = (
            matplotlib.colormaps["tab20"].colors
            if colours is None
            else colours
        )
        # reverse and select for fill + points.
        if vfp.vfp_attrs.orientation == "front":
            points_colours = colours[: 2 * n_interf + 1 : 2]
            fill_colours = colours[1 : 2 * n_interf + 2 : 2]
            # zorder should decrease away from fronting:
            points_zorder = np.arange(start=2 * n_interf + 1, stop=1, step=-2)
            fill_zorder = np.arange(start=2 * n_interf, stop=-1, step=-2)
        else:
            points_colours = colours[: 2 * n_interf - 1 : 2][
                ::-1
            ]  # keep point colour as found in front.
            fill_colours = colours[1 : 2 * n_interf + 2 : 2][::-1]
            # zorder increase as progress toward fronting.
            points_zorder = (
                np.arange(start=1, stop=2 * n_interf + 1, step=2) + 2
            )
            fill_zorder = np.arange(start=0, stop=2 * n_interf + 1, step=2)

        # attempt to recreate margin that would be found in vfp plot.
        z = vfp.z_and_sld()[0]
        def_xlower_lim, def_xupper_lim = self._calc_xlims(z)

        # plot surfaces.
        # if vfp.vfp_attrs.orientation == "front":
        for i, j in enumerate(surfaces):  # do the surfaces
            ax.plot(
                j,
                range(0, points),
                marker=".",
                zorder=points_zorder[i],
                color=points_colours[i % len(points_colours)],  # loop colours
            )

        for i in range(0, n_interf + 1):  # then do the fills
            if i == 0:
                ax.fill_betweenx(
                    y=range(0, points),
                    x1=def_xlower_lim,
                    x2=surfaces[i],
                    interpolate=True,
                    color=fill_colours[i % len(fill_colours)],
                    zorder=fill_zorder[i],
                )

            elif i < len(vfp.tup_thicks):
                ax.fill_betweenx(
                    y=range(0, points),
                    x1=surfaces[i - 1],
                    x2=surfaces[i],
                    where=surfaces[i] > surfaces[i - 1],
                    interpolate=True,
                    color=fill_colours[i % len(fill_colours)],
                    zorder=fill_zorder[i],
                )

            elif i == len(vfp.tup_thicks):
                ax.fill_betweenx(
                    y=range(0, points),
                    x1=surfaces[i - 1],
                    x2=def_xupper_lim,
                    where=def_xupper_lim > surfaces[i - 1],
                    interpolate=True,
                    color=fill_colours[i % len(fill_colours)],
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

    def plot(self, *args, **kwargs) -> None:
        """
        Wraps specific plot functions depending on PlotType.
        """
        plot_func = plot_dispatch.get(self)
        if plot_func:
            plot_func(self, *args, **kwargs)
        else:
            raise NotImplementedError(
                f"Plot function for {self.value} not implemented."
            )

    def _calc_xlims(self, z: np.ndarray) -> tuple[float, float]:
        """
        Calculates horizontal limits for axes given `z`.

        `z` maybe in ascending or descending order, so the tuple
        is sorted before being returned to ensure lower lim is
        always lower.

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
        lims = z[0] - margin, z[-1] + margin
        return sorted(lims)


# create a map of PlotType members to plot fns in PlotType.
plot_dispatch = {
    PlotType.SLD: PlotType._plot_sld,
    PlotType.VFP: PlotType._plot_vfp,
    PlotType.SURFACES: PlotType._plot_surfaces,
}


class AxesIndex(IntEnum):
    """
    Defines index of multiple axes.

    Intended to be created by passing a list of strings
    representing the required plots to
    `AxesIndex.from_requested_plots_list`.
    """

    FIRST = 0
    SECOND = auto()
    THIRD = auto()

    def __new__(cls, value):
        """
        Create's AxesIndex member with values defined
        above in the enum member definitions.
        Also set _plot_type attr to None.
        """
        member = int.__new__(cls, value)
        member._value_ = value
        member._plot_type = None
        return member

    @property
    def plot_type(self) -> PlotType:
        """
        The plot type of this axis.
        Set from the requested plots.
        """
        return self._plot_type

    @plot_type.setter
    def plot_type(self, value: PlotType):
        self._plot_type = value

    @classmethod
    def from_requested_plots_list(
        cls, requested_plots: list[str]
    ) -> list[AxesIndex]:
        """
        Creates AxesIndex from `requested_plots` list.

        Parameters
        ----------
        requested_plots : list[str]
            Plots requested. Strings can be all or
            some combination of "sld", "vfp",
            "surfaces".

        Returns
        -------
        list[AxesIndex]
        """
        # all possible plot types
        plot_types_map = {pt.value: pt for pt in PlotType}
        # filter the PlotTypes requested.
        requested_plot_types = [
            plot_types_map[plot_name] for plot_name in requested_plots
        ]
        req_plots_and_axis = []
        for pt in requested_plot_types:
            axis = cls(
                requested_plot_types.index(pt)
            )  # create AxesEnum from PlotType.
            axis.plot_type = pt  # set plot_type property to a PlotType
            req_plots_and_axis.append(axis)
        return req_plots_and_axis


def surfaces_for_display(
    vfp: BaseVFP, points: int, rng: np.random.Generator
) -> np.ndarray:
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
    rng : np.random.Generator
        An initialised pseudo random number generator.

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
            loc=interf_loc[i],
            scale=float(roughs[i]),
            size=points,
            random_state=rng,
        )

    if vfp.vfp_attrs.orientation == "front":
        idx_where_conformal = (vfp.vfp_attrs.conformal == 1).nonzero()[0]
        # insert the conformal interfaces.
        if num_conform > 0:
            for i in idx_where_conformal:
                interf_arr[i] = (
                    np.max(interf_arr[:i].T, axis=1) + vfp.tup_thicks[i]
                )

    elif vfp.vfp_attrs.orientation == "back":
        idx_where_conformal = (vfp.vfp_attrs.conformal == 1).nonzero()[0] - (
            vfp.vfp_attrs.conformal.size - 1
        )
        if num_conform > 0:
            for i in idx_where_conformal:
                interf_arr[i] = (
                    np.min(interf_arr[i + 1 :].T, axis=1)
                    - vfp.tup_thicks[::-1][i]
                )

    return interf_arr


def model_plot(  # noqa: PLR0913
    vfp: BaseVFP,
    plots_required: list[Literal["sld", "vfp", "surfaces"]],
    posterior_samples: dict[str, np.ndarray] | None,
    surface_points: int,
    surface_rng: np.random.Generator,
    fig: Figure | None,
    sld_plot_kwargs: dict | None,
    vfp_plot_kwargs: dict | None,
    surface_plot_kwargs: dict | None,
) -> tuple[Figure, Axes | np.ndarray[Axes]]:
    """
    Visualises the vfp model.

    See vfp.basevfp.plot for extended details.

    Parameters
    ----------
    vfp : BaseVFP
        The VFP object which describes the interface.
    plots_required : list[Literal["sld", "vfp", "surfaces"]]
        A list of plots required. Possible acceptable string values are
        "sld", "vfp", "surfaces". The order of the strings in the list
        will affect the order of the plot. Duplicates will be ignored.
    posterior_samples : dict[str, np.ndarray] | None
        Samples from the posterior to plot in the "sld" and "vfp" plots.
        The keys should match the names of varying parameters in the vfp.
        Array values should be 1D of parameter values.
        If None, no posterior samples will be plotted.
    surface_points : integer
        Number of points to simulate across each interface.
    surface_rng : np.random.Generator
        Random number generator for producing draws from each interface's
        modelled distribution.
    fig : Figure | None
        If supplied, plots will be plotted on `fig`. If None, a new Figure
        will be created.
    sld_plot_kwargs : dict | None
        Kwargs to be passed to PlotType._plot_sld.
    vfp_plot_kwargs : dict | None
        Kwargs to be passed to PlotType._plot_vfp.
    surface_plot_kwargs : dict | None
        Kwargs to be passed to PlotType._plot_surfaces.

    Returns
    -------
    tuple[Figure, Axes | np.ndarray[Axes]]
        Figure and axes objects.
    """
    if surface_points <= 0:
        raise ValueError("surface_points must be > 0.")

    # get axes index for required plots.
    axes_enum = AxesIndex.from_requested_plots_list(
        requested_plots=plots_required
    )

    # add on two additional points to create fill effect on surfaces
    surface_points += 2
    surfaces = surfaces_for_display(vfp, surface_points, surface_rng)

    # get original vfp varying_parameter values
    original_ps = copy.deepcopy(vfp.varying_parameters)

    # setup fig & axes.
    if fig is None:
        fig, _ = plt.subplots(
            nrows=len(plots_required),
            ncols=1,
            sharex=True,
            figsize=(8, 3 * len(plots_required)),
        )

    else:
        for i in range(len(plots_required)):
            fig.add_subplot(len(plots_required), 1, i)

    # get ax this way so that its a flat list for 1 or multiple axes.
    ax = fig.axes

    # create maps for kwargs that can be passed to plot_type.plot.
    kwarg_map = {
        PlotType.SLD: sld_plot_kwargs,
        PlotType.VFP: vfp_plot_kwargs,
        PlotType.SURFACES: surface_plot_kwargs,
    }

    # plot posterior samples:
    if posterior_samples is not None:
        # get length of each set of posterior samples.
        p_samps_lens = set([len(val) for val in posterior_samples.values()])
        # check they are the same length.
        if len(p_samps_lens) != 1:
            raise ValueError("Posterior samples are of different lengths.")

        plot_fn_args_map_posterior = {
            PlotType.SLD: (vfp, True),
            PlotType.VFP: (vfp, True),
        }

        length_of_samples = next(iter(p_samps_lens))
        for i in range(length_of_samples):
            vfp.varying_parameters = {
                key: values[i] for key, values in posterior_samples.items()
            }
            for axis in axes_enum:
                # can't plot a posterior on the surfaces plot.
                if axis.plot_type == PlotType.SURFACES:
                    continue
                plot_args = plot_fn_args_map_posterior.get(axis.plot_type)
                plot_kwargs = kwarg_map.get(axis.plot_type)
                plot_kwargs = plot_kwargs if plot_kwargs is not None else {}
                axis.plot_type.plot(ax[axis], *plot_args, **plot_kwargs)

    # plot main profiles.
    vfp.varying_parameters = original_ps  # set to original values.

    plot_fn_args_map = {
        PlotType.SLD: (vfp, False),
        PlotType.VFP: (vfp, False),
        PlotType.SURFACES: (vfp, surfaces, surface_points),
    }

    for axis in axes_enum:
        plot_args = plot_fn_args_map.get(axis.plot_type)
        plot_kwargs = kwarg_map.get(axis.plot_type)
        plot_kwargs = plot_kwargs if plot_kwargs is not None else {}
        axis.plot_type.plot(ax[axis], *plot_args, **plot_kwargs)

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
    av_slds = [0.5 * np.diff(sld_row) + sld_row[:-1] for sld_row in slds]

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
        all_slds += (
            stepper(zed, scale=0, loc=dist[i])[:, None] * delta_all_slds[i]
        )

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
