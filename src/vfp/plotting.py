"""Plot interfacial model from concrete implementations of ``BaseVFP``."""

from __future__ import annotations

import copy
from collections import defaultdict
from enum import IntEnum, StrEnum, auto
from functools import wraps
from typing import TYPE_CHECKING, Literal, Self

# third party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure, SubFigure
from scipy import stats

from vfp.calc import heaviside_step
from vfp.vfp_typing import (
    ParameterLike,
    SldPlotKwargType,
    SurfacePlotKwargType,
    VfpPlotKwargType,
)

if TYPE_CHECKING:
    from vfp.basevfp import V

okabe_ito_cmap = plt.get_cmap("okabe_ito")
okabe_ito_colours = okabe_ito_cmap(range(okabe_ito_cmap.N))[:, :3]
tab20_cmap = plt.get_cmap("tab20")
tab20_colours = tab20_cmap(range(tab20_cmap.N))[:, :3]

axtwinx_cache: dict[Figure | SubFigure, Axes] = {}


def call_counter(func):
    """Count how many times a specific function has been called."""
    func_counter = defaultdict(int)

    @wraps(func)
    def helper(plttype, ax, *args, **kwargs):
        fig = ax.get_figure()
        if ax is None:
            raise ValueError("ax not connected to a Figure.")
        func(plttype, ax, *args, _n_call=func_counter[(fig, func)], **kwargs)
        func_counter[(fig, func)] += 1

    return helper


def get_axtwinx(ax: Axes) -> Axes:
    """Get from cache or set up a twinx of an axis."""
    fig = ax.get_figure()
    if fig is None:
        raise ValueError("Axis is not set on a figure.")
    axtwinx = axtwinx_cache.get(fig)
    if axtwinx is None:
        axtwinx = ax.twinx()
        axtwinx_cache[fig] = axtwinx
    return axtwinx


def lighten_colour(
    col: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Lighten an rgb colour tuple."""
    return tuple(
        matplotlib.colors.hsv_to_rgb(
            np.minimum(
                matplotlib.colors.rgb_to_hsv(col) + np.asarray([0, 0, 0.2]), 1
            )
        )
    )


class PlotType(StrEnum):
    """Specific plots supported by this module.

    Each type has its own private plot method, with the ``plot``
    method designed to provide an interface to these methods.
    """

    SLD_PLOT = "sld"
    VFP_PLOT = "vfp"
    SURFACES_PLOT = "surfaces"

    def plot(self, *args, **kwargs) -> None:
        """Wrap specific plot functions depending on PlotType."""
        kw = kwargs[self]
        kw = kw if kw is not None else {}
        match self:
            case PlotType.SLD_PLOT:
                self._plot_sld(*args, **kw)
            case PlotType.VFP_PLOT:
                self._plot_vfp(*args, **kw)
            case PlotType.SURFACES_PLOT:
                self._plot_surfaces(*args, **kw)

    @call_counter
    def _plot_sld(  # noqa : PLR0913
        self,
        ax: Axes,
        vfp: V,
        align_at_interface: int,
        posterior_samples: dict[str, np.typing.NDArray[np.float64]] | None,
        *,
        microslice: bool = True,
        total_sld: bool = False,
        _n_call: int,
    ) -> None:
        """Plot sld profile.

        Parameters
        ----------
        ax : Axes
            Which axes to plot vfp profile on.
        vfp : VFP | refnxVFP | refl1dVFP
            Instantiated concrete class of ``BaseVFP`` from which to plot the
            interfacial model.
        align_at_interface : int
            Specifies which interface defines z = 0.
        posterior_samples : dict[str, np.typing.NDArray[np.float64]] | None
            Samples from the posterior to plot.

        Kwargs
        ------
        microslice : bool, optional
            Flag to plot sld as microsliced slabs as fed into refnx / refl1d.
            If False, continuous sld is plotted as calculated from vfp.
            By default True.
        total_sld : bool, optional
            If true, plots sldn (possibly) +/- sldm.
            Else, plots sldn, sldm separately.
            By default, False.
        """
        ax_twinx = get_axtwinx(ax)
        # get slds to plot (1d z, 2d all_slds (z points, sld type))
        z, all_slds = vfp.z_and_sld(align_at_interface=align_at_interface)
        # get lims that match vfp and surfaces.
        def_xlower_lim, def_xupper_lim = self._calc_xlims(z)
        # recreate z and all_slds with microslabs.
        if microslice:
            z, all_slds = _gen_sld_profile(vfp, z)

        ss_condition = vfp.vfp_attrs.spin_state if total_sld else "none"
        num_lines = len(ax.lines) + len(ax_twinx.lines)
        sld_to_plot, sld_label = _tot_sld(all_slds, ss_condition, _n_call)
        sld_colour = okabe_ito_colours[num_lines]
        ax.plot(
            z,
            sld_to_plot,
            color=sld_colour,
            label=sld_label,
        )
        num_lines += 1
        # if plotting slds separate & they are non zero.
        if not total_sld and all_slds[:, 2].any():
            m_sld_colour = okabe_ito_colours[num_lines]
            ax.plot(
                z,
                all_slds[:, 2],
                color=m_sld_colour,
                label=rf"$\mathrm{{SLD}}_{{\mathrm{{m}}, {_n_call}}}$",
            )
            num_lines += 1
        # plot sldi if any are nonzero.
        h, _ = ax.get_legend_handles_labels()
        # ax_twinx = None
        if all_slds[:, 1].any():
            i_sld_colour = okabe_ito_colours[num_lines]
            ax_twinx.plot(
                z,
                all_slds[:, 1],
                color=i_sld_colour,
                label=rf"$\mathrm{{SLD}}_{{\mathrm{{i}}, {_n_call}}}$",
            )
            ax_twinx.set_ylabel(
                r"$\mathrm{SLD}_{\mathrm{i}}$ /"
                r" $\mathrm{\AA{}}^{-2} \times 10^{-6}$",
            )
            ax_twinx.tick_params(axis="y")
            ax.set_zorder(
                ax_twinx.get_zorder() + 1
            )  # puts nsld and mslds above the isld.
            ax.patch.set_visible(
                False
            )  # make sure the isld isn't obscured by the first axis.
            ax_twinx_h, _ = ax_twinx.get_legend_handles_labels()
            h.extend(ax_twinx_h)
        if posterior_samples is not None:
            m_sequences, sequences, i_sequences = [], [], []
            p_samps_lens = set(
                [len(val) for val in posterior_samples.values()]
            )
            for i in range(next(iter(p_samps_lens))):
                vfp.varying_parameters = {
                    k: v[i] for k, v in posterior_samples.items()
                }
                z, all_slds = vfp.z_and_sld(
                    align_at_interface=align_at_interface
                )
                if microslice:
                    z, all_slds = _gen_sld_profile(vfp, z)
                sld_to_plot, _ = _tot_sld(all_slds, ss_condition, _n_call)
                if not total_sld and all_slds[:, 2].any():
                    m_sequences.append(np.column_stack((z, all_slds[:, 2])))
                if all_slds[:, 1].any():
                    i_sequences.append(np.column_stack((z, all_slds[:, 1])))
                sequences.append(np.column_stack((z, sld_to_plot)))
            ax.add_collection(
                LineCollection(
                    sequences,
                    colors=lighten_colour(sld_colour),
                    alpha=0.03,
                    zorder=0,
                )
            )
            if m_sequences:
                ax.add_collection(
                    LineCollection(
                        m_sequences,
                        colors=lighten_colour(m_sld_colour),
                        alpha=0.03,
                        zorder=0,
                    )
                )
            if i_sequences:
                assert ax_twinx is not None
                ax_twinx.add_collection(
                    LineCollection(
                        i_sequences,
                        colors=lighten_colour(i_sld_colour),
                        alpha=0.03,
                        zorder=0,
                    )
                )
        ax.legend(handles=h, frameon=False)
        ax.set_ylabel(r"SLD / $\mathrm{\AA{}}^{-2} \times 10^{-6}$")
        ax.set_xlim(def_xlower_lim, def_xupper_lim)
        ax.set_title("sld", alpha=0)

    @call_counter
    def _plot_vfp(  # noqa : PLR0913
        self,
        ax: Axes,
        vfp: V,
        align_at_interface: int,
        posterior_samples: dict[str, np.typing.NDArray[np.float64]] | None,
        *,
        layer_materials: dict[int, dict[str, ParameterLike]] | None = None,
        line_colours: np.typing.NDArray[np.float64] | None = None,
        posterior_colours: np.typing.NDArray[np.float64] | None = None,
        total_vf: bool = True,
        labels: list[str] | None = None,
        _n_call: int,
    ) -> None:
        """Plot the vfp profile on a given axis.

        Notes
        -----
        When orientation = back, the return from `vfp.vfs_for_display` values
        are reversed. Want to apply same colours to same material if one had
        two vfps with opposite orientations.

        Parameters
        ----------
        ax : Axes
            Which axes to plot vfp profile on.
        vfp : VFP | refnxVFP | refl1dVFP
            Instantiated concrete class of ``BaseVFP`` from which to plot the
            interfacial model.
        align_at_interface : int
            Specifies which interface defines z = 0.
        posterior_samples : dict[str, np.typing.NDArray[np.float64]] | None
            Samples from the posterior to plot.

        Kwargs
        ------
        layer_materials : dict[int, dict[str, ParameterLike]] | None,
            optional. Each key is the layer number (e.g fronting = 0), while
            the value should be a dict, with keys that are material names
            within a given layer and values that are material volume
            fractions. The material names are used as labels, and will
            overwrite the ``labels`` kwarg.
        line_colours : np.typing.NDArray[np.float64] | None, optional
            Colours to plot vfp profile. Defaults to the even colours in
            matplotlib's tab20.
        posterior_colours : np.typing.NDArray[np.float64] | None, optional
            Colours to plot the posterior samples. The default values are
            matplotlib's tab20 odd colours.
        total_vf : bool, optional.
            If true, plots the total_vf of the representative profiles by
            summing across all layers' volume fractions. Defaults to True.
        labels : list[str] | None , optional
            Labels to be applied to the legend of the volume fraction profile.
            Order of labels should match the order of layers in vfp, from
            fronting to backing.
        """
        # get default labels
        def_labels = [
            rf"$\mathrm{{Layer}}_{_n_call}$ {i}"
            for i in range(len(vfp.vfp_attrs.tup_thicks) + 1)
        ]
        def_labels[0], def_labels[-1] = (
            rf"$\mathrm{{Fronting}}_{_n_call}$",
            rf"$\mathrm{{Backing}}_{_n_call}$",
        )

        if labels is None:
            labels = def_labels
            labels = (
                labels[::-1]
                if vfp.vfp_attrs.orientation == "back"
                else labels
            )
        else:
            # merge labels with default labels. This way, if too many
            # labels are supplied, the excess are ignored. Otherwise if
            # too little are supplied, fall back on using a merge of
            # supplied and default.
            labels = [
                labels[i] if i < len(labels) else def_labels[i]
                for i in range(len(def_labels))
            ]
        line_colours = (
            line_colours if line_colours is not None else tab20_colours[::2]
        )
        posterior_colours = (
            posterior_colours
            if posterior_colours is not None
            else tab20_colours[1::2]
        )
        vfs = vfp.vfs_for_display()[0]
        z = vfp.z_and_sld(align_at_interface=align_at_interface)[0]
        xlower_lim, xupper_lim = self._calc_xlims(z)
        if layer_materials is not None:
            vfs, labels = self._recalc_vfs_by_materials(
                layer_materials, vfs, vfp.vfp_attrs.orientation
            )
        num_lines = len(ax.lines)
        n_layers = vfs.shape[0]
        colour_indices = np.asarray(
            [
                i % len(line_colours)
                for i in range(num_lines, n_layers + num_lines)
            ]
        )
        # restrict colours to this round of plotting.
        line_colours = (
            line_colours[colour_indices[::-1]]
            if vfp.vfp_attrs.orientation == "back"
            else line_colours[colour_indices]
        )
        posterior_colours = (
            posterior_colours[colour_indices[::-1]]
            if vfp.vfp_attrs.orientation == "back"
            else posterior_colours[colour_indices]
        )
        if vfp.vfp_attrs.orientation == "front":
            for i, lay_vfp in enumerate(vfs):
                ax.plot(
                    z,
                    lay_vfp,
                    label=labels[i],
                    zorder=i + num_lines,
                    color=line_colours[i],
                )

        elif vfp.vfp_attrs.orientation == "back":
            for i, lay_vfp in enumerate(vfs):
                ax.plot(
                    z,
                    lay_vfp,
                    label=labels[i],
                    color=line_colours[i],
                    zorder=len(vfp.vfp_attrs.tup_thicks) + num_lines - i,
                )

        if total_vf:
            ax.plot(
                z,
                np.sum(vfs, axis=0),
                label=rf"$\mathrm{{Total}}_{_n_call}$",
                linestyle="--",
                color="k",
            )

        if posterior_samples is not None:
            sequences = []
            p_samps_lens = set(
                [len(val) for val in posterior_samples.values()]
            )
            for i in range(next(iter(p_samps_lens))):
                vfp.varying_parameters = {
                    k: v[i] for k, v in posterior_samples.items()
                }
                vfs = vfp.vfs_for_display()[0]
                if layer_materials is not None:
                    vfs, _ = self._recalc_vfs_by_materials(
                        layer_materials, vfs, vfp.vfp_attrs.orientation
                    )
                z = vfp.z_and_sld(align_at_interface=align_at_interface)[0]
                lay_vf_seq = np.column_stack((z, vfs.T))
                lay_vfps = [
                    lay_vf_seq[:, np.r_[0, i]] for i in range(1, n_layers + 1)
                ]
                sequences.append(lay_vfps)
            for i in range(n_layers):
                ax.add_collection(
                    LineCollection(
                        [seq[i] for seq in sequences],
                        colors=posterior_colours[i % len(posterior_colours)],
                        alpha=0.03,
                        zorder=(
                            i + num_lines
                            if vfp.vfp_attrs.orientation == "front"
                            else len(vfp.vfp_attrs.tup_thicks) + num_lines - i
                        ),
                    )
                )
        ax.set_ylabel(r"Volume Fraction")
        ax.set_xlim(xlower_lim, xupper_lim)
        ax.legend(frameon=False)
        ax.set_title("vfp", alpha=0)

    def _plot_surfaces(  # noqa: PLR0913
        self,
        ax: Axes,
        vfp: V,
        align_at_interface: int,
        *,
        surface_points: int = 50,
        surface_rng: np.random.Generator | None = None,
        surface_colours: tuple[tuple[float, float, float], ...] | None = None,
    ) -> None:
        """Plot a stochastic simulation of layers.

        Parameters
        ----------
        ax : Axes
            Which axes to plot vfp profile on.
        vfp : VFP | refnxVFP | refl1dVFP
            Instantiated concrete class of ``BaseVFP`` from which to plot the
            interfacial model.
        align_at_interface : int
            Specifies which interface defines z = 0.

        Kwargs
        ------
        surface_points : integer, optional
            Number of points to simulate across each interface.
            By default, 50.
        surface_rng : np.random.Generator | None, optional
            Random number generator for producing draws from each interface's
            modelled distribution. If supplied, will generate deterministic
            draws so that the results are repeatable. If not supplied, a
            random seed will be set when calling this function.
        surface_colours : tuple[tuple[float, float, float], ...] | None,
            Optional. Colours to plot. Defaults to tab20.
        """
        surface_rng = (
            surface_rng
            if surface_rng is not None
            else np.random.default_rng()
        )

        if surface_points <= 0:
            raise ValueError("surface_points must be > 0.")

        # add on two additional points to create fill effect on surfaces
        surface_points += 2
        surfaces = surfaces_for_display(
            vfp, surface_points, surface_rng, align_at_interface
        )
        n_interf = len(vfp.vfp_attrs.tup_thicks)
        # get default colours if non specified.
        colours = (
            tab20_colours if surface_colours is None else surface_colours
        )
        # reverse and select for fill + points.
        points_colours = colours[: 2 * n_interf + 1 : 2]
        fill_colours = colours[1 : 2 * n_interf + 2 : 2]
        points_zorder = np.arange(start=2 * n_interf + 1, stop=1, step=-2)
        fill_zorder = np.arange(start=2 * n_interf, stop=-1, step=-2)
        if vfp.vfp_attrs.orientation == "back":
            fill_colours = fill_colours[::-1]
            fill_zorder = fill_zorder[::-1]
        # attempt to recreate margin that would be found in vfp plot.
        z = vfp.z_and_sld(align_at_interface=align_at_interface)[0]
        def_xlower_lim, def_xupper_lim = self._calc_xlims(z)
        # plot surfaces.
        for i, j in enumerate(surfaces):
            ax.plot(
                j,
                range(0, surface_points),
                marker=".",
                zorder=points_zorder[i],
                color=points_colours[i % len(points_colours)],  # loop colours
            )

        surfaces = (
            surfaces[::-1]
            if vfp.vfp_attrs.orientation == "back"
            else surfaces
        )
        for i in range(0, n_interf + 1):  # then do the fills
            if i == 0:
                ax.fill_betweenx(
                    y=range(0, surface_points),
                    x1=def_xlower_lim,
                    x2=surfaces[i],
                    interpolate=True,
                    color=fill_colours[i % len(fill_colours)],
                    zorder=fill_zorder[i],
                )

            elif i < len(vfp.vfp_attrs.tup_thicks):
                ax.fill_betweenx(
                    y=range(0, surface_points),
                    x1=surfaces[i - 1],
                    x2=surfaces[i],
                    where=surfaces[i] > surfaces[i - 1],
                    interpolate=True,
                    color=fill_colours[i % len(fill_colours)],
                    zorder=fill_zorder[i],
                )

            elif i == len(vfp.vfp_attrs.tup_thicks):
                ax.fill_betweenx(
                    y=range(0, surface_points),
                    x1=surfaces[i - 1],
                    x2=def_xupper_lim,
                    where=def_xupper_lim > surfaces[i - 1],
                    interpolate=True,
                    color=fill_colours[i % len(fill_colours)],
                    zorder=fill_zorder[i],
                )

        # define some y limits for ax[2] that allow for points > 0.
        ylower = 0.5
        yupper = (surface_points - 2) + (
            (surface_points - 1) - (surface_points - 2)
        ) / 2

        ax.set_yticks([])
        # set the x limits to the original x limits before plotting the fills.
        ax.set_xlim(def_xlower_lim, def_xupper_lim)
        ax.set_ylim(ylower, yupper)  # chop off the extra two points
        ax.set_title("surfaces", alpha=0)
        for border in ["top", "bottom", "left", "right"]:
            ax.spines[border].set_zorder(
                (len(vfp.vfp_attrs.tup_thicks) + 1) * 3
            )  # borders will be higher than surfaces and fills.

    def _calc_xlims(
        self, z: np.typing.NDArray[np.float64]
    ) -> np.typing.NDArray[np.float64]:
        """Calculate horizontal limits for axes given `z`.

        `z` maybe in ascending or descending order, so the tuple
        is sorted before being returned to ensure lower lim is
        always lower.

        Parameters
        ----------
        z : np.typing.NDArray[np.float64]
            The z coordinate over the vfp structure.

        Returns
        -------
        np.typing.NDArray[np.float64]
            Lower and upper x limits
        """
        margin: np.float64 = 0.05 * (z[-1] - z[0])
        lims = np.sort(np.array([z[0] - margin, z[-1] + margin]))
        return lims

    def _recalc_vfs_by_materials(
        self,
        layer_materials: dict[int, dict[str, ParameterLike]],
        vfs: np.typing.NDArray[np.float64],
        orientation: Literal["front", "back"],
    ) -> tuple[np.typing.NDArray[np.float64], list[str]]:
        """
        Calculate volume fraction profiles for each material.

        Parameters
        ----------
        layer_materials : dict[int, dict[str, ParameterLike]]
            The volume fractions of materials in each layer. Key is the layer
            index that a set of material occupies.
        vfs : np.typing.NDArray[np.float64]
            volume fraction profile of each layer.
        orientation : str
            Either "front" or "back" from `vfp.vfp_attrs.orientation`.

        Returns
        -------
        tuple[np.typing.NDArray[np.float64], list[str]]
            The first index is the volume fraction profile of each material.
            Second is the name of each material for label names.
        """
        all_mats = [
            mat_name
            for matfrac in layer_materials.values()
            for mat_name in matfrac.keys()
        ]

        # maintain the order of first appearance.
        unique_materials: list[str] = []
        for mat in all_mats:
            if mat not in unique_materials:
                unique_materials.append(mat)
        unique_materials: list[str] = (
            unique_materials[::-1]
            if orientation == "back"
            else unique_materials
        )

        lay_vfp_dict: dict[tuple[int, str], float] = {}

        def lm_lookup(n: int) -> int:
            """Handle orientation-dependent lookups.

            When orientation is back, match up the layer materials with the
            vfs.
            """
            if orientation == "front":
                return n
            else:
                return (len(layer_materials) - 1) - n

        for i, lay in enumerate(vfs):
            mats_in_layer = layer_materials[lm_lookup(i)]
            for mat_in_layer in mats_in_layer:
                lay_vfp_dict[i, mat_in_layer] = lay * float(
                    mats_in_layer[mat_in_layer]
                )

        # calculate the sum over all layers for each individual material.
        new_vfs = np.zeros(shape=(len(unique_materials), vfs.shape[1]))
        for i, uniq_mat in enumerate(unique_materials):
            new_vfs[i] = np.vstack(
                [v for k, v in lay_vfp_dict.items() if uniq_mat == k[1]]
            ).sum(axis=0)

        return new_vfs, unique_materials


class AxesIndex(IntEnum):
    """Defines index of multiple axes.

    Intended to be created by passing a list of strings representing the
    required plots to ``AxesIndex.from_requested_plots_list``.

    Example
    -------
    >>> from vfp.plotting import AxesIndex
    >>> AxesIndex.from_requested_plots_list(["vfp", "sld"])
    [<AxesIndex.FIRST: 0>, <AxesIndex.SECOND: 1>]
    """

    FIRST = 0
    SECOND = auto()
    THIRD = auto()

    def __new__(cls, value):
        """Create ``AxesIndex`` member from value defined above."""
        member = int.__new__(cls, value)
        member._value_ = value
        # set _plot_type to None, to be set via ``PlotType``.
        member._plot_type = None
        return member

    @property
    def plot_type(self) -> PlotType:
        """The plot type of this axis. Set from the requested plots."""
        return self._plot_type

    @plot_type.setter
    def plot_type(self, value: PlotType) -> None:
        if not isinstance(value, PlotType):
            raise TypeError(
                "Can only set AxesIndex.plot_type to a PlotType."
                f" Got {type(value)}."
            )
        self._plot_type = value

    @classmethod
    def from_requested_plots_list(
        cls, requested_plots: list[Literal["sld", "vfp", "surfaces"]]
    ) -> list[Self]:
        """Create ``AxesIndex``s from ``requested_plots`` list.

        Parameters
        ----------
        requested_plots : list[Literal["sld", "vfp", "surfaces"]]
            Plots requested. Strings can be all or some of "sld",
            "vfp", "surfaces".

        Returns
        -------
        list[AxesIndex]
        """
        # all possible plot types
        plot_types_map = {pt.value: pt for pt in PlotType}
        # validate
        if not all([st in plot_types_map.keys() for st in requested_plots]):
            raise ValueError(
                'Expected "sld", "vfp" or "surfaces" in requested plots. Got'
                f" {requested_plots}."
            )
        # filter the PlotTypes requested.
        requested_plot_types = [
            plot_types_map[plot_name] for plot_name in requested_plots
        ]
        req_plots_and_axis: list[Self] = []
        for pt in requested_plot_types:
            axis = cls(
                requested_plot_types.index(pt)
            )  # create AxesEnum from PlotType.
            axis.plot_type = pt  # set plot_type property to a PlotType
            req_plots_and_axis.append(axis)
        return req_plots_and_axis


def surfaces_for_display(
    vfp: V,
    points: int,
    rng: np.random.Generator,
    align_at_interface: int = 0,
) -> np.typing.NDArray[np.float64]:
    """Produce 2D array of RVs to describe each interface.

    The number of random variates is controlled by ``points``.
    Used to create a graphical representation of the modelled interfaces.

    Parameters
    ----------
    vfp : VFP | refnxVFP | refl1dVFP
        Instantiated concrete class of ``BaseVFP`` from which to plot the
        interfacial model.
    points : int
        Number of points to simulate across the surfaces.
    rng : np.random.Generator
        An initialised pseudo random number generator.
    align_at_interface : int
        Specifies which interface defines z = 0.

    Returns
    -------
    np.typing.NDArray[np.float64]
        2d array of shape = (Nlayers - 1, points)
    """
    if np.abs(align_at_interface) >= len(vfp.vfp_attrs.tup_thicks):
        raise ValueError("align_at_interface must be an index of the layers.")
    interf_loc = np.cumsum(vfp.vfp_attrs.tup_thicks)
    offset = interf_loc[align_at_interface]
    interf_loc = (
        -(interf_loc - offset)
        if vfp.vfp_attrs.orientation == "back"
        else interf_loc - offset
    )
    roughs = vfp.vfp_attrs.tup_roughs
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
    idx_where_conformal = (vfp.vfp_attrs.conformal == 1).nonzero()[0]
    # insert the conformal interfaces.
    if num_conform > 0:
        for i in idx_where_conformal:
            interf_arr[i] = (
                np.max(interf_arr[:i].T, axis=1) + vfp.vfp_attrs.tup_thicks[i]
            )
    return interf_arr


def model_plot(  # noqa: PLR0913
    vfp: V,
    plots_required: list[Literal["sld", "vfp", "surfaces"]],
    posterior_samples: dict[str, np.typing.NDArray[np.float64]] | None,
    align_at_interface: int,
    fig: Figure | None,
    sld_plot_kwargs: SldPlotKwargType | None,
    vfp_plot_kwargs: VfpPlotKwargType | None,
    surface_plot_kwargs: SurfacePlotKwargType | None,
) -> tuple[Figure, list[Axes]]:
    """Visualise the vfp model.

    See ``vfp.basevfp.plot`` for extended details.

    Parameters
    ----------
    vfp : VFP | refnxVFP | refl1dVFP
        Instantiated concrete class of ``BaseVFP`` from which to plot the
        interfacial model.
    plots_required : list[Literal["sld", "vfp", "surfaces"]]
        A list of plots required. Possible acceptable string values are
        "sld", "vfp", "surfaces". The order of the strings in the list
        will affect the order of the plot. Duplicates will be ignored.
    posterior_samples : dict[str, np.ndarray] | None
        Samples from the posterior to plot in the "sld" and "vfp" plots.
        The keys should match the names of varying parameters in the vfp.
        Array values should be 1D of parameter values.
        If None, no posterior samples will be plotted.
    align_at_interface : int
        Specifies which interface defines z = 0.
    fig : Figure | None
        If supplied, plots defined by ``plots_required`` will be plotted on
        ``fig``. If None, a new Figure will be created.
    sld_plot_kwargs : SldPlotKwargType | None
        Kwargs to be passed to ``PlotType._plot_sld``.
    vfp_plot_kwargs : VfpPlotKwargType | None
        Kwargs to be passed to ``PlotType._plot_vfp``.
    surface_plot_kwargs : SurfacePlotKwargType | None
        Kwargs to be passed to ``PlotType._plot_surfaces``.

    Returns
    -------
    tuple[Figure, list[Axes]]
        Figure and axes objects.
    """
    if np.abs(align_at_interface) >= len(vfp.vfp_attrs.tup_thicks):
        raise ValueError("align_at_interface must be an index of the layers.")
    # get axes index for required plots.
    if fig is not None:
        plots_required = [
            plot for plot in plots_required if plot != "surfaces"
        ]
    axes_enum = AxesIndex.from_requested_plots_list(
        requested_plots=plots_required
    )
    # get a copy of current varying_pars, reference after plotting posterior.
    # varying_parameters is not implemented for VFP.
    try:
        original_ps = copy.deepcopy(vfp.varying_parameters)
    except NotImplementedError:
        original_ps = None
    all_plot_kwargs = {
        PlotType.SLD_PLOT: sld_plot_kwargs,
        PlotType.VFP_PLOT: vfp_plot_kwargs,
        PlotType.SURFACES_PLOT: surface_plot_kwargs,
    }
    # setup fig & axes.
    if fig is None:
        fig, _ = plt.subplots(
            nrows=len(plots_required),
            ncols=1,
            sharex=True,
            figsize=(8, 3 * len(plots_required)),
        )
        # get ax this way so that its a flat list for 1 or multiple axes.
        # sorted by the vertical position of the axis in the plot (top to
        # bottom).
        ax: list[Axes] = sorted(
            fig.axes, key=lambda ax: ax.get_subplotspec().rowspan.start
        )
    else:
        # if fig has axes, we have to assume the subplots have original titles
        # won't over plot surfaces.
        original_plot_order = [
            axis.get_title()
            for axis in fig.axes
            if axis.get_title() not in ("surfaces", "")
        ]
        if not plots_required == original_plot_order:
            raise ValueError(
                "plots_required should follow the order of plots_required in"
                f" the original figure: {original_plot_order}."
            )
        ax = [a for a in fig.axes if a.get_title() not in ("surfaces", "")]

    # plot posterior samples:
    if posterior_samples is not None and original_ps is not None:
        # get length of each set of posterior samples.
        p_samps_lens = set([len(val) for val in posterior_samples.values()])
        # check they are the same length.
        if len(p_samps_lens) != 1:
            raise ValueError("Posterior samples are of different lengths.")

    # plot main profiles.
    if original_ps is not None:
        vfp.varying_parameters = original_ps  # set to original values.

    plot_fn_args_map = {
        PlotType.SLD_PLOT: (
            vfp,
            align_at_interface,
            posterior_samples,
        ),
        PlotType.VFP_PLOT: (vfp, align_at_interface, posterior_samples),
        PlotType.SURFACES_PLOT: (vfp, align_at_interface),
    }

    for axis in axes_enum:
        plot_args = plot_fn_args_map[axis.plot_type]
        axis.plot_type.plot(ax[axis], *plot_args, **all_plot_kwargs)

    ax[-1].set_xlabel(r"Distance over Interface / $\mathrm{\AA{}}$")
    return fig, ax


def _gen_sld_profile(
    vfp: V, z: np.typing.NDArray[np.float64]
) -> tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
    """Calculate sld profiles (nuclear, magnetic and imaginary) from the VFP.

    The purpose is to create the step-like affect to reconstruct the sld
    profile modelled. To do this, we reconstruct from ``vfp.dz``

    Parameters
    ----------
    vfp : VFP | refnxVFP | refl1dVFP
        Instantiated concrete class of ``BaseVFP`` from which to plot the
        interfacial model.
    z : np.typing.NDArray[np.float64]
        zeds from ``vfp.calc_z_and_slds``.

    Returns
    -------
    tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]
        Contains the z distance (first index in tuple) over the interface and
        a 2D array of slds in order of sldn, sldi, sldm.
    """
    vfp.process_model()
    # derive zeds and average slds from dzs and slds.
    all_slds = vfp.get_slds()
    mid_slds = np.vstack(
        [0.5 * np.diff(slds) + slds[:-1] for slds in all_slds]
    )
    av_slds = np.vstack([np.ones_like(row_slds) for row_slds in mid_slds])
    av_slds = av_slds * mid_slds
    # get z and dzs same orientation as front.
    z = -z if vfp.vfp_attrs.orientation == "back" else z
    dzs = (
        vfp.vfp_attrs.dz[::-1]
        if vfp.vfp_attrs.orientation == "back"
        else vfp.vfp_attrs.dz
    )
    reconstruc_zeds = np.ones(shape=(dzs.size + 1)) * z[0]
    reconstruc_zeds[1:] += np.cumsum(dzs)
    multiplier = 1
    zed_step_insert = reconstruc_zeds[1:] - (
        multiplier * float(vfp.vfp_attrs.max_delta_z) / 20
    )
    zed_step = np.sort(np.concatenate([reconstruc_zeds, zed_step_insert]))
    all_slds = np.ones_like(zed_step, dtype=float)[:, None] * av_slds[:, 0]
    # first value needs a difference of 0 as we start at this point.
    delta_all_slds = np.hstack(
        (np.zeros(shape=(3, 1)), (av_slds[:, 1:] - av_slds[:, :-1]))
    )

    # accumulate the sld of each step.
    # with scale = 0, this gives a 1 or 0 if x >= or < loc.
    for i in range(av_slds.shape[1] - 1):
        all_slds += (
            heaviside_step(zed_step, loc=reconstruc_zeds[i])[:, None]
            * delta_all_slds[:, i]
        )
    zed_step = -zed_step if vfp.vfp_attrs.orientation == "back" else zed_step
    return zed_step, all_slds


def _tot_sld(
    all_slds: np.typing.NDArray[np.float64], ss: str, overlay_counter: int
) -> tuple[np.typing.NDArray[np.float64], str]:
    """Get sld profile for plotting given spin state.

    The returned value is the rows of sldn (possibly) +/- sldm.

    Parameters
    ----------
    all_slds : np.typing.NDArray[np.float64]
        2D array containing sldn, sldi, sldm.
    ss : str
        Spin state for conditioning which sld is plotted.
    overlay_counter: int
        The number of overplots.

    Returns
    -------
    tuple[np.typing.NDArray[np.float64], str]
        sld for plotting in first index and label for sld plot in second.
    """
    if ss == "none":
        tot_sld = all_slds[:, 0]
        sld_label = rf"$\mathrm{{SLD}}_{{\mathrm{{n}}, {overlay_counter}}}$"
    elif ss == "down":
        tot_sld = all_slds[:, 0] - all_slds[:, 2]
        sld_label = (
            rf"$\mathrm{{SLD}}_{{\mathrm{{n}}, {overlay_counter}}} - $"
            rf"$\mathrm{{SLD}}_{{\mathrm{{m}}, {overlay_counter}}}$"
        )
    elif ss == "up":
        tot_sld = all_slds[:, 0] + all_slds[:, 2]
        sld_label = (
            rf"$\mathrm{{SLD}}_{{\mathrm{{n}}, {overlay_counter}}} + $"
            rf"$\mathrm{{SLD}}_{{\mathrm{{m}}, {overlay_counter}}}$"
        )
    return tot_sld, sld_label
