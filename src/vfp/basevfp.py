"""Methods for vfp shared by all types."""

from __future__ import annotations

# standard
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Generic, Literal, Self, TypeVar, cast

# third party
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# this package
from vfp.calc import (
    arr_to_tuple,
    calc_demag_array,
    calc_dzs,
    calc_indices,
    calc_vfp,
    calc_zeds,
    integrate_vfp,
    reduce_vfp_and_magcomp,
    transform_indices,
)
from vfp.plotting import model_plot
from vfp.vfp_typing import (
    ParameterLike,
    SldConstraintType,
    SldPlotKwargType,
    SurfacePlotKwargType,
    VFPAttrType,
    VfpPlotKwargType,
    flatten_composite_type_alias,
)

P = TypeVar("P", bound=ParameterLike)
"""Generic type of parameters within :class:`vfp.vfp_typing.ParameterLike`."""
V = TypeVar("V", bound="BaseVFP")
"""Generic subclass of :class:`BaseVFP`."""


@dataclass
class VFPAttributes:
    """Holds parameters given to concrete :class:`BaseVFP`s.

    Used for internal vfp calculations, and not intended to be directly set.
    Caching here is used to avoid having to recalculate particular properties
    during a call to :meth:`BaseVFP.process_model`.
    """

    nslds: np.typing.NDArray[np.float64]
    thicknesses: np.typing.NDArray[np.float64]
    roughnesses: np.typing.NDArray[np.float64]
    islds: np.typing.NDArray[np.float64]
    mslds: np.typing.NDArray[np.float64]
    spin_state: Literal["none", "up", "down"]
    orientation: Literal["front", "back"]
    demaglocs: np.typing.NDArray[np.float64]
    demagwidths: np.typing.NDArray[np.float64]
    sld_constraint: None | SldConstraintType
    max_delta_z: float
    conformal: np.typing.NDArray[np.float64]
    name: str

    _zeds_dependents: dict[str, tuple[float, ...]] = field(
        init=False, default_factory=dict[str, tuple], repr=False
    )
    """thicknesses and roughnesses when :func:`vfp.calc.calc_zeds` last
    called."""
    _vfp_dependents: dict[str, tuple[float, ...]] = field(
        init=False, default_factory=dict[str, tuple], repr=False
    )
    """thicknesses and roughnesses when :func:`vfp.calc.calc_vfp` last
    called."""
    _indices_dependents: dict[str, tuple[float, ...]] = field(
        init=False, default_factory=dict[str, tuple], repr=False
    )
    """Thickness, roughnesses, demag_locs, demag_widths and mslds when
    :func:`vfp.calc.calc_indices` last called."""
    _demag_arr_dependents: dict[str, tuple[float, ...]] = field(
        init=False, default_factory=dict[str, tuple[float, ...]], repr=False
    )
    """Thickness, roughnesses, demag_locs, demag_widths and mslds when
    :func:`vfp.calc.calc_demag_array` last called."""
    _cached_zeds: tuple[float, ...] = field(
        init=False, default_factory=tuple[float, ...], repr=False
    )
    _cached_vfp: tuple[tuple[float, ...], ...] = field(
        init=False, default_factory=tuple[tuple[float, ...], ...], repr=False
    )
    _cached_indices: tuple[int, ...] = field(
        init=False, default_factory=tuple[int, ...], repr=False
    )
    _cached_demag_arr: tuple[tuple[float, ...], ...] = field(
        init=False, default_factory=tuple[tuple[float, ...], ...], repr=False
    )

    @property
    def tup_thicks(self) -> tuple[float, ...]:
        """Tuple variant of :attr:`VFPAttributes.thicknesses` for caching."""
        return tuple(self.thicknesses.astype(float))

    @property
    def tup_mslds(self) -> tuple[float, ...]:
        """Tuple variant of :attr:`VFPAttributes.mslds` for caching."""
        return tuple(self.mslds.astype(float))

    @property
    def tup_demag_locs(self) -> tuple[float, ...]:
        """Tuple variant of :attr:`VFPAttributes.demaglocs` for caching."""
        return tuple(self.demaglocs.astype(float))

    @property
    def tup_demag_widths(self) -> tuple[float, ...]:
        """Tuple variant of :attr:`VFPAttributes.demagwidths` for caching."""
        return tuple(self.demagwidths.astype(float))

    @property
    def tup_roughs(self) -> tuple[float, ...]:
        """Tuple variant of :attr:`VFPAttributes.roughnesses` for caching.

        If a value in roughnesses is None, we set it to a dummy value of 1.
        This value is completely ignored during the vfp calculations, but
        is required for consistent array sizes.
        """
        rs = tuple(
            float(par) if par is not None else 1 for par in self.roughnesses
        )
        return rs

    @property
    def zeds(self) -> tuple[float, ...]:
        """Distance coordinate over total interface.

        If already calculated for combination of
        :attr:`VFPAttributes.tup_thicks` and
        :attr:`VFPAttributes.tup_roughs` will use cached value.
        """
        current_deps = (self.tup_thicks, self.tup_roughs)
        if current_deps == (
            self._zeds_dependents.get("tup_thicks"),
            self._zeds_dependents.get("tup_roughs"),
        ):
            return self._cached_zeds
        self._cached_zeds = calc_zeds(
            self.tup_roughs,
            self.tup_thicks,
            self.max_delta_z,
        )
        (
            self._zeds_dependents["tup_thicks"],
            self._zeds_dependents["tup_roughs"],
        ) = current_deps

        return self._cached_zeds

    @property
    def dz(self) -> np.typing.NDArray[np.float64]:
        """The thickness of each microslab.

        When ``orientation == back``, microslabs will have same thicknesses
        as front, just in reverse order.

        This isn't cached as this is only ever called once per call to
        :class:`BaseVFP.process_model`.
        """
        zds = self.zeds  # avoid calling the property more than once.
        dzs = calc_dzs(zds[0], zds[-1], len(zds), self.indices)
        # when orientation = back, slabs will have same thickness as front,
        # just in reverse order
        if self.orientation == "back":
            dzs = dzs[::-1]
        return dzs

    @property
    def vfp(self) -> tuple[tuple[float, ...], ...]:
        """Non-reduced layer volume fraction profile."""
        current_deps = (self.tup_thicks, self.tup_roughs)
        if current_deps == (
            self._vfp_dependents.get("tup_thicks"),
            self._vfp_dependents.get("tup_roughs"),
        ):
            return self._cached_vfp
        self._cached_vfp = calc_vfp(
            self.tup_roughs,
            self.tup_thicks,
            self.zeds,
            tuple(self.conformal),
        )
        (
            self._vfp_dependents["tup_thicks"],
            self._vfp_dependents["tup_roughs"],
        ) = current_deps
        return self._cached_vfp

    @property
    def demag_arr(self) -> tuple[tuple[float, ...], ...]:
        """Non-reduced magnetic demagnetisation of each layer."""
        current_deps = (
            self.tup_thicks,
            self.tup_roughs,
            self.tup_demag_locs,
            self.tup_demag_widths,
            self.tup_mslds,
        )
        if current_deps == (
            self._demag_arr_dependents.get("tup_thicks"),
            self._demag_arr_dependents.get("tup_roughs"),
            self._demag_arr_dependents.get("tup_demag_locs"),
            self._demag_arr_dependents.get("tup_demag_widths"),
            self._demag_arr_dependents.get("tup_mslds"),
        ):
            return self._cached_demag_arr
        self._cached_demag_arr = calc_demag_array(
            self.tup_demag_locs,
            self.tup_demag_widths,
            self.tup_mslds,
            self.zeds,
        )
        (
            self._demag_arr_dependents["tup_thicks"],
            self._demag_arr_dependents["tup_roughs"],
            self._demag_arr_dependents["tup_demag_locs"],
            self._demag_arr_dependents["tup_demag_widths"],
            self._demag_arr_dependents["tup_mslds"],
        ) = current_deps
        return self._cached_demag_arr

    @property
    def indices(self) -> tuple[int, ...]:
        """Indices where vfp is ~ invariant with next neighbouring point."""
        current_deps = (
            self.tup_thicks,
            self.tup_roughs,
            self.tup_demag_locs,
            self.tup_demag_widths,
            self.tup_mslds,
        )
        if current_deps == (
            self._indices_dependents.get("tup_thicks"),
            self._indices_dependents.get("tup_roughs"),
            self._indices_dependents.get("tup_demag_locs"),
            self._indices_dependents.get("tup_demag_widths"),
            self._indices_dependents.get("tup_mslds"),
        ):
            return self._cached_indices
        self._cached_indices = calc_indices(self.vfp, self.demag_arr)
        (
            self._indices_dependents["tup_thicks"],
            self._indices_dependents["tup_roughs"],
            self._indices_dependents["tup_demag_locs"],
            self._indices_dependents["tup_demag_widths"],
            self._indices_dependents["tup_mslds"],
        ) = current_deps
        return self._cached_indices


class BaseVFP(ABC, Generic[P]):
    """Base class of vfp classes in vfp.py.

    :func:`BaseVFP.process_model` is the main function.
    """

    def __repr__(self) -> str:
        """Get a string discription of the vfp.

        Currently not called by :class:`vfp.vfp.refnxVFP` or
        :class:`vfp.vfp.refl1dVFP`.

        Returns
        -------
        str
            Returns a printable representation of the VFP,
            describing VFP type, parameters and values.
        """
        s = (
            f"name: {self.vfp_attrs.name} \n"
            f"thicks: {self.vfp_attrs.thicknesses} \n"
            f"roughs: {self.vfp_attrs.roughnesses} \n"
            f"nslds: {self.vfp_attrs.nslds} \n"
            f"mslds: {self.vfp_attrs.mslds} \n"
            f"islds: {self.vfp_attrs.islds} \n"
            f"demag locations: {self.vfp_attrs.demaglocs} \n"
            f"demag widths: {self.vfp_attrs.demagwidths} \n"
            f"conformal: {self.vfp_attrs.conformal} \n"
        )
        return s

    def process_model(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the thickness and sld of microslices.

        Calculates the length of the vfp, the thicknesses of each microslice
        and calculates the sld of each microslice. Returns the coherent and
        imaginary sld values for each microslice and the thickness of each
        microslice given orientation of sample.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple of three arrays. In order, the arrays are 1)
            coherent (nsld +/- msld) microslices, 2) isld microslices, 3)
            thickness of each microslice.
            Each array has shape = zeds.size - self.indices
        """
        all_slds = self.get_slds()
        # total the nuclear and magnetic SLDs on given contrast.
        if self.vfp_attrs.spin_state == "none":
            coh_sld = all_slds[0]
        elif self.vfp_attrs.spin_state == "down":
            coh_sld = all_slds[0] - all_slds[2]
        elif self.vfp_attrs.spin_state == "up":
            coh_sld = all_slds[0] + all_slds[2]

        # and imaginary
        i_sld = all_slds[1]

        # get the average between each coherent and imaginary sld value.
        average_slds, average_islds = (
            0.5 * np.diff(slds) + slds[:-1] for slds in [coh_sld, i_sld]
        )

        # init arrays for final SLDs.
        return_slds, return_islds = [
            np.ones(slds.size) for slds in [average_slds, average_islds]
        ]

        if self.vfp_attrs.orientation == "front":
            # fill all but last with average SLDs.
            return_slds = return_slds * average_slds
            return_islds = return_islds * average_islds

        elif self.vfp_attrs.orientation == "back":
            # do the same but backwards for back orientations.
            return_slds = return_slds * average_slds[::-1]
            return_islds = return_islds * average_islds[::-1]

        return return_slds, return_islds, self.vfp_attrs.dz

    def get_slds(self, reduced: bool = True) -> np.ndarray:
        """Calculate slds via generation of volume fraction profile.

        Initially, the vol fraction profile is calculated, then it is reduced
        via :meth:`BaseVFP.init_demag`. Slds are calculated and then summed to
        give coherent slds (nuclear or nuclear +/- magnetic dependent on
        :attr:`BaseVFP.spin_state`) and imaginary slds.

        Parameters
        ----------
        reduced : bool
            If True/False, calculates the reduced/full SLD profiles

        Returns
        -------
        np.ndarray
            Three sld contributions across three rows as function of
            :attr:`VFPAttributes.zeds`. Coherent sld, imaginary sld, magnetic
            sld.
        """
        vfp = np.asarray(self.vfp_attrs.vfp)
        demag_arr = np.asarray(self.vfp_attrs.demag_arr)
        mag_comp = vfp * demag_arr
        if reduced:
            vfp, mag_comp = reduce_vfp_and_magcomp(
                vfp, mag_comp, np.asarray(self.vfp_attrs.indices)
            )
        all_slds = self.calc_slds(vfp, mag_comp)
        return all_slds

    def calc_slds(
        self,
        p_vfp: np.ndarray,
        demag_vfp: np.ndarray,
    ) -> np.ndarray:
        """Calculate coherent and imaginary slds.

        Slds are nuclear, imaginary and magnetic. Can be calculated with
        reduced or full VFP.

        Parameters
        ----------
        p_vfp : np.ndarray
            Possibly reduced vfp, else full vfp.
        demag_vfp : np.ndarray
            Possibly reduced demag_vfp.

        Returns
        -------
        np.ndarray
            Three sld contributions across three rows as function of
            :attr:`VFPAttributes.zeds`. Nuclear sld, imaginary sld, magnetic
            sld.
        """
        # possibly update nslds depending on user supplied constraint class.
        if self.vfp_attrs.sld_constraint is not None:
            layer_indices = self.vfp_attrs.sld_constraint.layer_choices()
            integrals = integrate_vfp(
                self.vfp_attrs.zeds,
                self.vfp_attrs.indices,
                arr_to_tuple(p_vfp),
                tuple(layer_indices),
            )
            # user defines a class with a callable, this should return a tuple
            # of two lists or tuples. The first is the indices at which SLD
            # values will be modified, and the second is the sld values to
            # change to.
            sld_const_res = self.vfp_attrs.sld_constraint(integrals)
            # check user has defined the return to be of the right type.
            if not all([isinstance(x, tuple | list) for x in sld_const_res]):
                raise TypeError(
                    "Expected the __call__ function of sld_constraint to"
                    " return a tuple of two tuple or lists. Got"
                    f" {type(sld_const_res[0])} and {type(sld_const_res[1])}"
                )
            layer_idxs, slds = sld_const_res
            # now check each value in both tuple / lists have right type.
            if not all([isinstance(idx, int) for idx in layer_idxs]):
                raise TypeError(
                    "Expected the first return value in sld_constraint"
                    " __call__ to contain only int. Got these types:"
                    f" {set([type(idx) for idx in layer_idxs])}."
                )
            # can't use ParameterLike in isinstance. The __value__ gives the
            # union of all types which should be what we want. Not pretty.
            if not all(
                [isinstance(sld, ParameterLike.__value__) for sld in slds]
            ):
                raise TypeError(
                    "Expected the second return value in sld_constraint"
                    " __call__ to contain only ParameterLike. Got these:"
                    f" {set([type(sld) for sld in slds])}."
                )
            for layer_idx, sld in zip(layer_idxs, slds, strict=True):
                self.vfp_attrs.nslds[layer_idx] = sld  # update

        # get float values from the Parameters in the attrs arrays.
        sld_values = [
            sld_pars.astype(float)
            for sld_pars in [
                self.vfp_attrs.nslds,
                self.vfp_attrs.mslds,
                self.vfp_attrs.islds,
            ]
        ]

        # calc nuclear_slds from red_vfps:
        nuc_and_i_slds = [
            p_vfp.T * sld_val for sld_val in [sld_values[0], sld_values[2]]
        ]

        # calc magnetic_slds
        sldm_layers = demag_vfp.T * sld_values[1]

        # sum slds over all layers.
        slds_over_z = [
            np.sum(arr, axis=1) for arr in [*nuc_and_i_slds, sldm_layers]
        ]

        # row 0 = nsld, row 1 = isld, row 2 = msld
        sum_slds = np.vstack(slds_over_z)

        return sum_slds

    def vfs_for_display(
        self, reduced: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get volume fraction profile for plotting.

        Parameters
        ----------
        reduced: bool, optional
            Flag to return reduced vfp. If False, get non-reduced vfp.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            First array is vfp (reduced or full). Second array is magnetic vfp
            (vfp * demag_arr) applied (reduced or full).
        """
        vfp = np.asarray(self.vfp_attrs.vfp)
        demag_arr = np.asarray(self.vfp_attrs.demag_arr)
        magcomp = vfp * demag_arr

        if reduced:
            vfp, magcomp = reduce_vfp_and_magcomp(
                vfp, magcomp, np.asarray(self.vfp_attrs.indices)
            )

        if self.vfp_attrs.orientation == "back":
            vfp = vfp[::-1]  # reverse order.
            magcomp = magcomp[::-1]

        return vfp, magcomp

    def z_and_sld(
        self, reduced: bool = True, align_at_interface: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get z and sld values from vfp for plotting.

        Returns z values from :attr:`VFPAttributes.zeds` and also returns
        non-microsliced sld values from :meth:`BaseVFP.get_slds` calculated
        from the VFP.

        Parameters
        ----------
        reduced : bool
            If False/True, will return full/reduced zs and slds.
        align_at_interface : int, optional
            Which interface index to set z = 0. Defaults to
            first interface.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            First array is z distance from fronting interface
            either reduced or full. Second array is slds (2d),
            coherent, imaginary, magnetic across columns.
            Either reduced or full.
        """
        offset = np.cumsum(self.vfp_attrs.tup_thicks)[align_at_interface]
        if np.abs(align_at_interface) >= len(self.vfp_attrs.thicknesses):
            raise ValueError(
                "align_at_interface must be an index of the layers."
            )
        z = np.array(self.vfp_attrs.zeds) - offset
        z = -z if self.vfp_attrs.orientation == "back" else z
        slds = self.get_slds(reduced=reduced)
        # conditionally remove z at indices.
        if reduced:
            delete_idx = transform_indices(self.vfp_attrs.indices)
            z = np.delete(z, delete_idx)
        return z, slds.T

    def sld_offset(self) -> float:
        """Get float to add to refnx or refl1d sld profile z coordinate.

        Returns
        -------
        float
            Offset to add to z values of an sld_profile.

        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from vfp import refnxVFP
        >>> from refnx.reflect import ReflectModel, SLD
        >>> thicknesses = (0, 20, 30)
        >>> roughnesses = (2, 4, 6)
        >>> slds = (2.07, 3.47, 0.21, 6.37)
        >>> refnx_vfp = refnxVFP(slds, thicknesses, roughnesses)
        >>> struc = SLD(2.07, name='Si') | refnx_vfp | SLD(6.37, name='D2O')
        >>> model = ReflectModel(struc)
        >>> z, sld = model.structure.sld_profile(max_delta_z=0.1)
        >>> plt.plot(z+refnx_vfp.sld_offset(), sld)
        >>> plt.show()
        """
        if self.vfp_attrs.orientation == "front":
            sldprof_offset_nr = -5 - (4 * self.vfp_attrs.tup_roughs[0])
            # round down like zstart
            sldprof_offset = np.floor(
                sldprof_offset_nr * (1 / self.vfp_attrs.max_delta_z)
            ) / (1 / self.vfp_attrs.max_delta_z)

        elif self.vfp_attrs.orientation == "back":
            zend_of_vfprofile_nr = (
                np.max(
                    np.sum(self.vfp_attrs.tup_thicks)
                    + 4 * np.array(self.vfp_attrs.tup_roughs)
                )
                + 5
            )
            # round up like zend
            zend_of_vfprofile = np.ceil(
                zend_of_vfprofile_nr * (1 / self.vfp_attrs.max_delta_z)
            ) / (1 / self.vfp_attrs.max_delta_z)

            # sld_profile zend defined by -5 + last slab location + 4 *
            # backing roughness. zend_of_vfprofile replicates the 4 *
            # backing roughness part. Then 5 + last microslice thickness
            # covers the -5 + last slab location part.
            zend_front = self.vfp_attrs.dz[-1] + zend_of_vfprofile
            sldprof_offset = -(zend_front - np.sum(self.vfp_attrs.tup_thicks))

        return sldprof_offset

    def plot(  # noqa : PLR0913
        self,
        plots_required: list[Literal["sld", "vfp", "surfaces"]] | None = None,
        posterior_samples: dict[str, np.ndarray] | None = None,
        align_at_interface: int | None = None,
        fig: Figure | None = None,
        sld_plot_kwargs: SldPlotKwargType | None = None,
        vfp_plot_kwargs: VfpPlotKwargType | None = None,
        surface_plot_kwargs: SurfacePlotKwargType | None = None,
    ) -> tuple[Figure, list[Axes]]:
        """Make a one to three axis figure to visualise the VFP model.

        By default the order of the plots are:
            Top plot = nsld / msld / isld
            Middle plot = volume fraction profiles
            Bottom plot = surface profiles
        This can be altered by specifying a different order in
        ``plots_required``.

        Parameters
        ----------
        plots_required : list[Literal["sld", "vfp", "surfaces"]] | None, opt
            A list of plots required. Possible acceptable string values are
            ``sld``, ``vfp``, ``surfaces``. The order of the strings in the
            list will affect the order of the plot. Duplicates are ignored.
        posterior_samples : dict[str, np.ndarray] | None, optional
            Samples from the posterior to plot in the "sld" and "vfp" plots.
            The keys should match the names of varying parameters in the vfp.
            Array values should be 1D of parameter values.
            By default is None.
        align_at_interface : int | None, optional
            Specifies which interface is defined as z = 0 by index.
            If not specified, defaults to first interface.
        fig : Figure | None, optional.
            If supplied, plots will be plotted on ``fig``.
            By default a new Figure will be created.
        sld_plot_kwargs : SldPlotKwargType | None, optional
            Kwargs to be passed to :meth:`vfp.plotting.PlotType._plot_sld`.
            By default None.
        vfp_plot_kwargs : VfpPlotKwargType | None, optional
            Kwargs to be passed to :meth:`vfp.plotting.PlotType._plot_vfp`.
            By default None.
        surface_plot_kwargs : SurfacePlotKwargType | None, optional
            Kwargs to be passed to
            :meth:`vfp.plotting.PlotType._plot_surfaces`. By default None.

        Returns
        -------
        tuple[Figure, list[Axes]]
        """
        # run check on unique vals in plots_required
        possible_plots = ["sld", "vfp", "surfaces"]
        if isinstance(plots_required, list):
            # remove duplicates, but preserve order.
            plots_required = list(dict.fromkeys(plots_required))
            if not all([ptype in possible_plots for ptype in plots_required]):
                raise ValueError(
                    'Check plots_required only contains "sld", "vfp",'
                    ' "surfaces".'
                )
        elif plots_required is None:
            plots_required = ["sld", "vfp", "surfaces"]
        else:
            raise TypeError(
                f"plots_required must be a list, got {type(plots_required)}."
            )

        align_at_interface = (
            0 if align_at_interface is None else align_at_interface
        )
        if not isinstance(align_at_interface, int):
            raise TypeError("align_at must be an integer")

        if np.abs(align_at_interface) >= len(self.vfp_attrs.thicknesses):
            raise ValueError(
                "align_at_interface must be an index of the layers."
            )

        fig, ax = model_plot(
            vfp=self,
            plots_required=plots_required,
            posterior_samples=posterior_samples,
            align_at_interface=align_at_interface,
            fig=fig,
            sld_plot_kwargs=sld_plot_kwargs,
            vfp_plot_kwargs=vfp_plot_kwargs,
            surface_plot_kwargs=surface_plot_kwargs,
        )

        return fig, ax

    def _init_vfp_attrs(
        self,
        arr_attrs: Sequence[Sequence[P | None]],
        other_attrs: tuple[
            list[int],
            Literal["front", "back"],
            Literal["none", "up", "down"],
            SldConstraintType | None,
            float,
        ],
        name: str,
    ) -> VFPAttributes:
        """Init :class:`VFPAttributes` with VFP input parameters."""
        (
            thicknesses,
            roughnesses,
            nslds,
            islds,
            mslds,
            demaglocs,
            demagwidths,
        ) = list(map(np.asarray, arr_attrs))
        (conformal, orientation, spin_state, sld_constraint, max_delta_z) = (
            other_attrs
        )
        attrs = VFPAttributes(
            nslds=nslds,
            thicknesses=thicknesses,
            roughnesses=roughnesses,
            islds=islds,
            mslds=mslds,
            spin_state=spin_state,
            orientation=orientation,
            demaglocs=demaglocs,
            demagwidths=demagwidths,
            sld_constraint=sld_constraint,
            max_delta_z=max_delta_z,
            conformal=np.asarray(conformal),
            name=name,
        )
        return attrs

    @property
    @abstractmethod
    def vfp_attrs(self) -> VFPAttributes:
        """Get :class:`VFPAttributes` attached to this vfp."""
        raise NotImplementedError

    @abstractmethod
    def set_parameter_prior(self, *args, **kwargs) -> None:
        """Set priors on fitting parameters."""
        raise NotImplementedError

    @abstractmethod
    def transform(
        self, wanted_vfp: Literal["VFP", "refnxVFP", "refl1dVFP"]
    ) -> V:
        """Transform a VFP of one type to another."""
        raise NotImplementedError

    @property
    @abstractmethod
    def varying_parameters(self) -> dict[str, ParameterLike]:
        """Get all varying parameters in vfp."""
        raise NotImplementedError

    @varying_parameters.setter
    @abstractmethod
    def varying_parameters(
        self, values_dict: dict[str, ParameterLike]
    ) -> None:
        """Set the values of the varying parameters."""
        raise NotImplementedError

    @abstractmethod
    def _createparam(
        self, params: Sequence[ParameterLike | None], nameid: str
    ) -> Sequence[P | None]:
        """Create parameters of specific type."""
        raise NotImplementedError

    @classmethod
    def from_transform(cls, vfp_attrs: VFPAttrType) -> Self:
        """Transform a dictionary of vfp attributes to a ``VFP``.

        Parameters
        ----------
        vfp_attrs : VfpAttrType
            Key names are parameter names, while dict values are
            values of each parameter.
        """
        attr_dict = {}
        # transform all arrays to lists to be compatible with init.
        for key, seq in vfp_attrs.items():
            if isinstance(seq, np.ndarray):
                seq = cast(np.ndarray, seq)
                attr_dict[key] = seq.astype(float).tolist()
            else:
                attr_dict[key] = seq
        # alter the roughness entry.
        attr_dict["roughnesses"] = [
            val if ~np.isnan(val) else "conformal" for val in attr_dict
        ]
        del attr_dict["conformal"]
        del attr_dict["name"]
        vfp = cls(**attr_dict)
        return vfp


def _check_init_input(  # noqa : PLR0912, PLR0913
    thicknesses: Sequence[ParameterLike],
    roughnesses: Sequence[ParameterLike | Literal["conformal"]],
    nslds: Sequence[ParameterLike],
    islds: Sequence[ParameterLike] | None,
    mslds: Sequence[ParameterLike] | None,
    demaglocs: Sequence[ParameterLike] | None,
    demagwidths: Sequence[ParameterLike] | None,
    spin_state: Literal["none", "up", "down"],
    max_delta_z: float,
) -> tuple[
    Sequence[ParameterLike | None],
    list[Sequence[ParameterLike]],
    Sequence[ParameterLike],
    Sequence[ParameterLike],
    list[int],
]:
    """Check the input values of a concrete :class:`BaseVFP` subclass."""
    if not demaglocs:  # look for empty lists or None.
        demaglocs: list[ParameterLike] = []

    if not demagwidths:
        demagwidths: list[ParameterLike] = []

    if len(demaglocs) != len(demagwidths):
        raise ValueError(
            """The number of the demagnetisation locations must be
                equal to the number of the demagnetisation widths."""
        )

    if len(demaglocs) % 2 > 0:
        raise ValueError(
            "The number of the demagnetisation locations and widths must be 0"
            " or even."
        )

    if len(thicknesses) != len(roughnesses):
        raise ValueError(
            """The number of roughness parameters must match
                the number of thickness parameters."""
        )

    # check roughness value below 0. Use float for compat with bumpsParameter.
    if any(
        [
            float(rough_val) <= 0
            for rough_val in roughnesses
            if not isinstance(rough_val, str)
        ]
    ):
        raise ValueError("Roughness parameters must be > 0 if not str.")

    if len(nslds) != len(thicknesses) + 1:
        raise ValueError(
            """The number of supplied SLD values must be 1 greater
                than the number of thickness parameters."""
        )

    # init a list of where conformal interfaces are:
    conformal: list[int] = []
    for roughness in roughnesses:
        # cannot use a type alias in isinstance so use its value attr
        possible_types = flatten_composite_type_alias(ParameterLike)
        possible_types.add(str)
        if isinstance(roughness, tuple(possible_types)):
            if isinstance(roughness, str) and roughness == "conformal":
                conformal.append(1)

            elif isinstance(roughness, str) and roughness != "conformal":
                raise ValueError(
                    "Any string within the roughness list must read"
                    " 'conformal'."
                )

            else:
                conformal.append(0)

        else:
            raise ValueError(
                "The entries within the roughness list must be a float,"
                " integer, refnx or bumps parameters, or a string =="
                " 'conformal'."
            )

    # can only have conformal roughnesses with more than one interface.
    # therefore the first interface cannot be conformal
    if any(conformal):
        idx_where_first_one = (np.array(conformal) == 1).nonzero()[0][0]

        if not idx_where_first_one > 0:
            raise ValueError(
                "Cannot specify the first interface to be conformal."
            )

    else:
        conformal = [0] * len(thicknesses)

    # where conformal in roughnesses, replace with None
    roughnesses_alt: list[ParameterLike | None] = [
        None if val == "conformal" else val for val in roughnesses
    ]

    # set mslds to a list of zeros if None.
    if mslds is None:
        mslds = [0] * (len(thicknesses) + 1)

    # now check for any non-zero values in list.
    # Use float for compat with bumpsParameter.
    if any([float(msld) > 0 for msld in mslds]):
        if spin_state == "none":
            raise ValueError(
                "If mslds is defined, the spin state passed to the VFP must"
                " be 'up' or 'down'."
            )

    # do same None check for islds
    if islds is None:
        islds = [0] * (len(thicknesses) + 1)

    # check the lengths of SLD arrays are the same.
    all_slds = [nslds, islds, mslds]
    if not all(len(sld) == len(all_slds[0]) for sld in all_slds):
        raise ValueError(
            "The number of supplied nuclear, magnetic and imaginary SLD"
            " values must be the same."
        )

    # Simple warning on max_delta_z being too low.
    if any(
        float(rough) < 2 * max_delta_z
        for rough in roughnesses_alt
        if rough is not None
    ):
        warnings.warn(
            "The microslice thickness is less than twice some of the"
            " roughness parameters. Consider reducing the max_delta_z of the"
            " VFP.",
            stacklevel=2,
        )
    return roughnesses_alt, all_slds, demaglocs, demagwidths, conformal
