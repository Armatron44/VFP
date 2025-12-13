from __future__ import annotations

# standard
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

# third party
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# this package
from vfp.calc import (
    calc_dzs,
    calc_vfp,
    calc_zeds,
    init_demag,
    integrate_vfp,
    transform_indices,
)
from vfp.plotting import model_plot
from vfp.vfp_typing import (
    ParameterLike,
    SLDConstraintType,
    SldPlotKwargType,
    SurfacePlotKwargType,
    VfpPlotKwargType,
)


@dataclass
class VFPAttributes:
    """
    Holds reference to the concrete VFP classes.
    The parameters held here can be updated by fitters and samplers.
    """

    nslds: np.ndarray
    thicknesses: np.ndarray
    roughnesses: np.ndarray
    islds: np.ndarray
    mslds: np.ndarray
    spin_state: Literal["none", "up", "down"]
    orientation: Literal["front", "back"]
    demaglocs: np.ndarray
    demagwidths: np.ndarray
    sld_constraint: None | SLDConstraintType
    max_delta_z: float
    conformal: np.ndarray
    name: str

    @property
    def tup_thicks(self) -> tuple[float, ...]:
        return tuple(self.thicknesses.astype(float))

    @property
    def tup_mslds(self) -> tuple[float, ...]:
        return tuple(self.mslds.astype(float))

    @property
    def tup_demag_locs(self) -> tuple[float, ...]:
        return tuple(self.demaglocs.astype(float))

    @property
    def tup_demag_widths(self) -> tuple[float, ...]:
        return tuple(self.demagwidths.astype(float))

    @property
    def tup_roughs(self) -> tuple[float, ...]:
        rs = tuple(
            float(par) if par is not None else 1 for par in self.roughnesses
        )
        return rs


class BaseVFP(ABC):
    """
    Handles common functions of VFP.

    `process_model` is the main function.
    """

    def __init__(self) -> None:
        # create vfp model.
        self.process_model()

    def __repr__(self) -> str:
        """
        Simple string description of the VFP.

        Currently not called by `refnxVFP` or `refl1dVFP`.

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
        """
        Calculates the thickness and sld of microslices.

        Main function of the `BaseVFP`. Calculates the length of the VFP,
        the thicknesses of each microslice and calculates the sld of each
        microslice. Returns the coherent and imaginary sld values for
        each microslice and the thickness of each microslice given orientation
        of sample.

        Returns
        -------
        np.array
            slds of microslices
            Shape = zeds.size - self.indices
        np.array
            islds of microslices
            Shape = zeds.size - self.indices
        np.array
            microslice thicknesses.
            Shape = zeds.size - self.indices
        """

        # calc z spectrum
        zeds = calc_zeds(
            self.vfp_attrs.tup_roughs,
            self.vfp_attrs.tup_thicks,
            self.vfp_attrs.max_delta_z,
        )

        zstart, zend, points = zeds[0], zeds[-1], zeds.size

        self.zeds = self._arrtotuple(zeds)
        "z space of interface as tuple for caching."

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

        self.dz = calc_dzs(zstart, zend, points, self.indices)
        "The thickness of each microslab"

        # when orientation = back, slabs will have same thickness as front,
        # just in reverse order
        if self.vfp_attrs.orientation == "back":
            self.dz = self.dz[::-1]
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
            # now set the final sld value to those from the micro arrays.
            # return_slds[-1] = coh_sld[-1]
            # return_islds[-1] = i_sld[-1]

        elif self.vfp_attrs.orientation == "back":
            # do the same but backwards for back orientations.
            return_slds = return_slds * average_slds[::-1]
            return_islds = return_islds * average_islds[::-1]
            # now set the final sld value to those from the micro arrays.
            # return_slds[0] = coh_sld[-1]
            # return_islds[0] = i_sld[-1]

        return return_slds, return_islds, self.dz

    def get_slds(self, reduced: bool = True) -> np.ndarray:
        """
        Calculate slds via generation of volume fraction profile.

        Initially, the vol fraction profile is calculated, then it is reduced
        via `self.init_demag`. slds are calculated and then summed to give
        a coherent slds (nuclear or nuclear +/- magnetic dependent on
        `self.spin_state`) and imaginary slds.

        Parameters
        ----------
        reduced : bool
            If True/False, calculates the reduced/full SLD profiles

        Returns
        -------
        np.ndarray
            Three sld contributions across three rows as function of
            `self.zeds`. Coherent sld, imaginary sld, magnetic sld.
        """

        # calculate volume fraction profiles of layers over interface.
        self.vfp = calc_vfp(
            self.vfp_attrs.tup_roughs,
            self.vfp_attrs.tup_thicks,
            self.zeds,
            tuple(self.vfp_attrs.conformal),
        )

        # calculate reduced volume fraction and magnetic profiles.
        red_vfp, red_demag_vfp, idx, demag_arr = init_demag(
            self.vfp_attrs.tup_demag_locs,
            self.vfp_attrs.tup_demag_widths,
            self.vfp_attrs.tup_mslds,
            self.zeds,
            self._arrtotuple(self.vfp),
        )

        self.indices = self._arrtotuple(idx)
        """Tuple version of arr where volume fraction values are
           approximately invariant."""

        # calculate the SLD values across reduced or full vfp:
        if reduced:
            all_slds = self.calc_slds(red_vfp, red_demag_vfp)
        else:
            all_slds = self.calc_slds(self.vfp, self.vfp * demag_arr)

        return all_slds

    def calc_slds(
        self,
        p_vfp: np.ndarray,
        demag_vfp: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates coherent and imaginary slds.

        Slds are nuclear, imaginary and magnetic.
        Can be calculated with reduced or full VFP.

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
            `self.zeds`. Nuclear sld, imaginary sld, magnetic sld.
        """
        # possibly update nslds depending on user supplied constraint class.
        if self.vfp_attrs.sld_constraint is not None:
            layer_indices = self.vfp_attrs.sld_constraint.layer_choices()
            integrals = integrate_vfp(
                self.zeds,
                self.indices,
                self._arrtotuple(p_vfp),
                tuple(layer_indices),
            )
            # user defines a class with a callable, which returns a list of
            # indices for modifying SLD values at those idxs.
            layer_idxs, slds = self.vfp_attrs.sld_constraint(integrals)
            for layer_idx, sld in zip(layer_idxs, slds, strict=False):
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
        """
        Get volume fraction profile for plotting.

        Parameters
        ----------
        reduced: bool, optional
            Flag to return reduced vfp. If False, get non-reduced vfp.

        Returns
        -------
        np.array
            vfp (reduced or full).
        np.array
            magnetic vfp after demag_f applied (reduced or full).
        """
        # update the model. Captures instances where parameters have changed.
        self.process_model()

        red_vfp, red_demag_vfp, _, demag_arr = init_demag(
            self.vfp_attrs.tup_demag_locs,
            self.vfp_attrs.tup_demag_widths,
            self.vfp_attrs.tup_mslds,
            self.zeds,
            self._arrtotuple(self.vfp),
        )

        if reduced:
            p_vfp = red_vfp
            demag_vfp = red_demag_vfp
        else:
            p_vfp = self.vfp
            demag_vfp = self.vfp * demag_arr

        if self.vfp_attrs.orientation == "back":
            p_vfp = p_vfp[::-1]  # reverse order.
            demag_vfp = demag_vfp[::-1]

        return p_vfp, demag_vfp

    def z_and_sld(
        self, reduced: bool = True, align_at_interface: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get z and sld values from vfp for plotting.

        Returns z values from `self.zeds` and also returns non-microsliced
        sld values from `self.get_slds` calculated from the VFP.

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
        self.process_model()  # update the model.
        offset = np.cumsum(self.vfp_attrs.tup_thicks)[align_at_interface]
        z = np.array(self.zeds) - offset
        z = -z if self.vfp_attrs.orientation == "back" else z
        slds = self.get_slds(reduced=reduced)
        # conditionally remove z at indices.
        if reduced:
            delete_idx = transform_indices(self.indices)
            z = np.delete(z, delete_idx)
        return z, slds.T

    def sld_offset(self) -> float:
        """
        Float to add to refnx or refl1d sld profile z coordinate.

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
        # update the model. Captures instances where parameters have changed.
        self.process_model()

        if self.vfp_attrs.orientation == "front":
            sldprof_offset_nr = -5 - (4 * self.tup_roughs[0])
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
            zend_front = self.dz[-1] + zend_of_vfprofile
            sldprof_offset = -(zend_front - np.sum(self.vfp_attrs.tup_thicks))

        return sldprof_offset

    def plot(  # noqa : PLR0913
        self,
        plots_required: list[Literal["sld", "vfp", "surfaces"]] | None = None,
        posterior_samples: dict[str, np.ndarray] | None = None,
        align_at: int | None = None,
        fig: Figure | None = None,
        sld_plot_kwargs: SldPlotKwargType | None = None,
        vfp_plot_kwargs: VfpPlotKwargType | None = None,
        surface_plot_kwargs: SurfacePlotKwargType | None = None,
    ) -> tuple[Figure, Axes | np.ndarray[Axes]]:
        """
        Makes a one to three axis figure to visualise the VFP model.

        By default the order of the plots are:
            Top plot = nsld / msld / isld
            Middle plot = volume fraction profiles
            Bottom plot = surface profiles
        This can be altered by specifying a different order in
        `plots_required`.

        Parameters
        ----------
        plots_required : list[Literal["sld", "vfp", "surfaces"]] | None, opt
            A list of plots required. Possible acceptable string values are
            "sld", "vfp", "surfaces". The order of the strings in the list
            will affect the order of the plot. Duplicates will be ignored.
        posterior_samples : dict[str, np.ndarray] | None, optional
            Samples from the posterior to plot in the "sld" and "vfp" plots.
            The keys should match the names of varying parameters in the vfp.
            Array values should be 1D of parameter values.
            By default is None.
        align_at : int | None, optional
            Specifies which interface is defined as z = 0 by index.
            If not specified, defaults to first interface.
        fig : Figure | None, optional.
            If supplied, plots will be plotted on `fig`.
            By default a new Figure will be created.
        sld_plot_kwargs : SldPlotKwargType | None, optional
            Kwargs to be passed to `vfp.plotting.PlotType._plot_sld`.
            By default None.
        vfp_plot_kwargs : VfpPlotKwargType | None, optional
            Kwargs to be passed to `vfp.plotting.PlotType._plot_vfp`.
            By default None.
        surface_plot_kwargs : SurfacePlotKwargType | None, optional
            Kwargs to be passed to `vfp.plotting.PlotType._plot_surfaces`.
            By default None.

        Returns
        -------
        tuple[Figure, Axes | np.ndarray[Axes]]
            Figure and axes objects.
        """
        # update the model. Captures instances where parameters have changed.
        self.process_model()

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

        if isinstance(align_at, int):
            if align_at > len(self.vfp_attrs.thicknesses) - 1:
                raise ValueError("align_at must be an index of the layers.")

        fig, ax = model_plot(
            vfp=self,
            plots_required=plots_required,
            posterior_samples=posterior_samples,
            align_at=align_at,
            fig=fig,
            sld_plot_kwargs=sld_plot_kwargs,
            vfp_plot_kwargs=vfp_plot_kwargs,
            surface_plot_kwargs=surface_plot_kwargs,
        )

        return fig, ax

    def _arrtotuple(
        self, arr: np.ndarray
    ) -> tuple[float, ...] | tuple[tuple[float, ...]]:
        """
        Convert arrays to tuples for caching.

        Parameters
        ----------
        arr : np.ndarray
            Array to convert to tuples.

        Returns
        -------
        tuple[float, ...] | tuple[tuple[float, ...]]
            tuple or nested tuple of floats.
        """
        if arr.ndim == 1:
            return tuple(val for val in arr)

        elif arr.ndim == 2:  # noqa : PLR2004
            return tuple([tuple([float(val) for val in row]) for row in arr])

    def _init_vfp_attrs(
        self,
        arr_attrs: list[
            list[ParameterLike | None] | tuple[ParameterLike] | list[int]
        ],
        other_attrs: list[
            Literal["front", "back"],
            Literal["none", "up", "down"],
            SLDConstraintType | None,
            float,
        ],
        name: str,
    ) -> VFPAttributes:
        """
        Inits a `VFPAttributes` to hold reference to child VFP input
        parameters.

        Returns
        -------
        VFPAttributes
        """
        (
            thicknesses,
            roughnesses,
            nslds,
            islds,
            mslds,
            demaglocs,
            demagwidths,
            conformal,
        ) = list(map(np.array, arr_attrs))
        orientation, spin_state, sld_constraint, max_delta_z = other_attrs
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
            conformal=conformal,
            name=name,
        )
        return attrs

    @property
    @abstractmethod
    def vfp_attrs(self) -> VFPAttributes:
        """
        Abstract property to implement in a child of `BaseVFP` to use the
        VFPAttribute dataclass.
        """
        raise NotImplementedError

    @abstractmethod
    def set_parameter_prior(self):
        """
        Abstract method to set priors on fitting parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self):
        """
        Abstract method to transform a VFP of one type to another.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def varying_parameters(self) -> dict[str, ParameterLike] | None:
        """
        Abstract method to get all varying parameters in vfp.
        """
        raise NotImplementedError

    @abstractmethod
    def _createparam(self):
        """
        Abstract method that should be implemented to return `ParameterLike`
        objects for fitting software. Not required for standard VFP.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError


def check_init_input(  # noqa : PLR0912, PLR0913
    thicknesses: tuple[ParameterLike] | list[ParameterLike],
    roughnesses: tuple[ParameterLike, str] | list[ParameterLike, str],
    nslds: tuple[ParameterLike] | list[ParameterLike],
    islds: tuple[ParameterLike] | list[ParameterLike] | None,
    mslds: tuple[ParameterLike] | list[ParameterLike] | None,
    demaglocs: tuple[ParameterLike] | list[ParameterLike] | None,
    demagwidths: tuple[ParameterLike] | list[ParameterLike] | None,
    spin_state: Literal["none", "up", "down"],
    max_delta_z: float,
) -> tuple[
    list[ParameterLike | None],
    list[list[ParameterLike]],
    list[ParameterLike] | tuple[ParameterLike],
    list[ParameterLike] | tuple[ParameterLike],
    list[int],
]:
    """
    Check over the input values to the VFP classes.

    Helper function as all types of VFP require similar input structure.
    """
    # now check given parameters
    if not demaglocs:  # look for empty lists or None.
        demaglocs = []

    if not demagwidths:
        demagwidths = []

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
        raise ValueError("Roughness parameters must be > 0 ")

    if len(nslds) != len(thicknesses) + 1:
        raise ValueError(
            """The number of supplied SLD values must be 1 greater
                than the number of thickness parameters."""
        )

    # init a list of where conformal interfaces are:
    conformal = []
    for roughness in roughnesses:
        # cannot use a type alias in isinstance so use its value attr
        if isinstance(roughness, str | ParameterLike.__value__):
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
                " interger, refnx.analysis.parameter or a string =="
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
