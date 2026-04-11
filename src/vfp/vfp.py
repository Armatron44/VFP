"""Concrete implementations of ``BaseVFP``."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal, Never, override

import numpy as np

from vfp.basevfp import BaseVFP, VFPAttributes, _check_init_input
from vfp.vfp_typing import ParameterLike, SldConstraintType

HAS_REFL1D = False
# Likely to change as bumps / refl1d are going through api refactor.
try:
    from bumps.parameter import (
        Expression,
    )
    from bumps.parameter import Parameter as bumpsParameter
    from bumps.parameter import (
        to_dict,
    )
    from refl1d.probe import NeutronProbe
    from refl1d.profile import Microslabs
    from refl1d.sample.layers import Layer

    HAS_REFL1D = True
except ImportError as ie:
    warnings.warn(
        f"{ie} compatible refl1d & bumps packages not installed.",
        stacklevel=2,
    )

HAS_REFNX = False
try:
    from refnx.analysis import (
        PDF,
        Interval,
    )
    from refnx.analysis import Parameter as refnxParameter
    from refnx.analysis import (
        Parameters,
        possibly_create_parameter,
    )
    from refnx.analysis.parameter import _BinaryOp
    from refnx.reflect import Component, Structure

    HAS_REFNX = True
except ImportError as ie:
    warnings.warn(
        f"{ie} compatible refnx package not installed.", stacklevel=2
    )


class VFP(BaseVFP):
    """Describes SLD profiles of interfaces from fronting to backing.

    SLD profiles are calculated by generating volume fraction profiles.
    These volume fraction profiles cannot be negative anywhere, and the total
    volume fraction must be one everywhere.
    """

    def __init__(  # noqa: PLR0913
        self,
        nslds: Sequence[ParameterLike],
        thicknesses: Sequence[ParameterLike],
        roughnesses: Sequence[ParameterLike | Literal["conformal"]],
        islds: Sequence[ParameterLike] | None = None,
        mslds: Sequence[ParameterLike] | None = None,
        spin_state: Literal["none", "up", "down"] = "none",
        orientation: Literal["front", "back"] = "front",
        demaglocs: Sequence[ParameterLike] | None = None,
        demagwidths: Sequence[ParameterLike] | None = None,
        sld_constraint: SldConstraintType | None = None,
        max_delta_z: float = 0.5,
    ) -> None:
        """Init a ``VFP``.

        Parameters
        ----------
        nslds : Sequence[ParameterLike]
            Nuclear scattering length densities of each material in the model.
        thicknesses : Sequence[ParameterLike]
            Thicknesses of layers in the model. These control the
            midpoint-to-midpoint width of a layer's transition to and from
            other materials.
        roughnesses : Sequence[ParameterLike]
            Roughnesses of layers. These control the width of interfaces
            between adjacent layers in the volume fraction profile.
        islds : Sequence[ParameterLike] | None, optional.
            Imaginary scattering length densities of each layer within the
            model. Defaults to None.
        mslds : Sequence[ParameterLike] | None, optional.
            Magnetic scattering length densities of each layer within the
            volume fraction profile. Defaults to None.
        spin_state : str, optional
            Defines if slds should be calculated as nuclear (spin_state =
            'none'), nuclear+magnetic (spin_state = 'up') or nuclear-magnetic
            (spin_state = 'down'). Defaults to 'none'.
        orientation : str, optional
            Defines if incident radiation passed through fronting or backing.
            Through the fronting = (orientation = 'front'), through
            the backing = (orientation = 'back'). Useful for co-refinement
            of solid-liquid NR data with air-solid x-ray reflectometry data.
            Optional, defaults to 'front'.
        demaglocs : Sequence[ParameterLike] | None, optional
            If supplied, must either be a tuple/list of an even number of
            ParameterLike objects. The parameters declare the centre point of
            a Gaussian CDF. The parameters are consecutive, so the z location
            of parameter 2 will be parameter 1 value + parameter 2 value.
            Defaults to None.
        demagwidths : Sequence[ParameterLike] | None. optional
            If supplied, must either be a tuple/list of an even number of
            ParameterLike objects. The parameters declare the width of a
            Gaussian CDF. Defaults to None.
        sld_constraint : SldConstraintType | None, optional
            User defined object used to handle SLD constraints between layers.
            Defaults to None.
        max_delta_z : float, optional
            Defines the approximate thickness of a microslice across the VFP.
            Defaults to 0.5 angstrom.
        """
        self.name = "VFP"
        # check some of the input pars & process roughnesses.
        checked_res = _check_init_input(
            thicknesses,
            roughnesses,
            nslds,
            islds,
            mslds,
            demaglocs,
            demagwidths,
            spin_state,
            max_delta_z,
        )
        roughnesses_alt, all_slds, demaglocs, demagwidths, conformal = (
            checked_res
        )

        arr_attrs = [
            thicknesses,
            roughnesses_alt,
            *all_slds,
            demaglocs,
            demagwidths,
            conformal,
        ]

        other_attrs = (orientation, spin_state, sld_constraint, max_delta_z)

        # init VFPAttrs object via parent class method
        self._vfp_attrs = self._init_vfp_attrs(
            arr_attrs, other_attrs, name=self.name
        )

    @override
    @property
    def vfp_attrs(self) -> VFPAttributes:
        return self._vfp_attrs

    @override
    def _createparam(
        self, params: Sequence[ParameterLike | None], nameid: str
    ):
        raise NotImplementedError

    @override
    def set_parameter_prior(self) -> None:
        raise NotImplementedError

    @override
    def transform(
        self, wanted_vfp: Literal["VFP", "refnxVFP", "refl1dVFP"]
    ) -> refnxVFP | refl1dVFP:
        """Transform a ``VFP`` to a ``refnxVFP`` or ``refl1dVFP``.

        Parameters
        ----------
        wanted_vfp : Literal["VFP", "refnxVFP", "refl1dVFP"]
            The desired type of vfp: "refnxVFP" or "refl1dVFP".

        Raises
        ------
        ValueError
            If vfp_type is not "refl1dVFP" or "refnxVFP".

        Returns
        -------
        refnxVFP | refl1dVFP
            Transformed version of VFP.
        """
        if wanted_vfp not in ["refnxVFP", "refl1dVFP"]:
            raise ValueError(
                'vfp_type must be either "refnxVFP" or "refl1dVFP".'
            )

        transformed_vfp = init_specific_vfp(
            self, wanted_vfp, self.vfp_attrs.__dict__
        )
        if isinstance(transformed_vfp, VFP):
            raise TypeError("transformed_vfp is same type.")
        return transformed_vfp

    @override
    @property
    def varying_parameters(self) -> Never:
        raise NotImplementedError


if HAS_REFNX:

    class refnxVFP(Component, BaseVFP):  # noqa: N801
        """VFP for use with refnx."""

        def __init__(  # noqa: PLR0913
            self,
            nslds: Sequence[ParameterLike],
            thicknesses: Sequence[ParameterLike],
            roughnesses: Sequence[ParameterLike | Literal["conformal"]],
            islds: Sequence[ParameterLike] | None = None,
            mslds: Sequence[ParameterLike] | None = None,
            spin_state: Literal["none", "up", "down"] = "none",
            orientation: Literal["front", "back"] = "front",
            demaglocs: Sequence[ParameterLike] | None = None,
            demagwidths: Sequence[ParameterLike] | None = None,
            sld_constraint: SldConstraintType | None = None,
            max_delta_z: float = 0.5,
        ) -> None:
            self.name = "refnxVFP"
            # check some of the input pars & process roughnesses.
            checked_res = _check_init_input(
                thicknesses,
                roughnesses,
                nslds,
                islds,
                mslds,
                demaglocs,
                demagwidths,
                spin_state,
                max_delta_z,
            )
            roughnesses_alt, all_slds, demaglocs, demagwidths, conformal = (
                checked_res
            )
            # convert parameters to refnxParameters
            thicknesses_p = self._createparam(thicknesses, "thicknesses")
            demaglocs_p = self._createparam(demaglocs, "demaglocs")
            demagwidths_p = self._createparam(demagwidths, "demagwidths")
            roughnesses_p = self._createparam(roughnesses_alt, "roughnesses")
            all_slds = map(
                self._createparam, all_slds, ["nsld", "isld", "msld"]
            )
            arr_attrs = [
                thicknesses_p,
                roughnesses_p,
                *all_slds,
                demaglocs_p,
                demagwidths_p,
                conformal,
            ]

            other_attrs = (
                orientation,
                spin_state,
                sld_constraint,
                max_delta_z,
            )

            # init VFPAttrs object via parent class method
            self._vfp_attrs = self._init_vfp_attrs(
                arr_attrs, other_attrs, name=self.name
            )

            # Init Component and get attrs in self.
            Component.__init__(self)

        @override
        @property
        def vfp_attrs(self) -> VFPAttributes:
            return self._vfp_attrs

        @property
        def parameters(self) -> Parameters:
            """Collates ``refnxParameter``s in ``self.vfp_attrs``.

            refnx uses this property when collating parameters in a component.
            Will return unique `refnxParameter`s in `nslds`, `thicknesses`,
            `roughnesses`, `mslds`, `islds` for varying and non-varying
            parameters. If not defined, `mslds` and `islds` are set as fixed
            at 0.

            If demaglocs and demagwidths are not defined, these will not be
            added to parameters.
            """
            # create a list of list of parameters
            llps = [
                lps
                for lps in [
                    self.vfp_attrs.thicknesses,
                    self.vfp_attrs.roughnesses,
                    self.vfp_attrs.demaglocs,
                    self.vfp_attrs.demagwidths,
                    self.vfp_attrs.nslds,
                    self.vfp_attrs.mslds,
                    self.vfp_attrs.islds,
                ]
                if lps.size > 0  # only accept non-empty par arrays.
            ]
            # flatten and remove Nones.
            flat_pars = [ps for lps in llps for ps in lps if ps is not None]
            # extract refnxParameters from _BinaryOps.
            ps = []
            for par in flat_pars:
                if isinstance(par, refnxParameter):
                    ps.append(par)
                elif isinstance(par, _BinaryOp):
                    ps.extend([_p for _p in par.dependencies()])
            # possible that ps now contains duplicates.
            ps = [par for i, par in enumerate(ps) if par not in ps[:i]]
            # now put all in Parameters container.
            p = Parameters(data=ps, name=self.name)
            return p

        @override
        def set_parameter_prior(
            self,
            priors: dict[
                str, dict[int, tuple[float, float] | PDF | Interval]
            ],
        ) -> None:
            """Set bounds on ``refnxParameter``s in ``self.vfp_attrs``.

            Use this function to set the prior for any parameters
            that are to be fit / sampled.

            The key names on the first level of the dictionary must
            match the names of the attributes in `self.vfp_attrs`. The
            key values of the second level of the dictionaries should
            match the indices of the parameters you wish to set priors for.

            Parameters
            ----------
            priors : dict[
                str, dict[int, tuple[float, float] | PDF | Interval]
            ]
                Nested dictionary of priors to be applied to
                `refnxParameter`s. The outer dictionary takes a str key to
                indicate what type of parameter (e.g 'thickness') should be
                given a prior. The available choices of parameters are those
                in `self.vfp_attrs`. The inner dictionary takes a int key to
                index into which specific parameter in the specified parameter
                type. The inner dictionary can take a tuple[float, float],
                `refnx.analysis.PDF` or `refnx.analysis.Interval` to be
                applied to the refnxParameters in `self.vfp_attrs`.
                See Example below.

            Example
            -------
            >>> import scipy.stats as stats
            >>> from vfp.vfp_refactor import refnxVFP
            >>> thicknesses = (0, 20)
            >>> roughnesses = (2, 1)
            >>> nslds = (2.07, 3.47, 6.37) # Si, SiO2, D2O
            >>> refnx_vfp = refnxVFP(nslds, thicknesses, roughnesses)
            ... # lets set uniform priors on the thickness and roughness of
            ... # SiO2 and set a gaussian prior on the sld of D2O with mean
            ... # 6.37 & std 0.03
            >>> prior_dict = {
            ... 'thicknesses' : {1 : (10, 30)},
            ... 'roughnesses' : {1 : (1, 4)},
            ... 'nslds' : {2 : PDF(stats.norm(6.37, 0.03))}
            ... }
            >>> refnx_vfp.set_parameter_prior(prior=prior_dict)
            >>> refnx_vfp.vfp_attrs.thicknesses[1]
            Parameter(value=20.0, name='refnxVFP - thicknesses - layer 1',
            ...       vary=True, bounds=Interval(lb=10.0, ub=30.0),
            ...       constraint=None)
            >>> refnx_vfp.vfp_attrs.roughnesses[1]
            Parameter(value=1.0,
            ...       name='refnxVFP - roughnesses - layer 1/backing',
            ...       vary=True, bounds=Interval(lb=1.0, ub=4.0),
            ...       constraint=None)
            >>> refnx_vfp.vfp_attrs.nslds[2]
            Parameter(value=6.37,
            ...       name='refnxVFP - nsld - backing', vary=True,
            ...       bounds=PDF(
            ...  <scipy.stats._distn_infrastructure.rv_continuous_frozen
            ...  object at 0x00000145898486E0>), constraint=None)
            """
            # take parameters from vfp_attrs as parameters property
            # is built from vfp_attrs. Take shallow copy, which will
            # update the attributes of vfp_attrs.
            pars_dict = self.vfp_attrs.__dict__

            # apply bounds nested dict to pars_dict:
            for par_type, pars in priors.items():
                for idx, prior in pars.items():
                    pars_dict[par_type][idx].bounds = prior
                    # set to vary if prior set.
                    pars_dict[par_type][idx].vary = True

        def slabs(self, structure: Structure | None = None) -> np.ndarray:
            """
            Generate array representation of the `refnxVFP`.

            A 2D np.array using the thicknesses, slds and islds of the
            microslabs.

            Parameters
            ----------
            structure : refnx.reflect.Structure, optional
                The refnx.reflect.Structure hosting this VFP component.
                Defaults to None.

            Raises
            ------
            TypeError: if the VFP is not part of a refnx.reflect.Structure,
                    this function will raise a ValueError.

            Returns
            -------
            np.array
                slabs is a 2d np.array with shape = (Nlayers, 5).
            """
            if structure is None:
                raise TypeError("VFP.slabs() requires a valid Structure")

            # use the process method of the VFP class to
            # return total slds, islds and thicknesses of each slab
            slds, islds, thicks = self.process_model()

            # init a 2D array (Nlayers, 5)
            slabs = np.zeros((len(thicks), 5))

            # now populate slabs with microslab thicknesses & SLDs.
            slabs[:, 0] = thicks
            slabs[:, 1] = slds
            slabs[:, 2] = islds
            return slabs

        @override
        def transform(
            self, wanted_vfp: Literal["VFP", "refnxVFP", "refl1dVFP"]
        ) -> VFP | refl1dVFP:
            """Transform ``refnxVFP`` to a ``VFP`` or ``refl1dVFP``.

            Parameters
            ----------
            wanted_vfp : Literal["VFP", "refnxVFP", "refl1dVFP"]
                The desired type of vfp: "refnxVFP" or "refl1dVFP".

            Raises
            ------
            ValueError
                If ``wanted_vfp`` is not "refl1dVFP" or "VFP".

            Returns
            -------
            VFP | refl1dVFP
                Transformed version of VFP.
            """
            if wanted_vfp not in ["vfp", "refl1d"]:
                raise ValueError('vfp_type must be either "vfp" or "refl1d".')

            transformed_vfp = init_specific_vfp(
                self, wanted_vfp, self.vfp_attrs.__dict__
            )
            if isinstance(transformed_vfp, refnxVFP):
                raise TypeError("transformed_vfp is same type.")
            return transformed_vfp

        @override
        @property
        def varying_parameters(self) -> dict[str, refnxParameter]:
            """Gets parameters that vary in this ``refnxVFP``.

            Returns
            -------
            dict[str, refnxParameter]
                Varying parameters of vfp.
            """
            # get varying parameters. Not set as attr as would
            # require re-init of vfp to change vary on any parameter.
            return {p.name: p for p in self.parameters if p.vary}

        @varying_parameters.setter
        def varying_parameters(
            self, values_dict: dict[str, ParameterLike]
        ) -> None:
            """Set the values of the ``refnxVFP.varying_parameters``.

            Parameters
            ----------
            values_dict: dict[str, ParameterLike]
                1D array of values for varying parameters.
            """
            # get varying parameters
            varying_pars = {p.name: p for p in self.parameters if p.vary}
            # and set value.
            for p_name, p in values_dict.items():
                varying_pars[p_name].value = float(p)

        @override
        def _createparam(  # noqa : PLR0912
            self,
            params: Sequence[ParameterLike | None],
            nameid: str,
        ) -> Sequence[ParameterLike | None]:
            """Get list of Parameters (or ops) / None.

            The parameters do not having to be varying.

            Parameters
            ----------
            params : Sequence[ParameterLike | None]]
                Sequence of parameter values.
            nameid : str
                The name of the collective parameters.
            """
            # create a list of strings that describe what each parameter is.
            # depends on which parameters we are dealing with.
            layer_strs = []
            if nameid in ("nsld", "msld", "isld"):
                for i in range(len(params)):
                    if i == 0:
                        layer_strs.append("fronting")
                    elif i == (len(params) - 1):
                        layer_strs.append("backing")
                    else:
                        layer_strs.append(f"layer {i}")
            elif nameid == "thicknesses":
                for i in range(len(params)):
                    layer_str = "fronting" if i == 0 else f"layer {i}"
                    layer_strs.append(layer_str)
            elif nameid in ("demagwidths", "demaglocs"):
                for i in range(len(params)):
                    peak_str = f"peak {(i + 2) // 2}"
                    side_str = "left" if i % 2 == 0 else "right"
                    layer_strs.append(peak_str + " " + side_str)
            elif nameid in ("roughnesses"):
                for i, par in enumerate(params):
                    if par is not None:
                        layer_before = "fronting" if i == 0 else f"layer {i}"
                        layer_after = (
                            "backing"
                            if i == (len(params) - 1)
                            else f"layer {i + 1}"
                        )
                        layer_strs.append(layer_before + "/" + layer_after)
                    else:
                        layer_strs.append(None)

            output: list[ParameterLike | None] = []
            for layer_str, par in zip(layer_strs, params, strict=False):
                if isinstance(par, _BinaryOp):
                    output.append(
                        par
                    )  # keep as _BinaryOp until parameters property.
                elif nameid == "roughnesses":
                    if par is not None:
                        output.append(
                            possibly_create_parameter(
                                par,
                                name=f"{self.name} - {nameid} - {layer_str}",
                            )
                        )
                    else:
                        output.append(None)
                else:
                    output.append(
                        possibly_create_parameter(
                            par,
                            name=f"{self.name} - {nameid} - {layer_str}",
                        )
                    )

            return output


if HAS_REFL1D:

    class refl1dVFP(Layer, BaseVFP):  # noqa: N801
        """VFP for use with refl1d."""

        def __init__(  # noqa: PLR0913
            self,
            nslds: Sequence[ParameterLike],
            thicknesses: Sequence[ParameterLike],
            roughnesses: Sequence[ParameterLike | Literal["conformal"]],
            islds: Sequence[ParameterLike] | None = None,
            mslds: Sequence[ParameterLike] | None = None,
            spin_state: Literal["none", "up", "down"] = "none",
            orientation: Literal["front", "back"] = "front",
            demaglocs: Sequence[ParameterLike] | None = None,
            demagwidths: Sequence[ParameterLike] | None = None,
            sld_constraint: SldConstraintType | None = None,
            max_delta_z: float = 0.5,
        ) -> None:
            self.name = "refl1dVFP"
            # check some of the input pars & process roughnesses.
            checked_res = _check_init_input(
                thicknesses,
                roughnesses,
                nslds,
                islds,
                mslds,
                demaglocs,
                demagwidths,
                spin_state,
                max_delta_z,
            )
            roughnesses_alt, all_slds, demaglocs, demagwidths, conformal = (
                checked_res
            )
            nslds, islds, mslds = all_slds

            # convert parameters to bumpsParameters
            thicknesses_p = self._createparam(thicknesses, "thicknesses")
            demaglocs_p = self._createparam(demaglocs, "demaglocs")
            demagwidths_p = self._createparam(demagwidths, "demagwidths")
            roughnesses_p = self._createparam(roughnesses_alt, "roughnesses")
            all_slds = map(
                self._createparam, all_slds, ["nsld", "isld", "msld"]
            )

            arr_attrs = [
                thicknesses_p,
                roughnesses_p,
                *all_slds,
                demaglocs_p,
                demagwidths_p,
                conformal,
            ]

            other_attrs = (
                orientation,
                spin_state,
                sld_constraint,
                max_delta_z,
            )

            # init VFPAttrs object via parent class method
            self._vfp_attrs = self._init_vfp_attrs(
                arr_attrs, other_attrs, name=self.name
            )

            # Init Layer and get attrs in self.
            Layer.__init__(self)

            # refl1d needs total thickness of the vfp as attr
            # at the beginning and throughout fitting.
            _, _, thicks = self.process_model()
            self.thickness = bumpsParameter(
                thicks.sum(), name=f"{self.name} - total thickness"
            )

        def set_parameter_prior(
            self, priors: dict[str, dict[int, tuple[float, float]]]
        ) -> None:
            """
            Set bounds on `bumpsParameter`s in `self.vfp_attrs`.

            Use this function to set the prior for any parameters
            that are to be fit / sampled.

            The key names on the first level of the dictionary must
            match the names of the attributes in `self.vfp_attrs`. The
            key values of the second level of the dictionaries should
            match the indices of the parameters you wish to set priors for.

            Parameters
            ----------
            priors: dict[str, dict[int, tuple[float, float]]]
                Nested dictionary of priors to be applied to
                `bumpsParameter`s. The outer dictionary takes a str key to
                indicate what type of parameter (e.g 'thickness') should be
                given a prior. The available choices of parameters are those
                in `self.vfp_attrs`. The inner dictionary takes a int key to
                index into which specific parameter in the specified parameter
                type. The inner dictionary can currently only take a
                tuple[float, float] for lower and upper bounds (flat prior)
                to apply to the `bumpsParameter`s in `self.vfp_attrs`.
                See Example below.

            Example
            -------
            >>> from vfp import refl1dVFP
            >>> import scipy.stats as stats
            >>> thicknesses = (0, 20)
            >>> roughnesses = (2, 1)
            >>> nslds = (2.07, 3.47, 6.37) # Si, SiO2, D2O
            >>> refl1d_vfp = refl1dVFP(nslds, thicknesses, roughnesses)
            # lets set uniform priors on the thickness and roughness of SiO2
            >>> prior_dict = {'thicknesses' : {1 : (10, 30)},
                              'roughnesses' : {1 : (1, 4)},
                              }
            >>> refl1d_vfp.set_parameter_prior(prior=prior_dict)
            >>> refl1d_vfp.vfp_attrs.thicknesses[1].bounds
            (10, 30)
            >>> refl1d_vfp.vfp_attrs.roughnesses[1].bounds
            (1, 4)
            """
            # TODO: how can I get this to work with
            # scipy.stats distributions?

            # take parameters from vfp_attrs as parameters property
            # is built from vfp_attrs. Take shallow copy, which will
            # update the attributes of vfp_attrs.
            pars_dict = self.vfp_attrs.__dict__

            # apply bounds nested dict to pars_dict:
            for par_type, pars in priors.items():
                for idx, prior in pars.items():
                    low, high = prior
                    pars_dict[par_type][idx].range(low, high)

        @property
        def vfp_attrs(self) -> VFPAttributes:
            """Return reference to VFPAttributes object setup in init."""
            return self._vfp_attrs

        def to_dict(self) -> dict[str | str, list[ParameterLike]]:
            """Get a dict repr of ``VFPattributes``.

            For use with bumps. Used when saving a refl1d model details
            as a .json file.

            Returns
            -------
            dict[str | str, list[ParameterLike]]
                repr of the refl1d.vfp_attrs.
            """
            return to_dict(self.vfp_attrs.__dict__)

        def layer_parameters(self) -> dict[str, list[bumpsParameter]]:
            """Get `bumpsParameter``s in ``refl1dVFP``.

            Will return key, value pairs of ``bumpsParameter``s in ``nslds``,
            ``thicknesses``, ``roughnesses``, ``mslds``, ``islds`` for varying
            and non-varying parameters. If not defined, ``mslds`` and
            ``islds`` are set as fixed at 0.

            If demaglocs and demagwidths are not defined, these will not be
            returned.

            Returns
            -------
            dict[str, list[bumpsParameter]]
                Parameters with key equal to the name of the ``refl1dVFP``
                parameters.
            """
            # remove empty arrays
            p_arr_dict = {}
            interest_keys = [
                "nslds",
                "thicknesses",
                "roughnesses",
                "islds",
                "mslds",
                "demaglocs",
                "demagwidths",
            ]
            for key, p_arr in self.vfp_attrs.__dict__.items():
                if key in interest_keys:
                    if p_arr.size > 0:
                        p_arr_dict[key] = p_arr

            # now extract just bumpsParameters, avoiding Expressions and Nones
            extract_ps = {}
            for key, par_arr in p_arr_dict.items():
                extract_ps[key] = []
                for par in par_arr:  # don't keep duplicates in the same list.
                    if isinstance(par, bumpsParameter):
                        if par not in extract_ps[key]:
                            extract_ps[key].append(par)
                    elif isinstance(par, Expression):
                        ext_pars = [_p for _p in par.parameters()]
                        for _p in ext_pars:
                            if _p not in extract_ps[key]:
                                extract_ps[key].append(_p)
            return extract_ps

        def render(self, probe: NeutronProbe, slabs: Microslabs) -> None:
            """Append microslice thickness, SLDs and iSLDs to ``Microslabs``.

            ``Microslabs`` is passed to the render function of the
            ``refl1dVFP`` by refl1d's ``Experiment``. Also updates the
            ``self.thickness`` value of the ``refl1dVFP``.

            Parameters
            ----------
            probe : refl1d.probe.NeutronProbe
                Passed to render functions of refl1d.layers,
                but not used here.
            slabs : refl1d.profile.Microslabs
                Has rho, irho, w and sigma properties.
            """
            # use the process method of the BaseVFP class to
            # return total slds, islds and thicknesses of each slab
            slds, islds, thicks = self.process_model()

            # update the self.thickness variable.
            self.thickness.value = thicks.sum()

            # now append slds, islds and thicks to slabs.
            for i in range(0, len(thicks)):
                slabs.append(rho=slds[i], irho=islds[i], w=thicks[i], sigma=0)

        def transform(
            self, wanted_vfp: Literal["VFP", "refnxVFP", "refl1dVFP"]
        ) -> VFP | refnxVFP:
            """Transform ``refl1dVFP`` to a ``VFP`` or ``refnxVFP``.

            Parameters
            ----------
            wanted_vfp : Literal["VFP", "refnxVFP", "refl1dVFP"]
                Either "VFP" or "refnxVFP".

            Raises
            ------
            ValueError
                If vfp_type is not "refnx" or "vfp".

            Returns
            -------
            VFP | refl1dVFP
                Transformed version of VFP.
            """
            if wanted_vfp not in ["vfp", "refnx"]:
                raise ValueError('vfp_type must be either "vfp" or "refnx".')

            transformed_vfp = init_specific_vfp(
                self, wanted_vfp, self.vfp_attrs.__dict__
            )
            if isinstance(transformed_vfp, refl1dVFP):
                raise TypeError("transformed_vfp is same type.")
            return transformed_vfp

        @override
        @property
        def varying_parameters(self) -> dict[str, bumpsParameter]:
            """Fit parameters from ``refl1dVFP``.

            Returns
            -------
            dict[str, bumpsParameter]
                Varying parameters.
            """
            ps = self.layer_parameters()
            varying_pars = {
                p.name: p
                for p_list in ps.values()
                for p in p_list
                if p.bounds is not None and p.name is not None
            }
            return varying_pars

        @varying_parameters.setter
        def varying_parameters(
            self, values_dict: dict[str, ParameterLike]
        ) -> None:
            """Set the values of ``refl1dVFP.varying_parameters``.

            Parameters
            ----------
            values_dict : dict[str, np.ndarray]
                1D array of values for varying parameters.
            """
            ps = self.layer_parameters()
            varying_pars = {
                p.name: p
                for p_list in ps.values()
                for p in p_list
                if p.bounds is not None and p.name is not None
            }
            for key, value in values_dict.items():
                varying_pars[key].value = float(value)

        @override
        def _createparam(  # noqa : PLR0912
            self,
            params: Sequence[ParameterLike | None],
            nameid: str,
        ) -> Sequence[ParameterLike | None]:
            """Create a list of ``bumpsParameter``s.

            Parameters
            ----------
            param : tuple[ParameterLike | None] | list[ParameterLike | None]
                Sequence of parameter values.
            nameid : str
                The name of the collective parameters.

            Returns
            -------
                list[ParameterLike | None]
            """
            # create a list of strings that describe what each parameter is.
            # depends on which parameters we are dealing with.
            layer_strs = []
            if nameid in ("nsld", "msld", "isld"):
                for i in range(len(params)):
                    if i == 0:
                        layer_strs.append("fronting")
                    elif i == (len(params) - 1):
                        layer_strs.append("backing")
                    else:
                        layer_strs.append(f"layer {i}")
            elif nameid == "thicknesses":
                for i in range(len(params)):
                    layer_str = "fronting" if i == 0 else f"layer {i}"
                    layer_strs.append(layer_str)
            elif nameid in ("demagwidths", "demaglocs"):
                for i in range(len(params)):
                    peak_str = f"peak {(i + 2) // 2}"
                    side_str = "left" if i % 2 == 0 else "right"
                    layer_strs.append(peak_str + " " + side_str)
            elif nameid in ("roughnesses"):
                for i, par in enumerate(params):
                    if par is not None:
                        layer_before = "fronting" if i == 0 else f"layer {i}"
                        layer_after = (
                            "backing"
                            if i == (len(params) - 1)
                            else f"layer {i + 1}"
                        )
                        layer_strs.append(layer_before + "/" + layer_after)
                    else:
                        layer_strs.append(None)

            output = []
            for layer_str, par in zip(layer_strs, params, strict=False):
                if nameid == "roughnesses":
                    if par is not None:
                        output.append(
                            bumpsParameter.default(
                                par,
                                name=f"{self.name} - {nameid} - {layer_str}",
                            )
                        )
                    else:
                        output.append(None)
                elif isinstance(par, Expression):
                    try:  # check we can extract all objects in par.
                        exp_ps = par.parameters()
                    except TypeError as terr:
                        raise ValueError(
                            f"Supplied expression {par} must contain "
                            "only bumpsParameter types. Check for "
                            "refl1d.sample.material.SLD objects and "
                            "similar in Expression."
                        ) from terr
                    else:
                        if not all(
                            [isinstance(_p, bumpsParameter) for _p in exp_ps]
                        ):
                            raise ValueError(
                                f"Supplied expression {par} must contain "
                                "only bumpsParameter types. Check for "
                                "refl1d.sample.material.SLD objects and "
                                "similar in Expression."
                            )
                    output.append(
                        par
                    )  # keep as Expression until parameters property.
                else:
                    output.append(
                        bumpsParameter.default(
                            par,
                            name=f"{self.name} - {nameid} - {layer_str}",
                        )
                    )

            return output


def init_specific_vfp(
    original_vfp: refnxVFP | refl1dVFP | VFP,
    vfp_type: Literal["VFP", "refnxVFP", "refl1dVFP"],
    vfp_dict: dict[
        str,
        np.ndarray | str | float | None | SldConstraintType,
    ],
) -> refnxVFP | refl1dVFP | VFP:
    """Load a type of VFP.

    Called by transform methods of child classes of `vfp.basevfp.BaseVFP`.

    Parameters
    ----------
    original_vfp : VFP | refnxVFP | refl1dVFP
        The original vfp to transform to a different type of VFP.
    vfp_type : str
        Type of the desired VFP type.
    vfp_dict : dict[str, np.ndarray | str | float | None | SldConstraintType]
        Original VFP attributes as a dictionary

    Returns
    -------
    refnxVFP | refl1dVFP | VFP
    """
    if vfp_type == "refnxVFP":
        if HAS_REFNX:
            target_vfp = refnxVFP
        else:
            raise ValueError(
                "Target vfp is a refnxVFP, and refnx is not an available"
                " dependency."
            )

    elif vfp_type == "refl1dVFP":
        if HAS_REFL1D:
            target_vfp = refl1dVFP
        else:
            raise ValueError(
                "Target vfp is a refl1dVFP, and refl1d is not an available"
                " dependency."
            )

    elif vfp_type == "VFP":
        target_vfp = VFP

    else:
        raise ValueError(
            "model_type must be 'VFP' / 'refnxVFP' / 'refl1dVFP'."
        )

    # if asked for the same type as original_vfp just return original_vfp.
    if isinstance(target_vfp, type(original_vfp)):
        warnings.warn(
            "Returning the original vfp as target is the same.", stacklevel=2
        )
        return original_vfp

    final_vfp = target_vfp.from_transform(vfp_dict)

    return final_vfp
