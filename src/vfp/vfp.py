from __future__ import annotations
import copy
from typing import Literal
from collections.abc import Callable
import warnings

import numpy as np

from vfp.basevfp import BaseVFP, VFPAttributes, check_init_input
from vfp.typing import ParameterLike

HAS_REFL1D = False
# This is likely to change as bumps / refl1d are going through an extensive refactor.
try:
    from bumps.parameter import Parameter as bumpsParameter, to_dict, Expression 
    from refl1d.sample.layers import Layer
    from refl1d.probe import NeutronProbe
    from refl1d.profile import Microslabs
    HAS_REFL1D = True
except ImportError as ie:
    warnings.warn(f"{ie} compatible refl1d & bumps packages not installed.")

HAS_REFNX = False    
try:
    from refnx.reflect import Component, Structure
    from refnx.analysis import Parameter as refnxParameter, Parameters, possibly_create_parameter, PDF, Interval
    from refnx.analysis.parameter import _BinaryOp
    HAS_REFNX = True
except ImportError as ie:
    warnings.warn(f"{ie} compatible refnx package not installed.")

class VFP(BaseVFP):
    """
    Describes SLD profiles of interfaces from fronting to backing.
    
    SLD profiles are calculated by generating volume fraction profiles.
    These volume fraction profiles are cannot be negative anywhere,
    and the total volume fraction must be one everywhere.

    Parameters
    ----------
    nslds : tuple[ParameterLike] | list[ParameterLike]
        Nuclear scattering length densities of each material in the model.
    thicknesses : tuple[ParameterLike] | list[ParameterLike]
        Thicknesses of layers in the model. These control the
        midpoint-to-midpoint width of a layer's transition to and from other
        materials.
    roughnesses : tuple[ParameterLike] | list[ParameterLike]
        Roughnesses of layers. These control the width of interfaces between
        adjacent layers in the volume fraction profile.
    islds : tuple[ParameterLike] | list[ParameterLike] | None
        Imaginary scattering length densities of each layer
        within the model. Optional, defaults to None.
    mslds : tuple[ParameterLike] | list[ParameterLike] | None
        Magnetic scattering length densities of each layer within the volume
        fraction profile. Optional, defaults to None.
    spin_state : str
        Defines if slds should be calculated as nuclear (spin_state = 'none'),
        nuclear+magnetic (spin_state = 'up') or
        nuclear-magnetic (spin_state = 'down').
        Optional, defaults to 'none'.
    orientation : str
        Defines if incident radiation passed through fronting or backing.
        Through the fronting = (orientation = 'front'), through
        the backing = (orientation = 'back'). Useful for co-refinement
        of solid-liquid NR data with air-solid x-ray reflectometry data.
        Optional, defaults to 'front'.
    demaglocs : tuple[ParameterLike] | list[ParameterLike] | None
        If supplied, must either be a tuple/list of an even number of ParameterLike objects.
        The parameters declare the centre point of a Gaussian CDF.
        The parameters are consecutive, so the z location of parameter 2 will be
        parameter 1 value + parameter 2 value.
        Optional, defaults to None.
    demagwidths : tuple[ParameterLike] | list[ParameterLike] | None
        If supplied, must either be a tuple/list of an even number of ParameterLike objects.
        The parameters declare the width of a Gaussian CDF.
        Optional, defaults to None.
    sld_constraint : Callable | None
        User defined object used to handle SLD constraints between layers.
        Optional, defaults to None.
    max_delta_z : float
        Defines the approximate thickness of a microslice across the VFP.
        Optional, defaults to 0.5 angstrom.
    """

    def __init__(
        self,
        nslds: tuple[ParameterLike] | list[ParameterLike],
        thicknesses: (
            tuple[ParameterLike] | list[ParameterLike]
        ),
        roughnesses: (
            tuple[ParameterLike, str] | list[ParameterLike, str]
        ),
        islds: (
            None
            | tuple[ParameterLike]
            | list[ParameterLike]
        ) = None,
        mslds: (
            None
            | tuple[ParameterLike]
            | list[ParameterLike]
        ) = None,
        spin_state: Literal["none", "up", "down"] = "none",
        orientation: Literal["front", "back"] = "front",
        demaglocs: (
            None
            | tuple[ParameterLike]
            | list[ParameterLike]
        ) = None,
        demagwidths: (
            None
            | tuple[ParameterLike]
            | list[ParameterLike]
        ) = None,
        sld_constraint: Callable | None = None,
        max_delta_z: float = 0.5,
    ) -> None:
        # check some of the input pars & process roughnesses.
        checked_res = check_init_input(
            thicknesses,
            roughnesses,
            nslds,
            islds,
            mslds,
            demaglocs,
            demagwidths,
            spin_state,
            max_delta_z
        )
        roughnesses_alt, all_slds, demaglocs, demagwidths, conformal = checked_res
        nslds, islds, mslds = all_slds
        # set priv attrs and set via setup_attrs method.
        self._thicknesses = np.array(thicknesses)
        self._roughnesses = np.array(roughnesses_alt)
        self._demaglocs = np.array(demaglocs)
        self._demagwidths = np.array(demagwidths)
        self._nslds = np.array(nslds)
        self._mslds = np.array(mslds)
        self._islds = np.array(islds)
        self._conformal = np.array(conformal)
        # the following attrs are vfp specific.
        self._name = "VFP"
        self._max_delta_z = max_delta_z
        self._orientation = orientation
        self._spin_state = spin_state
        self._sld_constraint = sld_constraint
        
        # init VFPAttrs object.
        self.vfp_attrs
        
        # get all attrs of parent.
        super().__init__()
    
    @property
    def vfp_attrs(self) -> VFPAttributes:
        """
        Use private attributes setup in __init__ to create a VFPAttributes object.
        """
        attrs = VFPAttributes(nslds=self._nslds,
                              thicknesses=self._thicknesses,
                              roughnesses=self._roughnesses,
                              islds=self._islds,
                              mslds=self._mslds,
                              spin_state=self._spin_state,
                              orientation=self._orientation,
                              demaglocs=self._demaglocs,
                              demagwidths=self._demagwidths,
                              sld_constraint=self._sld_constraint,
                              max_delta_z=self._max_delta_z,
                              conformal=self._conformal,
                              name=self._name)    
        return attrs
    
    @classmethod
    def from_transform(
        cls, 
        vfp_attrs: dict[str, np.ndarray | str | float | None | Callable]
    ) -> VFP:
        """
        Transforms a dictionary of vfp attributes to a `VFP`.

        Parameters
        ----------
        vfp_attrs : dict[str, np.ndarray  |  str  |  float  |  None  |  Callable])
            Original vfp attributes.    

        Returns
        -------
        VFP
            VFP object instantiated from `vfp_attrs`.
        """
        # remove attrs not in input pars.
        attr_dict = copy.deepcopy(vfp_attrs)
        del attr_dict['conformal'], attr_dict['name']
        # transform all arrays to lists to be compatible with init.
        for key, seq in attr_dict.items():
            attr_dict[key] = (
                seq.astype(float).tolist() if isinstance(seq, np.ndarray) else seq
            )
        attr_dict['roughnesses'] = [
            val if ~np.isnan(val) else 'conformal' for val in attr_dict['roughnesses']
        ]
        vfp = VFP(**attr_dict)
        return vfp
    
    def _createparam(self):
        """
        Not needed for the standard VFP.
        """
        pass
    
    def set_parameter_prior(self) -> None:
        """
        Not required for this class.
        """
        pass

    def transform(
        self,
        wanted_vfp: Literal['refnx', 'refl1d']
    ) -> refnxVFP | refl1dVFP:
        """
        Transform VFP to a refnxVFP or refl1dVFP.
        
        Parameters
        ----------
        wanted_vfp : str
            The desired type of VFP: "refnx" or "refl1d".

        Raises
        ------
        ValueError
            If vfp_type is not "refl1d" or "refnx".

        Returns
        -------
        refnxVFP | refl1dVFP
            Transformed version of VFP.
        """
        if wanted_vfp not in ['refnx', 'refl1d']:
            raise ValueError(f'vfp_type must be either "refnx" or "refl1d".')

        transformed_vfp = init_specific_VFP(self, wanted_vfp, self.vfp_attrs.__dict__)    
        return transformed_vfp

if HAS_REFNX:
    class refnxVFP(Component, BaseVFP):
        """
        VFP for use with refnx.
        
        Can be initialised from parameters,
        or initialised from `vfp.vfp.VFP`.
        """
        def __init__(
            self,
            nslds: tuple[ParameterLike] | list[ParameterLike],
            thicknesses: (
                tuple[ParameterLike] | list[ParameterLike]
            ),
            roughnesses: (
                tuple[ParameterLike, str] | list[ParameterLike, str]
            ),
            islds: (
                None
                | tuple[ParameterLike]
                | list[ParameterLike]
            ) = None,
            mslds: (
                None
                | tuple[ParameterLike]
                | list[ParameterLike]
            ) = None,
            spin_state: Literal["none", "up", "down"] = "none",
            orientation: Literal["front", "back"] = "front",
            demaglocs: (
                None
                | tuple[ParameterLike]
                | list[ParameterLike]
            ) = None,
            demagwidths: (
                None
                | tuple[ParameterLike]
                | list[ParameterLike]
            ) = None,
            sld_constraint: Callable | None = None,
            max_delta_z: float = 0.5,
        ) -> None:
            # check some of the input pars & process roughnesses.
            checked_res = check_init_input(
                thicknesses,
                roughnesses,
                nslds,
                islds,
                mslds,
                demaglocs,
                demagwidths,
                spin_state,
                max_delta_z
            )
            roughnesses_alt, all_slds, demaglocs, demagwidths, conformal = checked_res
            nslds, islds, mslds = all_slds
            
            # the following attrs are vfp specific.
            self._name = "refnxVFP"
            self._max_delta_z = max_delta_z
            self._orientation = orientation
            self._spin_state = spin_state
            self._sld_constraint = sld_constraint
            
            # convert parameters to `refnxParameter`s
            thicknesses = self._createparam(thicknesses, "thicknesses")
            demaglocs = self._createparam(demaglocs, "demaglocs")
            demagwidths = self._createparam(demagwidths, "demagwidths")
            roughnesses = self._createparam(roughnesses_alt, "roughnesses")
            all_slds_map = map(self._createparam, all_slds, ["nsld", "isld", "msld"])
            # finally, remove any duplicates that may exist in the sld pars.
            nslds, islds, mslds = self._remove_duplicate_pars(par_map=all_slds_map)
            
            # set priv attrs and set via setup_attrs method.
            self._thicknesses = np.array(thicknesses)
            self._roughnesses = np.array(roughnesses)
            self._demaglocs = np.array(demaglocs)
            self._demagwidths = np.array(demagwidths)
            self._nslds = np.array(nslds)
            self._mslds = np.array(mslds)
            self._islds = np.array(islds)
            self._conformal = np.array(conformal)
            
            # init VFPAttrs object.
            self.vfp_attrs
            
            # get all attrs of parent.
            super().__init__()
        
        @property
        def vfp_attrs(self) -> VFPAttributes:
            """
            Use private attributes setup in __init__ to create a VFPAttributes object.
            """
            attrs = VFPAttributes(nslds=self._nslds,
                                  thicknesses=self._thicknesses,
                                  roughnesses=self._roughnesses,
                                  islds=self._islds,
                                  mslds=self._mslds,
                                  spin_state=self._spin_state,
                                  orientation=self._orientation,
                                  demaglocs=self._demaglocs,
                                  demagwidths=self._demagwidths,
                                  sld_constraint=self._sld_constraint,
                                  max_delta_z=self._max_delta_z,
                                  conformal=self._conformal,
                                  name=self._name)
            return attrs
        
        @property
        def parameters(self) -> Parameters:
            """
            Collates `refnxParameter`s in `self.vfp_attrs` in
            `refnx.analysis.Parameters`
            
            refnx uses this property when collating all varying parameters
            in a model. If a parameter that is passed to the `refnxVFP` is
            a function of other parameters, these will not automatically
            tracked. These should be passed to the auxiliary parameters of
            a refnx's global objective.
            
            Returns
            -------
            `refnx.analysis.Parameters`
            """
            # create a list of list of parameters
            p = Parameters(name=self._name)
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
                if lps.size > 0 # only accept non-empty par arrays.
            ]

            p.extend(
                [ps for lps in llps for ps in lps if ps is not None]
            )  # add defined parameters to parameter list.
            return p
        
        def set_parameter_prior(
            self, 
            prior: dict[str, dict[int, tuple[float, float] | PDF | Interval]]
            ) -> None:
            """
            Set bounds on refnxParameters in `self.vfp_attrs`.
            
            Use this function to set the prior for any parameters
            that are to be fit / sampled.
            
            The key names on the first level of the dictionary must
            match the names of the attributes in self.vfp_attrs. The
            key values of the second level of the dictionaries should
            match the indices of the parameters you wish to set priors for.
            
            Parameters
            ----------
            bounds : dict
                Nested dictionary of tuple[float, float],
                `refnx.analysis.PDF` or `refnx.analysis.Interval`
                to be applied to the refnxParameters in `self.vfp_attrs`. 
            
            Example
            -------
            >>> from vfp.vfp_refactor import refnxVFP
            >>> import scipy.stats as stats
            >>> thicknesses = (0, 20)
            >>> roughnesses = (2, 1)
            >>> nslds = (2.07, 3.47, 6.37) # Si, SiO2, D2O
            >>> refnx_vfp = refnxVFP(nslds, thicknesses, roughnesses)
            # lets set uniform priors on the thickness and roughness of SiO2
            # and set a gaussian prior on the sld of D2O with mean 6.37 & std 0.03
            >>> prior_dict = {'thicknesses' : {1 : (10, 30)},
                              'roughnesses' : {1 : (1, 4)},
                              'nslds' : {2 : PDF(stats.norm(6.37, 0.03))}
                              }
            >>> refnx_vfp.set_parameter_prior(prior=prior_dict)
            >>> refnx_vfp.vfp_attrs.thicknesses[1]
            Parameter(value=20.0, name='refnxVFP - thicknesses - layer 1', vary=True, bounds=Interval(lb=10.0, ub=30.0), constraint=None)
            >>> refnx_vfp.vfp_attrs.roughnesses[1]
            Parameter(value=1.0, name='refnxVFP - roughnesses - layer 1/backing', vary=True, bounds=Interval(lb=1.0, ub=4.0), constraint=None)
            >>> refnx_vfp.vfp_attrs.nslds[2]
            Parameter(value=6.37, name='refnxVFP - nsld - backing', vary=True, bounds=PDF(<scipy.stats._distn_infrastructure.rv_continuous_frozen object at 0x00000145898486E0>), constraint=None)
            """
            # take parameters from vfp_attrs as parameters property
            # is built from vfp_attrs. Take shallow copy, which will
            # update the attributes of vfp_attrs.
            pars_dict = self.vfp_attrs.__dict__
            
            # apply bounds nested dict to pars_dict:
            for keys, pars in prior.items():
                for idx, prior in pars.items():
                    pars_dict[keys][idx].bounds = prior
                    # set to vary if prior set.
                    pars_dict[keys][idx].vary = True

        def slabs(self, structure: Structure | None = None) -> np.ndarray:
            """
            Generate array representation of the refnx VFP as a 2d np.array using the
            thicknesses, SLDs and iSLDs of the microslabs which represent the SLD profile.

            Parameters
            ----------
            structure : refnx.reflect.Structure, optional
                The refnx.reflect.Structure hosting this VFP component. Defaults to None.

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
        
        @classmethod
        def from_transform(
            cls, 
            vfp_attrs: dict[str, np.ndarray | str | float | None | Callable]
        ) -> refnxVFP:
            """
            Transforms a dictionary of vfp attributes to a `refnxVFP`.

            Parameters
            ----------
            vfp_attrs : dict[str, np.ndarray  |  str  |  float  |  None  |  Callable])
                Original vfp attributes.    

            Returns
            -------
            refnxVFP
                A refnxVFP object instantiated from `vfp_attrs`.
            """
            # remove attrs not in input pars.
            attr_dict = copy.deepcopy(vfp_attrs)
            del attr_dict['conformal'], attr_dict['name']
            # transform all arrays to lists to be compatible with init.
            for key, seq in attr_dict.items():
                attr_dict[key] = seq.tolist() if isinstance(seq, np.ndarray) else seq
            attr_dict['roughnesses'] = [
                val if val is not None else 'conformal' for val in attr_dict['roughnesses']
            ]
            refnxvfp = refnxVFP(**attr_dict)
            return refnxvfp
        
        def transform(self, wanted_vfp: Literal['vfp', 'refl1d']) -> VFP | refl1dVFP:
            """
            Transform `refnxVFP` to a `VFP` or `refl1dVFP`.
        
            Parameters
            ----------
            wanted_vfp : str
                Either "vfp" or "refl1d".

            Raises
            ------
            ValueError
                If vfp_type is not "refl1d" or "vfp".

            Returns
            -------
            VFP | refl1dVFP
                Transformed version of VFP.
            """
            if wanted_vfp not in ['vfp', 'refl1d']:
                raise ValueError(f'vfp_type must be either "vfp" or "refl1d".')
        
            transformed_vfp = init_specific_VFP(self, wanted_vfp, self.vfp_attrs.__dict__)
            return transformed_vfp
        
        def _createparam(
            self,
            params: (
                tuple[ParameterLike | None]
                | list[ParameterLike | None]
            ),
            nameid: str,
        ) -> list[refnxParameter | None]:
            """
            Creates a list of refnxParameters.

            Parameters
            ----------
            param : tuple[ParameterLike | None] | list[ParameterLike | None]
                Sequence of parameter values.
            nameid : str
                The name of the collective parameters.

            Returns
            -------
                list[refnxParameter | None]
            """ 
            # create a list of strings that describe what each parameter is.
            # depends on which parameters we are dealing with.
            layer_strs = []
            if nameid in ("nsld", "msld", "isld"):
                for i in range(len(params)):
                    if i == 0:
                        layer_strs.append('fronting')
                    elif i == (len(params) - 1):
                        layer_strs.append('backing')
                    else:
                        layer_strs.append(f'layer {i}') 
            elif nameid == 'thicknesses':
                for i in range(len(params)):
                    layer_str = 'fronting' if i == 0 else f'layer {i}'
                    layer_strs.append(layer_str)
            elif nameid in ("demagwidths", "demaglocs"):
                for i in range(len(params)):
                    peak_str = f'peak {(i + 2) // 2}'
                    side_str = 'left' if i % 2 == 0 else 'right' 
                    layer_strs.append(peak_str + ' '+ side_str)
            elif nameid in ("roughnesses"):
                for i, par in enumerate(params):
                    if par is not None:
                        layer_before = 'fronting' if i == 0 else f'layer {i}'
                        layer_after = 'backing' if i == (len(params) - 1) else f'layer {i + 1}'
                        layer_strs.append(layer_before + '/' + layer_after)
                    else:
                        layer_strs.append(None)

            output = []
            for layer_str, par in zip(layer_strs, params):
                if isinstance(par, _BinaryOp):
                    warnings.warn(
                        """Pass nucSLD parameters that are only part of a parameter operation 
                            (i.e f(p1, p2) = p1 + p2) to the auxiliary parameters argument of the objective."""
                    )
                else:
                    if nameid == 'roughnesses':
                        if par is not None:
                            output.append(
                                possibly_create_parameter(
                                    par, 
                                    name=f"{self._name} - {nameid} - {layer_str}")
                            )
                        else:
                            output.append(None)
                    else:
                        output.append(
                            possibly_create_parameter(
                                par, name=f"{self._name} - {nameid} - {layer_str}"
                            )
                        )

            return output         

if HAS_REFL1D:
    class refl1dVFP(Layer, BaseVFP):
        """
        VFP for use with refl1d.
        
        Can be initialised from parameters,
        or initialised from `vfp.vfp.VFP`.
        """
        def __init__(
            self,
            nslds: tuple[ParameterLike] | list[ParameterLike],
            thicknesses: (
                tuple[ParameterLike] | list[ParameterLike]
            ),
            roughnesses: (
                tuple[ParameterLike, str] | list[ParameterLike, str]
            ),
            islds: (
                None
                | tuple[ParameterLike]
                | list[ParameterLike]
            ) = None,
            mslds: (
                None
                | tuple[ParameterLike]
                | list[ParameterLike]
            ) = None,
            spin_state: Literal["none", "up", "down"] = "none",
            orientation: Literal["front", "back"] = "front",
            demaglocs: (
                None
                | tuple[ParameterLike]
                | list[ParameterLike]
            ) = None,
            demagwidths: (
                None
                | tuple[ParameterLike]
                | list[ParameterLike]
            ) = None,
            sld_constraint: Callable | None = None,
            max_delta_z: float = 0.5,
        ) -> None:
            # check some of the input pars & process roughnesses.
            checked_res = check_init_input(
                thicknesses,
                roughnesses,
                nslds,
                islds,
                mslds,
                demaglocs,
                demagwidths,
                spin_state,
                max_delta_z
                )
            roughnesses_alt, all_slds, demaglocs, demagwidths, conformal = checked_res
            nslds, islds, mslds = all_slds
            # the following attrs are vfp specific.
            self._name = "refl1dVFP"
            self._max_delta_z = max_delta_z
            self._orientation = orientation
            self._spin_state = spin_state
            self._sld_constraint = sld_constraint
            
            # convert parameters to `bumpsParameter`s
            thicknesses = self._createparam(thicknesses, "thicknesses")
            demaglocs = self._createparam(demaglocs, "demaglocs")
            demagwidths = self._createparam(demagwidths, "demagwidths")
            roughnesses = self._createparam(roughnesses_alt, "roughnesses")
            all_slds_map = map(self._createparam, all_slds, ["nsld", "isld", "msld"])
            # finally, remove any duplicates that may exist in the sld pars.
            nslds, islds, mslds = self._remove_duplicate_pars(par_map=all_slds_map)
            
            # set priv attrs and set via setup_attrs method.
            self._thicknesses = np.array(thicknesses)
            self._roughnesses = np.array(roughnesses)
            self._demaglocs = np.array(demaglocs)
            self._demagwidths = np.array(demagwidths)
            self._nslds = np.array(nslds)
            self._mslds = np.array(mslds)
            self._islds = np.array(islds)
            self._conformal = np.array(conformal)
            
            # init VFPAttrs object.
            self.vfp_attrs
            
            # get all attrs of parent.
            super().__init__()
            
            # refl1d needs total thickness of the vfp as attr
            # at the beginning and throughout fitting.
            _, _, thicks = self.process_model()
            self.thickness = bumpsParameter(
                thicks.sum(), name=f"{self._name} - total thickness"
            )
            
        def set_parameter_prior(
            self, 
            prior: dict[str, dict[int, tuple[float, float]]]
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
            bounds : dict
                Nested dictionary of tuple[float, float],
                to be applied to the `bumpsParameter`s in `self.vfp_attrs`. 
            
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
            # TODO: how can I get this to work with scipy.stats.distributions?
            
            # take parameters from vfp_attrs as parameters property
            # is built from vfp_attrs. Take shallow copy, which will
            # update the attributes of vfp_attrs.
            pars_dict = self.vfp_attrs.__dict__
            
            # apply bounds nested dict to pars_dict:
            for keys, pars in prior.items():
                for idx, prior in pars.items():
                    low, high = prior
                    pars_dict[keys][idx].range(low, high)
        
        @property
        def vfp_attrs(self) -> VFPAttributes:
            """
            Use private attributes setup in __init__ to create a VFPAttributes object.
            """
            attrs = VFPAttributes(nslds=self._nslds,
                                  thicknesses=self._thicknesses,
                                  roughnesses=self._roughnesses,
                                  islds=self._islds,
                                  mslds=self._mslds,
                                  spin_state=self._spin_state,
                                  orientation=self._orientation,
                                  demaglocs=self._demaglocs,
                                  demagwidths=self._demagwidths,
                                  sld_constraint=self._sld_constraint,
                                  max_delta_z=self._max_delta_z,
                                  conformal=self._conformal,
                                  name=self._name)
            return attrs
            
        def to_dict(self) -> dict[str | str, np.ndarray]:
            """
            Returns a dict repr of the VFPattributes for use with `bumps.parameters`.
            Used when saving a refl1d model details as a .json file.

            Returns
            -------
            dict[str | str, np.ndarray]
                repr of the refl1d.vfp_attrs.
            """
            
            return to_dict(self.vfp_attrs.__dict__)

        def layer_parameters(self) -> dict:
            """
            Get the fitting parameters of the refl1dVFP.

            Returns
            -------
            dictionary
                dict of parameters with equal to the name of the VFP arguments.
            """
            vfp_dict = self.to_dict()
            p = copy.deepcopy(vfp_dict)
            # remove those that are not to be fit.
            del p['spin_state'], p['orientation'], p['sld_constraint'], p['max_delta_z'], p['conformal'], p['name']
            # remove empty lists.
            p = {key : val for key, val in p.items() if val}
            return p

        def render(self,
                   probe: NeutronProbe,
                   slabs: Microslabs) -> None:
            """
            Appends the microslice thickness, SLDs and iSLDs to the
            Microslabs object passed to the render function of the VFP
            by refl1d's Experiment object. Also updates the self.thickness
            value of the VFP.

            Parameters
            ----------
            probe : refl1d.probe.NeutronProbe
                Passed to render functions of refl1d.layers but not used here.
            slabs : refl1d.profile.Microslabs
                Object which has rho, irho, w and sigma properties.
            """
            # use the process method of the BaseVFP class to
            # return total slds, islds and thicknesses of each slab
            slds, islds, thicks = self.process_model()

            # update the self.thickness variable.
            self.thickness.value = thicks.sum()

            # now append slds, islds and thicks to slabs.
            for i in range(0, len(thicks)):
                slabs.append(rho=slds[i], irho=islds[i], w=thicks[i], sigma=0)
        
        @classmethod
        def from_transform(
            cls, 
            vfp_attrs: dict[str, np.ndarray | str | float | None | Callable]
        ) -> refl1dVFP:
            """
            Transforms a dictionary of vfp attributes to a `refl1dVFP`.

            Parameters
            ----------
            vfp_attrs : dict[str, np.ndarray  |  str  |  float  |  None  |  Callable])
                Original vfp attributes.    

            Returns
            -------
            refl1dVFP
                A refl1dVFP object instantiated from `vfp_attrs`.
            """
            # remove attrs not in input pars.
            attr_dict = copy.deepcopy(vfp_attrs)
            del attr_dict['conformal'], attr_dict['name']
            # transform all arrays to lists to be compatible with init.
            for key, seq in attr_dict.items():
                attr_dict[key] = seq.tolist() if isinstance(seq, np.ndarray) else seq
            attr_dict['roughnesses'] = [
                val if val is not None else 'conformal' for val in attr_dict['roughnesses']
            ]
            refnxvfp = refl1dVFP(**attr_dict)
            return refnxvfp
        
        def transform(self, wanted_vfp: Literal['vfp', 'refnx']) -> VFP | refnxVFP:
            """
            Transform `refl1dVFP` to a `VFP` or `refnxVFP`.
        
            Parameters
            ----------
            wanted_vfp : str
                Either "vfp" or "refnx".

            Raises
            ------
            ValueError
                If vfp_type is not "refnx" or "vfp".

            Returns
            -------
            VFP | refl1dVFP
                Transformed version of VFP.
            """
            if wanted_vfp not in ['vfp', 'refnx']:
                raise ValueError(f'vfp_type must be either "vfp" or "refnx".')
        
            transformed_vfp = init_specific_VFP(self, wanted_vfp, self.vfp_attrs.__dict__)
            return transformed_vfp
                
        def _createparam(
            self,
            params: (
                tuple[ParameterLike | None]
                | list[ParameterLike | None]
            ),
            nameid: str,
        ) -> list[bumpsParameter | None]:
            """
            Creates a list of bumpsParameters.

            Parameters
            ----------
            param : tuple[ParameterLike | None] | list[ParameterLike | None]
                Sequence of parameter values.
            nameid : str
                The name of the collective parameters.

            Returns
            -------
                list[bumpsParameter | None]
            """ 
            # create a list of strings that describe what each parameter is.
            # depends on which parameters we are dealing with.
            layer_strs = []
            if nameid in ("nsld", "msld", "isld"):
                for i in range(len(params)):
                    if i == 0:
                        layer_strs.append('fronting')
                    elif i == (len(params) - 1):
                        layer_strs.append('backing')
                    else:
                        layer_strs.append(f'layer {i}') 
            elif nameid == 'thicknesses':
                for i in range(len(params)):
                    layer_str = 'fronting' if i == 0 else f'layer {i}'
                    layer_strs.append(layer_str)
            elif nameid in ("demagwidths", "demaglocs"):
                for i in range(len(params)):
                    peak_str = f'peak {(i + 2) // 2}'
                    side_str = 'left' if i % 2 == 0 else 'right' 
                    layer_strs.append(peak_str + ' '+ side_str)
            elif nameid in ("roughnesses"):
                for i, par in enumerate(params):
                    if par is not None:
                        layer_before = 'fronting' if i == 0 else f'layer {i}'
                        layer_after = 'backing' if i == (len(params) - 1) else f'layer {i + 1}'
                        layer_strs.append(layer_before + '/' + layer_after)
                    else:
                        layer_strs.append(None)

            output = []
            for layer_str, par in zip(layer_strs, params):
                if nameid == 'roughnesses':
                    if par is not None:
                        # I believe bumpsParameter.default acts like possibly_create_parameter.
                        output.append(
                            bumpsParameter.default(
                            par, 
                            name=f"{self._name} - {nameid} - {layer_str}"
                            )
                        )
                    else:
                        output.append(None)
                else:
                    if isinstance(par, Expression):
                        warnings.warn(
                        "If msld / isld parameters are part of a function" 
                        " (i.e f(p1, p2) = p1 + p2), they must be of type"
                        " bumps.parameter.Parameter. Do not use material or SLD objects."
                        )
                        output.extend(par.parameters())
                    else:
                        output.append(
                            bumpsParameter.default(
                            par, 
                            name=f"{self._name} - {nameid} - {layer_str}"
                            )
                        )
            return output

def init_specific_VFP(
    original_vfp: VFP | refnxVFP | refl1dVFP,
    vfp_type: Literal['vfp', 'refnx', 'refl1d'],
    vfp_dict: dict[str, np.ndarray  |  str  |  float  |  None  |  Callable]
) -> VFP | refnxVFP | refl1dVFP:
    """
    Helper function to load a type of VFP.
    
    Called by transform methods of child classes of `vfp.basevfp.BaseVFP`.
    
    Parameters
    ----------
    original_vfp : VFP | refnxVFP | refl1dVFP
        The original vfp to transform to a different type of VFP.
    vfp_type : str
        Type of the desired VFP type.
    vfp_dict : dict[str, np.ndarray  |  str  |  float  |  None  |  Callable]
        Original VFP attributes as a dictionary
        
    Returns
    -------
    refnxVFP | refldVFP | VFP
    """
    if vfp_type == 'refnx':
        if HAS_REFNX:
            target_vfp = refnxVFP
        else:
            raise ValueError(f'Target vfp is a refnxVFP, and refnx is not an available dependency.')    
    
    elif vfp_type == 'refl1d':
        if HAS_REFL1D:
            target_vfp = refl1dVFP
        else:
            raise ValueError(f'Target vfp is a refl1dVFP, and refl1d is not an available dependency.')          
    
    elif vfp_type == 'vfp':
        target_vfp = VFP
    
    else:
        raise ValueError("model_type must be 'vfp' / 'refnxVFP' / 'refl1dVFP'.")

    # if asked for the same type as original_vfp just return original_vfp.
    if isinstance(target_vfp, type(original_vfp)):
        warnings.warn(f'Returning the original vfp as target is the same.')
        return original_vfp
            
    final_vfp = target_vfp.from_transform(vfp_dict)
    
    return final_vfp