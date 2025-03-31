# standard
from __future__ import annotations
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Literal

# third party
import numpy as np
import matplotlib

# this module
from vfp.calc import calc_dzs, calc_zeds, init_demag, integrate_vfp, calc_vfp
from vfp.plotting import model_plot
from vfp.typing import ParameterLike


@dataclass
class VFPAttributes:
    """
    A dataclass to hold reference to the attributes of child classes that inherit BaseVFP.
    These attributes are set by the __init__ method of a VFP class.
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
    sld_constraint: None | Callable
    max_delta_z: float
    conformal: np.ndarray
    name: str


class BaseVFP(ABC):
    """
    Handles common functions of VFP.
    
    Process is the main function.
    """
    def __init__(self) -> None:
        # create vfp model.
        self.process_model()

    def __repr__(self) -> str:
        """
        Returns simple string description of the VFP.
        
        Currently not called by refnxVFP or refl1dVFP.

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
        
        Main function of the `BaseVFP`.
        Calculates the length of the VFP, the thicknesses of each microslice
        and calculates the sld of each microslice.
        Returns the coherent and imaginary sld values for each microslice and
        the thickness of each microslice given orientation of sample.

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
        # update tuple variants of parameters.
        self._tuple_pars()

        # calc z spectrum
        zeds = calc_zeds(self.tup_roughs, self.tup_thicks, self.vfp_attrs.max_delta_z)
        zstart, zend, points = zeds[0], zeds[-1], zeds.size

        # convert to tuple for caching.
        self.zeds = self._arrtotuple(zeds)
        
        # get the combined nuclear+/-magnetic SLDs (coherent) and the imaginary SLDs.
        slds_micro, islds_micro = self.get_slds()

        # get the thickness of each microslab.
        # uses caching and tuples defined above.
        self.dz = calc_dzs(zstart, zend, points, self.indices)

        # if VFP.orientation = back --> slabs will have same thickness,
        # just in reverse order
        if self.vfp_attrs.orientation == "back":
            self.dz = self.dz[::-1]

        # get the average between each coherent and imaginary SLD value.
        average_slds, average_islds = (
            0.5 * np.diff(slds) + slds[:-1] for slds in [slds_micro, 
                                                         islds_micro]
        )

        # init arrays for final SLDs.
        return_slds, return_islds = [
            np.ones(slds.size + 1) for slds in [average_slds,
                                                average_islds]
        ]

        if self.vfp_attrs.orientation == "front":
            # fill all but last with average SLDs.
            return_slds[:-1] = return_slds[:-1] * average_slds
            return_islds[:-1] = return_islds[:-1] * average_islds
            # now set the final sld value to those from the micro arrays.
            return_slds[-1] = slds_micro[-1]
            return_islds[-1] = islds_micro[-1]

        elif self.vfp_attrs.orientation == "back":
            # do the same but backwards for back orientations.
            return_slds[1:] = return_slds[1:] * average_slds[::-1]
            return_islds[1:] = return_islds[1:] * average_islds[::-1]
            # now set the final sld value to those from the micro arrays.
            return_slds[0] = slds_micro[-1]
            return_islds[0] = islds_micro[-1]

        return return_slds, return_islds, self.dz

    def get_slds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate slds via generation of VFP.

        Initially, the VFP is calculated, then it is reduced via
        `self.init_demag`. slds are calculated and then summed to give
        a coherent slds (nuclear or nuclear +/- magnetic dependent on 
        `self.spin_state`) and imaginary slds.

        Returns
        -------
        np.ndarray
            coherent slds (nuclear or nuclear +/- magnetic) (1d).
        np.ndarray
            imaginary slds (1d).
        """

        # calculate the volume fraction profiles of the layers in the interface.
        self.vfp = calc_vfp(
            self.tup_roughs, self.tup_thicks, self.zeds, tuple(self.vfp_attrs.conformal)
        )

        # using vfp from the above function,
        # calculate reduced volume fraction and magnetic profiles.
        self.red_vfp, self.demagf, idx, _ = init_demag(
            self.tup_demag_locs,
            self.tup_demag_widths,
            self.tup_mslds,
            self.zeds,
            self._arrtotuple(self.vfp),
        )

        self.indices = self._arrtotuple(idx)

        # calculate the SLD valus across reduced VFPs.
        all_slds = self.calc_slds()
        
        # now total the nuclear and magnetic SLDs on given contrast.
        # tot sld must either be addition or subtraction.
        if self.vfp_attrs.spin_state == "none":
            tot_sld = all_slds[0]
        elif self.vfp_attrs.spin_state == "down":
            tot_sld = all_slds[0] - all_slds[2]
        elif self.vfp_attrs.spin_state == "up":
            tot_sld = all_slds[0] + all_slds[2]

        return tot_sld, all_slds[1]

    def calc_slds(
        self, 
        reduced: bool = True
    ) -> np.ndarray:
        """
        Calculates coherent and imaginary slds.
        
        Slds are nuc, mag and imaginary.
        Can be calculated with reduced or full VFP.

        Parameters
        ----------
        reduced : bool
            If True/False, calculates the reduced/full SLD profiles

        Returns
        -------
        np.ndarray
            Three sld contributions across three rows as function of
            `self.zeds`. Coherent sld, imaginary sld, magnetic sld. 
            Shape = (3, z.size)
        """
        # if sld_constraint is not None, update self.nucSLDs depending on constraint.
        if self.vfp_attrs.sld_constraint:
            layer_indices = self.vfp_attrs.sld_constraint.layer_choices()
            
            integrals = integrate_vfp(
                self.zeds,
                self.indices,
                self._arrtotuple(self.red_vfp),
                tuple(layer_indices)
            )
            # user defines a class with a callable,
            # which returns an idx for modifying a particular SLD value.
            layer_loc, sld = self.vfp_attrs.sld_constraint(integrals)
            self.vfp_attrs.nslds[layer_loc] = sld

        demagf = self.demagf if reduced else self.vfs_for_display()[2]
        
        # get float values from the Parameters in the attrs arrays.
        sld_values = [sld_pars.astype(float) for sld_pars in [self.vfp_attrs.nslds,
                                                              self.vfp_attrs.mslds,
                                                              self.vfp_attrs.islds]]
        
        # calc nuclear_slds from red_vfps:
        nuc_and_i_slds = [
            self.red_vfp.T * sld_val for sld_val in [sld_values[0], 
                                                     sld_values[2]]
        ]

        # calc magnetic_slds
        sldm_layers = demagf.T * sld_values[1]
        
        slds_over_z = [
            np.sum(arr, axis=1) for arr in nuc_and_i_slds + [sldm_layers]
        ]
        # row 0 = nsld, row 1 = isld, row 2 = msld
        sum_slds = np.vstack(slds_over_z)

        return sum_slds

    def vfs_for_display(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get volume fraction profile for plotting
        
        Function useful for plotting:
        1. Reduced VF profile (defines nuclear SLD profile)
        2. Reduced magnetic composition profile (defines magnetic SLD profile)

        Returns
        -------
        reduced_VFP : np.array
            Shape = (Nlayers, len(z) - len(self.indices))
        reduced_magcomp : np.array
            Shape = (Nlayers, len(z) - len(self.indices))
        """
        # update the model. Captures instances where parameters have changed.
        self.process_model()

        reduced_VFP, reduced_magcomp, _, _ = init_demag(
            self.tup_demag_locs,
            self.tup_demag_widths,
            self.tup_mslds,
            self.zeds,
            self._arrtotuple(self.vfp),
        )

        if self.vfp_attrs.orientation == "back":
            reduced_VFP = reduced_VFP[::-1] # reverse order.
            reduced_magcomp = reduced_magcomp[::-1]

        return reduced_VFP, reduced_magcomp

    def z_and_sld(
        self, 
        reduced: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Plot slds calculated from the VFP.
        
        Returns z values from self.calc_zeds() and also returns
        sld values from self.calc_slds() calculated from the VFP.

        Parameters
        ----------
        reduced : Boolean
            If False/True, will return full/reduced zs and slds.

        Returns
        -------
        np.ndarray
            z distance from fronting interface either reduced or full.
        np.ndarray
            slds (2d), coherent, imaginary, magnetic. Either reduced or full.
        """
        self.process_model() # update the model.
        zeds = np.array(self.zeds)
        if self.vfp_attrs.orientation == "front":
            if reduced:
                slds = self.calc_slds()
                z = np.delete(zeds, self.indices)
            else:
                slds = self.calc_slds(reduced=False)
                z = zeds

        # if reverse orientation, subtract length of inteface & flip.
        if self.vfp_attrs.orientation == "back":
            offset = np.sum(self.tup_thicks)
            if reduced:
                slds = self.calc_slds()
                z = -(np.delete(zeds, self.indices) - offset)
            else:
                slds = self.calc_slds(reduced=False)
                z = -(zeds - offset)

        return z, slds

    def sld_offset(self) -> float:
        """
        Offset to apply to sld profile to set at appropriate z coordinate.

        Add the return value to the z values of the sld profile to set the
        sld profile to the correct location.

        Returns
        -------
        float
            Offset to add to z values of an sld_profile.
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
                np.max(np.sum(self.tup_thicks) + 4 * np.array(self.tup_roughs)) + 5
            )
            # round up like zend
            zend_of_vfprofile = np.ceil(
                zend_of_vfprofile_nr * (1 / self.vfp_attrs.max_delta_z)
            ) / (1 / self.vfp_attrs.max_delta_z)

            # sld_profile zend defined by -5 + last slab location + 4 * backing roughness.
            # zend_of_vfprofile replicates the 4 * backing roughness part.
            # Then 5 + last microslice thickness covers the -5 + last slab location part.
            zend_front = self.dz[-1] + zend_of_vfprofile
            sldprof_offset = -(zend_front - np.sum(self.tup_thicks))

        return sldprof_offset

    def plot(
        self,
        points: int = 50,
        microslice_sld: bool = True,
        total_sld: bool = False,
        total_vf: bool = True,
    ) -> tuple[
        matplotlib.figure.Figure, np.ndarray[matplotlib.axes._axes.Axes]
    ]:
        """
        Produces a three axis figure to visualise the VFP.
        
        Top plot = nsld / msld / isld
        Middle plot = volume fraction profiles
        Bottom plot = surface profiles

        Parameters
        ----------
        points : integer
            Number of points to simulate across the surfaces.
        microslice_sld : boolean
            If True, will return SLD profiles after microslicing the profile.
        total_sld : boolean
            If True, will return SLD+ or SLD- profiles. If false, the SLDn and SLDm parts
            will be plotted seperately.
        total_vf : boolean
            If True, will plot the sum of all layers' volume fractions.

        Returns
        -------
        fig, ax
            matplotlib.pyplot figure and axes objects.
        """

        # update the model. Captures instances where parameters have changed.
        self.process_model()

        fig, ax = model_plot(
            vfp=self,
            points=points,
            microslice=microslice_sld,
            total_sld=total_sld,
            total_vf=total_vf,
        )

        return fig, ax

    def _tuple_pars(self) -> None:
        """
        Converts attributes to tuples for the purposes of hashing.
        """
        self.tup_thicks = tuple(self.vfp_attrs.thicknesses.astype(float))
        self.tup_demag_locs = tuple(self.vfp_attrs.demaglocs.astype(float))
        self.tup_demag_widths = tuple(self.vfp_attrs.demagwidths.astype(float))
        self.tup_mslds = tuple(self.vfp_attrs.mslds.astype(float))
        
        # we need to put a hashable dummy value into the roughnesses.
        self.tup_roughs = tuple(
            float(par) if par is not None else 1 for par in self.vfp_attrs.roughnesses
        )

    def _arrtotuple(self, arr: np.ndarray) -> tuple:
        """
        Takes 1D/2D arrays and returns a tuple/nested tuple
        for the purposes of caching.

        Parameters
        ----------
        arr : np.ndarray
            Array to convert to tuples.
        """
        if arr.ndim == 1:
            return tuple(val for val in arr)

        elif arr.ndim == 2:
            return tuple([tuple([float(val) for val in row]) for row in arr])
    
    def _init_vfp_attrs(
        self, 
        arr_attrs: list[
            list[ParameterLike | None]
            | tuple[ParameterLike]
            | list[int]
        ],
        other_attrs: list[
            Literal['front', 'back'], 
            Literal['none', 'up', 'down'],
            Callable | None,
            float
        ],
        name: str
    ) -> VFPAttributes:
        """
        Setup object to hold reference to input parameters.
        """
        thicknesses, roughnesses, nslds, islds, mslds, demaglocs, demagwidths, conformal = list(map(np.array, arr_attrs))
        orientation, spin_state, sld_constraint, max_delta_z = other_attrs
        attrs = VFPAttributes(nslds=nslds,
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
                              name=name)    
        return attrs
    
    @property
    @abstractmethod
    def vfp_attrs(self) -> VFPAttributes:
        """
        Abstract property to implement in a child of BaseVFP to use the VFPAttribute dataclass.
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
    
    @abstractmethod
    def _createparam(self):
        """
        Abstract method that should be implemented to return ParameterLike
        objects for fitting software. Not required for standard VFP.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError


def check_init_input(thicknesses: tuple[ParameterLike] | list[ParameterLike], 
                     roughnesses: tuple[ParameterLike, str] | list[ParameterLike, str], 
                     nslds: tuple[ParameterLike] | list[ParameterLike], 
                     islds: tuple[ParameterLike] | list[ParameterLike] | None, 
                     mslds: tuple[ParameterLike] | list[ParameterLike] | None, 
                     demaglocs: tuple[ParameterLike] | list[ParameterLike] | None, 
                     demagwidths: tuple[ParameterLike] | list[ParameterLike] | None,
                     spin_state: Literal["none", "up", "down"],
                     max_delta_z: float) -> tuple[list[ParameterLike | None], list[list[ParameterLike]], list[ParameterLike] | tuple[ParameterLike], list[ParameterLike] | tuple[ParameterLike], list[int]]:
    """
    Check over the input values to the VFP classes.
    
    Helper function as all types of VFP require similar input structure.
    """
    # now check given parameters
    if not demaglocs: # look for empty lists or None.
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
            "The number of the demagnetisation locations and widths must be 0 or even."
        )

    if len(thicknesses) != len(roughnesses):
        raise ValueError(
            """The number of roughness parameters must match 
                the number of thickness parameters."""
        )
    
    # check roughness value below 0. Use float for compat with bumpsParameter.
    if any([float(rough_val) <= 0 for rough_val in roughnesses if not isinstance(rough_val, str)]):
        raise ValueError(f'Roughness parameters must be > 0 ')

    if len(nslds) != len(thicknesses) + 1:
        raise ValueError(
            """The number of supplied SLD values must be 1 greater 
                than the number of thickness parameters."""
        )

    # init a list of where conformal interfaces are:
    conformal = []
    for roughness in roughnesses:
        # cannot use a type alias in isinstance so use its value attr
        if isinstance(roughness, (str, ParameterLike.__value__)):
            if isinstance(roughness, str) and roughness == "conformal":
                conformal.append(1)

            elif isinstance(roughness, str) and roughness != "conformal":
                raise ValueError(
                    "Any string within the roughness list must read 'conformal'."
                )

            else:
                conformal.append(0)

        else:
            raise ValueError(
                """The entries within the roughness list must be a float, interger, 
                    refnx.analysis.parameter or a string == 'conformal'."""
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
        if spin_state == 'none':
            raise ValueError("If mslds is defined, the spin state passed to the VFP must be 'up' or 'down'.")


    # do same None check for islds
    if islds is None:
        islds = [0] * (len(thicknesses) + 1)

    # check the lengths of SLD arrays are the same.
    all_slds = [nslds, islds, mslds]
    if not all(len(sld) == len(all_slds[0]) for sld in all_slds):
        raise ValueError("The number of supplied nuclear, magnetic and imaginary SLD values must be the same.")

    # Simple warning on max_delta_z being too low.
    if any((float(rough) < 2 * max_delta_z for rough in roughnesses_alt if rough is not None)):
        warnings.warn(
        "The microslice thickness is less than twice some of the"
        " roughness parameters. Consider reducing the max_delta_z of the VFP."
        )

    return roughnesses_alt, all_slds, demaglocs, demagwidths, conformal