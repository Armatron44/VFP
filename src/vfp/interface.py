import numpy as np
import warnings
from basevfp import BaseVFP

from refnx.reflect import Component
from refnx.analysis import Parameter, Parameters, possibly_create_parameter
from refl1d.model import Layer
from bumps.parameter import Parameter as bumpsParameter, to_dict

warnings.simplefilter("once")

class refnxVFP(Component, BaseVFP):
    """
    How does this work?
    -------------------

    In ReflectModel (or MixedReflectModel), the reflectivity function is used to calculate the generative 
    (the simulated reflectivity) for a given set of parameters. The generative is used when fitting a 
    dataset, when carrying out posterior sampling & when estimating the model evidence.
    
    The reflectivity function requires a slab representation 
    (an array of slab (shape = 2+N, 4 where N is number of layers) parameters - thicknesses, roughnesses etc) 
    of the structure.
    
    This slab representation is returned by the structure.slab() method, which uses the slab method of 
    each component within a structure to return a concatenated array of slabs.
    
    The VFP uses thickness and roughness parameters to build a volume fraction profile of the interface.
    Using SLD parameters (nuclear, magnetic, imaginary), the SLD profile of the interface is calculated
    from the volume fraction profile.
    The SLD profile is a microslabbed approximation of the continuous interface.
    The size of the microslabs is default at 0.5 Å.
    
    The above workflow requires that the VFP component has a slab method which will return an array of
    microslabs to the ReflectModel object. In the slab method below we use the __call__ method of the 
    VFP component to do this calculation.
    
    Parameters
    ----------
    nucSLDs : array of floats or Parameters    
              Nuclear scattering length densities of each layer within the volume fraction profile.
    thicknesses : tuple of floats or Parameters
                  Thicknesses of layers - These control the midpoint-to-midpoint width of a layer's volume fraction
                  profile.
    roughnesses : tuple of floats or Parameters
                  Roughnesses of layers - These control the width of interfaces between two adjacent layers in the
                  volume fraction profile.
    nuciSLDs : array of floats or Parameters - optional   
               Imaginary scattering length densities of each layer within the volume fraction profile.
    magSLDs : array of floats or Parameters - optional    
              Magnetic scattering length densities of each layer within the volume fraction profile.
    spin_state : string - optional
                 string used to define if SLDs should be calculated as nuclear (spin_state = 'none'), 
                 nuclear+magnetic (spin_state = 'up') or nuclear-magnetic (spin_state = 'down')
    orientation : string - optional
                  string used to define if incident radiation pass through fronting or backing, 
                  Through the fronting = (orientation = 'front'), through the backing = (orientation = 'back').
    demaglocs : list of Parameters - optional - but if supplied must either be a list of 2 or 4 parameters.
                The parameters declare the centre point of a Gaussian CDF.
                The parameters are consecutive, so the z location of parameter 2 will be parameter 1 value 
                + parameter 2 value.
    demagwidths : list of Parameters - optional - but if supplied must either be a list of 2 or 4 parameters.
                  The parameters declare the width of a Gaussian CDF.
                  Must either be a list of 2 or 4 parameters.
    max_delta_z : float - optional
                  Defines the approximate thickness of a microslice across the VFP.              
    """

    def __init__(
        self,
        nucSLDs,
        thicknesses,
        roughnesses,
        nuciSLDs=None,
        magSLDs=None,
        spin_state='none',
        orientation='front',
        demaglocs=[],
        demagwidths=[],
        SLD_constraint=None,
        max_delta_z=0.5
    ):
        
        # init the variables in the Component class (importantly, the _interfaces variable.)
        super().__init__()
        
        # the following options are not refnx or refl1d parameters.
        self.name = 'refnx VFP'
        self.max_delta_z = max_delta_z
        self.orientation = orientation
        self.spin_state = spin_state
        self.SLD_constraint = SLD_constraint
        
        if len(demaglocs) != len(demagwidths):
            raise ValueError(
                "The number of the demagnetisation locations must be equal to the number of the demagnetisation widths."
            )

        while len(demaglocs) not in [0, 2, 4]:
            raise ValueError(
                "The number of the demagnetisation locations (and widths) must be 0, 2 or 4."
            )

        if len(thicknesses) != len(roughnesses):
            raise ValueError(
                "The number of roughness parameters must match the number of thickness parameters."
            )
        
        if len(nucSLDs) != len(thicknesses) + 1:
            raise ValueError(
                "The number of supplied SLD values must be 1 greater than the number of thickness parameters."
            )
        
        # init a list of where conformal interfaces are:
        conformal = []
        
        for i in roughnesses:
            if isinstance(i, (float, int, str, Parameter)):
                if isinstance(i, str) and i == 'conformal':
                    conformal.append(1)

                elif isinstance(i, str) and i != 'conformal':
                    raise ValueError(
                        "Any string within the roughness list must read 'conformal'."
                    )
                
                else:
                    conformal.append(0)

            else:
                raise ValueError(
                    "The entries within the roughness list must be a float, interger, refnx.analysis.parameter or a string == 'conformal'."
                )
            
        # can only have conformal roughnesses with more than one interface.
        # therefore the first interface cannot be conformal
        if np.any(conformal):
            idx_where_first_one = np.asarray(np.array(conformal) == 1).nonzero()[0][0] # quicker than where and nonzero!
            
            if idx_where_first_one > 0:
                self.conformal = conformal
            
            else:
                raise ValueError(
                    "Cannot specify the first interface to be conformal."
                )
        
        else:
            self.conformal = np.zeros(len(thicknesses))

        # where conformal in roughnesses, replace value with None
        roughnesses_alt = [None if i == 'conformal' else i for i in roughnesses]
        
        # if specified, the following should be refnx parameters.
        self.thicknesses = [possibly_create_parameter(j, name=f"{self.name} - Layer{i} - thick") for i, j in enumerate(thicknesses)]
        self.demaglocs = [possibly_create_parameter(j, name=f"{self.name} - demaglocs{i}") for i, j in enumerate(demaglocs)]
        self.demagwidths = [possibly_create_parameter(j, name=f"{self.name} - demagwidths{i}") for i, j in enumerate(demagwidths)]
        
        # create parameters for any roughness values that aren't parameters
        self.roughnesses_p = [possibly_create_parameter(j, name=f"{self.name} - Layer{i}/Layer{i+1} - rough") for i, j in enumerate(roughnesses_alt) if not j is None]

        # now we have to create a list of the above roughness parameters, and where they are not, replace with 1 as a "dummy value".
        # the dummy value will have no impact on the resulting VFP, but a finite non-zero value is required.
        self.roughnesses = []
        tick = 0
        for i in conformal:
            if i == 0:
                self.roughnesses.append(self.roughnesses_p[tick])
                tick += 1

            else:
                self.roughnesses.append(i)
        
        self.nucSLDs = nucSLDs
        # magSLDs
        # if magSLDs not defined by user, init an array of zeros of len = number of layers.
        if magSLDs is not None:
            self.magSLDs = magSLDs

            if self.spin_state != 'up' and self.spin_state != 'down': #check spin state is either 'up' or 'down'
                raise ValueError(
                    "If magnetic SLDs are used, the spin state passed to the VFP must be 'up' or 'down'."
                )
        
        else:
            self.magSLDs = np.zeros(len(self.thicknesses) + 1)

        # same for nuciSLDs.
        if nuciSLDs is not None:
            self.nuciSLDs = nuciSLDs
        
        else:
            self.nuciSLDs = np.zeros(len(self.thicknesses) + 1)

        # check the lengths of SLD arrays are the same.
        if len(self.nuciSLDs) != len(self.nucSLDs) or len(self.magSLDs) != len(self.nucSLDs):
            raise ValueError(
                "The number of supplied nuclear, magnetic and imaginary SLD values must be the same."
            )

        # we need lists of parameters used in the nucSLDs, magSLDs and nuciSLDs
        nucSLDs_p = self._createparam(self.nucSLDs, "nSLD")
        magSLDs_p = self._createparam(self.magSLDs, "magSLD")
        nuciSLDs_p = self._createparam(self.nuciSLDs, "niSLD")

        # finally, remove any duplicates from nucSLDs_p, magSLDs_p, nuciSLDs_p
        self.nucSLDs_p = [j for i, j in enumerate(nucSLDs_p) if j not in nucSLDs_p[:i]]
        self.magSLDs_p = [j for i, j in enumerate(magSLDs_p) if j not in magSLDs_p[:i]]
        self.nuciSLDs_p = [j for i, j in enumerate(nuciSLDs_p) if j not in nuciSLDs_p[:i]]

        # Simple warning on max_delta_z being too low.
        if np.any(np.array(self.roughnesses_p) < 2*self.max_delta_z):
            warnings.warn(
                          "The microslice thickness is less than twice some of the roughness parameters. Consider reducing the max_delta_z of the VFP."
                          )
    
    def __str__(self):
        """Overwrites the refnx.reflect.Component.__str__ method, so that printing the VFP
        object will return the output from self.__repr__(), defined in BaseVFP.

        Returns:
            string: Simple printable description of refnx VFP.
        """
        return self.__repr__()

    @property
    def parameters(self):
        """Collates all accessible refnx.analysis.Parameter(s) in the refnxVFP to a list, llps.
        Then adds all parameters in llps to the refnx.analysis.Parameters object, p.
        The list is then used by refnx to track which parameters are varying in the model.
        
        Returns:
            Parameters: sequence of parameters.
        """
        # create a list of list of parameters
        # if a particular sublist of parameters is None, then don't add to llps.
        p = Parameters(name=self.name)
        llps = [lps for lps in 
                [self.thicknesses, self.roughnesses_p, self.demaglocs, self.demagwidths, self.nucSLDs_p, self.magSLDs_p, self.nuciSLDs_p] 
                if lps]
        p.extend([ps for lps in llps for ps in lps]) # add defined parameters to parameter list.
        return p

    def slabs(self, structure=None):
        """Generate array representation of the refnx VFP as a 2d np.array using the
        thicknesses, SLDs and iSLDs of the microslabs which represent the SLD profile.

        Args:
            structure (refnx.reflect.Structure, optional): The refnx.reflect.Structure hosting this VFP component. Defaults to None.

        Raises:
            ValueError: if the VFP is not part of a refnx.reflect.Structure, this function will raise a ValueError.

        Returns:
            np.array: slabs is a 2d np.array with shape = (Nlayers, 5).
        """
        if structure is None:
            raise ValueError("VFP.slabs() requires a valid Structure")
        
        # use the __call__ method of the VFP class to
        # return total slds, islds and thicknesses of each slab
        slds, islds, thicks = self.process_model()

        # init a 2D array (Nlayers, 5)
        slabs = np.zeros((len(thicks), 5))
            
        # now populate slabs with microslab thicknesses & SLDs.
        slabs[:, 0] = thicks
        slabs[:, 1] = slds
        slabs[:, 2] = islds
        return slabs
    
class refl1dVFP(Layer, BaseVFP):
    """Describes SLD profiles of whole interfaces from fronting to backing. 
    SLD profiles are calculated via generating volume fraction profiles, which avoid negative volume fractions while keeping the total
    volume fraction = 1. This object should be passed to a refl1d.model.Stack, with the fronting and backing refl1d.material.SLD
    surrounding the refl1dVFP.

    Args:
        Layer (refl1d.model.Layer): Abstract base class of material components in refl1d.
        BaseVFP (vfp.basevfp.BaseVFP): Parent class to refnxVFP and refl1dVFP.
    """
    def __init__(self, nucSLDs, thicknesses, roughnesses, nuciSLDs=None, magSLDs=None, spin_state='none',
                 orientation='front', demaglocs=[], demagwidths=[], SLD_constraint=None, max_delta_z=0.5):
        """Initialises the refl1dVFP variables. A number of checks are conducted on the length of the sequences passed to the VFP to
        check they are of the required relative lengths, and contain the correct type of input.

        Args:
            nucSLDs (np.array): Nuclear scattering length densities of each layer (fronting to backing) within the volume fraction profile.
                                Numpy array of floats and/or bumps.parameter(s). Length of nucSLDs = len(thicknesses) + 1.
            thicknesses (list/tuple): Thicknesses of layers in the VFP. Sequence should start with fronting thickness (= 0), and end with 
                                      the thickness of the penultimate layer (layer before the backing material).
                                      The thicknesses control the midpoint-to-midpoint distances of adjacent interfaces.
                                      List/tuple of floats and/or bumps.parameter(s).
            roughnesses (list/tuple): Roughnesses of layers in the VFP. Sequence should start with roughness of the fronting and end with
                                      the roughness of the penultimate layer (layer before the backing material).
                                      The roughnesses control the width of the interfaces.
                                      By passing a string which reads 'conformal' in the roughnesses sequence, the interface at that index
                                      will be conformal to the interfaces before it. 
                                      List/tuple of floats and/or bumps.parameter(s) and/or strings.
            nuciSLDs (np.array, optional): Imaginary nuclear scattering length densities of each layer (fronting to backing) within the 
                                           volume fraction profile. Defaults to None.
                                           Numpy array of floats and/or bumps.parameter(s). Length of nuciSLDs = len(nucSLDs).
            magSLDs (np.array, optional): Magnetic scattering length densities of each layer (fronting to backing) within the volume 
                                          fraction profile. Defaults to None.
                                          Numpy array of floats and/or bumps.parameter(s). Length of magSLDs = len(nucSLDs).
            spin_state (str, optional): Defines if nucSLDs should be calculated as nuclear (spin_state = 'none'), 
                                        nuclear+magnetic (spin_state = 'up') or nuclear-magnetic (spin_state = 'down'). Defaults to 'none'.
            orientation (str, optional): Defines if incident radiation passes through fronting or backing. 
                                         Through the fronting = (orientation = 'front'), through the backing = (orientation = 'back').
                                         Useful if co-refining NR and XRR, where the orientation of the substrate may be reversed between
                                         measurements. Defaults to 'front'.
            demaglocs (list, optional): List of bumps.parameter(s), which must be of length 0, 2 or 4. 
                                        The parameters define the centre point of Gaussian CDFs, which are used to create demagnetised
                                        regions. The parameters are consectuive, so the z location of parameter 2 will be parameter 1 + 
                                        parameter 2. Defaults to [].
            demagwidths (list, optional): List of bumps.parameter(s), which must be of length 0, 2 or 4. 
                                          The parameters define the width of Gaussian CDFs, which are used to create demagnetised
                                          regions. Defaults to [].
            SLD_constraint (Class, optional): User supplied object that has a layer_choice method and a __call__ method. The user class
                                              is used to calculate the integral of two layers in the VFP, which is useful for
                                              constraining some functions of parameters, such as constraining the surface excess of 
                                              materials over two layers. The layer_choice method will return the indices of two layers 
                                              in the VFP which are to be integrated. The __call__ method of the SLD_constraint object
                                              takes the integrals of the two layers' vfps as arguments, and calculates the nucSLD of
                                              one of the constrained layers via the function of the two integrals. It also returns the
                                              index of the nucSLD which needs to be altered.
                                              Defaults to None.
            max_delta_z (float, optional): Defines the thickness of the uncombined microslices across the VFP. 
                                           The thickness of combined microslices will be a multiple of max_delta_z. Defaults to 0.5 Å.
        """
        # the following variables are not refl1d parameters.
        self.name = 'refl1d VFP'
        self.max_delta_z = max_delta_z
        self.orientation = orientation
        self.spin_state = spin_state
        self.SLD_constraint = SLD_constraint

        if len(demaglocs) != len(demagwidths):
            raise ValueError(
                "The number of the demagnetisation locations must be equal to the number of the demagnetisation widths."
            )

        while len(demaglocs) not in [0, 2, 4]:
            raise ValueError(
                "The number of the demagnetisation locations (and widths) must be 0, 2 or 4."
            )

        if len(thicknesses) != len(roughnesses):
            raise ValueError(
                "The number of roughness parameters must match the number of thickness parameters."
            )
        
        if len(nucSLDs) != len(thicknesses) + 1:
            raise ValueError(
                "The number of supplied SLD values must be 1 greater than the number of thickness parameters."
            )
        
        # init a list of where conformal interfaces are:
        conformal = []
        
        for i in roughnesses:
            if isinstance(i, (float, int, str, bumpsParameter)):
                if isinstance(i, str) and i == 'conformal':
                    conformal.append(1)

                elif isinstance(i, str) and i != 'conformal':
                    raise ValueError(
                        "Any string within the roughness list must read 'conformal'."
                    )
                
                else:
                    conformal.append(0)

            else:
                raise ValueError(
                    "The entries within the roughness list must be a float, interger, bumps.parameter or a string == 'conformal'."
                )
            
        # can only have conformal roughnesses with more than one interface.
        # therefore the first interface cannot be conformal
        if np.any(conformal):
            idx_where_first_one = np.asarray(np.array(conformal) == 1).nonzero()[0][0]
            
            if idx_where_first_one > 0:
                self.conformal = conformal
            
            else:
                raise ValueError(
                    "Cannot specify the first interface to be conformal."
                )
        
        else:
            self.conformal = np.zeros(len(thicknesses))

        # where conformal in roughnesses, replace value with None
        roughnesses_alt = [None if i == 'conformal' else i for i in roughnesses]
        
        # if specified, the following should be refl1d parameters.
        self.thicknesses = [bumpsParameter.default(j, name=f"{self.name} - Layer{i} - thick") if not isinstance(j, bumpsParameter) else j for i, j in enumerate(thicknesses)]
        self.demaglocs = [bumpsParameter.default(j, name=f"{self.name} - demaglocs{i}") if not isinstance(j, bumpsParameter) else j for i, j in enumerate(demaglocs)]
        self.demagwidths = [bumpsParameter.default(j, name=f"{self.name} - demagwidths{i}") if not isinstance(j, bumpsParameter) else j for i, j in enumerate(demagwidths)]
        # create parameters for any roughness values that aren't parameters
        self.roughnesses_p = [bumpsParameter.default(j, name=f"{self.name} - Layer{i}/Layer{i+1} - rough") if not j is None and not isinstance(j, bumpsParameter) else j for i, j in enumerate(roughnesses_alt)]

        # now we have to create a list of the above roughness parameters, and where they are not, replace with 1 as a "dummy value".
        # the dummy value will have no impact on the resulting VFP, but a finite non-zero value is required.
        self.roughnesses = []
        tick = 0
        for i in conformal:
            if i == 0:
                self.roughnesses.append(self.roughnesses_p[tick])
                tick += 1

            else:
                self.roughnesses.append(i)

        self.nucSLDs = nucSLDs
        # magSLDs
        # if magSLDs not defined by user, init an array of zeros of len = number of layers.
        if magSLDs is not None:
            self.magSLDs = magSLDs

            if self.spin_state != 'up' and self.spin_state != 'down': #check spin state is either 'up' or 'down'
                raise ValueError(
                    "If magnetic SLDs are used, the spin state passed to the VFP must be 'up' or 'down'."
                )
        
        else:
            self.magSLDs = np.zeros(len(self.thicknesses) + 1)

        # same for nuciSLDs.
        if nuciSLDs is not None:
            self.nuciSLDs = nuciSLDs
        
        else:
            self.nuciSLDs = np.zeros(len(self.thicknesses) + 1)

        # check the lengths of SLD arrays are the same.
        if len(self.nuciSLDs) != len(self.nucSLDs) or len(self.magSLDs) != len(self.nucSLDs):
            raise ValueError(
                "The number of supplied nuclear, magnetic and imaginary SLD values must be the same."
            )

        # we need lists of parameters used in the nucSLDs, magSLDs and nuciSLDs
        nucSLDs_p = self._createparam(self.nucSLDs, "nSLD")
        magSLDs_p = self._createparam(self.magSLDs, "magSLD")
        nuciSLDs_p = self._createparam(self.nuciSLDs, "niSLD")

        # finally, remove any duplicates from nucSLDs_p, magSLDs_p, nuciSLDs_p
        # need to remove any appearances of nucSLDs in magSLDs and nuciSLDs - these appear from functions of parameters.
        self.nucSLDs_p = [j for i, j in enumerate(nucSLDs_p) if j not in nucSLDs_p[:i]]
        self.magSLDs_p = [j for i, j in enumerate(magSLDs_p) if j not in magSLDs_p[:i] and nucSLDs_p[:i]]
        self.nuciSLDs_p = [j for i, j in enumerate(nuciSLDs_p) if j not in nuciSLDs_p[:i] and nucSLDs_p[:i]]

        # Simple warning on max_delta_z being too low.
        if np.any(np.array(self.roughnesses_p) < 2*self.max_delta_z):
            warnings.warn(
                          "The microslice thickness is less than twice some of the roughness parameters. Consider reducing the max_delta_z of the VFP."
                          )

        # refl1d requires the total thickness of the vfp as a parameter at the beginning and throughout fitting.
        # this must be defined as self.thickness.
        _, _, thicks = self.process_model()
        total_thicks = thicks.sum()
        self.thickness = bumpsParameter.default(total_thicks, name=f"{self.name} - total thickness")
    
    def to_dict(self):
        """Returns a dictionary representation of the VFP using the to_dict function of bumps.parameters.
        Used when saving a refl1d model details as a .json file.

        Returns:
            dictionary: dictionary representation of the parameters in the VFP.
        """
        return to_dict({'type': 'VFP',
                        'name': self.name,
                        'thicks': self.thicknesses,
                        'roughs': self.roughnesses_p,
                        'nucSLDs': self.nucSLDs_p,
                        'magSLDs': self.magSLDs_p,
                        'nuciSLDs': self.nuciSLDs_p,
                        'demaglocs': self.demaglocs,
                        'demagwidths': self.demagwidths})

    def layer_parameters(self):
        """Takes all the parameters which define the refl1d VFP, and organises them
        into a dictionary. The keys of the dictionary are the VFP arguments.

        Returns:
            dictionary: dict of parameters with equal to the name of the VFP arguments.
        """
        # init a list of parameter lists.
        # only include the parameter list if its not empty.
        llps = [lps for lps in 
            [self.thicknesses, self.roughnesses_p, self.demaglocs, self.demagwidths, self.nucSLDs_p, self.magSLDs_p, self.nuciSLDs_p] 
            if lps]
        
        # create a list of keys for the parameter dictionary.
        keys = ['thicknesses', 'roughnesses', 'demag_locs', 'demag_widths', 'nucSLDs', 'magSLDs', 'nuciSLDs']
        
        # get indices of where parameter lists aren't zero.
        keys_idx = [i for i, j in 
                    enumerate([self.thicknesses, self.roughnesses_p, self.demaglocs, self.demagwidths, self.nucSLDs_p, self.magSLDs_p, self.nuciSLDs_p])
                    if j]
        
        # create dictionary with keys, indices and the list of parameter lists.
        p = dict(zip([keys[i] for i in keys_idx], llps))
        return p

    def render(self, probe, slabs):
        """Appends the microslice thickness, SLDs and iSLDs to the Microslabs object passed to the render 
        function of the VFP by refl1d's Experiment object. Also updates the self.thickness value of the VFP.

        Args:
            probe (refl1d.probe.NeutronProbe): Passed to render functions of refl1d.layers but not used here.
            slabs (refl1d.profile.Microslabs): Object which has rho, irho, w and sigma properties.
        """
        
        # use the __call__ method of the VFP class to
        # return total slds, islds and thicknesses of each slab
        slds, islds, thicks = self.process_model()

        # update the self.thickness variable.
        self.thickness.value = thicks.sum()

        # now append slds, islds and thicks to slabs.
        for i in range(0, len(thicks)):
            slabs.append(rho=slds[i], irho=islds[i], w=thicks[i], sigma=0)