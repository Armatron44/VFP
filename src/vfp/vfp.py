import numpy as np
from refnx.reflect import Component
from refnx.reflect.interface import Step
from refnx.analysis import Parameter, Parameters, possibly_create_parameter
from refnx.analysis.parameter import _BinaryOp
import warnings
from scipy import stats
from methodtools import lru_cache
import matplotlib.pyplot as plt
from matplotlib import colormaps

class VFP(Component):
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
        super().__init__() # inherit the Component class.
        
        # the following options are not refnx.parameters.
        self.name = 'VFP'
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
            idx_where_first_one = np.where(np.array(conformal) == 1)[0][0]
            
            if idx_where_first_one > 0:
                self.conformal = conformal
            
            else:
                raise ValueError(
                    "Cannot specify the first interface to be conformal."
                )
        
        else:
            self.conformal = np.zeros(len(thicknesses))
        
        # if specified, the following should be refnx parameters.
        self.thicknesses = [possibly_create_parameter(j, name=f"{self.name} - Layer{i} - thick") for i, j in enumerate(thicknesses)]
        self.demaglocs = [possibly_create_parameter(j, name=f"{self.name} - demaglocs{i}") for i, j in enumerate(demaglocs)]
        self.demagwidths = [possibly_create_parameter(j, name=f"{self.name} - demagwidths{i}") for i, j in enumerate(demagwidths)]
        
        # where conformal in roughnesses, replace value with None
        roughnesses_alt = [None if i == 'conformal' else i for i in roughnesses]
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

        # set self.nucSLDs to be the float, parameter or _BinaryOp objects in the nucSLDs array.
        self.nucSLDs = nucSLDs

        # we need lists of parameters used in the nucSLDs, magSLDs and nuciSLDs
        # nucSLDs
        nucSLDs_p = []
        for k, i in enumerate(nucSLDs):
            if isinstance(i, Parameter):
                nucSLDs_p.append(i)
            
            elif isinstance(i, _BinaryOp): # can't handle _BinaryOp so warn user that the parameters that define must go to auxiliary params
                warnings.warn(
                    "Pass nucSLD parameters that are only part of a parameter operation (i.e f(p1, p2) = p1 + p2) to the auxiliary parameters argument of the objective."
                              )
            
            elif not isinstance(i, Parameter) and not isinstance(i, _BinaryOp):
                nucSLDs_p.append(possibly_create_parameter(i, name=f"{self.name} - Layer{k} - nSLD"))
        
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

        magSLDs_p = []
        if (self.magSLDs != 0).any(): # check if magSLDs doesn't only contain zeros - if it does, we don't need parameters.
            for k, i in enumerate(self.magSLDs[self.magSLDs != 0]):
                if isinstance(i, Parameter):
                    magSLDs_p.append(i)
                
                elif isinstance(i, _BinaryOp):
                    warnings.warn(
                        "Pass magSLD parameters that are only part of a parameter operation (i.e f(p1, p2) = p1 + p2) to the auxiliary parameters argument of the objective."
                                  )
                
                elif not isinstance(i, Parameter) and not isinstance(i, _BinaryOp):
                    magSLDs_p.append(possibly_create_parameter(i, name=f"{self.name} - Layer{k} - magSLD"))

        # nuciSLDs
        nuciSLDs_p = []
        if (self.nuciSLDs != 0).any(): # check if nuciSLDs doesn't only contain zeros - if it does, we don't need parameters.
            for k, i in enumerate(self.nuciSLDs[self.nuciSLDs != 0]):
                if isinstance(i, Parameter):
                    nuciSLDs_p.append(i)

                elif isinstance(i, _BinaryOp):
                    warnings.warn(
                        "Pass nuciSLD parameters that are only part of a parameter operation (i.e f(p1, p2) = p1 + p2) to the auxiliary parameters argument of the objective."
                                  )
                
                elif not isinstance(i, Parameter) and not isinstance(i, _BinaryOp):
                    nuciSLDs_p.append(possibly_create_parameter(i, name=f"{self.name} - Layer{k} - niSLD"))

        # finally, remove any duplicates from nucSLDs_p, magSLDs_p, nuciSLDs_p
        self.nucSLDs_p = [j for i, j in enumerate(nucSLDs_p) if j not in nucSLDs_p[:i]]
        self.magSLDs_p = [j for i, j in enumerate(magSLDs_p) if j not in magSLDs_p[:i]]
        self.nuciSLDs_p = [j for i, j in enumerate(nuciSLDs_p) if j not in nuciSLDs_p[:i]]

        # Simple warning on max_delta_z being too low.
        if np.any(np.array(self.roughnesses_p) < 2*self.max_delta_z):
            warnings.warn("The microslice thickness is less than twice some of the roughness parameters. Consider reducing the max_delta_z of the VFP.")

        # init class variables via call method.
        self()

    def __repr__(self):
        s = (f"VFP: \n"
             f"Thicks - {self.thicknesses} \n"
             f"Roughs - {self.roughnesses_p} \n"
             f"Nuclear SLDs - {self.nucSLDs_p} \n"
             f"Magnetic SLDs - {self.magSLDs_p} \n"
             f"Imaginary SLDs - {self.nuciSLDs_p} \n"
             f"Demag Locations - {self.demag_locs} \n"
             f"Demag Widths - {self.demag_widths} \n")
        return s
    
    @property
    def parameters(self):
        # create a list of list of parameters
        # if a particular sublist of parameters is None, then don't add to llps.
        p = Parameters(name=self.name)
        llps = [lps for lps in 
                [self.thicknesses, self.roughnesses_p, self.demag_locs, self.demag_widths, self.nucSLDs_p, self.magSLDs_p, self.nuciSLDs_p] 
                if lps]
        p.extend([ps for lps in llps for ps in lps]) # add defined parameters to parameter list.
        return p
    
    @classmethod
    def consecutive(cls, arr, stepsize=1):
        """
        Splits an array into sub arrays where the difference between neighbouring units is not 1.
        """
        return np.split(arr, np.where(np.diff(arr) != stepsize)[0] + 1)

    @lru_cache(maxsize=2)
    @classmethod
    def calc_dzs(cls, zstart, zend, points, idxs):
        """
        Calculates the thickness (z) of each microslice after reducing the VFP in self.init_demag().
        Cached class method so this calculation is not repeated for VFPs with the same input values.

        The thickness of each microslice is approximately the value of self.max_delta_z prior to
        reduction.

        Parameters
        ----------
        zstart : float
                 z value of where VFP starts.
        zend : float
               z value of where VFP ends.
        points : integer
                 number of points in the VFP. Points = int((zend - zstart) / self.max_delta_z)
                 defined in self.calc_zeds()
        idxs : tuple
               indices of nodes in the VFP that are approximately equal to a neighbouring node 
               as defined in self.init_demag(). These indices are used to calculate the thickness
               of each microslice across an uneven z space after reduction.
        
        Returns
        -------
        dzs : np.array of microslice thicknesses (1d).
        """
        
        idxs = np.array(idxs)
        
        # find thickness of microslabs without reduction.
        delta_step = (-zstart + zend)/(points - 1)

        # if idxs is empty, then each dz is 1 * delta_step.
        if not idxs.any():
            dzs = np.ones(points) * delta_step

        # if there are indices, then dzs needs to be altered to include slabs that are > delta_step   
        else: 
            indexs = cls.consecutive(idxs) # list of n arrays (n is the number of zones where there is invariance in demag factor)
            indexs_diffs = [j[-1]-j[0] for j in indexs] # find length of each zone and return in a list.
            indexs_starts = [j[0] for j in indexs] # where does each zone start?
            indexs_ends = [j[-1] for j in indexs] # where does each zone end?
        
            # calculate the distance between indicies of interest
            index_gaps = np.array([j - (indexs_ends[i-1] + 1) for i, j in enumerate(indexs_starts) if i > 0])
            new_points = points - (np.array(indexs_diffs).sum() + len(indexs)) # number of slabs required.
            new_indexs_starts = [indexs_starts[0] + index_gaps[:i].sum() for i in range(0, len(indexs))]
            
            dzs = np.ones(new_points) * delta_step # init an array for dzs. make all values delta step to begin with.
            if len(new_indexs_starts) > 1:
                for i, j in enumerate(new_indexs_starts): # find places where delta step needs to be altered.
                    dzs[j] = ((indexs_diffs[i] + 1)* delta_step) + dzs[j-1]
            
            # alter dz in the one place required.
            else:
                dzs[int(new_indexs_starts[0])] = ((indexs_diffs[0] + 1) * delta_step) + dzs[int(new_indexs_starts[0]-1)]
        
        return dzs

    def z_and_SLD_scatter(self, imag=False, reduced=True):
        """
        Function used for plotting SLDs from VFP.
        Returns z values from self.calc_zeds() and also returns
        SLD values from self.calc_slds() calculated from the VFP.

        Parameters
        ----------
        imag : Boolean
               If False/True, will return coherent/imaginary SLDs.
        reduced : Boolean
                  If False/True, will return full/reduced zs and SLDs.

        Notes
        -----
        The VFP SLDs here do not match the SLDs used in the refnx structure to 
        which this VFP belongs. This is because the SLDs in the refnx structure 
        are the average of the VFP SLDs that are returned here. The averaging is
        done in the __call__ method and is fed through to the slabs method.
        
        Returns
        -------
        x : z (1d) - either reduced or full.
        y : SLDs (1d) - either reduced or full.
        """
        self() # use __call__ method to update VFPs if parameters have been updated.

        zeds = np.array(self.zeds)

        if self.orientation in ('front'):
            if reduced is True:
                SLDs = self.calc_slds()
                if imag is False:
                    y = SLDs[0]
                    sep_n = SLDs[2]
                    sep_m = SLDs[3]
                    x = np.delete(zeds, self.indices)

                else:
                    y = SLDs[1]
                    sep_n = SLDs[2]
                    sep_m = SLDs[3]
                    x = np.delete(zeds, self.indices)

            else:
                SLDs = self.calc_slds(reduced=False)
                if imag is False:
                    y = SLDs[0]
                    sep_n = SLDs[2]
                    sep_m = SLDs[3]
                    x = zeds

                else:
                    y = SLDs[1]
                    sep_n = SLDs[2]
                    sep_m = SLDs[3]
                    x = zeds

        if self.orientation in ('back'):
            offset = np.cumsum(self.thicks)[-1]
            if reduced is True:
                SLDs = self.calc_slds()
                if imag is False:
                    y = SLDs[0]
                    sep_n = SLDs[2]
                    sep_m = SLDs[3]
                    x = -(np.delete(zeds, self.indices) - offset)
                    
                else:
                    y = SLDs[1]
                    sep_n = SLDs[2]
                    sep_m = SLDs[3]
                    x = -(np.delete(zeds, self.indices) - offset)

            else:
                SLDs = self.calc_slds(reduced=False)
                if imag is False:
                    y = SLDs[0]
                    sep_n = SLDs[2]
                    sep_m = SLDs[3]
                    x = -(zeds - offset)
                
                else:
                    y = SLDs[1]
                    sep_n = SLDs[2]
                    sep_m = SLDs[3]
                    x = -(zeds - offset)
        return x, y, sep_n, sep_m
    
    @lru_cache(maxsize=2)
    @classmethod
    def calc_zeds(cls, rough, thick, mxdz):
        """
        Returns array of z values for VFP calculated using roughnesses & thicknesses.
        
        Cached class method so that VFPs that share same the arguments aren't required
        to re-calculate (e.g multiple contrasts).
        
        Parameters
        ----------
        rough : tuple of roughness values
                used in calculation of zstart and zend.
        thick : tuple of thickness values
                used in calculation of zend.
        mxdz :  float - self.max_delta_z as defined in the init fuction.
                used to calculate the number of points in returned z array.
        
        Returns
        -------
        zstart : float - start z value of VFP.
        zend : float - end z value of VFP.
        points : integer - number of points in zeds.
        zeds : np.array (1d) of length points.
        """
        # convert rough & thick tuples to arrays.
        rough = np.array(rough)
        thick = np.array(thick)
        
        # find the start of VF profile.
        zstart_nr = -5 - (4 * rough[0])

        # set to the next lower multiple of mxdz
        zstart = np.floor(zstart_nr * (1 / mxdz)) / (1 / mxdz)
        
        # find the point at which VF profile has reached ~ 1 on backing side.
        zend_of_vfprofile = np.max(thick.sum() + 4 * rough)
        zend_nr = 5 + zend_of_vfprofile #add a small offset

        # set to the next higher multiple of mxdz
        zend = np.ceil(zend_nr * (1 / mxdz)) / (1 / mxdz)
        
        # calculate number of points required in z array.
        #TODO is np.rint required here? could use int((-zstart + zend)/mxdz + 1)
        points = np.rint((-zstart + zend)/mxdz + 1).astype(int) 

        zeds = np.linspace(zstart, zend, num=points)
        
        return zstart, zend, points, zeds
        
    @classmethod
    def one_minus_cdf(cls, interf_choice, x, cumthick, rough): 
        """
        Returns 1-CDF for a given set of thickness and roughness that describe an interface.
        
        Parameters
        ----------
        interf_choice : integer
                        selects which interf to calculate the CDF for
        x : np.array
            z values over which VFP will be calculated.
        cumthick : np.array
                   cumulative thicknesses of the layers.
        rough : np.array
                list of roughnesses of the layers.
        
        Returns
        -------
        one_minus_cdf : np.array (1d)
        """
        # TODO: add other distribution types to this function?
        # exponential, uniform (straight line CDF), ...
        one_minus_cdf = 1-stats.norm.cdf(x, loc=cumthick[interf_choice], scale=rough[interf_choice])
        return one_minus_cdf
   
    @lru_cache(maxsize=2)
    @classmethod
    def calc_vfp(cls, rough, thick, zeds, conformal):
        """
        Returns volume fraction profile for each layer.
        This method is a cached class method so that the VFP
        does not have to be recalculated for contrasts with
        the same sample structure.
        
        Parameters
        ----------
        thick : tuple
                tuple of thicknesses values of the layers.
        rough : tuple
                tuple of roughnesses values of the layers.
        zeds : tuple
               tuple of z values across VFP. 
        conformal : tuple
                    tuple of 0 and 1s. 
                    1 indicates conformal interface to everything before, 0 indicates non-conformal interface.
        
        Returns
        -------
        vfp : np.array (2d) - Shape = (Nlayers, len(z))
        """
        rough = np.array(rough)
        thick = np.array(thick)
        z = np.array(zeds)
        conformal = np.array(conformal)

        cumthick = np.cumsum(thick)
        num_layers = len(thick) + 1
        num_conform = np.sum(conformal)

        if num_conform != 0:
            num_nonconform_b4 = np.where(conformal == 1)[0][0] + 1
        else:
            num_nonconform_b4 = num_layers

        # to initiate, calculate VFP of those interfaces before the first conformal
        # interface via product of CDFs.
        vfs_init = np.ones((num_layers, z.shape[0]))
        for i in range(0, num_nonconform_b4 - 1): # i = 0, 1, 2, ..., num_nonconform_b4 - 2
            vfs_init[i] = cls.one_minus_cdf(i, z, cumthick, rough)
            
        mlti_arr = np.ones((num_layers, z.shape[0]))
        for i in range(1, num_nonconform_b4): # i = 1, 2, 3, ..., num_nonconform_b4 - 1
            mlti_arr[i] = 1 - cls.one_minus_cdf(i - 1, z, cumthick, rough)
                
        mlti_arr_cp = np.cumprod(mlti_arr, axis=0)
        vfp = vfs_init * mlti_arr_cp

        # TODO: refactor below?
        # if there are conformal interfaces, we need to modify the first conformal interface and all after it.
        if num_conform != 0:
            idx_conformal = np.where(conformal == 1)[0] # find conformal interfaces
            for j, i in enumerate(idx_conformal): # for each conformal interface
                non_conform_b4 = np.where(conformal[:i+1] == 0)[0] # finds indices of non-conformal interfaces before this conformal interface.

                # find where there is a jump between non-conformal interfaces:
                locator = np.zeros_like(non_conform_b4)
                locator[1:] = np.where(np.diff(non_conform_b4) > 1, 1, 0) # array value = 1 where there is a jump & 0 where there isn't.
                
                # init some offsets.
                thick_offsets = np.zeros(len(non_conform_b4)) # length of non-conformal interfaces
                idx_interest = cls.consecutive(np.where(conformal[:i+1] == 1)[0]) # finds the indices of conformal intefaces & splits depending on if consecutive.
                non_conf_batch = 0 # init a ticker whose value depends on locator.
                
                # set the thick_offset values
                for m in range(len(thick_offsets)):
                    if locator[m] == 1:
                        non_conf_batch += 1
                    thick_offsets[m] = thick[idx_interest[non_conf_batch][0]:idx_interest[non_conf_batch][-1] + 1].sum()
                
                # init a cumulative thickness offset array.
                cthick_off = np.zeros_like(cumthick)
                # add the thick_offsets to the cumulative thicknesses to those that correspond to non-conformal interfaces.
                cthick_off[non_conform_b4] = cumthick[non_conform_b4] + thick_offsets 
                
                # calculate the required CDFs but offset with thickness of layers.
                mlti_off_arr = np.ones((len(non_conform_b4), z.shape[0]))
                # init an adjuster so that we calculate the correct CDFs.
                num_consec_conform = np.zeros_like(non_conform_b4)
                num_consec_conform[1:] = np.diff(non_conform_b4) - 1
                adjuster = np.zeros_like(non_conform_b4)
                for m in range(len(adjuster)):
                    adjuster[m] = num_consec_conform[:m+1].sum()
                # calculate the CDFs
                for m in range(len(non_conform_b4)): # m = 0, 1, 2, ..., len(non_conform_b4)-1
                    mlti_off_arr[m] = 1 - cls.one_minus_cdf(m + adjuster[m], z, cthick_off, rough) # offset by adjust to calculate relevant CDFs.

                # need the product of the calculated CDFs.
                mlti_off_arr_prod = np.prod(mlti_off_arr, axis=0)
                vfp_off_sum = 1 - mlti_off_arr_prod
                vfp[i] = vfp_off_sum - np.sum(vfp[:i].T, axis=1)
                vfp[i+1] = 1 - vfp_off_sum # set the next layer's VFP to 1-everything else - required for backing VFP.

                # work out if additional interfaces required after this conformal interface
                if i+1 == len(thick): # no interfaces after this conformal interface
                    interf_after = 0
                elif i+1 < len(thick) and i == idx_conformal[-1]: # only non-conformal interfaces after this conformal interface
                    interf_after = len(thick) - (i + 1)
                elif i+1 < len(thick): # there are both conformal and non-conformal interfaces after this conformal interface.
                    interf_after = idx_conformal[j+1] - (i + 1) # the number of non-conformal interfaces after this conformal interface but before the next conformal interface.
                else:
                    pass
                
                if interf_after > 0: # are there non-conformal interfaces after?
                    vfs_after_init = np.ones((interf_after + 1, z.shape[0]))
                    for m in range(i + 1, i + interf_after + 1): # m = i+1, i+1+1, i+1+2, ..., i+interf_after
                        vfs_after_init[m-(i+1)] = cls.one_minus_cdf(m, z, cumthick, rough)
            
                    mlti_after_arr = np.ones((interf_after+1, z.shape[0]))
                    for m in range(i + 2, i + interf_after + 2): # m = i+2, i+2+1, i+2+2, ..., i+1+interf_after
                        mlti_after_arr[m-(i+1)] = 1 - cls.one_minus_cdf(m - 1, z, cumthick, rough)
                
                    mlti_b4_arr_cp = np.cumprod(mlti_after_arr, axis=0)
                    vfp[i+1:i+2+interf_after] = (1 - vfp_off_sum) * vfs_after_init * mlti_b4_arr_cp

        return vfp
    
    @lru_cache(maxsize=2)
    @classmethod
    def init_demag(cls, locs, widths, mSLDs, zeds, vfp):
        """
        Calculates the product of the VFP and the magnetic 'deadness' --> mag_comp. 
        This is used for calculating the magnetic SLDs of the layers.
        mag_comp and the original VFP are then reduced by finding the regions of
        mag_comp that do not vary by < 1e-5.
        
        Returns the following: 
        1. reduced_vfp and reduced_magcomp - used in the calculation of SLDs.
        2. idxs - the indices of where points were removed from VFP and mag_comp.
        3. demag_arr - Magnetic deadlayer peak, not reduced.

        Parameters
        ----------
        locs : tuple
               tuple of scale values to describe demagnetisation peak(s).
        widths : tuple
                 tuple of scale values to describe demagnetisation peak(s).
        mSLDs : tuple
                tuple of magnetic SLD values of the layers.
        zeds : tuple
               tuple of z values across VFP.         
        vfp : tuple
              Nested tuple (2d) containing VFP of each layer.
        
        Returns
        -------
        reduced_vfp : np.array (2d) - Shape = (Nlayers, len(z) - len(idxs))
                      Reduced VFPs.
        reduced_magcomp : np.array (2d) - Shape = (Nlayers, len(z) - len(idxs))
                          Reduced mag_comp.
        idxs : np.array (1d)
                  Indices of where to remove points from vfp and mag_comp.
        demag_arr : np.array (2d) - Shape = (Nlayers, len(z))
                    Magnetic deadlayer peak before multiplication with VFP.
                    Not reduced.
        """
        locs = np.array(locs)
        widths = np.array(widths)
        mSLDs = np.array(mSLDs)
        zeds = np.array(zeds)
        vfp = np.array(vfp)

        # init an array for any magnetic deadness.
        demag_arr = np.ones((mSLDs.shape[0], zeds.shape[0]))

        # use the following function to model "dead" structure in magnetic layers.
        # it should differ from unity if there are peaks and widths supplied.
        demag_factor = cls.get_demag(zeds, locs, widths)
        
        # now apply demag_factor to all layers that have a magnetic component.
        for i in range(0, mSLDs.shape[0]):
            if mSLDs[i] != 0:
                demag_arr[i] = demag_arr[i] * demag_factor
        
        # calculate magnetic composition of each layer over the interface using VFPs.
        mag_comp = vfp * demag_arr 

        # find the regions of the interface where the VFPs are approximately invariant.
        difference_arr = np.abs(np.diff(mag_comp, axis=1)) < 1e-5
        reduce_diff_arr = np.all(difference_arr, axis=0)
        indices_full = np.nonzero(reduce_diff_arr)

        # shift indices along by 1 & don't take last value of indices_full.
        idxs = (indices_full[0] + 1)[:-1]

        # now remove parts of the vfps and mag_comp where they are ~ invariant.
        reduced_vfp = np.delete(vfp, idxs, 1)
        reduced_magcomp = np.delete(mag_comp, idxs, 1)
        return reduced_vfp, reduced_magcomp, idxs, demag_arr
    
    @classmethod
    def get_demag(cls, dist, locs, widths):
        """
        Calculates the magnetic deadness across the interface given a set of 
        locs and widths parameters. The magnetic deadness can be described by
        up to two peaks calculated from normal cdf functions using locs and
        widths to define their location and scale respectively.
        
        The magnetic deadness is default at 1, which represents a state of
        no magnetic deadness (layers with a magnetic SLD will have the full 
        extent of that magnetic SLD applied).

        Introducing a peak(s) will decrease the magnetic deadness from 1 to
        a range of values between 0 & 1 across the interface. This will
        reduce the extent of magnetic SLD applied to a layer in that region
        of the inverse peak.
        
        Parameters
        ----------
        dist : np.array (1d)
               z values of VFP.
        locs : np.array (1d)
               values of locs parameters.
        widths : np.array (1d)
                 values of widths parameters.

        Returns
        -------
        np.ones_like(dist) : np.array(1d)
                             returned if no location or width values.
        1-peak : np.array(1d)
                 returned if only two locs and two width values passed to function.
        demag_f : np.array (1d)
                  returned if all 4 locs and width values passed to function.
        """
        # if locs does not contain any non-zero values, then just return an array of ones.
        if not locs.any():
            return np.ones_like(dist)
        
        # allow if any values in locs are non-zero and check if all values are non-zero (allow if they are not). 
        # or it allows if only two loc parameters given.
        elif locs.any() and not locs.all() or len(locs) == 2:
            split_loc = np.split(locs, len(locs) / 2)
            split_width = np.split(widths, len(widths) / 2)
            
            for i, j in enumerate(split_loc):
                if j.all():
                    peak = stats.norm.cdf(dist, loc=j[0], scale=split_width[i][0]) * (1-stats.norm.cdf(dist, loc=j[0] + j[1], scale=split_width[i][1]))
                    return 1 - peak
                    
        # if demag_locs and demag_widths have 4 values in them, can return two peaks.
        else:
            cumlocs = np.cumsum(locs)
            peak_1 = stats.norm.cdf(dist, loc=cumlocs[0], scale=widths[0]) * (1-stats.norm.cdf(dist, loc=cumlocs[1], scale=widths[1]))
            peak_2 = stats.norm.cdf(dist, loc=cumlocs[1], scale=widths[1]) * stats.norm.cdf(dist, loc=cumlocs[2], scale=widths[2]) * (1-stats.norm.cdf(dist, loc=cumlocs[3], scale=widths[3]))
            demag_f = 1 - (peak_1 + peak_2) # can have two dead layers.
            return demag_f
        
    @lru_cache(maxsize=2)
    @classmethod
    def integrate_vfp(cls, zeds, indexs, red_vfps, first_layer, second_layer):
        """
        Parameters
        ----------
        zeds : tuple
               tuple of z values across VFP.
        idxs : tuple
               indices of nodes in the VFP that are approximately equal to a neighbouring node 
               as defined in self.init_demag(). These indices are used to calculate the thickness
               of each microslice across an uneven z space after reduction.
        red_vfps : tuple
                   Nested tuple (2d) containing VFP of each layer.
        first_layer : integer
                      idx used to indicate which 
        second_layer : integer
                       idx used to point to which other layer to integrate
        """
        zs = np.array(zeds)
        idxs = np.array(indexs)
        red_vfp = np.array(red_vfps)

        integrate_over = np.delete(zs, idxs) # get zed values to integrate over.

        return np.trapz(red_vfp[first_layer], x=integrate_over), np.trapz(red_vfp[second_layer], x=integrate_over)
    
    def calc_slds(self, reduced=True):
        """
        Calculates coherent (nuclear or nuclear +/- magnetic depending on self.spin_state) 
        and imaginary SLDs with VFPs (reduced or full).
        
        Parameters
        ----------
        reduced : Boolean
                  If True/False, calculates the reduced/full SLD profiles
        
        Returns
        -------
        tot_sld : coherent SLDs (nuclear or nuclear +/- magnetic SLD)
                  np.array (1d) - Shape = (Nlayers, len(z))
        sum_isldn_list : imaginary SLDs 
                         np.array (1d) - Shape = (Nlayers, len(z))
        sum_sldn_list : coherent nuclear SLD.
                        np.array(1d) - Shape = (Nlayers, len(z)) 
        sum_sldm_list : coherent magnetic SLD.
                        np.array(1d) - Shape = (Nlayers, len(z)) 
        """
        # if SLD_constraint is not None, update self.nucSLDs depending on constraint.
        if self.SLD_constraint is not None:
            first_layer, second_layer = self.SLD_constraint.layer_choice() # user defines which layers participate in the constraint.
            int_vfp1, int_vfp2 = self.integrate_vfp(self.zeds, self.indices, self.arrtotuple(self.red_vfp), first_layer, second_layer)
            layer_loc, SLD = self.SLD_constraint(int_vfp1, int_vfp2) # user defines a class with a callable, which returns an idx for modifying a particular SLD value.
            self.nucSLDs[layer_loc] = SLD

        if reduced is True:
            vfp = self.red_vfp
            demagf = self.demagf

            sldn_values = [float(i) for i in self.nucSLDs]
            isldn_values = [float(i) for i in self.nuciSLDs]
            sldn_list = vfp.T * sldn_values
            isldn_list = vfp.T * isldn_values
            sum_sldn_list = np.sum(sldn_list, 1)
            sum_isldn_list = np.sum(isldn_list, 1)
        
            # now calculate the magnetic SLD profile.
            sldm_values = [float(i) for i in self.magSLDs]
            sldm_list = demagf.T * sldm_values
            sum_sldm_list = np.sum(sldm_list, 1)
        
            # now total the nuclear and magnetic SLDs on given contrast.
            # tot sld must either be addition or subtraction.
            if self.spin_state in ('none'):
                tot_sld = sum_sldn_list
            elif self.spin_state in ('down'):
                tot_sld = sum_sldn_list - sum_sldm_list
            elif self.spin_state in ('up'):
                tot_sld = sum_sldn_list + sum_sldm_list
        
        else:
            vfp = self.vfp
            demagf = self.vfs_for_display()[2]

            sldn_values = [float(i) for i in self.nucSLDs]
            isldn_values = [float(i) for i in self.nuciSLDs]
            sldn_list = vfp.T * sldn_values
            isldn_list = vfp.T * isldn_values
            sum_sldn_list = np.sum(sldn_list, 1)
            sum_isldn_list = np.sum(isldn_list, 1)
        
            sldm_values = [float(i) for i in self.magSLDs]
            sldm_list = demagf.T * sldm_values
            sum_sldm_list = np.sum(sldm_list, 1)
        
            if self.spin_state in ('none'):
                tot_sld = sum_sldn_list
            elif self.spin_state in ('down'):
                tot_sld = sum_sldn_list - sum_sldm_list
            elif self.spin_state in ('up'):
                tot_sld = sum_sldn_list + sum_sldm_list
        
        return tot_sld, sum_isldn_list, sum_sldn_list, sum_sldm_list
    
    def get_slds(self):
        """
        Calculate SLDs from VFPs.
        Initially, the VFP is calculated and then it is reduced via self.init_demag.
        
        Returns
        -------
        SLD : coherent SLDs (nuclear or nuclear +/- magnetic) (1d).
        iSLD : imaginary SLDs (1d).
        """

        # calculate the volume fraction profiles of the layers in the interface.
        self.vfp = self.calc_vfp(self.roughs, self.thicks, self.zeds, tuple(self.conformal))

        # using vfp from the above function, calculate reduced volume fraction and magnetic profiles.
        self.red_vfp, self.demagf, idx = self.init_demag(self.demag_locs, 
                                                         self.demag_widths, 
                                                         self.mSLDs, 
                                                         self.zeds,
                                                         self.arrtotuple(self.vfp))[:3]
        
        self.indices = self.arrtotuple(idx)

        # calculate the SLD valus across reduced VFPs.
        SLD, iSLD, _, _ = self.calc_slds()

        return SLD, iSLD

    def arrtotuple(self, arr):
        """
        Takes 1D/2D arrays and returns a tuple/nested tuple for the purposes of caching.
        """
        if arr.ndim == 1:
            return tuple(i for i in arr)
        
        elif arr.ndim == 2:
            return tuple([tuple([float(i) for i in row]) for row in arr])

    def __call__(self):
        """
        Main function of the VFP class, and is called via the slabs() method.
        Calculates the length of the VFP, the thicknesses of each microslice
        and calculates the SLD of each microslice.
        Returns the coherent and imaginary SLD values for each microslice and
        the thickness of each microslice given orientation of sample.
        
        Returns
        -------
        return_slds : np.array (1d) - Shape = (len(zeds) - self.indices)
        return_islds : np.array (1d) - Shape = (len(zeds) - self.indices)
        self.dz : np.array (1d) - Shape = (len(zeds) - self.indices)
        """
        
        # convert the tuples containing parameters to float values 
        # and keep as tuples for the purposes of caching.
        self.thicks = tuple((float(i) for i in self.thicknesses))
        self.roughs = tuple((float(i) for i in self.roughnesses))
        self.demag_locs = tuple((float(i) for i in self.demaglocs))
        self.demag_widths = tuple((float(i) for i in self.demagwidths))
        self.mSLDs = tuple((float(i) for i in self.magSLDs))

        # some class variables required in calc_dzs are defined in the calc_zeds function.
        self.zstart, self.zend, self.points, zeds = self.calc_zeds(self.roughs, self.thicks, self.max_delta_z)
        
        # convert to tuple for caching.
        self.zeds = self.arrtotuple(zeds)

        # get the combined nuclear+/-magnetic SLDs (coherent) and the imaginary SLDs.
        SLDs_micro, iSLDs_micro = self.get_slds()

        # get the thickness of each microslab.
        # uses caching and tuples defined above.
        self.dz = self.calc_dzs(self.zstart, self.zend, self.points, self.indices)

        if self.orientation in ('front'):
            pass
        
        # if self.orientation = back --> slabs will have same thickness, just in reverse order
        elif self.orientation in ('back'):
            self.dz = self.dz[::-1]

        # get the average between each coherent and imaginary SLD value.
        average_slds = 0.5 * np.diff(SLDs_micro) + SLDs_micro[:-1]
        average_islds = 0.5 * np.diff(iSLDs_micro) + iSLDs_micro[:-1]
        
        # init arrays for final SLDs.
        return_slds = np.ones(average_slds.shape[0] + 1)
        return_islds = np.ones(average_islds.shape[0] + 1)
        
        if self.orientation in ('front'):
            # fill all but last with average SLDs.
            return_slds[:-1] = return_slds[:-1] * average_slds
            return_islds[:-1] = return_islds[:-1] * average_islds
            # now set the final sld value to those from the micro arrays.
            return_slds[-1] = SLDs_micro[-1]
            return_islds[-1] = iSLDs_micro[-1]

        elif self.orientation in ('back'):
            # do the same but backwards for back orientations.
            return_slds[1:] = return_slds[1:] * average_slds[::-1]
            return_islds[1:] = return_islds[1:] * average_islds[::-1]
            # now set the final sld value to those from the micro arrays.
            return_slds[0] = SLDs_micro[-1]
            return_islds[0] = iSLDs_micro[-1]

        return return_slds, return_islds, self.dz
    
    def vfs_for_display(self):
        """
        Function useful for plotting: 
        1. Reduced VF profile (defines nuclear SLD profile)
        2. Reduced VF x magnetic composition profile (defines magnetic SLD profile)
        3. Magnetic "deadness" (not reduced).
        
        Notes
        -----
        To plot the the Magnetic "deadness", use the  z_and_SLD_scatter(reduced=False) VFP method.

        Returns
        -------
        reduced_VFP : np.array (2d) - Shape = (Nlayers, len(z) - len(self.indices))
        reduced_magcomp : np.array (2d) - Shape = (Nlayers, len(z) - len(self.indices))
        demag_arr : np.array (2d) - Shape = (Nlayers, len(z))
        """

        self() # use __call__ method to update VFPs if parameters have been updated.

        reduced_VFP, reduced_magcomp, _, demag_arr = self.init_demag(self.demag_locs, 
                                                                       self.demag_widths, 
                                                                       self.mSLDs,
                                                                       self.zeds,
                                                                       self.arrtotuple(self.vfp))
        
        if self.orientation in ('front'):
            pass

        elif self.orientation in ('back'):
            reduced_VFP = reduced_VFP[::-1] # reverse order.

        return reduced_VFP, reduced_magcomp, demag_arr
    
    def SLD_offset(self):
        """
        Returns an offset that can be applied to the z values of a sld_profile 
        of a refnx structure so that the sld_profile will align with the VFPs.

        Returns
        -------
        sldprof_offset : float
        """

        self() # use __call__ method to update VFPs if parameters have been updated.

        if self.orientation in ('front'):
            sldprof_offset_nr = -5 - (4 * self.roughs[0])
            # round down like zstart
            sldprof_offset = np.floor(sldprof_offset_nr * (1 / self.max_delta_z)) / (1 / self.max_delta_z)
        
        elif self.orientation in ('back'): 
            #TODO: follow the calc in calc_zeds?:
            zend_of_vfprofile_nr = np.max((np.cumsum(self.thicks) + 4 * np.array(self.roughs)))
            # round up like zend
            zend_of_vfprofile = np.ceil(zend_of_vfprofile_nr * (1 / self.max_delta_z)) / (1 / self.max_delta_z)

            # refnx sld_profile zend defined by -5 + last slab location + 4 * backing roughness.
            # zend_of_vfprofile replicates the 4 * backing roughness part.
            # Then 5 + last microslice thickness covers the -5 + last slab location part.

            zend_front = 5 + self.dz[-1] + zend_of_vfprofile
            sldprof_offset = -(zend_front - np.cumsum(self.thicks)[-1]) #TODO: simplify - this can be self.thicks.sum()
        
        return sldprof_offset
    
    def surfaces_for_display(self, points=50):
        """
        Produces a 2D array of random variates from each interface.
        The number of random variates is controlled by points.
        Used to create a representation of the modelled interfaces.

        Parameters
        ----------
        points : integer
                 Number of points to simulate across the surfaces.

        Returns
        -------
        interf_list : np.array (2d) - Shape = (Nlayers - 1, points)
        """
        self() # use __call__ method to update VFPs if parameters have been updated.
        
        thicks = self.thicks

        if self.orientation in ('front'):
            interf_loc = np.cumsum(thicks)
            roughs = self.roughnesses

        elif self.orientation in ('back'):
            interf_loc = np.cumsum(thicks)
            interf_loc = np.fabs(interf_loc - interf_loc[-1])[::-1]
            roughs = self.roughnesses[::-1]

        interf_arr = np.ones((len(interf_loc), points))
        num_conform = np.sum(self.conformal)
        idx_where_conformal = np.where(np.array(self.conformal) == 1)[0]
        
        # return the non-conformal interfaces.
        for i in range(0, len(interf_arr)):
            interf_arr[i] = stats.norm.rvs(loc=interf_loc[i], scale=float(roughs[i]), size=points)

        # insert the conformal interfaces.
        if num_conform != 0:
            for i in idx_where_conformal:
                interf_arr[i] = np.max(interf_arr[:i].T, axis=1) + thicks[i]

        return interf_arr

    def plot(self, points=50, microslice_SLD=True, total_SLD=False, total_VF=True):
        """
        Produces a three axis figure on the same x axis.
        Top plot = nSLD / mSLD / iSLD
        Middle plot = volume fraction profiles
        Bottom plot = surface profiles

        Parameters
        ----------
        points : integer
                 Number of points to simulate across the surfaces.
        microslice_SLD : boolean
                         If True, will return SLD profiles equivalent to those generated 
                         with refnx's structure.sld_profile() method
        total_SLD : boolean
                    If True, will return SLD+ or SLD- profiles. If false, the SLDn and SLDm parts
                    will be plotted seperately.
        total_VF : boolean
                   If True, will plot the sum of all layers' volume fractions.

        Returns
        -------
        fig, ax : matplotlib.pyplot figure and axes objects. 
        """
        
        self() # use __call__ method to update VFPs if parameters have been updated.

        if points <= 0:
            raise ValueError("points must be > 0.")

        points = points + 2 # add on two additional points to create fill effect on y axis of bottom plot
        surfaces = self.surfaces_for_display(points=points)
        
        # define some colours to use for the surface plot.
        colours = colormaps['tab20'].colors

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 9))

        # ax[0] - nSLD / mSLD / iSLD. Only plots mSLD and iSLD curves if they are not zero.
        if microslice_SLD == True:
            z, SLDn, SLDm, SLDi = self._gen_sld_profile(max_delta_z=self.max_delta_z)
            
            if total_SLD == True:
                if self.spin_state in ('none'):
                    tot_sld = SLDn
                    ax[0].plot(z + self.SLD_offset(), tot_sld, color='k', label=r'$\mathrm{SLD}_{\mathrm{n}}$')

                elif self.spin_state in ('down'):
                    tot_sld = SLDn - SLDm
                    ax[0].plot(z + self.SLD_offset(), tot_sld, color='k', label=r'$\mathrm{SLD}_{\mathrm{n}} - \mathrm{SLD}_{\mathrm{m}}$')

                elif self.spin_state in ('up'):
                    tot_sld = SLDn + SLDm
                    ax[0].plot(z + self.SLD_offset(), tot_sld, color='k', label=r'$\mathrm{SLD}_{\mathrm{n}} + \mathrm{SLD}_{\mathrm{m}}$')

                if SLDi.any(): # if SLDi contains any number that is not zero
                    ax[0].plot(z + self.SLD_offset(), SLDi, color='tab:red', label=r'$\mathrm{SLD}_{\mathrm{i}}$')

            else:
                ax[0].plot(z + self.SLD_offset(), SLDn, color='k', label=r'$\mathrm{SLD}_{\mathrm{n}}$')
                if SLDm.any():
                    ax[0].plot(z + self.SLD_offset(), SLDm, color='tab:grey', label=r'$\mathrm{SLD}_{\mathrm{m}}$')

                if SLDi.any():
                    ax[0].plot(z + self.SLD_offset(), SLDi, color='tab:red', label=r'$\mathrm{SLD}_{\mathrm{i}}$')
        
        else: # if you want the original non-microsliced SLD profile.
            z, tot_sld, SLDn, SLDm = self.z_and_SLD_scatter()
            SLDi = self.z_and_SLD_scatter(imag=True)[1]
            
            if total_SLD == True:
                if self.spin_state in ('none'):
                    ax[0].plot(z, tot_sld, color='k', label=r'$\mathrm{SLD}_{\mathrm{n}}$')

                elif self.spin_state in ('down'):
                    ax[0].plot(z, tot_sld, color='k', label=r'$\mathrm{SLD}_{\mathrm{n}} - \mathrm{SLD}_{\mathrm{m}}$')     

                elif self.spin_state in ('up'):
                    ax[0].plot(z, tot_sld, color='k', label=r'$\mathrm{SLD}_{\mathrm{n}} + \mathrm{SLD}_{\mathrm{m}}$')

                if SLDi.any():
                    ax[0].plot(z, SLDi, color='tab:red', label=r'$\mathrm{SLD}_{\mathrm{i}}$')

            else:
                ax[0].plot(z, SLDn, color='k', label=r'$\mathrm{SLD}_{\mathrm{n}}$')
                
                if SLDm.any():
                    ax[0].plot(z, SLDm, color='tab:grey', label=r'$\mathrm{SLD}_{\mathrm{m}}$')

                if SLDi.any():
                    ax[0].plot(z, SLDi, color='tab:red', label=r'$\mathrm{SLD}_{\mathrm{i}}$')
        
        # ax[1] - Volume fractions
        vfs = self.vfs_for_display()[0]
        z = self.z_and_SLD_scatter()[0]

        if self.orientation in ('front'):
            for i, j in enumerate(vfs):
                if i == 0:
                    ax[1].plot(z, j.T, label=f'Fronting')

                elif i+1 == len(vfs):
                    ax[1].plot(z, j.T, label=f'Backing')

                else:
                    ax[1].plot(z, j.T, label=f'Layer {i}')
        
        elif self.orientation in ('back'):
            for i, j in enumerate(vfs):
                if i == 0:
                    ax[1].plot(z, j.T, label=f'Backing', color=colours[2*len(self.thicks) - 2*i], zorder=len(self.thicks)-i)

                elif i+1 == len(vfs):
                    ax[1].plot(z, j.T, label=f'Fronting', color=colours[2*len(self.thicks) - 2*i], zorder=len(self.thicks)-i)

                else:
                    ax[1].plot(z, j.T, label=f'Layer {(len(vfs)-1)-(i)}', color=colours[2*len(self.thicks) - 2*i], zorder=len(self.thicks)-i)

        if total_VF == True:
            ax[1].plot(z, np.sum(vfs.T, axis=1), label=r'Total', linestyle='--', color='k')

        # ax[2] - surface plots
        def_xlower_lim, def_xupper_lim = ax[1].get_xlim() # get the default x limits from ax1 before filling

        if self.orientation in ('front'):
            for i, j in enumerate(surfaces): # do the surfaces
                ax[2].plot(j, range(0, points), marker='.', zorder=(2*len(self.thicks)+1 - 2*i))
            
            for i in range(0, len(self.thicks)+1): # then do the fills
                if i == 0:
                    ax[2].fill_betweenx(y=range(0, points), x1=def_xlower_lim-1, x2=surfaces[i], interpolate=True, color=colours[2*i+1], zorder=(2*len(self.thicks) - 2*i))
                
                elif i < len(self.thicks):
                    ax[2].fill_betweenx(y=range(0, points), x1=surfaces[i-1], x2=surfaces[i], where=surfaces[i]>surfaces[i-1], interpolate=True, zorder=(2*len(self.thicks) - 2*i), color=colours[2*i+1])
                
                elif i == len(self.thicks):
                    ax[2].fill_betweenx(y=range(0, points), x1=surfaces[i-1], x2=def_xupper_lim+1, where=def_xupper_lim+1>surfaces[i-1], interpolate=True, zorder=(2*len(self.thicks) - 2*i), color=colours[2*i+1])
        
        elif self.orientation in ('back'):
            for i, j in enumerate(surfaces): # do the surfaces
                ax[2].plot(j, range(0, points), marker='.', color=colours[(len(self.thicks)+2 - 2*i)], zorder=(len(self.thicks)-1 + 2*i))

            for i in range(0, len(self.thicks)+1): # then do the fills
                if i == 0:
                    ax[2].fill_betweenx(y=range(0, points), x1=def_xlower_lim-1, x2=surfaces[i], interpolate=True, color=colours[2*len(self.thicks) + 1 - 2*i], zorder=2*i)
                
                elif i < len(self.thicks):
                    ax[2].fill_betweenx(y=range(0, points), x1=surfaces[i-1], x2=surfaces[i], where=surfaces[i]>surfaces[i-1], interpolate=True, color=colours[2*len(self.thicks) + 1 - 2*i], zorder=2*i)
                
                elif i == len(self.thicks):
                    ax[2].fill_betweenx(y=range(0, points), x1=surfaces[i-1], x2=def_xupper_lim+1, where=def_xupper_lim+1>surfaces[i-1], interpolate=True, color=colours[2*len(self.thicks) + 1 - 2*i], zorder=2*i)
        
        # formatting
        ax[0].legend(frameon=False)
        ax[0].set_ylabel(r'SLD / $\mathrm{\AA{}}^{-2} \times 10^{-6}$')
        ax[1].set_ylabel(r'Volume Fraction')
        ax[1].legend(frameon=False)

        # define some y limits for ax[2] that allow for points > 0.
        ylower = 0.5
        yupper = (points - 2) + ((points - 1) - (points - 2)) / 2

        ax[2].set_xlabel(r'Distance over Interface / $\mathrm{\AA{}}$')
        ax[2].set_yticks([])
        # set the x limits to the original x limits before plotting the fills.
        ax[2].set_xlim(def_xlower_lim, def_xupper_lim) 
        ax[2].set_ylim(ylower, yupper) # chop off the extra two points
        
        for border in ['top', 'bottom', 'left', 'right']:
            ax[2].spines[border].set_zorder((len(self.thicks)+1)*3) # borders will be higher than surfaces and fills.

        return fig, ax

    def slabs(self, structure=None):
        """
        Slab representation of the VFP, as an array.

        Parameters
        ----------
        structure : refnx.reflect.Structure
                    The Structure hosting this VFP component.
        
        Returns
        -------
        slabs : np.array (2d) - Shape = (Nlayers, 5)
        """
        if structure is None:
            raise ValueError("VFP.slabs() requires a valid Structure")
        
        # use the __call__ method of the VFP class to
        # return total slds, islds and thicknesses of each slab
        slds, islds, thicks = self()

        # init a 2D array (Nlayers, 5)
        slabs = np.zeros((len(thicks), 5))
            
        # now populate slabs with microslab thicknesses & SLDs.
        slabs[:, 0] = thicks
        slabs[:, 1] = slds
        slabs[:, 2] = islds
        return slabs

    def _gen_sld_profile(self, max_delta_z):

        # use the __call__ method of the VFP class to
        # return islds and thicknesses of each slab
        _, islds, thicks = self()

        # grab the original nSLD and mSLDs
        _, _, nSLD, mSLD = self.calc_slds()

        # get the average between each nuclear and magnetic SLD value.
        average_nslds = 0.5 * np.diff(nSLD) + nSLD[:-1]
        average_mslds = 0.5 * np.diff(mSLD) + mSLD[:-1]
        
        # init arrays for final SLDs.
        nslds = np.ones(average_nslds.shape[0] + 1)
        mslds = np.ones(average_mslds.shape[0] + 1)
        
        if self.orientation in ('front'):
            # fill all but last with average SLDs.
            nslds[:-1] = nslds[:-1] * average_nslds
            mslds[:-1] = mslds[:-1] * average_mslds
            # now set the final sld value to those from the micro arrays.
            nslds[-1] = nSLD[-1]
            mslds[-1] = mSLD[-1]

        elif self.orientation in ('back'):
            # do the same but backwards for back orientations.
            nslds[1:] = nslds[1:] * average_nslds[::-1]
            mslds[1:] = mslds[1:] * average_mslds[::-1]
            # now set the final sld value to those from the micro arrays.
            nslds[0] = nSLD[-1]
            mslds[0] = mSLD[-1]

        # init a 2D array (Nlayers, 5)
        microslices = np.zeros((len(thicks), 5))
            
        # populate microslices with microslab thicknesses & SLDs.
        microslices[:, 0] = thicks
        microslices[:, 1] = nslds
        microslices[:, 2] = mslds
        microslices[:, 3] = islds

        # calc how many layers, total z distance, start and end points.
        nlayers = np.size(microslices, 0)
        dist = np.cumsum(microslices[:, 0])
        zstart = -5
        zend = 5 + dist[-1]

        # workout how much space the SLD profile should encompass
        # (z array not provided)
        # use twice as many points as the real SLD profile
        max_delta_z = float(max_delta_z) / 2
        npnts = int(np.ceil((zend - zstart) / max_delta_z)) + 1
        zed = np.linspace(zstart, zend, num=npnts)

        # the output arrays
        nsld = np.ones_like(zed, dtype=float) * microslices[0, 1]
        msld = np.ones_like(zed, dtype=float) * microslices[0, 2]
        isld = np.ones_like(zed, dtype=float) * microslices[0, 3]

        # work out the step in SLD at an interface
        # the delta arrays are shape (nlayers - 1)
        delta_nSLD = microslices[1:, 1] - microslices[:-1, 1]
        delta_mSLD = microslices[1:, 2] - microslices[:-1, 2]
        delta_iSLD = microslices[1:, 3] - microslices[:-1, 3]

        # use erf for roughness function, but step if the roughness is zero
        f = Step()
        sigma = microslices[:, 4]

        # accumulate the SLD of each step.
        for i in range(nlayers-1):
            nsld += delta_nSLD[i] * f(zed, scale=sigma[i], loc=dist[i])
            msld += delta_mSLD[i] * f(zed, scale=sigma[i], loc=dist[i])
            isld += delta_iSLD[i] * f(zed, scale=sigma[i], loc=dist[i])
        
        return zed, nsld, msld, isld