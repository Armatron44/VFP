import numpy as np
from refnx.reflect import Structure, Component
from refnx.analysis import Parameter, Parameters, possibly_create_parameter
from scipy import special
from scipy import stats
from methodtools import lru_cache

EPS = np.finfo(float).eps

class VFP(Component):
    """
    
    ### how does this work? ###
    
    In ReflectModel (or MixedReflectModel), the reflectivity function is used to calculate the generative 
    (the simulated reflectivity) for a given set of parameters.
    The generative is used when fitting a dataset, when carrying out posterior sampling & when estimating
    the model evidence.
    
    The reflectivity function requires a slab representation 
    (an array of slab (shape = 2+N, 4 where N is number of layers) parameters - thicknesses, roughnesses etc) 
    of the structure.
    
    This slab representation is returned by the structure.slab() method, which uses the slab method of 
    each component within a structure to return a concatenated array of slabs.
    
    The VFP uses thickness and roughness parameters to build a volume fraction profile of the interface.
    Using SLD parameters (nuclear, magnetic, imaginary), the SLD profile of the interface is calculated
    from the volume fraction profile.
    The SLD profile is a microslabbed approximation of the continuous interface.
    The size of the microslabs is hard-coded at 0.5 Å, set by self.max_delta_z in the __init__ method.
    
    The above workflow requires that the VFP component needs a slab method which will return an array of
    microslabs to the ReflectModel object.
    In the slab method below we use the __call__ method of the VFP component to do this calculation.
    
    Parameters
    ----------
    extent : float or Parameter
        Total length of volume fraction profile.
    nSLDs : array of floats or Parameters    
        Nuclear scattering length densities of each layer within the volume fraction profile.
    niSLDs : array of floats or Parameters    
        Imaginary scattering length densities of each layer within the volume fraction profile.
    mSLDs : array of floats or Parameters    
        Magnetic scattering length densities of each layer within the volume fraction profile.
    thicknesses : tuple of floats or Parameters
        Thicknesses of layers - These control the midpoint-to-midpoint width of a layer's volume fraction
        profile.
    roughnesses : tuple of floats or Parameters
        Roughnesses of layers - These control the width of interfaces between two adjacent layers in the
        volume fraction profile.
    contrast : string
        string used to select which SLDs are used to calculate the scattering length density profile.
        Useful if setting up a global optimisation model with multiple contrasts (but the same structure).
    demaglocs : list of Parameters
        The parameters declare the centre point of a Gaussian CDF.
        The parameters are consecutive, so the z location of parameter 2 will be parameter 1 value 
        + parameter 2 value.
        Must either be a list of 2 or 4 parameters.
    demagwidths : list of Parameters
        The parameters declare the width of a Gaussian CDF.
        Must either be a list of 2 or 4 parameters.
    """

    def __init__(
        self,
        extent,
        nSLDs,
        niSLDs,
        mSLDs,
        thicknesses,
        roughnesses,
        contrast,
        demaglocs=[],
        demagwidths=[]
    ):
        super().__init__() #inherit the Component class.
        
        #hardcode some values
        self.name = ""
        self.max_delta_z = 0.5
        
        #now use inputs to define class variables.
        self.mSLDs = mSLDs
        self.thicknesses = thicknesses
        self.roughnesses = roughnesses
        self.demag_locs = demaglocs
        self.demag_widths = demagwidths
        self.contrast = contrast
        
        #and seperate out the nSLD and niSLDs depending on contrast.
        if self.contrast == 'Air':
            self.nSLDs = np.array([nSLDs[i] for i in [0, 1, 2, 3, 4]])
            self.niSLDs = np.array([niSLDs[i] for i in [0, 1, 2, 3, 4]])
        
        #if extent that is passed to the VFP is a float, turn it into a parameter.
        self.extent = possibly_create_parameter(
            extent, name="%s - VFP extent", units="Å"
        )

    @lru_cache(maxsize=2)
    @classmethod
    def get_dzs(cls, ex, mxdz, thicks, roughs, locs, widths):
        """
        This function returns the thicknesses of each microslice.
        """
        
        def consecutive(indic, stepsize=1): #internal func for finding consecutive indices.
            return np.split(indic, np.where(np.diff(indic) != stepsize)[0]+1) #splits indic into sub arrays where the difference between neighbouring units in indic is not 1.
        
        delta_step = ex/(cls.knots-1) #gives the thickness of the microslabs without the larger slab approximation.
        
        if not cls.ind.any(): #if cls.ind is empty, then dzs is just number of points (cls.knots) * distance per point (delta_step).
            dzs = np.ones(cls.knots)*delta_step
            
        else: #if we have indices, then dzs needs to be altered to include slabs that are > delta_step.
            indexs = consecutive(cls.ind) #list of n arrays (n is the number of zones where there is invariance in demag factor)
            indexs_diffs = [j[-1]-j[0] for j in indexs] #find length of each zone and return in a list.
            indexs_starts = [j[0] for j in indexs] #where does each zone start?
            indexs_ends = [j[-1] for j in indexs] #where does each zone start?
        
            #calculate the distance between indicies of interest in units of indices.
            index_gaps = np.array([j - (indexs_ends[i-1]+1) for i, j in enumerate(indexs_starts) if i > 0])
            new_knots = (cls.knots) - (np.array(indexs_diffs).sum()+len(indexs))
            new_indexs_starts = [indexs_starts[0] + index_gaps[:i].sum() for i, j in enumerate(indexs)]
            
            dzs = np.ones(new_knots)*delta_step #init an array for dzs. make all values delta step to begin with.
            if len(new_indexs_starts) > 1:
                for i, j in enumerate(new_indexs_starts): #find places where delta step needs to be altered.
                    dzs[j] = ((indexs_diffs[i]+1)*delta_step)+dzs[j-1]
            else:
                dzs[int(new_indexs_starts[0])] = ((indexs_diffs[0]+1)*delta_step)+dzs[int(new_indexs_starts[0]-1)] #alter dz in the one place required.
        return dzs
        
    def get_x_and_y_scatter(self, imag=False, reduced=True): #utility function.
        """
        use the imag argument to get imaginary SLD profile nodes.
        use reduced argument to get distance array without removing slabs of similar SLD.
        """
        if reduced is True:
            if imag is False:
                y = np.array([float(i) for i in self.get_slds()[0]])
                x = np.delete(np.array([float(i) for i in self.get_zeds(self.roughnesses, self.thicknesses)]), self.indices)
            else:
                y = np.array([float(i) for i in self.get_slds()[1]])
                x = np.delete(np.array([float(i) for i in self.get_zeds(self.roughnesses, self.thicknesses)]), self.indices)
            return x, y
        else:
            x = np.array([float(i) for i in self.get_zeds(self.roughnesses, self.thicknesses)])
            return x
            
    @classmethod
    def get_zeds(cls, rough, thick): #return a list of zeds for volume fractions calculations.
        cls.knots = int(cls.ex/cls.mxdz)
        zstart = -50 - (4 * rough[0]) #zstart --> can be problematic if thick[1]-(rough[1]x4) < -constant - (4 x rough[0]) - therefore, make constant large enough.
        zend = 5 + np.cumsum(thick)[-1] + 4 * rough[-1] #similar problem here if rough[-1] is << rough[-2]
        zed = np.linspace(float(zstart), float(zend), num=cls.knots)
        return zed
        
    @classmethod
    def get_erf(cls, layer_choice, loc, rough, thick): #return erf for a specific layer
        erf = (1-np.array([0.5*(1 + special.erf((float(i)-loc[layer_choice])/
               float(rough[layer_choice])/np.sqrt(2))) for i in cls.get_zeds(rough, thick)]))
        return erf 
   
    @lru_cache(maxsize=2)
    @classmethod
    def get_vfs(cls, rough, thick, ex, mxdz):
        """
        This function calculates the volume fraction profile for the model.
        The profile has to be editted to match the intended model.
        The integer supplied in the brackets is the layer identifier, 
        where 0 is the fronting layer.
        """
        
        rough = np.array(rough)
        thick = np.array(thick)
        loc = np.cumsum(thick)
        cls.ex = ex
        cls.mxdz = mxdz
        
        #get_erf is a function to calculate the Gaussian CDF profile given the thickness and roughness parameters.
        cls.vfs = np.array([cls.get_erf(0, loc, rough, thick)*cls.get_erf(1, loc, rough, thick)*cls.get_erf(2, loc, rough, thick)*cls.get_erf(3, loc, rough, thick), #air
                            (1-cls.get_erf(0, loc, rough, thick))*cls.get_erf(1, loc, rough, thick)*cls.get_erf(2, loc, rough, thick)*cls.get_erf(3, loc, rough, thick), #FeOx
                            (1-cls.get_erf(1, loc, rough, thick))*cls.get_erf(2, loc, rough, thick)*cls.get_erf(3, loc, rough, thick), #Fe
                            (1-cls.get_erf(2, loc, rough, thick))*cls.get_erf(3, loc, rough, thick), #SiO2
                            (1-cls.get_erf(3, loc, rough, thick)) #Si
                            ])
        return cls.vfs
    
    @lru_cache(maxsize=2)
    @classmethod
    def init_demag(cls, rough, thick, locs, widths):
        """
        Function to reduce the volume fraction and declare if any layers that are 
        magnetic have any region of no magnetic character.
        """
        rough = np.array(rough)
        thick = np.array(thick)
        
        dist = cls.get_zeds(rough, thick)
        locs = np.array(locs)
        widths = np.array(widths)
        
        no_demag_factor = np.ones_like(dist)
        
        #use the following function to model "dead" structure in magnetic layers.
        #it should differ from unity if there are peaks and widths supplied.
        demag_factor = cls.get_demag(dist, locs, widths)
        
        #using demag factor and no_demag_factor, we declare if any regions have some/no magnetic character.
        demag_arr = np.vstack((no_demag_factor, no_demag_factor, no_demag_factor, no_demag_factor, no_demag_factor)) #as we have 4 layers, we must supply 4 demag_factors.
        mag_comp = cls.vfs*demag_arr #calculate magnetic composition of each layer over the interface using volume fraction profiles.
        
        #find the regions of the interface where the volume fraction profiles are approximately invariant.
        #handles all regions that are invariant.
        indices_full = np.nonzero((np.abs(np.diff(mag_comp, axis=1)) < 1e-8) & (mag_comp[:, :-1] > 0.5))
        cls.ind = indices_full[1]
        
        #now remove parts of the vfps and mag_comp where they are invariant.
        reduced_vfs = np.delete(cls.vfs, cls.ind, 1)
        reduced_magcomp = np.delete(mag_comp, cls.ind, 1)
        
        return reduced_vfs, reduced_magcomp, cls.ind, demag_factor
    
    @classmethod
    def get_demag(cls, dist, locs, widths):
        """
        Function that creates either one or two peaks depending on demag_locs & demag_widths.
        """
        if not locs.any():#if loc does not contain non-zero parameters, then just return an array of ones.
            return np.ones_like(dist)
        
        #check if any parameters in locs are non-zero and check if all parameters are non-zero (allow if they are not). 
        #Also checks if only two loc parameters given and allows if true.
        elif locs.any() and not locs.all() or len(locs) == 2:
            split_loc = np.split(locs, len(locs)/2)
            split_width = np.split(widths, len(widths)/2)
            for i, j in enumerate(split_loc):
                if j.all():
                    peak = stats.norm.cdf(dist, loc=j[0], scale=split_width[i][0])*(1-stats.norm.cdf(dist, loc=j[0]+j[1], scale=split_width[i][1]))
                    return 1-peak
                    
        # if demag_locs and demag_widths have 4 values in them, can return two peaks.
        else:
            peak_1 = stats.norm.cdf(dist, loc=locs[0], scale=widths[0])*(1-stats.norm.cdf(dist, loc=locs[0]+locs[1], scale=widths[1]))
            peak_2 = stats.norm.cdf(dist, loc=locs[0]+locs[1], scale=widths[1])*stats.norm.cdf(dist, loc=locs[0]+locs[1]+locs[2], scale=widths[2])*(1-stats.norm.cdf(dist, loc=locs[0]+locs[1]+locs[2]+locs[3], scale=widths[3]))
            demag_f = 1-(peak_1+peak_2) #can have two dead layers.
            return demag_f
    
    def get_slds(self):
        """
        Returns the total and imaginary SLD of given contrasts.
        """
        
        #get floats of parameters so hashing recognition works...
        thicks = tuple(np.array([float(i) for i in self.thicknesses]))
        roughs = tuple(np.array([float(i) for i in self.roughnesses]))
        demag_locs = tuple(np.array([float(i) for i in self.demag_locs]))
        demag_widths = tuple(np.array([float(i) for i in self.demag_widths]))
        
        #calculate the volume fraction profiles of the layers in the interface.
        #this function returns a class variable, and so we don't assign a value to its output.
        self.get_vfs(roughs, thicks, self.extent.value, self.max_delta_z)
        
        #using cls.vfs from the above function, calculate reduced volume fraction and magnetic profiles.
        red_vfs, demagf, self.indices = self.init_demag(roughs, thicks, demag_locs, demag_widths)[:3]
        
        #now we calculate the nuclear (real & imaginary) SLD profiles.
        sldn_values = [float(i) for i in self.nSLDs]
        isldn_values = [float(i) for i in self.niSLDs]
        sldn_list = red_vfs.T*sldn_values
        isldn_list = red_vfs.T*isldn_values
        sum_sldn_list = np.sum(sldn_list, 1)
        sum_isldn_list = np.sum(isldn_list, 1)
        
        #now calculate the magnetic SLD profile.
        # TODO: insert some kind of check on SLDm being empty?
        sldm_values = [float(i) for i in self.mSLDs]
        sldm_list = demagf.T*sldm_values
        sum_sldm_list = np.sum(sldm_list, 1)
        
        #now total the nuclear and magnetic SLDs on given contrast.
        #tot sld must either be addition or subtraction.
        if self.contrast in ('Air'):
            tot_sld = sum_sldn_list + sum_sldm_list
        return tot_sld, sum_isldn_list

    def __call__(self):
        """
        This is the main function of the VFP component.
        
        Here we get the slds from the volume fractions,
        then we find the average slds between consecutive points.
        """
        #we have to convert the following to tuples for the purposes of caching. 
        thicks = tuple(np.array([float(i) for i in self.thicknesses]))
        roughs = tuple(np.array([float(i) for i in self.roughnesses]))
        demag_locs = tuple(np.array([float(i) for i in self.demag_locs]))
        demag_widths = tuple(np.array([float(i) for i in self.demag_widths]))
        
        #get the combined nuclear+/-magnetic SLDs and the imaginary SLDs.
        self.SLDs_micro, self.iSLDs_micro = self.get_slds()
        
        #get the thickness of each microslab.
        #uses caching and tuples defined above.
        self.dz = self.get_dzs(self.extent.value, self.max_delta_z, thicks, roughs, demag_locs, demag_widths)
        
        #now we take the average of each sld value
        average_slds = 0.5*np.diff(self.SLDs_micro)+self.SLDs_micro[:-1]
        average_islds = 0.5*np.diff(self.iSLDs_micro)+self.iSLDs_micro[:-1]
        
        return_slds = np.append(average_slds, self.SLDs_micro[-1])
        return_islds = np.append(average_islds, self.iSLDs_micro[-1])
        
        return return_slds, return_islds, self.dz

    @property
    def parameters(self):
        p = Parameters(name=self.name)
        p.extend([self.extent])
        return p
        
    def p_equivs(self):
        #as slds and dzs are not automatically returned as parameters
        #use this function to return the parameter values after fitting.
        dzs_par_list = Parameters(name='dzs')
        vs_par_list = Parameters(name='slds')
        for i, j in enumerate(self.dz):
            pdz = Parameter(value=j)
            dzs_par_list.append(pdz)
            pvs = Parameter(value=self.SLDs_micro[i])
            vs_par_list.append(pvs)
        p = Parameters(name=self.name)
        p.extend([self.extent, dzs_par_list, vs_par_list])
        return p
    
    def vfs_for_display(self):
        thicks = tuple(np.array([float(i) for i in self.thicknesses]))
        roughs = tuple(np.array([float(i) for i in self.roughnesses]))
        demag_locs = tuple(np.array([float(i) for i in self.demag_locs]))
        demag_widths = tuple(np.array([float(i) for i in self.demag_widths]))
        
        volfracs = self.init_demag(roughs, thicks, demag_locs, demag_widths)[0]
        demagf = self.init_demag(roughs, thicks, demag_locs, demag_widths)[1]
        demag_peak = self.init_demag(roughs, thicks, demag_locs, demag_widths)[3]
        
        return volfracs, demagf, demag_peak
        
    def logp(self):
        return 0

    def slabs(self, structure=None):
        """
        Slab representation of the VFP, as an array.

        Parameters
        ----------
        structure : refnx.reflect.Structure
            The Structure hosting this VFP component.
        """
        if structure is None:
            raise ValueError("VFP.slabs() requires a valid Structure")
        
        slds, islds, thicks = self() #use the __call__ method of the VFP class.
        slabs = np.zeros((len(thicks), 5)) #init a 2D array (N x 5), where N = the length of the microslabs
        
        #now populate slabs with microslab thicknesses & SLDs.
        slabs[:, 0] = thicks
        slabs[:, 1] = slds
        slabs[:, 2] = islds
        return slabs