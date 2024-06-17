import numpy as np
import warnings
from calc import calc_dzs, calc_zeds, init_demag, integrate_vfp, calc_vfp
from plotting import model_plot

from refnx.analysis import possibly_create_parameter
from refnx.analysis.parameter import _BinaryOp
from bumps.parameter import Operator 
from bumps.parameter import Parameter as bumpsParameter


class BaseVFP:
    def __init__(self):
        # create vfp model.
        self.process_model()

    def __repr__(self):
        """Returns simple string description of the VFP.

        Returns:
            str : Returns a printable representation of the VFP, describing VFP type, parameters and values.
        """

        s = (f"VFP Name - {self.name} \n"
             f"Thicks - {self.thicknesses} \n"
             f"Roughs - {self.roughnesses_p} \n"
             f"Nuclear SLDs - {self.nucSLDs_p} \n"
             f"Magnetic SLDs - {self.magSLDs_p} \n"
             f"Imaginary SLDs - {self.nuciSLDs_p} \n"
             f"Demag Locations - {self.demaglocs} \n"
             f"Demag Widths - {self.demagwidths} \n")
        return s

    def process_model(self):
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
        # update tuple variants of parameters.
        self._tuple_pars()

        # some class variables required in calc_dzs are defined in the calc_zeds function.
        self.zstart, self.zend, self.points, zeds = calc_zeds(self.roughs, 
                                                              self.thicks, 
                                                              self.max_delta_z)
        
        # convert to tuple for caching.
        self.zeds = self._arrtotuple(zeds)

        # get the combined nuclear+/-magnetic SLDs (coherent) and the imaginary SLDs.
        SLDs_micro, iSLDs_micro = self.get_slds()

        # get the thickness of each microslab.
        # uses caching and tuples defined above.
        self.dz = calc_dzs(self.zstart, self.zend, self.points, self.indices)

        if self.orientation in ('front'):
            pass
        
        # if VFP.orientation = back --> slabs will have same thickness, just in reverse order
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
        self.vfp = calc_vfp(self.roughs, self.thicks, self.zeds, tuple(self.conformal))

        # using vfp from the above function, calculate reduced volume fraction and magnetic profiles.
        self.red_vfp, self.demagf, idx = init_demag(self.demag_locs, 
                                                    self.demag_widths, 
                                                    self.mSLDs, 
                                                    self.zeds,
                                                    self._arrtotuple(self.vfp))[:3]
        
        self.indices = self._arrtotuple(idx)

        # calculate the SLD valus across reduced VFPs.
        SLD, iSLD, _, _ = self.calc_slds()

        return SLD, iSLD
    
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
            first_layer, second_layer = self.SLD_constraint.layer_choice() 
            int_vfp1, int_vfp2 = integrate_vfp(self.zeds, 
                                               self.indices, 
                                               self._arrtotuple(self.red_vfp), 
                                               first_layer, 
                                               second_layer)
            # user defines a class with a callable, 
            # which returns an idx for modifying a particular SLD value.
            layer_loc, SLD = self.SLD_constraint(int_vfp1, int_vfp2) 
            self.nucSLDs[layer_loc] = SLD

        if reduced is True:
            demagf = self.demagf
            sldn_values = [float(i) for i in self.nucSLDs]
            isldn_values = [float(i) for i in self.nuciSLDs]
            sldn_list = self.red_vfp.T * sldn_values
            isldn_list = self.red_vfp.T * isldn_values

        else:
            demagf = self.vfs_for_display()[2]
            sldn_values = [float(i) for i in self.nucSLDs]
            isldn_values = [float(i) for i in self.nuciSLDs]
            sldn_list = self.vfp.T * sldn_values
            isldn_list = self.vfp.T * isldn_values

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
        
        return tot_sld, sum_isldn_list, sum_sldn_list, sum_sldm_list
    
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

        # update the model. Captures instances where parameters have changed.
        self.process_model()

        reduced_VFP, reduced_magcomp, _, demag_arr = init_demag(self.demag_locs, 
                                                                self.demag_widths, 
                                                                self.mSLDs,
                                                                self.zeds,
                                                                self._arrtotuple(self.vfp))
        
        if self.orientation in ('front'):
            pass

        elif self.orientation in ('back'):
            reduced_VFP = reduced_VFP[::-1] # reverse order.

        return reduced_VFP, reduced_magcomp, demag_arr
    
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
        # update the model. Captures instances where parameters have changed.
        self.process_model()

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
            offset = np.sum(self.thicks)
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
    
    def SLD_offset(self):
        """
        Returns an offset that can be applied to the z values of a sld_profile 
        of a refnx structure so that the sld_profile will align with the VFPs.

        Returns
        -------
        sldprof_offset : float
        """

        # update the model. Captures instances where parameters have changed.
        self.process_model()

        if self.orientation in ('front'):
            sldprof_offset_nr = -5 - (4 * self.roughs[0])
            # round down like zstart
            sldprof_offset = np.floor(sldprof_offset_nr * (1 / self.max_delta_z)) / (1 / self.max_delta_z)
        
        elif self.orientation in ('back'): 
            zend_of_vfprofile_nr = np.max(np.sum(self.thicks) + 4 * np.array(self.roughs)) + 5
            # round up like zend
            zend_of_vfprofile = np.ceil(zend_of_vfprofile_nr * (1 / self.max_delta_z)) / (1 / self.max_delta_z)

            # sld_profile zend defined by -5 + last slab location + 4 * backing roughness.
            # zend_of_vfprofile replicates the 4 * backing roughness part.
            # Then 5 + last microslice thickness covers the -5 + last slab location part.
            zend_front = self.dz[-1] + zend_of_vfprofile
            sldprof_offset = -(zend_front - np.sum(self.thicks))
        
        return sldprof_offset

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
        
        # update the model. Captures instances where parameters have changed.
        self.process_model()

        fig, ax = model_plot(VFP=self,
                             points=points,
                             microslice_SLD=microslice_SLD, 
                             total_SLD=total_SLD, 
                             total_VF=total_VF)

        return fig, ax
    
    def _tuple_pars(self):
        self.thicks = tuple((float(i) for i in self.thicknesses))
        self.roughs = tuple((float(i) for i in self.roughnesses))
        self.demag_locs = tuple((float(i) for i in self.demaglocs))
        self.demag_widths = tuple((float(i) for i in self.demagwidths))
        self.mSLDs = tuple((float(i) for i in self.magSLDs))
    
    def _arrtotuple(self, arr):
        """
        Takes 1D/2D arrays and returns a tuple/nested tuple for the purposes of caching.
        """
        if arr.ndim == 1:
            return tuple(i for i in arr)
        
        elif arr.ndim == 2:
            return tuple([tuple([float(i) for i in row]) for row in arr])
        
    def _createparam(self, param, nameid):
        output = []
        if self.name in ('refnx VFP'):
            if nameid in ("magSLD", "niSLD"):
                if (param != 0).any():
                    for k, i in enumerate(param[param != 0]):
                        if isinstance(i, _BinaryOp):  # can't handle _BinaryOp so warn user that the parameters that define must go to auxiliary params
                            warnings.warn(
                            "Pass magSLD / nuciSLD parameters that are only part of a parameter operation (i.e f(p1, p2) = p1 + p2) to the auxiliary parameters argument of the objective."
                            )
                        else:
                            output.append(possibly_create_parameter(i, name=f"{self.name} - Layer{k} - {nameid}"))
            
            else:
                for k, i in enumerate(param):
                    if isinstance(i, _BinaryOp):
                        warnings.warn(
                        "Pass nucSLD parameters that are only part of a parameter operation (i.e f(p1, p2) = p1 + p2) to the auxiliary parameters argument of the objective."
                        )
                    else:
                        output.append(possibly_create_parameter(i, name=f"{self.name} - Layer{k} - {nameid}"))
        
        elif self.name in ('refl1d VFP'):
            if nameid in ("magSLD", "niSLD"):
                if (param != 0).any():
                    for k, i in enumerate(param[param != 0]):
                        if isinstance(i, bumpsParameter):
                            output.append(i)
                        elif isinstance(i, Operator):
                            warnings.warn(
                            "If magSLD / nuciSLD parameters are part of a function (i.e f(p1, p2) = p1 + p2), they must be of type bumps.parameter.Parameter. Do not use material or SLD objects.."
                            )
                            output.extend(i.parameters())
                        elif not isinstance(i, bumpsParameter) and not isinstance(i, Operator):
                            output.append(bumpsParameter(i, name=f"{self.name} - Layer{k} - {nameid}"))
            
            else:
                for k, i in enumerate(param):
                    if isinstance(i, bumpsParameter):
                        output.append(i)
                    elif isinstance(i, Operator): 
                        warnings.warn(
                            "If nucSLD parameters are part of a function (i.e f(p1, p2) = p1 + p2), they must be of type bumps.parameter.Parameter. Do not use material or SLD objects."
                                    )
                        output.extend(i.parameters())
                    elif not isinstance(i, bumpsParameter) and not isinstance(i, Operator):
                        output.append(bumpsParameter(i, name=f"{self.name} - Layer{k} - {nameid}"))

        return output