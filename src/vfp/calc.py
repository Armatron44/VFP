import numpy as np
import scipy
from functools import lru_cache
import itertools

def consecutive(arr, stepsize=1):
    """
    Splits an array into sub arrays where the difference between neighbouring units is not 1.
    """
    return np.split(arr, np.where(np.diff(arr) != stepsize)[0] + 1)

@lru_cache(maxsize=6)
def calc_dzs(zstart, zend, points, idxs):
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
        indexs = consecutive(idxs) # list of n arrays (n is the number of zones where there is invariance in demag factor)
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
    
@lru_cache(maxsize=6)
def calc_zeds(rough, thick, mxdz):
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
    points = np.rint((-zstart + zend)/mxdz + 1).astype(int) 

    zeds = np.linspace(zstart, zend, num=points)
    
    return zstart, zend, points, zeds

def one_minus_cdf(x, cumthick, rough): 
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
    one_minus_cdf = 1-scipy.stats.norm.cdf(x, loc=cumthick, scale=rough)
    return one_minus_cdf

@lru_cache(maxsize=6)
def calc_vfp(rough, thick, zeds, conformal):
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

    # identify conformal interfaces and get the number of consecutive non-conformal and conformal interfaces.
    # the length of the two lists is 2*n where n is a batch of consecutive conformal interfaces.
    check_conform = []
    part_count = []
    for k, g in itertools.groupby(conformal):
        check_conform.append(k)
        part_count.append(sum(1 for _ in g))

    # init vfp array to fill.
    vfp = np.zeros((num_layers, len(z)))
    prior_surface = np.ones((num_layers, len(z))) # will be used to keep track of preceeding layers' vfps.

    counter = 0
    for ii, set in enumerate(check_conform):
        if set == 1:
            for part in range(part_count[ii]):
                # init a 2D array with number of rows = number of non-conformal interfaces 
                # before this conformal interface, but after the preceeding conformal interface.
                prior_nonconform = np.ones((part_count[ii - 1], len(z)))
                for prior_part in range(part_count[ii - 1]):
                    prior_nonconform[prior_part] = 1 - one_minus_cdf(z, 
                                                                     cumthick[counter - prior_part - 1 - part] + 
                                                                     cumthick[counter] - cumthick[counter - part - 1],
                                                                     rough[counter - prior_part - 1 - part])
                full_int = np.cumprod(prior_nonconform, axis=0)
                vf_sum = np.cumsum(vfp, axis=0) 
                # vfp of conformal materials is calculated as 1 - cumprod(nonconform_CDF_shifted) - sum of vfps of all previous layers.  
                vfp[counter, :] = 1 - full_int[-1] - vf_sum[-1]
                # update prior_surface for next iteration if nonconformal interface.
                # for materials after a conformal interface, the prior_surface is cumprod(nonconform_CDF_shifted)
                prior_surface[counter + 1, :] = full_int[-1]
                counter += 1
        
        else:
            for part in range(part_count[ii]):
                end_interf = one_minus_cdf(z, cumthick[counter], rough[counter])
                start_interf = prior_surface[counter, :]
                vfp[counter, :] = end_interf * start_interf
                # update prior_surface for next iteration.
                # it takes the value of current CDF multiplied by the cumulative product of all CDFs before.
                prior_surface[counter + 1, :] = (1 - end_interf) * prior_surface[counter]
                counter += 1
    
    # calculate the backing material vfp.
    vf_sum = np.cumsum(vfp, axis=0)
    vfp[counter, :] = 1 - vf_sum[-1] # the backing material is simply 1-everything else.
    return vfp

@lru_cache(maxsize=6)
def init_demag(locs, widths, mSLDs, zeds, vfp):
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
    demag_arr = np.ones((len(mSLDs), len(zeds)))

    # use the following function to model "dead" structure in magnetic layers.
    # it should differ from unity if there are peaks and widths supplied.
    demag_factor = get_demag(zeds, locs, widths)
    
    # now apply demag_factor to all layers that have a magnetic component.
    for i in range(0, len(mSLDs)):
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

def get_demag(dist, locs, widths):
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
                peak = scipy.stats.norm.cdf(dist, loc=j[0], scale=split_width[i][0]) * (1-scipy.stats.norm.cdf(dist, loc=j[0] + j[1], scale=split_width[i][1]))
                return 1 - peak
                
    # if demag_locs and demag_widths have 4 values in them, can return two peaks.
    else:
        cumlocs = np.cumsum(locs)
        peak_1 = scipy.stats.norm.cdf(dist, loc=cumlocs[0], scale=widths[0]) * (1-scipy.stats.norm.cdf(dist, loc=cumlocs[1], scale=widths[1]))
        peak_2 = scipy.stats.norm.cdf(dist, loc=cumlocs[1], scale=widths[1]) * scipy.stats.norm.cdf(dist, loc=cumlocs[2], scale=widths[2]) * (1-scipy.stats.norm.cdf(dist, loc=cumlocs[3], scale=widths[3]))
        demag_f = 1 - (peak_1 + peak_2) # can have two dead layers.
        return demag_f
    
@lru_cache(maxsize=6)
def integrate_vfp(zeds, indexs, red_vfps, first_layer, second_layer):
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

    first_lay_int = scipy.integrate.simpson(red_vfp[first_layer], x=integrate_over)
    secon_lay_int = scipy.integrate.simpson(red_vfp[second_layer], x=integrate_over)

    return first_lay_int, secon_lay_int