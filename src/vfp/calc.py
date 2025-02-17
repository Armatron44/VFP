# standard
from functools import lru_cache
import itertools

# third party
import numpy as np
import scipy

def consecutive(arr: np.ndarray) -> list[np.ndarray]:
    """
    Splits an array into a list of 1d arrays. 
    
    Splitting occurs where the difference between 
    neighbouring values is not +1.
    
    Parameters
    ----------
    arr : np.ndarray
        Array to be split where the values are not consecutive.
    
    Returns 
    -------
    list[np.ndarray]
        A list of arrays that all have consecutive values. 
    
    Examples
    --------
    >>> import numpy as np
    >>> from vfp.calc import consecutive
    >>> idxs = np.array([1, 2, 3, 5, 6])
    >>> consecutive(idxs)
    [array([1, 2, 3]), array([5, 6])]
    """
    
    return np.split(arr, (np.diff(arr) != 1).nonzero()[0] + 1)

@lru_cache(maxsize=2)
def calc_dzs(
    zstart: float, zend: float, points: int, idxs: tuple[int]
) -> np.ndarray:
    """
    Calculates the thickness (z) of each microslice.

    By default, the thickness of each microslice is
    (-`zstart` + `zend`) / (`points` - 1).
    
    However, where a microslice's index is defined in `idxs`,
    the microslice is combined with the previous microslice.  

    Parameters
    ----------
    zstart : float
        z value of where VFP starts.
    zend : float
        z value of where VFP ends.
    points : int
        number of microslices in the VFP model.
    idxs : tuple[int]
        indices of nodes in the VFP that are approximately equal to a neighbouring node
        as defined in self.init_demag(). These indices are used to calculate the thickness
        of each microslice across an uneven z space after reduction.

    Returns
    -------
    np.array
        microslice thicknesses (1d).
        
    Examples
    --------
    >>> import numpy as np
    >>> from vfp.calc import calc_dzs
    >>> indices = (2, 3, 4) # the 3rd, 4th and 5th microslices have same SLD.
    >>> calc_dzs(zstart=-3, zend=3, points=7, idxs=indices)
    array([1., 1., 4., 1.])
    """

    idxs = np.array(idxs)

    # find thickness of microslabs without reduction.
    delta_step = (-zstart + zend) / (points - 1)

    # if idxs is empty, then each dz is 1 * delta_step.
    if not idxs.any():
        dzs = np.ones(points) * delta_step

    # if there are indices, then dzs needs to be altered
    # to include slabs that are > delta_step
    else:
        indexs = consecutive(idxs)
        indexs_diffs = [
            j[-1] - j[0] for j in indexs
        ]  # find length of each zone and return in a list.
        indexs_starts = [j[0] for j in indexs]  # where does each zone start?
        indexs_ends = [j[-1] for j in indexs]  # where does each zone end?

        # calculate the distance between indicies of interest
        index_gaps = np.array(
            [
                j - (indexs_ends[i - 1] + 1)
                for i, j in enumerate(indexs_starts)
                if i > 0
            ]
        )
        # number of slabs required.
        new_points = points - (np.array(indexs_diffs).sum() + len(indexs))
        new_indexs_starts = [
            indexs_starts[0] + index_gaps[:i].sum()
            for i in range(0, len(indexs))
        ]

        # init an array for dzs. make all values delta step to begin with.
        dzs = np.ones(new_points) * delta_step

        # find places where delta step needs to be altered.
        if len(new_indexs_starts) > 1:
            for i, j in enumerate(new_indexs_starts):
                dzs[j] = ((indexs_diffs[i] + 1) * delta_step) + dzs[j - 1]

        # alter dz in the one place required.
        else:
            dzs[int(new_indexs_starts[0])] = (
                (indexs_diffs[0] + 1) * delta_step
            ) + dzs[int(new_indexs_starts[0] - 1)]

    return dzs

@lru_cache(maxsize=2)
def calc_zeds(
    rough: tuple[float], thick: tuple[float], mxdz: float
) -> np.ndarray:
    """
    Calculates the z values over which the interface is defined.
    
    The range of z values is defined by `thick` and `rough`,
    while the spacing is defined by `mxdz`.

    Parameters
    ----------
    rough : tuple[float]
        Roughnesses of layers in model. 
        Used in calculation of the start and end of zeds.
    thick : tuple[float]
        Thicknesses of layers in model. 
        Used to calculate the end of the zeds.
    mxdz : float
        Used to calculate the number of points in returned array.

    Returns
    -------
    np.array
        Distance points.
    
    Examples
    --------
    >>> from vfp.calc import calc_zeds    
    >>> zs = calc_zeds(rough=(1,), thick=(2,), mxdz=0.5)
    >>> print(zs)
    [-9.  -8.5 -8.  -7.5 -7.  -6.5 -6.  -5.5 -5.  -4.5 -4.  -3.5 -3.  -2.5
     -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2.   2.5  3.   3.5  4.   4.5
      5.   5.5  6.   6.5  7.   7.5  8.   8.5  9.   9.5 10.  10.5 11. ]
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
    zend_nr = 5 + zend_of_vfprofile  # add a small offset

    # set to the next higher multiple of mxdz
    zend = np.ceil(zend_nr * (1 / mxdz)) / (1 / mxdz)

    # calculate number of points required in z array.
    points = np.rint((-zstart + zend) / mxdz + 1).astype(int)

    zeds = np.linspace(zstart, zend, num=points)

    return zeds

def one_minus_cdf(
    z: np.ndarray, cumthick: np.ndarray, rough: np.ndarray
) -> np.ndarray:
    """
    Returns the inverse (1-CDF) of a normal CDF. 
    
    The CDF is defined by the cumulative thicknesses, `cumthick`, of the layers
    and their roughnesses, `rough`.

    Parameters
    ----------
    z : np.array
        z values over which VFP will be calculated.
    cumthick : np.array
        cumulative thicknesses of the layers.
    rough : np.array
        roughnesses of the model layers.

    Returns
    -------
    np.array
        1-CDF
    
    """
    # TODO: add other distribution types to this function?
    # exponential, uniform (straight line CDF), ...
    one_minus_cdf = 1 - scipy.stats.norm.cdf(z, loc=cumthick, scale=rough)
    return one_minus_cdf

@lru_cache(maxsize=2)
def calc_vfp(
    rough: tuple[float], 
    thick: tuple[float], 
    zeds: tuple[float], 
    conformal: tuple[int]
) -> np.ndarray:
    """
    Returns the volume fraction profile for each layer.

    Parameters
    ----------
    thick : tuple[float]
        thicknesses values of the layers.
    rough : tuple[float]
        roughnesses values of the layers.
    zeds : tuple[float]
        z values across VFP.
    conformal : tuple[int]
        sequence of 0 and 1s.
        1 indicates conformal interface to everything before,
        0 indicates non-conformal interface.

    Returns
    -------
    np.array
        VFP values. Shape = (Nlayers, len(z))
    """
    rough = np.array(rough)
    thick = np.array(thick)
    z = np.array(zeds)
    conformal = np.array(conformal)

    cumthick = np.cumsum(thick)
    num_layers = len(thick) + 1

    # identify conformal interfaces and get the number of
    # consecutive non-conformal and conformal interfaces.
    # the length of the two lists is 2*n where n is a
    # batch of consecutive conformal interfaces.
    check_conform = []
    part_count = []
    for k, g in itertools.groupby(conformal):
        check_conform.append(k)
        part_count.append(sum(1 for _ in g))

    # init vfp array to fill.
    vfp = np.zeros((num_layers, len(z)), dtype=float)
    prior_surface = np.ones(
        (num_layers, len(z)), dtype=float
    )  # will be used to keep track of preceeding layers' vfps.

    counter = 0
    for ii, set in enumerate(check_conform):
        if set == 1:
            for part in range(part_count[ii]):
                # init a 2D array with num rows = number of non-conformal interfaces
                # before this conformal interface, but after the preceeding conformal interface.
                prior_nonconform = np.ones((part_count[ii - 1], len(z)))
                for prior_part in range(part_count[ii - 1]):
                    prior_nonconform[prior_part] = 1 - one_minus_cdf(
                        z,
                        cumthick[counter - prior_part - 1 - part]
                        + cumthick[counter]
                        - cumthick[counter - part - 1],
                        rough[counter - prior_part - 1 - part],
                    )
                full_int = np.cumprod(prior_nonconform, axis=0)
                vf_sum = np.cumsum(vfp, axis=0)
                # vfp of conformal materials is calculated as:
                # 1 - cumprod(nonconform_CDF_shifted) - sum of vfps of all previous layers.
                vfp[counter, :] = 1 - full_int[-1] - vf_sum[-1]
                # update prior_surface for next iteration if nonconformal interface.
                # for materials after a conformal interface,
                # the prior_surface is cumprod(nonconform_CDF_shifted)
                prior_surface[counter + 1, :] = full_int[-1]
                counter += 1

        else:
            for part in range(part_count[ii]):
                end_interf = one_minus_cdf(
                    z, cumthick[counter], rough[counter]
                )
                start_interf = prior_surface[counter, :]
                vfp[counter, :] = end_interf * start_interf
                # update prior_surface for next iteration.
                # it takes the value of current CDF multiplied by
                # the cumulative product of all CDFs before.
                prior_surface[counter + 1, :] = (
                    1 - end_interf
                ) * prior_surface[counter]
                counter += 1

    # calculate the backing material vfp.
    vf_sum = np.cumsum(vfp, axis=0)
    vfp[counter, :] = (
        1 - vf_sum[-1]
    )  # the backing material is simply 1-everything else.
    return vfp

@lru_cache(maxsize=2)
def init_demag(
    locs: tuple[float], 
    widths: tuple[float], 
    mSLDs: tuple[float], 
    zeds: tuple[float], 
    vfp: tuple[tuple[float]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the product of the VFP and the demagnetisation factor.
    
    Regions in mag_comp and the VFP are then removed if the 
    difference between neighbouring units in :math:`mag_comp < 10^{-5}'. 
    These are referred to as the "reduced" VFP and `mag_comp`.
    
    Where these regions have been deleted, a record is kept in `idxs`
    for use in `calc_dzs`.

    Returns the following:
    1. reduced_vfp and reduced_magcomp - used in the calculation of SLDs.
    2. idxs - the indices of where points were removed from VFP and mag_comp.
    3. demag_arr - demagnetisation array, not reduced.

    Parameters
    ----------
    locs : tuple[float]
        values to describe demagnetisation peak(s) locations.
    widths : tuple[float]
        values to describe demagnetisation peaks(s) widths.
    mSLDs : tuple[float]
        tuple of magnetic SLD values of the layers.
    zeds : tuple[float]
        tuple of z values across VFP.
    vfp : tuple[tuple[float]]
        Nested tuple (2d) containing VFP of each layer.

    Returns
    -------
    np.array
        Reduced VFPs. (2d) - Shape = (Nlayers, len(z) - len(idxs))
    np.array
        Reduced mag_comp. (2d) - Shape = (Nlayers, len(z) - len(idxs))
    np.array
        Indices of where to remove points from vfp and mag_comp.
    np.array
        Shape = (Nlayers, len(z))
        Magnetic demagnetisation before multiplication with VFP.
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
    demag_factor = 1 - get_demag(zeds, locs, widths)

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

def get_demag(
    dist: np.ndarray, locs: np.ndarray, widths: np.ndarray
) -> np.ndarray:
    """
    Calculates the demagnetisation function across the interface. 
    
    The function is described by *n* peaks, where *n* is half the number of
    `locs` and `widths` parameters. The *n peaks are calculated from normal 
    CDFs using `locs` and `widths` to define their location and scale respectively.

    The demagnetisation is set at zero by default, which represents a state of
    no demagnetisation (layers with a magnetic SLD will have the full extent of
    that magnetic SLD applied).

    Introducing peak(s) will increase the magnetic deadness from 0 to
    a range of values between 0--1 across the interface. This will
    reduce the extent of magnetic SLD in the region of the peak(s).

    Parameters
    ----------
    dist : np.array
        z values of VFP.
    locs : np.array
        values of locs parameters.
    widths : np.array
        values of widths parameters.

    Returns
    -------
    np.array
        Peak(s) that describe the demagnetisation factor.
        
    Examples
    --------
    >>> import numpy as np
    >>> from vfp.calc import get_demag
    >>> get_demag(dist=np.linspace(0, 10, 11), 
    >>>           locs=np.array([4, 3]), 
    >>>           widths=np.array([1, 1]))
    array([3.16712418e-05, 1.34989803e-03, 2.27501254e-02, 1.58650229e-01,
           4.99325051e-01, 8.22204042e-01, 8.22204042e-01, 4.99325051e-01,
           1.58650229e-01, 2.27501254e-02, 1.34989803e-03])
    """
    demag_f = np.zeros_like(dist)
    cumlocs = np.cumsum(locs)
    
    # number of peaks is half size of number of locs & width parameters. 
    npeaks = int(cumlocs.size / 2)
    prev_down = 1 - demag_f
    for n in range(npeaks):
        up = scipy.stats.norm.cdf(dist, loc=cumlocs[2*n], scale=widths[2*n])
        down = scipy.stats.norm.cdf(dist, loc=cumlocs[2*n + 1], scale=widths[2*n + 1])
        peak = prev_down * up * (1 - down)
        demag_f += peak
        prev_down = down
    
    return demag_f

@lru_cache(maxsize=2)
def integrate_vfp(
    zeds: tuple[float],
    indexs: tuple[int],
    red_vfps: tuple[float],
    layer_indices: tuple[int]
) -> list[float]:
    """
    Calculates integrals of specific VFP layers.
    
    The integrals that are calculated are specified by `layer_indices`.
    Integration is calculated via Simpson's rule.

    Parameters
    ----------
    zeds : tuple[float]
        tuple of z values across VFP.
    indexs : tuple[int]
        indices of nodes in the VFP that are approximately equal to a neighbouring node
        as defined in self.init_demag(). These indices are used to calculate the
        thickness of each microslice across an uneven z space after reduction.
    red_vfps : tuple[float]
        Nested tuple (2d) containing VFP of each layer.
    layer_indices : tuple[ind]
        Indices of layers of which to calculate the integral.

    Returns
    -------
    list[float]
        The integrals of the layers.
        
    Examples
    --------
    >>> import numpy as np
    >>> import scipy
    >>> from vfp.calc import integrate_vfp
    >>> z = np.linspace(0, 100, 1001)
    >>> pdfs = scipy.stats.norm.pdf(x=z[:, np.newaxis], loc=[35, 60], scale=[3, 4]).T
    >>> pdfs[0] = pdfs[0] * 3 # make the first peak have an integral of 3.
    >>> integrate_vfp(zeds=tuple(z), 
    >>>               indexs=(), 
    >>>               red_vfps=tuple(tuple(i) for i in pdfs), 
    >>>               layer_indices=(0, 1))
    [np.float64(3.0), np.float64(1.0000000000000002)]
    """
    if not layer_indices:
        raise ValueError(f'layer_indices must be defined.')

    zs = np.array(zeds)
    idxs = np.array(indexs)
    red_vfp = np.array(red_vfps)
    layer_indices = list(layer_indices)

    if idxs.size > 0:
        integrate_over = np.delete(
            zs, idxs
        )  # get zed values to integrate over.
    else:
        integrate_over = zs
        
    integrals = []
    for lidx in layer_indices:
        layer_integral = scipy.integrate.simpson(red_vfp[lidx], x=integrate_over)
        integrals.append(layer_integral)    

    return integrals