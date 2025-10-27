# standard
import itertools
from functools import lru_cache

# third party
import numpy as np
import scipy

# microslices of diff 1e-5 with neighbouring slices are equivalent.
MICROSLICE_EQUIVALENCE_THRESHOLD = 1e-5


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
    if arr.size > 1:
        return np.split(arr, (np.diff(arr) != 1).nonzero()[0] + 1)
    else:
        return [arr]


@lru_cache(maxsize=2)
def calc_dzs(
    zstart: float, zend: float, points: int, idxs: tuple[int, ...]
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
    idxs : tuple[int, ...]
        indices of differences between neighbouring pairs in the VFP that are
        approximately equal to a neighbouring node as defined in
        `self.init_demag`. These indices are used to calculate the thickness
        of each microslice across an uneven z space after reduction.
        dz calculated over idxs i, i+1, ..., n will be (1 + n - i) * spacing.

    Returns
    -------
    np.array
        microslice thicknesses (1d).

    Examples
    --------
    >>> import numpy as np
    >>> from vfp.calc import calc_dzs
    >>> np.linspace(-3, 3, 7) # location of z points
    array([-3, -2, -1, 0, 1, 2, 3])
    >>> indices = (2, 3, 4) # skip the diffences at indices 2, 3 & 4.
    >>> dzs = calc_dzs(zstart=-3, zend=3, points=7, idxs=indices)
    >>> dzs
    array([1., 1., 3., 1.])
    >>> -3 + dzs.sum()
    np.float64(3.0)
    """
    idxs = np.array(idxs)

    # find thickness of microslabs without reduction.
    delta_step = (-zstart + zend) / (points - 1)

    # set up thicknesses.
    dzs = np.ones(points - 1) * delta_step
    # if there are indices, then dzs needs to be altered
    # to include slabs that are > delta_step
    if idxs.any():
        indexs = consecutive(idxs)
        block_thicks = np.array([delta_step * (arr.size) for arr in indexs])
        indexs_starts = np.array([j[0] for j in indexs])
        dzs[indexs_starts] = block_thicks
        indices_to_remove = np.concatenate([arr[1:] for arr in indexs])
        dzs = np.delete(dzs, indices_to_remove)
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

    The CDF is defined by the cumulative thicknesses, `cumthick`, of the
    layers and their roughnesses, `rough`.

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
    conformal: tuple[int],
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

    # group consecutive non-conformal and conformal
    # interfaces into batches and get the number of
    # interfaces in these batches.
    batch_conformal = []
    batch_count = []
    for is_conformal, batcher in itertools.groupby(conformal):
        batch_conformal.append(is_conformal)
        batch_count.append(len(list(batcher)))

    # init vfp array to fill.
    vfp = np.zeros((num_layers, len(z)), dtype=float)
    prior_surface = np.ones(
        (num_layers, len(z)), dtype=float
    )  # will be used to keep track of preceeding layers' vfps.

    counter = 0
    for i, is_conformal in enumerate(batch_conformal):
        if is_conformal == 1:
            for num_interf in range(batch_count[i]):
                # init a 2D array with num rows = number of non-conformal
                # interfaces before this conformal interface, but after the
                # preceeding conformal interface.
                prior_nonconform = np.ones((batch_count[i - 1], len(z)))
                for prior_part in range(batch_count[i - 1]):
                    prior_nonconform[prior_part] = 1 - one_minus_cdf(
                        z,
                        cumthick[counter - prior_part - 1 - num_interf]
                        + cumthick[counter]
                        - cumthick[counter - num_interf - 1],
                        rough[counter - prior_part - 1 - num_interf],
                    )
                full_int = np.cumprod(prior_nonconform, axis=0)
                vf_sum = np.cumsum(vfp, axis=0)
                vfp[counter, :] = 1 - full_int[-1] - vf_sum[-1]
                # update prior_surface for next iteration if nonconformal
                # interface. For materials after a conformal interface,
                # the prior_surface is cumprod(nonconform_CDF_shifted)
                prior_surface[counter + 1, :] = full_int[-1]
                counter += 1

        else:
            for _ in range(batch_count[i]):
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
    mslds: tuple[float],
    zeds: tuple[float],
    vfp: tuple[tuple[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the product of the VFP and the demagnetisation factor.

    Regions in mag_comp and the VFP are then removed if the
    difference between neighbouring units in `mag_comp` :math:`< 10^{-5}`.
    These are referred to as the "reduced" VFP and `mag_comp`.

    Where these regions have been deleted, a record is kept in `idxs`
    for use in `calc_dzs`.

    Returns the following:
    1 & 2. reduced_vfp and reduced_magcomp - used in the calculation of SLDs.
    3. idxs - the indices of where points were removed from VFP and mag_comp.
    4. demag_arr - demagnetisation array, not reduced.

    Parameters
    ----------
    locs : tuple[float]
        values to describe demagnetisation peak(s) locations.
    widths : tuple[float]
        values to describe demagnetisation peaks(s) widths.
    mslds : tuple[float]
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
        Indices of where vfp is ~ invariant with next neighbouring point.
    np.array
        Shape = (Nlayers, len(z))
        Magnetic demagnetisation before multiplication with VFP.
        Not reduced.
    """
    locs = np.array(locs)
    widths = np.array(widths)
    mslds = np.array(mslds)
    zeds = np.array(zeds)
    vfp = np.array(vfp)

    # init an array for any magnetic deadness.
    demag_arr = np.ones((len(mslds), len(zeds)))

    # use the following function to model "dead" structure in magnetic layers.
    # it should differ from unity if there are peaks and widths supplied.
    demag_factor = 1 - get_demag(zeds, locs, widths)

    # now apply demag_factor to all layers that have a magnetic component.
    for i in range(0, len(mslds)):
        if mslds[i] != 0:
            demag_arr[i] = demag_arr[i] * demag_factor

    # calculate magnetic composition of each layer over interface using VFPs.
    mag_comp = vfp * demag_arr
    # find regions of interface where VFPs are approximately invariant.
    difference_arr = (
        np.abs(np.diff(mag_comp, axis=1)) < MICROSLICE_EQUIVALENCE_THRESHOLD
    )
    reduce_diff_arr = np.all(difference_arr, axis=0)
    (indices_full,) = np.nonzero(reduce_diff_arr)
    # now remove parts of the vfps and mag_comp where they are ~ invariant.
    # remove the i+1 values, except the last in a block
    to_delete_indices = transform_indices(indices_full)
    reduced_vfp = np.delete(vfp, to_delete_indices, 1)
    reduced_magcomp = np.delete(mag_comp, to_delete_indices, 1)
    return reduced_vfp, reduced_magcomp, indices_full, demag_arr


def transform_indices(indices: tuple[int, ...] | np.ndarray) -> np.ndarray:
    """
    Use with vfp.indices to transform to indices suitable for
    reducing values from zed, vfp and sld.

    Parameters
    ----------
    indices : tuple[int, ...] | np.ndarray
        indices where the next point is roughly invariant.

    Returns
    -------
    np.ndarray
    """
    indices = np.asarray(indices)
    to_delete_indices = indices + 1
    seperate_indices = consecutive(to_delete_indices)
    final_to_delete_indices = np.concatenate(
        [arr[:-1] for arr in seperate_indices]
    )
    return final_to_delete_indices


def get_demag(
    dist: np.ndarray, locs: np.ndarray, widths: np.ndarray
) -> np.ndarray:
    """
    Calculates the demagnetisation function across the interface.

    The function is described by :math:`n` peaks, where :math:`n` is half the
    number of `locs` and `widths` parameters. The :math:`n` peaks are
    calculated from normal CDFs using `locs` and `widths` to define their
    location and scale respectively.

    The demagnetisation is set at zero by default, which represents a state of
    no demagnetisation (layers with a magnetic SLD will have the full extent
    of that magnetic SLD applied).

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
        up = scipy.stats.norm.cdf(
            dist, loc=cumlocs[2 * n], scale=widths[2 * n]
        )
        down = scipy.stats.norm.cdf(
            dist, loc=cumlocs[2 * n + 1], scale=widths[2 * n + 1]
        )
        peak = prev_down * up * (1 - down)
        demag_f += peak
        prev_down = down

    return demag_f


@lru_cache(maxsize=2)
def integrate_vfp(
    zeds: tuple[float],
    indexs: tuple[int],
    red_vfps: tuple[float],
    layer_indices: tuple[int],
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
        indices of nodes in the VFP that are approximately equal to a
        neighbouring node as defined in `self.init_demag`. These indices are
        used to calculate the thickness of each microslice across an uneven z
        space after reduction.
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
    >>> pdfs = scipy.stats.norm.pdf(
    ...     x=z[:, np.newaxis],
    ...     loc=[35, 60],
    ...     scale=[3, 4]
    ... ).T
    >>> pdfs[0] = pdfs[0] * 3 # make the first peak have an integral of 3.
    >>> integrate_vfp(zeds=tuple(z),
    ...               indexs=(),
    ...               red_vfps=tuple(tuple(i) for i in pdfs),
    ...               layer_indices=(0, 1))
    [np.float64(3.0), np.float64(1.0000000000000002)]
    """
    if not layer_indices:
        raise ValueError("layer_indices must be defined.")

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
        layer_integral = scipy.integrate.simpson(
            red_vfp[lidx], x=integrate_over
        )
        integrals.append(layer_integral)

    return integrals


def heaviside_step(z: np.ndarray, loc: float = 0) -> np.ndarray:
    """
    Get heaviside step function over support `z`, where centre is `loc`.

    output y = 1 if z >= loc else 0.

    Parameters
    ----------
    z : np.ndarray
        points at which to evaluate function.
    loc : float, optional
        Location of transition.

    Returns
    -------
    np.ndarray
    """
    centred_z = z - loc
    f = np.empty_like(centred_z)
    f[centred_z < 0] = 0
    f[centred_z >= 0] = 1
    return f
