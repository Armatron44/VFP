import numpy as np
from refnx.reflect.interface import Step
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import colormaps

def surfaces_for_display(VFP, points=50):
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

    thicks = VFP.thicks

    if VFP.orientation in ('front'):
        interf_loc = np.cumsum(thicks)
        roughs = VFP.roughnesses

    elif VFP.orientation in ('back'):
        interf_loc = np.cumsum(thicks)
        interf_loc = np.fabs(interf_loc - interf_loc[-1])[::-1]
        roughs = VFP.roughnesses[::-1] 

    interf_arr = np.ones((len(interf_loc), points))
    num_conform = np.sum(VFP.conformal)

    # return the non-conformal interfaces.
    for i in range(0, len(interf_arr)):
        interf_arr[i] = stats.norm.rvs(loc=interf_loc[i], scale=float(roughs[i]), size=points)

    if VFP.orientation in ('front'):
        idx_where_conformal = np.where(np.array(VFP.conformal) == 1)[0]
        # insert the conformal interfaces.
        if num_conform != 0:
            for i in idx_where_conformal:
                interf_arr[i] = np.max(interf_arr[:i].T, axis=1) + thicks[i]
    
    elif VFP.orientation in ('back'):
        idx_where_conformal = np.where(np.array(VFP.conformal) == 1)[0] - (len(VFP.conformal) - 1)
        if num_conform != 0:
            for i in idx_where_conformal:
                interf_arr[i] = np.min(interf_arr[i+1:].T, axis=1) - thicks[::-1][i]

    return interf_arr

def model_plot(VFP, points=50, microslice_SLD=True, total_SLD=False, total_VF=True):
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

    if points <= 0:
        raise ValueError("points must be > 0.")

    points = points + 2 # add on two additional points to create fill effect on y axis of bottom plot
    surfaces = surfaces_for_display(VFP, points=points)
    
    # define some colours to use for the surface plot.
    colours = colormaps['tab20'].colors

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 9))

    # ax[0] - nSLD / mSLD / iSLD. Only plots mSLD and iSLD curves if they are not zero.
    if microslice_SLD == True:
        z, SLDn, SLDm, SLDi = _gen_sld_profile(VFP)
        
        if total_SLD == True:
            if VFP.spin_state in ('none'):
                tot_sld = SLDn
                ax[0].plot(z + VFP.SLD_offset(), tot_sld, color='k', label=r'$\mathrm{SLD}_{\mathrm{n}}$')

            elif VFP.spin_state in ('down'):
                tot_sld = SLDn - SLDm
                ax[0].plot(z + VFP.SLD_offset(), tot_sld, color='k', label=r'$\mathrm{SLD}_{\mathrm{n}} - \mathrm{SLD}_{\mathrm{m}}$')

            elif VFP.spin_state in ('up'):
                tot_sld = SLDn + SLDm
                ax[0].plot(z + VFP.SLD_offset(), tot_sld, color='k', label=r'$\mathrm{SLD}_{\mathrm{n}} + \mathrm{SLD}_{\mathrm{m}}$')

            if SLDi.any(): # if SLDi contains any number that is not zero
                ax0_twinx = ax[0].twinx()
                ax0_twinx.plot(z + VFP.SLD_offset(), SLDi, color='tab:red')

        else:
            ax[0].plot(z + VFP.SLD_offset(), SLDn, color='k', label=r'$\mathrm{SLD}_{\mathrm{n}}$')
            if SLDm.any():
                ax[0].plot(z + VFP.SLD_offset(), SLDm, color='tab:grey', label=r'$\mathrm{SLD}_{\mathrm{m}}$')

            if SLDi.any():
                ax0_twinx = ax[0].twinx()
                ax0_twinx.plot(z + VFP.SLD_offset(), SLDi, color='tab:red')
    
    else: # if you want the original non-microsliced SLD profile.
        z, tot_sld, SLDn, SLDm = VFP.z_and_SLD_scatter()
        SLDi = VFP.z_and_SLD_scatter(imag=True)[1]
        
        if total_SLD == True:
            if VFP.spin_state in ('none'):
                ax[0].plot(z, tot_sld, color='k', label=r'$\mathrm{SLD}_{\mathrm{n}}$')

            elif VFP.spin_state in ('down'):
                ax[0].plot(z, tot_sld, color='k', label=r'$\mathrm{SLD}_{\mathrm{n}} - \mathrm{SLD}_{\mathrm{m}}$')     

            elif VFP.spin_state in ('up'):
                ax[0].plot(z, tot_sld, color='k', label=r'$\mathrm{SLD}_{\mathrm{n}} + \mathrm{SLD}_{\mathrm{m}}$')

            if SLDi.any():
                ax0_twinx = ax[0].twinx()
                ax0_twinx.plot(z, SLDi, color='tab:red')

        else:
            ax[0].plot(z, SLDn, color='k', label=r'$\mathrm{SLD}_{\mathrm{n}}$')
            
            if SLDm.any():
                ax[0].plot(z, SLDm, color='tab:grey', label=r'$\mathrm{SLD}_{\mathrm{m}}$')

            if SLDi.any():
                ax0_twinx = ax[0].twinx()
                ax0_twinx.plot(z, SLDi, color='tab:red')
    
    # ax[1] - Volume fractions
    vfs = VFP.vfs_for_display()[0]
    z = VFP.z_and_SLD_scatter()[0]

    if VFP.orientation in ('front'):
        for i, j in enumerate(vfs):
            if i == 0:
                ax[1].plot(z, j.T, label=f'Fronting')

            elif i+1 == len(vfs):
                ax[1].plot(z, j.T, label=f'Backing')

            else:
                ax[1].plot(z, j.T, label=f'Layer {i}')
    
    elif VFP.orientation in ('back'):
        for i, j in enumerate(vfs):
            if i == 0:
                ax[1].plot(z, j.T, label=f'Backing', color=colours[(2*len(VFP.thicks) - 2*i) % len(colours)], zorder=len(VFP.thicks)-i)

            elif i+1 == len(vfs):
                ax[1].plot(z, j.T, label=f'Fronting', color=colours[(2*len(VFP.thicks) - 2*i) % len(colours)], zorder=len(VFP.thicks)-i)

            else:
                ax[1].plot(z, j.T, label=f'Layer {len(vfs)-(i+1)}', color=colours[(2*len(VFP.thicks) - 2*i) % len(colours)], zorder=len(VFP.thicks)-i)

    if total_VF == True:
        ax[1].plot(z, np.sum(vfs.T, axis=1), label=r'Total', linestyle='--', color='k')

    # ax[2] - surface plots
    def_xlower_lim, def_xupper_lim = ax[1].get_xlim() # get the default x limits from ax1 before filling

    if VFP.orientation in ('front'):
        for i, j in enumerate(surfaces): # do the surfaces
            ax[2].plot(j, range(0, points), marker='.', zorder=(2*len(VFP.thicks)+1 - 2*i))
        
        for i in range(0, len(VFP.thicks)+1): # then do the fills
            if i == 0:
                ax[2].fill_betweenx(y=range(0, points), x1=def_xlower_lim-1, x2=surfaces[i], interpolate=True, 
                                    color=colours[(2 * i % len(colours)) + 1], zorder=(2*len(VFP.thicks) - 2*i))
            
            elif i < len(VFP.thicks):
                ax[2].fill_betweenx(y=range(0, points), x1=surfaces[i-1], x2=surfaces[i], where=surfaces[i]>surfaces[i-1], 
                                    interpolate=True, zorder=(2*len(VFP.thicks) - 2*i), color=colours[(2 * i % len(colours)) + 1])
            
            elif i == len(VFP.thicks):
                ax[2].fill_betweenx(y=range(0, points), x1=surfaces[i-1], x2=def_xupper_lim+1, where=def_xupper_lim+1>surfaces[i-1], 
                                    interpolate=True, zorder=(2*len(VFP.thicks) - 2*i), color=colours[(2 * i % len(colours)) + 1])
    
    elif VFP.orientation in ('back'):
        for i, j in enumerate(surfaces): # do the surfaces
            ax[2].plot(j, range(0, points), marker='.', color=colours[(len(VFP.thicks) - 2*i) % len(colours)], zorder=(len(VFP.thicks)+1 + 2*i))

        for i in range(0, len(VFP.thicks)+1): # then do the fills
            if i == 0:
                ax[2].fill_betweenx(y=range(0, points), x1=def_xlower_lim-1, x2=surfaces[i], 
                                    interpolate=True, color=colours[(2*len(VFP.thicks) + 1 - 2*i) % len(colours)], zorder=2*i)

            elif i < len(VFP.thicks):
                ax[2].fill_betweenx(y=range(0, points), x1=surfaces[i-1], x2=surfaces[i], where=surfaces[i]>surfaces[i-1], 
                                    interpolate=True, color=colours[(2*len(VFP.thicks) + 1 - 2*i) % len(colours)], zorder=2*i)
            
            elif i == len(VFP.thicks):
                ax[2].fill_betweenx(y=range(0, points), x1=surfaces[i-1], x2=def_xupper_lim+1, where=def_xupper_lim+1>surfaces[i-1], 
                                    interpolate=True, color=colours[(2*len(VFP.thicks) + 1 - 2*i) % len(colours)], zorder=2*i)
    
    # formatting
    ax[0].legend(frameon=False)
    ax[0].set_ylabel(r'SLD / $\mathrm{\AA{}}^{-2} \times 10^{-6}$')
    ax[1].set_ylabel(r'Volume Fraction')
    ax[1].legend(frameon=False)

    # format the right-hand side y axis.
    try:
        ax0_twinx.set_ylabel(r'$\mathrm{SLD}_{\mathrm{i}}$ / $\mathrm{\AA{}}^{-2} \times 10^{-6}$', color='tab:red')
        ax0_twinx.tick_params(axis='y', colors='tab:red')
        ax[0].set_zorder(ax0_twinx.get_zorder()+1) # puts nSLD and mSLDs above the iSLD.
        ax[0].patch.set_visible(False) # make sure the iSLD isn't completely obscured by the first axis.
    except:
        pass
    
    # define some y limits for ax[2] that allow for points > 0.
    ylower = 0.5
    yupper = (points - 2) + ((points - 1) - (points - 2)) / 2

    ax[2].set_xlabel(r'Distance over Interface / $\mathrm{\AA{}}$')
    ax[2].set_yticks([])
    # set the x limits to the original x limits before plotting the fills.
    ax[2].set_xlim(def_xlower_lim, def_xupper_lim) 
    ax[2].set_ylim(ylower, yupper) # chop off the extra two points
    
    for border in ['top', 'bottom', 'left', 'right']:
        ax[2].spines[border].set_zorder((len(VFP.thicks)+1)*3) # borders will be higher than surfaces and fills.

    return fig, ax

def _gen_sld_profile(VFP):

    # use the __call__ method of the VFP class to
    # return islds and thicknesses of each slab
    _, islds, thicks = VFP.process_model()

    # grab the original nSLD and mSLDs
    _, _, nSLD, mSLD = VFP.calc_slds()

    # get the average between each nuclear and magnetic SLD value.
    average_nslds = 0.5 * np.diff(nSLD) + nSLD[:-1]
    average_mslds = 0.5 * np.diff(mSLD) + mSLD[:-1]
    
    # init arrays for final SLDs.
    nslds = np.ones(average_nslds.shape[0] + 1)
    mslds = np.ones(average_mslds.shape[0] + 1)
    
    if VFP.orientation in ('front'):
        # fill all but last with average SLDs.
        nslds[:-1] = nslds[:-1] * average_nslds
        mslds[:-1] = mslds[:-1] * average_mslds
        # now set the final sld value to those from the micro arrays.
        nslds[-1] = nSLD[-1]
        mslds[-1] = mSLD[-1]

    elif VFP.orientation in ('back'):
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
    max_delta_z = float(VFP.max_delta_z) / 2
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

    sigma = microslices[:, 4]

    stepper = Step()
    # accumulate the SLD of each step.
    for i in range(nlayers-1):
        nsld += delta_nSLD[i] * stepper(zed, scale=sigma[i], loc=dist[i])
        msld += delta_mSLD[i] * stepper(zed, scale=sigma[i], loc=dist[i])
        isld += delta_iSLD[i] * stepper(zed, scale=sigma[i], loc=dist[i])
    
    return zed, nsld, msld, isld