from interface import refnxVFP, refl1dVFP

def VFP(nucSLDs, 
        thicknesses, 
        roughnesses,
        model_type='refnx', 
        nuciSLDs=None, 
        magSLDs=None,
        spin_state='none',
        orientation='front', 
        demaglocs=[], 
        demagwidths=[], 
        SLD_constraint=None, 
        max_delta_z=0.5):
    
    if model_type in ('refnx'):
        vfp = refnxVFP(nucSLDs,
                       thicknesses,
                       roughnesses,
                       nuciSLDs=nuciSLDs, 
                       magSLDs=magSLDs, 
                       spin_state=spin_state,
                       orientation=orientation, 
                       demaglocs=demaglocs, 
                       demagwidths=demagwidths, 
                       SLD_constraint=SLD_constraint, 
                       max_delta_z=max_delta_z)
    
    elif model_type in ('refl1d'):
        vfp = refl1dVFP(nucSLDs,
                        thicknesses,
                        roughnesses,
                        nuciSLDs=nuciSLDs, 
                        magSLDs=magSLDs, 
                        spin_state=spin_state,
                        orientation=orientation, 
                        demaglocs=demaglocs, 
                        demagwidths=demagwidths, 
                        SLD_constraint=SLD_constraint, 
                        max_delta_z=max_delta_z)    

    else:
        raise ValueError(
            'model_type must be "refnx" or "refl1d".'
            )   
    
    return vfp