
def thetahod_literature(paper):
    ''' best-fit HOD parameters from the literature.

    Currently, HOD values from the following papers are available:
    * 'parejko2013_lowz'
    * 'manera2015_lowz_ngc'
    * 'manera2015_lowz_sgc'
    * 'redi2014_cmass'
    '''
    if paper == 'parejko2013_lowz':
        # lowz catalog from Parejko+2013 Table 3. Note that the
        # parameterization is slightly different so the numbers need to
        # be converted.
        p_hod = {
            'logMmin': 13.25,
            'sigma_logM': 0.43,  # 0.7 * sqrt(2) * log10(e)
            'logM0': 13.27,  # log10(kappa * Mmin)
            'logM1': 14.18,
            'alpha': 0.94
        }
    elif paper == 'manera2015_lowz_ngc':
        # best-fit HOD of the lowz catalog NGC from Table 2 of Manera et al.(2015)
        p_hod = {
            'logMmin': 13.20,
            'sigma_logM': 0.62,
            'logM0': 13.24,
            'logM1': 14.32,
            'alpha': 0.9
        }
    elif paper == 'manera2015_lowz_sgc':
        # best-fit HOD of the lowz catalog SGC from Table 2 of Manera et al.(2015)
        # Manera+(2015) actually uses a redshift dependent HOD. The HOD that's
        # currently implemented is primarily for the 0.2 < z < 0.35 population,
        # which has nbar~3x10^-4 h^3/Mpc^3
        p_hod = {
            'logMmin': 13.14,
            'sigma_logM': 0.55,
            'logM0': 13.43,
            'logM1': 14.58,
            'alpha': 0.93
        }
    elif paper == 'reid2014_cmass':
        # best-fit HOD from Reid et al. (2014) Table 4
        p_hod = {
            'logMmin': 13.03,
            'sigma_logM': 0.38,
            'logM0': 13.27,
            'logM1': 14.08,
            'alpha': 0.76
        }
    else:
        raise NotImplementedError

    return p_hod
