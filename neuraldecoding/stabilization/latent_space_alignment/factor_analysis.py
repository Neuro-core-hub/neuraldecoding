"""Direct adaptation of factor analysis and stabilization method from Degenhart, et. al. 2020
Original paper: https://www.nature.com/articles/s41551-020-0542-9
Original repository: https://github.com/alandegenhart/stabilizedbci 

Author: Brianna M. Karpowicz (bkarpowicz3)
Last updated: 11/6/2021
"""


import warnings
from scipy import linalg
import numpy as np

def get_factor_analysis_loading(train_data, n_components, n_restarts=5):
    '''
    Runs multiple Factor Analysis (FA) models and returns loading matrix of 
    model with maximum log likelihood

    Parameters
    ----------
    train_data: numpy.array
        array of data to fit FA on, should be of shape trials x time x neurons 
    n_components: int 
        number of FA components to fit 
    n_restarts: int
        number of FA models to fit and select from 

    Returns
    -------
    loading: numpy.array 
        the loading matrix of the FA model with maximum log likelihood
    psi: numpy.array 
        diagonal matrix of errors for the FA model with max log likelihood
    d: numpy.array 
        vector of means for the FA model with max log likelihood
    '''
    input_data = np.vstack(train_data).astype(float)

    models = []
    for i in range(0, n_restarts): 
        d, c, psi, _, diags = fit_factor_analysis(input_data, 
                                                  n_latents=n_components,
                                                  min_priv_var=0.1,
                                                  ll_diff_thresh=0.00001)

        models.append((c, psi, d, diags))

    # get log likelihood from each model
    model_ll = [m[-1]['ll'] for m in models]
    # get final log likelihood from each model 
    model_final_ll = [ll[~np.isnan(ll)][-1] for ll in model_ll]
    # get model with max log likelihood and use this moving forward
    model_max_ll = models[np.argmax(model_final_ll)]
    loading = model_max_ll[0]
    psi = model_max_ll[1]
    d = model_max_ll[2]

    return loading, psi, d

def fit_factor_analysis(x, n_latents, max_n_its=100000, ll_diff_thresh=1e-8,
    min_priv_var=0.01, C_init=None, PSI_init=None):
    '''
    Fits a factor analysis model: 
    x_t = C * l_t + d + ep_t

    Assuming: 
    l_t ~ N(0, I)
    ep_t ~ N(O, psi)

    Parameters
    ----------
    x: numpy.array 
        array of data to fit FA on, should be 2D of shape (trials * time) x neurons
    n_latents: int
        the number of latent variables to fit the model for 
    max_n_its (optional): int 
        the maximum number of iterations to run the EM algorithm for 
    ll_diff_thresh (optional): float 
        the stopping criteria for fitting EM 
    min_priv_var (optional): float 
        a threshold for minimum private variance values (diagonals of psi); enforced 
        after the M step of each iteration
    C_init (optional): numpy.array
        if provided, will be the initial value of C to fit EM with, if None then 
        a random C will be used 
    PSI_init (optional): numpy.array 
        if provided, will be the initial values of psi to fit EM with, if None then 
        a random psi will be used

    Returns
    ----------
    d: numpy.array
        vector of estimated means
    c: numpy.array
        estimated loading matrix 
    psi: numpy.array
        diagonal matrix of estimated private noise variances 
    conv: bool
        True if the final EM iteration converged 
    diags: dict
        dictionary containing initial C and Psi and log likelihoods for each iteration
    '''

    # set up a function for stopping criteria
    stopfcn = lambda cL, pL, iL : (cL- pL)
    # number of observed variables 
    n_obs_vars = x.shape[1]
    # calculate mean 
    d = np.nanmean(x, axis=0)

    diags = {}
    if n_latents == 0: # easy case 
        c = np.zeros((n_obs_vars, 1))

        psi = np.nanvar(x)
        psi[psi < min_priv_var] = min_priv_var;
        psi = np.diag(psi)

        conv = True
    else: # need to run EM 
        # subtract mean from x, row-wise
        x_mean_ctr = x - d
        # arrange centered data into blocks according to observed variables 
        # done for computational efficiency 
        x_ctr_blocked = smpmat2blockstr(x_mean_ctr)

        # initialize estimates for c and psi 
        if C_init == None or PSI_init == None: 
            print('Initializing randomly.')
            # initialize C 
            c_init = np.random.randn(n_obs_vars, n_latents)*10
            # initialize psi 
            psi_init = np.diag(np.nanvar(x, axis=0)) - np.diag(np.diag(np.matmul(c_init, c_init.T)))
        
        if C_init != None: 
            c_init = C_init
        if PSI_init != None: 
            psi_init = PSI_init

        diags['cInit'] = c_init 
        diags['psiInit'] = psi_init 

        print('Done with initialization. Fitting with EM.')

        diags['ll'] = np.full((max_n_its+1,), np.nan)

        c = c_init
        psi = psi_init 

        # enforce private noise floor on psi 
        psi_diag = np.array(np.diag(psi))
        psi_diag[psi_diag < min_priv_var] = min_priv_var 
        psi = np.diag(psi_diag)

        # calculate initial log likelihood
        init_ll, ll_precomp = fa_data_log_likelihood(x_ctr_blocked, c, psi)
        diags['ll'][0] = init_ll
        prev_ll = init_ll

        cur_it = 1
        stop_crit = np.inf 
        while cur_it <= max_n_its and stop_crit > ll_diff_thresh: 
            # run the E step 
            block_e_str = fast_fa_e_step(x_ctr_blocked, c, psi)

            # run the M step 
            if cur_it == 1: 
                c, psi, precomp = fast_fa_m_step(block_e_str, x_ctr_blocked)
            else: 
                c, psi, _ = fast_fa_m_step(block_e_str, x_ctr_blocked, precomp)

            # enforce private noise floor 
            psi_diag = np.array(psi) 
            psi_diag[psi_diag < min_priv_var] = min_priv_var 
            psi = np.diag(psi_diag)
            
            cur_ll, _ = fa_data_log_likelihood(x_ctr_blocked, c, psi, ll_precomp)
            diags['ll'][cur_it] = cur_ll

            if cur_ll == -1*np.inf: 
                stop_crit = np.inf
            else: 
                stop_crit = stopfcn(cur_ll, prev_ll, init_ll)
            prev_ll = cur_ll

            print('EM Iteration: %d LL: %f Stop Crit: %f' % (cur_it, cur_ll, stop_crit), end="\r", flush=True)
            
            cur_it += 1

        conv = stop_crit <= ll_diff_thresh
        print('\n')
        
    return d, c, psi, conv, diags

def fa_data_log_likelihood(block_x_data, c, psi, precomp=None):
    '''
    Calculates the log likelihood of observed data under an FA model

    Parameters
    ----------
    block_x_data: list of dicts 
        list of dictionaries of data in block form, from call to smpmat2blockstr
    c: numpy.array
        loading matrix 
    psi: numpy.array 
        array of private variances 
    precomp (optional): dict
        pre-computed saved values from a previous call to this function

    Returns
    -------
    ll: float 
        the log-likelihood
    precomp: dict
        saved values to be reused in future functional calls
    '''
    if precomp == None: 
        precomp = {}
        precomp['n_blocks'] = len(block_x_data)
        precomp['pi_consts'] = np.zeros((1, precomp['n_blocks']))
        precomp['covs'] = {}

        for b in range(0, precomp['n_blocks']): 
            n_block_vars = len(block_x_data[b]['block_x_inds'])
            precomp['pi_consts'][b] = -0.5 * block_x_data[b]['nsmps'] * n_block_vars * np.log(2*np.pi)

            # normalize by N and not N-1
            nsmps = block_x_data[b]['nsmps']
            if nsmps > 1: 
                precomp['covs'][b] = ((nsmps-1)/nsmps) * np.cov(block_x_data[b]['obsx'], rowvar=False)
            else: 
                precomp['covs'][b] = np.zeros((n_block_vars, n_block_vars))

    n_blocks = precomp['n_blocks']
    pi_consts = precomp['pi_consts']
    covs = precomp['covs']
    ll = sum(pi_consts)[0]

    for b in range(0, n_blocks): 
        n_block_smps = block_x_data[b]['nsmps']
        block_var_inds = block_x_data[b]['block_x_inds']
        cur_C = c[block_var_inds, :]
        cur_cov = np.matmul(cur_C, cur_C.T) + psi[block_var_inds,:][:,block_var_inds]
        ll = ll - 0.5*n_block_smps*(np.linalg.slogdet(cur_cov)[1] + np.trace(np.linalg.lstsq(cur_cov, covs[b])[0]))

    return ll, precomp

def fast_fa_e_step(blockstr, c, psi):
    '''
    Computes the E step for fitting FA models 

    Parameters
    ----------    
    blockstr: list of dicts 
        list of dictionaries of data in block form, from call to smpmat2blockstr
    c: numpy.array
        loading matrix
    psi: numpy.array
        array of private variances 

    Returns
    -------
    block_e_str: list of dicts 
        a list with a dictionary containing posterior means and posterior covariance
        for each entry in the input blockstr. means are contained in the key 'post_means'
        and covariances in 'post_cov'
    '''
    n_latents = c.shape[1]
    n_blocks = len(blockstr) 

    block_e_str = []
    # work backwards to implicitly preallocate structure 
    for b in range(n_blocks-1, -1, -1): 
        cur_inds = blockstr[b]['block_x_inds']

        block_c = c[cur_inds, :]
        block_psi = psi[cur_inds, :][:, cur_inds]

        denom = np.linalg.inv(np.matmul(block_c, block_c.T) + block_psi)
        core_computation = np.matmul(block_c.T, denom)

        block_e_str.append({'post_means': np.matmul(blockstr[b]['obsx'], core_computation.T),
                            'post_cov': np.eye(n_latents) - np.matmul(core_computation, block_c)
                           })
    
    return block_e_str

def fast_fa_m_step(block_e_str, block_x_data, precomp=None):
    '''
    Computes the M step for fitting FA models 

    Parameters
    ----------  
    block_e_str: list of dicts  
        list of dictionaries outputted by the E step, from a call to fast_fa_e_step
    block_x_data: list of dicts 
        list of dictionaries of data in block form, from call to smpmat2blockstr
    precomp (optional): dict
        saved pre-computed values from an earlier call to this function

    Returns
    -------
    C: numpy.array
        the estimated loading matrix 
    psi: numpy.array
        the estimated private variances 
    precomp: dict 
        pre-computed values saved to be used in later calls to this function
    '''
    # do precomputations if needed 
    if precomp == None: 
        precomp = {}
        # compute C row-wise, in blocks 
        # determine what row-wise blocks are 
        x_block_inds = np.array([bxd['block_x_inds'] for bxd in block_x_data])
        row_wise_blockstr = find_minimal_disjoint_integer_sets(x_block_inds)

        n_var_blocks = len(row_wise_blockstr)
        for b in range(0, n_var_blocks): 
            block_cell_inds = row_wise_blockstr[b]['cell_inds']
            n_cell_inds = len(block_cell_inds)

            row_wise_blockstr[b]['x_block_cols'] = []
            for c in range(0, n_cell_inds): 
                cur_cell_inds = block_cell_inds[c]
                row_wise_blockstr[b]['x_block_cols'].append(np.intersect1d(
                    block_x_data[cur_cell_inds]['block_x_inds'], 
                    row_wise_blockstr[b]['vls'])
                )
        
        # number of observed variables 
        n_vars = np.max([rwbs['vls'] for rwbs in row_wise_blockstr])+1
        # number of latents 
        n_latents = len(block_e_str[0]['post_cov'])
        # precompute square of observations 
        obs_sq = np.nansum(blockstr2smpmat(block_x_data)**2, axis=0)
        # precompute number of observations for each variable 
        n_obs = np.nansum(~np.isnan(blockstr2smpmat(block_x_data)), axis=0)
    else: 
        row_wise_blockstr = precomp['row_wise_blockstr']
        n_vars = precomp['n_vars']
        n_latents = precomp['n_latents']
        obs_sq = precomp['obs_sq']
        n_obs = precomp['n_obs']

    precomp['row_wise_blockstr'] = row_wise_blockstr
    precomp['n_vars'] = n_vars 
    precomp['n_latents'] = n_latents 
    precomp['obs_sq'] = obs_sq
    precomp['n_obs'] = n_obs
    
    C = np.full((n_vars, n_latents), np.nan)
    psi = np.full((n_vars, 1), np.nan)

    # compute C and take care of some computations for psi in blocks, row-wise
    n_var_blocks = len(row_wise_blockstr)
    for b in range(0, n_var_blocks): 
        block_var_inds = row_wise_blockstr[b]['vls']
        n_block_vars = len(block_var_inds)

        x_block_inds = row_wise_blockstr[b]['cell_inds']
        n_x_blocks = len(x_block_inds)

        term1 = np.zeros((n_block_vars, n_latents))
        term2 = np.zeros((n_latents, n_latents))

        for xB in range(0, n_x_blocks): 
            cur_x_block_ind = x_block_inds[xB]
            cur_x_block = block_x_data[cur_x_block_ind]
            cur_e_step_block = block_e_str[cur_x_block_ind]

            cur_x_block_var_inds = row_wise_blockstr[b]['x_block_cols'][xB]

            term1 += np.matmul(cur_x_block['obsx'][:, cur_x_block_var_inds].T, cur_e_step_block['post_means'])
            term2 += cur_x_block['nsmps'] * cur_e_step_block['post_cov'] + \
                np.matmul(cur_e_step_block['post_means'].T, cur_e_step_block['post_means'])
        
        C[block_var_inds, :] = np.matmul(term1, np.linalg.inv(term2))
        psi[block_var_inds] = np.expand_dims(obs_sq[block_var_inds] - 1 * np.sum(term1*C[block_var_inds,:], axis=1),axis=1)

    psi=np.diag(psi/n_obs)

    return C, psi, precomp

def smpmat2blockstr(x):
    '''
    Converts a matrix of samples with potentially missing data indicated by NaNs 
    to a structure organized by block, where each block has the same non-missing
    variables.

    Parameters
    ----------  
    x: numpy.array
        matrix of samples of shape (trials * time) x neurons with missing values 
        indicated by NaN

    Returns
    -------
    blockstr: list of dicts
        a list of dicts of length B, where B is the number of blocks. Each dict 
        contains: 
            nsmps: number of samples in this block 
            block_x_inds: columns in the x matrix for the variables represented 
                in this block
            orig_rows: vector listing the rows the samples of x for this block 
                were taken from
            obsx: a matrix of size samples x dimensions that are non-NaN for this
                block
    '''
    # get non-NaN values 
    x_not_nan_inds = ~np.isnan(x)
    # get unique blocks 
    unique_rows = np.unique(x_not_nan_inds, axis=0)
    # find the index of the unique rows where each row matches
    row_inds = np.array([np.unique(np.where(unique_rows == x_not_nan_inds[ii,:])[0])[0] \
        for ii in range(0, x_not_nan_inds.shape[0]) \
            if x_not_nan_inds[ii,:] in unique_rows])
    n_blocks = unique_rows.shape[0]

    block_str = []
    for b in range(0, n_blocks):
        orig_rows = np.where(row_inds == b)[0]
        block_x_inds = np.where(unique_rows[b, :])[0]
        dictb = {}
        dictb['nsmps'] = len(orig_rows)
        dictb['block_x_inds'] = block_x_inds
        dictb['orig_rows'] = orig_rows
        dictb['obsx'] = x[orig_rows,:][:,block_x_inds]
        block_str.append(dictb)

    return block_str

def find_minimal_disjoint_integer_sets(input_cell): 
    '''
    Given sets of possibly overlapping integers, this function will return another
    collection of sets so that intersection of any two of the returned sets
    is the null set and the union of the returned sets equals the union of
    the original sets. This collection will be the smallest collection 
    possible with this property. 

    Parameters
    ---------- 
    input_cell: numpy.array
        array where each entry contains a vector of integers

    Returns
    -------
    set_str: list of dicts 
        a list of length S, where S is the number of returned sets. Each dict has 
        keys: 
            vls: values in the set 
            cell_inds: indices into input cell of the original sets that contained 
                these values
    '''
    cell_length = len(input_cell)

    min_vl = np.min(input_cell)
    max_vl = np.max(input_cell)

    unique_vls = np.array(range(min_vl, max_vl+1))
    n_unique_vls = max_vl - min_vl + 1

    ind_matrix = np.zeros((n_unique_vls, cell_length), dtype=bool)
    for i in range(0, cell_length): 
        ind_matrix[input_cell[i]-min_vl, i] = True 
    
    bad_rows = np.sum(ind_matrix, axis=1) == 0
    unique_vls[bad_rows] = []
    ind_matrix[bad_rows, :] = np.empty((1, ind_matrix.shape[1]))
 
    unique_rows = np.unique(ind_matrix)
    # find the index of the unique rows where each row matches
    row_map = np.array([np.unique(np.where(unique_rows == ind_matrix[ii])[0])[0] \
        for ii in range(0, ind_matrix.shape[0]) if ind_matrix[ii,:] in unique_rows])
    n_unique_rows = unique_rows.shape[0]

    setstr = []
    for r in range(n_unique_rows-1, -1, -1): 
        setstr.append({'vls': unique_vls[row_map == r],
                       'cell_inds': np.where(unique_rows[r])[0]
                       })
    
    return setstr

def blockstr2smpmat(blockstr): 
    '''
    Function to convert data stored in a block structure to matrix with missing
    values indicated by NaN.

    Parameters
    ---------- 
    blockstr: list of dicts
        data stored in a block structure as defined by smpmat2blockstr

    Returns
    -------
    x: numpy.array
        data in matrix form of shape (trials * time) x neurons with missing values 
        indicated by NaN.
    '''
    n_blocks = len(blockstr)

    n_block_smps = np.array([b['nsmps'] for b in blockstr])
    n_total_smps = sum(n_block_smps)

    n_cols = np.max([b['block_x_inds'] for b in blockstr])+1

    x = np.full((n_total_smps, n_cols), np.nan)
    for b in range(0, n_blocks): 
        x_inds = blockstr[b]['orig_rows'][:,None]
        y_inds = blockstr[b]['block_x_inds'][None,:]
        x[x_inds, y_inds] = blockstr[b]['obsx']

    return x

def get_stabilization_matrices(loading, psi, d): 
    '''
    Function to get matrices so latent state (stabilized representation of
    neural signals) can be calculated as: l_t = beta*y_t + o,
    where l_t is latent state, y_t is vector of neural data and beta and o
    are matrices calculated by this function. 

    Parameters
    ---------- 
    loading: numpy.array
        loading matrix of FA model ('C')
    psi: numpy.array
        psi matrix of FA model
    d: numpy.array
        vector of means stored in FA model

    Returns
    -------
    beta: numpy.array
        matrix in above latent state equation
    o: numpy.array
        matrix in above latent state equation
    '''

    # x=b/A solves xA = B which is the equivalent of A'x' = B'
    # use np.linalg.lstsq(A.T, b.T).T
    b = loading.T
    A = (loading@loading.T + psi)
    beta = np.linalg.lstsq(A.T, b.T)[0].T
    o = -1*beta@d

    return beta, o

def update_factor_analysis_loading(day0_loading, calibration_data, n_components, 
    n_restarts=5, n_stable_rows=60, threshold=0.01): 
    '''
    Returns the FA model of the calibration data aligned to the day0_loading 

    Parameters
    ---------- 
    day0_loading: numpy.array
        loading matrix from day 0 FA model, to be aligned to
    calibration_data: numpy.array
        array of data to fit new FA on, should be of shape trials x time x neurons 
    n_components: int 
        number of FA components to fit on calibration data
    n_restarts: int
        number of FA models to fit and select from 
    n_stable_rows: int
        the number of rows of the loading matrix to use for alignment
    threshold: float
        the threshold to use when screening out rows for possible
        alignment.  Any row which has an l_2 norm less than th in either m_1 or
        m_2 will not be considered when selecting rows to use for alignment.

    Returns
    ---------- 
    dayk_loading: numpy.array
        aligned loading matrix for day K data 
    aligned_channels: numpy.array 
        array indicating which channels were selected for alignment 
    '''
    # Fit a FA model to the calibration data 
    dayk_loading, psi, d = get_factor_analysis_loading(calibration_data, 
                                               n_components=n_components, 
                                               n_restarts=n_restarts)

    # Rotate the FA model 
    W, aligned_channels = align_loading_matrices(day0_loading, dayk_loading, 
                                                 n_stable_rows=n_stable_rows, 
                                                 threshold=threshold)
    dayk_loading = np.matmul(dayk_loading, W.T)

    # get the stabilization matrices 
    beta, o = get_stabilization_matrices(dayk_loading, psi, d)

    return beta, o, aligned_channels

def align_loading_matrices(m1, m2, n_stable_rows=60, threshold=0.01): 
    '''
    Returns the alignment matrix W that aligns m2 to m1 and the rows used
    for that alignment.

    Parameters
    ---------- 
    m1: numpy.array
        the first loading matrix, to align m2 to
    m2: numpy.array 
        the second loading matrix, to align to m1
    n_stable_rows: int
        the number of rows of the loading matrix to use for alignment
    threshold: float
        the threshold to use when screening out rows for possible
        alignment.  Any row which has an l_2 norm less than th in either m_1 or
        m_2 will not be considered when selecting rows to use for alignment.

    Returns
    ---------- 
    align_rows: numpy.array 
        array indicating which channels were selected for alignment 
    W: numpy.array 
        the matrix that will align m2 to m1 so that m1 - m2*W' is minimized
    '''
    align_rows = identify_stable_loading_rows(m1, m2, n_stable_rows, threshold)
    W = learn_optimal_orthonormal_transformation(m1[align_rows,:], m2[align_rows,:])

    return W, align_rows

def identify_stable_loading_rows(m1, m2, n_stable_rows=60, threshold=0.01): 
    '''
    Returns the stable rows used for alignment.

    Parameters
    ---------- 
    m1: numpy.array
        the first loading matrix, to align m2 to
    m2: numpy.array 
        the second loading matrix, to align to m1
    n_stable_rows: int
        the number of rows of the loading matrix to use for alignment
    threshold: float
        the threshold to use when screening out rows for possible
        alignment.  Any row which has an l_2 norm less than th in either m_1 or
        m_2 will not be considered when selecting rows to use for alignment.

    Returns
    ---------- 
    align_rows: numpy.array 
        array indicating which channels were selected for alignment 
    '''
    n_rows = m1.shape[0]
    n_latents = m1.shape[1]

    m1_norms = np.sqrt(np.sum(m1**2, axis=1))
    small_m1_rows = np.where(m1_norms < threshold)[0]

    m2_norms = np.sqrt(np.sum(m2**2, axis=1))
    small_m2_rows = np.where(m2_norms < threshold)[0]

    small_rows = np.union1d(small_m1_rows, small_m2_rows)
    clean_rows = np.setdiff1d(range(0, n_rows), small_rows) 
    n_clean_rows = len(clean_rows)

    m1_clean = m1[clean_rows, :]
    m2_clean = m2[clean_rows, :]

    cur_rows = np.array(range(0, n_clean_rows))
    n_drop_rows = max(n_clean_rows - n_stable_rows, 0)
    for i in range(1, n_drop_rows+1): 
        # Perform alignment using the rows that were identified last iteration
        m1_cur = m1_clean[cur_rows, :]
        m2_cur = m2_clean[cur_rows, :]

        W = learn_optimal_orthonormal_transformation(m1_cur, m2_cur)
        row_delta = np.sqrt(np.sum((m1_cur - np.matmul(m2_cur,W))**2, axis=1))

        # Identify rows to keep this iteration 
        n_keep_rows = n_clean_rows - i
        sort_order = np.argsort(row_delta, axis=0)
        best_sorted_rows = sort_order[0:n_keep_rows]
        cur_rows = np.array(sorted(cur_rows[best_sorted_rows]))

    align_rows_enum = clean_rows[cur_rows].T
    n_align_rows = len(align_rows_enum)

    align_rows = np.zeros((n_rows,), dtype=bool)
    align_rows[align_rows_enum] = True

    if n_align_rows < n_stable_rows: 
        warnings.warn('Too many small value rows: Unable to return the requested number of alignment rows.')

    if n_align_rows < n_latents: 
        warnings.warn('Number of alignment rows is less than the number of latent variables in the model.')

    return align_rows

def learn_optimal_orthonormal_transformation(m1, m2):
    ''' 
    Finds the alignment matrix that aligns m2 to m1 according to: 
    S = m1'*m2; 
    [U, ~, V] = svd(S); 
    T = U*V'; 

    Parameters
    ---------- 
    m1: numpy.array
        the first loading matrix, to align m2 to
    m2: numpy.array 
        the second loading matrix, to align to m1

    Returns
    -------
    T: numpy.array 
        the learned transformation matrix 
    '''
    S = np.matmul(m1.T, m2)
    U, _, V = np.linalg.svd(S)
    V = V.T # due to differences in Matlab and numpy impls 
    T = np.matmul(U, V.T)
    return T