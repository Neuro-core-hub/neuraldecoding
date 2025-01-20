import torch
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.base import BaseEstimator
import sys

EPSILON = sys.float_info.epsilon

# ------------------------
# --- Legacy Functions ---
# ------------------------

def add_training_noise(x,
                       bias_neural_std=None,
                       noise_neural_std=None,
                       noise_neural_walk_std=None,
                       bias_allchans_neural_std=None,
                       device='cpu'):
    """Function to add different types of noise to training input data to make models more robust.
       Identical to the methods in Willet 2021.

    Args:
        x (tensor):                     neural data of shape [batch_size x num_chans x conv_size]
        bias_neural_std (float):        std of bias noise
        noise_neural_std (float):       std of white noise
        noise_neural_walk_std (float):  std of random walk noise
        bias_allchans_neural_std (float): std of bias noise, bias is same across all channels
        device (device):                torch device (cpu or cuda)
    """
    if bias_neural_std:
        # bias is constant across time (i.e. the 3 conv inputs), but different for each channel & batch
        # biases = torch.normal(0, bias_neural_std, x.shape[:2]).unsqueeze(2).repeat(1, 1, x.shape[2])
        biases = torch.normal(torch.zeros(x.shape[:2]), bias_neural_std).unsqueeze(2).repeat(1, 1, x.shape[2])
        x = x + biases.to(device=device)

    if noise_neural_std:
        # adds white noise to each channel and timepoint (independent)
        # noise = torch.normal(0, noise_neural_std, x.shape)
        noise = torch.normal(torch.zeros_like(x), noise_neural_std)
        x = x + noise.to(device=device)

    if noise_neural_walk_std:
        # adds a random walk to each channel (noise is summed across time)
        # noise = torch.normal(0, noise_neural_walk_std, x.shape).cumsum(dim=2)
        noise = torch.normal(torch.zeros_like(x), noise_neural_walk_std).cumsum(dim=2)
        x = x + noise.to(device=device)

    if bias_allchans_neural_std:
        # bias is constant across time (i.e. the 3 conv inputs), and same for each channel
        biases = torch.normal(torch.zeros((x.shape[0], 1, 1)), bias_allchans_neural_std).repeat(1, x.shape[1], x.shape[2])
        x = x + biases.to(device=device)

    return x

# ------------------------
# --- Helper Functions ---
# ------------------------

def create_snippets(neural, 
                    kinematic, 
                    snip_min_bins=1, 
                    snip_max_bins=5, 
                    pad_bins=0):
        """Divides neural and kinematic data into short time snippets (used by other augmentation functions)

        Args:
            neural (array):         (N, NumChans) array of neural data
            kinematic (array):      (N, NumFings) array of velocities
            snip_min_bins (int):    minimum number of bins for a random snippet
            snip_max_bins (int):    maximum number of bins for a random snippet
            pad_bins (int):        number of extra bins to pad the beginning&end (overlaps with neighboring snippets)

        Returns:
            snip_list: list of tuples [(neursnip1, kinsnip1), (neursnip2, kinsnip2)...]
        """
        # create a bunch of random snippet lengths (more than enough)
        lengths = np.random.randint(snip_min_bins, snip_max_bins, size=round(neural.shape[0] / snip_min_bins))

        # if pad bins, add zeros on the very ends
        if pad_bins:
            neural = np.vstack((np.zeros([pad_bins, neural.shape[1]]),
                                neural,
                                np.zeros([pad_bins, neural.shape[1]])))
            kinematic = np.vstack((np.zeros([pad_bins, kinematic.shape[1]]),
                                kinematic,
                                np.zeros([pad_bins, kinematic.shape[1]])))

        # extract snippets
        ptr = pad_bins      # if not padding then starts at 0
        snip_list = []
        for length in lengths:
            snip_list.append((neural[ptr-pad_bins:ptr+length+pad_bins, :], kinematic[ptr-pad_bins:ptr+length+pad_bins, :]))
            ptr += length
            if ptr >= neural.shape[0] - (length + 2*pad_bins):
                break

        return snip_list

def add_noise_white(X,
                    std):
    """
        Adds white noise to each channel and timepoint (independent)

        Args:
            X (ndarray): numpy array of size [N, numfeats]
            std (float): standard deviation

        Returns:
            noise_X (ndarray): numpy array of size [N, numfeats] with noise added
    """
    noise = np.random.normal(loc = 0.0, scale = std, size = X.shape)
    noise_X = X + noise
    return noise_X

def add_noise_random_walk(X,
                          std):
    """
        Adds a random walk to each channel (noise is summed across time)
        
        Args:
            X (ndarray): numpy array of size [N, numfeats]
            std (float): standard deviation

        Returns:
            noise_X (ndarray): numpy array of size [N, numfeats] with noise added
    """
    noise = np.random.normal(loc=0.0, scale=std, size=X.shape)
    noise = np.cumsum(noise, axis=0)
    noise_X = X + noise
    return noise_X

def add_noise_constant(X,
                       std,
                       type = 'same'):
    """
        Bias is constant across time (i.e. the 3 conv inputs), can be different/same for each channel

        Args:
            X (ndarray): numpy array of size [N, numfeats]
            std (float): standard deviation
            type (string): Default 'same', options include 'different', 'same', determining whether bias is same for each channel

        Returns:
            bias_X (ndarray): numpy array of size [N, numfeats] with constant noise added
    """
    if(type == 'same'):
        biases = np.full(X.shape, np.random.normal(loc=0.0, scale=std))
    elif(type == 'different'):
        biases = np.random.normal(loc=0.0, scale=std, size=(1, X.shape[1]))
        biases = np.tile(biases, (X.shape[0], 1))
    else:
        Warning("Not a valid type, defaulting to same")
        biases = np.full(X.shape, np.random.normal(loc=0.0, scale=std))

    bias_X = X + biases
        
    return bias_X

def add_noise_white_tensor(X,
                    std,
                    device='cpu'):
    """
        Adds white noise to each channel and timepoint (independent)

        Args:
            X (tensor): tensor of size [N, numfeats]
            std (float): standard deviation
            device (device): Default 'cpu', torch device

        Returns:
            noise_X (tensor): tensor of size [N, numfeats] with noise added
    """
    noise = torch.normal(torch.zeros_like(X), std, device=device)
    noise_X = X + noise
    return noise_X

def add_noise_constant_tensor(X,
                       std,
                       type = 'same',
                       device = 'cpu'):
    """
        Bias is constant across time (i.e. the 3 conv inputs), can be different/same for each channel

        Args:
            X (tensor): tensor of size [N, numfeats]
            std (float): standard deviation
            type (string): Default 'same', options include 'different', 'same', determining whether bias is same for each channel
            device (device): Default 'cpu', torch device

        Returns:
            bias_X (tensor): tensor of size [N, numfeats] with constant noise added
    """
    if(type == 'same'):
        biases = torch.full(X.shape, torch.normal(mean=0.0, std=std), device=device)
    elif(type == 'different'):
        biases = torch.normal(mean=0.0, std=std, size=(1, X.shape[1]), device=device)
        biases = biases.repeat(X.shape[0], 1)
    else:
        Warning("Not a valid type, defaulting to same")
        biases = torch.full(X.shape, torch.normal(mean=0.0, std=std), device=device)

    bias_X = X + biases
        
    return bias_X

def normalize_moving_average(X, 
                             window,
                             axis
                             ):
    """
        Normalizes data to 0 mean 1 std using the giving window length 

        Args:
            X (ndarray): numpy array of size [N, numfeats]
            window (int): Number of values to normalize over
            axis (int): The axis to normalize over
            
        Returns:
            normalize_X: Normalized input data
    """
    if(window!=0):
        df = pd.DataFrame(X)
        df_norm = (df - df.rolling(window, min_periods=1, axis = axis).mean()) / df.rolling(window,min_periods=1, axis = axis).std()
        data_norm = df_norm.to_numpy()
        data_norm[np.isnan(data_norm)] = 0      # fill NaNs with 0 (i.e. the first few bins are zero)
    else:
        data_norm = (X - X.mean(axis=axis, keepdims=True)) / X.std(axis=axis, keepdims=True)
    normalize_X = data_norm
    return normalize_X

def normalize_scaler(X,
                     normalizer):
    """
        Normalizes data with given sklearn scaler.

        Args:
            X (ndarray): numpy array of size [N, numfeats]
            Normalizer (scipy.scaler): sklearn scaler
        
        Returns:
            normalize_X (ndarray): Normalized input data
    """
    if not isinstance(normalizer, BaseEstimator):
        raise ValueError("Not a valid sklearn scaler")
    normalizer.fit(X)
    normalize_X = normalizer.transform(X)
    return normalize_X, normalizer

# ----------------------
# --- Util Functions ---
# ----------------------

def add_time_history(X,
                     Y,
                     num_bins = 0,
                     padding = None,
                     reshape = False
                     ):
    """
        Takes in neural data X and behavior Y and returns two "adjusted" neural data and behavior matrices
        based on the optional params. The number of historical bins of neural data can be set.

        Args:
            X (ndarray): The neural data, which should be [t, neu] in size, where t is the numebr of smaples and neu is the number
                    of neurons.
            Y (ndarray): The behavioral data, which should be [t, dim] in size, where t is the number of samples and dim is the
                    number of states.
            num_bins (int, optional): Default 0. The number of bins to append to each sample of neural data from the previous 'hist' bins.
                    padding (string): Default None, disabled. This fills previous neural data with values before the experiment began. A single
                    scalar wil fill all previous neural data with that value. Otherwise, a [1,neu] ndarray equal to the first
                    dimension of X (# of neurons) should represent the value to fill for each channel.
            reshape (bool, optional): Default False. If history is added, will return the adjusted matrices either in 2d or 3d form (2d has the history appended
                    as extra columns, 3d has history as a third dimension. For example, reshape true returns a sample as:
                    [1, neu*hist+1] whereas reshape false returns: [1, neu, hist+1]. 

        Returns:
            adjX (ndarray): The adjusted neural data.
            adjY (ndarray): The adjusted behavioral data.
    """
    nNeu = X.shape[1]
    if padding is not None:
        if isinstance(padding, np.ndarray):
            Xadd = np.tile(padding, num_bins)
            Yadd = np.zeros((num_bins, Y.shape[1]))
        else:
            Xadd = np.ones((num_bins, nNeu))*padding
            Yadd = np.zeros((num_bins, Y.shape[1]))
        X = np.concatenate((Xadd, X))
        Y = np.concatenate((Yadd, Y))

    #reshape data to include historical bins
    adjX = np.zeros((X.shape[0]-num_bins, nNeu, num_bins+1))
    for h in range(num_bins+1):
        adjX[:,:,h] = X[h:X.shape[0]-num_bins+h,:]
    adjY = Y[num_bins:,:]

    if reshape:
        #NOTE: History will be succesive to each column (ie with history 5, columns 0-5 will be channel 1, 6-10
        # channel 2, etc..
        adjX = adjX.reshape(adjX.shape[0],-1)

    return adjX, adjY

def add_lag(X,
            Y,
            lag = 0
            ):
    """
        Takes in neural data X and behavior Y and returns two "adjusted" neural data and behavior matrices
        based on the optional params. The amount of lag between neural data and behavior can be set in units.

        Args:
            X (ndarray): The neural data, which should be [t, neu] in size, where t is the numebr of smaples and neu is the number
                    of neurons.
            Y (ndarray): The behavioral data, which should be [t, dim] in size, where t is the number of samples and dim is the
                    number of states.
            lag (int, optional): Defaults to 0. The number of bins to lag the neural data relative to the behavioral data. For example,
                    add_lag(X,Y, lag=1) will return X[0:-1] for adjX and Y[1:] for adjY.

        Returns:
            adjX (ndarray): The adjusted neural data.
            adjY (ndarray): The adjusted behavioral data.
    """
    adjX = X[0:-lag,:]
    adjY = Y[lag:,:]
    
    return adjX, adjY

def add_noise(X,
              method,
              std,
              type = None,
              device = 'cpu'):
    """
        Add noise with method provided.
        N.B. Outputs from pytorch tensor version and numpy ndarray version will be different, even with the same seed.

        Args:
            X (ndarray/tensor): numpy array/pytorch tensor of size [N, numfeats], N.B. random_walk cannot be performed on pytorch tensor
            method (string): method of adding the noise, options include 'white', 'random_walk', and 'constant'.
            std (float): standard deviation
            type (string): Default None, only for 'constant' method, options include 'different', 'same', determining whether bias is same for each channel
            device (string): Default 'cpu', only for pytorch tensor inputs.

        Returns:
            noise_X (ndarray.tensor): numpy array/pytorch tensor of size [N, numfeats] with noise added
    """
    if isinstance(X, np.ndarray):
        if(method == 'white'):
            noise_X = add_noise_white(X,std)
        elif(method == 'random_walk'):
            noise_X = add_noise_random_walk(X,std)
        elif(method == 'constant'):
            noise_X = add_noise_constant(X,std,type)
        else:
            Warning("Not a valid method, defaulting to white")
            noise_X = add_noise_white(X,std)
        return noise_X
    elif isinstance(X, torch.Tensor):
        if(method == 'white'):
            noise_X = add_noise_white_tensor(X,std,device)
        elif(method == 'constant'):
            noise_X = add_noise_constant_tensor(X,std,type,device)
        else:
            Warning("Not a valid method, defaulting to white")
            noise_X = add_noise_white_tensor(X,std,device)
        return noise_X
    else:
        raise TypeError("X must be a np.ndarray or a torch.Tensor.")

def normalize(X, 
              method = 'moving_average',
              **kwargs
              ):
    """
        Wrapper function for different normalization methods

        Args:
            X (ndarray): numpy array of size [N, numfeats]
            method (string): Method to use for normalization. Options includes 'moving_average', 'scipy'.
            **kwargs: additional parameters. For moving average, it includes 'win_size', and 'axis'. For scipy, it includes 'normalizer'.

        Returns:
            normalize_X (ndarray): Normalized input data
            normalizer (scipy.scaler): Scipy normalizer, none if method is moving average
    """

    if(method == "moving_average"):
        win_size = kwargs.get('win_size')
        axis = kwargs.get('axis')
        if win_size is None or axis is None:
            raise ValueError("moving_average method requires 'win_size' and 'axis'")
        normalize_X = normalize_moving_average(X, win_size, axis)
        return normalize_X, None
    elif(method == "scipy"):
        normalizer = kwargs.get('normalizer')
        if normalizer is None:
            raise ValueError("scipy method requires 'normalizer'")
        normalize_X, normalizer = normalize_scaler(X, normalizer)
        return normalize_X, normalizer

def resample(Y,
             num_resamp_floor = 2e4,
             method = None,
             start = None,
             end = None,
             **kwargs
             ):
    """
        Resample data to most closely match a specified distribution (tri, gauss, uni)

        Args:
            Y (ndarray): numpy array of size [N, numfeats]
            num_resamp_floor (int): Default 2e4 (as PyBMI code), minimum number of samples resampled
            start (int): Default None (not slicing), starting index for the resampling dimension 
            end (int): Default None (not slicing), ending index for the resampling dimension 
            method (string): Default None, Target distribution for resampling, current options: 'std_gauss', 'gauss', 'tri', 'uni', None
            **kwargs: Additional args for parameters of distribution. ('mean', 'std' for gauss)

        Returns:
            idx (ndarray): Resampling indices for given distribution (None if method = None)
    """
    num_resamp = int(np.max((num_resamp_floor,Y.shape[0])))

    vy = Y[:,start:end] #pull off velocities
        # N.B. In the original code the slicing should correspond to start = int(Y.shape[1] / 2), end = None
    

    if isinstance(method, str):
        if method == 'std_gauss':
            ny = (vy - np.mean(vy))/np.std(vy)
            pd_vec = 2*np.random.randn(num_resamp,int(vy.shape[1]))
        elif method == 'gauss':
            # this method specifies specific parameters of gaussian (rather than a standard gaussian)
            mean = kwargs.get('mean')
            std = kwargs.get('std')
            ny = vy
            if mean is None or std is None:
                raise ValueError("gauss method requires 'mean', and 'std'")
            pd_vec = np.random.normal(mean, std, (num_resamp,int(vy.shape[1])))
        elif method == 'tri':
            # left = kwargs.get('left')
            # mode = kwargs.get('mode')
            # right = kwargs.get('right')
            # if left is None or mode is None or right is None:
            #     raise ValueError("tri method requires 'left', 'mode', and 'right'")
            # pd_vec = np.random.triangular(left, mode,right, (num_resamp,int(vy.shape[1])))
            ny = (vy - np.mean(vy))/np.std(vy)
            np.random.triangular(-4, 0, 4, (num_resamp, int(vy.shape[1])))
        elif method == 'uni':
            # low = kwargs.get('low')
            # high = kwargs.get('high')
            
            # if low is None or high is None: 
            #     raise ValueError("uni method requires 'low', and 'high'")
            # pd_vec = np.random.uniform(low, high, (num_resamp,int(vy.shape[1])))
            ny = (vy - np.mean(vy))/np.std(vy)
            pd_vec = np.random.uniform(-4, 4, (num_resamp, int(vy.shape[1])))
        else:
            Warning("Not a valid distribution defaulting to None")
            pd_vec = None

    elif method == None:
        pd_vec = None
    else:
        Warning("Not a valid distribution, defaulting to None")
        pd_vec = None

    if pd_vec is not None:
        kdt = KDTree(ny)
        idx = kdt.query(pd_vec)[1]
    else:
        idx = None
    
    return idx

def time_warp(neural, kinematic, snip_min_bins=30, snip_max_bins=150, min_warp=0.5, max_warp=2.0):
    """Randomly stretches neural & kinematic data in time

    Args:
        neural (array):         (N, NumChans) array of neural data
        kinematic (array):      (N, NumFings) array of velocities
        snip_min_bins (int):    minimum number of bins for a random snippet
        snip_max_bins (int):    maximum number of bins for a random snippet
        min_warp (float):       min (slowest) warp. E.g. 0.5 means a snippet could be at 50% speed
        max_warp (float):       max (flastest) warp. E.g. 1.5 means a snippet could be at 150% speed

    Returns:
        (neural_warp, kinematic_warp): augmented data arrays
    """

    # divide data into snippets
    snip_list = create_snippets(neural, kinematic, snip_min_bins, snip_max_bins)

    # decide how much to warp each snippet (uniform random)
    warp_factors = np.random.uniform(min_warp, max_warp, len(snip_list))

    # warp each snippet and concat together
    #   0.5 means slower, so 200% as many samples
    #   2.0 means faster, so 50% as many samples
    #   k -> 1/k samples in the augmented version
    neural_warp_list = []
    kin_warp_list = []
    for snip, warp in zip(snip_list, warp_factors):
        snip_neu = snip[0]
        snip_kin = snip[1]

        # setup new time vectors
        t_orig = np.linspace(0, 1, snip_neu.shape[0])
        t_warp = np.linspace(0, 1, round(snip_neu.shape[0] * (1. / warp)))

        # init warped data
        snip_neu_warp = np.zeros((len(t_warp), snip_neu.shape[1]))
        snip_kin_warp = np.zeros((len(t_warp), snip_kin.shape[1]))

        # warp each neural channel
        for chan in range(snip_neu.shape[1]):
            snip_neu_warp[:, chan] = np.interp(t_warp, t_orig, snip_neu[:, chan])

        # warp each kinematic channel
        for chan in range(snip_kin.shape[1]):
            snip_kin_warp[:, chan] = np.interp(t_warp, t_orig, snip_kin[:, chan])

        # save the warped snippet
        neural_warp_list.append(snip_neu_warp)
        kin_warp_list.append(snip_kin_warp)

    # return as array
    neural_warp = np.vstack(neural_warp_list)
    kinematic_warp = np.vstack(kin_warp_list)
    return neural_warp, kinematic_warp

def random_shift(neural, 
                 kinematic, 
                 snip_min_bins=30, 
                 snip_max_bins=150, 
                 shift_max_bins=1
                 ):
    """Randomly shifts y (kinematic) dimensions in time to help reduce correlation among dimensions. Each y-dimension
    has an independent shift applied (+-shift_max_bins), with possibility for 0 shift. Applies shifts to randomly
    selected snippets.

    Args:
        neural (ndarray):         (N, NumChans) array of neural data
        kinematic (ndarray):      (N, NumFings) array of velocities
        snip_min_bins (int):    minimum number of bins for a random snippet
        snip_max_bins (int):    maximum number of bins for a random snippet
        shift_max_bins (int):   (absolute value) maximum number of bins for time shift. E.g. 2 means up to +-2 bins.

    Returns:
        (neural_warp, kinematic_warp): augmented data ndarrays
    """

    # divide data into snippets (snippets have an extra bin at the beginning & end, to be cropped)
    pad = shift_max_bins
    snip_list = create_snippets(neural, kinematic, snip_min_bins, snip_max_bins, pad_bins=pad)

    # decide shift amounts for each kinematic dimension, for each snippet
    shift_amts = np.random.randint(-1*shift_max_bins, shift_max_bins, size=(kinematic.shape[1], len(snip_list)))

    # apply shifts to each snippet and concat together
    neural_shift_list = []
    kin_shift_list = []
    for snip, shift in zip(snip_list, shift_amts.T):
        # shift each kinematic channel
        #     - +1 means kinematics are shifted forward in time (delayed 1 bin)
        #     - shift has shape (NumFings,)
        snip_kin = snip[1]
        snip_kin_shift = np.zeros_like(snip_kin[pad:-1*pad, :])
        for chan in range(snip_kin.shape[1]):
            # remove the padded bins at start & end, and apply the shift
            end_idx = -1*pad-shift[chan] if (-1*pad-shift[chan] != 0) else snip_kin.shape[0]
            snip_kin_shift[:, chan] = snip_kin[pad-shift[chan]:end_idx, chan]

        # save the shifted snippet
        neural_shift_list.append(snip[0][pad:-1*pad, :])    # neural remains un-shifted, just remove the padding
        kin_shift_list.append(snip_kin_shift)

    # return as array
    neural_shift = np.vstack(neural_shift_list)
    kinematic_shift = np.vstack(kin_shift_list)
    return neural_shift, kinematic_shift

def remove_zeros(X,
                 Y):
    """
    Remove pairs of data that contains 0 in Y

    Args:
        X (ndarray): numpy array of size [N, numfeats]
        Y (ndarray): numpy array of size [N, numfeats]

    Returns:
        no_zero_X (ndarray): numpy array of size [N, numfeats] with zero Y pairs removed
        no_zero_X (ndarray): numpy array of size [N, numfeats] with zero Y pairs removed
    """
    mask = np.all(Y!=0, axis=1)
    no_zero_X = X[mask]
    no_zero_Y = Y[mask]
    return no_zero_X, no_zero_Y