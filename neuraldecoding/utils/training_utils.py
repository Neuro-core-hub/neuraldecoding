import numpy as np
import torch

def calc_corr(y1,y2):
    """Calculates the correlation between y1 and y2 (tensors)"""
    corr = []
    for i in range(y1.shape[1]):
        corr.append(np.corrcoef(y1[:, i], y2[:, i])[1, 0])
    return corr

class OutputScaler:
    def __init__(self, gains, biases):
        """An object to linearly scale data, like the output of a neural network

        Args:
            gains (1d np array):  [1,NumOutputs] array of gains
            biases (1d np array):           [1,NumOutputs] array of biases
        """
        self.gains = gains
        self.biases = biases

    def fit(self, model, loader, device = 'cpu', dtype = torch.float32, num_outputs=2, verbose=True):
        self.device = device
        self.dtype = dtype
        self.gains, self.biases = generate_output_scaler(model, loader, device = device, dtype = dtype, num_outputs=num_outputs, verbose=verbose, get_param = True)

    def scale(self, data):
        """
        data should be an numpy array/tensor of shape [N, NumOutputs]
        :param data:    np.ndarray or torch.Tensor, data to scale
        :return scaled_data np.ndarray or torch.Tensor, returns either according to what was input
        """

        # check if input is tensor or numpy
        isTensor = False
        if type(data) is torch.Tensor:
            isTensor = True
            data = data.cpu().detach().numpy()
        N = data.shape[0]

        # scale data
        scaled_data = np.tile(self.gains, (N, 1)) * data + np.tile(self.biases, (N, 1))

        # convert back to tensor if needed
        if isTensor:
            scaled_data = torch.from_numpy(scaled_data)

        return scaled_data

    def unscale(self, data):
        """Data should be an numpy array/tensor of shape [N, NumOutputs].
            Performs the inverse of the scale function (used in Refit)"""
        N = data.shape[0]
        is_tensor = False
        if type(data) is torch.Tensor:
            is_tensor = True
            data = data.cpu().detach().numpy()
        # unscaled_data = (data / np.tile(self.gains, (N, 1))) - np.tile(self.biases, (N, 1))
        unscaled_data = (data - np.tile(self.biases, (N, 1))) / np.tile(self.gains, (N, 1))
        if is_tensor:
            unscaled_data = torch.from_numpy(unscaled_data).to(device=self.device, dtype=self.dtype)
        return unscaled_data

def generate_output_scaler(model, loader, device = 'cpu', dtype = 'float32', num_outputs=2, verbose=True, get_param = False):
    """Returns a scaler object that scales the output of a decoder

    Args:
        model:      model
        loader:     dataloader
        num_outputs:  how many outputs (2)

    Returns:
        scaler: An OutputScaler object that takes returns scaled version of input data. If refit, this is the
                composition of the original and new scalers.
    """
    # fit constants using regression
    model.eval()  # set model to evaluation mode
    batches = len(list(loader))
    with torch.no_grad():
        for x, y in loader:

            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)
            yhat = model(x)

            if isinstance(yhat, tuple):
                # RNNs return y, h
                yhat = yhat[0]

            num_samps = yhat.shape[0]
            num_outputs = yhat.shape[1]
            yh_temp = torch.cat((yhat, torch.ones([num_samps, 1]).to(device)), dim=1)

            # train ~special~ theta
            # (scaled velocities are indpendent of each other - this is the typical method)
            # Theta has the following form: [[w_x,   0]
            #                                [0,   w_y]
            #                                [b_x, b_y]]
            theta = torch.zeros((num_outputs + 1, num_outputs)).to(device=device, dtype=dtype)
            for i in range(num_outputs):
                yhi = yh_temp[:, (i, -1)]
                thetai = torch.matmul(torch.mm(torch.pinverse(torch.mm(torch.t(yhi), yhi)), torch.t(yhi)), y[:, i])
                theta[i, i] = thetai[0]  # gain
                theta[-1, i] = thetai[1]  # bias
                if verbose:
                    print("Finger %d RR Calculated Gain, Offset: %.6f, %.6f" % (i, thetai[0], thetai[1]))

    gains = np.zeros((1, num_outputs))
    biases = np.zeros((1, num_outputs))
    for i in range(num_outputs):
        gains[0, i] = theta[i, i]
        biases[0, i] = theta[num_outputs, i]

    if get_param:
        return gains, biases
    else:
        return OutputScaler(gains, biases)