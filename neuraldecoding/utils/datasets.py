import torch
from torch.utils.data import Dataset

class BehaviorDataset(Dataset):
    """ Torch Dataset for predicting finger position/velocity from neural data. """

    def __init__(self,
                 data,
                 xkey,
                 ykey):
        """
        Args:
            data (dict): Input data dictionary containing the data and relevant information.
            xkey (str): Location of the neural data in the data dictionary.
            ykey (str): Location of the finger data in the data dictionary.
        """
        # store the processed X/Y data
        if not isinstance(data[xkey], torch.Tensor):
            data[xkey] = torch.Tensor(data[xkey])
        if not isinstance(data[ykey], torch.Tensor):
            data[ykey] = torch.Tensor(data[ykey])
        self.neural = data[xkey]
        self.kin = data[ykey]

    def __len__(self):
        return len(self.neural)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        neu = self.neural[idx, :]
        kin = self.kin[idx, :] 

        sample = {'neu': neu, 'kin': kin}

        return sample

class BehaviorDatasetCustom(Dataset):
    """Torch Dataset for predicting finger position/velocity from neural data.
       Has functionality to return other data in the dataset such as trial lengths.
       Also allows for interpipe parameters to be passed through.
       May be slower than the default BehaviorDataset, but allows for more flexibility.
    """

    def __init__(self,
                 data,
                 interpipe,
                 xkey,
                 ykey,
                 otherdatakeys=None):
        """
        Args:
            data (dict): Input data dictionary containing the data and relevant information.
            interpipe (dict): Interpipe dictionary containing information shared between blocks of the preprocessing pipeline.
            xkey (str): Location of the neural data in the data dictionary.
            ykey (str): Location of the finger data in the data dictionary.
            otherkeys (str, list, optional): Additional locations in the data dictionary to include. Defaults to None.
            otherinterpipekeys (str, list, optional): Additional keys in the interpipe dictionary to include. Defaults to None.
        """
        # store the processed X/Y data
        if not isinstance(data[xkey], torch.Tensor):
            data[xkey] = torch.Tensor(data[xkey])
        if not isinstance(data[ykey], torch.Tensor):
            data[ykey] = torch.Tensor(data[ykey])
        self.neural = data[xkey]
        self.kin = data[ykey]
        self.params = {}

        # store any other data that is needed for processing
        if isinstance(otherdatakeys, str):
            otherdatakeys = [otherdatakeys]

        self.otherkeys = otherdatakeys
        for key in otherdatakeys or []:
            setattr(self, key, data[key])


    def __len__(self):
        return len(self.neural)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        neu = self.neural[idx, :]
        kin = self.kin[idx, :] 

        sample = {'neu': neu, 'kin': kin}
        
        for key in self.otherkeys or []:
            sample[key] = getattr(self, key)[idx]

        return sample