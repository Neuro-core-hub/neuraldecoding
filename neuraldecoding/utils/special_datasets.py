import torch
from torch.utils.data import Dataset

class BehaviorDatasetCustom(Dataset):
    """Torch Dataset for predicting finger position/velocity from neural data.
       Has functionality to return other data in the dataset such as trial lengths, onset, movement direction, etc.
       Also allows for interpipe parameters to be passed through.
       May be slower than the default BehaviorDataset, but allows for more flexibility.
    """

    def __init__(self,
                 data,
                 xkey,
                 ykey,
                 otherdatakeys=None,
                 otherdatakeys_data=None,
                 device='cpu'):
        """
        Args:
            data (dict): Input data dictionary containing the data and relevant information.
            interpipe (dict): Interpipe dictionary containing information shared between blocks of the preprocessing pipeline.
            xkey (str): Location of the neural data in the data dictionary.
            ykey (str): Location of the finger data in the data dictionary.
            otherkeys (str, list, optional): Names of additional useful data to include outside of neural and behavioral data. Defaults to None.
            otherdatakeys_data (str, list, optional): Keys in the data dictionary corresponding to otherkeys. Defaults to None.
        """
        # store the processed X/Y data
        if not isinstance(data[xkey], torch.Tensor):
            data[xkey] = torch.Tensor(data[xkey], device=device, dtype=torch.float32)
        if not isinstance(data[ykey], torch.Tensor):
            data[ykey] = torch.Tensor(data[ykey], device=device, dtype=torch.float32)
        self.neural = data[xkey]
        self.kin = data[ykey]
        self.params = {}

        # store any other data that is needed for processing
        if isinstance(otherdatakeys, str):
            otherdatakeys = [otherdatakeys]

        # loop through other potential data and store it as attributes
        self.otherkeys = otherdatakeys
        for name, key in zip(otherdatakeys or [], otherdatakeys_data or []):
            setattr(self, name, data[key])

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