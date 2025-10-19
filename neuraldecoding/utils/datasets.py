import torch
from torch.utils.data import Dataset

class NeuralDecodingDataset(Dataset):
    """Torch Dataset for predicting behavior from neural data.
       Has functionality to return other data in the dataset such as trial lengths.
       Also allows for interpipe parameters to be passed through.
    """

    def __init__(self,
                 data,
                 interpipe,
                 xkey,
                 ykey,
                 otherdatakeys=None,
                 otherinterpipekeys=None):
        """
        Args:
            data (dict): Input data dictionary containing the data and relevant information.
            xkey (str): Location of the neural data in the data dictionary.
            ykey (str): Location of the behavior data in the data dictionary.
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
        if isinstance(otherinterpipekeys, str):
            otherinterpipekeys = [otherinterpipekeys]

        self.otherkeys = otherdatakeys or [] + otherinterpipekeys or []
        for key in otherdatakeys or []:
            setattr(self, key, data[key])
        for key in otherinterpipekeys or []:
            setattr(self, key, interpipe[key])


    def __len__(self):
        return len(self.neural)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        neu = self.neural[idx, :]
        kin = self.kin[idx, :] 

        sample = {'neural': neu, 'behavior': kin}
        
        for key in self.otherkeys or []:
            sample[key] = getattr(self, key)[idx]

        return sample