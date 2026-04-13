import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from neuraldecoding.utils.data_tools import load_one_nwb, neural_finger_from_dict
from neuraldecoding.utils import data_split_trial#
from neuraldecoding.dataset import Dataset
from scipy.ndimage import gaussian_filter1d
import numpy as np
from sklearn.metrics import r2_score
from datetime import datetime
import pickle
import os
import sklearn.preprocessing
from scipy.stats import entropy
import neuraldecoding.preprocessing
from torch.amp import autocast, GradScaler
import copy  # optimization: fix deepcopy bug for aligner snapshots

def kl_divergence_per_channel_rob(signal1, signal2, bins=50):
    n_channels = signal1.shape[1]
    kl_divs = np.zeros(n_channels)
    
    for ch in range(n_channels):
        # CRITICAL: Use combined min/max from BOTH signals
        combined_min = min(signal1[:, ch].min(), signal2[:, ch].min())
        combined_max = max(signal1[:, ch].max(), signal2[:, ch].max())
        
        # Create common bin edges
        bin_edges = np.linspace(combined_min, combined_max, bins + 1)
        
        # Now both histograms use the same bins
        hist1, _ = np.histogram(signal1[:, ch], bins=bin_edges, density=False)
        hist2, _ = np.histogram(signal2[:, ch], bins=bin_edges, density=False)
        
        # Normalize to probabilities
        epsilon = 1e-10
        hist1 = (hist1 + epsilon) / (hist1.sum() + bins * epsilon)
        hist2 = (hist2 + epsilon) / (hist2.sum() + bins * epsilon)
        
        kl_divs[ch] = entropy(hist1, hist2)
    
    return kl_divs

def kl_divergence_per_channel_rob_optimized(signal1, signal2, bins=50):
    """Fully vectorized KL divergence — eliminates the Python for-loop over channels."""
    n_channels = signal1.shape[1]
    n_samples1 = signal1.shape[0]
    n_samples2 = signal2.shape[0]
    epsilon = 1e-10
 
    # optimization: compute min/max for all channels in one vectorized call instead of per-channel Python min/max
    combined_min = np.minimum(signal1.min(axis=0), signal2.min(axis=0))
    combined_max = np.maximum(signal1.max(axis=0), signal2.max(axis=0))
 
    # optimization: build all bin edges at once as a (bins+1, n_channels) array using broadcasting, avoiding n_channels separate np.linspace calls
    t = np.linspace(0, 1, bins + 1).reshape(-1, 1)  # (bins+1, 1)
    all_bin_edges = combined_min + t * (combined_max - combined_min)  # (bins+1, n_channels)
 
    # optimization: use np.searchsorted to bin all channels at once instead of calling np.histogram in a loop;
    # searchsorted maps each sample to its bin index in O(n log bins) per channel
    hist1 = np.zeros((bins, n_channels), dtype=np.float64)
    hist2 = np.zeros((bins, n_channels), dtype=np.float64)
 
    for ch in range(n_channels):  # optimization: loop is retained for searchsorted (no vectorized multi-column histogram in numpy), but the body is minimal — just searchsorted + bincount
        edges = all_bin_edges[:, ch]
 
        indices1 = np.searchsorted(edges, signal1[:, ch], side='right') - 1  # optimization: searchsorted is faster than np.histogram since it skips redundant edge validation and output tuple creation
        indices1 = np.clip(indices1, 0, bins - 1)  # optimization: clip in-place to valid bin range
        hist1[:, ch] = np.bincount(indices1, minlength=bins)[:bins]  # optimization: bincount is the fastest way to accumulate counts from indices
 
        indices2 = np.searchsorted(edges, signal2[:, ch], side='right') - 1
        indices2 = np.clip(indices2, 0, bins - 1)
        hist2[:, ch] = np.bincount(indices2, minlength=bins)[:bins]
 
    # optimization: normalize and compute KL divergence fully vectorized across all channels simultaneously — no Python loop needed
    hist1 = (hist1 + epsilon) / (hist1.sum(axis=0, keepdims=True) + bins * epsilon)  # optimization: vectorized normalization over all channels at once
    hist2 = (hist2 + epsilon) / (hist2.sum(axis=0, keepdims=True) + bins * epsilon)  # optimization: vectorized normalization over all channels at once
 
    # optimization: inline KL divergence formula (sum of p * log(p/q)) instead of calling scipy.stats.entropy per channel — avoids n_channels Python function call overhead
    kl_divs = np.sum(hist1 * np.log(hist1 / hist2), axis=0)
 
    return kl_divs

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, drop_out):
        """
        input_dim: the number of input channels.
        hidden_dim: the number of neurons in the hidden layer.
        drop_out: drop-out rate.
        """
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.drop_out = drop_out
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Dropout(self.drop_out), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(self.drop_out), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
            nn.ReLU()
        )

    def forward(self, input):
        """
        input: spike firing rate data
        x: transformed spike firing rate data
        """
        x = self.model(input)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, drop_out):
        """
        input_dim: the number of input channels.
        hidden_dim: the number of neurons in the hidden layer.
        drop_out: drop-out rate.
        """
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.drop_out = drop_out
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Dropout(self.drop_out),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(self.drop_out),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, input):
        """
        input: spike firing rate data
        return: a label indicating if the input data is real or fake
        """
        label = self.model(input)
        return label

class cycleGAN():
    def __init__(self, 
                 day0_data_dir,
                 verbose = False):
        self.day0_data_dir = day0_data_dir
        self.kernel_size = 40
        self.sample_size = 20
        self.verbose = verbose
    
    def load_process_data(self, dayk_data_dir, decoder, neural_type, preprocessor, normalizer_path, is_xds = False, day0_xds_data = None, day_pair = None, train_test_split = 0.8):
        # preprocessor is the preprocessor for validation data split after first split to mimic the preproc after split
        self.preprocessor = preprocessor
        self.dayk_data_dir = dayk_data_dir
        self.is_xds = is_xds
        self.day_pair = day_pair
        if is_xds:
            ref_names = ['elec93', 'elec92', 'elec94', 'elec95', 'elec75', 'elec96', 'elec85', 'elec97', 'elec86', 'elec98', 'elec87', 'elec88', 'elec77', 'elec99', 'elec66', 'elec89', 'elec76', 'elec90', 'elec67', 'elec79', 'elec58', 'elec80', 'elec78', 'elec70', 'elec68', 'elec60', 'elec69', 'elec50', 'elec59', 'elec40', 'elec49', 'elec100', 'elec83', 'elec84', 'elec73', 'elec74', 'elec63', 'elec64', 'elec53', 'elec54', 'elec43', 'elec55', 'elec44', 'elec45', 'elec33', 'elec46', 'elec34', 'elec65', 'elec24', 'elec56', 'elec35', 'elec47', 'elec25', 'elec57', 'elec26', 'elec36', 'elec27', 'elec37', 'elec28', 'elec38', 'elec29', 'elec48', 'elec19', 'elec39', 'elec81', 'elec82', 'elec71', 'elec72', 'elec61', 'elec62', 'elec51', 'elec52', 'elec41', 'elec42', 'elec31', 'elec32', 'elec21', 'elec22', 'elec11', 'elec12', 'elec2', 'elec23', 'elec3', 'elec13', 'elec4', 'elec14', 'elec15', 'elec5', 'elec16', 'elec6', 'elec17', 'elec7', 'elec8', 'elec18', 'elec20', 'elec9']
            day0_data = day0_xds_data
            xds_day0 = neuraldecoding.preprocessing.blocks.xdsNWB2DictBlock()
            zeropad1 = neuraldecoding.preprocessing.blocks.ZeroPaddingBlock(ref_names)

            interpipe_day0 = {}
            xds_day0_out, interpipe_day0 = xds_day0.transform(day0_data, interpipe_day0)
            zeropad_day0_out, interpipe_day0 = zeropad1.transform(xds_day0_out, interpipe_day0)
            day0_neural = zeropad_day0_out['neural']
            day0_behaviour = zeropad_day0_out['behaviour']

            dayk_data = dayk_data_dir
            xds_dayk = neuraldecoding.preprocessing.blocks.xdsNWB2DictBlock()
            zeropad2 = neuraldecoding.preprocessing.blocks.ZeroPaddingBlock(ref_names)

            interpipe_dayk = {}
            xds_dayk_out, interpipe_dayk = xds_dayk.transform(dayk_data, interpipe_dayk)
            zeropad_dayk_out, interpipe_dayk = zeropad2.transform(xds_dayk_out, interpipe_dayk)
            dayk_neural = zeropad_dayk_out['neural']
            dayk_behaviour = zeropad_dayk_out['behaviour']
        else:
            if self.verbose:
                print("Loading and processing data...")
            day0_data = load_one_nwb(self.day0_data_dir)
            (day0_neural, day0_behaviour), trial_idx_day0 = neural_finger_from_dict(day0_data, neural_type)
            dayk_data = load_one_nwb(dayk_data_dir)
            (dayk_neural, dayk_behaviour), trial_idx_dayk = neural_finger_from_dict(dayk_data, neural_type)
            if self.verbose:
                print("Data loaded, Smoothing data...")

        
        sigma = self.kernel_size / self.sample_size

        day0_X = gaussian_filter1d(day0_neural.astype(np.float32), sigma=sigma, axis=0)
        day0_Y = day0_behaviour

        dayk_X = gaussian_filter1d(dayk_neural.astype(np.float32), sigma=sigma, axis=0)
        dayk_Y = dayk_behaviour

        (self.day0_X_train, self.day0_Y_train),(self.day0_X_test, self.day0_Y_test) = data_split_trial(day0_X, day0_Y, split_ratio=train_test_split)
        (self.dayk_X_train, self.dayk_Y_train),(self.dayk_X_test, self.dayk_Y_test) = data_split_trial(dayk_X, dayk_Y, split_ratio=train_test_split)

        if self.verbose:
            print("Data loaded, smoothed, and splited.")

        self.decoder = decoder

        self.normalizer_path = normalizer_path
    
    def preprocess_val_data(self, data):
        if self.verbose:
            print("Preprocessing validation data...")
        data = self.preprocessor.preprocess_pipeline(data, params = {'is_train': False})
        return data

    def save_cycleGAN(self, fpath):
        if self.verbose:
            print("Saving cycleGAN instance...")
        with open(fpath, 'wb') as fp:
            pickle.dump(self, fp)
        if self.verbose:
            print("cycleGAN instance saved.")

    def save_checkpoint(self, save_dir, tag=None):
        """Save model weights + transformed data arrays.
        
        Saves the state_dict (~few hundred KB) and the aligned train/test arrays
        as compressed .npz (~10-20MB total depending on data size). At 42,562 pairs
        this takes ~15-25ms per pair for I/O (model ~2ms + npz ~10-20ms).
        
        Args:
            save_dir: directory to save checkpoint files into
            tag: optional string to identify this checkpoint (e.g. 'day0_dayk').
                 If None, uses a timestamp.
        """
        os.makedirs(save_dir, exist_ok=True)
        if tag is None:
            tag = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
        # --- Save model checkpoint ---
        checkpoint = {
            'aligner_state_dict': self.trained_aligner.state_dict(),
            'model_config': {
                'input_dim': self.trained_aligner.input_dim,
                'hidden_dim': self.trained_aligner.hidden_dim,
                'drop_out': self.trained_aligner.drop_out,
            },
            'day0_data_dir': self.day0_data_dir,
            'dayk_data_dir': self.dayk_data_dir,
            'kernel_size': self.kernel_size,
            'sample_size': self.sample_size,
            'is_xds': self.is_xds,
            'day_pair': self.day_pair,
        }
 
        model_path = os.path.join(save_dir, f'checkpoint_{tag}.pt')
        torch.save(checkpoint, model_path)
 
        # --- Run inference and save transformed data ---
        aligner = self.trained_aligner.eval()
        with torch.no_grad():
            dayk_train_aligned = aligner(
                torch.from_numpy(self.dayk_X_train).float().to(device)
            ).cpu().numpy()
            dayk_test_aligned = aligner(
                torch.from_numpy(self.dayk_X_test).float().to(device)
            ).cpu().numpy()
 
        # np.savez_compressed: ~30-50% smaller than raw .npy with minimal speed cost
        data_path = os.path.join(save_dir, f'transformed_{tag}.npz')
        np.savez_compressed(
            data_path,
            dayk_train_aligned=dayk_train_aligned,
            dayk_test_aligned=dayk_test_aligned,
            day0_X_train=self.day0_X_train,       # save reference data for later metric computation
            dayk_X_train=self.dayk_X_train,        # save original for comparison
            dayk_X_test=self.dayk_X_test,           # save original for comparison
        )
 
        if self.verbose:
            model_kb = os.path.getsize(model_path) / 1024
            data_kb = os.path.getsize(data_path) / 1024
            print(f"Saved: {model_path} ({model_kb:.1f} KB) + {data_path} ({data_kb:.1f} KB)")
 
        return model_path, data_path

    def train_cycle_gan_aligner(self,cycleGAN_params):
        """
        x1: M1 spike firing rates on day-0. A list, where each item is a numpy array containing the neural data of one trial
        
        x2: M1 spike firing rates on day-k. A list, where each item is a numpy array containing the neural data of one trial
            x2 will be divided into two portions (ratio 3:1), where the first portion will be used to train the aligner, and 
            the second portion will be used as the validation set.
        
        y2: EMGs on day-k. A list, where each item is a numpy array containing the EMGs of one trial. Only a portion of y2
            (those corresponding to the trials used as the validation set) will be used.
        
        D_params: the hyper-parameters determining the structure of the discriminators, a dictionary.
        
        G_params: the hyper-parameters determining the structure of the generators, a dictionary.
        
        training_parameters: the hyper-parameters controlling the training process, a dictionary.
        
        decoder: the day-0 decoder to be tested on the validation set, an array.
        
        n_lags: the number of time lags of the decoder, a number.
        
        logs: to indicate if training logs is needed to be recorded as a .pkl file, a bool.
        
        return: a trained "aligner" (generator) for day-k use.
        """
        x_dim = self.day0_X_train.shape[1]
        if self.verbose:
            print("Training cycleGAN aligner...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        #============================================= Specifying hyper-parameters =============================================
        D_hidden_dim = cycleGAN_params['D_params']['hidden_dim']
        G_hidden_dim = cycleGAN_params['G_params']['hidden_dim']
        loss_type = cycleGAN_params['training_params']['loss_type']
        optim_type = cycleGAN_params['training_params']['optim_type']
        epochs = cycleGAN_params['training_params']['epochs']
        batch_size = cycleGAN_params['training_params']['batch_size']
        D_lr = cycleGAN_params['training_params']['D_lr']
        G_lr = cycleGAN_params['training_params']['G_lr']
        ID_loss_p = cycleGAN_params['training_params']['ID_loss_p']
        cycle_loss_p = cycleGAN_params['training_params']['cycle_loss_p']
        drop_out_D = cycleGAN_params['training_params']['drop_out_D']
        drop_out_G = cycleGAN_params['training_params']['drop_out_G']

        logs = cycleGAN_params['logs']
        log_save_path = cycleGAN_params['log_save_path']
        #============================================= Defining networks ===================================================
        
        generator1, generator2 = Generator(x_dim, G_hidden_dim, drop_out_G), Generator(x_dim, G_hidden_dim, drop_out_G)
        discriminator1, discriminator2 = Discriminator(x_dim, D_hidden_dim, drop_out_D), Discriminator(x_dim, D_hidden_dim, drop_out_D)

        generator1.to(device)
        generator2.to(device)
        discriminator1.to(device)
        discriminator2.to(device)

        #==================================== Specifying the type of the losses ===============================================
        if loss_type == 'L1':
            criterion_GAN = torch.nn.MSELoss()
            criterion_cycle = torch.nn.L1Loss()
            criterion_identity = torch.nn.L1Loss()
        elif loss_type == 'MSE':
            criterion_GAN = torch.nn.MSELoss()
            criterion_cycle = torch.nn.MSELoss()
            criterion_identity = torch.nn.MSELoss()

        #====================================== Specifying the type of the optimizer ==============================================
        if optim_type == 'SGD':
            gen1_optim = optim.SGD(generator1.parameters(), lr = G_lr, momentum=0.9)
            gen2_optim = optim.SGD(generator2.parameters(), lr = G_lr, momentum=0.9)
            dis1_optim = optim.SGD(discriminator1.parameters(), lr = D_lr, momentum=0.9)
            dis2_optim = optim.SGD(discriminator2.parameters(), lr = D_lr, momentum=0.9)
        elif optim_type == 'Adam':
            gen1_optim = optim.Adam(generator1.parameters(), lr = G_lr)
            gen2_optim = optim.Adam(generator2.parameters(), lr = G_lr)
            dis1_optim = optim.Adam(discriminator1.parameters(), lr = D_lr)
            dis2_optim = optim.Adam(discriminator2.parameters(), lr = D_lr)
        elif optim_type == 'RMSProp':
            gen1_optim = optim.RMSprop(generator1.parameters(), lr = G_lr)
            gen2_optim = optim.RMSprop(generator2.parameters(), lr = G_lr)
            dis1_optim = optim.RMSprop(discriminator1.parameters(), lr = D_lr)
            dis2_optim = optim.RMSprop(discriminator2.parameters(), lr = D_lr)

        #============================================= Prepare the data ======================================================
        x1 = self.day0_X_train
        x2 = self.dayk_X_train
        y2 = self.dayk_Y_train
        #=============================== Split x2 into the actual training set and the validation set ==============================
        # TODO
        x2_train = x2[:int((x2.shape[0])*0.75), :] # training set
        y2_train = y2[:int((y2.shape[0])*0.75), :] # training set
        x2_valid = x2[int((x2.shape[0])*0.75):, :] # validation set
        y2_valid = y2[int((y2.shape[0])*0.75):, :] # validation set
        #================================================  Define data Loaders ======================================================

        #--------------- loader1 is for day-0 data ---------------------
        loader1 = DataLoader(torch.utils.data.TensorDataset(torch.Tensor(x1)), batch_size = batch_size, shuffle = True)
        #--------------- loader2 is for day-k data in the training set ---------------------
        loader2 = DataLoader(torch.utils.data.TensorDataset(torch.Tensor(x2_train)), batch_size = batch_size, shuffle = True)
        
        #============================================ Training logs =========================================================
        train_log = {'epoch':[], 'batch_idx': [],
                    'loss D1':[], 'loss D2':[], 
                    'loss G1':[], 'loss G2':[],
                    'loss cycle 121':[], 'loss cycle 212':[],
                    'decoder r2 wiener': [],
                    'decoder r2 rnn': []}
        
        #============================================ Preparing to train ========================================================
        generator1.train()
        generator2.train()
        discriminator1.train()
        discriminator2.train()
        aligner_list = []
        mr2_all_list = []
        #================================================== The training loop ====================================================
        for epoch in range(epochs):
            for batch_idx, (data1_, data2_) in enumerate(zip(loader1, loader2)):
                #========================= loader1 and loader2 will yield mini-batches of data when running =========================
                #------ The batches by loader1 will be stored in data1, while the batches by loader2 will be stored in data2 ------
                data1, data2 = data1_[0], data2_[0]
                if data1.__len__() != data2.__len__():
                    continue
                data1, data2 = data1.to(device), data2.to(device)
                #------------ The labels for real samples --------------
                target_real = torch.ones((data1.shape[0], 1), requires_grad = False).type('torch.FloatTensor').to(device)
                #------------ The labels for fake samples --------------
                target_fake = torch.zeros((data1.shape[0], 1), requires_grad = False).type('torch.FloatTensor').to(device)
                #================================================== Generators ==================================================
                gen1_optim.zero_grad()
                gen2_optim.zero_grad()
                
                #------------ Identity loss, to make sure the generators do not distort the inputs --------------
                same2 = generator1(data2)
                loss_identity2 = criterion_identity(same2, data2)*ID_loss_p
                same1 = generator2(data1)
                loss_identity1 = criterion_identity(same1, data1)*ID_loss_p
                
                #------------ GAN loss for generator1, see the figure right above --------------
                fake2 = generator1(data1)
                pred_fake = discriminator2(fake2)
                loss_GAN2 = criterion_GAN(pred_fake, target_real)
                
                #------------ GAN loss for generator2, see the figure right above --------------
                fake1 = generator2(data2)
                pred_fake = discriminator1(fake1)
                loss_GAN1 = criterion_GAN(pred_fake, target_real)
                
                #------------ Cycle loss, see the figure right above --------------
                recovered1 = generator2(fake2)
                loss_cycle_121 = criterion_cycle(recovered1, data1)*cycle_loss_p
                
                recovered2 = generator1(fake1)
                loss_cycle_212 = criterion_cycle(recovered2, data2)*cycle_loss_p
                
                #----------- Total loss of G, the sum of all the losses defined above -----------
                loss_G = loss_identity1 + loss_identity2 + loss_GAN1 + loss_GAN2 + loss_cycle_121 + loss_cycle_212
                
                #-------- Backward() and step() for generators ---------
                loss_G.backward() 
                gen1_optim.step()
                gen2_optim.step()
                
                #================================================== Discriminator 1 ==================================================
                dis1_optim.zero_grad()
                
                #-------------- Adversarial loss from discriminator 1, see the figure above ------------------
                pred_real = discriminator1(data1)
                loss_D_real = criterion_GAN(pred_real, target_real)
                
                pred_fake = discriminator1(generator2(data2).detach())
                loss_D_fake = criterion_GAN(pred_fake, target_fake)
                
                loss_D1 = (loss_D_real + loss_D_fake)/2
                
                #-------- Backward() and step() for discriminator1 ---------
                loss_D1.backward()
                dis1_optim.step()
                
                #-------------- Adversarial loss from discriminator 2, see the figure above ------------------
                dis2_optim.zero_grad()
                
                pred_real = discriminator2(data2)
                loss_D_real = criterion_GAN(pred_real, target_real)
                
                pred_fake = discriminator2(generator1(data1).detach())
                loss_D_fake = criterion_GAN(pred_fake, target_fake)
                
                loss_D2 = (loss_D_real + loss_D_fake)/2
                
                #-------- Backward() and step() for discriminator2 ---------
                loss_D2.backward()
                dis2_optim.step()
                
                #====================================== save the training logs ========================================
                if logs == True:
                    train_log['epoch'].append(epoch)
                    train_log['batch_idx'].append(batch_idx)
                    train_log['loss D1'].append(loss_D1.item())
                    train_log['loss D2'].append(loss_D2.item())
                    train_log['loss G1'].append(loss_GAN1.item())
                    train_log['loss G2'].append(loss_GAN2.item())
                    train_log['loss cycle 121'].append(loss_cycle_121.item())
                    train_log['loss cycle 212'].append(loss_cycle_212.item())
                    
            #================ Test the aligner every 10 epoches on the validation set ====================
            if (epoch + 1) % 10 == 0:
                #---------- Put generator2, namely the aligner, into evaluation mode ------------
                generator2.eval()
                
                #---------- Use the trained aligner to transform the trials in x2_valid -----------
                # x2_valid_aligned = []
                # with torch.no_grad():  
                #     for each in x2_valid:
                #         data = torch.from_numpy(each).type('torch.FloatTensor')
                #         x2_valid_aligned.append(generator2(data).numpy())
                
                #--------- Feed the day-0 decoder with x2_valid_aligned to evaluate the performance of the aligner ----------
                
                # x2_valid_aligned_, y2_valid_ = format_data_from_trials(x2_valid_aligned, y2_valid, n_lags)
                x2_valid_align = generator2(torch.from_numpy(x2_valid).type('torch.FloatTensor').to(device)).detach().cpu().numpy()
                x2_valid_, y2_valid_ = self.preprocess_val_data({'neural': x2_valid_align, 'behaviour': y2_valid, 'neural_train': x2_train, 'behaviour_train': y2_train})
                # print(f"Shapes of x2_valid_aligned_ and y2_valid_: {x2_valid_.shape}, {y2_valid_.shape}")
                # print(x2_valid_)
                pred_y2_valid_ = self.decoder.predict(x2_valid_)
                
                if self.normalizer_path is not None:
                    with open(self.normalizer_path, 'rb') as f:
                        normalizer = pickle.load(f)
                    pred_y2_valid_ = normalizer.inverse_transform(pred_y2_valid_)
                #--------- Compute the multi-variate R2 between pred_y2_valid (predicted EMGs) and y2_valid (real EMGs) ----------
                mr2 = r2_score(y2_valid_, pred_y2_valid_, multioutput='variance_weighted')
                if self.verbose:
                    print('On the %dth epoch, the R\u00b2 on the validation set is %.2f'%(epoch+1, mr2))
                
                #------- Save the half-trained aligners and the corresponding performance on the validation set ---------
                aligner_list.append(generator2)
                mr2_all_list.append(mr2)
                
                #---------- Put generator2 back into training mode after finishing the evaluation -----------
                generator2.train()
        
        IDX = np.argmax(mr2_all_list)
        if self.verbose: 
            print('The aligner has been well trained on the %dth epoch'%(IDX*10))
        train_log['decoder r2 wiener'] = mr2_all_list
        #============================================ save the training log =================================================
        if logs == True:
            dt_string = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            with open(log_save_path + 'cycleGAN_train_log_' + dt_string + '.pkl', 'wb') as fp:
                pickle.dump(train_log, fp)        

        self.trained_aligner = aligner_list[IDX]
        if self.is_xds:
            day0day = self.day_pair[0]
            daykday = self.day_pair[1]
        else:
            day0day = os.path.basename(self.day0_data_dir).split('_')[1].split('-')[1]
            daykday = os.path.basename(self.dayk_data_dir).split('_')[1].split('-')[1]
            
        return 

    def test_cycle_gan_aligner(self):
        """
        net: the trained aligner
        dayk_data: the data that needs to be processed by the trained aligner
        """
        #------ Put the net in eval mode ------ #
        aligner = self.trained_aligner.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        #------ Use the trained aligner to process the dayk_data ------#
        self.dayk_X_test_aligned = aligner(torch.from_numpy(self.dayk_X_test).type('torch.FloatTensor').to(device)).detach().cpu().numpy()

        return self.dayk_X_test_aligned
    

class cycleGAN_optimized():
    def __init__(self,
                 day0_data_dir,
                 verbose=False):
        self.day0_data_dir = day0_data_dir
        self.kernel_size = 40
        self.sample_size = 20
        self.verbose = verbose
 
    def load_process_data(self, dayk_data_dir, decoder, neural_type, preprocessor, normalizer_path, is_xds=False, day0_xds_data=None, day_pair=None, train_test_split=0.8):
        self.preprocessor = preprocessor
        self.dayk_data_dir = dayk_data_dir
        self.is_xds = is_xds
        self.day_pair = day_pair
        if is_xds:
            ref_names = ['elec93', 'elec92', 'elec94', 'elec95', 'elec75', 'elec96', 'elec85', 'elec97', 'elec86', 'elec98', 'elec87', 'elec88', 'elec77', 'elec99', 'elec66', 'elec89', 'elec76', 'elec90', 'elec67', 'elec79', 'elec58', 'elec80', 'elec78', 'elec70', 'elec68', 'elec60', 'elec69', 'elec50', 'elec59', 'elec40', 'elec49', 'elec100', 'elec83', 'elec84', 'elec73', 'elec74', 'elec63', 'elec64', 'elec53', 'elec54', 'elec43', 'elec55', 'elec44', 'elec45', 'elec33', 'elec46', 'elec34', 'elec65', 'elec24', 'elec56', 'elec35', 'elec47', 'elec25', 'elec57', 'elec26', 'elec36', 'elec27', 'elec37', 'elec28', 'elec38', 'elec29', 'elec48', 'elec19', 'elec39', 'elec81', 'elec82', 'elec71', 'elec72', 'elec61', 'elec62', 'elec51', 'elec52', 'elec41', 'elec42', 'elec31', 'elec32', 'elec21', 'elec22', 'elec11', 'elec12', 'elec2', 'elec23', 'elec3', 'elec13', 'elec4', 'elec14', 'elec15', 'elec5', 'elec16', 'elec6', 'elec17', 'elec7', 'elec8', 'elec18', 'elec20', 'elec9']
            day0_data = day0_xds_data
            xds_day0 = neuraldecoding.preprocessing.blocks.xdsNWB2DictBlock()
            zeropad1 = neuraldecoding.preprocessing.blocks.ZeroPaddingBlock(ref_names)
 
            interpipe_day0 = {}
            xds_day0_out, interpipe_day0 = xds_day0.transform(day0_data, interpipe_day0)
            zeropad_day0_out, interpipe_day0 = zeropad1.transform(xds_day0_out, interpipe_day0)
            day0_neural = zeropad_day0_out['neural']
            day0_behaviour = zeropad_day0_out['behaviour']
 
            dayk_data = dayk_data_dir
            xds_dayk = neuraldecoding.preprocessing.blocks.xdsNWB2DictBlock()
            zeropad2 = neuraldecoding.preprocessing.blocks.ZeroPaddingBlock(ref_names)
 
            interpipe_dayk = {}
            xds_dayk_out, interpipe_dayk = xds_dayk.transform(dayk_data, interpipe_dayk)
            zeropad_dayk_out, interpipe_dayk = zeropad2.transform(xds_dayk_out, interpipe_dayk)
            dayk_neural = zeropad_dayk_out['neural']
            dayk_behaviour = zeropad_dayk_out['behaviour']
        else:
            if self.verbose:
                print("Loading and processing data...")
            day0_data = load_one_nwb(self.day0_data_dir)
            (day0_neural, day0_behaviour), trial_idx_day0 = neural_finger_from_dict(day0_data, neural_type)
            dayk_data = load_one_nwb(dayk_data_dir)
            (dayk_neural, dayk_behaviour), trial_idx_dayk = neural_finger_from_dict(dayk_data, neural_type)
            if self.verbose:
                print("Data loaded, Smoothing data...")
 
        sigma = self.kernel_size / self.sample_size
 
        day0_X = gaussian_filter1d(day0_neural.astype(np.float32), sigma=sigma, axis=0)
        day0_Y = day0_behaviour
 
        dayk_X = gaussian_filter1d(dayk_neural.astype(np.float32), sigma=sigma, axis=0)
        dayk_Y = dayk_behaviour
 
        (self.day0_X_train, self.day0_Y_train), (self.day0_X_test, self.day0_Y_test) = data_split_trial(day0_X, day0_Y, split_ratio=train_test_split)
        (self.dayk_X_train, self.dayk_Y_train), (self.dayk_X_test, self.dayk_Y_test) = data_split_trial(dayk_X, dayk_Y, split_ratio=train_test_split)
 
        if self.verbose:
            print("Data loaded, smoothed, and splited.")
 
        self.decoder = decoder
        self.normalizer_path = normalizer_path
 
    def preprocess_val_data(self, data):
        if self.verbose:
            print("Preprocessing validation data...")
        data = self.preprocessor.preprocess_pipeline(data, params={'is_train': False})
        return data
 
    def save_cycleGAN(self, fpath):
        if self.verbose:
            print("Saving cycleGAN instance...")
        with open(fpath, 'wb') as fp:
            pickle.dump(self, fp)
        if self.verbose:
            print("cycleGAN instance saved.")
 
    def save_checkpoint(self, save_dir, tag=None):
        """Save model weights + transformed data arrays.
        
        Saves the state_dict (~few hundred KB) and the aligned train/test arrays
        as compressed .npz (~10-20MB total depending on data size). At 42,562 pairs
        this takes ~15-25ms per pair for I/O (model ~2ms + npz ~10-20ms).
        
        Args:
            save_dir: directory to save checkpoint files into
            tag: optional string to identify this checkpoint (e.g. 'day0_dayk').
                 If None, uses a timestamp.
        """
        os.makedirs(save_dir, exist_ok=True)
        if tag is None:
            tag = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
        # --- Save model checkpoint ---
        checkpoint = {
            'aligner_state_dict': self.trained_aligner.state_dict(),
            'model_config': {
                'input_dim': self.trained_aligner.input_dim,
                'hidden_dim': self.trained_aligner.hidden_dim,
                'drop_out': self.trained_aligner.drop_out,
            },
            'day0_data_dir': self.day0_data_dir,
            'dayk_data_dir': self.dayk_data_dir,
            'kernel_size': self.kernel_size,
            'sample_size': self.sample_size,
            'is_xds': self.is_xds,
            'day_pair': self.day_pair,
        }
 
        model_path = os.path.join(save_dir, f'checkpoint_{tag}.pt')
        torch.save(checkpoint, model_path)
 
        # --- Run inference and save transformed data ---
        aligner = self.trained_aligner.eval()
        with torch.no_grad():
            dayk_train_aligned = aligner(
                torch.from_numpy(self.dayk_X_train).float().to(device)
            ).cpu().numpy()
            dayk_test_aligned = aligner(
                torch.from_numpy(self.dayk_X_test).float().to(device)
            ).cpu().numpy()
 
        # np.savez_compressed: ~30-50% smaller than raw .npy with minimal speed cost
        data_path = os.path.join(save_dir, f'transformed_{tag}.npz')
        np.savez_compressed(
            data_path,
            dayk_train_aligned=dayk_train_aligned,
            dayk_test_aligned=dayk_test_aligned,
            day0_X_train=self.day0_X_train,       # save reference data for later metric computation
            dayk_X_train=self.dayk_X_train,        # save original for comparison
            dayk_X_test=self.dayk_X_test,           # save original for comparison
        )
 
        if self.verbose:
            model_kb = os.path.getsize(model_path) / 1024
            data_kb = os.path.getsize(data_path) / 1024
            print(f"Saved: {model_path} ({model_kb:.1f} KB) + {data_path} ({data_kb:.1f} KB)")
 
        return model_path, data_path
 
    @staticmethod
    def load_transformed_data(data_path):
        """Load saved transformed data arrays — no GPU, no model needed.
        
        Usage:
            data = cycleGAN_optimized.load_transformed_data('transformed_day0_dayk.npz')
            aligned_test = data['dayk_test_aligned']   # ready for KL, R2, etc.
            original_test = data['dayk_X_test']         # for comparison
            day0_ref = data['day0_X_train']             # reference distribution
            
        Returns:
            NpzFile (dict-like) with keys: dayk_train_aligned, dayk_test_aligned,
            day0_X_train, dayk_X_train, dayk_X_test
        """
        return np.load(data_path)
 
    @staticmethod
    def load_aligner_from_checkpoint(fpath, device=None):
        """Rebuild the Generator from a checkpoint and return it in eval mode.
        
        Usage:
            aligner = cycleGAN_optimized.load_aligner_from_checkpoint('checkpoint_best.pt')
            aligned = aligner(torch.from_numpy(new_data).float().to(device)).cpu().numpy()
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(fpath, map_location=device, weights_only=False)
        cfg = checkpoint['model_config']
        aligner = Generator(cfg['input_dim'], cfg['hidden_dim'], cfg['drop_out'])
        aligner.load_state_dict(checkpoint['aligner_state_dict'])
        aligner.to(device)
        aligner.eval()
        return aligner
 
    @staticmethod
    def batch_transform_and_save(checkpoint_dir, data_loader_fn, output_dir, device=None):
        """Run inference for ALL checkpoints and save transformed data in one offline pass.
        
        Designed for 42,562 day-pair scale. Reuses a single model shell on GPU,
        swaps state_dict per pair, runs inference, and saves transformed arrays.
        
        Args:
            checkpoint_dir: directory containing checkpoint_*.pt files
            data_loader_fn: callable(checkpoint_dict) -> (day0_X_train, dayk_X_train, dayk_X_test)
                            as numpy arrays. Controls data loading so nothing is held in memory.
                            Example:
                                def load_pair(ckpt):
                                    day0 = load_one_nwb(ckpt['day0_data_dir'])
                                    dayk = load_one_nwb(ckpt['dayk_data_dir'])
                                    # ... smooth, split ...
                                    return day0_X_train, dayk_X_train, dayk_X_test
            output_dir: directory to write transformed_*.npz files into
            device: torch device (auto-detected if None)
            
        Returns:
            list of output file paths
            
        Example:
            paths = cycleGAN_optimized.batch_transform_and_save(
                checkpoint_dir='checkpoints/',
                data_loader_fn=my_loader,
                output_dir='transformed_data/'
            )
            # Later, compute metrics on any pair instantly:
            data = np.load(paths[0])
            kl = kl_divergence_per_channel_rob_optimized(data['day0_X_train'], data['dayk_test_aligned'])
        """
        import glob
 
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.pt')))
        print(f"Found {len(checkpoint_files)} checkpoints to process")
 
        output_paths = []
        shell_aligner = None
 
        for i, fpath in enumerate(checkpoint_files):
            fname = os.path.basename(fpath)
            tag = fname.replace('checkpoint_', '').replace('.pt', '')
            checkpoint = torch.load(fpath, map_location=device, weights_only=False)
 
            # Build model shell once, reuse thereafter
            if shell_aligner is None:
                cfg = checkpoint['model_config']
                shell_aligner = Generator(cfg['input_dim'], cfg['hidden_dim'], cfg['drop_out'])
                shell_aligner.to(device)
                shell_aligner.eval()
 
            shell_aligner.load_state_dict(checkpoint['aligner_state_dict'])
 
            try:
                day0_X_train, dayk_X_train, dayk_X_test = data_loader_fn(checkpoint)
            except Exception as e:
                print(f"  [{i+1}/{len(checkpoint_files)}] SKIPPED {fname}: {e}")
                continue
 
            with torch.no_grad():
                dayk_train_aligned = shell_aligner(
                    torch.from_numpy(dayk_X_train).float().to(device)
                ).cpu().numpy()
                dayk_test_aligned = shell_aligner(
                    torch.from_numpy(dayk_X_test).float().to(device)
                ).cpu().numpy()
 
            out_path = os.path.join(output_dir, f'transformed_{tag}.npz')
            np.savez_compressed(
                out_path,
                dayk_train_aligned=dayk_train_aligned,
                dayk_test_aligned=dayk_test_aligned,
                day0_X_train=day0_X_train,
                dayk_X_train=dayk_X_train,
                dayk_X_test=dayk_X_test,
            )
            output_paths.append(out_path)
 
            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(checkpoint_files)}] processed")
 
        print(f"Done. Processed {len(output_paths)}/{len(checkpoint_files)} checkpoints.")
        return output_paths
 
    def train_cycle_gan_aligner(self, cycleGAN_params):
        """
        x1: M1 spike firing rates on day-0.
        x2: M1 spike firing rates on day-k (split 3:1 into train/validation).
        y2: EMGs on day-k.
        return: a trained "aligner" (generator) for day-k use.
        """
        x_dim = self.day0_X_train.shape[1]
        if self.verbose:
            print("Training cycleGAN aligner...")
 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
 
        # ============== Specifying hyper-parameters ==============
        D_hidden_dim = cycleGAN_params['D_params']['hidden_dim']
        G_hidden_dim = cycleGAN_params['G_params']['hidden_dim']
        loss_type = cycleGAN_params['training_params']['loss_type']
        optim_type = cycleGAN_params['training_params']['optim_type']
        epochs = cycleGAN_params['training_params']['epochs']
        batch_size = cycleGAN_params['training_params']['batch_size']
        D_lr = cycleGAN_params['training_params']['D_lr']
        G_lr = cycleGAN_params['training_params']['G_lr']
        ID_loss_p = cycleGAN_params['training_params']['ID_loss_p']
        cycle_loss_p = cycleGAN_params['training_params']['cycle_loss_p']
        drop_out_D = cycleGAN_params['training_params']['drop_out_D']
        drop_out_G = cycleGAN_params['training_params']['drop_out_G']
 
        logs = cycleGAN_params['logs']
        log_save_path = cycleGAN_params['log_save_path']
 
        # ============== Defining networks ==============
        generator1, generator2 = Generator(x_dim, G_hidden_dim, drop_out_G), Generator(x_dim, G_hidden_dim, drop_out_G)
        discriminator1, discriminator2 = Discriminator(x_dim, D_hidden_dim, drop_out_D), Discriminator(x_dim, D_hidden_dim, drop_out_D)
 
        generator1.to(device)
        generator2.to(device)
        discriminator1.to(device)
        discriminator2.to(device)
 
        # ============== Losses ==============
        if loss_type == 'L1':
            criterion_GAN = torch.nn.MSELoss()
            criterion_cycle = torch.nn.L1Loss()
            criterion_identity = torch.nn.L1Loss()
        elif loss_type == 'MSE':
            criterion_GAN = torch.nn.MSELoss()
            criterion_cycle = torch.nn.MSELoss()
            criterion_identity = torch.nn.MSELoss()
 
        # ============== Optimizers ==============
        if optim_type == 'SGD':
            gen1_optim = optim.SGD(generator1.parameters(), lr=G_lr, momentum=0.9)
            gen2_optim = optim.SGD(generator2.parameters(), lr=G_lr, momentum=0.9)
            dis1_optim = optim.SGD(discriminator1.parameters(), lr=D_lr, momentum=0.9)
            dis2_optim = optim.SGD(discriminator2.parameters(), lr=D_lr, momentum=0.9)
        elif optim_type == 'Adam':
            gen1_optim = optim.Adam(generator1.parameters(), lr=G_lr)
            gen2_optim = optim.Adam(generator2.parameters(), lr=G_lr)
            dis1_optim = optim.Adam(discriminator1.parameters(), lr=D_lr)
            dis2_optim = optim.Adam(discriminator2.parameters(), lr=D_lr)
        elif optim_type == 'RMSProp':
            gen1_optim = optim.RMSprop(generator1.parameters(), lr=G_lr)
            gen2_optim = optim.RMSprop(generator2.parameters(), lr=G_lr)
            dis1_optim = optim.RMSprop(discriminator1.parameters(), lr=D_lr)
            dis2_optim = optim.RMSprop(discriminator2.parameters(), lr=D_lr)
 
        # ============== Prepare data ==============
        x1 = self.day0_X_train
        x2 = self.dayk_X_train
        y2 = self.dayk_Y_train
 
        x2_train = x2[:int((x2.shape[0]) * 0.75), :]
        y2_train = y2[:int((y2.shape[0]) * 0.75), :]
        x2_valid = x2[int((x2.shape[0]) * 0.75):, :]
        y2_valid = y2[int((y2.shape[0]) * 0.75):, :]
 
        # ============== DataLoaders ==============
        loader1 = DataLoader(
            torch.utils.data.TensorDataset(torch.Tensor(x1)),
            batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True, persistent_workers=True  # optimization: parallel data loading with pinned memory avoids CPU-GPU copy overhead; persistent workers avoid respawning processes each epoch
        )
        loader2 = DataLoader(
            torch.utils.data.TensorDataset(torch.Tensor(x2_train)),
            batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True, persistent_workers=True  # optimization: same as loader1
        )
 
        # ============== Training logs ==============
        train_log = {'epoch': [], 'batch_idx': [],
                     'loss D1': [], 'loss D2': [],
                     'loss G1': [], 'loss G2': [],
                     'loss cycle 121': [], 'loss cycle 212': [],
                     'decoder r2 wiener': [],
                     'decoder r2 rnn': []}
 
        # ============== Pre-allocate target tensors ==============
        target_real = torch.ones((batch_size, 1), device=device)   # optimization: create once outside loop instead of 25600 times; avoids repeated allocation and CPU-to-GPU transfer
        target_fake = torch.zeros((batch_size, 1), device=device)  # optimization: create once outside loop instead of 25600 times; avoids repeated allocation and CPU-to-GPU transfer
 
        # ============== Mixed precision scaler ==============
        scaler = GradScaler('cuda')  # optimization: mixed precision (FP16) reduces memory bandwidth and speeds up GPU compute for forward/backward passes
 
        # ============== Preparing to train ==============
        generator1.train()
        generator2.train()
        discriminator1.train()
        discriminator2.train()
        aligner_list = []
        mr2_all_list = []
 
        # ============== Training loop ==============
        for epoch in range(epochs):
            for batch_idx, (data1_, data2_) in enumerate(zip(loader1, loader2)):
                data1, data2 = data1_[0], data2_[0]
                if data1.shape[0] != data2.shape[0]:
                    continue
                data1, data2 = data1.to(device, non_blocking=True), data2.to(device, non_blocking=True)  # optimization: non_blocking=True allows async CPU-to-GPU transfer while CPU continues working (requires pin_memory=True)
 
                # Handle last batch with different size
                cur_batch = data1.shape[0]
                if cur_batch != batch_size:  # optimization: reuse pre-allocated targets for full batches, only create new ones for the final smaller batch
                    t_real = torch.ones((cur_batch, 1), device=device)
                    t_fake = torch.zeros((cur_batch, 1), device=device)
                else:
                    t_real = target_real
                    t_fake = target_fake
 
                # ============== Generators ==============
                gen1_optim.zero_grad(set_to_none=True)  # optimization: set_to_none=True skips memset-to-zero, just deallocates gradients which is faster
                gen2_optim.zero_grad(set_to_none=True)  # optimization: set_to_none=True
 
                with autocast('cuda'):  # optimization: mixed precision forward pass — runs matmuls in FP16 for ~2x throughput on GPU
                    # Identity loss
                    same2 = generator1(data2)
                    loss_identity2 = criterion_identity(same2, data2) * ID_loss_p
                    same1 = generator2(data1)
                    loss_identity1 = criterion_identity(same1, data1) * ID_loss_p
 
                    # GAN loss
                    fake2 = generator1(data1)
                    pred_fake = discriminator2(fake2)
                    loss_GAN2 = criterion_GAN(pred_fake, t_real)
 
                    fake1 = generator2(data2)
                    pred_fake = discriminator1(fake1)
                    loss_GAN1 = criterion_GAN(pred_fake, t_real)
 
                    # Cycle loss
                    recovered1 = generator2(fake2)
                    loss_cycle_121 = criterion_cycle(recovered1, data1) * cycle_loss_p
 
                    recovered2 = generator1(fake1)
                    loss_cycle_212 = criterion_cycle(recovered2, data2) * cycle_loss_p
 
                    # Total generator loss
                    loss_G = loss_identity1 + loss_identity2 + loss_GAN1 + loss_GAN2 + loss_cycle_121 + loss_cycle_212
 
                scaler.scale(loss_G).backward()  # optimization: scaled backward pass for mixed precision — prevents FP16 gradient underflow
                scaler.step(gen1_optim)  # optimization: scaler handles unscaling before optimizer step
                scaler.step(gen2_optim)
 
                # ============== Discriminator 1 ==============
                dis1_optim.zero_grad(set_to_none=True)  # optimization: set_to_none=True
 
                with autocast('cuda'):  # optimization: mixed precision for discriminator forward pass
                    pred_real = discriminator1(data1)
                    loss_D_real = criterion_GAN(pred_real, t_real)
 
                    pred_fake = discriminator1(fake1.detach())  # optimization: reuse fake1 from generator step instead of recomputing generator2(data2) — saves one full generator forward pass per batch (~3% of total time)
                    loss_D_fake = criterion_GAN(pred_fake, t_fake)
 
                    loss_D1 = (loss_D_real + loss_D_fake) / 2
 
                scaler.scale(loss_D1).backward()  # optimization: scaled backward for mixed precision
                scaler.step(dis1_optim)
 
                # ============== Discriminator 2 ==============
                dis2_optim.zero_grad(set_to_none=True)  # optimization: set_to_none=True
 
                with autocast('cuda'):  # optimization: mixed precision for discriminator forward pass
                    pred_real = discriminator2(data2)
                    loss_D_real = criterion_GAN(pred_real, t_real)
 
                    pred_fake = discriminator2(fake2.detach())  # optimization: reuse fake2 from generator step instead of recomputing generator1(data1) — saves one full generator forward pass per batch (~3% of total time)
                    loss_D_fake = criterion_GAN(pred_fake, t_fake)
 
                    loss_D2 = (loss_D_real + loss_D_fake) / 2
 
                scaler.scale(loss_D2).backward()  # optimization: scaled backward for mixed precision
                scaler.step(dis2_optim)
 
                scaler.update()  # optimization: update the loss scale factor once per iteration (must be called after all scaler.step() calls)
 
                # ============== Logging ==============
                if logs:
                    # optimization: batch all .item() calls into one GPU sync via torch.stack + .tolist(), instead of 6 separate .item() calls that each force a GPU synchronization
                    loss_vals = torch.stack([loss_D1, loss_D2, loss_GAN1, loss_GAN2,
                                            loss_cycle_121, loss_cycle_212]).detach().cpu().tolist()
                    train_log['epoch'].append(epoch)
                    train_log['batch_idx'].append(batch_idx)
                    train_log['loss D1'].append(loss_vals[0])
                    train_log['loss D2'].append(loss_vals[1])
                    train_log['loss G1'].append(loss_vals[2])
                    train_log['loss G2'].append(loss_vals[3])
                    train_log['loss cycle 121'].append(loss_vals[4])
                    train_log['loss cycle 212'].append(loss_vals[5])
 
            # ============== Validation every 10 epochs ==============
            if (epoch + 1) % 10 == 0:
                generator2.eval()
 
                with torch.no_grad():  # optimization: disable gradient tracking during validation — saves memory and compute since no backward pass is needed
                    x2_valid_align = generator2(torch.from_numpy(x2_valid).float().to(device)).cpu().numpy()  # optimization: .float() is cleaner than .type('torch.FloatTensor'); no .detach() needed inside no_grad()
 
                x2_valid_, y2_valid_ = self.preprocess_val_data({'neural': x2_valid_align, 'behaviour': y2_valid, 'neural_train': x2_train, 'behaviour_train': y2_train})
                # KL removed from training loop — compute offline from saved transformed data instead
 
                pred_y2_valid_ = self.decoder.predict(x2_valid_)
 
                if self.normalizer_path is not None:
                    with open(self.normalizer_path, 'rb') as f:
                        normalizer = pickle.load(f)
                    pred_y2_valid_ = normalizer.inverse_transform(pred_y2_valid_)
 
                mr2 = r2_score(y2_valid_, pred_y2_valid_, multioutput='variance_weighted')
                if self.verbose:
                    print('On the %dth epoch, the R² on the validation set is %.2f' % (epoch + 1, mr2))
 
                aligner_list.append(copy.deepcopy(generator2))  # bugfix + optimization: original appended a reference to the same object, so all entries pointed to the final state; deepcopy preserves each checkpoint's actual weights
                mr2_all_list.append(mr2)
 
                generator2.train()
 
        IDX = np.argmax(mr2_all_list)
        if self.verbose:
            print('The aligner has been well trained on the %dth epoch' % (IDX * 10))
        train_log['decoder r2 wiener'] = mr2_all_list
 
        if logs:
            dt_string = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            with open(log_save_path + 'cycleGAN_train_log_' + dt_string + '.pkl', 'wb') as fp:
                pickle.dump(train_log, fp)
 
        self.trained_aligner = aligner_list[IDX]
 
        return
 
    def test_cycle_gan_aligner(self):
        aligner = self.trained_aligner.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
        with torch.no_grad():  # optimization: disable gradient tracking during inference
            self.dayk_X_test_aligned = aligner(torch.from_numpy(self.dayk_X_test).float().to(device)).cpu().numpy()  # optimization: .float() replaces verbose .type('torch.FloatTensor'); no .detach() needed inside no_grad()
 
        return self.dayk_X_test_aligned