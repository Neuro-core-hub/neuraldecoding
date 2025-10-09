import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from neuraldecoding.utils.data_tools import load_one_nwb, neural_finger_from_dict
from neuraldecoding.utils import data_split_trial
from scipy.ndimage import gaussian_filter1d
import numpy as np
from sklearn.metrics import r2_score
from datetime import datetime
import pickle
import os

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
    
    def load_process_data(self, dayk_data_dir, decoder, neural_type, preprocessor, normalizer_path, train_test_split = 0.8):
        # preprocessor is the preprocessor for validation data split after first split to mimic the preproc after split
        self.preprocessor = preprocessor
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
                #------------ The labels for real samples --------------
                target_real = torch.ones((data1.shape[0], 1), requires_grad = False).type('torch.FloatTensor')
                #------------ The labels for fake samples --------------
                target_fake = torch.zeros((data1.shape[0], 1), requires_grad = False).type('torch.FloatTensor')

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
                x2_valid_align = generator2(torch.from_numpy(x2_valid).type('torch.FloatTensor')).detach().numpy()
                x2_valid_, y2_valid_ = self.preprocess_val_data({'neural': x2_valid_align, 'behaviour': y2_valid})
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
        return 

    def test_cycle_gan_aligner(self):
        """
        net: the trained aligner
        dayk_data: the data that needs to be processed by the trained aligner
        """
        #------ Put the net in eval mode ------ #
        aligner = self.trained_aligner.eval()
        
        #------ Use the trained aligner to process the dayk_data ------#
        self.dayk_X_test_aligned = aligner(torch.from_numpy(self.dayk_X_test).type('torch.FloatTensor')).detach().numpy()

        return self.dayk_X_test_aligned