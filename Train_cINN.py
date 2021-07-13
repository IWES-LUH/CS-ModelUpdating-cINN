from cINN import InvertibleNN as inn
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import sys
from datetime import datetime
import os
from Eval_Models import evalModel

case_name = 'Red_Inp_Full_Out_R6m_uni'
sample_file = 'unused/Samples_n_40000_R_6.00_red_Inp_uni_varMat_0.60_varLoc_0.150_RndSeed_5370.mat'
in_sel = 'R6_red_Inp'  # R6_red_Inp, R9_red_Inp, R12_red_Inp, R15_red_Inp, R6_full_Inp, R9_full_Inp, custom,
out_sel = 'ALL'  # ALL, KsMs, KsMs_old, custom
check_int = 100
COUT = True

# Set Hyper Parameter Set
hps = {
    # Experiment Params:
	'PCA_in': False,
	'PCA_cond': False,
    'clamp': 2.0,  # necessary input for Couplinblock GLOW
    'n_CouplingBlock': 11,  # number of coupling block
    'permute_soft': True,  # permutation soft of coupling block
    'subnet_layer': 1,  # number of dense hidden layer in affine coupling block subnet
    'subnet_dim': 285,  # perceptron numbers per hidden layer in affine coupling block subnet
    'condnet': False,  # condition net layer number
    'condnet_layer': 1,  # condition net layer number
    'condnet_dim': 100,  # output dimension of condition net
    'condnet_activation': nn.ReLU(),  # activation function for condition net
    'activation': nn.PReLU(),  # activation function of each hidden layer in affine coupling block subnet
    'drop_out': True,
    'drop_out_rate': 0.075,

    # Training Params:
    'optimizer': 'adagrad',  # optimizer
    'learning_rate': 0.25,  # learning rate optimizer
    'optim_eps': 1e-6,
    'weight_decay': 2e-5,
    'lr_scheduler': False,
    'lr_epoch_step': 500,
    'lr_factor': 0.75,
    'epochs': 4500,
    'batch_size': 64,  # batch size
    'test_split': 0.25,
    'grad_clip': 1}

# Prepare Data
if in_sel in ['R6_red_Inp']:
    in_sel = np.arange(18)
elif in_sel in ['R9_red_Inp', 'R12_red_Inp', 'R15_red_Inp']:
    in_sel = np.arange(16)
elif in_sel in ['R6_full_Inp']:
    in_sel = np.arange(29)
elif in_sel in ['R9_full_Inp']:
    in_sel = np.arange(26)
else:
    in_sel = np.arange(10)

if out_sel == 'ALL':
    out_sel = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
               18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
               30, 31, 32, 33, 34, 35]
elif out_sel == 'KsMs':
    out_sel = np.arange(23)  # realKsMs but works only with Samples labeling KsMs
elif out_sel == 'KsMs_old':
    out_sel = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
else:
    out_sel = []

now = datetime.now()
folder = './Current_Training/' + now.strftime('%Y-%m-%d-%H-%M_') + case_name + '/'

try:
    os.mkdir('./Current_Training/')
except:
    pass

try:
    os.mkdir(folder)
except:
    pass

fmat = './Samples/' + sample_file

if COUT:
    cout = sys.stdout = open(folder + 'TrainProgress.txt', 'w')

print('Selected mat file:')
print(sample_file)

print('Selected input idx:')
print(in_sel)

print('Selected output idx:')
print(out_sel)

# Load Data
mat = scipy.io.loadmat(fmat)
In_np = np.transpose(mat['In'])
Out_np = np.transpose(mat['Out'])

Ch_In = np.transpose(mat['Ch_In'])
Ch_Out = np.transpose(mat['Ch_Out'])

ch_i = [Ch_In[0][i][0] for i in range(Ch_In[0].size)]
ch_o = [Ch_Out[0][i][0] for i in range(Ch_Out[0].size)]

ch_in_sel = [ch_i[k] for k in in_sel]
in_idx = [True if k in ch_in_sel else False for k in ch_i]

ch_out_sel = [ch_o[k] for k in out_sel]
out_idx = [True if k in ch_out_sel else False for k in ch_o]

shuffling = np.random.permutation(In_np.shape[0])
In = torch.tensor(In_np[:, in_idx], dtype=torch.float)
Out = torch.tensor(Out_np[:, out_idx], dtype=torch.float)

In_test = In.clone()
model = inn(In, Out)
model.update_hps(hps=hps)
model.train_model(In, Out, verbose=True, tune_report=False, check_int=check_int, check_dir=folder,
                  save_file=folder + 'Model_trained.pt')
model.print_training(folder + 'Final_progress.png')

evalModel(folder=folder, modelfile='Model_trained.pt',
          sample_file=sample_file, in_sel=in_sel, out_sel=out_sel)

if COUT:
    cout.close()
