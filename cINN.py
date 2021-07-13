'''
BSD 3-Clause License

Copyright (c) 2021, Leibniz Universität Hannover, Institut für Windenergiesysteme
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

from time import time
import torch
import torch.nn as nn
import torch.optim
import numpy as np
import math
import os
from hyperopt import hp
from StandardScaler import StandardScaler as stdScaler
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
from sklearn.decomposition import PCA

from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

from FrEIA.framework import InputNode, OutputNode, ConditionNode, Node, GraphINN
from FrEIA.modules import AllInOneBlock


class InvertibleNN(GraphINN):
    """This class builds a conditional invertible neural network

    Based on the hyper parameter set hps, an object of this class creates a
    conditional invertible neural network. It is equipped with class methods
    to train the model (train_model) and for a trained model to evaluate the
    inverse path (inverse). In order to optimize the hyperparameters a tune
    method is implemented (hps_tune) based on ray.tune and hyperopt modules.
    Additional save (save_state) and load (load_state) methods are
    incorporated.

    Attributes
    ----------
    hps: dict
        a dictionary with all hyper parameters necessary for this model
    in_scaler: StandardScaler.StandardScaler
        an object to standardize the input and reverse the standardization
    out_scaler: StandardScaler.StandardScaler
        an object to standardize the conditions
    optimizer: torch.nn optimizer
        a torch.nn optimizer object used for the training process
    train_acc: dict
        contains for each epoch a dict with the training accuracies
    train_loss: dict
        contains for each epoch a dict with the training losses
    val_acc: dict
        contains for each epoch a dict with the validation accuracies
    val_loss: dict
        contains for each epoch a dict with the validation losses

    Methods
    -------
    update_hps(hps=None)
        updates the model hps attribute. a partial hps dict is also valid
    train_model(inputs, conditions, verbose=0, tune_report=False, init=True,
                optimizer=None, epoch_resume=0, check_int=None, 
                check_dir='./ChkPnt', save_file=None)
        trains the model. Checkpoint intervals can be defined.
    inverse(conditions, lat_sample_size=1000, mean_out=False,
            numpy_flag=True, standardized_out=False)
        calculates the inputs for given conditions.
    hps_tune(inputs, conditions, epochs=50, config=None, n_samples=10,
             cpus_per_trial=8, gpus_per_trial=2, max_n_epochs=None,
             search_algo=None)
        performes a hyper parameter tuning for a given search space (config)
    save_state(fname, epoch=None)
        saves the model to the file fname
    load_state(fname)
        loads the model from the file fname
    load_state(fname)
        static method to create a model from the file fname
    resume_training(fname, inputs, conditions, verbose=0, epochs=2,
                    learning_rate=None, check_int=None,
                    check_dir='./ChkPnt', save_file=None)
        method to resume training by x epochs
    default_hps()
        a static method, that returns the default hyper parameter set
    """

    def __init__(self, inputs=None, conditions=None, in_scaler=None, cond_scaler=None, **kwargs):

        # Set given hyper parameters oder generate default values
        if 'hps' in kwargs:
            self.hps = kwargs.get("hps")
        else:
            self.hps = self.default_hps()

        if self.hps['batch_norm']:
            self.hps['drop_out'] = False

        # Save in- and output scalers
        if not in_scaler:
            self.in_scaler = stdScaler()
        else:
            self.in_scaler = in_scaler

        if not cond_scaler:
            self.cond_scaler = stdScaler()
        else:
            self.cond_scaler = cond_scaler

        # save condition PCA
        if self.hps['PCA_cond']:
            self.cond_pca = PCA()
        else:
            self.cond_pca = None

        if self.hps['PCA_in']:
            self.in_pca = PCA()
        else:
            self.in_pca = None

        # Define input and condition dimensions
        if not self.hps['feat_in']:
            if torch.is_tensor(inputs) or isinstance(inputs, np.ndarray):
                self.hps['feat_in'] = inputs.shape[-1]  # number of input features
            else:
                raise (
                    'Please introduce either a dummy input with correct feature number or pass an hps with key ''feat_in'' ')

        if not self.hps['feat_cond']:
            if torch.is_tensor(conditions) or isinstance(conditions, np.ndarray):
                self.hps['feat_cond'] = conditions.shape[-1]  # number of conditional features
            else:
                raise (
                    'Please introduce either a dummy output with correct feature number or pass an hps with key ''feat_cond'' ')

        # generate Model Nodes
        # Input node
        nodes = [InputNode(self.hps['feat_in'], name='input')]

        if self.hps['condnet']:
            cond = ConditionNode(self.hps['condnet_dim'], name='condition')
        else:
            cond = ConditionNode(self.hps['feat_cond'], name='condition')

        # Generate the argument dict for the selected CouplingBlock
        dict_CB = {'subnet_constructor': self.__genSubnet, 'affine_clamping': self.hps['clamp'],
                   'permute_soft': self.hps['permute_soft']}

        # Chain demanded number of coupling blocks
        for i in range(self.hps['n_CouplingBlock']):
            nodes.append(Node(nodes[-1],
                              AllInOneBlock,
                              dict_CB,
                              conditions=cond,
                              name=F'coupling_{i}'))

        nodes.append(OutputNode([nodes[-1].out0], name='Output'))

        if 'verbose' in kwargs:
            verb = kwargs['verbose']
            if type(verb) is not bool:
                verb = False
        else:
            verb = False

        # Build network
        super().__init__(nodes + [cond], verbose=verb)

        self.__init_losses()

        if self.hps['condnet']:
            self.condnet = self.__genCondnet()
        else:
            self.condnet = []

        # Move to preselected device
        self.to(self.hps['device'])

        # Collect all trainable parameters
        self.trainable_parameters = [p for p in self.parameters() if p.requires_grad]

        if self.hps['condnet']:
            self.trainable_parameters += list(self.condnet.parameters())

        self.optimizer = self.getOptimizer()

    def update_hps(self, hps=None):
        """ method to update the hyper parameter set"""
        if hps:
            for k, v in hps.items():
                self.hps[k] = v

        self.__init__(self, hps=self.hps)

    def train_model(self, inputs, conditions, verbose=True, tune_report=False,
                    init=True, optimizer=None, epoch_resume=0,
                    check_int=None, check_dir='./ChkPnt', save_file=None):
        """training method of the model"""

        self.__init_losses(epoch_resume=epoch_resume)

        # Initialize trainable parameters
        if init and epoch_resume == 0:
            for param in [p for p in self.parameters() if p.requires_grad]:
                param.data = 0.025 * torch.randn_like(param)

        # Get Data Loaders
        train_loader, val_loader = self.__getDataLoad(inputs, conditions, setscaler=True)

        # Get Optimizer
        if optimizer:
            self.optimizer = optimizer

        # Initialize Output
        t_start = time()
        if verbose:
            print('\n\n The Models hyperparameter set is:')
            for key, value in self.hps.items():
                print(key, ' : ', value)

            print('\n\n| Epoch:    |  Time:  | Loss_maxL_train: |  | Acc_i_val: | L_maxL_val: |')
            print('|-----------|---------|------------------|  |------------|-------------|')

        # Initialize learining rate scheduler
        if self.hps['lr_scheduler']:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hps['lr_epoch_step'],
                                                           gamma=self.hps['lr_factor'])

        # Start Training
        for i_epoch in range(epoch_resume, self.hps['epochs']):
            t_epoch = time()

            self.__train_process(train_loader, i_epoch=i_epoch)

            self.__val_process(val_loader, i_epoch=i_epoch)

            # Output epoch step
            if verbose:
                print('| %4d/%4d | %6ds | %16.5f |  | %10.6f | %11.5f |' % (
                    i_epoch + 1, self.hps['epochs'], min((time() - t_epoch), 99999),
                    min(self.train_loss[i_epoch], 9999),
                    min(np.mean(self.val_acc[i_epoch], axis=0), 1.),
                    min(self.val_loss[i_epoch], 9999)))

            # Step learining rate scheduler
            if self.hps['lr_scheduler']:
                lr_scheduler.step()

            # Report to hps-tuning
            if tune_report:
                if math.isnan(self.train_loss[i_epoch]):
                    done_flag = True
                else:
                    done_flag = False

                tune.report(done=done_flag,
                            L_maxL_train=self.train_loss[i_epoch],
                            Acc_in_val=np.mean(self.val_acc[i_epoch], axis=0),
                            L_maxL_val=self.val_loss[i_epoch])

            # Save Checkpoint
            if check_int and (i_epoch + 1) % check_int == 0:
                try:
                    os.rmdir(check_dir)
                except:
                    pass

                try:
                    os.mkdir(check_dir)
                except:
                    pass

                fname = check_dir + '/ChkPnt_Ep_' + str(i_epoch + 1) + '.pt'
                self.save_state(fname, epoch=i_epoch)
                fname = check_dir + '/TrainingProgress_Ep_' + str(i_epoch + 1) + '.png'
                self.print_training(fname)

        if verbose:
            print('|-----------|---------|------------------|  |------------|-------------|')
            print(f"\n\nTraining took {(time() - t_start) / 60:.2f} minutes\n")

        if save_file:
            self.save_state(save_file)

    def inverse(self, conditions, lat_sample_size=1000, mean_out=False, numpy_flag=True, standardized_out=False):
        """ Method to inverse the model """

        self.eval()

        # check data type of conditions
        if isinstance(conditions, np.ndarray):
            if self.hps['PCA_cond']:
                conditions = self.cond_pca.transform(conditions)
            conditions = torch.tensor(conditions, dtype=torch.float, device=self.hps['device'])
        elif torch.is_tensor(conditions):
            if self.hps['PCA_cond']:
                conditions = conditions.cpu().detach().numpy()
                conditions = self.cond_pca.transform(conditions)
                conditions = torch.tensor(conditions, dtype=torch.float, device=self.hps['device'])
            else:
                conditions = conditions.clone()
                conditions = conditions.to(device=self.hps['device'])
        else:
            raise ('The input data needs to be type of torch.tensor or numpy.ndarray')

        # expand dimension if only a single dimension exists
        if conditions.dim() == 1:
            conditions = torch.unsqueeze(conditions, 0)

        # check dimensions:
        if conditions.shape[1] != self.hps['feat_cond']:
            raise ('The input data dimension does not match the models output size of %d features.' % self.hps[
                'feat_cond'])

        n = conditions.shape[0]

        rnd_lat = torch.randn(lat_sample_size * n, self.hps['feat_in'],
                              dtype=torch.float,
                              device=self.hps['device'])

        # scale conditions
        conditions = self.cond_scaler.transform(conditions, set_par=False)
        # and expand to the latent sampling size
        conditions = torch.repeat_interleave(conditions, lat_sample_size, dim=0)

        with torch.no_grad():
            if self.hps['condnet']:
                inputs, jac = self.forward(rnd_lat, c=self.condnet(conditions), rev=True, jac=False)
            else:
                inputs, jac = self.forward(rnd_lat, c=conditions, rev=True, jac=False)

            inputs = torch.reshape(inputs, (n, lat_sample_size, self.hps['feat_in']))
            if not standardized_out:
                inputs = self.in_scaler.inverse_transform(inputs)
                if self.hps['PCA_in']:
                    inputs = self.in_pca.inverse_transform(inputs)
                    inputs = torch.tensor(inputs, dtype=torch.float, device=self.hps['device'])
        if mean_out:
            inputs = torch.mean(inputs, 1)
            if numpy_flag:
                return inputs.cpu().detach().numpy()
            else:
                return inputs
        else:
            if numpy_flag:
                inputs = inputs.cpu().detach().numpy()
            return [inputs[i, :, :] for i in range(n)]

    @staticmethod
    def hps_tune(inputs, conditions, epochs=50, config=None, n_samples=10, cpus_per_trial=8, gpus_per_trial=2,
                 max_n_epochs=None, search_algo=None):

        if not max_n_epochs or max_n_epochs > epochs:
            scheduler = ASHAScheduler(metric="L_maxL_train", mode="min")
        else:
            scheduler = ASHAScheduler(metric="L_maxL_train", mode="min", grace_period=max_n_epochs)

        if not config:
            config = {
                'n_CouplingBlock': hp.quniform('n_CouplingBlock', 4, 60, 4),
                'subnet_layer': hp.quniform('subnet_layer', 1, 10, 1),
                'subnet_dim': hp.quniform('subnet_dim', 20, 200, 5),
                'activation': hp.choice('activation',
                                        [nn.ReLU(), nn.LeakyReLU(), nn.PReLU(), nn.Tanhshrink(), nn.Tanh(), nn.CELU()]),
                'optimizer': hp.choice('optimizer', ['adamax', 'adadelta', 'adagrad', 'adamw', 'adam']),
                'learning_rate': hp.quniform('learning_rate', 0.01, 0.05, 0.005),
                'batch_size': hp.choice('batch_size', [128, 256, 512])
            }

        def train_tuning(config):

            # Get Hyperparameters
            hps = InvertibleNN.default_hps()

            # Defines how to cast values default ist float
            integers = ['n_CouplingBlock', 'subnet_layer',
                        'subnet_dim', 'condnet_layer',
                        'condnet_dim', 'batch_size']

            # Define tuning
            for key in config:
                if key in integers:
                    hps[key] = int(config[key])
                else:
                    hps[key] = config[key]

            hps['epochs'] = epochs

            model = InvertibleNN(inputs, conditions, hps=hps)

            model.train_model(inputs, conditions, verbose=False, tune_report=True)

        # Defining search algorithm
        if not search_algo:
            search_algo = HyperOptSearch(space=config, metric="L_maxL_val", mode="min")

        result = tune.run(
            train_tuning,
            resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
            num_samples=n_samples,
            scheduler=scheduler,
            search_alg=search_algo)

        best_trial = result.get_best_trial("L_maxL_val", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["L_maxL_val"]))
        print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["Acc_in_val"]))

        return result

    def save_state(self, fname, epoch=None):
        if not epoch:
            torch.save({'model': self.state_dict(),
                        'hps': self.hps,
                        'opt': self.optimizer.state_dict(),
                        'condnet': self.condnet,
                        'in_scaler': self.in_scaler.save(),
                        'cond_scaler': self.cond_scaler.save(),
                        'in_pca': self.in_pca,
                        'cond_pca': self.cond_pca,
                        'train_loss': self.train_loss,
                        'val_loss': self.val_loss,
                        'val_acc': self.val_acc}, fname)
        else:
            torch.save({'model': self.state_dict(),
                        'hps': self.hps,
                        'opt': self.optimizer.state_dict(),
                        'condnet': self.condnet,
                        'in_scaler': self.in_scaler.save(),
                        'cond_scaler': self.cond_scaler.save(),
                        'in_pca': self.in_pca,
                        'cond_pca': self.cond_pca,
                        'epoch': epoch,
                        'train_loss': self.train_loss,
                        'val_loss': self.val_loss,
                        'val_acc': self.val_acc}, fname)

    def load_state(self, fname):
        dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        data = torch.load(fname, map_location=dev)
        data['model'] = {k: v for k, v in data['model'].items() if 'tmp_var' not in k}

        data['hps']['device'] = dev

        self.update_hps(data['hps'])

        self.load_state_dict(data['model'])
        self.in_scaler.load(data['in_scaler'])
        self.cond_scaler.load(data['cond_scaler'])  # new class definition

        self.optimizer.load_state_dict(data['opt'])

        self.condnet = data['condnet']

        if 'in_pca' in data.keys():
            self.in_pca = data['in_pca']
        if 'cond_pca' in data.keys():
            self.cond_pca = data['cond_pca']
        if 'train_loss' in data.keys():
            self.train_loss = data['train_loss']
        if 'val_loss' in data.keys():
            self.val_loss = data['val_loss']
        if 'val_acc' in data.keys():
            self.val_acc = data['val_acc']

        if 'epoch' in data.keys():
            return data['epoch']

        self.trainable_parameters = [p for p in self.parameters() if p.requires_grad]

        if self.hps['condnet']:
            self.trainable_parameters += list(self.condnet.parameters())

    @staticmethod
    def loadmodel(fname, verbose=False):
        dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        data = torch.load(fname, map_location=dev)
        data['model'] = {k: v for k, v in data['model'].items() if 'tmp_var' not in k}

        data['hps']['device'] = dev

        # TODO remove when compatibility is not necessary anymore
        if 'indiv_loss_weight' not in data['hps'].keys():
            data['hps']['indiv_loss_weight'] = None

        hps = InvertibleNN.default_hps()

        for k, v in data['hps'].items():
            if k in hps.keys():
                hps[k] = v

        model = InvertibleNN(hps=hps, verbose=verbose)

        model.load_state_dict(data['model'])
        model.in_scaler.load(data['in_scaler'])
        model.cond_scaler.load(data['cond_scaler'])  # new class definition

        model.optimizer.load_state_dict(data['opt'])

        model.condnet = data['condnet']

        if 'in_pca' in data.keys():
            model.in_pca = data['in_pca']
        if 'cond_pca' in data.keys():
            model.cond_pca = data['cond_pca']
        if 'train_loss' in data.keys():
            model.train_loss = data['train_loss']
        if 'val_loss' in data.keys():
            model.val_loss = data['val_loss']
        if 'val_acc' in data.keys():
            model.val_acc = data['val_acc']

        model.trainable_parameters = [p for p in model.parameters() if p.requires_grad]

        if model.hps['condnet']:
            model.trainable_parameters += list(model.condnet.parameters())

        return model

    @staticmethod
    def resume_training(fname, inputs, conditions, verbose=True, epochs=2, learning_rate=None,
                        check_int=None, check_dir='./ChkPnt', save_file=None):
        # TODO implement this method
        model = InvertibleNN.loadmodel(fname, verbose=0)
        if learning_rate is not None:
            print(model.optimizer)
            for g in model.optimizer.param_groups:
                g['lr'] = learning_rate
                model.hps['learnin_rate'] = learning_rate
            print(model.optimizer)
        else:
            opti = None

        n = np.where(np.array(model.train_loss) != None)
        last_ep = n[0][-1] + 1
        model.hps['epochs'] += epochs

        model.train_model(inputs, conditions, verbose=verbose, tune_report=False,
                          init=False, optimizer=None, epoch_resume=last_ep,
                          check_int=check_int, check_dir=check_dir, save_file=save_file)

        return model

    @staticmethod
    def default_hps():
        """ Return default hyper-parameters """
        hps_dict = {

            # Model Size:
            'feat_in': None,
            'feat_cond': None,

            # Experiment Params:
            'PCA_in': False,  # apply PCA on inputs
            'PCA_cond': False,  # apply PCA on conditions
            'clamp': 2.0,  # necessary input for Couplinblock GLOW
            'n_CouplingBlock': 17,  # number of coupling block
            'permute_soft': False,  # permutation soft of coupling block
            'subnet_layer': 1,  # number of dense hidden layer in affine coupling block subnet
            'subnet_dim': 65,  # perceptron numbers per hidden layer in affine coupling block subnet
            'activation': nn.PReLU(),  # activation function of each hidden layer in affine coupling block subnet
            'condnet': False,
            'condnet_layer': 3,  # condition net layer number
            'condnet_dim': 100,  # output dimension of condition net
            'condnet_activation': nn.ReLU(),  # activation function for condition net

            # Training Params:
            'batch_norm': False,
            'drop_out': True,
            'drop_out_rate': 0.05,
            'optimizer': 'adagrad',  # optimizer
            'learning_rate': 0.05,  # learning rate optimizer
            'optim_eps': 1e-6,
            'weight_decay': 2e-5,
            'lr_scheduler': False,
            'lr_epoch_step': 500,
            'lr_factor': 0.5,
            'epochs': 2000,
            'batch_size': 128,  # batch size
            'test_split': 0.25,
            'grad_clip': 2,
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',  # device name
        }

        return hps_dict

    def __init_losses(self, epoch_resume=0):
        if epoch_resume == 0:
            self.train_loss = [None] * self.hps['epochs']
            self.val_loss = [None] * self.hps['epochs']
            self.val_acc = [None] * self.hps['epochs']
        else:
            self.train_loss = self.train_loss + [None] * (self.hps['epochs'] - epoch_resume)
            self.val_loss = self.val_loss + [None] * (self.hps['epochs'] - epoch_resume)
            self.val_acc = self.val_acc + [None] * (self.hps['epochs'] - epoch_resume)

    def __genCondnet(self):

        dim_mult = np.array(self.hps['condnet_dim'] / self.hps['feat_cond']).repeat(self.hps['condnet_layer'] + 1)
        steps = np.arange(0, self.hps['condnet_layer'] + 1, 1, dtype=int)
        dims = np.rint(self.hps['feat_cond'] * dim_mult ** (steps / steps[-1])).astype(int)

        modules = []

        for i in range(0, self.hps['condnet_layer']):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            if self.hps['batch_norm']:
                modules.append(nn.BatchNorm1d(dims[i + 1]))
            modules.append(self.hps['condnet_activation'])

        return nn.Sequential(*modules)

    def __genSubnet(self, c_in, c_out):
        modules = [nn.Linear(c_in, self.hps['subnet_dim']), self.hps['activation']]

        for i in range(self.hps['subnet_layer']):
            modules.append(nn.Linear(self.hps['subnet_dim'], self.hps['subnet_dim']))
            if self.hps['drop_out']:
                modules.append(nn.Dropout(p=self.hps['drop_out_rate']))
            if self.hps['batch_norm']:
                modules.append(nn.BatchNorm1d(self.hps['subnet_dim']))
            modules.append(self.hps['activation'])

        modules.append(nn.Linear(self.hps['subnet_dim'], c_out))

        return nn.Sequential(*modules)

    def __getDataLoad(self, inputs, conditions, setscaler=False):

        if not torch.is_tensor(inputs):
            if self.hps['PCA_in']:
                if setscaler:
                    self.in_pca.fit(inputs)
                inputs = self.in_pca.transform(inputs)

            inputs = torch.tensor(inputs, dtype=torch.float, device=self.hps['device'])
        else:
            if self.hps['PCA_in']:
                inputs = inputs.cpu().detach().numpy()
                if setscaler:
                    self.in_pca.fit(inputs)
                inputs = self.in_pca.transform(inputs)
                inputs = torch.tensor(inputs, dtype=torch.float, device=self.hps['device'])
            else:
                inputs = inputs.clone()
                inputs = inputs.to(device=self.hps['device'])

        if not torch.is_tensor(conditions):
            if self.hps['PCA_cond']:
                if setscaler:
                    self.cond_pca.fit(conditions)
                conditions = self.cond_pca.transform(conditions)

            conditions = torch.tensor(conditions, dtype=torch.float, device=self.hps['device'])
        else:
            if self.hps['PCA_cond']:
                conditions = conditions.cpu().detach().numpy()
                if setscaler:
                    self.cond_pca.fit(conditions)
                conditions = self.cond_pca.transform(conditions)
                conditions = torch.tensor(conditions, dtype=torch.float, device=self.hps['device'])
            else:
                conditions = conditions.clone()
                conditions = conditions.to(device=self.hps['device'])

        inputs = self.in_scaler.transform(inputs, set_par=setscaler)

        conditions = self.cond_scaler.transform(conditions, set_par=setscaler)

        t_idx = int(inputs.shape[0] * (1 - self.hps['test_split']))

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(inputs[:t_idx, :], conditions[:t_idx, :]),
            batch_size=self.hps['batch_size'], shuffle=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(inputs[t_idx:, :], conditions[t_idx:, :]),
            batch_size=self.hps['batch_size'], shuffle=True, drop_last=True)

        return train_loader, val_loader

    def getOptimizer(self):

        if self.hps['optimizer'].lower() == 'adadelta':
            return torch.optim.Adadelta(self.trainable_parameters, lr=self.hps['learning_rate'],
                                        eps=self.hps['optim_eps'], weight_decay=self.hps['weight_decay'])
        elif self.hps['optimizer'].lower() == 'adagrad':
            return torch.optim.Adagrad(self.trainable_parameters, lr=self.hps['learning_rate'],
                                       eps=self.hps['optim_eps'], weight_decay=self.hps['weight_decay'])
        elif self.hps['optimizer'].lower() == 'adamw':
            return torch.optim.AdamW(self.trainable_parameters, lr=self.hps['learning_rate'], betas=(0.8, 0.9),
                                     eps=self.hps['optim_eps'], weight_decay=self.hps['weight_decay'], amsgrad=False)
        elif self.hps['optimizer'].lower() == 'adamax':
            return torch.optim.Adamax(self.trainable_parameters, lr=self.hps['learning_rate'], betas=(0.8, 0.9),
                                      eps=self.hps['optim_eps'], weight_decay=self.hps['weight_decay'])
        else:
            return torch.optim.Adam(self.trainable_parameters, lr=self.hps['learning_rate'], betas=(0.8, 0.9),
                                    eps=self.hps['optim_eps'], weight_decay=self.hps['weight_decay'])

    @staticmethod
    def __max_Likelyhood(z, log_jac_det):
        return torch.mean(0.5 * torch.sum(z ** 2, dim=1) - log_jac_det)

    @staticmethod
    def __R2(inputs, targets):
        R2 = [np.corrcoef(inputs[:, i].cpu().detach().numpy(), targets[:, i].cpu().detach().numpy())[
                  0, 1] ** 2 for i in range(inputs.shape[1])]
        return np.array(R2)

    def __train_process(self, train_loader, i_epoch=0):
        self.train()
        loss_history = []

        for x, y in train_loader:

            x, y = x.to(self.hps['device']), y.to(self.hps['device'])

            self.optimizer.zero_grad()

            # Forward step:
            if self.hps['condnet']:
                z, log_jac_det = self.forward(x, c=self.condnet(y), rev=False, jac=True)
            else:
                z, log_jac_det = self.forward(x, c=y, rev=False, jac=True)

            # maxlikelyhood loss:
            l_maxL = self.__max_Likelyhood(z, log_jac_det)

            l_maxL.backward()

            loss_history.append(l_maxL.item())

            nn.utils.clip_grad_norm_(self.trainable_parameters, self.hps['grad_clip'])

            self.optimizer.step()

        epoch_losses = np.mean(np.array(loss_history), axis=0)

        self.train_loss[i_epoch] = epoch_losses

    def __val_process(self, val_loader, i_epoch=0):

        self.eval()
        with torch.no_grad():

            loss_history = []
            acc_in = []
            max_it = 150
            it = 0
            for x, y in val_loader:
                it += 1
                if it > max_it:
                    break

                x, y = x.to(self.hps['device']), y.to(self.hps['device'])

                # Forward step:
                if self.hps['condnet']:
                    z, log_jac_det = self.forward(x, c=self.condnet(y), rev=False, jac=True)
                else:
                    z, log_jac_det = self.forward(x, c=y, rev=False, jac=True)

                # maxlikelyhood loss:
                l_maxL = self.__max_Likelyhood(z, log_jac_det)

                # Backward step:
                z = torch.randn_like(z, dtype=torch.float, device=self.hps['device'])
                if self.hps['condnet']:
                    x_rec, log_jac_det = self.forward(z, c=self.condnet(y), rev=True, jac=False)
                else:
                    x_rec, log_jac_det = self.forward(z, c=y, rev=True, jac=False)

                if self.hps['PCA_in']:
                    x = x.cpu().detach().numpy()
                    x_rec = x_rec.cpu().detach().numpy()
                    x = self.in_scaler.inverse_transform(x)
                    x = x.cpu().detach().numpy()
                    x = self.in_pca.inverse_transform(x)
                    x_rec = self.in_scaler.inverse_transform(x_rec)
                    x_rec = x_rec.cpu().detach().numpy()
                    x_rec = self.in_pca.inverse_transform(x_rec)
                    x = torch.tensor(x)
                    x_rec = torch.tensor(x_rec)

                acc_in.append(self.__R2(x, x_rec))

                loss_history.append(l_maxL.item())

            epoch_losses = np.mean(np.array(loss_history), axis=0)

            self.val_loss[i_epoch] = epoch_losses
            self.val_acc[i_epoch] = np.mean(np.stack(acc_in, axis=0), axis=0)

    def print_training(self, fname='Training.png'):

        fig, axs = plt.subplots(1, 2, figsize=[38, 15], dpi=200)
        L_maxL_val = np.array(self.val_loss)

        L_maxL_train = np.array(self.train_loss)

        n = np.where(L_maxL_train != None)
        n = n[0][-1] + 1

        val_acc = np.vstack(self.val_acc[:n])

        ep = np.arange(0, n, 1)
        axs[0].plot(ep, L_maxL_val[:n], 'r', label='L_maxL_val')
        axs[0].plot(ep, L_maxL_train[:n], 'r:', label='L_maxL_train')
        axs[0].set_ylim([min([np.amin(L_maxL_train[:n]), np.amin(L_maxL_val[:n])]), 100])
        axs[0].set_title('Losses')
        axs[0].grid(b=True, which='major', axis='both')
        axs[0].legend(loc='upper right')

        feat = val_acc.shape[-1]
        cm = plt.get_cmap('gist_rainbow')
        cNorm = colors.Normalize(vmin=0, vmax=feat - 1)
        scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
        axs[1].set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(feat)])
        for i in range(feat):
            axs[1].plot(ep, val_acc[:n, i], label='Ch_' + str(i))
            axs[1].annotate(xy=(ep[-1], val_acc[n - 1, i]), xytext=(5, 0), textcoords='offset points',
                            text='Ch_' + str(i), va='center')
            axs[1].set_title('Accuracy, mean accuracy= %6.5f' % (np.mean(val_acc[n - 1, :])))
        axs[1].set_ylim([np.amin(val_acc), np.amax(val_acc)])
        axs[1].grid(b=True, which='major', axis='both')
        axs[1].legend(loc='upper left')

        plt.savefig(fname)
        plt.close('all')

    def print_training2(self, fname='Training.png'):

        matplotlib.rcParams['text.usetex'] = True

        fig, axs1 = plt.subplots(figsize=[12, 4], dpi=300)


        L_maxL_val = np.array(self.val_loss)
        L_maxL_train = np.array(self.train_loss)

        n = np.where(L_maxL_train != None)
        n = n[0][-1] + 1

        val_acc = np.vstack(self.val_acc[:n])

        ep = np.arange(0, n, 1)

        matplotlib.rcParams['text.usetex'] = True
        axs2 = axs1.twinx()

        matplotlib.rcParams['text.usetex'] = True

        ep = np.arange(0, n, 4)
        feat = val_acc.shape[-1]
        cm = plt.get_cmap('gist_rainbow')
        cNorm = colors.Normalize(vmin=0, vmax=feat - 1)
        scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
        axs1.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(feat)])
        for i in range(feat):
            axs1.plot(ep, val_acc[ep, i], label='Ch_' + str(i))
            # axs1.annotate(xy=(ep[-1], val_acc[n - 1, i]), xytext=(5, 0), textcoords='offset points',
            #                text='Ch_' + str(i), va='center')
        axs1.set_ylim([0, 1])
        axs1.grid(b=True, which='major', axis='both')

        axs2.plot(ep, L_maxL_val[ep], 'k', label='L_maxL_val')
        axs2.plot(ep, L_maxL_train[ep], 'k:', label='L_maxL_train')
        axs2.set_ylim([-60, 100])  # [min([np.amin(L_maxL_train[:n]), np.amin(L_maxL_val[:n])]), 100])
        axs2.grid(b=True, which='major', axis='both')

        axs2.set_xlim([0, n])
        axs1.set_xlabel(r'training epochs')

        axs1.set_title(r'$\mathcal{L}_{min,val}$ = %5.2f, $\overline{acc}_{val}$ = %6.5f' % (np.min(L_maxL_val), np.mean(val_acc[n - 1, :])))
        axs1.set_yticks(np.linspace(axs1.get_yticks()[0], axs1.get_yticks()[-1], len(axs2.get_yticks())))

        axs1.yaxis.tick_right()
        axs1.yaxis.set_label_position("right")
        axs2.yaxis.tick_left()
        axs2.yaxis.set_label_position("left")
        axs2.set_ylabel(r'neg log likelihood loss')
        axs1.set_ylabel(r'input feature accuracy')
        plt.savefig(fname)
        plt.close('all')
