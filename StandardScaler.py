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

import torch


class StandardScaler:
    def __init__(self, std=1, mu=0):
        self.par = {'std': torch.tensor(std, dtype=torch.float),
                    'mu': torch.tensor(mu, dtype=torch.float),
                    'shift': torch.tensor(0, dtype=torch.float),
                    'scale': torch.tensor(1, dtype=torch.float),
                    'device': None}

        self.par['device'] = self.par['scale'].device.type

    def transform(self, inputs, set_par=False):
        # TODO Check for std=0 due to no varying values
        # TODO Think about PCA or any Data preprocessing

        if self.par['device'] != 'cpu' and not torch.cuda.is_available():
            self.copy2dev('cpu')

        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.float, device=self.par['device'])

        if inputs.device.type != self.par['device']:
            if set_par:
                self.copy2dev(inputs.device.type)
                realocate = False
            else:
                dev = inputs.device.type
                inputs = inputs.to(self.par['device'])
                realocate = True
        else:
            realocate = False

        if set_par:

            mu = inputs.mean(0, keepdim=False)
            std = inputs.std(0, unbiased=False, keepdim=False)
            tmp = std == 0
            std[tmp] = 1

            self.par['shift'] = mu + self.par['mu']
            self.par['scale'] = std / self.par['std']

            inputs -= self.par['shift']
            inputs /= self.par['scale']

        else:
            inputs -= self.par['shift']
            inputs /= self.par['scale']

        if realocate:
            inputs = inputs.to(dev)

        return inputs

    def inverse_transform(self, inputs):

        if self.par['device'] != 'cpu' and not torch.cuda.is_available():
            self.copy2dev('cpu')

        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.float, device=self.par['device'])

        if inputs.device.type != self.par['device']:
            dev = inputs.device.type
            inputs = inputs.to(self.par['device'])
            realocate = True
        else:
            realocate = False

        inputs *= self.par['scale']
        inputs += self.par['shift']

        if realocate:
            inputs = inputs.to(dev)

        return inputs

    def save(self):
        return self.par

    def load(self, par):
        self.par = par

    def copy2dev(self, device):
        self.par['mu'] = self.par['mu'].to(device)
        self.par['std'] = self.par['std'].to(device)
        self.par['shift'] = self.par['shift'].to(device)
        self.par['scale'] = self.par['scale'].to(device)
        self.par['device'] = device
