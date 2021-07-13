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
