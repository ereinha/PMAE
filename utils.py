import matplotlib.pyplot as plt
import numpy as np
import DataSet
import json

# Make 2-dimensional histogram
def make_hist2d(group_num, group_size, step, names, inputs, outputs, scaler, event_type, file_path, lower=None, upper=None):
    inputs = scaler.inverse_transform(inputs)
    outputs = scaler.inverse_transform(outputs)
    if lower is None:
        lower = np.min((outputs[:,group_num*group_size+step], inputs[:,group_num*group_size+step]))
    if upper is None:
        upper = np.max((outputs[:,group_num*group_size+step], inputs[:,group_num*group_size+step]))
    varname = names[group_num*group_size+step]
    heatmap, xedges, yedges = np.histogram2d(inputs[:,group_num*group_size+step],
                                             outputs[:,group_num*group_size+step],
                                             bins=30,
                                             range=[[lower, upper], [lower, upper]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    #Plot heatmap
    plt.imshow(heatmap.T,
               extent=extent,
               origin='lower')
    plt.plot([lower, upper],
             [lower, upper],
             color='blue')
    fig = plt.gcf()
    plt.set_cmap('gist_heat_r')
    plt.xlabel('%s True' % varname)
    plt.ylabel('%s Pred' % varname)
    plt.title('Frequency Heatmap (' + event_type + ')')
    plt.xlim(lower, upper)
    plt.ylim(lower, upper)
    plt.colorbar()
    plt.savefig(file_path + '/hist2d_' + event_type)
    plt.show()

# Custom loss function expects shape [batch_size, num_particles, 3] where 3 items are pt, eta, phi
class custom_loss(LossFunction):
    def __init__(self, phi_limit, alpha=0.4, beta=.5, gamma=1.):
        self.phi_limit = phi_limit
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.loss_mask = loss_mask
        self.zero_padded = zero_padded
    def compute_loss(self, output, target, loss_mask=[], zero_padded=[]):
        loss = 0
        for i in range(output.size()[1]):
            if i in (self.loss_mask):
                continue
            elif i in self.zero_padded:
                continue
            elif i % 3 == 0:
                loss += torch.mean((target[:,i] - output[:,i])**2 + torch.gt(output[:,i], lower_pt_limit[(i + 3) % 4]).long() * \
                    (self.gamma / (1 + torch.exp(-(output[:,i] - lower_pt_limit[(i + 3) % 4]) * 3)) - self.gamma) + \
                        torch.le(output[:,i], lower_pt_limit[(i + 3) % 4]).long()*(self.gamma/2 - self.gamma))
            elif i % 3 == 1:
                loss += torch.mean((target[:,i] - output[:,i])**2 - output[:,i]**2 * self.beta)
            elif i % 3 == 2:
                loss += torch.mean(torch.le(torch.abs(output[:,i]), self.phi_limit).long() *\
                    ((torch.sin(((output[:,i] - target[:,i]) / self.phi_limit - .5) * 3.14159265) + 1)**2 +\
                        (torch.sin(((output[:,i] - target[:,i]) / self.phi_limit - .5) * 3.14159265) + 1) * 2) * self.alpha +\
                    torch.gt(torch.abs(output[:,i]), self.phi_limit).long() *\
                    (((torch.sin(((self.phi_limit * torch.sign(output[:,i]) - target[:,i]) / self.phi_limit  - .5) * \
                                    3.14159265) + 1)**2 +\
                        (torch.sin(((self.phi_limit * torch.sign(output[:,i]) - target[:,i]) / self.phi_limit  - .5) * \
                                3.14159265) + 1) * 2) * self.alpha +\
                    (self.phi_limit*torch.sign(output[:,i]) - output[:,i])**2))
        return loss / (output.size()[1] - len(self.zero_padded) - len(self.loss_mask))

# Dataset class
class DataLabelDataset(Dataset):
    def __init__(self, data, labels, dtype: str = 'numpy'):
        super(DataLabelDataset, self).__init__()
        if dtype == 'numpy':
            self.data = torch.from_numpy(data)
            self.labels = torch.from_numpy(labels)
        elif dtype == 'torch':
            self.data = data
            self.labels = labels
    def __getitem__(self, index):
        return self.config[index], self.labels[index]

    def __len__(self):
        return len(self.data)

# Custom SGD optimizer
class SGDWithSaturatingMomentumAndDecay(optim.Optimizer):
    def __init__(self, params, lr=None, momentum=0, max_momentum=0.99, epochs_to_saturate=100, batches_per_epoch=1, weight_decay=0, lr_decay=0.1, min_lr=1e-6, resume_epoch=0):
        if lr is not None and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, max_momentum=max_momentum, epochs_to_saturate=epochs_to_saturate, batches_per_epoch=batches_per_epoch, weight_decay=weight_decay, lr_decay=lr_decay, min_lr=min_lr, resume_epoch=resume_epoch)
        super(SGDWithSaturatingMomentumAndDecay, self).__init__(params, defaults)

        for group in self.param_groups:
            # Adjust initial learning rate and momentum based on resume epoch
            steps_to_saturate = group['epochs_to_saturate'] * group['batches_per_epoch']
            resumed_steps = group['resume_epoch'] * group['batches_per_epoch']
            max_momentum = group['max_momentum']
            momentum_step = (max_momentum - group['momentum']) / steps_to_saturate
            group['momentum'] = min(group['momentum'] + momentum_step * resumed_steps, max_momentum)
            group['lr'] = max(group['lr'] * (group['lr_decay'] ** resumed_steps), group['min_lr'])

    def step(self, closure=None):
        for group in self.param_groups:
            steps_to_saturate = group['epochs_to_saturate'] * group['batches_per_epoch']
            max_momentum = group['max_momentum']
            momentum_step = (max_momentum - group['momentum']) / steps_to_saturate

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if group['weight_decay'] != 0:
                    d_p.add_(p.data, alpha=group['weight_decay'])

                if group['momentum'] != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(d_p)
                    d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])

            # Increment momentum and decay learning rate after the step
            group['momentum'] = min(group['momentum'] + momentum_step, max_momentum)
            group['lr'] = max(group['lr'] * group['lr_decay'], group['min_lr'])

def parse_model_name(str:model_name):
    elements = model_name.split('_')[1:]  # remove the first 'Model' element

    # Initialize an empty dictionary to hold the key-value pairs
    json_dict = {}

    # Parse the elements
    for element in elements:
        key, value = element[0], element[1:]

        # Try converting values to float or int if possible, else leave them as string
        try:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass

        # Construct the dictionary
        json_dict[key] = value

    # Convert the dictionary to a JSON string
    json_string = json.dumps(json_dict, indent=4)

    return json.loads(json_string)