import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn

class ContinualBackprop(object):
    def __init__(
            self,
            net,
            step_size=0.001,
            loss='mse',
            opt='sgd',
            beta=0.9,
            beta_2=0.999,
            replacement_rate=0.001,
            decay_rate=0.9,
            device='cpu',
            maturity_threshold=100,
            util_type='contribution',
            init='kaiming',
            accumulate=False,
            momentum=0,
            outgoing_random=False,
            weight_decay=0
    ):
        self.net = net
        self.decay_rate = decay_rate
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold

        # define the optimizer
        if opt == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=step_size, 
                               momentum=momentum, weight_decay=weight_decay)
        elif opt == 'adam':
            self.opt = optim.Adam(self.net.parameters(), lr=step_size, 
                                betas=(beta, beta_2), weight_decay=weight_decay)

        # define the loss function
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[loss]

        # Initialize running means for feature tracking
        self.running_means = {}
        
    def learn(self, x, target):
        """Implements CBP learning as described in the paper"""
        output, features = self.net.predict(x)
        loss = self.loss_func(output, target)
        
        # Standard backprop step
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        return loss.detach(), output.detach(), features

    def update_utilities(self, name, features, output):
        if name not in self.utilities:
            return
        
        layer = dict(self.clf.named_modules())[name]
        
        # Calculate mean-corrected features (hl,i,t - fl,i,t)
        if name not in self.running_means:
            self.running_means[name] = features.mean(0)
        else:
            self.running_means[name] = self.decay_rate * self.running_means[name] + \
                                      (1 - self.decay_rate) * features.mean(0)
        
        mean_corrected = torch.abs(features - self.running_means[name])
        
        # Calculate weight terms from equation 6
        if isinstance(layer, nn.Linear):
            outgoing_weights = torch.abs(layer.weight).sum(dim=0)  # Σ|wl,i,k,t|
            incoming_weights = torch.abs(layer.weight).sum(dim=1)  # Σ|wl-1,j,i,t|
            
            # Calculate yl,i,t according to equation 6
            y_l_i_t = (mean_corrected @ outgoing_weights.unsqueeze(1)) / (incoming_weights + 1e-8)
            
            # Update running average according to equation 7
            self.utilities[name] = self.decay_rate * self.utilities[name] + \
                                  (1 - self.decay_rate) * y_l_i_t.mean(0)


#hi1

