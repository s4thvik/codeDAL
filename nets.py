# nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import models
from continual_backprop import ContinualBackprop

class CustomResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super(CustomResNet18, self).__init__()
        # Initialize ResNet18 from torchvision without pretrained weights
        self.model = models.resnet18(pretrained=False)
        # Modify the final fully connected layer to match num_classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        out = self.model(x)
        return out

    def predict(self, x):
        features = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.model.avgpool(x)
        features = x.view(x.size(0), -1)
        out = self.model.fc(features)
        return out, features

class ContinualBackpropNet(nn.Module):
    def __init__(self, net_cls, params, device, replacement_rate=1e-4, 
                 maturity_threshold=500, decay_rate=0.9):
        super().__init__()
        self.clf = net_cls(num_classes=params['num_classes'])
        self.device = device
        self.clf.to(device)
        self.params = params
        
        # Initialize CBP object
        self.cbp = ContinualBackprop(
            net=self.clf,
            step_size=params['optimizer_args']['lr'],
            loss='nll',  # for classification
            opt='sgd',
            momentum=params['optimizer_args']['momentum'],
            replacement_rate=replacement_rate,
            decay_rate=decay_rate,
            device=device,
            maturity_threshold=maturity_threshold,
            weight_decay=params['optimizer_args']['weight_decay']
        )
        
        # CBP specific parameters
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.decay_rate = decay_rate
        
        # Initialize utilities and ages for each layer
        self.utilities = {}
        self.ages = {}
        self.running_means = {}
        
        for name, layer in self.clf.named_modules():
            if isinstance(layer, nn.Linear):
                self.utilities[name] = torch.zeros(layer.out_features).to(device)
                self.ages[name] = torch.zeros(layer.out_features).to(device)
                
    def train_model(self, labeled_data):
        """Train using CBP instead of standard backprop"""
        from tqdm import tqdm
        self.train()
        total_loss = 0
        metrics = {}
        n_epochs = self.params['train_args'].get('num_epochs', 100)
        
        # Create epoch-level progress bar
        epoch_pbar = tqdm(range(n_epochs), desc='Training Epochs', leave=True)
        
        for epoch in epoch_pbar:
            loader = DataLoader(labeled_data, **self.params['train_args'])
            # Create batch-level progress bar
            batch_pbar = tqdm(loader, desc=f'Epoch {epoch+1}', leave=False)
            epoch_loss = 0
            
            for x, y, idxs in batch_pbar:
                x, y = x.to(self.device), y.to(self.device)
                loss, output, features = self.train_on_batch(x, y)
                epoch_loss += loss.item()
                
                # Update utilities and perform selective reinitialization
                for name, layer in self.clf.named_modules():
                    if isinstance(layer, nn.Linear):
                        self.update_utilities(name, features, output)
                        self._selective_reinit(name, layer)
                        self.ages[name] += 1
                        
                        metrics[name] = {
                            'utility': self.utilities[name].mean().item(),
                            'age': self.ages[name].mean().item()
                        }
                
                # Update batch progress bar
                batch_pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
            
            avg_epoch_loss = epoch_loss / len(loader)
            total_loss += avg_epoch_loss
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({'avg_loss': f'{avg_epoch_loss:.4f}'})
        
        avg_loss = total_loss / n_epochs
        return avg_loss, metrics

    def forward(self, x):
        return self.clf(x)

    def train_on_batch(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        output, features = self.clf.predict(x)
        loss = F.cross_entropy(output, y)
        
        # Standard backprop
        loss.backward()
        self.cbp.opt.step()
        self.cbp.opt.zero_grad()
        
        # Selective reinitialization
        for name, module in self.clf.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self._selective_reinit(name, module)
                
        return loss, output, features

    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=torch.long)
        loader = DataLoader(data, **self.params['test_args'])
        with torch.no_grad():
            for x, _, idxs in loader:
                x = x.to(self.device)
                out = self.clf(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds.numpy()

    def predict_prob(self, data):
        self.clf.eval()
        probs = torch.zeros([len(data), self.params['num_classes']])
        loader = DataLoader(data, **self.params['test_args'])
        with torch.no_grad():
            for x, _, idxs in loader:
                x = x.to(self.device)
                out = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs

    def update_utilities(self, name, features, output):
        if name not in self.utilities:
            return
            
        layer = dict(self.clf.named_modules())[name]
        
        # Update running mean of features
        if name not in self.running_means:
            self.running_means[name] = features.mean(0)
        else:
            self.running_means[name] = (self.decay_rate * self.running_means[name] + 
                                      (1 - self.decay_rate) * features.mean(0))
        
        # Calculate mean-corrected contribution (equation 6 from paper)
        mean_corrected = torch.abs(features - self.running_means[name])  # [batch_size, feature_dim]
        
        # Get weight dimensions
        out_features, in_features = layer.weight.shape
        
        # Calculate weight terms
        outgoing_weights = torch.abs(layer.weight).sum(dim=0)  # [in_features]
        incoming_weights = torch.abs(layer.weight).sum(dim=1)  # [out_features]
        
        # Calculate contribution term
        contribution = mean_corrected * outgoing_weights.unsqueeze(0)  # [batch_size, in_features]
        contribution = contribution.mean(0)  # [in_features]
        
        # Reshape contribution to match incoming weights
        contribution = contribution.unsqueeze(1).expand(-1, out_features)  # [in_features, out_features]
        
        # Calculate instantaneous utility
        instantaneous_utility = (contribution / (incoming_weights + 1e-8)).mean(0)  # [out_features]
        
        # Update utility using equation 7 from the paper
        self.utilities[name] = (self.decay_rate * self.utilities[name] + 
                              (1 - self.decay_rate) * instantaneous_utility)

    def _selective_reinit(self, name, module):
        if name not in self.utilities:
            return
            
        # Only consider units past protection period
        mature_units = self.ages[name] > self.maturity_threshold
        if not mature_units.any():
            return
            
        num_reinit = int(self.replacement_rate * mature_units.sum().item())
        if num_reinit > 0:
            utilities_mature = self.utilities[name][mature_units]
            _, indices = torch.topk(-utilities_mature, num_reinit)
            units_to_reinit = torch.where(mature_units)[0][indices]
            
            # Zero outgoing weights for new units (as per lines 504-506)
            with torch.no_grad():
                module.weight[units_to_reinit].zero_()
                if module.bias is not None:
                    module.bias.data[units_to_reinit] = 0
                    
            # Reset ages for protection period
            self.ages[name][units_to_reinit] = 0

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        out = x.view(x.size(0), -1)
        e1 = out
        out = self.linear(out)
        return out, e1

    def get_embedding_dim(self):
        return 512

def ResNet18(num_classes=100):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

class MNIST_Net(nn.Module):
    def __init__(self, num_classes=10):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

class SVHN_Net(nn.Module):
    def __init__(self, num_classes=10):
        super(SVHN_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 64 * 8 * 8)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 256

class CIFAR10_Net(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 8 * 8)
        e1 = F.relu(self.fc1(x))
        x = self.fc2(e1)
        return x, e1

    def get_embedding_dim(self):
        return 512


#hi3
