# nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import models

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

class ContinualBackpropNet(nn.Module):
    def __init__(self, net_cls, params, device, replacement_rate=0.0001, maturity_threshold=100, decay_rate=0.9):
        super(ContinualBackpropNet, self).__init__()
        self.clf = net_cls(num_classes=params['num_classes']).to(device)
        self.params = params
        self.optimizer = SGD(self.clf.parameters(), **params['optimizer_args'])
        # Modified scheduler for better learning rate decay
        self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.device = device
        
        # CBP parameters
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.decay_rate = decay_rate
        
        # Track utilities and ages for each layer
        self.utilities = {}
        self.ages = {}
        self.features = {}
        
        # Initialize utilities and ages for each layer
        for name, module in self.clf.named_modules():
            if isinstance(module, nn.Linear):
                num_units = module.out_features
                self.utilities[name] = torch.zeros(num_units).to(device)
                self.ages[name] = torch.zeros(num_units).to(device)
            elif isinstance(module, nn.Conv2d):
                num_units = module.out_channels
                self.utilities[name] = torch.zeros(num_units).to(device)
                self.ages[name] = torch.zeros(num_units).to(device)
            
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.register_forward_hook(self._feature_hook(name))

    def _feature_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook

    def _compute_utility(self, name, module):
        if name not in self.features:
            return
            
        features = self.features[name]
        weights = module.weight.data
        
        if isinstance(module, nn.Conv2d):
            # For Conv2d, average across spatial dimensions
            features = features.mean(dim=(2, 3))  # Average across H,W
            
        # Compute mean-corrected contribution utility
        mean_activations = features.mean(dim=0, keepdim=True)
        contribution = torch.abs(features - mean_activations).mean(dim=0)
        
        # Compute adaptation utility based on layer type
        if isinstance(module, nn.Conv2d):
            # For Conv2d: average across input channels, kernel height, and width
            adaptation = 1 / (weights.abs().mean(dim=(1, 2, 3)) + 1e-10)
        else:  # Linear layer
            # For Linear: average across input features
            adaptation = 1 / (weights.abs().mean(dim=1) + 1e-10)
        
        # Ensure dimensions match
        if contribution.shape != adaptation.shape:
            if len(contribution.shape) > len(adaptation.shape):
                contribution = contribution.mean(dim=tuple(range(1, len(contribution.shape))))
            else:
                adaptation = adaptation.mean(dim=tuple(range(1, len(adaptation.shape))))
        
        # Update utility with decay
        self.utilities[name] = self.decay_rate * self.utilities[name] + \
                             (1 - self.decay_rate) * (contribution * adaptation)
        self.ages[name] += 1

    def _selective_reinit(self, name, module):
        if name not in self.utilities:
            return
            
        # Find eligible units for reinitialization
        mature_units = self.ages[name] > self.maturity_threshold
        if not mature_units.any():
            return
            
        # Calculate number of units to reinitialize
        num_reinit = int(self.replacement_rate * mature_units.sum().item())
        if num_reinit == 0:
            return
            
        # Select units with lowest utility
        utilities_mature = self.utilities[name][mature_units]
        _, indices = torch.topk(-utilities_mature, num_reinit)
        units_to_reinit = torch.where(mature_units)[0][indices]
        
        # Reinitialize selected units
        with torch.no_grad():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight[units_to_reinit])
                if module.bias is not None:
                    module.bias.data[units_to_reinit] = 0
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight[units_to_reinit])
                if module.bias is not None:
                    module.bias.data[units_to_reinit] = 0
                    
            # Reset ages for reinitialized units
            self.ages[name][units_to_reinit] = 0

    def train_model(self, data):
        loader = DataLoader(data, **self.params['train_args'])
        total_loss = 0
        num_batches = 0
    
        for epoch in range(self.params['n_epoch']):
            epoch_loss = 0
            self.clf.train()
        
            with tqdm(loader, desc=f"Epoch {epoch+1}/{self.params['n_epoch']}") as t:
                for x, y, idxs in t:
                    x, y = x.to(self.device), y.to(self.device)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    out = self.clf(x)
                    loss = F.cross_entropy(out, y)
                
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                
                    # Apply CBP
                    for name, module in self.clf.named_modules():
                        if isinstance(module, (nn.Linear, nn.Conv2d)):
                            self._compute_utility(name, module)
                            self._selective_reinit(name, module)
                
                    epoch_loss += loss.item()
                    num_batches += 1
                    t.set_postfix(loss=loss.item())
                
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
            self.scheduler.step()
        
        return avg_loss

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

# Define ResNet-18 for CIFAR-100
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
        return out, e1  # return embedding for compatibility with current structure

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


#hi6
