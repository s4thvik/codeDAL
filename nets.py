import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

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

class Net:
    def __init__(self, net_cls, params, device):
        self.net_cls = net_cls
        self.params = params
        self.device = device

        num_classes = self.params['num_classes']
        self.clf = self.net_cls(num_classes=num_classes).to(self.device)

        # Initialize optimizer and scheduler
        self.optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
        self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)  # Adjusted step_size for 100 epochs

        # Initialize seen classes
        self.seen_classes = []
        self.num_seen_classes = 0

    def update_seen_classes(self, seen_classes):
        self.seen_classes = seen_classes
        self.num_seen_classes = len(seen_classes)
        self.seen_classes_tensor = torch.tensor(self.seen_classes).to(self.device)

    def train(self, data):
        n_epoch = self.params['n_epoch']
        self.clf.train()

        loader = DataLoader(data, shuffle=True, **self.params['train_args'])
        for epoch in range(1, n_epoch + 1):
            with tqdm(loader, ncols=100, leave=True, desc=f"Epoch {epoch}/{n_epoch}") as t:
                for batch_idx, (x, y, idxs) in enumerate(t):
                    x, y = x.to(self.device), y.to(self.device)

                    self.optimizer.zero_grad()
                    out, _ = self.clf(x)

                    # Select outputs for seen classes
                    out = out[:, self.seen_classes]

                    loss = F.cross_entropy(out, y)
                    loss.backward()
                    self.optimizer.step()
                    t.set_postfix(loss=loss.item())

            self.scheduler.step()

    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=torch.long)
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x = x.to(self.device)
                out, _ = self.clf(x)

                # Select outputs for seen classes
                out = out[:, self.seen_classes]

                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds

    def predict_prob(self, data):
        self.clf.eval()
        num_samples = len(data)
        probs = torch.zeros([num_samples, self.num_seen_classes])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x = x.to(self.device)
                out, _ = self.clf(x)

                # Select outputs for seen classes
                out = out[:, self.seen_classes]

                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()  # Enable dropout layers
        num_samples = len(data)
        probs = torch.zeros([num_samples, self.num_seen_classes])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for _ in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x = x.to(self.device)
                    out, _ = self.clf(x)

                    # Select outputs for seen classes
                    out = out[:, self.seen_classes]

                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs

    def get_embeddings(self, data):
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x = x.to(self.device)
                _, e1 = self.clf(x)
                embeddings[idxs] = e1.cpu()
        return embeddings

# Define other networks similarly, adding num_classes as a parameter

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

