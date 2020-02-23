
import torch
import torch.nn as nn
import torch.nn.functional as F

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

class ConvNet_1_0_0(nn.Module):
    def __init__(self):
        super(ConvNet_1_0_0, self).__init__()
        self.conv1 = nn.Conv1d(5, 10, 4 ) # n * 5 * (28)
        self.conv2 = nn.Conv1d(10, 10, 4) #n * 10 * (24)
        self.fc1 = nn.Linear(10 * 13, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x,2)
        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ConvNet_1_0_1(nn.Module):
    def __init__(self):
        super(ConvNet_1_0_1, self).__init__()
        self.conv1 = nn.Conv1d(5, 10, 4 )
        self.fc1 = nn.Linear(140, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        x = F.max_pool1d(x,2)
        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConvNet_1_0_2(nn.Module):
    def __init__(self):
        super(ConvNet_1_0_2, self).__init__() # bs * 5 * 32
        self.conv1 = nn.Conv1d(5, 10, 4) # bs * 10 * 28+1
        self.fc0 = nn.Linear(10*29, 64) # bs * 64
        self.conv2 = nn.Conv1d(1, 10, 4) # bs * 10 * 60
        self.fc1 = nn.Linear(10 * 30, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc0(x))
        x = torch.unsqueeze(x, 1)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x,2)
        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ConvNet_1_0_3(nn.Module):
    def __init__(self):
        super(ConvNet_1_0_3, self).__init__() # bs * 5 * 32
        self.conv1 = nn.Conv1d(5, 10, 4) # bs * 10 * 28+1
        self.fc0 = nn.Linear(10*29, 64) # bs * 64
        self.conv2 = nn.Conv1d(1, 10, 4) # bs * 10 * 60
        self.fc1 = nn.Linear(10 * 30, 128)
        self.dr0 = nn.Dropout(.2)
        self.fc2 = nn.Linear(128, 64)
        self.dr1 = nn.Dropout(.2)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc0(x))
        x = torch.unsqueeze(x, 1)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x,2)
        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.dr0(x)
        x = F.relu(self.fc2(x))
        x = self.dr1(x)
        x = self.fc3(x)
        return x

class TransformerNet_1_0_0(nn.Module):

    def __init__(self):
        super(TransformerNet_1_0_0, self).__init__() # bs * 5 * 32
        self.encoder0 = nn.TransformerEncoderLayer(d_model=32, nhead=8, dim_feedforward=512)
        self.fc0 = nn.Linear(32 * 5, 5)

    def forward(self, x):
        x = self.encoder0(x)
        x = x.view(-1, num_flat_features(x))
        x = self.fc0(x)
        return x

class TransformerNet_1_0_1(nn.Module):

    def __init__(self):
        super(TransformerNet_1_0_1, self).__init__() # bs * 5 * 32
        self.encoder0 = nn.TransformerEncoderLayer(d_model=32, nhead=8)
        self.fc0 = nn.Linear(32 * 5, 5)

    def forward(self, x):
        x = self.encoder0(x)
        x = x.view(-1, num_flat_features(x))
        x = self.fc0(x)
        return x

class TransformerNet_1_0_2(nn.Module):

    def __init__(self):
        super(TransformerNet_1_0_2, self).__init__() # bs * 5 * 32
        self.encoder0 = nn.TransformerEncoderLayer(d_model=32, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder0, num_layers=3)
        self.fc0 = nn.Linear(32 * 5, 5)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, num_flat_features(x))
        x = self.fc0(x)
        return x




