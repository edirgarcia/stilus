
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from stilus.data.sets import MidiDataset
from torch.utils.data import DataLoader


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

class AbstractMidiNet(pl.LightningModule):

    def __init__(self):
        super(AbstractMidiNet, self).__init__()
        # Calls super. Body is implemented in each concrete model class
        #self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        self.midi_dataset = MidiDataset("training_data.npy")
        self.midi_test_dataset = MidiDataset("test_data.npy", self.midi_dataset.mean, self.midi_dataset.std)
        self.val_test_dataset = MidiDataset("validation_data.npy", self.midi_dataset.mean, self.midi_dataset.std)

    def forward(self, x):
        # Is implemented in each concrete model class
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return {'test_loss': loss}

    def test_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': test_loss_mean}

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     loss = self.criterion(y_hat, y)
    #     return {'val_loss': loss}
    
    # def validation_end(self, outputs):
    #     val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     return {'val_loss': val_loss_mean}

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.midi_dataset, batch_size=128, shuffle=True)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.midi_test_dataset, batch_size=128, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_test_dataset, batch_size=64, shuffle=False)

    

class ConvNet_1_0_0(AbstractMidiNet):

    def __init__(self):
        super(ConvNet_1_0_0, self).__init__()
        self.conv1 = nn.Conv1d(5, 10, 4 ) # bs * 5 * (28)
        self.conv2 = nn.Conv1d(10, 10, 4) # bs * 10 * (24)
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

class ConvNet_1_0_1(AbstractMidiNet):

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


class ConvNet_1_0_2(AbstractMidiNet):

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

class ConvNet_1_0_3(AbstractMidiNet):

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

class TransformerNet_1_0_0(AbstractMidiNet):

    def __init__(self):
        super(TransformerNet_1_0_0, self).__init__() # bs * 5 * 32
        self.encoder0 = nn.TransformerEncoderLayer(d_model=32, nhead=8, dim_feedforward=512)
        self.fc0 = nn.Linear(32 * 5, 5)

    def forward(self, x):
        x = self.encoder0(x)
        x = x.view(-1, num_flat_features(x))
        x = self.fc0(x)
        return x

class TransformerNet_1_0_1(AbstractMidiNet):

    def __init__(self):
        super(TransformerNet_1_0_1, self).__init__() # bs * 5 * 32
        self.encoder0 = nn.TransformerEncoderLayer(d_model=32, nhead=8)
        self.fc0 = nn.Linear(32 * 5, 5)

    def forward(self, x):
        x = self.encoder0(x)
        x = x.view(-1, num_flat_features(x))
        x = self.fc0(x)
        return x

class TransformerNet_1_0_2(AbstractMidiNet):

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




