
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

class WeightedL1Loss(nn.Module):
    def __init__(self):
        super(WeightedL1Loss, self).__init__()
        self.internal_loss = nn.L1Loss()

        cuda0 = torch.device('cuda:0')

        self.importance = torch.tensor([[1.0], [1.0], [1.0], [1.0], [1.0]], requires_grad=True, device=cuda0 )
        
    def forward(self, y_hat, y):
        
        #print("y_hat->",y_hat.shape)
        #print("importance->",self.importance.shape)
        x = y_hat * self.importance[:, None].T
        #print("x->",x.shape)
        
        return self.internal_loss(x[0], y)
    

class AbstractMidiNet(pl.LightningModule):

    def __init__(self):
        super(AbstractMidiNet, self).__init__()
        # Calls super. Body is implemented in each concrete model class
        self.criterion = nn.MSELoss()
        #self.criterion = WeightedL1Loss()
        #for logging
        self.total_epochs = 0
        self.total_batches = 0
        self.data_set = False

    def forward(self, x):
        # Is implemented in each concrete model class
        pass

    def training_step(self, batch, batch_idx):

        if not self.data_set:
            raise Exception("Data path has not been set, can't train")

        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.logger.experiment.add_scalar("training_loss", loss, self.total_batches)
        self.total_batches += 1
        return {'loss': loss}

    def set_data_path(self, data_path):
        self.midi_dataset = MidiDataset(data_path + "/training_data.npy")
        #self.midi_test_dataset = MidiDataset("test_data.npy", self.midi_dataset.mean, self.midi_dataset.std)
        self.midi_val_dataset = MidiDataset(data_path + "/validation_data.npy", self.midi_dataset.mean, self.midi_dataset.std)
        self.name = type(self).__name__ + "_" + data_path.replace("data/", "")
        self.data_set = True

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     loss = self.criterion(y_hat, y)
    #     return {'test_loss': loss}

    # def test_end(self, outputs):
    #     test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     return {'test_loss': test_loss_mean}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return {'val_loss': loss}
    
    def validation_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("val_loss", val_loss_mean, self.total_epochs)
        self.total_epochs += 1
        return {'val_loss': val_loss_mean}

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def train_dataloader(self):
        return DataLoader(self.midi_dataset, batch_size=128, shuffle=True)

    # def test_dataloader(self):
    #     return DataLoader(self.midi_test_dataset, batch_size=128, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.midi_val_dataset, batch_size=128, shuffle=False)

    

class ConvNet_1_0_0(AbstractMidiNet):

    def __init__(self):
        super(ConvNet_1_0_0, self).__init__()
        self.conv1 = nn.Conv1d(5, 10, 4 ) # bs * 5 * (60)
        self.conv2 = nn.Conv1d(10, 10, 4) # bs * 10 * (56)
        self.fc1 = nn.Linear(10 * 29, 128)
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
        self.conv1 = nn.Conv1d(5, 10, 4 ) # bs * 5 * (60)
        self.fc1 = nn.Linear(300, 128)
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
        super(ConvNet_1_0_2, self).__init__() # bs * 5 * 64
        self.conv1 = nn.Conv1d(5, 10, 4) # bs * 10 * 60+1
        self.fc0 = nn.Linear(10*61, 64) # bs * 64
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
        self.conv1 = nn.Conv1d(5, 10, 4) # bs * 10 * 60+1
        self.fc0 = nn.Linear(10*61, 64) # bs * 64
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
        super(TransformerNet_1_0_0, self).__init__() # bs * 5 * 64
        self.encoder0 = nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=512)
        self.fc0 = nn.Linear(64 * 5, 5)

    def forward(self, x):
        x = self.encoder0(x)
        x = x.view(-1, num_flat_features(x))
        x = self.fc0(x)
        return x

class TransformerNet_1_0_1(AbstractMidiNet):

    def __init__(self):
        super(TransformerNet_1_0_1, self).__init__() # bs * 5 * 64
        self.encoder0 = nn.TransformerEncoderLayer(d_model=64, nhead=8)
        self.fc0 = nn.Linear(64 * 5, 5)

    def forward(self, x):
        x = self.encoder0(x)
        x = x.view(-1, num_flat_features(x))
        x = self.fc0(x)
        return x

class TransformerNet_1_0_2(AbstractMidiNet):

    def __init__(self,):
        super(TransformerNet_1_0_2, self).__init__() # bs * 5 * 32
        self.encoder0 = nn.TransformerEncoderLayer(d_model=64, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder0, num_layers=3)
        self.fc0 = nn.Linear(64 * 5, 5)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, num_flat_features(x))
        x = self.fc0(x)
        return x




