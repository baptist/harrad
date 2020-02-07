import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2DNet(nn.Module):

    def __init__(self, num_classes, sample_size=(1, 30, 256)):

        super(CNN2DNet, self).__init__()


        time_pooling = np.zeros((4,), dtype=np.int32)
        time_pooling[:] = int((float(sample_size[1]) / 2) ** .25)
        res = ((float(sample_size[1])/2) ** .25) - time_pooling[0]
        for i in range(int(round(res * 4))):
            time_pooling[i] += 1

        self.temporal_size = sample_size[1]
        for i in range(len(time_pooling)):
            self.temporal_size //= time_pooling[i]

        self.spatial_size = sample_size[2] // 2 // 2 // 2 // 2

        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(in_channels=sample_size[0], out_channels=8, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            #nn.Dropout2d(0.1),
            nn.MaxPool2d((time_pooling[0], 2)),
            # Conv2
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Dropout2d(0.1),
            nn.MaxPool2d((time_pooling[1], 2)),
            # Conv3
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Dropout2d(0.1),
            nn.MaxPool2d((time_pooling[2], 2)),
            # Conv4
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Dropout2d(0.3),
        )  # --- Conv sequential ends ---

        self.pool = nn.MaxPool2d((time_pooling[3], 2))

        # --- Linear sequential starts ---
        self.classifier = nn.Sequential(
            nn.Linear(64 * self.spatial_size * self.temporal_size, 128),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )  # --- Linear sequential ends ---

        self.feature_size = self.spatial_size*self.temporal_size
        self.sample_size = sample_size

    def modify_input(self, x):
        x = x.contiguous().view(-1, *self.sample_size)
        return x

    def forward(self, x):

        x = self.modify_input(x)
        # Forward on Conv
        x = self.features(x)
        x = self.pool(x)
        # Flatten
        x = x.view(-1, 64 * self.spatial_size * self.temporal_size)
        # Forward on Linear
        x = self.classifier(x)

        return x


class CNN1DLSTMNet(nn.Module):

    def __init__(self, num_classes, input_channels=1, sample_size=(30, 256)):

        super(CNN1DLSTMNet, self).__init__()

        # Define all layers containing learnable weights.
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.spatial_size = sample_size[1] // 3 // 3 // 3

        self.fc1 = nn.Linear(32 * self.spatial_size, 64)

        self.lstm = nn.LSTM(64, 32, 1, batch_first=True, bidirectional=True)

        self.c1_dropout = nn.Dropout2d(0.1)
        self.c2_dropout = nn.Dropout2d(0.3)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.2)

        self.pool = nn.MaxPool1d(3)

        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.sample_size = sample_size

    def init_states(self, batch_size):
        return torch.zeros(batch_size, 2, 32).cuda(), torch.zeros(batch_size, 2, 32).cuda()

    def forward(self, x):

        hidden0, cell0 = self.init_states(x.size(0))

        x = x.contiguous().view(-1, 1, self.sample_size[1])

        x = self.pool(self.c1_dropout(F.elu(self.conv1(x))))
        x = self.pool(self.c1_dropout(F.elu(self.conv2(x))))
        x = self.pool(self.c1_dropout(F.elu(self.conv3(x))))

        x = x.view(-1, 32 * self.spatial_size)

        x = self.dropout2(F.elu(self.fc1(x)))
        x = x.view(-1, self.sample_size[0], 64)

        self.lstm.flatten_parameters()
        x, (hidden, cell) = self.lstm(x, (hidden0.permute(1, 0, 2).contiguous(), cell0.permute(1, 0, 2).contiguous()))
        x = hidden.permute(1, 0, 2).contiguous().view(-1, 64)
        x = self.dropout5(F.elu(self.fc2(x)))
        x = self.fc3(x)


        return x


class LSTMNet(nn.Module):

    def __init__(self, num_classes, sample_size=(30, 256)):

        super(LSTMNet, self).__init__()

        # Define all layers containing learnable weights.

        self.fc1 = nn.Linear(sample_size[1], 64)

        self.lstm = nn.LSTM(64, 32, 1, batch_first=True, bidirectional=True)

        self.c1_dropout = nn.Dropout2d(0.1)
        self.c2_dropout = nn.Dropout2d(0.3)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.sample_size = sample_size

    def init_states(self, batch_size):
        return torch.zeros(batch_size, 2, 32).cuda(), torch.zeros(batch_size, 2, 32).cuda()

    def forward(self, x):

        hidden0, cell0 = self.init_states(x.size(0))

        x = x.contiguous().view(-1, 1, self.sample_size[1])

        x = self.dropout2(F.elu(self.fc1(x)))

        x = x.view(-1, self.sample_size[0], 64)

        self.lstm.flatten_parameters()
        x, (hidden, cell) = self.lstm(x, (hidden0.permute(1, 0, 2).contiguous(), cell0.permute(1, 0, 2).contiguous()))
        x = hidden.permute(1, 0, 2).contiguous().view(-1, 64)
        x = self.dropout5(F.elu(self.fc2(x)))
        x = self.fc3(x)



        return x


class CNN2DLSTMNet(nn.Module):

    def __init__(self, num_classes, sample_size=(1, 30, 80, 128)):

        super(CNN2DLSTMNet, self).__init__()

        # Define all layers containing learnable weights.
        self.conv1 = nn.Conv2d(in_channels=sample_size[0], out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.range_size = sample_size[2] // 3 // 3 // 2
        self.doppler_size = sample_size[3] // 3 // 3 // 2

        self.fc1 = nn.Linear(32 * self.range_size * self.doppler_size, 64)

        self.lstm = nn.LSTM(64, 32, 1, batch_first=True, bidirectional=True)

        self.c1_dropout = nn.Dropout2d(0.1)
        self.c2_dropout = nn.Dropout2d(0.3)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.2)

        self.pool3 = nn.MaxPool2d(3)
        self.pool2 = nn.MaxPool2d(2)

        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.sample_size = sample_size

    def init_states(self, batch_size):
        return torch.zeros(batch_size, 2, 32).cuda(), torch.zeros(batch_size, 2, 32).cuda()

    def forward(self, x):

        hidden0, cell0 = self.init_states(x.size(0))

        x = x.contiguous().view(-1, 1, *self.sample_size[2:])

        x = self.pool3(self.c1_dropout(F.elu(self.conv1(x))))
        x = self.pool3(self.c1_dropout(F.elu(self.conv2(x))))
        x = self.pool2(self.c1_dropout(F.elu(self.conv3(x))))

        x = x.view(-1, 32 * self.range_size * self.doppler_size)

        x = self.dropout2(F.elu(self.fc1(x)))
        x = x.view(-1, self.sample_size[1], 64)

        self.lstm.flatten_parameters()
        x, (hidden, cell) = self.lstm(x, (hidden0.permute(1, 0, 2).contiguous(), cell0.permute(1, 0, 2).contiguous()))
        x = hidden.permute(1, 0, 2).contiguous().view(-1, 64)
        x = self.dropout5(F.elu(self.fc2(x)))
        x = self.fc3(x)

        return x


class CNN3DNet(nn.Module):

    def __init__(self, num_classes, sample_size=(1, 30, 80, 128)):

        super(CNN3DNet, self).__init__()

        self.num_classes = num_classes

        time_pooling = np.zeros((4,), dtype=np.int32)
        time_pooling[:] = int((float(sample_size[1]) / 2) ** .25)
        res = ((float(sample_size[1]) / 2) ** .25) - time_pooling[0]
        for i in range(int(round(res * 4))):
            time_pooling[i] += 1

        self.range_size = sample_size[2] // 2 // 2 // 2 // 2
        self.doppler_size = sample_size[3] // 2 // 2 // 2 // 2
        self.sample_size = sample_size

        self.temporal_size = sample_size[1]
        for i in range(len(time_pooling)):
            self.temporal_size //= time_pooling[i]

        self.feature_size = self.range_size * self.doppler_size * self.temporal_size

        # --- Conv sequential starts ---
        self.features = nn.Sequential(
            # Conv1
            nn.Conv3d(in_channels=sample_size[0], out_channels=8, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Dropout3d(0.1),
            nn.MaxPool3d((time_pooling[0], 2, 2)),
            # Conv2
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Dropout3d(0.1),
            nn.MaxPool3d((time_pooling[1], 2, 2)),
            # Conv3
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Dropout3d(0.3),
            nn.MaxPool3d((time_pooling[2], 2, 2)),
            # Conv4
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Dropout3d(0.3),

        )  # --- Conv sequential ends ---
        self.pool = nn.MaxPool3d((time_pooling[3], 2, 2))

        # --- Linear sequential starts ---
        self.classifier = nn.Sequential(
            nn.Linear(64 * self.feature_size, 128),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )  # --- Linear sequential ends ---



    def modify_input(self, x):
        x = x.contiguous().view(-1, *self.sample_size)
        return x

    def forward(self, x):
        x = self.modify_input(x)
        # Forward on Conv
        x = self.features(x)
        x = self.pool(x)
        # Flatten
        x = x.view(-1, 64 * self.range_size * self.doppler_size * self.temporal_size)
        # Forward on Linear
        x = self.classifier(x)

        return x