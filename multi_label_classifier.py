from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import dataset_processing
import torch.optim as optim
import torch

DATA_PATH        = 'data'
TRAIN_DATA       = 'train_img'
TEST_DATA        = 'test_img'
TRAIN_IMG_FILE   = 'train_img.txt'
TEST_IMG_FILE    = 'test_img.txt'
TRAIN_LABEL_FILE = 'train_label.txt'
TEST_LABEL_FILE  = 'test_label.txt'

NLABELS = 5
batch_size = 16

transformations = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
dset_train = dataset_processing.DatasetProcessing(
    DATA_PATH, TRAIN_DATA, TRAIN_IMG_FILE, TRAIN_LABEL_FILE, transformations)

dset_test = dataset_processing.DatasetProcessing(
    DATA_PATH, TEST_DATA, TEST_IMG_FILE, TEST_LABEL_FILE, transformations)

train_loader = DataLoader(dset_train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0
                         )

test_loader = DataLoader(dset_test,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=0
                         )

class MultiLabelNN(nn.Module):
    def __init__(self, nlabel):
        super(MultiLabelNN, self).__init__()
        self.nlabel = nlabel
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, 5, 1, 1),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 16, 5, 1, 1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(46656, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.nlabel)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        res = x.view(x.size(0), -1)
        out = self.dense(res)
        return out

use_gpu = torch.cuda.is_available()
model = MultiLabelNN(NLABELS)
if use_gpu:
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MultiLabelMarginLoss()

epochs = 10
for epoch in range(epochs):
    ### training phase
    total_training_loss = 0.0
    # total = 0.0
    model.train()
    for iter, traindata in enumerate(train_loader, 0):
        train_inputs, train_labels = traindata
        if use_gpu:
            train_inputs, train_labels = train_inputs.cuda(), train_labels.cuda()

        train_outputs = model(train_inputs)
        loss = criterion(train_outputs, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # total += train_labels.size(0)
        total_training_loss += loss.item()
        print('Training Phase: Epoch: [%2d][%2d/%2d]\tIteration Loss: %.3f' %
              (iter, epoch+1, epochs, loss.item() / train_labels.size(0)))
    ### testing phase
    model.eval()
    with torch.no_grad():
        for iter, testdata in enumerate(test_loader, 0):
            test_inputs, test_labels = testdata
            if use_gpu:
                test_inputs, test_labels = test_inputs.cuda(), test_labels.cuda()
    
            test_outputs = model(test_inputs)
            test_loss = criterion(test_outputs, test_labels)
            print('Testing Phase: Epoch: [%2d][%2d/%2d]\tIteration Loss: %.3f' %
                  (iter, epoch+1, epochs, test_loss.item() / test_labels.size(0)))
