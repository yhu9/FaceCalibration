
import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class STNkd(nn.Module):
    def __init__(self, k=128):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k*k)
        self.relu = nn.ReLU()

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True,k=128):
        super(PointNetfeat, self).__init__()
        self.stn = STNkd(k=k)
        self.conv1 = torch.nn.Conv1d(k, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.global_feat = global_feat

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        pointfeat = x
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 128)
        return x, trans

class PointNet(nn.Module):
    def __init__(self, k=128, feature_transform=False):
        super(PointNet, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True,k=k)

    def forward(self, x):
        x, trans_feat = self.feat(x)
        return x

if __name__ == '__main__':

    m = PointNet(k=128)
    x = torch.randn(10,128,1000)
    out = m(x)
    print(out.shape)


