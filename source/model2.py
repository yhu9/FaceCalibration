import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from pointnet2d import PointNet as PNet2D
from pointnet3d import PointNet as PNet3D
from pointnetkd import PointNet as PNetkD

class DeltaEstimator(nn.Module):

    def __init__(self):
        super(DeltaEstimator,self).__init__()

        self.pnet2d = PNet2D()
        self.pnet128d = PNetkD(k=128)

        self.fc1 = nn.Linear(128,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,1)

    def forward(self,x):

        x = self.pnet2d(x)
        x = x.T.unsqueeze(0)
        x = self.pnet128d(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x

class FEstimator(nn.Module):

    def __init__(self):
        super(FEstimator,self).__init__()

        self.pnet2d = PNet2D()
        self.pnet128d = PNetkD(k=128)
        self.fc1 = nn.Linear(129,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,1)

    def forward(self,x,f):

        x = self.pnet2d(x)
        x = x.T.unsqueeze(0)
        x = self.pnet128d(x)

        x = torch.cat((x,f),dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(2, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,1]).astype(np.float32))).view(1,4).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 2, 2)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
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
        x = x.view(-1, 1024)

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
    def __init__(self, global_feat=True, feature_transform = False,d=2):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(d, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            #return x, trans, trans_feat
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNet(nn.Module):
    def __init__(self, n=2, feature_transform=False):
        super(PointNet, self).__init__()
        self.feature_transform = feature_transform
        self.n = n
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n)
        #self.bn1 = nn.BatchNorm1d(512)
        #self.bn2 = nn.BatchNorm1d(256)
        #self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        #x = F.relu(self.bn1(self.fc1(x)))
        #x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_finetune(self,x):
        x, trans,trans_feat = self.feat(x)

        return x

class FinalLayer(nn.Module):
    def __init__(self,n=2):
        super(FinalLayer, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

class PointNetSmall(nn.Module):
    def __init__(self, n=2, feature_transform=False):
        super(PointNetSmall, self).__init__()
        self.feature_transform = feature_transform
        self.n = n
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, n)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = torch.clamp(torch.sigmoid(self.fc1(x)),1e-4,1.0)
        return x

if __name__ == '__main__':

    m = FEstimator()
    x = torch.randn((80,2,68))
    f = torch.randn((1,1)) * 1000 + 500
    p = m(x,f)

    print(p.shape)

    m = DeltaEstimator()
    x = torch.randn((70,2,68))
    y = m(x)
    print(y.shape)

    quit()

    m = PointNet(n=1)
    x = torch.randn((1,200))
    l1 = torch.nn.Linear(200,128)
    quit()

    sim_data = Variable(torch.rand(32,3,2500))
    #sim_data = Variable(torch.rand(32,2,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNet(k=5)
    out, _, _ = cls(sim_data)
    print('input size ',sim_data.shape)
    print('class', out.size())
