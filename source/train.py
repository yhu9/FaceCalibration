
import itertools

import torch

import losses
import dataloader
from model2 import PointNet
from optimizer import Optimizer

def train(device='cuda'):

    # create our dataloader
    data = dataloader.Data(face3dmm='lm')

    # mean shape and eigenvectors for 3dmm
    # data3dmm = dataloader.SyntheticLoader()
    loader = dataloader.SyntheticLoader()
    center = torch.tensor([loader.w/2,loader.h/2,1])

    # optimizer
    optim = Optimizer(center,gt=None)
    optim.to_cuda()

    optim.sfm_opt = torch.optim.Adam(optim.sfm_net.parameters(),lr=1e-4)
    optim.calib_opt = torch.optim.Adam(optim.calib_net.parameters(),lr=1e-3)

    # start training
    for epoch in itertools.count():
        for i in range(2000):
            batch = loader[i]
            optim.sfm_opt.zero_grad()
            optim.calib_opt.zero_grad()
            x = batch['x_img'].float()
            fgt = batch['f_gt'].float()
            shape_gt = batch['x_w_gt'].float()

            # forward prediction
            K = optim.predict_intrinsic(x)
            S = optim.get_shape(x)

            # compute error and step
            f_error = torch.abs(K.mean(0)[0,0] - fgt) / fgt
            s_error = losses.compute_reprojection_error(x.permute(0,2,1),S,K,show=False)

            # s_error = torch.mean(torch.pow(S - shape_gt,2).sum(1))
            loss = f_error + s_error
            loss.backward()
            optim.sfm_opt.step()
            optim.calib_opt.step()

            print(f"epoch: {epoch} | iter: {i} | f_error: {f_error.item():.3f} | f/fgt: {K.mean(0)[0,0].item():.2f}/{fgt.item():.2f} | S_err: {s_error.item():.3f} ")

        optim.save(f"{epoch:02d}_")

if __name__ == '__main__':

    train()

