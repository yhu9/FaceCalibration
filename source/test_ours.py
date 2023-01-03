
import itertools
import argparse
import os
import sys

from numpy.core.records import fromarrays
import scipy.io as sio
import torch
import numpy as np

import dataloader

import util
import time
import losses
from optimizer import Optimizer

#import BiwiLoader
#import BiwiidLoader
#import Cad120Loader

def getLoader(db):
    if db == 'syn':
        loader = dataloader.TestLoader(f_test)
    elif db == 'human36':
        loader = dataloader.Human36Loader()
    elif db == 'cad120':
        loader = dataloader.Cad120Loader()
    elif db == 'biwi':
        loader = dataloader.BIWILoader()
    elif db == 'biwiid':
        loader = dataloader.BIWIIDLoader()
    elif db == 'checkerboard':
        loader = dataloader.CheckerboardLoader()

    return loader

##############################################################################################
##############################################################################################
##############################################################################################

class DataRecorder():
    def __init__(self):

        self.data = {}
        self.out_dir = os.path.join('results','analysis')

    def add_item(self,name,val):
        if not name in self.data.keys():
            self.data[name] = []

        if torch.is_tensor(val):
            val = val.detach().cpu().numpy()
        self.data[name].append(val)

    def save_data(self,out_dir='out'):
        out_data = {}
        for key,val in self.data.items():
            out_data[key] = np.stack(self.data[key],axis=-1)

        out_file = os.path.join(self.out_dir,'bias_analysis.mat')
        sio.savemat(out_file,out_data)

##############################################################################################
##############################################################################################
##############################################################################################
# test on dataset
def test(args):
    outfile = args.out
    
    # data to save from testing
    data = []
    errors = {'error_2d': [], 'error_3d': [], 'error_reld': [], 'error_relf': [],
            'error_px': [], 'error_py': []}

    # set random seed for reproducibility of test set
    f_vals = [i*100 for i in range(5,15)]
    np.random.seed(0)
    torch.manual_seed(0)
    for f_test in f_vals:
        # create dataloader
        loader = dataloader.TestLoader(f_test)

        k = 0
        while True:
            sample = loader[k]

            # load the data
            shape_gt = sample['x_w_gt']
            fgt = sample['f_gt']
            x = sample['x_img'].permute(0,2,1)[::5]
            K_gt = sample['K']

            # make prediction without any optimization
            b = x.shape[0]
            _,R,T = util.EPnP_(x.permute(0,2,1),
                    shape_gt.unsqueeze(0).repeat(b,1,1),K_gt.unsqueeze(0).repeat(b,1,1))
            dgt = torch.norm(T,dim=1)

            # ground truth values
            gt = {}
            gt['f'] = fgt
            gt['S'] = shape_gt

            # perform prediction
            center = torch.tensor([loader.w/2,loader.h/2,1])
            optim = Optimizer(center,gt=gt)
            optim.load('00_')

            if args.opt:
                if args.opt == 'AO':
                    optim.sfm_opt.param_groups[0]['lr'] = 6
                    try:
                        S, K, R, T = optim.dualoptimization(x,max_iter=10)
                    except:
                        print("OOPS! SOMETHING WENT WRONG! PROLLY PYTORCH'S SVD GOT A NAN SOMEHOW UHHHH")
                        sys.exit(1)
                elif args.opt == 'JO':
                    S,K,R,T = optim.jointoptimization(x,max_iter=100)
                elif args.opt == 'SO':
                    S,K,R,T = optim.sequentialoptimization(x)
                else:
                    S, K, R, T = optim.dualoptimization(x)
            else:
                K = optim.predict_intrinsic(x)
                S = optim.get_shape(x)
                #S = optim.predict_shape(x)
                Xc, R, T = util.EPnP_(x.permute(0,2,1),S,K)

            # get predicted intrinsics
            f = torch.mean(K[:,0,0])
            px = torch.mean(K[:,0,2])
            py = torch.mean(K[:,1,2])

            reproj_error = torch.median(losses.getError(x,S,R,T,K,show=False,loss='l2'))
            reconstruction_error = torch.norm(shape_gt.unsqueeze(0) - S,dim=2).mean()

            d = torch.norm(T,dim=1)
            d_error = util.getDepthError(d,dgt)
            rel_error = torch.median(d_error)
            f_error = torch.median(torch.abs(fgt - f) / fgt)
            px_error = torch.abs(px - K_gt[0,2]) / K_gt[0,2]
            py_error = torch.abs(py - K_gt[1,2]) / K_gt[1,2]

            print(f"f/sequence: {f_test}/{k}  | f/fgt: {f.mean().item():.3f}/{fgt.item():.3f} |  f_error_rel: {f_error.item():.4f}  | rmse: {reconstruction_error.item():.4f}  | rel rmse: {rel_error.item():.4f}    | 2d error: {reproj_error.item():.4f}")

            sample = {}
            sample['x'] = x.detach().cpu().numpy()
            sample['K'] = K.detach().cpu().numpy()
            sample['K_gt'] = K_gt.detach().cpu().numpy()
            sample['d'] = d.detach().cpu().numpy()
            sample['d_gt'] = dgt.cpu().numpy()
            sample['shape_pred'] = S.detach().cpu().numpy()
            sample['shape_gt'] = shape_gt.detach().cpu().numpy()
            sample['error_2d'] = reproj_error.cpu().data.item()
            sample['error_reld'] = rel_error.cpu().data.item()
            sample['error_relf'] = f_error.cpu().data.item()
            # sample['filename'] = sample['fname']
            data.append(sample)

            errors['error_2d'].append(reproj_error.detach().cpu().item())
            errors['error_3d'].append(reconstruction_error.detach().cpu().item())
            errors['error_reld'].append(rel_error.detach().cpu().item())
            errors['error_relf'].append(f_error.detach().cpu().item())
            errors['error_px'].append(px_error.cpu().data.item())
            errors['error_py'].append(py_error.cpu().data.item())
            k = k + 1

            if k == 5: break

    print(f"MEAN seterror_2d: {np.median(errors['error_2d'])}")
    print(f"MEAN seterror_3d: {np.median(errors['error_3d'])}")
    print(f"MEAN seterror_reld: {np.median(errors['error_reld'])}")
    print(f"MEAN seterror_relf: {np.median(errors['error_relf'])}")
    print(f"Mean error px: {np.median(errors['error_px'])}")
    print(f"Mean error py: {np.median(errors['error_py'])}")


    # prepare output file
    sio.savemat(outfile,{'data': data})

##############################################################################################
##############################################################################################
# testing on different datasets
def testReal(args):
    modelin = args.model
    outfile = args.out
    db = args.db

    # data to save from testing
    data = []
    errors = {'error_2d': [], 'error_3d': [], 'error_reld': [], 'error_relf': [],
            'error_px': [], 'error_py': []}

    # define loader
    loader = getLoader(db)
    #loader = OriginalLoader.get_loader(db)
    #loader = get_loader(db)

    # get 3D shape
    np.random.seed(0)
    torch.manual_seed(0)
    for sub in range(len(loader)):
        batch = loader[sub]
        x = batch['x_img']
        K_gt = batch['K_gt']
        dgt = batch['d'].squeeze()
        fgt = K_gt[0,0]
        b = x.shape[0]

        # ground truth values
        gt = {}
        gt['f'] = K_gt[0,0]

        # create our optimizer
        center = torch.tensor([loader.w/2,loader.h/2,1])
        optim = Optimizer(center,gt=gt)
        optim.load('00_')

        # optimize 3D shape
        # pred = optim.opt_shape_dlt()

        # perform prediction
        if args.opt:
            if args.opt == 'AO':
                S, K, R, T = optim.dualoptimization(x,max_iter=5)
            elif args.opt == 'JO':
                S,K,R,T = optim.jointoptimization(x,max_iter=30)
            elif args.opt == 'SO':
                S,K,R,T = optim.seqopitmization(x)
            else:
                S, K, R, T = optim.dualoptimization(x)
        else:
            K = optim.predict_intrinsic(x)
            S = optim.get_shape(x)
            #R, T = optim.get_pose(x,S,K)
            Xc, R, T = util.EPnP_(x.permute(0,2,1),S,K)

        # get predicted intrinsics
        f = torch.mean(K[:,0,0])
        px = torch.mean(K[:,0,2])
        py = torch.mean(K[:,1,2])

        # get errors
        #reproj_errors2 = util.getError(ptsI,shape,R,T,K)
        reproj_errors2 = losses.getError(x,S,R,T,K,show=False,loss='l2')
        #reproj_errors2 = util.getReprojError2(x,shape,R,T,K,show=False,loss='l2')
        #rel_errors = util.getRelReprojError3(d_gt,S,R,T)
        d = torch.norm(T,dim=1)
        d_error = util.getDepthError(d,dgt)

        reproj_error = reproj_errors2.mean()
        rel_error = torch.median(d_error)
        f_error = torch.mean(torch.abs(fgt - f) / fgt)
        px_error = torch.abs(px - K_gt[0,2]) / K_gt[0,2]
        py_error = torch.abs(py - K_gt[1,2]) / K_gt[1,2]

        # save final prediction
        sample = {}
        sample['x'] = x.detach().cpu().numpy()
        sample['K'] = K.detach().cpu().numpy()
        sample['K_gt'] = K_gt.detach().cpu().numpy()
        sample['d'] = d.detach().cpu().numpy()
        sample['d_gt'] = dgt.cpu().numpy()
        sample['shape_pred'] = S.detach().cpu().numpy()
        sample['error_2d'] = reproj_error.cpu().data.item()
        sample['error_reld'] = rel_error.cpu().data.item()
        sample['error_relf'] = f_error.cpu().data.item()
        if 'shape' in batch.keys():
            sgt = batch['shape'].detach().cpu()
            s = S.mean(0).detach().cpu()
            rmse = util.getShapeError(s,sgt)
            sample['shape'] = sgt.numpy()
            errors['error_3d'].append(rmse.data.item())
            print(f"rmse: {rmse:.3f}")
        errors['error_2d'].append(reproj_error.cpu().data.item())
        errors['error_reld'].append(rel_error.cpu().data.item())
        errors['error_relf'].append(f_error.cpu().data.item())
        errors['error_px'].append(px_error.cpu().data.item())
        errors['error_py'].append(py_error.cpu().data.item())
        data.append(sample)

        f_x = torch.mean(f.detach()).cpu().item()
        print(f" f/fgt: {f_x:.3f}/{fgt.item():.3f} |  f_error_rel: {f_error.item():.4f}  | rel_d: {rel_error.item():.4f}    | 2d error: {reproj_error.item():.4f}")

        #end for

    print(f"MEAN seterror_2d: {np.median(errors['error_2d'])}")
    print(f"MEAN seterror_3d: {np.median(errors['error_3d'])}")
    print(f"MEAN seterror_reld: {np.median(errors['error_reld'])}")
    print(f"MEAN seterror_relf: {np.median(errors['error_relf'])}")
    print(f"Mean error px: {np.median(errors['error_px'])}")
    print(f"Mean error py: {np.median(errors['error_py'])}")
    # prepare output file
    sio.savemat(outfile,{'data': data})


####################################################################################3
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="training arguments")
    parser.add_argument("--ckpt", default=False)
    parser.add_argument("--out",default="results/exp.mat")
    parser.add_argument("--db", default="syn")
    parser.add_argument("--device",default='cpu')
    parser.add_argument("--opt", default=False)
    parser.add_argument("--ft",default=False, action="store_true")
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    if args.db == 'syn':
        test(args)
    else:
        testReal(args)



