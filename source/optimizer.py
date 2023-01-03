
import os
import yaml

import util
import losses
import dataloader
import matplotlib.pyplot as plt
#from TDDFA import TDDFA

import torch
import torchvision
import numpy as np
from model2 import PointNet
import kornia as kn

import BPnP

# optimizer for our camera calibration
class Optimizer():
    def __init__(self,center, gt=None,sfm_net=None,calib_net=None):
        self.bpnp = BPnP.BPnP.apply

        # mean shape and eigenvectors for 3dmm
        self.visualize = False
        self.data3dmm = dataloader.SyntheticLoader()
        self.pbias = 300
        self.fbias = 300
        self.center = center
        self.gt = gt

        if not sfm_net and not calib_net:
            self.reset(3,199)
        else:
            self.sfm_net = sfm_net
            self.calib_net = calib_net

        self.model_path = os.path.join('model','test5_prev')
        self.delta = torch.zeros((68,3),requires_grad=True)
        #self.calib_net.eval()
        #self.sfm_net.eval()
        #self.delta_net.eval()

        #self.sfm_opt = torch.optim.Adam(self.sfm_net.parameters(),lr=5e-1)
        #self.calib_opt = torch.optim.Adam(self.calib_net.parameters(),lr=1e-4)
        #self.calib_opt = torch.optim.Adam(self.calib_net.parameters(),lr=7e-4)
        #self.delta_opt = torch.optim.Adam(self.delta_net.parameters(),lr=2e-4)
        self.opt = torch.optim.Adam(list(self.sfm_net.parameters()) + list(self.calib_net.parameters()),
                lr = 1e-3)
        self.sfm_opt = torch.optim.Adam(self.sfm_net.parameters(),lr=1e-3)
        self.calib_opt = torch.optim.Adam(self.calib_net.parameters(),lr=1e-2)


    def set_eval(self):
        self.sfm_net.eval()
        self.calib_net.eval()

    def set_train(self):
        self.sfm_net.train()
        self.calib_net.train()

    # set visualization flag
    def set_visualize(self):
        self.visualize = True
        self.log = {}

    # show plot of optimization on synthetic data
    def show_visualization_syn(self):
        fig,ax = plt.subplots(3,2,figsize=(12,12))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        num_iter = len(self.log['f'])
        ax[0,0].plot(np.array(self.log['e2d']))
        ax[0,0].set_ylabel('Reprojection Error',fontsize=16)
        ax[0,1].plot(np.array(self.log['f']),label='prediction')
        ax[0,1].set_ylabel('Focal Length',fontsize=16)
        ax[1,0].plot(np.array(self.log['px']),label='prediction')
        ax[1,0].set_ylabel('X Principal Point',fontsize=16)
        ax[1,1].plot(np.array(self.log['py']),label='prediction')
        ax[1,1].set_ylabel('Y Principal Point',fontsize=16)
        ax[2,0].plot(np.array(self.log['derr']))
        ax[2,0].set_ylabel('Depth Error',fontsize=16)
        ax[2,1].plot(np.array(self.log['rmse']))
        ax[2,1].set_ylabel('Shape Error',fontsize=16)

        if 'f' in self.gt:
            ax[0,1].plot(np.array(list(range(num_iter))),np.ones(num_iter)*self.gt['f'].item(),
                    label='gt')
        if 'px' in self.gt:
            ax[1,0].plot(np.array(list(range(num_iter))),np.ones(num_iter)*self.gt['px'].item(),
                    label='gt')
        if 'py' in self.gt:
            ax[1,1].plot(np.array(list(range(num_iter))),np.ones(num_iter)*self.gt['py'].item(),
                    label='gt')
        ax[0,1].legend()
        ax[1,0].legend()
        ax[1,1].legend()
        for i in range(3):
            for j in range(2):
                ax[i,j].grid(visible=True)
                ax[i,j].set_xlabel('Iteration',fontsize=16)
                ax[i,j].tick_params(axis='both',which='both',labelsize=14)
        plt.tight_layout()
        fig.subplots_adjust(hspace=.25)
        plt.show()

    # show plot of optimization iteration
    def show_visualization(self):
        fig,ax = plt.subplots(2,2,figsize=(12,12))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        num_iter = len(self.log['f'])
        ax[0,0].plot(np.array(self.log['f']),label='prediction')
        ax[0,0].set_ylabel('Focal Length',fontsize=18)
        ax[0,1].plot(np.array(self.log['derr']),label='prediction')
        ax[0,1].set_ylabel('Depth Error',fontsize=18)
        ax[1,0].plot(np.array(self.log['px']),label='prediction')
        ax[1,0].set_ylabel('X Principal Point',fontsize=18)
        ax[1,1].plot(np.array(self.log['py']),label='prediction')
        ax[1,1].set_ylabel('Y Principal Point',fontsize=18)
        ax[0,0].grid(visible=True)
        ax[0,1].grid(visible=True)
        ax[1,0].grid(visible=True)
        ax[1,1].grid(visible=True)

        if 'f' in self.gt:
            ax[0,0].plot(np.array(list(range(num_iter))),np.ones(num_iter)*self.gt['f'].item(),
                    label='gt')
        if 'px' in self.gt:
            ax[1,0].plot(np.array(list(range(num_iter))),np.ones(num_iter)*self.gt['px'].item(),
                    label='gt')
        if 'py' in self.gt:
            ax[1,1].plot(np.array(list(range(num_iter))),np.ones(num_iter)*self.gt['py'].item(),
                    label='gt')
        ax[0,0].legend()
        ax[1,0].legend()
        ax[1,1].legend()
        for i in range(2):
            for j in range(2):
                ax[i,j].set_xlabel('Iteration',fontsize=18)
                ax[i,j].tick_params(axis='both',which='both',labelsize=14)
        plt.tight_layout()
        fig.subplots_adjust(hspace=.2)
        plt.show()

    # visualize optimizer
    def log_err(self,pred):
        for key,val in pred.items():
            if torch.is_tensor(val): val = torch.mean(val).detach().cpu().numpy()
            if not key in self.log: self.log[key] = []
            self.log[key].append(val)

        if 'd' in self.gt and 'dpred' in self.gt:
            if not 'derr' in self.log: self.log['derr'] = []
            d_error = util.getDepthError(self.gt['dpred'],self.gt['d']).mean()
            self.log['derr'].append(d_error.detach().cpu().item())
        if 'S' in self.gt and 'S_pred' in self.gt:
            if not 'rmse' in self.log: self.log['rmse'] = []
            if not 's_pred' in self.log: self.log['s_pred'] = []
            rmse = util.getShapeError(self.gt['S_pred'],self.gt['S'])
            self.log['s_pred'].append(self.gt['S_pred'].detach().cpu().numpy())
            self.log['rmse'].append(rmse.detach().cpu().item())

    # console logger
    def console_log(self,pred):
        out_str = ""
        for key,val in pred.items():
            if torch.is_tensor(val): val = torch.mean(val).detach().cpu().numpy()
            out_str += f"{key}: {val:.3f} | "
        out_str += self.gt_log()
        print(out_str)
        return out_str

    # create logging text
    def gt_log(self):
        out_str = ""
        keys = self.gt.keys()
        if self.gt:
            if 'f' in keys and 'f_pred' in keys:
                fgt = self.gt['f'].detach().cpu().item()
                fpred = self.gt['f_pred'].detach().cpu().item()
                ferr = np.abs(fgt - fpred) / fgt
                out_str += f"fgt/f/err: {fgt:.2f}/{fpred:2.f}/{ferr:.3f} | "
            if 'S_pred' in keys and 'S' in keys:
                S = self.gt['S']
                S_pred = self.gt['S_pred']
                S_err = torch.norm(S.unsqueeze(0) - S_pred,dim=2).mean().detach().cpu().item()
                out_str += f"S_err: {S_err:3f} | "
                # torch.norm(S - S_pred)
        return out_str

    # optimize the shape parameters given K
    def opt_shape(self,ptsI,K,max_iter=5,log=True):
        b = ptsI.shape[0]
        for i in range(max_iter):
            self.sfm_opt.zero_grad()
            # self.delta_opt.zero_grad()

            # predict S
            S = self.get_shape(ptsI)
            #S = self.get_shape(ptsI).mean(0).unsqueeze(0).repeat(b,1,1)

            # differentiable PnP pose estimation
            Xc, R, T = util.EPnP_(ptsI.permute(0,2,1),S,K)

            # Get Error
            error2d = losses.getError(ptsI,S,R,T,K,show=False,loss='l2').mean()
            #error2d = torch.log(losses.getXcError(ptsI,Xc,K))
            #error2d = losses.getXcError(ptsI,Xc,K)
            error_p = losses.compute_principal_error(K,self.center.unsqueeze(0)).mean()
            error_motion = torch.log(losses.motionError(Xc.mean(1))).mean()

            # apply loss
            loss = error2d
            loss.backward()
            self.sfm_opt.step()

            # log results
            f = torch.mean(K[:,0,0])
            px = torch.mean(K[:,0,2])
            py = torch.mean(K[:,1,2])
            pred = {'iter': i, 'error': loss, 'e_pr': error_p,'e_m': error_motion,'f': f,
                    'e2d': error2d,'px': px, 'py': py}
            self.gt['S_pred'] = S.mean(0).detach().cpu()
            self.gt['dpred'] = torch.norm(T,dim=1)
            self.console_log(pred)

            # log results
            if self.visualize:
                self.log_err(pred)

        pred['S'] = S.detach()
        return pred

    # optimize the shape parameters given K
    def opt_shape_mu(self,ptsI,K,max_iter=5,log=True):
        b = ptsI.shape[0]
        n = ptsI.shape[2]
        mu_s = self.get_mean_shape().detach()
        for i in range(max_iter):
            # self.sfm_opt.zero_grad()
            self.delta_opt.zero_grad()

            # predict S
            #S = self.get_shape(ptsI)
            #S = self.get_shape(ptsI).mean(0).unsqueeze(0).repeat(b,1,1)
            S = mu_s + self.delta_net(ptsI).view(b,n,3)

            # differentiable PnP pose estimation
            Xc, R, T = util.EPnP_(ptsI.permute(0,2,1),S,K)

            # error2d
            error2d = losses.getError(ptsI,S,R,T,K,show=False,loss='l2')

            # apply loss
            loss = error2d
            loss.backward()
            # self.sfm_opt.step()
            self.delta_opt.step()
            f = torch.mean(K[:,0,0])
            px = torch.mean(K[:,0,2])
            py = torch.mean(K[:,1,2])

            # log results
            pred = {'iter': i, 'error': loss, 'f': f, 'e2d': error2d, 'px': px, 'py': py}
            self.gt['S_pred'] = S
            print(self.console_log(pred))

        pred['S'] = S.detach()
        return pred

    # optimize the shape using direct linear transform
    def opt_shape_dlt(self,ptsI,max_iter=100):
        b = ptsI.shape[0]
        for i in range(max_iter):
            S = self.get_shape(ptsI).mean(0).unsqueeze(0).repeat(b,1,1)

            # direct linear transform
            P = util.DLT(ptsI,S)
            Xc = torch.cat([S,torch.ones((b,68,1)).to(ptsI.device)],dim=2).permute(0,2,1)

            # get error
            error2d = losses.getXcError(ptsI,Xc,P)

            # get motion error
            # error_motion = torch.log10(losses.motionError(Xc))

            # apply loss
            loss = error2d
            loss.backward()
            self.sfm_opt.step()

            # log results
            if self.gt:
                self.gt['S_pred'] = S
            pred = {'iter': i, 'error': loss}
            self.console_log(pred)

        pred['S'] = S.detach()
        return pred

    # optimize the K parameters given S
    # given single 3D shape S estimate 3D model
    def opt_calib(self,ptsI,S,max_iter=5,still=False):
        b = ptsI.shape[0]
        for i in range(max_iter):
            self.calib_opt.zero_grad()

            # predict K
            K = self.predict_intrinsic(ptsI).mean(0).repeat(b,1,1)

            # pose estimation
            Xc, R, T = util.EPnP_(ptsI.permute(0,2,1),S,K)

            # get Error
            #error2d = torch.log(losses.getXcError(ptsI,Xc,K))
            error2d = torch.log(losses.getError(ptsI,S,R,T,K,show=False,loss='l2').mean())
            e2d = losses.getError(ptsI,S,R,T,K,show=False,loss='l2').mean()
            #error2d = util.getReprojError2(ptsI,S,R,T,K.mean(0),show=False,loss='l2')
            if still:
                error_motion = torch.log(losses.motionError(Xc.mean(1))).mean()*1.0
            else:
                error_motion = torch.log(losses.motionError(Xc.mean(1))).mean()*1e-2
            #error_motion = torch.log(losses.motionError(T).mean())
            error_p = losses.compute_principal_error(K,self.center.unsqueeze(0)).mean()

            # apply loss
            loss = error2d + error_p*1e-2
            #loss = error2d + error_motion + error_p*1e-2
            loss.backward()
            self.calib_opt.step()

            # log results
            f = torch.mean(K[:,0,0])
            px = torch.mean(K[:,0,2])
            py = torch.mean(K[:,1,2])
            pred = {'iter': i, 'error': loss, 'e_pr': error_p,'e_m': error_motion,'f': f,
                    'e2d': error2d,'px': px, 'py': py,'e2d': e2d}
            self.gt['S_pred'] = S.mean(0).detach().cpu()
            self.gt['dpred'] = torch.norm(T,dim=1)
            self.console_log(pred)

            # visualize optimization
            if self.visualize:
                self.log_err(pred)

        pred['K'] = K.detach()
        return pred

    # simultaneously optimize both K and S
    def opt_both(self,ptsI,max_iter=100,log=True,gt_data=None):
        if gt_data: gt = {}
        for i in range(max_iter):
            self.opt.zero_grad()

            # predict K
            K = self.predict_intrinsic(ptsI)

            # predict S
            S = self.get_shape(ptsI).mean(0)

            # get 2D reprojection error
            error2d = losses.getError(ptsI,S,R,T,K,show=False,loss='l2').mean()
            #error2d = util.getReprojError2(ptsI,shape,R,T,K,show=True,loss='l2')

            # get principal error
            error_p = losses.compute_principal_error(K,self.center.unsqueeze(0)).mean()

            # apply loss
            loss = error2d + error_p*1e-5
            loss.backward()
            self.opt.step()

            # log results
            pred = {'iter': i, 'error': loss, 'f': K[0,0], 'e2d': error2d, 'px': K[0,2], 'py': K[1,2]}
            #pred = {'iter': i, 'error': loss, 'f': K[:,0,0], 'e2d': error2d, 'px': K[:,0,2],
            #        'py': K[:,1,2]}
            self.console_log(pred,gt)

        return pred

    # perform Alternating optimization AO
    def dualoptimization(self, x, max_iter=5):
        # get initial shape and intrinsics
        b = x.shape[0]
        K = self.predict_intrinsic(x).detach()
        S = self.get_shape(x).detach()
        pred = {'S': S, 'K': K}

        still = True if self.get_initial_motion(x) < 50 else False
        print(self.get_initial_motion(x))
        print(still)

        #create param optimizer
        best = 100000
        for i in range(max_iter):
            pred = self.opt_calib(x, pred['S'].mean(0).unsqueeze(0).repeat(b,1,1),still=still,max_iter=5)
            pred = self.opt_shape(x, pred['K'].mean(0).unsqueeze(0).repeat(b,1,1),max_iter=5)
            loss = pred['error']

            #if ((torch.abs(best - loss) <= 0.01 or best < loss) and i >= 5) or i == max_iter: break
            if i == max_iter: break
            if loss < best: best = loss

        # show optimization curve
        if self.visualize:
            if 'S' in self.gt:
                self.show_visualization_syn()
            else:
                self.show_visualization()

        # get final prediction
        S = self.get_shape(x).mean(0).unsqueeze(0).repeat(b,1,1)
        K = self.predict_intrinsic(x).mean(0).unsqueeze(0).repeat(b,1,1)
        Xc, R, T = util.EPnP_(x.permute(0,2,1),S,K)
        return S, K, R, T

    # perform Joint optimization JO
    def jointoptimization(self,x,max_iter=100,still=False):
        b = x.shape[0]

        for i in range(max_iter):
            self.calib_opt.zero_grad()
            self.sfm_opt.zero_grad()

            K = self.predict_intrinsic(x)
            S = self.get_shape(x)

            # pose estimation
            Xc,R,T = util.EPnP_(x.permute(0,2,1),S,K)

            if still:
                error_motion = torch.log10(losses.motionError(T))*2
            else:
                error_motion = torch.log10(losses.motionError(T))*.2

            # get error
            error_p = losses.compute_principal_error(K,self.center.unsqueeze(0)).mean()
            error2d = torch.pow(torch.log10(losses.getXcError(x,Xc,K)),5)

            # apply loss
            # loss = error2d + torch.mean(error_motion) + error_p*1e-2
            loss = error2d + error_p*1e-2
            loss.backward()
            self.calib_opt.step()
            self.sfm_opt.step()
            f = torch.mean(K[:,0,0])
            px = torch.mean(K[:,0,2])
            py = torch.mean(K[:,1,2])

            # log results
            pred = {'iter': i, 'error': loss, 'e_pr': error_p,'e_m': error_motion,'f': f,
                    'e2d': error2d,'px': px, 'py': py}
            self.console_log(pred)

        S = self.get_shape(x)
        K = self.predict_intrinsic(x)
        Xc,R,T = util.EPnP_(x.permute(0,2,1),S,K)
        return S, K, R, T

    # perform Joint optimization JO
    def sequentialoptimization(self,x,max_iter=100,still=False):
        b = x.shape[0]
        K = self.predict_intrinsic(x)

        # optimize 3D shape
        pred = self.opt_shape_dlt(x,max_iter=max_iter)
        S = pred['S']

        # optimize intrinsics
        for i in range(max_iter):
            self.calib_opt.zero_grad()

            K = self.predict_intrinsic(x)

            # pose estimation
            Xc,R,T = util.EPnP_(x.permute(0,2,1),S,K)
            error2d = torch.log(losses.getXcError(x,Xc,K))

            if still:
                error_motion = torch.log(losses.motionError(Xc))*2
            else:
                error_motion = torch.log(losses.motionError(Xc))*.2

            # get principal error
            error_p = losses.compute_principal_error(K,self.center.unsqueeze(0)).mean()

            # apply loss
            loss = error2d + error_motion + error_p*1e-2
            loss.backward()
            self.calib_opt.step()
            f = torch.mean(K[:,0,0])
            px = torch.mean(K[:,0,2])
            py = torch.mean(K[:,1,2])

            # log results
            pred = {'iter': i, 'error': loss, 'e_pr': error_p,'e_m': error_motion,'f': f,
                    'e2d': error2d,'px': px, 'py': py}
            self.console_log(pred)

        S = self.get_shape(x)
        K = self.predict_intrinsic(x)
        Xc,R,T = util.EPnP_(x.permute(0,2,1),S,K)
        return S, K, R, T

    # perform dualoptimization with bpnp
    def dualoptimization_bpnp(self,x,max_iter=5,still=False):
        b = x.shape[0]
        K = self.predict_intrinsic(x).detach()
        S = self.get_shape(x).detach()
        pred = {'S': S, 'K': K}
        still = True if self.get_initial_motion(x) < 50 else False
        #create param optimizer
        best = 100000
        for i in range(max_iter):
            pred = self.opt_shape(x, pred['K'].mean(0).unsqueeze(0).repeat(b,1,1),max_iter=5)
            pred = self.opt_calib(x, pred['S'].mean(0).unsqueeze(0).repeat(b,1,1),still=still,max_iter=5)
            #pred = self.opt_shape_dlt(x,max_iter=5)
            loss = pred['error']

            #if ((torch.abs(best - loss) <= 0.01 or best < loss) and i >= 5) or i == max_iter: break
            if i == max_iter: break
            if loss < best: best = loss

        # get final prediction
        with torch.no_grad():
            S = self.get_shape(x).mean(0).unsqueeze(0).repeat(b,1,1)
            K = self.predict_intrinsic(x).mean(0).unsqueeze(0).repeat(b,1,1)
            Xc, R, T = util.EPnP_(x.permute(0,2,1),S,K)

        #R,T = self.get_pose(x,S,K)
        return S, K, R, T

    # perform joint optimization with our 3dmm + bpnp
    def oursbpnp_optimization(self,x,max_iter=5,still=False):
        b = x.shape[0]
        ini_pose = torch.zeros((b,6))
        ini_pose[:,5] = 99

        self.calib_opt.param_groups[0]['lr'] = 1e-4
        self.sfm_opt.param_groups[0]['lr'] = 1e-2

        # start main loop
        for i in range(max_iter):

            self.calib_opt.zero_grad()
            self.sfm_opt.zero_grad()

            # predict intrinsic and 3D shape
            K = self.predict_intrinsic(x).mean(0)
            S = self.get_shape(x).mean(0)

            # pose estimation
            pose = self.bpnp(x.permute(0,2,1),S,K,ini_pose)
            pred = BPnP.batch_project(pose,S,K).permute(0,2,1)

            # get error
            error2d = torch.norm(pred - x,dim=1).mean()
            error_p = losses.compute_principal_error(K.unsqueeze(0),self.center.unsqueeze(0)).mean()

            # optimize loss
            #loss = error2d + error_p*1e-2
            loss = error2d
            loss.backward()
            self.calib_opt.step()
            self.sfm_opt.step()

            # log results
            f = K[0,0]
            px = K[0,2]
            py = K[1,2]
            pred = {'iter': i, 'error': loss, 'e2d': error2d, 'e_p': error_p,
                    'f': f, 'px': px, 'py': py}
            self.console_log(pred)

        with torch.no_grad():
            S = self.get_shape(x).mean(0).unsqueeze(0).repeat(b,1,1)
            K = self.predict_intrinsic(x).mean(0).unsqueeze(0).repeat(b,1,1)
            Xc,R,T = util.EPnP_(x.permute(0,2,1),S,K)

        return S,K,R,T

    # perform Joint Optimization with BPnP
    def bpnp_optimization(self,x,max_iter=5):
        b = x.shape[0]
        n = x.shape[2]

        sfm_net = torchvision.models.vgg11()
        sfm_net.classifier = torch.nn.Linear(25088,68*3)
        #sfm_net = (10*torch.randn(68*3)).requires_grad_()
        calib_param = (torch.randn(3)).requires_grad_()
        self.calib_opt = torch.optim.Adam({calib_param},lr=5e-1)
        self.sfm_opt = torch.optim.Adam(sfm_net.parameters(),lr=1e-5)
        #self.sfm_opt = torch.optim.SGD([sfm_net],lr=1e-5)
        ini_pose = torch.zeros((b,6))
        ini_pose[:,5] = 99

        mu_s = self.get_mean_shape().detach()[0]
        #self.plot3d(mu_s.cpu().numpy())
        #quit()

        # perform optimization
        for i in range(max_iter):
            self.calib_opt.zero_grad()
            self.sfm_opt.zero_grad()

            # K = self.predict_intrinsic(x).mean(0)
            param = torch.sigmoid(calib_param)
            f = param[0]*1000 + 300
            px = param[1]*1000
            py = param[2]*1000
            K = torch.zeros(3,3)
            K[0,0] = f
            K[1,1] = f
            K[0,2] = px
            K[1,2] = py
            K[2,2] = 1

            #S = sfm_net.view(n,3)
            S = mu_s + sfm_net(torch.ones(1,3,32,32)).view(n,3)

            pose = self.bpnp(x.permute(0,2,1),S,K,ini_pose)
            pred = BPnP.batch_project(pose,S,K).permute(0,2,1)

            # get principal error
            error_p = losses.compute_principal_error(K.unsqueeze(0),self.center.unsqueeze(0)).mean()
            e2d = torch.norm(pred - x,dim=-1).mean()

            # apply loss
            #loss = e2d + error_p*1e-2
            loss = e2d
            loss.backward()

            # log results
            pred = {'iter': i, 'error': loss, 'e2d': e2d, 'e_p': error_p, 'f': f, 'px': px, 'py': py}
            self.console_log(pred)

            self.sfm_opt.step()
            self.calib_opt.step()

        R = kn.angle_axis_to_rotation_matrix(pose[:, 0:3].reshape(b, 3))
        T = pose[:,3:]
        S = mu_s + sfm_net(torch.ones(1,3,32,32)).view(n,3).detach()
        K = K.detach()
        return S.unsqueeze(0).repeat(b,1,1),K.unsqueeze(0).repeat(b,1,1),R,T

    # get mean shape S
    def get_mean_shape(self):
        return self.data3dmm.get_3dmm(torch.zeros((1,199,1)))

    # visualize 3d points
    def plot3d(self,pt):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(pt[:,0],pt[:,1],pt[:,2])
        plt.show()

    # forward functions for solving 3D shape S
    def get_shape(self,ptsI):
        #x = ptsI.permute(1,2,0).view(2,-1).unsqueeze(0)
        #betas = self.sfm_net(x).repeat(ptsI.shape[0],1)
        #S = self.data3dmm.get_3dmm(betas.unsqueeze(-1))
        #delta_S = self.delta_net(x).view(-1,68,3).repeat(ptsI.shape[0],1,1)*.1
        b = ptsI.shape[0]
        betas = self.sfm_net(ptsI).mean(0).unsqueeze(0).repeat(b,1)
        S = self.data3dmm.get_3dmm(betas.unsqueeze(-1))
        return S

        # return S + self.delta.unsqueeze(0)

    # forward function for solving intrinsic matrix K
    def predict_intrinsic(self,ptsI):
        #x = ptsI.permute(1,2,0).view(2,-1).unsqueeze(0)
        #k = self.calib_net(x).repeat(ptsI.shape[0],1)
        k = self.calib_net(ptsI)
        k[:,0] = k[:,0] + self.fbias
        k[:,1:] = k[:,1:] + self.pbias
        return util.create_K(k)

    # forward function for solving the pose given current estimation S,K
    def get_pose(self,ptsI,S,K):
        km,c_w,scaled_betas,alphas = util.EPnP(ptsI,S,K)
        #km,c_w,scaled_betas,alphas = util.EPnP_single(ptsI,S,K)
        _, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,S,ptsI)
        return R, T

    # reset the networks to random initiailztino
    def reset(self,n1, n2):
        self.calib_net = PointNet(n=n1)
        self.sfm_net = PointNet(n=n2)
        self.delta_net = PointNet(n=68*3)

    # get iinitial motion
    def get_initial_motion(self,x):
        b = x.shape[0]

        # initialize dummy values
        K = torch.zeros(3,3)
        K[0,0] = 500
        K[1,1] = 500
        K[0,2] = 320
        K[1,2] = 240
        S = self.get_mean_shape().detach()
        K = K.unsqueeze(0).repeat(b,1,1)
        S = S.repeat(b,1,1)

        # get pose with dummy values
        Xc,R,T = util.EPnP_(x.permute(0,2,1),S,K)

        # get initial motion
        l = losses.motionError(Xc.mean(1)).mean()
        return l

    # to cuda
    def to_cuda(self):
        self.calib_net
        self.sfm_net

    def save(self,token=''):
        print("saving!")
        torch.save(self.calib_net.state_dict(),self.model_path + os.sep + token + 'calib_net.pt')
        torch.save(self.sfm_net.state_dict(),self.model_path + os.sep + token + 'sfm_net.pt')

    # load network from directory
    def load(self,token=''):
        self.sfm_net.load_state_dict(torch.load(self.model_path + os.sep + token + 'sfm_net.pt'))
        self.calib_net.load_state_dict(torch.load(self.model_path + os.sep + token + 'calib_net.pt'))
        self.sfm_opt = torch.optim.Adam(self.sfm_net.parameters(),lr=5e-1)
        self.calib_opt = torch.optim.Adam(self.calib_net.parameters(),lr=1e-4)
        #self.sfm_net.eval()
        #self.calib_net.eval()

##############################################################################################
def create_networks(n1=3, n2=199):
    return PointNet(n=n1), PointNet(n=n2)

if __name__ == '__main__':
    sfm_net, calib_net = create_networks(3,199)

    optim = Optimizer(sfm_net,calib_net)
    print("hello world")


