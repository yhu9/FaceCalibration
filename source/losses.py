import torch
import matplotlib.pyplot as plt

import util

# compute principal error
# K         (M,3,3)
# p_hat     (1,3)
#
def compute_principal_error(K,p_hat):
    #if len(p_hat.shape) <= 2: p_hat = p_hat.unsqueeze(0)
    p = K[:,:,2]
    diff = p - p_hat
    return torch.mean(torch.norm(diff,dim=1))
    #return torch.mean(torch.pow(p - p_hat,2))

# compute reprojection error using epnp and intrinsic matrix A
# x         (M,68,2)
# S         (M,68,3)
# A         (M,3,3)
#
def compute_reprojection_error(x,S,A,show=False,loss='l2'):

    Xc, R, T = util.EPnP_(x,S,A)
    pc = torch.bmm(S,R.permute(0,2,1)) + T.unsqueeze(1)
    proj = torch.bmm(pc,A.permute(0,2,1))
    pimg_pred = proj / proj[:,:,-1].unsqueeze(-1)

    diff = x - pimg_pred[:,:,:2]
    error = torch.norm(diff,dim=2).mean()
    if show:
        pta = x[0].cpu().numpy()
        ptc = pimg_pred[0].detach().cpu().numpy()

        plt.scatter(pta[:,0],pta[:,1],s=15,facecolors='none',edgecolors='green')
        plt.scatter(ptc[:,0],ptc[:,1],s=10,marker='.',edgecolors='red')
        plt.show()

    return error

# batched reprojection error using intrinsics and extrinsics on world coordinates
#
#INPUT:
#   pimg                (M,2,N)
#   pw                  (N,3) or (M,N,3)
#   R                   (M,3,3)
#   T                   (M,3)
#   A                   (M,3,3)
#
#OUTPUT:
#   error               (M)
def getError(pimg,pw,R,T,A,show=False,loss='l2'):
    M = pimg.shape[0]
    N = pimg.shape[2]

    if len(pw.shape) == 3:
        pc = torch.bmm(R,pw.permute(0,2,1))
    else:
        pc = torch.bmm(R,torch.stack(M*[pw]).permute(0,2,1))
    pct = pc + T.unsqueeze(2)
    if len(A.shape) == 3:
        proj = torch.bmm(A,pct)
    else:
        proj = torch.bmm(torch.stack([A]*M),pct)
    proj_img = proj / proj[:,-1,:].unsqueeze(1)
    pimg_pred = proj_img[:,:2,:]

    # losses
    error = torch.norm(pimg - pimg_pred,dim=1)
    #error = torch.norm(pimg - pimg_pred,dim=1).mean()

    if show:
        pta = pimg[0].T.cpu().numpy()
        ptc = pimg_pred[0].detach().T.cpu().numpy()
        plt.scatter(pta[:,0],pta[:,1],s=15,facecolors='none',edgecolors='green')
        plt.scatter(ptc[:,0],ptc[:,1],s=10,marker='.',edgecolors='red')
        plt.show()
        quit()

    return error

def getXcError(pimg,Xc,A,show=False,loss='l2'):
    M = pimg.shape[0]
    N = pimg.shape[2]

    proj = torch.bmm(A,Xc)
    proj_img = proj / proj[:,-1,:].unsqueeze(1)
    proj_img = proj_img[:,:2,:]

    error = torch.norm(proj_img - pimg,dim=1).mean()

    return error

def getKinvError(pimg,Xc,A):
    M = pimg.shape[0]
    N = pimg.shape[2]

    d = Xc[:,2,:]
    proj = torch.bmm(torch.inverse(A),add_ones(pimg,1))
    proj = proj * d.unsqueeze(1)

    diff = Xc - proj
    return torch.norm(diff,dim=1).mean(1).sum()

def motionError(T):
    M = T.shape[0]
    c1 = T[:M-1]
    c2 = T[1:]
    diff = c1 - c2
    return torch.norm(diff,dim=-1)

def add_ones(x,dim):
    s = list(x.shape)
    s[dim] = 1
    return torch.cat((x,torch.ones(s).to(x.device)),dim=dim)

