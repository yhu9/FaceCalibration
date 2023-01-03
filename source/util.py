
import math

import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import pptk
import os

#import kornia as K
#import pytorch3d


def view_results(imgs,data):
    f_err = data['f_err']
    d_err = data['d_err']
    s_err = data['rmse']
    p_err = data['procrustes']
    loss = data['loss']
    fpred = data['fpred']
    fgt = data['fgt']
    R = data['R']
    T = data['T']
    K = data['K']
    M = imgs.shape[0]
    h, w, _ = imgs[0].shape

    proj1 = project(data['shape'][-1],K,R[0],T[0])
    proj2 = project(data['shape'][-1], K, R[M//3], T[M//3])
    proj3 = project(data['shape'][-1], K, R[(2*M)//3], T[(2*M)//3])
    proj4 = project(data['shape'][-1], K, R[M-1], T[M-1])

    proj1[:,1] = proj1[:,1] * -1
    proj2[:,1] = proj2[:, 1] * -1
    proj3[:,1] = proj3[:, 1] * -1
    proj4[:,1] = proj4[:, 1] * -1
    p1 = proj1 + np.array(([w//2, h//2]))
    p2 = proj2 + np.array(([w//2, h//2]))
    p3 = proj3 + np.array(([w//2, h//2]))
    p4 = proj4 + np.array(([w//2, h//2]))

    dim = (640,480)
    im1 = cv2.resize(drawpts(imgs[0].copy(),p1,size=5,color=[0,255,0]),dim)
    im2 = cv2.resize(drawpts(imgs[M//3].copy(),p2,size=5,color=[0,255,0]),dim)
    im3 = cv2.resize(drawpts(imgs[(2*M)//3].copy(),p3,size=5,color=[0,255,0]),dim)
    im4 = cv2.resize(drawpts(imgs[M-1].copy(),p4,size=5,color=[0,255,0]),dim)

    iterations = f_err.shape[0]
    fig, axs = plt.subplots(2,6)
    axs[0,0].imshow(cv2.resize(imgs[0].astype(np.uint8),dim))
    axs[0,1].imshow(cv2.resize(imgs[M//3].astype(np.uint8),dim))
    axs[0,2].imshow(cv2.resize(imgs[(2*M)//3].astype(np.uint8),dim))
    axs[0,3].imshow(cv2.resize(imgs[M-1].astype(np.uint8),dim))
    axs[0, 0].axis('off')
    axs[0, 1].axis('off')
    axs[0, 2].axis('off')
    axs[0, 3].axis('off')
    axs[0, 0].set_title('Seq 1')
    axs[0, 1].set_title(f"Seq {M//3}")
    axs[0, 2].set_title(f"Seq {(2*M)//3}")
    axs[0, 3].set_title(f"Seq {M}")
    axs[1,0].imshow(cv2.resize(im1,dim))
    axs[1,1].imshow(cv2.resize(im2,dim))
    axs[1,2].imshow(cv2.resize(im3,dim))
    axs[1,3].imshow(cv2.resize(im4,dim))
    axs[1, 0].axis('off')
    axs[1, 1].axis('off')
    axs[1, 2].axis('off')
    axs[1, 3].axis('off')
    axs[1, 0].set_title(f"Reproj Error {data['e2d'][0]:.3f}")
    axs[1, 1].set_title(f"Reproj Error {data['e2d'][M//3]:.3f}")
    axs[1, 2].set_title(f"Reproj Error {data['e2d'][(2*M)//3]:.3f}")
    axs[1, 3].set_title(f"Reproj Error {data['e2d'][M-1]:.3f}")
    axs[0,4].plot(loss)
    axs[0,5].plot(d_err)
    #axs[1,4].plot(s_err)
    axs[1, 4].plot(s_err)
    axs[1,5].plot(fpred,label='prediction')
    axs[1,5].plot(fgt,label='Ground Truth')
    axs[0, 4].set_title('Loss')
    axs[0, 5].set_title('Depth Error')
    axs[1, 4].set_title('Shape error')
    axs[1, 5].set_title('Focal Prediction')
    axs[0, 4].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    axs[0, 5].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    axs[1, 4].ticklabel_format(axis='y',style='sci',scilimits=(0,0))

    axs[1,5].legend()
    fig.set_size_inches(24,6)
    fig.tight_layout()
    fig.savefig('results/example/example.png')

    return None

# depth error function
# compute error as relative distance between the predicted translation to the true center of the camera coordinates for each frame
def getDepthError(d, dgt):
    diff = dgt - d
    err = torch.abs(diff) / dgt
    return err

def getRelTransError(c,T):
    diff = c - T
    err = torch.norm(diff,dim=1) / torch.norm(c,dim=1)
    return err.mean()

def getReprojErrorDLT(x_i,shape):
    m = x_i.shape[0]
    P = batchDLT(x_i, shape)
    s = torch.stack(m * [torch.cat((shape, torch.ones(shape.shape[0], 1)), dim=1).T])
    proj = torch.bmm(P, s)
    proj = proj / proj[:, 2].unsqueeze(1)
    diff = proj[:, :2, :] - x_i

    return torch.norm(diff,dim=1).mean()

# eliminate Noisy 2D correspondences according to the Fundamental Matrix Constraint
# and RANSAC inlier selection
#
# x_i     -> (M,N,2)            : 2D points
# mindist -> scaler             : minimum absolute pixel distance for inlier selection
def getCorrespondence(x_i,mindist=0.1,x_i_gt=None):
    M, N, _ = x_i.shape
    face_indices = torch.arange(53215)
    pts = []
    pt_idx = []
    cam_idx = []
    for i in range(M-1):

        src_pts = x_i[0]
        dst_pts = x_i[i+1]

        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_8POINT, mindist)

        x1 = np.concatenate([src_pts, np.ones((N, 1))], axis=1)
        x2 = np.concatenate([dst_pts,np.ones((N,1))],axis=1)
        x1 = torch.tensor(x1,dtype=torch.float32).unsqueeze(1)
        x2 = torch.tensor(x2,dtype=torch.float32).unsqueeze(2)
        F = torch.tensor(F,dtype=torch.float32)
        F = torch.stack(53215*[F])
        d = torch.bmm(torch.bmm(x1,F),x2).squeeze()
        idx = torch.argsort(d.abs())
        p1 = torch.index_select(torch.tensor(src_pts),0,idx[:30])
        p2 = torch.index_select(torch.tensor(dst_pts),0,idx[:30])
        idx1 = torch.index_select(face_indices,0,idx[:30])
        idx2 = np.ones(p1.shape[0], dtype=np.int) * i

        # visualize correspondence selection ratio for each view
        # diff = x_i - x_i_gt
        # d = np.linalg.norm(diff, axis=2)
        # a = np.linspace(0, 1, 1000)
        # src_d = d[0]
        # dst_d = d[i+1]

        # DO NOT ADD VIEW IF IT DOES NOT EVEN PRODUCE 100 FRAMES FOR FUNDAMENTAL MATRIX ASSUMPTION
        if np.sum(mask) <= 20: continue
        if i == 0:
            pts.append(np.squeeze(p1))
            pt_idx.append(idx1.numpy())
            cam_idx.append(idx2)

        pts.append(np.squeeze(p2.numpy()))
        pt_idx.append(idx1)
        cam_idx.append(idx2)

    # SOME ERROR CHECKING FOR POSSIBLE FUTURE PROBLEMS
    if len(pts) < 20:
        print("LESS THAN 20 VALID VIEWS FOUND")
        quit()

    return np.concatenate(pts), np.concatenate(pt_idx), np.concatenate(cam_idx)

def viewpts2d(pts,w,h):
    window = np.zeros((h,w))
    for i in range(pts.shape[0]):
        z = pts[i,2]
        x = int(pts[i,0] / z + w//2)
        y = int(pts[i,1] / z + h//2)

        cv2.circle(window,(x,y),2,255,-1)

    cv2.imshow('img',window)
    cv2.waitKey(1)

    return

def getFMatrices(x_i):
    M,N,_ = x_i.shape
    pair = []
    for i in range(M):
        for j in range(M):
            if i >= j: continue

            src_pts = x_i[i].reshape(-1,1,2)
            dst_pts = x_i[j].reshape(-1,1,2)
            F, mask = cv2.findFundamentalMat(src_pts,dst_pts,cv2.RANSAC,0.1,confidence=0.999999)
            p1 = src_pts[mask.ravel().astype(np.bool)]
            p2 = dst_pts[mask.ravel().astype(np.bool)]

    return pair

# gets M-1 fundamental matrices using first frame as key frame
# x_i   -> (M,N,2)  -> np.float array
#
# return:
# pair  -> (M,3,3)  -> np.float array
def getFMatricesSequential(x_i):
    M, N, _ = x_i.shape
    pair = []
    for i in range(1,M-1):
        src_pts = x_i[0]
        dst_pts = x_i[i]
        F, mask = cv2.findFundamentalMat(src_pts,dst_pts,cv2.FM_8POINT)
        pair.append(F)

    return np.stack(pair)

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        #m.weight.data.fill_(0.00)
        m.bias.data.fill_(0.00)
    if type(m) == torch.nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)
        #m.weight.data.fill_(0.00)
        m.bias.data.fill_(0.00)

# control points set as arbitrary basis vector in R4
#Input:
#   Pw              (N,3)
#Output:
#   control_w
def getControlPoints(Pw):
    if Pw.is_cuda:
        control_w = torch.zeros((4,4)).float().cuda()
    else:
        control_w = torch.zeros((4,4)).float()

    control_w[0,0] = 1
    control_w[1,1] = 1
    control_w[2,2] = 1
    control_w[-1,:] = 1

    return control_w

def getControlPoints_():
    control_w = torch.zeros((4,4)).float()
    control_w[0,0] = 1
    control_w[1,1] = 1
    control_w[2,2] = 1
    control_w[-1,:] = 1

    return control_w

# control points set as arbitrary basis vector in R4
#Input:
#   Pw              (b,N,3)
#Output:
#   control_w       (bx4x4)
def getBatchControlPoints(Pw):
    b = Pw.shape[0]
    if Pw.is_cuda:
        control_w = torch.zeros((b,4,4)).float().cuda()
    else:
        control_w = torch.zeros((b,4,4)).float()

    control_w[:,0,0] = 1
    control_w[:,1,1] = 1
    control_w[:,2,2] = 1
    control_w[:,-1,:] = 1

    return control_w


'''     control points using svd and finding principal components
#Input:
# Pw                (N,3)
#
#Output:
# control_w         (4,4)
def getControlPoints(Pw):
    if Pw.is_cuda:
        ones = torch.ones((1,4)).cuda()
    else:
        ones = torch.ones((1,4))
    c = torch.mean(Pw,dim=0)
    centered = Pw-c
    u,d,v = torch.svd(centered)
    control_w = v + c.unsqueeze(1)
    control_w = torch.cat([control_w,c.unsqueeze(1)],dim=1)
    control_w = torch.cat([control_w,ones],dim=0)

    return control_w
'''

#Input:
# Pw                (N,3)
# control_w         (4,4)
#
#Output:
# alphas            (3,N)
def solveAlphas(Pw,control_w):
    N = Pw.shape[0]
    if Pw.is_cuda:
        ones = torch.ones((1,N)).cuda()
    else:
        ones = torch.ones((1,N))
    ph_w = torch.cat([Pw.T,ones],dim=0)
    alphas, LU = torch.solve(ph_w,control_w)

    return alphas

# pw            (M,N,3)
# control_w     (M,N,3)
def solveAlphas_(pw,control_w):
    b = pw.shape[0]
    n = pw.shape[1]

    ones = torch.ones((b,n,1)).to(pw.device)
    ph_w = torch.cat([pw,ones],dim=-1)

    alphas = torch.linalg.solve(control_w,ph_w.permute(0,2,1))

    return alphas

#Input:
# Pw                (b,N,3)
# control_w         (b,4,4)
#
#Output:
# alphas            (b,3,N)
def batchSolveAlphas(Pw,control_w):
    b = Pw.shape[0]
    if Pw.is_cuda:
        ones = torch.ones((b,1,68)).cuda()
    else:
        ones = torch.ones((b,1,68))

    ph_w = torch.cat([Pw.permute(0,2,1),ones],dim=1)
    alphas, LU = torch.solve(ph_w,control_w)

    return alphas

#Input:
#   alphas                  (3,N)
#   p_img                   (M,N,2)
#   px                      scalar
#   py                      scalar
#   f                       scalar
#Output:
#
#   M                       (M,2*N,12)
def setupM(alphas,p_img,px,py,f):
    views = p_img.shape[0]
    N = p_img.shape[1]
    if p_img.is_cuda:
        M = torch.zeros((views,2*N,12)).float().cuda()
    else:
        M = torch.zeros((views,2*N,12))

    M[:,0::2,0] = alphas[0,:]*f
    M[:,0::2,1] = 0
    M[:,0::2,2] = alphas[0,:].unsqueeze(0) * (px - p_img[:,:,0])
    M[:,0::2,3] = alphas[1,:]*f
    M[:,0::2,4] = 0
    M[:,0::2,5] = alphas[1,:].unsqueeze(0) * (px-p_img[:,:,0])
    M[:,0::2,6] = alphas[2,:]*f
    M[:,0::2,7] = 0
    M[:,0::2,8] = alphas[2,:].unsqueeze(0) * (px-p_img[:,:,0])
    M[:,0::2,9] = alphas[3,:]*f
    M[:,0::2,10] = 0
    M[:,0::2,11] = alphas[3,:].unsqueeze(0) * (px-p_img[:,:,0])
    M[:,1::2,0] = 0
    M[:,1::2,1] = alphas[0,:]*f
    M[:,1::2,2] = alphas[0,:].unsqueeze(0) * (py-p_img[:,:,1])
    M[:,1::2,3] = 0
    M[:,1::2,4] = alphas[1,:]*f
    M[:,1::2,5] = alphas[1,:].unsqueeze(0) * (py-p_img[:,:,1])
    M[:,1::2,6] = 0
    M[:,1::2,7] = alphas[2,:]*f
    M[:,1::2,8] = alphas[2,:].unsqueeze(0) * (py-p_img[:,:,1])
    M[:,1::2,9] = 0
    M[:,1::2,10] = alphas[3,:]*f
    M[:,1::2,11] = alphas[3,:].unsqueeze(0) * (py-p_img[:,:,1])

    return M

#Input:
#   alphas                  (M,4,N)
#   p_img                   (M,N,2)
#   px                       (M)
#   py                       (M)
#   f                       (M)
def setupM_(alphas,p_img,K):
    b = p_img.shape[0]
    n = p_img.shape[1]

    my_matrix = torch.zeros((b,2*n,12)).float().to(p_img.device)
    f = (K[:,0,0]+ K[:,1,1])*0.5
    px = K[:,0,2]
    py = K[:,1,2]

    my_matrix[:,0::2,0] = alphas[:,0,:] * f.unsqueeze(1)
    my_matrix[:,0::2,1] = 0
    my_matrix[:,0::2,2] = alphas[:,0,:] * (px.unsqueeze(1) - p_img[:,:,0])
    my_matrix[:,0::2,3] = alphas[:,1,:]*f.unsqueeze(1)
    my_matrix[:,0::2,4] = 0
    my_matrix[:,0::2,5] = alphas[:,1,:] * (px.unsqueeze(1) - p_img[:,:,0])
    my_matrix[:,0::2,6] = alphas[:,2,:]*f.unsqueeze(1)
    my_matrix[:,0::2,7] = 0
    my_matrix[:,0::2,8] = alphas[:,2,:] * (px.unsqueeze(1) - p_img[:,:,0])
    my_matrix[:,0::2,9] = alphas[:,3,:]*f.unsqueeze(1)
    my_matrix[:,0::2,10] = 0
    my_matrix[:,0::2,11] = alphas[:,3,:] * (px.unsqueeze(1) - p_img[:,:,0])
    my_matrix[:,1::2,0] = 0
    my_matrix[:,1::2,1] = alphas[:,0,:]*f.unsqueeze(1)
    my_matrix[:,1::2,2] = alphas[:,0,:] * (py.unsqueeze(1) - p_img[:,:,1])
    my_matrix[:,1::2,3] = 0
    my_matrix[:,1::2,4] = alphas[:,1,:]*f.unsqueeze(1)
    my_matrix[:,1::2,5] = alphas[:,1,:] * (py.unsqueeze(1) - p_img[:,:,1])
    my_matrix[:,1::2,6] = 0
    my_matrix[:,1::2,7] = alphas[:,2,:]*f.unsqueeze(1)
    my_matrix[:,1::2,8] = alphas[:,2,:] * (py.unsqueeze(1) - p_img[:,:,1])
    my_matrix[:,1::2,9] = 0
    my_matrix[:,1::2,10] = alphas[:,3,:]*f.unsqueeze(1)
    my_matrix[:,1::2,11] = alphas[:,3,:] * (py.unsqueeze(1) - p_img[:,:,1])

    return my_matrix

#Input:
#   alphas                  (3,N)
#   p_img                   (M,N,2)
#   px                       (M)
#   py                       (M)
#   f                       (M)
#Output:
#
#   M                       (M,2*N,12)
def setupM_single(alphas,p_img,px,py,f):
    views = p_img.shape[0]
    N = p_img.shape[1]
    if p_img.is_cuda:
        M = torch.zeros((views,2*N,12)).float().cuda()
    else:
        M = torch.zeros((views,2*N,12))

    view_alphas = torch.stack(views*[alphas])

    M[:,0::2,0] = view_alphas[:,0,:]*f.unsqueeze(1)
    M[:,0::2,1] = 0
    M[:,0::2,2] = view_alphas[:,0,:].unsqueeze(0) * (px.unsqueeze(1) - p_img[:,:,0])
    M[:,0::2,3] = view_alphas[:,1,:]*f.unsqueeze(1)
    M[:,0::2,4] = 0
    M[:,0::2,5] = view_alphas[:,1,:].unsqueeze(0) * (px.unsqueeze(1) - p_img[:,:,0])
    M[:,0::2,6] = view_alphas[:,2,:]*f.unsqueeze(1)
    M[:,0::2,7] = 0
    M[:,0::2,8] = view_alphas[:,2,:].unsqueeze(0) * (px.unsqueeze(1) - p_img[:,:,0])
    M[:,0::2,9] = view_alphas[:,3,:]*f.unsqueeze(1)
    M[:,0::2,10] = 0
    M[:,0::2,11] = view_alphas[:,3,:].unsqueeze(0) * (px.unsqueeze(1) - p_img[:,:,0])
    M[:,1::2,0] = 0
    M[:,1::2,1] = view_alphas[:,0,:]*f.unsqueeze(1)
    M[:,1::2,2] = view_alphas[:,0,:].unsqueeze(0) * (py.unsqueeze(1) - p_img[:,:,1])
    M[:,1::2,3] = 0
    M[:,1::2,4] = view_alphas[:,1,:]*f.unsqueeze(1)
    M[:,1::2,5] = view_alphas[:,1,:].unsqueeze(0) * (py.unsqueeze(1) - p_img[:,:,1])
    M[:,1::2,6] = 0
    M[:,1::2,7] = view_alphas[:,2,:]*f.unsqueeze(1)
    M[:,1::2,8] = view_alphas[:,2,:].unsqueeze(0) * (py.unsqueeze(1) - p_img[:,:,1])
    M[:,1::2,9] = 0
    M[:,1::2,10] = view_alphas[:,3,:]*f.unsqueeze(1)
    M[:,1::2,11] = view_alphas[:,3,:].unsqueeze(0) * (py.unsqueeze(1) - p_img[:,:,1])

    return M

'''
#Input:
#   alphas                  (3,N)
#   p_img                   (N,2)
#   px                      scalar
#   py                      scalar
#   f                       scalar
#Output:
#
#   M                       (2*N,12)
def setupM_single(alphas,p_img,px,py,f):
    N = p_img.shape[0]
    if p_img.is_cuda:
        M = torch.zeros((2*N,12)).float().cuda()
    else:
        M = torch.zeros((2*N,12))

    M[0::2,0] = alphas[0,:]*f
    M[0::2,1] = 0
    M[0::2,2] = alphas[0,:] * (px - p_img[:,0])
    M[0::2,3] = alphas[1,:]*f
    M[0::2,4] = 0
    M[0::2,5] = alphas[1,:] * (px-p_img[:,0])
    M[0::2,6] = alphas[2,:]*f
    M[0::2,7] = 0
    M[0::2,8] = alphas[2,:] * (px-p_img[:,0])
    M[0::2,9] = alphas[3,:]*f
    M[0::2,10] = 0
    M[0::2,11] = alphas[3,:] * (px-p_img[:,0])
    M[1::2,0] = 0
    M[1::2,1] = alphas[0,:]*f
    M[1::2,2] = alphas[0,:] * (py-p_img[:,1])
    M[1::2,3] = 0
    M[1::2,4] = alphas[1,:]*f
    M[1::2,5] = alphas[1,:]* (py-p_img[:,1])
    M[1::2,6] = 0
    M[1::2,7] = alphas[2,:]*f
    M[1::2,8] = alphas[2,:] * (py-p_img[:,1])
    M[1::2,9] = 0
    M[1::2,10] = alphas[3,:]*f
    M[1::2,11] = alphas[3,:] * (py-p_img[:,1])

    return M
'''

#Input:
# control_w             (4x4)
#
#Output:
# d12                   scalar
# d13                   scalar
# d14                   scalar
# d23                   scalar
# d24                   scalar
# d34                   scalar
#
def getDistances(control_w):
    p1 = control_w[:3,0]
    p2 = control_w[:3,1]
    p3 = control_w[:3,2]
    p4 = control_w[:3,3]

    d12 = torch.norm(p1-p2,p=2)
    d13 = torch.norm(p1-p3,p=2)
    d14 = torch.norm(p1-p4,p=2)
    d23 = torch.norm(p2-p3,p=2)
    d24 = torch.norm(p2-p4,p=2)
    d34 = torch.norm(p3-p4,p=2)

    return d12,d13,d14,d23,d24,d34

# scale control points and get camera coordinates via method similar to procrustes
#Input:
#   c_c                 (3x4)
#   c_w                 (3x4)
#   alphas              (4xN)
#   x_w                 (Nx3)
#
#Output:
#   sc_c                (3x4)
#   sx_c                (Nx3)
#   s                   (1)
def scaleControlPoints_single(c_c,c_w,alphas,x_w):
    x_c = torch.mm(c_c,alphas)

    centered_xw = x_w - torch.mean(x_w,dim=0).unsqueeze(0)
    d_w = torch.norm(centered_xw,p=2,dim=1)
    centered_xc = x_c - torch.mean(x_c,dim=1).unsqueeze(1)
    d_c = torch.norm(centered_xc,p=2,dim=0)

    # least square solution to scale
    s = 1.0 /((1.0/torch.sum(d_c*d_c) * torch.sum(d_c*d_w)))

    # apply scale onto c_c and recompute the camera coordinates
    sc_c = c_c / s
    sx_c = torch.mm(sc_c,alphas).T

    # fix the sign so negative depth is not possible
    negdepth_mask = sx_c[:,-1] < 0
    sx_c = sx_c * (negdepth_mask.float().unsqueeze(1) * -2 +1)

    return sc_c, sx_c, s

# scale control points and get camera coordinates via method similar to procrustes
#Input:
#   c_c                 (Mx3x4)
#   c_w                 (3x4)
#   alphas              (M,4,N)
#   x_w                 (M,N,3)
#
#Output:
#   sc_c                (Mx3x4)
#   sx_c                (MxNx3)
#   s                   (M)
def scaleControlPoints_(c_c,c_w,alphas,x_w):
    views = c_c.shape[0]
    x_c = torch.bmm(c_c,alphas)

    # center x_w and x_c
    centered_xw = x_w - torch.mean(x_w,dim=1).unsqueeze(1)

    # get depth to world points
    d_w = torch.norm(centered_xw,dim=2)

    # get depth to control points
    centered_xc = x_c - torch.mean(x_c,dim=2).unsqueeze(2)
    d_c = torch.norm(centered_xc,dim=1)

    # least square solution to scale
    #s = 1.0 /((1.0/(torch.sum(d_c*d_c,dim=1)) * torch.sum(d_c*d_w,dim=1)))
    #s = 1.0 /( torch.sum(d_c*d_w,dim=1)/ torch.sum(d_c*d_c,dim=1))
    s = torch.sum(d_c*d_w,dim=1) / torch.sum(d_c*d_c,dim=1)

    # apply scale onto c_c and recompute the camera coordinates
    sc_c = c_c * s.unsqueeze(1).unsqueeze(1)
    sx_c = torch.bmm(sc_c,alphas).permute(0,2,1)

    # fix the sign so negative depth is not possible
    negdepth_mask = sx_c[:,:,-1] < 0
    # sx_c = sx_c * (negdepth_mask.float().unsqueeze(2) * -2 +1)

    # scale control points
    #mask = sc_c[:,:,-1] < 0
    #print(torch.any(negdepth_mask))
    #quit()

    return s

# scale control points and get camera coordinates via method similar to procrustes
#Input:
#   c_c                 (Mx3x4)
#   c_w                 (3x4)
#   alphas              (4xN)
#   x_w                 (Nx3)
#
#Output:
#   sc_c                (Mx3x4)
#   sx_c                (MxNx3)
#   s                   (M)
def scaleControlPoints(c_c,c_w,alphas,x_w):
    views = c_c.shape[0]
    rep_alpha = torch.stack(views*[alphas])
    x_c = torch.bmm(c_c,rep_alpha)

    # center x_w and x_c
    centered_xw = x_w - torch.mean(x_w,dim=0).unsqueeze(0)
    d_w = torch.norm(centered_xw,p=2,dim=1)
    centered_xc = x_c - torch.mean(x_c,dim=2).unsqueeze(2)
    d_c = torch.norm(centered_xc,p=2,dim=1)

    # least square solution to scale
    s = 1.0 /((1.0/(torch.sum(d_c*d_c,dim=1)) * torch.sum(d_c*d_w.unsqueeze(0),dim=1)))

    # apply scale onto c_c and recompute the camera coordinates
    sc_c = c_c / s.unsqueeze(1).unsqueeze(1)
    sx_c = torch.bmm(sc_c,rep_alpha).permute(0,2,1)

    # fix the sign so negative depth is not possible
    #print(sx_c.shape)
    negdepth_mask = sx_c[:,:,-1] < 0
    sx_c = sx_c * (negdepth_mask.float().unsqueeze(2) * -2 +1)

    return sc_c, sx_c, s

# batched DLT error as SUM{(x1-y1)^2, ..., (x-y)^2}
# x_i   -> (M,2,N)
# shape -> (N,3)
def dltReprojectionError(x_i,shape):
    if len(shape.shape) == 3:
        P = batchDLT(x_i, shape)
        ones = torch.ones(shape.shape[0],shape.shape[1]).to(shape.device)
        s = torch.cat((shape,ones.unsqueeze(-1)),dim=-1)
        proj = torch.bmm(P,s.permute(0,2,1))
        proj = proj / proj[:,2].unsqueeze(1)
    else:
        P = DLT(x_i,shape)
        proj = projectDLT(P,shape)
    diff = proj[:,:2,:] - x_i
    error = torch.abs(diff).sum(1).mean(1)
    return error

# batched DLT error as SUM{(x1-y1)^2, ..., (x-y)^2}
# P   -> (M,3,4)
# shape -> (N,3)
def projectDLT(P,shape):
    M = P.shape[0]
    s = torch.stack(M * [torch.cat((shape, torch.ones(shape.shape[0], 1).to(shape.device)), dim=1).T])
    proj = torch.bmm(P,s)
    proj = proj / proj[:,2].unsqueeze(1)
    return proj

# batched DLT error as SUM{(x1-y1)^2, ..., (x-y)^2}
# F     -> (M,3,3)  -> float tensor
# f     -> (1)      -> float tensor
def mendoncaError(F,f):

    M = F.shape[0]

    # create intrinsic matrix
    A = torch.zeros((3,3))
    A[0,0] = f
    A[1,1] = f
    A[2,2] = 1

    # stack for batch matrix multiplication
    A_t = torch.stack(M*[A.T])
    A_ = torch.stack(M*[A])

    # get essential matrix
    EM = torch.bmm(torch.bmm(A_t,F),A_)

    # get singular values
    u,d,v = torch.svd(EM)
    r = d[:,0]
    s = d[:,1]

    # cost function
    cost = (1/M) * (r-s) / (r+s)

    return torch.sum(cost)

# Method for getting extrinsics after finding world coordinates and the camera coord
# solved using idea similar to procrustes and closest Rotation matrix
#Input:
#   x_c                 (M,N,3)
#   p_w                 (M,N,3)
#Output:
#   R                   (Mx3x3)
#   T                   (Mx3x1)
def getExtrinsics_(x_c,p_w):
    c_center = torch.mean(x_c,dim=1)
    w_center = torch.mean(p_w,dim=1)

    # center the  3d shapes
    x_c_centered = x_c - c_center.unsqueeze(1)
    p_w_centered = p_w - w_center.unsqueeze(1)

    # create martrix to solve
    M = x_c.shape[0]
    Matrix = torch.bmm(x_c_centered.permute(0,2,1),p_w_centered)

    u,d,v = torch.svd(Matrix)

    #R = torch.bmm(u,v.permute(0,2,1))
    R = torch.bmm(u,v.permute(0,2,1))

    # solve T using centers
    #rot_w_center = torch.bmm(R,rep_w_center)

    # T = c_center - w_center.squeeze()
    T = c_center

    return R,T

# Method for getting extrinsics after finding world coordinates and the camera coord
# solved using idea similar to procrustes and closest Rotation matrix
#Input:
#   x_c                 (MxNx3)
#   p_w                 (Nx3)
#Output:
#   R                   (Mx3x3)
#   T                   (Mx3x1)
def getExtrinsics(x_c,p_w):
    c_center = torch.mean(x_c,dim=1)
    w_center = torch.mean(p_w,dim=0)

    # center the  3d shapes
    x_c_centered = x_c - c_center.unsqueeze(1)
    p_w_centered = p_w - w_center.unsqueeze(0)

    # create martrix to solve
    M = x_c.shape[0]
    if p_w.is_cuda:
        Matrix = torch.zeros((M,3,3)).cuda()
    else:
        Matrix = torch.zeros((M,3,3))
    for i in range(M):
        m1 = x_c_centered[i].unsqueeze(2)
        m2 = p_w_centered.unsqueeze(1)
        tmpM = torch.bmm(m1,m2)
        Matrix[i] = torch.sum(tmpM,dim=0)

    u,d,v = torch.svd(Matrix)

    #R = torch.bmm(u,v.permute(0,2,1))
    R = torch.bmm(u,v.permute(0,2,1))

    # solve T using centers
    rep_w_center = torch.stack(M * [w_center]).unsqueeze(2)
    rot_w_center = torch.bmm(R,rep_w_center)
    T = c_center - rot_w_center.squeeze()

    return R,T

# prot of matlab's procrustes analysis function in numpy
# https://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

# pts       (b,M,2,N)
# shape     (b,N,3)
# K         (3,3)
def getBatchError(pts,shape,K):
    e2d = []
    e3d = []
    e2d_all = []
    e3d_all = []
    d_all = []
    b = pts.shape[0]
    for i in range(b):
        pW = shape[i]
        pI = pts[i]
        M = pI.shape[0]
        km, c_w, scaled_betas, alphas = EPnP(pI,pW,K)

        Xc, R, T, mask = optimizeGN(km,c_w,scaled_betas,alphas,pW,pI,K)

        error2d = getReprojError2(pI,pW,R,T,K,loss='l2')
        error3d = getRelReprojError3(pC,pW,R,T)
        d = torch.norm(pC,p=2,dim=1).mean(1)

        e2d.append(error2d.mean())
        e3d.append(error3d.mean())
        e2d_all.append(error2d)
        e3d_all.append(error3d)
        d_all.append(d)
    errorShape = torch.mean(torch.norm(xw_gt - shape,dim=2),dim=1)
    return torch.stack(e2d),torch.stack(e3d),errorShape,torch.stack(e2d_all),torch.stack(e3d_all),torch.stack(d_all)

def getError2(pimg,pcam,A,show=False):
    M = pimg.shape[0]
    N = pimg.shape[1]

    proj = torch.bmm(torch.stack(M*[A]),pcam)
    proj = proj_img = proj / proj[:,-1,:].unsqueeze(1)
    pimg_pred = proj_img[:,:2,:]
    diff = pimg - pimg_pred
    error = torch.mean(torch.norm(diff,p=2,dim=1),dim=1)

    return error

# batched reprojection error using intrinsics and extrinsics on world coordinates
#
#INPUT:
#   xcam                (M,3,N)
#   pw                  (N,3)
#   R                   (M,3,3)
#   T                   (M,3)
#   A                   (3,3)
#
#OUTPUT:
#   error               (M)
def getReprojError3(xcam,pw,R,T):

    M = xcam.shape[0]
    pc = torch.bmm(R,torch.stack(M*[pw]).permute(0,2,1))
    pct = pc + T.unsqueeze(2)
    diff = pct - xcam
    error  = torch.mean(torch.norm(diff,p=2,dim=1),dim=1)

    return error

def getReprojError(xcam,pw,R,T):
    M = xcam.shape[0]
    pc = torch.bmm(R,torch.stack(M*[pw]).permute(0,2,1))
    pct = pc + T.unsqueeze(2)

    error = torch.mean(torch.abs(pct - xcam))

    return error

#xcam = torch.Size([16, 3, 6800])
#ximg = torch.Size([16, 6800, 3])
#kinv = torch.Size([16, 3, 3])
def getPCError(xcam,ximg,kinv,mode='l1'):

    #torch.set_printoptions(profile='full')
    M = xcam.shape[0]
    proj = torch.bmm(kinv,ximg.permute(0,2,1))

    xcam_pred = proj*xcam[:,2,:].unsqueeze(1)
    if mode == 'l1':
        return torch.nn.functional.l1_loss(xcam_pred[:,:2,:],xcam[:,:2,:])
    else:
        return torch.nn.functional.mse_loss(xcam_pred[:,:2,:],xcam[:,:2,:])

# get error in fundamental matrix
#INPUT:
#   pI ->   (M,2,N)
#   f ->    (3,3)
#   R ->    (M,3,3)
#   T ->    (M,3)
#OUTPUT:
#
def FundamentalError(pI,K,R,T):
    M = pI.shape[0]
    N = pI.shape[2]

    tx = torch.zeros((M,3,3))
    tx[:,0,1] = -T[:,2]
    tx[:,0,2] = T[:,1]
    tx[:,1,0] = T[:,2]
    tx[:,1,2] = -T[:,0]
    tx[:,2,0] = -T[:,1]
    tx[:,2,1] = T[:,0]

    kinv = torch.zeros((3,3))
    kinv[0,0] = 1/K[0,0]
    kinv[1,1] = 1/K[1,1]
    kinv[2,2] = 1

    E = torch.bmm(R,tx)
    F = torch.bmm(torch.bmm(kinv,E),kinv)

    return error

#INPUT
# s1        (68,3)
# s2        (68,3)
#OUTPUT
# scalar
def getShapeError(s1,s2):
    diff = s1 - s2
    error = torch.mean(torch.norm(diff,dim=1))
    return error

#INPUT
# pI        (M,2,N)
# pC        (M,3,N)
def solvef(pI,pC):
    M = pI.shape[0]
    N = pI.shape[2]
    fx = (pC[:,2,:] / pC[:,0,:]) * (pI[:,0,:] - 1)
    fy = (pC[:,2,:] / pC[:,1,:])* (pI[:,1,:] - 1)
    f = torch.cat([fx,fy]).mean()
    return f

#xcam = torch.Size([16, 3, 6800])
#ximg = torch.Size([16, 6800, 3])
#kinv = torch.Size([16, 3, 3])
def getRelPCError(xcam,ximg,kinv,mode='l1'):

    #torch.set_printoptions(profile='full')
    M = xcam.shape[0]
    proj = torch.bmm(kinv,ximg.permute(0,2,1))

    xcam_pred = proj*xcam[:,2,:].unsqueeze(1)
    diff = xcam_pred[:,:2,:] - xcam[:,:2,:]
    if mode == 'l1':
        l1_reldiff = torch.mean(torch.abs(diff) / xcam[:,2,:])
        return l1_reldiff
    else:
        l2_reldiff = torch.mean(torch.log(torch.sum(torch.pow(diff,2),1) / torch.pow(xcam[:,2,:],2)))
        return l2_reldiff

# getTimeConsistency(pW,R,T)
def getTimeConsistency(pW,R,T):
    M = R.shape[0]
    pc = torch.bmm(R,torch.stack(M*[pW]).permute(0,2,1))
    pct = pc + T.unsqueeze(2)

    error = torch.mean(torch.norm(pct[:M-1,:,:] - pct[1:,:,:],dim=1))
    return error

# getTimeConsistency(pW,R,T)
def getTimeConsistency2(pW,R,T):
    M = R.shape[0]
    pc = torch.bmm(R,torch.stack(M*[pW]).permute(0,2,1))
    pct = pc + T.unsqueeze(2)

    error = torch.norm(pct[:M-1,:,:] - pct[1:,:,:],dim=1)
    return error

#INPUT:
# T     (M,3)
#OUTPUT:
# SCALAR
def getTConsistency(T):

    t1 = T[1:]
    t2 = T[:-1]

    error = torch.mean(torch.sum(torch.pow(t1 - t2,2),1))

    return error

#INPUT:
# R     (M,3,3)
#OUTPUT:
# SCALAR
def getRConsistency(R):

    r1 = R[1:]
    r2 = R[:-1]

    error = torch.mean(torch.pow(r1 - r2,2))
    return error

#INPUT:
# pI        (M,2,N)
# pW        (N,3)
# kinv      (3,3)
# R         (M,3,3)
# T         (M,3)
#OUTPUT:
# SCALAR
def get3DConsistency(pI,pW,kinv,R,T):
    kinv[0,0] = 1/100
    kinv[1,1] = 1/100
    M = pI.shape[0]
    N = pI.shape[2]

    pC = torch.bmm(R,torch.stack(M*[pW.T])) + T.unsqueeze(2)
    z = pC[:,2,:]
    ones = torch.ones(M,1,N).to(pI.device)

    pI_h = torch.cat((pI,ones),dim=1)
    pI_proj = torch.bmm(torch.stack(M*[kinv]),pI_h)
    pC_proj = pI_proj * z.unsqueeze(1)

    le_gt = torch.mean(pW[36:42,:],dim=0)
    re_gt = torch.mean(pW[42:48,:],dim=0)
    d_gt = torch.norm(le_gt - re_gt)

    le = torch.mean(pC_proj[:,:,36:42],dim=2)
    re = torch.mean(pC_proj[:,:,42:48],dim=2)
    d = torch.norm(le - re,dim=1)

    return torch.mean(torch.abs(d - d_gt))

# batched reprojection error using intrinsics and extrinsics on world coordinates
#
#INPUT:
#   xcam                (M,3,N)
#   pw                  (N,3)
#   R                   (M,3,3)
#   T                   (M,3)
#   A                   (3,3)
#
#OUTPUT:
#   error               (M)
'''
def getRelReprojError3(xcam,pw,R,T):

    M = xcam.shape[0]
    pc = torch.bmm(R,torch.stack(M*[pw]).permute(0,2,1))
    pct = pc + T.unsqueeze(2)

    #import pptk
    #x1 = xcam[1].T
    #x2 = pct[1].T
    #x1 = x1.cpu().data.numpy()
    #x2 = x2.cpu().data.numpy()
    #pts = np.concatenate((x1,x2),axis=0)

    diff = pct - xcam
    d = torch.norm(xcam,p=2,dim=1)
    error = torch.norm(diff,p=2,dim=1)

    #print(torch.mean(error/d,dim=1).mean())
    #v = pptk.viewer(pts)
    #v.set(point_size=10)
    #quit()
    return torch.mean(error / d,dim=1)
'''

def getRelReprojError3(xcam,pw,R,T):

    M = xcam.shape[0]
    pc = torch.bmm(R,torch.stack(M*[pw]).permute(0,2,1))
    pct = pc + T.unsqueeze(2)

    dgt = torch.norm(torch.mean(pct,dim=2),dim=1)
    d = torch.norm(torch.mean(xcam,dim=2),dim=1)
    diff = torch.abs(dgt - d)

    return diff / d

# batched reprojection error using intrinsics and extrinsics on world coordinates
#
#INPUT:
#   pimg                (M,2,N)
#   pw                  (N,3)
#   R                   (M,3,3)
#   T                   (M,3)
#   A                   (3,3)
#
#OUTPUT:
#   error               (M)
def getReprojError2_(pimg,pct,A,show=False,loss='l2'):

    M = pimg.shape[0]
    N = pimg.shape[2]
    proj = torch.bmm(torch.stack(M*[A]),pct)
    proj_img = proj / proj[:,-1,:].unsqueeze(1)

    pimg_pred = proj_img[:,:2,:]
    diff = pimg - pimg_pred
    if loss == 'l2':
        error  = torch.norm(diff,p=2,dim=1).mean(1)
    elif loss == 'l1':
        error = torch.abs(diff).mean(1).mean(1)

    if show:
        #import pptk
        #x = pct[-1].T.detach().cpu().numpy()
        #v = pptk.viewer(x)
        #v.set(point_size=1.1)
        #for i in range(M):
        #    pt1 = pimg[i].T.cpu().numpy()
        #    scatter(pt1)
        pta = pimg[0].T.cpu().numpy()
        ptb = pimg[-1].T.cpu().numpy()
        ptc = pimg_pred[0].detach().T.cpu().numpy()
        ptd = pimg_pred[-1].detach().T.cpu().numpy()
        plt.scatter(pta[:,0],pta[:,1],s=15,facecolors='none',edgecolors='green')
        #plt.scatter(ptb[:,0],pta[:,1],s=15,facecolors='none',edgecolors='green')
        plt.scatter(ptb[:,0],ptb[:,1],s=10,marker='.',edgecolors='red')
        #plt.scatter(ptb[:,0],pta[:,1],s=10,marker='.',edgecolors='red')

        #plt.xlim((-320,320))
        #plt.ylim((-240,240))
        plt.show()

        quit()

    return error


# batched reprojection error using intrinsics and extrinsics on world coordinates
#
#INPUT:
#   pimg                (M,2,N)
#   pw                  (N,3)
#   R                   (M,3,3)
#   T                   (M,3)
#   A                   (3,3)
#
#OUTPUT:
#   error               (M)
def getReprojError2(pimg,pw,R,T,A,show=False,loss='l2'):

    M = pimg.shape[0]
    N = pimg.shape[2]
    pc = torch.bmm(R,torch.stack(M*[pw]).permute(0,2,1))
    pct = pc + T.unsqueeze(2)
    proj = torch.bmm(torch.stack(M*[A]),pct)
    proj_img = proj / proj[:,-1,:].unsqueeze(1)
    pimg_pred = proj_img[:,:2,:]

    #diff1 = pimg - pimg_pred
    #diff2 = pimg*-1 - pimg_pred
    #if loss == 'l2':
    #    error1 = torch.norm(diff1,p=2,dim=1)
    #    error2 = torch.norm(diff2,p=2,dim=1)
    #elif loss == 'l1':
    #    error1 = torch.abs(diff1)
    #    error2 = torch.abs(diff2)
    #error = error1 if torch.mean(error1) < torch.mean(error2) else error2

    error = torch.norm(pimg - pimg_pred,dim=1).mean()

    if show:
        for i in range(pimg.shape[0]):
            pta = pimg[i].T.cpu().numpy()
            ptc = pimg_pred[i].detach().T.cpu().numpy()

            plt.scatter(pta[:,0],pta[:,1],s=15,facecolors='none',edgecolors='green')
            plt.scatter(ptc[:,0],ptc[:,1],s=10,marker='.',edgecolors='red')
            plt.show()
            break

    return error
# computes the beta values as b11,b12,b22 in that order using distance constraints
#Input:
#   v                   (Mx12x2)
#   d                   (6)
#
#Output:
#   betas               (Mx3x1)
def getBetaN2(v,d):
    views = v.shape[0]
    v1 = v[:,:3,:]
    v2 = v[:,3:6,:]
    v3 = v[:,6:9,:]
    v4 = v[:,9:12,:]

    if v.is_cuda:
        M = torch.zeros((views,6,3)).cuda()
    else:
        M = torch.zeros((views,6,3))

    M00 = torch.sum(v1[:,:,0]**2 - 2*v1[:,:,0]*v2[:,:,0] + v2[:,:,0]**2,dim=1)
    M01 = torch.sum(2*v1[:,:,0]*v1[:,:,1] - 2*v1[:,:,0]*v2[:,:,1] - 2*v1[:,:,1]*v2[:,:,0] + 2*v2[:,:,0]*v2[:,:,1],dim=1)
    M02 = torch.sum(v1[:,:,1]**2 - 2*v1[:,:,1]*v2[:,:,1] + v2[:,:,1]**2,dim=1)
    M10 = torch.sum(v1[:,:,0]**2 - 2*v1[:,:,0]*v3[:,:,0] + v3[:,:,0]**2,dim=1)
    M11 = torch.sum(2*v1[:,:,0]*v1[:,:,1] - 2*v1[:,:,0]*v3[:,:,1] - 2*v1[:,:,1]*v3[:,:,0] + 2*v3[:,:,0]*v3[:,:,1],dim=1)
    M12 = torch.sum(v1[:,:,1]**2 - 2*v1[:,:,1]*v3[:,:,1] + v3[:,:,1]**2,dim=1)
    M20 = torch.sum(v1[:,:,0]**2 - 2*v1[:,:,0]*v4[:,:,0] + v4[:,:,0]**2,dim=1)
    M21 = torch.sum(2*v1[:,:,0]*v1[:,:,1] - 2*v1[:,:,0]*v4[:,:,1] - 2*v1[:,:,1]*v4[:,:,0] + 2*v4[:,:,0]*v4[:,:,1],dim=1)
    M22 = torch.sum(v1[:,:,1]**2 - 2*v1[:,:,1]*v4[:,:,1] + v4[:,:,1]**2,dim=1)
    M30 = torch.sum(v2[:,:,0]**2 - 2*v2[:,:,0]*v3[:,:,0] + v3[:,:,0]**2,dim=1)
    M31 = torch.sum(2*v2[:,:,0]*v2[:,:,1] - 2*v2[:,:,0]*v3[:,:,1] - 2*v2[:,:,1]*v3[:,:,0] + 2*v3[:,:,0]*v3[:,:,1],dim=1)
    M32 = torch.sum(v2[:,:,1]**2 - 2*v2[:,:,1]*v3[:,:,1] + v3[:,:,1]**2,dim=1)
    M40 = torch.sum(v2[:,:,0]**2 - 2*v2[:,:,0]*v4[:,:,0] + v4[:,:,0]**2,dim=1)
    M41 = torch.sum(2*v2[:,:,0]*v2[:,:,1] - 2*v2[:,:,0]*v4[:,:,1] - 2*v2[:,:,1]*v4[:,:,0] + 2*v4[:,:,0]*v4[:,:,1],dim=1)
    M42 = torch.sum(v2[:,:,1]**2 - 2*v2[:,:,1]*v4[:,:,1] + v4[:,:,1]**2,dim=1)
    M50 = torch.sum(v3[:,:,0]**2 - 2*v3[:,:,0]*v4[:,:,0] + v4[:,:,0]**2,dim=1)
    M51 = torch.sum(2*v3[:,:,0]*v3[:,:,1] - 2*v3[:,:,0]*v4[:,:,1] - 2*v3[:,:,1]*v4[:,:,0] + 2*v4[:,:,0]*v4[:,:,1],dim=1)
    M52 = torch.sum(v3[:,:,1]**2 - 2*v3[:,:,1]*v4[:,:,1] + v4[:,:,1]**2,dim=1)

    M[:,0,0] = M00
    M[:,0,1] = M01
    M[:,0,2] = M02
    M[:,1,0] = M10
    M[:,1,1] = M11
    M[:,1,2] = M12
    M[:,2,0] = M20
    M[:,2,1] = M21
    M[:,2,2] = M22
    M[:,3,0] = M30
    M[:,3,1] = M31
    M[:,3,2] = M32
    M[:,4,0] = M40
    M[:,4,1] = M41
    M[:,4,2] = M42
    M[:,5,0] = M50
    M[:,5,1] = M51
    M[:,5,2] = M52

    #mtm_inv_mt = torch.bmm(torch.inverse(torch.bmm(M.permute(0,2,1),M)),M.permute(0,2,1))
    #betas = torch.bmm(mtm_inv_mt,torch.stack(views*[d]).unsqueeze(-1))
    #mtb[0]

    # not sure which would be more stable solution
    # solve for betas using inverse of MtM
    #MtM = torch.bmm(M.permute(0,2,1),M)
    #b = torch.stack(views*[d])
    #MtM_inv = torch.inverse(MtM)
    #Mtb = torch.bmm(M.permute(0,2,1),b.unsqueeze(-1))
    #beta = torch.bmm(MtM_inv,Mtb)
    #beta = torch.bmm(torch.bmm(torch.inverse(MtM), M.permute(0,2,1)),b.unsqueeze(-1))

    # solve lstsq using qr decomposition
    q,r = torch.qr(M)
    b = torch.stack(views*[d])
    qtb = torch.bmm(q.permute(0,2,1),b.unsqueeze(-1))
    betas = torch.bmm(torch.inverse(r),qtb)
    #try:
    #betas = torch.bmm(torch.inverse(r),qtb)
    #except:
    #    noise = 1e-4 * r.mean().detach() * torch.rand(r.shape).to(r.device)
    #    betas = torch.bmm(torch.inverse(r + noise),qtb)

    # solve lstsq using svd on batch
    # https://gist.github.com/gngdb/611d8f180ef0f0baddaa539e29a4200e
    '''
    U_double,D_double,V_double = torch.svd(M.double())
    U = U_double.float()
    D = D_double.float()
    V = V_double.float()
    b = torch.stack(views*[d])
    Utb = torch.bmm(U.permute(0,2,1),b.unsqueeze(-1))
    D_inv = torch.diag_embed(1.0/D)
    VS = torch.bmm(V,D_inv)
    betas = torch.bmm(VS, Utb)
    '''

    return betas

# assumes betas are in order b11,b12,b22
#
#INPUT:
#   v               (Mx12x2)
#   beta            (Mx4)
#OUTPUT:
#   c_c             (Mx3x4)
def getControlPointsN2(v,beta):
    M = v.shape[0]
    b1 = beta[:,2]
    b2 = beta[:,3]
    #b1 = torch.sqrt(torch.abs(beta[:,0]))
    #b2 = torch.sqrt(torch.abs(beta[:,2])) * torch.sign(beta[:,1]) * torch.sign(beta[:,0])
    p = v[:,:,0]*b1.unsqueeze(1) + v[:,:,1]*b2.unsqueeze(1)

    c_c = p.reshape((M,4,3)).permute(0,2,1)
    return c_c

# optimize via gauss newton the betas
# get the optimzed rotation, translation, and camera coord
def optimize_betas_gauss_newton(km, cw, betas, alphas, x_w, x_img, A):

    M = km.shape[0]
    beta_opt, err = gauss_newton(km,betas)

    # compute control points using optimized betas
    kmsum = beta_opt[:,0].unsqueeze(1)*km[:,:,0] + beta_opt[:,1].unsqueeze(1)*km[:,:,1] + beta_opt[:,2].unsqueeze(1)*km[:,:,2] + beta_opt[:,3].unsqueeze(1)*km[:,:,3]
    c_c = kmsum.reshape((M,4,3)).permute(0,2,1)

    # check sign of the determinent to keep orientation
    sign1 = torch.sign(torch.det(cw[:3,:3]))
    sign2 = sign_determinant(c_c)

    # get extrinsics
    cc = c_c * (sign1*sign2).unsqueeze(1).unsqueeze(1)
    rep_alpha = torch.stack(M*[alphas])
    Xc_opt = torch.bmm(cc,rep_alpha)

    return Xc_opt

def gauss_newton(km,betas,max_iter=10):
    L = compute_L6_10(km)
    rho = compute_rho().to(km.device)

    # repeat below code for more iterations of gauss newton, but 4-5 should be enough
    for i in range(max_iter):
        betas,err = gauss_newton_step(betas,rho,L)

    return betas, err

def gauss_newton_step(betas,rho,L):
    M = betas.shape[0]
    A,b = computeJacobian(betas,rho,L)

    ata = torch.bmm(A.permute(0,2,1),A)
    q,r = torch.qr(A)
    qtb = torch.bmm(q.permute(0,2,1),b.unsqueeze(-1))
    r_inv = torch.inverse(r)

    rinv_qtb = torch.bmm(r_inv,qtb)
    next_betas = betas.unsqueeze(-1) + rinv_qtb

    error = torch.bmm(b.view((M,1,6)),b.view((M,6,1)))
    return next_betas.squeeze(-1), error

# compute the derivatives of the eigenvector summation for gauss newton
#
#INPUT:
# km            (M,12,4)
#OUTPUT:
# L             (M,6,10)
def compute_L6_10(km):

    M = km.shape[0]
    L = torch.zeros((M,6,10)).to(km.device)
    v1 = km[:,:,0]
    v2 = km[:,:,1]
    v3 = km[:,:,2]
    v4 = km[:,:,3]

    # compute differenes
    dx112 = v1[:,0] - v1[:,3];
    dx113 = v1[:,0] - v1[:,6];
    dx114 = v1[:,0] - v1[:,9];
    dx123 = v1[:,3] - v1[:,6];
    dx124 = v1[:,3] - v1[:,9];
    dx134 = v1[:,6] - v1[:,9];
    dy112 = v1[:,1] - v1[:,4];
    dy113 = v1[:,1] - v1[:,7];
    dy114 = v1[:,1] - v1[:,10];
    dy123 = v1[:,4] - v1[:,7];
    dy124 = v1[:,4] - v1[:,10];
    dy134 = v1[:,7] - v1[:,10];
    dz112 = v1[:,2] - v1[:,5];
    dz113 = v1[:,2] - v1[:,8];
    dz114 = v1[:,2] - v1[:,11];
    dz123 = v1[:,5] - v1[:,8];
    dz124 = v1[:,5] - v1[:,11];
    dz134 = v1[:,8] - v1[:,11];

    dx212 = v2[:,0] - v2[:,3];
    dx213 = v2[:,0] - v2[:,6];
    dx214 = v2[:,0] - v2[:,9];
    dx223 = v2[:,3] - v2[:,6];
    dx224 = v2[:,3] - v2[:,9];
    dx234 = v2[:,6] - v2[:,9];
    dy212 = v2[:,1] - v2[:,4];
    dy213 = v2[:,1] - v2[:,7];
    dy214 = v2[:,1] - v2[:,10];
    dy223 = v2[:,4] - v2[:,7];
    dy224 = v2[:,4] - v2[:,10];
    dy234 = v2[:,7] - v2[:,10];
    dz212 = v2[:,2] - v2[:,5];
    dz213 = v2[:,2] - v2[:,8];
    dz214 = v2[:,2] - v2[:,11];
    dz223 = v2[:,5] - v2[:,8];
    dz224 = v2[:,5] - v2[:,11];
    dz234 = v2[:,8] - v2[:,11];

    dx312 = v3[:,0] - v3[:,3];
    dx313 = v3[:,0] - v3[:,6];
    dx314 = v3[:,0] - v3[:,9];
    dx323 = v3[:,3] - v3[:,6];
    dx324 = v3[:,3] - v3[:,9];
    dx334 = v3[:,6] - v3[:,9];
    dy312 = v3[:,1] - v3[:,4];
    dy313 = v3[:,1] - v3[:,7];
    dy314 = v3[:,1] - v3[:,10];
    dy323 = v3[:,4] - v3[:,7];
    dy324 = v3[:,4] - v3[:,10];
    dy334 = v3[:,7] - v3[:,10];
    dz312 = v3[:,2] - v3[:,5];
    dz313 = v3[:,2] - v3[:,8];
    dz314 = v3[:,2] - v3[:,11];
    dz323 = v3[:,5] - v3[:,8];
    dz324 = v3[:,5] - v3[:,11];
    dz334 = v3[:,8] - v3[:,11];

    dx412 = v4[:,0] - v4[:,3];
    dx413 = v4[:,0] - v4[:,6];
    dx414 = v4[:,0] - v4[:,9];
    dx423 = v4[:,3] - v4[:,6];
    dx424 = v4[:,3] - v4[:,9];
    dx434 = v4[:,6] - v4[:,9];
    dy412 = v4[:,1] - v4[:,4];
    dy413 = v4[:,1] - v4[:,7];
    dy414 = v4[:,1] - v4[:,10];
    dy423 = v4[:,4] - v4[:,7];
    dy424 = v4[:,4] - v4[:,10];
    dy434 = v4[:,7] - v4[:,10];
    dz412 = v4[:,2] - v4[:,5];
    dz413 = v4[:,2] - v4[:,8];
    dz414 = v4[:,2] - v4[:,11];
    dz423 = v4[:,5] - v4[:,8];
    dz424 = v4[:,5] - v4[:,11];
    dz434 = v4[:,8] - v4[:,11];

    L[:,0,0] =        dx112 * dx112 + dy112 * dy112 + dz112 * dz112;      #b1*b1
    L[:,0,1] = 2.0 *  (dx112 * dx212 + dy112 * dy212 + dz112 * dz212);    #b1*b2
    L[:,0,2] =        dx212 * dx212 + dy212 * dy212 + dz212 * dz212;      #b2*b2
    L[:,0,3] = 2.0 *  (dx112 * dx312 + dy112 * dy312 + dz112 * dz312);    #b1*b3
    L[:,0,4] = 2.0 *  (dx212 * dx312 + dy212 * dy312 + dz212 * dz312);    #b2*b3
    L[:,0,5] =        dx312 * dx312 + dy312 * dy312 + dz312 * dz312;      #b3*b3
    L[:,0,6] = 2.0 *  (dx112 * dx412 + dy112 * dy412 + dz112 * dz412);    #b1*b4
    L[:,0,7] = 2.0 *  (dx212 * dx412 + dy212 * dy412 + dz212 * dz412);    #b2*b4
    L[:,0,8] = 2.0 *  (dx312 * dx412 + dy312 * dy412 + dz312 * dz412);    #b3*b4
    L[:,0,9] =       dx412 * dx412 + dy412 * dy412 + dz412 * dz412;      #b4*b4

    L[:,1,0] =        dx113 * dx113 + dy113 * dy113 + dz113 * dz113;
    L[:,1,1] = 2.0 *  (dx113 * dx213 + dy113 * dy213 + dz113 * dz213);
    L[:,1,2] =        dx213 * dx213 + dy213 * dy213 + dz213 * dz213;
    L[:,1,3] = 2.0 *  (dx113 * dx313 + dy113 * dy313 + dz113 * dz313);
    L[:,1,4] = 2.0 *  (dx213 * dx313 + dy213 * dy313 + dz213 * dz313);
    L[:,1,5] =        dx313 * dx313 + dy313 * dy313 + dz313 * dz313;
    L[:,1,6] = 2.0 *  (dx113 * dx413 + dy113 * dy413 + dz113 * dz413);
    L[:,1,7] = 2.0 *  (dx213 * dx413 + dy213 * dy413 + dz213 * dz413);
    L[:,1,8] = 2.0 *  (dx313 * dx413 + dy313 * dy413 + dz313 * dz413);
    L[:,1,9] =       dx413 * dx413 + dy413 * dy413 + dz413 * dz413;

    L[:,2,0] =        dx114 * dx114 + dy114 * dy114 + dz114 * dz114;
    L[:,2,1] = 2.0 *  (dx114 * dx214 + dy114 * dy214 + dz114 * dz214);
    L[:,2,2] =        dx214 * dx214 + dy214 * dy214 + dz214 * dz214;
    L[:,2,3] = 2.0 *  (dx114 * dx314 + dy114 * dy314 + dz114 * dz314);
    L[:,2,4] = 2.0 *  (dx214 * dx314 + dy214 * dy314 + dz214 * dz314);
    L[:,2,5] =        dx314 * dx314 + dy314 * dy314 + dz314 * dz314;
    L[:,2,6] = 2.0 *  (dx114 * dx414 + dy114 * dy414 + dz114 * dz414);
    L[:,2,7] = 2.0 *  (dx214 * dx414 + dy214 * dy414 + dz214 * dz414);
    L[:,2,8] = 2.0 *  (dx314 * dx414 + dy314 * dy414 + dz314 * dz414);
    L[:,2,9] =       dx414 * dx414 + dy414 * dy414 + dz414 * dz414;

    L[:,3,0] =        dx123 * dx123 + dy123 * dy123 + dz123 * dz123;
    L[:,3,1] = 2.0 *  (dx123 * dx223 + dy123 * dy223 + dz123 * dz223);
    L[:,3,2] =        dx223 * dx223 + dy223 * dy223 + dz223 * dz223;
    L[:,3,3] = 2.0 *  (dx123 * dx323 + dy123 * dy323 + dz123 * dz323);
    L[:,3,4] = 2.0 *  (dx223 * dx323 + dy223 * dy323 + dz223 * dz323);
    L[:,3,5] =        dx323 * dx323 + dy323 * dy323 + dz323 * dz323;
    L[:,3,6] = 2.0 *  (dx123 * dx423 + dy123 * dy423 + dz123 * dz423);
    L[:,3,7] = 2.0 *  (dx223 * dx423 + dy223 * dy423 + dz223 * dz423);
    L[:,3,8] = 2.0 *  (dx323 * dx423 + dy323 * dy423 + dz323 * dz423);
    L[:,3,9] =       dx423 * dx423 + dy423 * dy423 + dz423 * dz423;

    L[:,4,0] =        dx124 * dx124 + dy124 * dy124 + dz124 * dz124;
    L[:,4,1] = 2.0 *  (dx124 * dx224 + dy124 * dy224 + dz124 * dz224);
    L[:,4,2] =        dx224 * dx224 + dy224 * dy224 + dz224 * dz224;
    L[:,4,3] = 2.0 * ( dx124 * dx324 + dy124 * dy324 + dz124 * dz324);
    L[:,4,4] = 2.0 * (dx224 * dx324 + dy224 * dy324 + dz224 * dz324);
    L[:,4,5] =        dx324 * dx324 + dy324 * dy324 + dz324 * dz324;
    L[:,4,6] = 2.0 * ( dx124 * dx424 + dy124 * dy424 + dz124 * dz424);
    L[:,4,7] = 2.0 * ( dx224 * dx424 + dy224 * dy424 + dz224 * dz424);
    L[:,4,8] = 2.0 * ( dx324 * dx424 + dy324 * dy424 + dz324 * dz424);
    L[:,4,9] =       dx424 * dx424 + dy424 * dy424 + dz424 * dz424;

    L[:,5,0] =        dx134 * dx134 + dy134 * dy134 + dz134 * dz134;
    L[:,5,1] = 2.0 * ( dx134 * dx234 + dy134 * dy234 + dz134 * dz234);
    L[:,5,2] =        dx234 * dx234 + dy234 * dy234 + dz234 * dz234;
    L[:,5,3] = 2.0 * ( dx134 * dx334 + dy134 * dy334 + dz134 * dz334);
    L[:,5,4] = 2.0 * ( dx234 * dx334 + dy234 * dy334 + dz234 * dz334);
    L[:,5,5] =        dx334 * dx334 + dy334 * dy334 + dz334 * dz334;
    L[:,5,6] = 2.0 *  (dx134 * dx434 + dy134 * dy434 + dz134 * dz434);
    L[:,5,7] = 2.0 *  (dx234 * dx434 + dy234 * dy434 + dz234 * dz434);
    L[:,5,8] = 2.0 *  (dx334 * dx434 + dy334 * dy434 + dz334 * dz434);
    L[:,5,9] =       dx434 * dx434 + dy434 * dy434 + dz434 * dz434;

    return L

# so we don't use cw here since we use the same control points in the world system
# but if you don't do this, you must change rho accordingly using cw
#
#INPUT:
# cw            (4x4)
#OUTPUT:
# rho           (6)
def compute_rho():

    rho = torch.zeros(6);
    rho[0] = 2
    rho[1] = 2
    rho[2] = 1
    rho[3] = 2
    rho[4] = 1
    rho[5] = 1
    return rho

def computeJacobian_(current_betas,rho,L):

    device = current_betas.device
    M = current_betas.shape[0]
    A = torch.zeros((M,6,4)).to(device)
    b = torch.zeros((M,6)).to(device)
    B = torch.zeros((M,10)).to(device)

    cb = current_betas

    B[:,0] = cb[:,0] * cb[:,0]
    B[:,1] = cb[:,0] * cb[:,1]
    B[:,2] = cb[:,1] * cb[:,1]
    B[:,3] = cb[:,0] * cb[:,2]
    B[:,4] = cb[:,1] * cb[:,2]
    B[:,5] = cb[:,2] * cb[:,2]
    B[:,6] = cb[:,0] * cb[:,3]
    B[:,7] = cb[:,1] * cb[:,3]
    B[:,8] = cb[:,2] * cb[:,3]
    B[:,9] = cb[:,3] * cb[:,3]

    A[:,0,0]=2*cb[:,0]*L[:,0,0]+cb[:,1]*L[:,0,1]+cb[:,2]*L[:,0,3]+cb[:,3]*L[:,0,6];
    A[:,0,1]=cb[:,0]*L[:,0,1]+2*cb[:,1]*L[:,0,2]+cb[:,2]*L[:,0,4]+cb[:,3]*L[:,0,7];
    A[:,0,2]=cb[:,0]*L[:,0,3]+cb[:,1]*L[:,0,4]+2*cb[:,2]*L[:,0,5]+cb[:,3]*L[:,0,8];
    A[:,0,3]=cb[:,0]*L[:,0,6]+cb[:,1]*L[:,0,7]+cb[:,2]*L[:,0,8]+2*cb[:,3]*L[:,0,9];
    b[:,0] = rho[0]-torch.bmm(L[:,0,:].view((M,1,10)),B.view((M,10,1))).squeeze();

    A[:,1,0]=2*cb[:,0]*L[:,1,0]+cb[:,1]*L[:,1,1]+cb[:,2]*L[:,1,3]+cb[:,3]*L[:,1,6];
    A[:,1,1]=cb[:,0]*L[:,1,1]+2*cb[:,1]*L[:,1,2]+cb[:,2]*L[:,1,4]+cb[:,3]*L[:,1,7];
    A[:,1,2]=cb[:,0]*L[:,1,3]+cb[:,1]*L[:,1,4]+2*cb[:,2]*L[:,1,5]+cb[:,3]*L[:,1,8];
    A[:,1,3]=cb[:,0]*L[:,1,6]+cb[:,1]*L[:,1,7]+cb[:,2]*L[:,1,8]+2*cb[:,3]*L[:,1,9];
    b[:,1] = rho[1]-torch.bmm(L[:,1,:].view((M,1,10)),B.view((M,10,1))).squeeze();

    A[:,2,0]=2*cb[:,0]*L[:,2,0]+cb[:,1]*L[:,2,1]+cb[:,2]*L[:,2,3]+cb[:,3]*L[:,2,6];
    A[:,2,1]=cb[:,0]*L[:,2,1]+2*cb[:,1]*L[:,2,2]+cb[:,2]*L[:,2,4]+cb[:,3]*L[:,2,7];
    A[:,2,2]=cb[:,0]*L[:,2,3]+cb[:,1]*L[:,2,4]+2*cb[:,2]*L[:,2,5]+cb[:,3]*L[:,2,8];
    A[:,2,3]=cb[:,0]*L[:,2,6]+cb[:,1]*L[:,2,7]+cb[:,2]*L[:,2,8]+2*cb[:,3]*L[:,2,9];
    b[:,2] = rho[2]-torch.bmm(L[:,2,:].view((M,1,10)),B.view((M,10,1))).squeeze();

    A[:,3,0]=2*cb[:,0]*L[:,3,0]+cb[:,1]*L[:,3,1]+cb[:,2]*L[:,3,3]+cb[:,3]*L[:,3,6];
    A[:,3,1]=cb[:,0]*L[:,3,1]+2*cb[:,1]*L[:,3,2]+cb[:,2]*L[:,3,4]+cb[:,3]*L[:,3,7];
    A[:,3,2]=cb[:,0]*L[:,3,3]+cb[:,1]*L[:,3,4]+2*cb[:,2]*L[:,3,5]+cb[:,3]*L[:,3,8];
    A[:,3,3]=cb[:,0]*L[:,3,6]+cb[:,1]*L[:,3,7]+cb[:,2]*L[:,3,8]+2*cb[:,3]*L[:,3,9];
    b[:,3] = rho[3]-torch.bmm(L[:,3,:].view((M,1,10)),B.view((M,10,1))).squeeze();

    A[:,4,0]=2*cb[:,0]*L[:,4,0]+cb[:,1]*L[:,4,1]+cb[:,2]*L[:,4,3]+cb[:,3]*L[:,4,6];
    A[:,4,1]=cb[:,0]*L[:,4,1]+2*cb[:,1]*L[:,4,2]+cb[:,2]*L[:,4,4]+cb[:,3]*L[:,4,7];
    A[:,4,2]=cb[:,0]*L[:,4,3]+cb[:,1]*L[:,4,4]+2*cb[:,2]*L[:,4,5]+cb[:,3]*L[:,4,8];
    A[:,4,3]=cb[:,0]*L[:,4,6]+cb[:,1]*L[:,4,7]+cb[:,2]*L[:,4,8]+2*cb[:,3]*L[:,4,9];
    b[:,4] = rho[4]-torch.bmm(L[:,4,:].view((M,1,10)),B.view((M,10,1))).squeeze();

    A[:,5,0]=2*cb[:,0]*L[:,5,0]+cb[:,1]*L[:,5,1]+cb[:,2]*L[:,5,3]+cb[:,3]*L[:,5,6];
    A[:,5,1]=cb[:,0]*L[:,5,1]+2*cb[:,1]*L[:,5,2]+cb[:,2]*L[:,5,4]+cb[:,3]*L[:,5,7];
    A[:,5,2]=cb[:,0]*L[:,5,3]+cb[:,1]*L[:,5,4]+2*cb[:,2]*L[:,5,5]+cb[:,3]*L[:,5,8];
    A[:,5,3]=cb[:,0]*L[:,5,6]+cb[:,1]*L[:,5,7]+cb[:,2]*L[:,5,8]+2*cb[:,3]*L[:,5,9];
    b[:,5] = rho[5]-torch.bmm(L[:,5,:].view((M,1,10)),B.view((M,10,1))).squeeze();

    return A,b


def computeJacobian(current_betas,rho,L):

    device = current_betas.device
    M = current_betas.shape[0]
    A = torch.zeros((M,6,4)).to(device)
    b = torch.zeros((M,6)).to(device)
    B = torch.zeros((M,10)).to(device)

    cb = current_betas

    B[:,0] = cb[:,0] * cb[:,0]
    B[:,1] = cb[:,0] * cb[:,1]
    B[:,2] = cb[:,1] * cb[:,1]
    B[:,3] = cb[:,0] * cb[:,2]
    B[:,4] = cb[:,1] * cb[:,2]
    B[:,5] = cb[:,2] * cb[:,2]
    B[:,6] = cb[:,0] * cb[:,3]
    B[:,7] = cb[:,1] * cb[:,3]
    B[:,8] = cb[:,2] * cb[:,3]
    B[:,9] = cb[:,3] * cb[:,3]

    A[:,0,0]=2*cb[:,0]*L[:,0,0]+cb[:,1]*L[:,0,1]+cb[:,2]*L[:,0,3]+cb[:,3]*L[:,0,6];
    A[:,0,1]=cb[:,0]*L[:,0,1]+2*cb[:,1]*L[:,0,2]+cb[:,2]*L[:,0,4]+cb[:,3]*L[:,0,7];
    A[:,0,2]=cb[:,0]*L[:,0,3]+cb[:,1]*L[:,0,4]+2*cb[:,2]*L[:,0,5]+cb[:,3]*L[:,0,8];
    A[:,0,3]=cb[:,0]*L[:,0,6]+cb[:,1]*L[:,0,7]+cb[:,2]*L[:,0,8]+2*cb[:,3]*L[:,0,9];
    b[:,0] = rho[0]-torch.bmm(L[:,0,:].view((M,1,10)),B.view((M,10,1))).squeeze();

    A[:,1,0]=2*cb[:,0]*L[:,1,0]+cb[:,1]*L[:,1,1]+cb[:,2]*L[:,1,3]+cb[:,3]*L[:,1,6];
    A[:,1,1]=cb[:,0]*L[:,1,1]+2*cb[:,1]*L[:,1,2]+cb[:,2]*L[:,1,4]+cb[:,3]*L[:,1,7];
    A[:,1,2]=cb[:,0]*L[:,1,3]+cb[:,1]*L[:,1,4]+2*cb[:,2]*L[:,1,5]+cb[:,3]*L[:,1,8];
    A[:,1,3]=cb[:,0]*L[:,1,6]+cb[:,1]*L[:,1,7]+cb[:,2]*L[:,1,8]+2*cb[:,3]*L[:,1,9];
    b[:,1] = rho[1]-torch.bmm(L[:,1,:].view((M,1,10)),B.view((M,10,1))).squeeze();

    A[:,2,0]=2*cb[:,0]*L[:,2,0]+cb[:,1]*L[:,2,1]+cb[:,2]*L[:,2,3]+cb[:,3]*L[:,2,6];
    A[:,2,1]=cb[:,0]*L[:,2,1]+2*cb[:,1]*L[:,2,2]+cb[:,2]*L[:,2,4]+cb[:,3]*L[:,2,7];
    A[:,2,2]=cb[:,0]*L[:,2,3]+cb[:,1]*L[:,2,4]+2*cb[:,2]*L[:,2,5]+cb[:,3]*L[:,2,8];
    A[:,2,3]=cb[:,0]*L[:,2,6]+cb[:,1]*L[:,2,7]+cb[:,2]*L[:,2,8]+2*cb[:,3]*L[:,2,9];
    b[:,2] = rho[2]-torch.bmm(L[:,2,:].view((M,1,10)),B.view((M,10,1))).squeeze();

    A[:,3,0]=2*cb[:,0]*L[:,3,0]+cb[:,1]*L[:,3,1]+cb[:,2]*L[:,3,3]+cb[:,3]*L[:,3,6];
    A[:,3,1]=cb[:,0]*L[:,3,1]+2*cb[:,1]*L[:,3,2]+cb[:,2]*L[:,3,4]+cb[:,3]*L[:,3,7];
    A[:,3,2]=cb[:,0]*L[:,3,3]+cb[:,1]*L[:,3,4]+2*cb[:,2]*L[:,3,5]+cb[:,3]*L[:,3,8];
    A[:,3,3]=cb[:,0]*L[:,3,6]+cb[:,1]*L[:,3,7]+cb[:,2]*L[:,3,8]+2*cb[:,3]*L[:,3,9];
    b[:,3] = rho[3]-torch.bmm(L[:,3,:].view((M,1,10)),B.view((M,10,1))).squeeze();

    A[:,4,0]=2*cb[:,0]*L[:,4,0]+cb[:,1]*L[:,4,1]+cb[:,2]*L[:,4,3]+cb[:,3]*L[:,4,6];
    A[:,4,1]=cb[:,0]*L[:,4,1]+2*cb[:,1]*L[:,4,2]+cb[:,2]*L[:,4,4]+cb[:,3]*L[:,4,7];
    A[:,4,2]=cb[:,0]*L[:,4,3]+cb[:,1]*L[:,4,4]+2*cb[:,2]*L[:,4,5]+cb[:,3]*L[:,4,8];
    A[:,4,3]=cb[:,0]*L[:,4,6]+cb[:,1]*L[:,4,7]+cb[:,2]*L[:,4,8]+2*cb[:,3]*L[:,4,9];
    b[:,4] = rho[4]-torch.bmm(L[:,4,:].view((M,1,10)),B.view((M,10,1))).squeeze();

    A[:,5,0]=2*cb[:,0]*L[:,5,0]+cb[:,1]*L[:,5,1]+cb[:,2]*L[:,5,3]+cb[:,3]*L[:,5,6];
    A[:,5,1]=cb[:,0]*L[:,5,1]+2*cb[:,1]*L[:,5,2]+cb[:,2]*L[:,5,4]+cb[:,3]*L[:,5,7];
    A[:,5,2]=cb[:,0]*L[:,5,3]+cb[:,1]*L[:,5,4]+2*cb[:,2]*L[:,5,5]+cb[:,3]*L[:,5,8];
    A[:,5,3]=cb[:,0]*L[:,5,6]+cb[:,1]*L[:,5,7]+cb[:,2]*L[:,5,8]+2*cb[:,3]*L[:,5,9];
    b[:,5] = rho[5]-torch.bmm(L[:,5,:].view((M,1,10)),B.view((M,10,1))).squeeze();

    return A,b

def sign_determinant(c):

    c0 = c[:,:,0]
    c1 = c[:,:,1]
    c2 = c[:,:,2]
    c3 = c[:,:,3]

    v1 = c0 - c3
    v2 = c1 - c3
    v3 = c2 - c3
    M = torch.stack((v1,v2,v3),2)
    signs = torch.sign(torch.det(M))

    return signs

def Rx(x):
    batchsize = x.shape[0]
    sinx = torch.sin(x)
    cosx = torch.cos(x)
    Rx = torch.zeros((batchsize,3,3)).cuda()
    Rx[:,0,0] = 1
    Rx[:,0,1] = 0
    Rx[:,0,2] = 0
    Rx[:,1,0] = 0
    Rx[:,1,1] = cosx
    Rx[:,1,2] = -sinx
    Rx[:,2,0] = 0
    Rx[:,2,1] = sinx
    Rx[:,2,2] = cosx
    return Rx

def Ry(x):
    batchsize = x.shape[0]
    sinx = torch.sin(x)
    cosx = torch.cos(x)
    Ry = torch.zeros((batchsize,3,3)).cuda()
    Ry[:,0,0] = cosx
    Ry[:,0,1] = 0
    Ry[:,0,2] = sinx
    Ry[:,1,0] = 0
    Ry[:,1,1] = 1
    Ry[:,1,2] = 0
    Ry[:,2,0] = -sinx
    Ry[:,2,1] = 0
    Ry[:,2,2] = cosx
    return Ry

def Rz(x):
    batchsize = x.shape[0]
    sinx = torch.sin(x)
    cosx = torch.cos(x)
    Rz = torch.zeros((batchsize,3,3)).cuda()
    Rz[:,0,0] = cosx
    Rz[:,0,1] = -sinx
    Rz[:,0,2] = 0
    Rz[:,1,0] = sinx
    Rz[:,1,1] = cosx
    Rz[:,1,2] = 0
    Rz[:,2,0] = 0
    Rz[:,2,1] = 0
    Rz[:,2,2] = 1
    return Rz

def R(thetax,thetay,thetaz):
    rx = Rx(thetax)
    ry = Ry(thetay)
    rz = Rz(thetaz)
    return torch.bmm(rz,torch.bmm(ry,rx))

def quat2euler(quat):

    dist = torch.norm(quat,p=2,dim=0)
    q = quat / dist.unsqueeze(0)
    rx = torch.atan2(2*(q[0]*q[1] + q[2]*q[3]),1-2*(q[1]**2+q[2]**2));
    ry = torch.asin(2*(q[0]*q[2] - q[3]*q[1]));
    rz = torch.atan2(2*(q[0]*q[3] + q[1]*q[2]),1-2*(q[2]**2+q[3]**2));

    return rx, ry,rz

# euler angles in x,y,z
def euler2rotm(rx,ry,rz):
    Rx = np.array([[1,0,0],[0,torch.cos(x),-torch.sin(x)],[0,torch.sin(x),torch.cos(x)]])
    Ry = np.array([[torch.cos(y),0,torch.sin(y)],[0,1,0],[-torch.sin(y),0,torch.cos(y)]])
    Rz = np.array([[torch.cos(z),-torch.sin(z),0],[torch.sin(z),torch.cos(z),0],[0,0,1]])
    return Rx @ Ry @ Rz

# create 3DMM using alphas for shape eigen vectors, and betas for expression eigen vectors
def create3DMM(mu_s, mu_exp, s_eigen, exp_eigen, alphas, betas):
    shape_cov = torch.matmul(s_eigen,alphas)
    exp_cov = torch.matmul(exp_eigen,betas)

    shape = (mu_s + shape_cov.view((53215,3))) + (mu_exp + exp_cov.view((53215,3)))

    return shape/1000

# rotate and translate the shape according to rotation translation and scale factor
def align(shape,s,R,T):
    return s*(torch.matmul(shape,R) + T)

# apply orthographic projection
def project(shape):
    ortho = torch.Tensor([[1,0,0],[0,1,0]]).float().cuda()
    return torch.matmul(shape,ortho.T)

# predict a 3DMM model according to parameters
def predict(s,R,T,alphas,betas):
    shape = create3DMM(alphas,betas)
    shape = align(shape,s,R,T)
    shape = project(shape)
    return shape

# visualize 2d points
def drawpts(img, pts, size=1,fill=-1,color=[255,255,255]):

    for p in pts:
        cv2.circle(img,(int(p[0]),int(p[1])),size,color,fill)

    return img

# project a set of points using camera projection params
# projection of x2 onto x1 using K,R,T
def project(s,K,R,T):
    proj = np.matmul(R, s.T) + T[:, np.newaxis]
    proj = np.matmul(K,proj)
    proj = proj / np.expand_dims(proj[2,:],axis=0)
    return proj[:2,:].T

def scatter(pts,color=[255,0,0]):
    plt.scatter(pts[:,0],pts[:,1])
    return

# extrinsic matrix defined by 6 degrees of freedom
def getA(rx,ry,rz,tx,ty,tz):
    batchsize = rx.size()[0]
    R = torch.zeros((batchsize,3,4)).cuda()
    Rx = torch.zeros((batchsize,3,3)).cuda()
    Ry = torch.zeros((batchsize,3,3)).cuda()
    Rz = torch.zeros((batchsize,3,3)).cuda()
    cosrx = torch.cos(rx)
    sinrx = torch.sin(rx)
    cosry = torch.cos(ry)
    sinry = torch.sin(ry)
    cosrz = torch.cos(rz)
    sinrz = torch.sin(rz)
    Rx[:,0,0] = cosrx
    Rx[:,0,1] = sinrx
    Rx[:,1,0] = -sinrx
    Rx[:,1,1] = cosrx
    Rx[:,2,2] = 1
    Ry[:,0,0] = 1
    Ry[:,1,1] = cosry
    Ry[:,1,2] = sinry
    Ry[:,2,1] = -sinry
    Ry[:,2,2] = cosry
    Rz[:,0,0] = cosrz
    Rz[:,0,1] = sinrz
    Rz[:,1,0] = -sinrz
    Rz[:,1,1] = cosrz
    Rz[:,2,2] = 1

    R[:,:3,:3] = torch.bmm(torch.bmm(Rz,Ry),Rx)
    R[:,0,3] = tx
    R[:,1,3] = ty
    R[:,2,3] = tz
    R[tz < -1,:,3] = R[tz < -1,:,3]*-1

    return R

# batched epnp alg
# x_i       (m,n,2)
# x_w       (m,n,3)
# K         (m,3,3)
def EPnP_(x_i,x_w,K):
    b = x_i.shape[0]

    # get control points
    c_w = getControlPoints(x_w).to(x_i.device).unsqueeze(0).repeat(b,1,1)

    # solve alphas
    alphas = solveAlphas_(x_w,c_w)

    # create matrix
    Matrix = setupM_(alphas,x_i,K)

    # get svd
    u,d,v = torch.svd(Matrix)
    km = v[:,:,-4:]

    # solve N = 1
    beta_n1 = torch.zeros((b,4)).to(x_i.device)
    beta_n1[:,3] = 1
    c_c_n1 = km[:,:,-1].reshape((b,4,3)).permute(0,2,1)
    s = scaleControlPoints_(c_c_n1,c_w,alphas,x_w)
    scaled_betas = beta_n1 * s.unsqueeze(1)

    # solve extrinsics
    Xc, R, T = optimizeGN_(km,c_w,scaled_betas,alphas,x_w,x_i)

    return Xc, R, T

# epnp algorithm to solve for camera pose with gauss newton
#INPUT:
# x_img         (M,2,N)
# x_w           (N,3)
# K             (M,3,3)
def EPnP_single(x_img,x_w,K):
    M = x_img.shape[0]
    N = x_img.shape[2]
    f = K[:,0,0]
    px = K[:,0,2]
    py = K[:,1,2]

    # get control points
    c_w = getControlPoints(x_w)

    # solve alphas
    alphas = solveAlphas(x_w,c_w)
    if x_img.is_cuda:
        alphas = alphas.cuda()
    Matrix = setupM_single(alphas,x_img.permute(0,2,1),px,py,f)

    if ~torch.all(Matrix == Matrix):
        print(Matrix)
        print(f)
        print(x_w)
        return

    u,d,v = torch.svd(Matrix)
    km = v[:,:,-4:]

    # solve N=1
    beta_n1 = torch.zeros((M,4))
    if x_img.is_cuda:
        beta_n1 = beta_n1.cuda()
    beta_n1[:,3] = 1
    c_c_n1 = km[:,:,-1].reshape((M,4,3)).permute(0,2,1)
    _, x_c_n1,s1 = scaleControlPoints(c_c_n1,c_w[:3,:],alphas,x_w)
    mask1 = s1 == s1
    Rn1, Tn1 = getExtrinsics(x_c_n1[mask1],x_w)

    return km, c_w, beta_n1 / s1.unsqueeze(1), alphas

# epnp algorithm to solve for camera pose with gauss newton
#INPUT:
# x_img         (M,2,N)
# x_w           (N,3)
# K             (3,3)
def EPnP(x_img,x_w,K):
    M = x_img.shape[0]
    N = x_img.shape[2]
    f = K[0,0]
    px = K[0,2]
    py = K[1,2]

    # get control points
    c_w = getControlPoints(x_w)

    # solve alphas
    alphas = solveAlphas(x_w,c_w)
    if x_img.is_cuda:
        alphas = alphas.cuda()
    Matrix = setupM(alphas,x_img.permute(0,2,1),px,py,f)

    if ~torch.all(Matrix == Matrix):
        print(Matrix)
        print(f)
        print(x_w)
        return

    u,d,v = torch.svd(Matrix)
    km = v[:,:,-4:]

    # solve N=1
    beta_n1 = torch.zeros((M,4))
    if x_img.is_cuda:
        beta_n1 = beta_n1.cuda()
    beta_n1[:,3] = 1
    c_c_n1 = km[:,:,-1].reshape((M,4,3)).permute(0,2,1)
    _, x_c_n1,s1 = scaleControlPoints(c_c_n1,c_w[:3,:],alphas,x_w)
    mask1 = s1 == s1
    Rn1, Tn1 = getExtrinsics(x_c_n1[mask1],x_w)
    reproj_error2_n1 = getReprojError2(x_img[mask1],x_w,Rn1,Tn1,K,loss='l2')

    return km, c_w, beta_n1 / s1.unsqueeze(1), alphas

    # solve N=2
    d12, d13, d14, d23, d24, d34 = getDistances(c_w)
    distances = torch.stack([d12,d13,d14,d23,d24,d34])**2
    betasq_n2 = getBetaN2(km[:,:,-2:],distances).squeeze()
    b1_n2 = torch.sqrt(torch.abs(betasq_n2[:,0]))
    b2_n2 = torch.sqrt(torch.abs(betasq_n2[:,2])) * torch.sign(betasq_n2[:,1]) * torch.sign(betasq_n2[:,0]).detach()
    beta_n2 = torch.zeros((M,4))
    if x_img.is_cuda:
        beta_n2 = beta_n2.cuda()
    beta_n2[:,2] = b1_n2
    beta_n2[:,3] = b2_n2
    c_c_n2 = getControlPointsN2(km[:,:,-2:],beta_n2)
    _,x_c_n2,s2 = scaleControlPoints(c_c_n2,c_w[:3,:],alphas,x_w)
    mask2 = s2 == s2
    Rn2,Tn2 = getExtrinsics(x_c_n2[mask2],x_w)
    reproj_error2_n2 = getReprojError2(x_img[mask2],x_w,Rn2,Tn2,K,loss='l2')

    if torch.any(mask1 != mask2):
        error1 = torch.zeros(M).to(reproj_error2_n1.device) + 10000
        error2 = torch.zeros(M).to(reproj_error2_n2.device) + 10000
        error1[mask1] = reproj_error2_n1
        error2[mask2] = reproj_error2_n2
    else:
        error1 = reproj_error2_n1
        error2 = reproj_error2_n2


    s = torch.stack((s1,s2))

    # determine best solution in terms of 2d reprojection
    mask = error1 > error2
    mask = mask.long()
    betas = torch.stack((beta_n1,beta_n2))
    best_betas = betas.gather(0,torch.stack(4*[mask],dim=1).unsqueeze(0)).squeeze()
    best_scale = s.gather(0,mask.unsqueeze(0)).squeeze()
    scaled_betas = best_betas / best_scale.unsqueeze(1)

    return km, c_w, scaled_betas, alphas

# DLT with single 3D shape and multiple 2D images
# pi        (M,2,N)
# pw        (M,N,3)
def DLT(pi,pw):
    M = pi.shape[0]
    N = pi.shape[2]
    V = torch.zeros((M,2*N,12)).to(pi.device)

    V[:,0::2,0] = pw[:,:,0]
    V[:,0::2,1] = pw[:,:,1]
    V[:,0::2,2] = pw[:,:,2]
    V[:,0::2,3] = torch.ones((M,N))
    V[:,0::2,4] = torch.zeros((M,N))
    V[:,0::2,5] = torch.zeros((M,N))
    V[:,0::2,6] = torch.zeros((M,N))
    V[:,0::2,7] = torch.zeros((M,N))
    V[:,0::2,8] = -pi[:,0,:]*pw[:,:,0]
    V[:,0::2,9] = -pi[:,0,:]*pw[:,:,1]
    V[:,0::2,10] = -pi[:,0,:]*pw[:,:,2]
    V[:,0::2,11] = -pi[:,0,:]
    V[:,1::2,0] = torch.zeros((M,N))
    V[:,1::2,1] = torch.zeros((M,N))
    V[:,1::2,2] = torch.zeros((M,N))
    V[:,1::2,3] = torch.zeros((M,N))
    V[:,1::2,4] = pw[:,:,0]
    V[:,1::2,5] = pw[:,:,1]
    V[:,1::2,6] = pw[:,:,2]
    V[:,1::2,7] = torch.ones((M,N))
    V[:,1::2,8] = -pi[:,1,:]*pw[:,:,0]
    V[:,1::2,9] = -pi[:,1,:]*pw[:,:,1]
    V[:,1::2,10] = -pi[:,1,:]*pw[:,:,2]
    V[:,1::2,11] = -pi[:,1,:]

    u,d,v = torch.svd(V)
    P = v[:,:,-1]
    P = P.reshape((M,3,4))
    return P

# optimize gauss newton on extrinsics
# km        (m,12,4)
# c_w       (m,4,4)
# scaled_betas      (m,3,4)
# alphas            (m,4,68)
# x_w               (m,68,3)
# x_i               (m,68,2)
def optimizeGN_(km,c_w,scaled_betas,alphas,x_w,x_img):

    b = km.shape[0]

    # optimize betas
    beta_opt, err = gauss_newton(km,scaled_betas)

    # get camera coordinates
    kmsum = beta_opt[:,0].unsqueeze(1)*km[:,:,0] + beta_opt[:,1].unsqueeze(1)*km[:,:,1] + beta_opt[:,2].unsqueeze(1)*km[:,:,2] + beta_opt[:,3].unsqueeze(1)*km[:,:,3]
    c_c = kmsum.reshape((b,4,3)).permute(0,2,1)

    # check sign of the determinent to keep orientation
    sign1 = torch.sign(torch.det(c_w[:,:3,:3]))
    sign2 = sign_determinant(c_c)

    # get extrinsics
    cc = c_c * (sign1*sign2).unsqueeze(1).unsqueeze(1)
    xc_opt = torch.bmm(cc,alphas)

    # fix the sign
    xc_opt = xc_opt * torch.sign(xc_opt[:,2,:].sum(1)).unsqueeze(-1).unsqueeze(-1)

    # get the extrinsics
    r_opt,t_opt = getExtrinsics_(xc_opt.permute(0,2,1),x_w)

    return xc_opt, r_opt, t_opt

# Gauss Newton Optimization on extrinsic
#INPUT:
#   km      (M,12,4)
#   c_w     (4,4)
#   scaled_betas    (M,4)
#   alphas          (4,N)
#   x_w             (N,3)
#   x_img           (M,2,N)
def optimizeGN(km,c_w,scaled_betas,alphas,x_w,x_img):
    M = km.shape[0]
    beta_opt, err = gauss_newton(km,scaled_betas)

    # get camera coordinates
    kmsum = beta_opt[:,0].unsqueeze(1)*km[:,:,0] + beta_opt[:,1].unsqueeze(1)*km[:,:,1] + beta_opt[:,2].unsqueeze(1)*km[:,:,2] + beta_opt[:,3].unsqueeze(1)*km[:,:,3]
    c_c = kmsum.reshape((M,4,3)).permute(0,2,1)

    # check sign of the determinent to keep orientation
    sign1 = torch.sign(torch.det(c_w[:3,:3]))
    sign2 = sign_determinant(c_c)

    # get extrinsics
    cc = c_c * (sign1*sign2).unsqueeze(1).unsqueeze(1)
    rep_alpha = torch.stack(M*[alphas])
    xc_opt = torch.bmm(cc,rep_alpha)

    # fix the sign
    xc_opt = xc_opt * torch.sign(xc_opt[:,2,:].sum(1)).unsqueeze(-1).unsqueeze(-1)

    # get the extrinsics
    r_opt,t_opt = getExtrinsics(xc_opt.permute(0,2,1),x_w)

    return xc_opt, r_opt, t_opt, beta_opt

# scatter 3d points
def scatter3d(pts):
    v = pptk.viewer(pts.detach().cpu().numpy())
    v.set(point_size=0.2)
    return v

# get rodriguez angles from rotation matrices.
# we use stackoverflow solution here: https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix
# output angles are in zyx
def rotm2eul(R):
    if len(R.shape) == 2:
        rx = np.arctan2(R[2,1],R[2,2])
        ry = np.arctan2(-R[2,0],np.sqrt(R[2,1]*R[2,1] + R[2,2]*R[2,2]))
        rz = np.arctan2(R[1,0],R[0,0])
        return np.hstack([rx,ry,rz])
    if len(R.shape) == 3:
        rx = np.arctan2(R[:,2,1],R[:,2,2])
        ry = np.arctan2(-R[:,2,0],np.sqrt(R[:,2,1]*R[:,2,1] + R[:,2,2]*R[:,2,2]))
        rz = np.arctan2(R[:,1,0],R[:,0,0])
        return np.stack([rz,ry,rx],axis=1)

# solve rotation translation
def rotm2axisangle(R):
    if len(R.shape) == 2:
        theta = np.arccos((np.trace(R) - 1) / 2)
        axis = np.hstack([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
        return axis*theta
    if len(R.shape) == 3:
        #axis = K.rotation_matrix_to_angle_axis(R)
        #return axis.detach().cpu().numpy()

        diag = np.trace(R,axis1=1,axis2=2)
        diag[diag <= -1] = diag[diag < -1] * -1
        diag[diag >= 3] = 1
        theta = np.arccos((diag-1) / 2)
        axis = np.stack([R[:,2,1]-R[:,1,2],R[:,0,2]-R[:,2,0],R[:,1,0]-R[:,0,1]],axis=1)
        return axis*theta[:,np.newaxis]

def create_K(param):
    M = param.shape[0]
    K = torch.zeros((M,3,3)).float().to(param.device)
    K[:,0,0] = param[:,0]
    K[:,1,1] = param[:,0]
    K[:,0,2] = param[:,1]
    K[:,1,2] = param[:,2]
    K[:,2,2] = 1
    return K

# plot input 2D points as 3x3 grid
def show_input(data):
    fig,ax = plt.subplots(3,3,figsize=(12,12))
    x = data['x_img'].clone()
    x = x[::100 // 9]
    s = 320

    for i in range(3):
        for j in range(3):
            canvas = np.ones((s,s,3))
            lm = x[3*i+j]
            lm[:,0] = (lm[:,0] - 320)
            lm[:,1] = (240 - lm[:,1])
            lm = lm / (torch.amax(torch.abs(lm)) + 20)
            lm = lm * (s/2) + s//2
            canvas = drawpts(canvas,lm,size=2,color=[1,0,0])
            ax[i,j].set_title(f"frame {int((3*i+j)*100/9):03d}")
            ax[i,j].imshow(canvas)
            ax[i,j].tick_params(axis='both',which='both',labelsize=0)

    plt.tight_layout()
    fig.subplots_adjust(hspace=.25)
    plt.show()

# save images to create video from
def save_video_log(opt,data):
    num_iter = len(opt.log['f'])
    plt.ion()
    fig = plt.figure(figsize=(12,12))
    gs = fig.add_gridspec(4,5)
    ax0 = fig.add_subplot(gs[0:3,0:3],projection='3d')
    ax1 = fig.add_subplot(4,5,4)
    ax2 = fig.add_subplot(4,5,5)
    ax3 = fig.add_subplot(4,5,9)
    ax4 = fig.add_subplot(4,5,10)
    ax5 = fig.add_subplot(4,5,14)
    ax6 = fig.add_subplot(4,5,15)
    ax7 = fig.add_subplot(4,5,16)
    ax8 = fig.add_subplot(4,5,17)
    ax9 = fig.add_subplot(4,5,18)
    ax10 = fig.add_subplot(4,5,19)
    ax11 = fig.add_subplot(4,5,20)

    # draw synthetic images used
    img_axs = [ax7,ax8,ax9,ax10,ax11]
    x = data['x_img'].clone()
    x = x[::100//5]
    s = 320
    for i in range(5):
        canvas = np.ones((s,s,3))
        lm = x[i]
        lm[:,0] = (lm[:,0] - 320)
        lm[:,1] = (240 - lm[:,1])
        lm = lm / (torch.amax(torch.abs(lm)) + 20)
        lm = lm * (s / 2) + s// 2
        canvas = drawpts(canvas,lm,size=2,color=[1,0,0])
        img_axs[i].set_title(f"frame {int(20*i):03d}")
        img_axs[i].imshow(canvas)
        img_axs[i].tick_params(axis='both',which='both',labelsize=0)

    S = opt.gt['S']
    S_x = opt.log['s_pred'][0]
    ax0.scatter(S[:,0],S[:,1],S[:,2],'b')
    d0 = ax0.scatter(S_x[:,0],S_x[:,1],S_x[:,2],'r')
    d1 = ax1.scatter(np.arange(num_iter),np.array(opt.log['e2d']))
    d2 = ax2.scatter(np.arange(num_iter),np.array(opt.log['f']),label='prediction')
    d3 = ax3.scatter(np.arange(num_iter),np.array(opt.log['px']),label='prediction')
    d4 = ax4.scatter(np.arange(num_iter),np.array(opt.log['py']),label='prediction')
    d5 = ax5.scatter(np.arange(num_iter),np.array(opt.log['derr']))
    d6 = ax6.scatter(np.arange(num_iter),np.array(opt.log['rmse']))

    # set labels
    ax0.set_title(f"RMSE: {opt.log['rmse'][0]:.3f}",fontsize=14)
    ax1.set_ylabel('Reprojection Error',fontsize=14)
    ax2.set_ylabel('Focal Length',fontsize=14)
    ax3.set_ylabel('X Principal Point',fontsize=14)
    ax4.set_ylabel('Y Principal Point',fontsize=14)
    ax5.set_ylabel('Depth Error',fontsize=14)
    ax6.set_ylabel('Shape Error',fontsize=14)

    # show gt values
    if 'f' in opt.gt:
        ax2.axhline(y=opt.gt['f'].item(),color='g',label='gt')
    if 'px' in opt.gt:
        ax3.axhline(y=opt.gt['px'].item(),color='g',label='gt')
    if 'py' in opt.gt:
        ax4.axhline(y=opt.gt['py'].item(),color='g',label='gt')

    # show optimization of variables
    x_val = np.arange(num_iter)
    plt.tight_layout()
    fig.subplots_adjust(hspace=.22)
    for i in range(num_iter):

        s_x = opt.log['s_pred'][i]
        d0._offsets3d = (s_x[:,0],s_x[:,1],s_x[:,2])
        #ax0.scatter(S[:,0],S[:,1],S[:,2],'b')
        #ax0.scatter(S_x[:,0],S_x[:,1],S_x[:,2],'r')
        ax0.set_title(f"RMSE: {opt.log['rmse'][i]:.3f}",fontsize=16)
        d1.set_offsets(np.c_[x_val[:i],np.array(opt.log['e2d'][:i])])
        d2.set_offsets(np.c_[x_val[:i],np.array(opt.log['f'][:i])])
        d3.set_offsets(np.c_[x_val[:i],np.array(opt.log['px'][:i])])
        d4.set_offsets(np.c_[x_val[:i],np.array(opt.log['py'][:i])])
        d5.set_offsets(np.c_[x_val[:i],np.array(opt.log['derr'][:i])])
        d6.set_offsets(np.c_[x_val[:i],np.array(opt.log['rmse'][:i])])

        # plot and save each iteration
        fig.canvas.draw()
        fig.canvas.flush_events()
        if i == 0: continue
        fig.savefig(os.path.join('supp_results',f"iter{i:03d}.png"), bbox_inches='tight', pad_inches=0,dpi=100)

    plt.ioff()
    plt.close()

if __name__ == '__main__':

    F = torch.rand((100,3,3)).float()
    A = torch.rand((3,3)).float()
    u,d,v = torch.svd(F)

    d = torch.stack(100*[torch.eye(3)]) * d.unsqueeze(-1)
    F = torch.bmm(torch.bmm(u,d),v)

    mendoncaError(F,A)

    # import dataloader
    #
    # #facemodel = dataloader.Face3DMM()
    # M = 100;
    # N = 68;
    # synth = dataloader.TestLoader(400)
    # sequence = synth[0]
    # x_cam_gt = sequence['x_cam_gt']
    # x_w_gt = sequence['x_w_gt']
    # f_gt = sequence['f_gt']
    # x_img = sequence['x_img']
    # data3dmm = dataloader.SyntheticLoader()
    # mu_lm = torch.from_numpy(data3dmm.mu_lm).float()
    # shape = mu_lm
    # shape[:,2] = shape[:,2] * -1;
    #
    # one  = torch.ones(M,1,68)
    # x_img_one = torch.cat([x_img,one],dim=1)
    #
    # #f = torch.relu(out[:,199]).mean()
    # error2d = []
    # error3d = []
    # fvals = []
    # f_gt = 400
    # for diff in np.linspace(-200,200,20):
    #     K = torch.zeros((3,3)).float()
    #     f = f_gt + diff
    #     fvals.append(f)
    #     K[0,0] = f;
    #     K[1,1] = f;
    #     K[2,2] = 1;
    #     K[0,2] = 320;
    #     K[1,2] = 240;
    #     px = 320;
    #     py = 240;
    #
    #     # get control points
    #     Xc, R, T = EPnP(x_img,shape,K)
    #
    #     reproj_error2 = getReprojError2(x_img,shape,R,T,K)
    #     reproj_error3 = getReprojError3(x_cam_gt,shape,R,T)
    #     rel_error = getRelReprojError3(x_cam_gt,shape,R,T)
    #
    #     error2d.append(reproj_error2.mean().item())
    #     error3d.append(rel_error.mean().item())
    #
    # data = {}
    # data['fvals'] = np.array(fvals)
    # data['error2d'] = np.array(error2d)
    # data['error3d'] = np.array(error3d)
    #
    # print(fvals)
    # print(error2d)
    # print(error3d)
    #
    # import scipy.io
    # scipy.io.savemat('exp4.mat',data)
    # quit()
    #
    # print(torch.mean(reproj_error2))
    # print(torch.mean(reproj_error3))
    # print(torch.mean(rel_error))
    # quit()

    '''
    mu_s = facemodel.mu_shape
    mu_exp = facemodel.mu_exp
    s_eigen = facemodel.shape_eigenvec
    exp_eigen = facemodel.exp_eigenvec
    lm = facemodel.lm

    alphas  = torch.matmul(torch.randn(199),torch.eye(199)).float().cuda()
    betas = torch.matmul(torch.randn(29),torch.eye(29)).float().cuda()

    euler = np.random.rand(3)
    R = torch.Tensor(euler2rotm(euler)).float().cuda()
    T = torch.randn((1,3)).float().cuda() * 10
    s = torch.randn(1).float().cuda()

    shape = create3DMM(mu_s,mu_exp,s_eigen,exp_eigen,alphas,betas)
    shape = align(shape,s,R,T)
    shape = project(shape)

    keypoints = shape[lm,:]
    print(shape.shape)
    print(keypoints.shape)

    pts = keypoints.detach().cpu().numpy()
    print(pts.shape)

    import scipy.io
    scipy.io.savemat('pts.mat',{'pts': pts})
    '''
