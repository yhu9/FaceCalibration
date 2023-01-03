
import os
import random

from skimage import io, transform
import scipy.io
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
import numpy as np
from torch.utils.data import Dataset, DataLoader
#import pptk

import util
import torch
# the main data class which has iterators to all datasets
# add more datasets accordingly


def get_root_dir(db):
    if db == "cad120":
        return os.path.join('..','data','rgbd-human','cad-120')
    elif db == 'biwi':
        return os.path.join('..','data','rgbd-human','kinect_head_pose_db','hpdb')
    elif db == 'biwiid':
        return os.path.join('..','data','rgbd-human','BIWI-I')
    elif db == 'human36':
        return os.path.join('..','data','human3.6m_downloader','training','subject')
    else:
        return ''

class Data():
    def __init__(self,db='all',face3dmm='lm'):
        if face3dmm == 'lm':
            self.dataloader = SyntheticLoader()
        else:
            self.dataloader = SyntheticLoader2()
        self.M = self.dataloader.M
        self.N = self.dataloader.N
        self.batchsize = 4
        self.shuffle = True
        self.transform = True
        self.batchloader = DataLoader(self.dataloader,
                batch_size=self.batchsize,
                shuffle=self.shuffle,
                num_workers=4)

    def __len__(self):
        return len(self.dataloader)

    def printinfo(self):
        print(f"RANDOM TRANSFORMS: TRUE")
        print(f"SHUFFLE: TRUE")
        print(f"BATCH SIZE: {self.batchsize}")
        print()

class TestData():
    def __init__(self):
        self.batchsize = 4

    def createLoader(self,f):
        self.dataloader = TestLoader(f)
        self.batchloader = DataLoader(self.dataloader,
                batch_size = self.batchsize,
                shuffle=False,
                num_workers=0)

        return self.batchloader

    def __len__(self):
        return len(self.dataloader)


class ImageLoader():

    def __init__(self,db='biwiid',addnoise=False):
        root_dir = get_root_dir(db)
        get_img_loader(db)

class TestLoader(Dataset):
    def __init__(self,f):
        root_dir = os.path.join("../data/synthetic_principal",f"sequencef{f:04d}")

        self.root_dir = root_dir
        self.files = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]
        self.files = sorted(self.files,key=str.lower)
        self.M = 100
        self.N = 68

        self.w = 640
        self.h = 480
        self.px = self.w / 2
        self.py = self.h / 2

    def __len__(self):

        return len(self.files)

    def __getitem__(self,idx):
        fname = self.files[idx]
        data = scipy.io.loadmat(fname)
        self.M = 100
        self.N = 68

        tmp = data['sequence'][0,0]
        x_w = tmp['x_w']
        x_img_gt = tmp['x_img_true']
        x_cam = tmp['x_cam']
        R = tmp['R']
        T = tmp['T']
        f = torch.Tensor(tmp['f'].astype(np.float)[0]).float()
        d = np.mean(T[:,2])

        x_img = x_img_gt

        sample = {}
        sample['fname'] = fname
        sample['x_w_gt'] = torch.from_numpy(x_w - x_w.mean(0)).float()
        sample['x_cam_gt'] = torch.from_numpy(x_cam).float()
        sample['x_img'] = torch.from_numpy(x_img).float()
        sample['x_img_gt'] = torch.from_numpy(x_img_gt).float()
        #sample['x_img_gt'] = torch.from_numpy(x_img_gt).float().permute(0,2,1)
        sample['f_gt'] = torch.Tensor(f).float()
        sample['d_gt'] = torch.Tensor([d]).float()
        sample['T_gt'] = T

        if 'K' in data.keys():
            sample['K'] = torch.from_numpy(data['K']).float()
        else:
            sample['K'] = torch.tensor([[f,0,self.w/2],[0,f,self.h/2],[0,0,1]],dtype=torch.float32)

        return sample

class Cad60Loader(Dataset):
    def __init__(self):
        if os.path.isdir("../data"):
            self.root_dir = os.path.join("../data/tmp/cad60/processed")
        else:
            self.root_dir = os.path.join("../data0/tmp/cad60/processed")
        files = os.listdir(self.root_dir)
        sorted(files,key=str.lower)

        self.all_paths = [os.path.join(self.root_dir,f) for f in files]

    def __len__(self):

        return len(self.all_paths)

    def __getitem__(self,idx):
        path = self.all_paths[idx]
        #print(f"load: {path}")
        data = scipy.io.loadmat(path)

        x2d = data['x2d']
        xcam = data['xcam']
        f = data['fgt'][0,0]
        M = x2d.shape[0]
        N = x2d.shape[1]

        pts = x2d.reshape((M*N,2))
        x2d = np.transpose(x2d,(0,2,1))
        xcam = np.transpose(xcam,(0,2,1))

        sample = {}
        sample['x_img_gt'] = torch.from_numpy(x2d).float()
        sample['x_img'] = torch.from_numpy(pts).float()
        sample['x_cam_gt'] = torch.from_numpy(xcam).float()
        sample['f_gt'] = torch.from_numpy(np.array([f]).astype(np.float)).float()

        return sample

class Cad120Loader(Dataset):
    def __init__(self):
        if os.path.isdir("../data"):
            #self.root_dir = os.path.join("../data/tmp/cad120/processed")
            #shape_dir = "../data/face_alignment/300W_LP/Code/ModelGeneration/shape_simple.mat"
            self.root_dir = os.path.join("../data/tmp/cad120/processed_orig")
            shape_dir = "../data/face_alignment/300W_LP/Code/ModelGeneration/shape_simple.mat"
        files = os.listdir(self.root_dir)
        files = sorted(files,key=str.lower)

        self.all_paths = [os.path.join(self.root_dir,f) for f in files]
        print(files)

        self.w = 640
        self.h = 480
        self.px = 320
        self.py = 240

    def __len__(self):

        return len(self.all_paths)

    def __getitem__(self,idx):
        path = self.all_paths[idx]
        data = scipy.io.loadmat(path)

        x2d = data['x2d_raw']
        f = data['fgt'][0,0]
        M = x2d.shape[0]
        N = x2d.shape[1]

        x2d = np.transpose(x2d,(0,2,1))

        K_gt = torch.tensor([[f, 0, self.px],[0,f,self.py],[0,0,1]],dtype=torch.float32)

        sample = {}
        if 'imgs' in data.keys(): sample['imgs'] = data['imgs']
        sample['x_w_gt'] = None
        sample['x_img_gt'] = None
        sample['K_gt'] = K_gt
        sample['x_img'] = torch.from_numpy(x2d).float()
        sample['d'] = torch.from_numpy(data['depth'].astype(np.float32)).float().mean(1)
        sample['f_gt'] = torch.from_numpy(np.array([f]).astype(np.float)).float()

        return sample

class Human36Loader(Dataset):
    def __init__(self):
        if os.path.isdir("../data"):
            self.root_dir = os.path.join("../data/tmp/human36/processed")
        else:
            self.root_dir = os.path.join("../data0/tmp/human36/processed")
        files = os.listdir(self.root_dir)
        #files = sorted(files,key=str.lower)
        files.sort()
        self.all_paths = [os.path.join(self.root_dir,f) for f in files]

        self.w = 1000
        self.h = 1000
        self.px = 500
        self.py = 500

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self,idx):
        path = self.all_paths[idx]
        print(path)
        #path = '../data/tmp/human36/processed/S1_Eating 2.55011271_sequence.mat'
        #print(f"load: {path}")
        data = scipy.io.loadmat(path)

        x2d = data['x2d_raw']
        xcam = data['xcam']
        f = data['fgt'][0,0]

        x2d = np.transpose(x2d,(0,2,1))
        xcam = np.transpose(xcam,(0,2,1))

        K_gt = torch.tensor([[f, 0, self.px],[0,f,self.py],[0,0,1]],dtype=torch.float32)

        sample = {}
        sample['x_w_gt'] = None
        sample['x_img_gt'] = None
        sample['K_gt'] = K_gt
        sample['x_img'] = torch.from_numpy(x2d).float()
        sample['x_cam'] = torch.from_numpy(xcam).float()
        sample['d'] = torch.from_numpy(data['depth'].astype(np.float32)).float().mean(1)
        sample['f_gt'] = torch.from_numpy(np.array([f]).astype(np.float)).float()

        return sample

class BIWIID3ddfaLoader(Dataset):

    def __init__(self):
        self.root_dir = os.path.join("/home/huynshen/projects/git/3DDFA_V2/biwiid_faces/processed")
        files = os.listdir(self.root_dir)
        #files = sorted(files,key=str.lower)

        self.all_paths = [os.path.join(self.root_dir,f) for f in files]
        #files.sort()

    def __len__(self):

        return len(self.all_paths)

    def __getitem__(self,idx):
        path = self.all_paths[idx]
        #print(f"load: {path}")
        data = scipy.io.loadmat(path)

        x2d = data['x2d']
        f = data['fgt'][0,0]
        x2d = np.transpose(x2d,(0,2,1))
        #xcam = np.transpose(xcam,(0,2,1))

        sample = {}
        sample['x_w_gt'] = None
        sample['x_img_gt'] = None
        sample['x_img'] = torch.from_numpy(x2d).float()
        sample['x_cam'] = None
        sample['f_gt'] = torch.from_numpy(np.array([f]).astype(np.float)).float()

        return sample

class BIWIIDLoader(Dataset):

    def __init__(self):
        if os.path.isdir("../data"):
            #self.root_dir = os.path.join("../data/tmp/biwi-i/processed")
            self.root_dir = os.path.join("../data/tmp/biwi-i/processed_orig")
        files = os.listdir(self.root_dir)
        #files = sorted(files,key=str.lower)

        files.sort()
        self.all_paths = [os.path.join(self.root_dir,f) for f in files]

        self.w = 640
        self.h = 480
        self.px = 320
        self.py = 240

    def __len__(self):

        return len(self.all_paths)

    def __getitem__(self,idx):
        path = self.all_paths[idx]
        print(f"load: {path}")
        data = scipy.io.loadmat(path,squeeze_me=True)

        x2d = data['x2d_raw']
        #x2d = data['x2d']
        #x2d[:,:,0] = x2d[:,:,0] + self.w
        #x2d[:,:,1] = x2d[:,:,1] + self.h
        x2d[:,:,0] = x2d[:,:,0] / self.w * self.px
        x2d[:,:,1] = x2d[:,:,1] / self.h * self.py
        #xcam = data['xcam']
        f = data['fgt']
        x2d = np.transpose(x2d,(0,2,1))
        #xcam = np.transpose(xcam,(0,2,1))

        K_gt = torch.tensor([[f,0,self.px],[0,f,self.py],[0,0,1]],dtype=torch.float32)

        sample = {}
        sample['fname'] = data['fname']
        sample['x_img'] = torch.from_numpy(x2d).float()
        sample['K_gt'] = K_gt
        #sample['x_cam'] = torch.from_numpy(xcam).float()
        sample['d'] = torch.from_numpy(data['depth'].astype(np.float32)).float().mean(1)
        sample['f_gt'] = torch.from_numpy(np.array([f]).astype(np.float)).float()

        return sample

class CheckerboardLoader(Dataset):
    def __init__(self):

        self.root_dir = '/home/huynshen/projects/git/A-flexible-new-technique-for-camera-calibration/zhang'
        self.file = 'checkerboard.mat'

    def __len__(self):

        return 1

    def __getitem__(self,idx):
        path = os.path.join(self.root_dir,self.file)
        data = scipy.io.loadmat(path,struct_as_record=False,squeeze_me=True)
        Fs = data['data'].Fs
        x_w_gt = data['data'].M
        x_img = data['data'].x
        weights = data['data'].ws
        K_gt = data['data'].K

        sample = {}
        sample['x_img'] = torch.from_numpy(x_img).float().permute(2,0,1)
        sample['x_w_gt'] = torch.from_numpy(x_w_gt).float()
        sample['K_gt'] = torch.from_numpy(K_gt).float()
        sample['Fs'] = torch.from_numpy(Fs).float()
        sample['weights'] = torch.from_numpy(weights).float()
        return sample

class MeritonLoader(Dataset):
    def __init__(self):
        self.root_dir = './extras/meriton_college.mat'
        data_dict = scipy.io.loadmat(self.root_dir,squeeze_me=True,struct_as_record=False)
        self.A = data_dict['data'].A_gt
        self.F = data_dict['data'].F
        pts = data_dict['data'].pts
        obs_file = '/home/huynshen/Downloads/meriton_college_1/2D/nview-corners_masa'
        p_w_file = '/home/huynshen/Downloads/meriton_college_1/3D/p3d'
        observation = np.genfromtxt(obs_file)
        self.p_w = np.genfromtxt(p_w_file)

        m1 = observation[:,0] > 0
        m2 = observation[:,1] > 0
        m3 = observation[:,2] > 0
        m = m1 & m2 & m3

        p1 = pts[0].x2d[observation[:,0][m].astype(np.int),:]
        p2 = pts[1].x2d[observation[:,1][m].astype(np.int),:]
        p3 = pts[2].x2d[observation[:,2][m].astype(np.int),:]
        self.x_img = np.stack((p1,p2,p3))
        self.p_w = self.p_w[m]

    def __len__(self):
        return 1

    def __getitem__(self,idx):

        sample = {}
        sample['x_img'] = torch.from_numpy(self.x_img).float().permute(0,2,1)
        sample['x_w_gt'] = torch.from_numpy(self.p_w).float()
        sample['Fs'] = torch.from_numpy(self.F).float()
        sample['K_gt'] = torch.from_numpy(self.A).float()

        return sample

class BIWILoader(Dataset):
    def __init__(self):
        if os.path.isdir("../data"):
            self.root_dir = os.path.join("../data/tmp/biwi/processed")
            self.shape_dir = os.path.join("../data/tmp/biwi/shape")
        files = os.listdir(self.root_dir)
        files = sorted(files,key=str.lower)

        self.w = 640
        self.h = 480
        self.px = 320
        self.py = 240
        self.res = 300

        self.all_paths = [os.path.join(self.root_dir,f) for f in files]
        self.maxangle = 20

    def __len__(self):

        return len(self.all_paths)

    def __getitem__(self,idx):
        path = self.all_paths[idx]
        #print(f"load: {path}")
        data = scipy.io.loadmat(path)
        print(data.keys())
        quit()

        shape_path = os.path.join(self.shape_dir,f"{idx+1:02d}_shape.mat")
        shape_data = scipy.io.loadmat(shape_path)

        x2d = data['x2d']
        x2d[:,:,0] = x2d[:,:,0] + self.px
        x2d[:,:,1] = x2d[:,:,1] + self.py
        xcam = data['xcam']
        f = data['fgt'][0,0]

        K_gt = torch.tensor([[f,0,self.px],[0,f,self.py],[0,0,1]],dtype=torch.float32)

        x2d = np.transpose(x2d,(0,2,1))
        xcam = np.transpose(xcam,(0,2,1))

        sample = {}
        sample['x_w_gt'] = None
        sample['x_img'] = torch.from_numpy(x2d).float()
        sample['K_gt'] = K_gt
        sample['x_cam'] = torch.from_numpy(xcam).float()
        sample['d'] = torch.norm(sample['x_cam'].mean(2),dim=1)
        sample['f_gt'] = torch.from_numpy(np.array([f]).astype(np.float)).float()

        S = torch.from_numpy(shape_data['mean_lm']).float()
        S = S - S.mean(0).unsqueeze(0)
        S[:,2] = S[:,2] * -1
        sample['shape'] = S

        return sample

class AnalysisLoader(Dataset):
    def __init__(self):
        if os.path.isdir("../data"):
            self.root_dir = os.path.join("../data/tmp/biwi/processed")
            shape_dir = "../data/face_alignment/300W_LP/Code/ModelGeneration/shape_simple.mat"
            self.shape_dir = os.path.join("../data/tmp/biwi/shape")
        else:
            self.root_dir = os.path.join("../data0/tmp/biwi/processed")
            shape_dir = "../data0/face_alignment/300W_LP/Code/ModelGeneration/shape_simple.mat"
            self.shape_dir = os.path.join("../data0/tmp/biwi/shape")
        subjects = [f"{sub:02d}" for sub in range(1,25)]
        files = os.listdir(self.root_dir)
        files = sorted(files,key=str.lower)
        self.all_paths = [os.path.join(self.root_dir,f) for f in files]

        shape_data = scipy.io.loadmat(shape_dir)
        mu_lm = shape_data['mu_lm']
        mu_exp = shape_data['mu_exp']
        lm_eigenvec = shape_data['lm_eigenvec']
        exp_eigenvec = shape_data['exp_eigenvec']

        self.mu_lm = mu_lm.T
        self.mu_lm = self.mu_lm - np.mean(self.mu_lm,0)
        self.mu_exp = mu_exp.T
        self.mu_exp = self.mu_exp - np.mean(self.mu_exp,0)

        self.lm_eigenvec = lm_eigenvec
        self.exp_eigenvec = exp_eigenvec

        self.minangle = 20
        self.maxangle = 20

    def __len__(self):
        return 24

    def __getitem__(self,idx):
        path = self.all_paths[idx]
        #print(f"load: {path}")
        data = scipy.io.loadmat(path)

        shape_path = os.path.join(self.shape_dir,f"{idx+1:02d}_shape.mat")
        shape_data = scipy.io.loadmat(shape_path)

        x_w = shape_data['mean_lm']
        x_w[:,2] = x_w[:,2] * -1
        x2d = data['x2d']
        xcam = data['xcam']
        f = data['fgt'][0,0]
        R = data['R']
        M = x2d.shape[0]
        N = x2d.shape[1]

        # find valid views
        validview = []
        for i in range(M):
            r = Rotation.from_matrix(R[i])
            angles = r.as_euler('zyx',degrees=False)
            angles = np.arcsin(np.sin(angles)) * 180 / np.pi
            if np.any(np.abs(angles) > self.maxangle): continue
            else: validview.append(i)

        x2d = x2d[validview]
        xcam = xcam[validview]
        M = x2d.shape[0]
        N = x2d.shape[1]

        pts = x2d.reshape((M*N,2))
        x2d = np.transpose(x2d,(0,2,1))
        xcam = np.transpose(xcam,(0,2,1))

        sample = {}
        sample['x_w_gt'] = torch.from_numpy(x_w).float()
        sample['x_img_gt'] = torch.from_numpy(x2d).float()
        sample['x_img'] = torch.from_numpy(pts).float()
        sample['x_cam_gt'] = torch.from_numpy(xcam).float()
        sample['f_gt'] = torch.from_numpy(np.array([f]).astype(np.float)).float()

        return sample

# loads M views and N points depending on the setting
# ablation for synthetic dataset
class MNLoader(Dataset):
    def __init__(self,f,M=100,N=68,addnoise=True,seed=0):
        if os.path.isdir("../data"):
            root_dir = os.path.join("../data/synthetic_3dface",f"sequencef{f:04d}")
        else:
            root_dir = os.path.join("../data0/synthetic_3dface",f"sequencef{f:04d}")
        random.seed(seed)
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]
        self.files = sorted(self.files,key=str.lower)
        self.addnoise = addnoise

        #self.transform = transforms.Compose([ToTensor()])
        if os.path.isdir("../data"):
            root_dir = "../data/face_alignment/300W_LP/Code/ModelGeneration/shape_simple.mat"
        else:
            root_dir = "../data0/face_alignment/300w_lp/code/modelgeneration/shape_simple.mat"

        # load shape data
        shape_data = scipy.io.loadmat(root_dir)
        self.mu_s = shape_data['mu_lm'].T
        self.mu_exp = shape_data['mu_exp'].T
        lm_eigenvec = shape_data['lm_eigenvec']
        exp_eigenvec = shape_data['exp_eigenvec']
        sigma = np.diag(shape_data['sigma'][:,0] / 100)

        if N > 68: N=68

        #lm_eigenvec = lm_eigenvec.reshape(53215,3,199)
        #exp_eigenvec = exp_eigenvec.reshape(53215,3,29)
        lm_eigenvec = lm_eigenvec.reshape(68,3,199)
        exp_eigenvec = exp_eigenvec.reshape(68,3,29)
        self.lm_eigenvec = torch.from_numpy(lm_eigenvec[:N].reshape(-1,199)).float()
        self.exp_eigenvec = torch.from_numpy(exp_eigenvec[:N].reshape(-1,29)).float()
        self.mu_s = torch.from_numpy(self.mu_s[:N]).float()
        self.mu_exp = torch.from_numpy(self.mu_exp[:N]).float()
        self.sigma = torch.from_numpy(sigma).float()

        # bideo sequence length
        self.M = M
        self.N = N

        # get indices
        indices = []
        le = list(range(36,42)) + list(range(17,22))
        re = list(range(42,48)) + list(range(22,27))
        nose = list(range(27,36))
        mouth = list(range(48,68))
        jaw = list(range(0,17))
        random.shuffle(le)
        random.shuffle(re)
        random.shuffle(nose)
        random.shuffle(mouth)
        random.shuffle(jaw)
        lm = [le,re,nose,mouth,jaw]
        for i in range(68):
            idx = lm[i % len(lm)].pop()
            indices.append(idx)
            if len(lm[i % len(lm)]) == 0:
                lm.pop(i % len(lm))
        self.indices = indices

    def __len__(self):

        return len(self.files)

    def __getitem__(self,idx):
        fname = self.files[idx]
        data = scipy.io.loadmat(fname)

        tmp = data['sequence'][0,0]
        x_w = tmp['x_w']
        x_img_gt = tmp['x_img_true']
        x_cam = tmp['x_cam']
        R = tmp['R']
        T = tmp['T']
        f = torch.Tensor(tmp['f'].astype(np.float)[0]).float()
        d = np.mean(T[:,2])

        if self.addnoise:
            le = x_img_gt[:,36,:]
            re = x_img_gt[:,45,:]
            std = np.max(np.linalg.norm(le - re,axis=1)*0.05)
            noise = np.random.rand(self.M,self.N,2) * std
        else:
            noise = 0

        x_img_gt = x_img_gt[:self.M,:self.N,:]
        x_w = x_w[:self.N,:]
        x_cam = x_cam[:self.M,:self.N]
        x_img_gt[:,:,0] = x_img_gt[:,:,0] - 320
        x_img_gt[:,:,1] = x_img_gt[:,:,1] - 240
        x_img = x_img_gt + noise
        x_img = x_img.reshape((self.M*self.N,2))

        sample = {}
        sample['x_w_gt'] = torch.from_numpy(x_w).float()
        sample['x_cam_gt'] = torch.from_numpy(x_cam).float().permute(0,2,1)
        sample['x_img'] = torch.from_numpy(x_img).float()
        sample['x_img_gt'] = torch.from_numpy(x_img_gt).float().permute(0,2,1)
        sample['f_gt'] = torch.Tensor(f).float()

        return sample

    def get3DMM_(self,a):
        a = a.unsqueeze(-1)
        delta = torch.mm(torch.mm(self.lm_eigenvec,self.sigma),a).view(self.N,3)
        shape = self.mu_s + delta
        shape = shape - shape.mean(0).unsqueeze(0)
        return shape

class SyntheticLoader2(Dataset):

    def __init__(self):

        #self.transform = transforms.Compose([ToTensor()])
        if os.path.isdir("../data"):
            root_dir = '/home/huynshen/data/face_alignment/300W_LP/Code/ModelGeneration/Model_Shape.mat'

        # load shape data
        shape_data = scipy.io.loadmat(root_dir)
        self.sigma = shape_data['sigma'][:,0]
        self.s_eig = shape_data['shape_eigenvec']
        self.s_mu = shape_data['mu_shape']

        # video sequence length
        self.M = 100
        self.N = self.s_mu.shape[0] // 3

        # extra boundaries on camera coordinates
        self.maxn = 500
        self.minn = 20
        self.w = 640
        self.h = 480
        self.minf = 300; self.maxf = 1500
        self.minz = 800; self.maxz = 8000;
        self.max_rx = 20;
        self.max_ry = 20; self.max_rz = 20;
        self.std = 0.05

    def __len__(self):
        return 10000

    def __getitem__(self,idx):

        # data holders
        M = self.M
        x_w = np.zeros((self.N,3))

        # define intrinsics
        while True:
            f = 500 + np.random.randn() * (self.maxf - self.minf);
            if f >= self.minf and f < self.maxf: break
        K = np.array([[f,0,self.w/2],[ 0,f,self.h/2], [0,0,1]])

        # create random 3dmm shape
        s,alpha = self.generateRandomFace()

        #import pptk
        #v = pptk.viewer(s)
        #v.set(point_size=1.1)

        # create random 3dmm expression
        #beta = np.random.randn(29) * 0.1
        #e = np.sum(np.expand_dims(beta,0)*self.exp_eigenvec,1)
        #exp = e.reshape(N,3)

        # define depth
        tz = np.random.random() * (self.maxz-self.minz) + self.minz
        minz = np.maximum(tz - 500,self.minz)
        maxz = np.minimum(tz + 500,self.maxz)

        # get initial and final rotation
        while True:
            r_init, q_init = self.generateRandomRotation()
            r_final, q_final = self.generateRandomRotation()
            t_init = self.generateRandomTranslation(K,minz,maxz)
            t_final = self.generateRandomTranslation(K,minz,maxz)

            ximg_init = self.project2d(r_init,t_init,K,s)
            ximg_final = self.project2d(r_final,t_final,K,s)
            if np.any(np.amin(ximg_init,axis=0) < -320): continue
            if np.any(np.amin(ximg_final,axis=0) < -320): continue
            if np.any(np.amin(ximg_init,axis=1) < -240): continue
            if np.any(np.amin(ximg_final,axis=1) < -240): continue
            init = np.amax(ximg_init,axis=0)
            final = np.amax(ximg_final,axis=0)
            if init[0] > 320: continue
            if final[0] > 320: continue
            if init[1] > 240: continue
            if final[1] > 240: continue
            break
        d = (t_init[2] + t_final[2]) / 2

        # interpolate quaternion using spherical linear interpolation
        qs = np.stack((q_init,q_final))
        Rs = Rotation.from_quat(qs)
        times = np.linspace(0,1,M)
        slerper = Slerp([0,1],Rs)
        rotations = slerper(times)
        matrices = rotations.as_matrix()

        T = np.stack((np.linspace(t_init[0],t_final[0],M),
                np.linspace(t_init[1],t_final[1],M),
                np.linspace(t_init[2],t_final[2],M))).T

        x_cam = np.matmul(matrices,np.stack(M*[x_w.T])) + np.expand_dims(T,-1)
        proj = np.matmul(np.stack(M*[K]),x_cam)
        proj = proj / np.expand_dims(proj[:,2,:],1)
        proj = proj.transpose(0,2,1)
        x_img_true = proj[:,:,:2]

        #le = x_img_true[:,36,:]
        #re = x_img_true[:,45,:]
        #std = np.mean(np.linalg.norm(le - re,axis=1)*0.05)

        noise = np.random.randn(x_img_true.shape[0],x_img_true.shape[1],x_img_true.shape[2]) * self.std
        x_img = x_img_true + noise

        # create dictionary for results
        sample = {}
        sample['alpha_gt'] = torch.from_numpy(alpha).float()
        sample['x_w_gt'] = torch.from_numpy(x_w).float()
        sample['x_cam_gt'] = torch.from_numpy(x_cam).float()
        sample['x_img'] = x_img
        sample['x_img_gt'] = x_img_true
        sample['f_gt'] = torch.Tensor([f]).float()
        #sample['x_img_norm'] = torch.from_numpy(x_img_norm).float()
        #sample['K_gt'] = torch.from_numpy(K).float()
        #sample['R_gt'] = torch.from_numpy(R).float()
        #sample['Q_gt'] = torch.from_numpy(Q).float()
        #sample['T_gt'] = torch.from_numpy(T).float()
        return sample

    def generateRandomFace(self):
        N = self.s_mu.shape[0] // 3
        alphas = np.random.randn(199)[:,np.newaxis]
        d = np.matmul(self.s_eig,np.diag(self.sigma))
        d = np.matmul(d,alphas)
        s = self.s_mu + d
        s = s / 1000
        s = s.reshape((N,3))
        s[2,:] = s[2,:] * -1

        return s, alphas

    def generateRandomRotation(self):

        ax = self.max_rx;
        ay = self.max_ry;
        az = self.max_rz;
        rx = np.random.random()*2*ax - ax;
        ry = np.random.random()*2*ay - ay;
        rz = np.random.random()*2*az - az;

        r = Rotation.from_euler('zyx',[rz,ry,rx],degrees=True)
        q = r.as_quat()

        return r,q

    def generateRandomTranslation(self,K,minz,maxz,w=640,h=480):

        xvec = np.array([[w],[w/2],[1]])
        yvec = np.array([[h/2],[h],[1]])
        vz = np.array([[0],[0],[1]]);
        vx = np.matmul(np.linalg.inv(K),xvec)
        vy = np.matmul(np.linalg.inv(K),yvec)
        vx = np.squeeze(vx)
        vy = np.squeeze(vy)
        vz = np.array([0,0,1])
        thetax = np.arctan2(np.linalg.norm(np.cross(vz,vy)),np.dot(vz,vy));
        thetay = np.arctan2(np.linalg.norm(np.cross(vz,vx)),np.dot(vz,vx));

        tz = np.random.random()*(maxz-minz) + minz;
        maxx = tz * np.tan(thetax);
        maxy = tz * np.tan(thetay);
        tx = np.random.random()*maxx*2 - maxx;
        ty = np.random.random()*maxy*2 - maxy;
        t = [tx,ty,tz];

        return np.array(t)

    def project2d(self,r,t,K,pw):

        R = r.as_matrix()
        xc = np.matmul(R,pw.T) + np.expand_dims(t,1);

        proj = np.matmul(K,xc)
        proj = proj / proj[2,:]
        ximg = proj.T

        return ximg

class SyntheticLoader(Dataset):

    def __init__(self):

        root_dir = "../extras/shape_simple.mat"

        # load shape data
        shape_data = scipy.io.loadmat(root_dir)
        mu_lm = shape_data['mu_lm']
        mu_exp = shape_data['mu_exp']
        self.lm_eigenvec = shape_data['lm_eigenvec']
        exp_eigenvec = shape_data['exp_eigenvec']
        self.sigma = shape_data['sigma'] / 100

        self.mu_lm = mu_lm.T
        self.mu_lm = self.mu_lm - np.mean(self.mu_lm,0)

        self.mu_exp= mu_exp.T
        self.mu_exp= self.mu_exp - np.mean(self.mu_exp,0)

        self.exp_eigenvec = exp_eigenvec

        # video sequence length
        self.M = 100
        self.N = 68
        self.res = 640

        # extra boundaries on camera coordinates
        self.w = 640
        self.h = 480
        self.minf = 300; self.maxf = 1500
        self.minz = 800; self.maxz = 8000;
        self.max_rx = 20;
        self.max_ry = 20; self.max_rz = 20;
        self.xstd = 1; self.ystd = 1;

    def __len__(self):
        return 10000

    # betas (199,1)
    def get_3dmm(self,betas):
        b = betas.shape[0]
        mu_s  = torch.from_numpy(self.mu_lm).float().to(betas.device).clone()
        lm_eig = torch.from_numpy(self.lm_eigenvec).float().to(betas.device)
        sigma = torch.from_numpy(self.sigma).float().to(betas.device)

        mu_s[:,2] = mu_s[:,2] * -1
        sigma = torch.diag(sigma.squeeze())

        # adjust batch size
        lm_eig = torch.mm(lm_eig, sigma).unsqueeze(0).repeat(b,1,1)

        # predict batchwise 3d shapes
        shape = mu_s.unsqueeze(0).repeat(b,1,1) + torch.bmm(lm_eig,betas).view(b,-1,3)
        shape = shape - shape.mean(1).unsqueeze(1)
        return shape

    def __getitem__(self,idx):
        # data holders
        M = self.M
        N = self.N
        x_w = np.zeros((N,3));
        x_cam = np.zeros((M,N,3));
        x_img = np.zeros((M,N,2));
        x_img_true = np.zeros((M,N,2));

        # define intrinsics
        f = self.minf + random.random() * (self.maxf - self.minf);
        K = np.array([[f,0,self.w/2],[ 0,f,self.h/2], [0,0,1]])

        # create random 3dmm shape
        alpha = np.random.randn(199,1) * 5
        lm_eigenvec = np.matmul(self.lm_eigenvec,np.diag(self.sigma[:,0]))
        s = np.squeeze(np.matmul(lm_eigenvec,alpha))
        s = s.reshape(N,3)
        lm = self.mu_lm + s
        lm = lm - np.expand_dims(np.mean(lm,axis=0),axis=0)
        x_w = lm

        # create random 3dmm expression
        beta = np.random.randn(29) * 0.1
        e = np.sum(np.expand_dims(beta,0)*self.exp_eigenvec,1)
        exp = e.reshape(N,3)

        # define depth
        tz = np.random.random() * (self.maxz-self.minz) + self.minz
        minz = np.maximum(tz - 500,self.minz)
        maxz = np.minimum(tz + 500,self.maxz)

        # get initial and final rotation
        while True:
            r_init, q_init = self.generateRandomRotation()
            r_final, q_final = self.generateRandomRotation()
            t_init = self.generateRandomTranslation(K,minz,maxz)
            t_final = self.generateRandomTranslation(K,minz,maxz)

            ximg_init = self.project2d(r_init,t_init,K,x_w)
            ximg_final = self.project2d(r_final,t_final,K,x_w)
            if np.any(np.amin(ximg_init,axis=0) < 0): continue
            if np.any(np.amin(ximg_final,axis=0) < 0): continue
            if np.any(np.amin(ximg_init,axis=1) < 0): continue
            if np.any(np.amin(ximg_final,axis=1) < 0): continue
            init = np.amax(ximg_init,axis=0)
            final = np.amax(ximg_final,axis=0)
            if init[0] > self.w: continue
            if final[0] > self.w: continue
            if init[1] > self.h: continue
            if final[1] > self.h: continue
            break
        d = (t_init[2] + t_final[2]) / 2

        # interpolate quaternion using spherical linear interpolation
        qs = np.stack((q_init,q_final))
        Rs = Rotation.from_quat(qs)
        times = np.linspace(0,1,M)
        slerper = Slerp([0,1],Rs)
        rotations = slerper(times)
        matrices = rotations.as_matrix()

        T = np.stack((np.linspace(t_init[0],t_final[0],M),
                np.linspace(t_init[1],t_final[1],M),
                np.linspace(t_init[2],t_final[2],M))).T

        x_cam = np.matmul(matrices,np.stack(M*[x_w.T])) + np.expand_dims(T,-1)
        proj = np.matmul(np.stack(M*[K]),x_cam)
        proj = proj / np.expand_dims(proj[:,2,:],1)
        proj = proj.transpose(0,2,1)
        x_img_true = proj[:,:,:2]

        #le = x_img_true[:,36,:]
        #re = x_img_true[:,45,:]
        #std = np.mean(np.linalg.norm(le - re,axis=1)*0.05)

        noise = np.random.randn(M,N,2)

        x_img = x_img_true + noise

        # create dictionary for results
        sample = {}
        sample['alpha_gt'] = torch.from_numpy(alpha).float()
        sample['x_w_gt'] = torch.from_numpy(x_w).float()
        sample['x_cam_gt'] = torch.from_numpy(x_cam).float()
        sample['x_img'] = torch.from_numpy(x_img).float().permute(0,2,1)
        sample['x_img_gt'] = torch.from_numpy(x_img_true).float().permute(0,2,1)
        sample['f_gt'] = torch.Tensor([f]).float()
        #sample['x_img_norm'] = torch.from_numpy(x_img_norm).float()
        #sample['K_gt'] = torch.from_numpy(K).float()
        #sample['R_gt'] = torch.from_numpy(R).float()
        #sample['Q_gt'] = torch.from_numpy(Q).float()
        #sample['T_gt'] = torch.from_numpy(T).float()
        return sample

    def generateRandomRotation(self):

        ax = self.max_rx;
        ay = self.max_ry;
        az = self.max_rz;
        rx = np.random.random()*2*ax - ax;
        ry = np.random.random()*2*ay - ay;
        rz = np.random.random()*2*az - az;

        r = Rotation.from_euler('zyx',[rz,ry,rx],degrees=True)
        q = r.as_quat()

        return r,q

    def generateRandomTranslation(self,K,minz,maxz,w=640,h=480):

        xvec = np.array([[w],[w/2],[1]])
        yvec = np.array([[h/2],[h],[1]])
        vz = np.array([[0],[0],[1]]);
        vx = np.matmul(np.linalg.inv(K),xvec)
        vy = np.matmul(np.linalg.inv(K),yvec)
        vx = np.squeeze(vx)
        vy = np.squeeze(vy)
        vz = np.array([0,0,1])
        thetax = np.arctan2(np.linalg.norm(np.cross(vz,vy)),np.dot(vz,vy));
        thetay = np.arctan2(np.linalg.norm(np.cross(vz,vx)),np.dot(vz,vx));

        tz = np.random.random()*(maxz-minz) + minz;
        maxx = tz * np.tan(thetax);
        maxy = tz * np.tan(thetay);
        tx = np.random.random()*maxx*2 - maxx;
        ty = np.random.random()*maxy*2 - maxy;
        t = [tx,ty,tz];

        return np.array(t)

    def project2d(self,r,t,K,pw):

        R = r.as_matrix()
        xc = np.matmul(R,pw.T) + np.expand_dims(t,1);

        proj = np.matmul(K,xc)
        proj = proj / proj[2,:]
        ximg = proj.T

        return ximg

# LOADER FOR BIWI KINECT DATASET ONLY
class Face3DMM():
    def __init__(self,
            ):

        # load shape data
        shape_data = scipy.io.loadmat(shape_path)
        self.lm = torch.from_numpy(shape_data['keypoints'].astype(np.int32)).long().squeeze()
        self.mu_shape = torch.from_numpy(mu_shape.reshape(53215,3)).float()
        self.mu_shape = self.mu_shape - self.mu_shape.mean(0).unsqueeze(0)
        self.mu_shape = self.mu_shape / torch.max(torch.abs(self.mu_shape))
        self.shape_eigenvec = torch.from_numpy(shape_eigenvec).float()

        # load expression data
        exp_data = scipy.io.loadmat(exp_path)
        mu_exp = exp_data['mu_exp']
        exp_eigenvec = exp_data['w_exp']

        self.mu_exp = torch.from_numpy(mu_exp.reshape(53215,3)).float()
        self.exp_eigenvec = torch.from_numpy(exp_eigenvec).float()

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # torch image: C X H X W
        img = transform.resize(image, (new_h, new_w))
        img = img.transpose((2, 0, 1))

        return img

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # swap color axis because
        # numpy image: H x W x C
        return torch.from_numpy(data).float()

#Batch LOader
#BatchLoader = DataLoader(LFWLoader(),batch_size=8,shuffle=True,num_workers=4)

# UNIT TESTING
if __name__ == '__main__':

    # test dataloaders
    loader = MeritonLoader()
    print(loader.A.shape)
    print(loader.x_img.shape)
    print(loader.p_w.shape)
    quit()

    loader = CheckerboardLoader()
    data = loader[0]
    print(data['x_img'].shape)
    data['x_w_gt']
    data['K_gt']
    data['Fs']
    data['weights']
    quit()

    loader = AllLoader()
    data = loader[0]
    print(data.keys())
    quit()

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


    loader = MNLoader(400)
    quit()

    loader = AnalysisLoader()
    sample = loader[1]

    print(sample['x_w_gt'].shape)
    print(sample['x_cam_gt'].shape)
    print(sample['x_img'].shape)
    print(sample['x_img_gt'].shape)
    print(sample['f_gt'].shape)

    quit()

    loader = BIWILoader()
    sample = loader[1]
    print(sample['x_img'].shape)
    print(sample['x_img_gt'].shape)
    print(sample['f_gt'].shape)
    print(sample['x_cam_gt'].shape)

    import matplotlib.pyplot as plt
    for i in range(499):
        x = sample['x_img_gt'].numpy()[i].T
        print(x.shape)
        plt.scatter(x[:68,0],x[:68,1])
        plt.show()
    quit()

    loader = BIWIIDLoader()
    sample = loader[1]
    ximg = sample['x_img_gt']
    M = ximg.shape[0]
    for i in range(M):
        fig = plt.figure()
        ax = Axes3D(fig)
        x = sample['x_cam_gt'].numpy()[i].T
        ax.scatter(x[:,0],x[:,1],x[:,2])
        plt.show()
    print(ximg.shape)
    quit()

    loader = TestLoader(1000)
    sample = loader[1]

    for i in range(100):
        x = sample['x_img_gt'].numpy()[i].T
        plt.scatter(x[:,0],x[:,1])
        plt.show()
    quit()

    # test training loader
    loader = SyntheticLoader()
    sample = loader[1]
    print("sample made")
    print(sample.keys())

