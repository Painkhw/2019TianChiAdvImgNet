import torch
import torchvision
import os
import gc
from torch import nn,Tensor
from glob import glob
from torch.optim import SGD,Adam
from torch import optim
from torchvision.models.vgg import make_layers,cfgs
from torch.nn import functional as F
from torchvision.transforms.functional import resize
from torchvision.transforms.functional import to_tensor,to_pil_image
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from torchvision import transforms
from torchvision.transforms import ToTensor,ToPILImage
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim.optimizer import Optimizer
from PIL import Image as im


def save(pic,fp):
    for i,f in enumerate(fp):
        p=to_pil_image(pic[i])
        p.save(f)

def data_clip(data1,data2,clip=32,res_dir="pre1/"):
    #data1=Data(manifest,path)
    #data2=Data(manifest,"pre/")

    dataloader1=DataLoader(data1,batch_size=10)
    dataloader2=DataLoader(data2,batch_size=10)
    for (pic1, pic_id1, label1, target1),(pic2, pic_id2, label2, target2) in zip(dataloader1,dataloader2):
        diff=torch.sign(pic2-pic1)
        pic2=pic1+diff*clip/255
        pic2=Tensor(np.clip(pic2.numpy(),0,1))
        save(pic2,[res_dir+e for e in pic_id1])

def show(pic,tool="plt"):
    if(tool=="plt"):
        pic=list(pic.numpy())
        pic=np.stack(pic,axis=-1)
        plt.imshow(pic)
    if(tool=="PIL"):
        pic=to_pil_image(pic)
        pic.show()


class Data(Dataset):
    def __init__(self,manifest,path,device="cpu"):
        self.manifest=pd.read_csv(manifest)
        self.path=path
        self.device=device

    def __getitem__(self, item):
        imageid,label,target=self.manifest.iloc[item]
        image=im.open(self.path+imageid,"r")
        image=to_tensor(image)
        if(self.device=="cpu"):
            return image,imageid,label,target
        else:
            return image.cuda(), imageid, label, target

    def __len__(self):
        return self.manifest.shape[0]

class FGSM(Optimizer):
    #默认梯度上升
    def __init__(self,params,lr=1):
        defaults=dict(lr=lr)
        super(FGSM, self).__init__(params, defaults)
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = torch.sign(p.grad.data)
                p.data.add_(group['lr'], d_p)

class MIFGSM(Optimizer)
    def __init__(self,params, lr=1):
        defaults=dict(lr=lr)
        super(MIFGSM, self).__init__(params,defaults)
    def setp(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data.view(p.data.size(0),-1)
                d_p = torch.sign(d_p/torch.norm(d_p,p=1,dim=(1),keepdim=True))


class FGM(optim.Adam):
    #默认梯度下降
    def __init__(self,params,lr=1):
        super(FGM, self).__init__(params, lr=lr)
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p=p.grad.data.view(p.data.size(0),-1)
                d_p = d_p/torch.norm(d_p,dim=(1),keepdim=True)
                p.grad.data=d_p.view(-1,3,299,299)

        super(FGM,self).step()

def gs_filter(n=5,delta=1):
    c_i=(n-1)/2
    c_j=(n-1)/2
    weight=torch.zeros((n,n))
    for i in range(n):
        for j in range(n):
            weight[i,j]=np.exp((-(i-c_i)**2-(j-c_j)**2 )/2*delta)
    weight/=weight.sum()
    return weight

def resize_ad(path="data/images_raw/",res_path="pre/",origin=299,resize_list=[10]):
    manifest = "data/dev.csv"
    resize_list.append(origin)
    data = Data_resize(path=path, manifest=manifest,resize_list=resize_list)
    for i in range(len(data)):
        print(i)
        pic, pic_id, label, target=data[i]
        pic.save(res_path+pic_id)

class Data_resize:
    def __init__(self,manifest,path,resize_list=[100]):
        self.manifest=pd.read_csv(manifest)
        self.path=path
        self.resize_list=resize_list

    def __getitem__(self, item):
        imageid,label,target=self.manifest.iloc[item]
        image=im.open(self.path+imageid,"r")
        for e in self.resize_list:
            image = resize(image, (e, e), im.BILINEAR)

        return image,imageid,label,target

    def __len__(self):
        return self.manifest.shape[0]
