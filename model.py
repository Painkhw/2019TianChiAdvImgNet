#数据点附近的10个位置分别求更新梯度方向，然后取平均获得更稳定的评估。
#clip到32像素内，得分为0.38分
from function import *

manifest="data/dev.csv"
#df=pd.read_csv(manifest)

path="data/images_raw/"
data=Data(path=path,manifest=manifest,device="gpu")
dataloader=DataLoader(data,batch_size=40)

#res18
model1=torchvision.models.ResNet( torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
state=torch.load(glob("resnet18*")[0])
model1.load_state_dict(state)

# ##res34
# model2=torchvision.models.ResNet( torchvision.models.resnet.Bottleneck,[3, 4, 6, 3])
# state=torch.load(glob("resnet34*")[0])
# model2.load_state_dict(state)

##res50
model3= torchvision.models.ResNet( torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])
state=torch.load(glob("resnet50*")[0])
model3.load_state_dict(state)

# ##vgg16
# model= torchvision.models.VGG(make_layers(cfgs["D"], batch_norm="False"))
# state=torch.load(glob("resnet50*")[0])
# model.load_state_dict(state)

##res101
model4=torchvision.models.ResNet( torchvision.models.resnet.Bottleneck, [3, 4, 23, 3])
state=torch.load(glob("resnet101*")[0])
model4.load_state_dict(state)

#inceptionv3
# model4 = torchvision.models.Inception3()
# state_dict = torch.load(glob("inception*")[0])
# model4.load_state_dict(state_dict)


models={
   "resnet18":nn.DataParallel(model1.cuda()).train(),
    #"resnet34":nn.DataParallel(model2).cuda().train(),
   "resnet50":nn.DataParallel(model3.cuda()).train(),
   "resnet101":nn.DataParallel(model4.cuda()).train(),
   #"inceptionv3":nn.DataParallel(model4).cuda().train()
}

# dd=next(iter(dataloader))
# try:
#     dd[0]=dd[0].cuda()
#     out=model(dd[0])
# except RuntimeError:
#     print("eee")
#     pass

for pic,pic_id,label,target in dataloader:
    for name, model in models.items():
        out = model(pic)
        pre=out.clone().detach().max(dim=1).indices
        a=(pre.cpu()+1 == label).clone().detach().sum().numpy()
        b=pre.shape[0]
        print(name,":",a,b)


def LossFunc(input,label,n_top=50):
    input=torch.log_softmax(input, 1)
    x=input.sort(dim=1,descending=True)
    out=-x.values[:,:n_top].mean()-2*F.nll_loss(input,label)/n_top
    return out

def adv(models,optimizer,pic,pic1,label,n_it_ad,i):
    _=[model.eval() for model in models.values()]
    for n_it in range(n_it_ad):
        loss=[]
        for model in models.values():
            out = model(pic)
            loss.append(LossFunc(out, label.cuda(), n_top=n_top))  # +torch.abs(pic-pic1).max()*255/eposile
        loss=sum(loss)/len(models)#平均损失
        print(f"batch {i},iter {n_it},loss {loss.cpu().item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        diff = pic.detach() - pic1.detach()  # 新图片和原始图片的像素差
        print("diff: ", diff.abs().max().item() * 255)
        diff = np.clip(diff.cpu().numpy(), -eposile / 255, eposile / 255)  # 限制在32像素以内
        diff = Tensor(diff)
        pic.data = diff.cuda() + pic1.data
        pic.data[pic.data < 0] = 0
        pic.data[pic.data > 1] = 1

    return pic

def train(models,pic,label,epoch):
    # 更新被攻击的模型
    _=[model.train() for model in models.values()]
    for ii in range(3):
        loss=[]
        for model in models.values():
            out = model(pic)
            loss.append(nn.CrossEntropyLoss()(out, label.cuda()))
        loss=sum(loss)
        print("train model, loss", loss.cpu().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for name,model in models.items():
        torch.save(model.state_dict(), f"{name}_params_{epoch}.pt")


max_epoch=50#总最大迭代次数
eposile=32#像素差阈值
n_top=50#在计算对抗样本时，topn的错误标签的概率被增大
n_it_ad=50#算对抗样本时的迭代次数
n_it_tr=3#更新模型时的迭代次数
lr_ad=2/255#对抗攻击时，步长为2个像素
lr_tr=0.0001#更新模型时的步长

for epoch in range(max_epoch):
    for i,(pic,pic_id,label,target) in enumerate(dataloader):# one epoch
        label=label-1
        target=target-1#对齐pytorch的标签和比赛中的标签
        print(i)
        pic1 = pic.clone().detach()  # 原始图片

        # 攻击
        pic=Variable(pic+(torch.rand(pic.size())/255*eposile).cuda(),requires_grad=True)#adv初始位置
        optimizer = FGM([pic], lr=lr_ad)#求解对抗样本
        pic = adv(models,optimizer,pic,pic1,label,n_it_ad,i)

        #保存这一轮的结果
        pic = pic.cpu().detach().clone()
        pic[pic < 0] = 0
        pic[pic > 1] = 1  # pic为这一代的对抗样本
        if ("pre_" + str(epoch) not in os.listdir()):
            os.mkdir("pre_" + str(epoch))
        save(pic, ["pre_" + str(epoch) + "/" + p_id for p_id in pic_id])  # 保存

        #更新模型
        pic = Variable(pic.cuda(), requires_grad=True)
        params=[]
        for model in models.values():
            params+=list(model.parameters())
        optimizer = Adam(params, lr=lr_tr)  # 更新模型
        train(models, pic, label, epoch)





