'''
Author:Xuelei Chen(chenxuelei@hotmail.com)
Usgae:
python train.py TRAIN_RAW_IMAGE_FOLDER TRAIN_REFERENCE_IMAGE_FOLDER
'''
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms

from model import PhysicalNN
from uwcc import uwcc
import shutil
import os
from torch.utils.data import DataLoader
import sys


def main():

    best_loss = 9999.0


    lr = 0.001
    batchsize = 1
    n_workers = 2
    epochs = 3000
    ori_fd = "/home/data/hope/night_hope/6/low"
    ucc_fd = "/home/data/hope/night_hope/6/high"
    ori_dirs = [os.path.join(ori_fd, f) for f in os.listdir(ori_fd)]
    ucc_dirs = [os.path.join(ucc_fd, f) for f in os.listdir(ucc_fd)]

    #create model
    model = PhysicalNN()
    model = nn.DataParallel(model)   ##并行运算
    model = model.cuda()

    #define optimizer  优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #define criterion  损失函数
    criterion = nn.MSELoss()

    #load data
    trainset = uwcc(ori_dirs, ucc_dirs, train=True)
    #迭代器
    trainloader = DataLoader(trainset, batchsize, shuffle=True, num_workers=n_workers)


    #train
    for epoch in range(epochs):

        tloss = train(trainloader, model, optimizer, criterion)

        print('Epoch:[{}/{}] Loss{}'.format(epoch,epochs, tloss))
        is_best = tloss < best_loss
        best_loss = min(tloss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best)
    print('Best Loss: ', best_loss)

def train(trainloader, model, optimizer, criterion):
    losses = AverageMeter()
    model.train()

    for i, sample in enumerate(trainloader):
        ori, ucc = sample
        ori = ori.cuda()
        ucc = ucc.cuda()

        corrected = model(ori)  #调用模型
        loss = criterion(corrected,ucc)    #损失
        losses.update(loss)    #记录

        optimizer.zero_grad()   #清空过往梯度
        loss.backward()     #反向传播，计算当前梯度
        optimizer.step()     #根据梯度更新网络参数
    return losses.avg


def save_checkpoint(state, is_best):
    """Saves checkpoint to disk"""
    freq = 500
    epoch = state['epoch'] 

    filename = './checkpoints/model_tmp.pth'
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    torch.save(state, filename)

    if epoch%freq==0:
        shutil.copyfile(filename, './checkpoints/model_{}.pth'.format(epoch))
    if is_best:
        shutil.copyfile(filename, './checkpoints/model_best_{}.pth'.format(epoch))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
