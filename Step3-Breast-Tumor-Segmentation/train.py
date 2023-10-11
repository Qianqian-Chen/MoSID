import torch
import numpy as np
import torch.optim as optim
from options.Options import Options_x
from dataset.dataset_lits_train import Lits_DataSet
from dataset.dataset_lits_val import Val_DataSet
from Model.networks import MoSID
from torch.utils.data import DataLoader
from utils.common import adjust_learning_rate
from utils import logger, util
import torch.nn as nn
from utils.metrics import LossAverage, DiceLoss
import os
import time
from test import test_all
from collections import OrderedDict

def val(val_dataloader, epoch):
    since = time.time()

    Loss = LossAverage()
    DICE_Loss = LossAverage()
    BCE_Loss = LossAverage()
    Supervised_Loss = LossAverage()
    max_pooling_3d = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

    for i, (DCE, sub, ADC, T2W, Infor_DCE, Infor_ADC, Infor_T2, gt) in enumerate(val_dataloader):
        b, c, l, w, e = DCE.shape[0], DCE.shape[1], DCE.shape[2], DCE.shape[3], DCE.shape[4]

        DCE = DCE.view(-1, 1, l, w, e).to(device)
        sub = sub.view(-1, 1, l, w, e).to(device)
        ADC = ADC.view(-1, 1, l, w, e).to(device)
        T2W = T2W.view(-1, 1, l, w, e).to(device)
        Infor_DCE = Infor_DCE.view(-1, 1, l, w, e).to(device)
        Infor_ADC = Infor_ADC.view(-1, 1, l, w, e).to(device)
        Infor_T2 = Infor_T2.view(-1, 1, l, w, e).to(device)
        gt = gt.view(-1, 1, l, w, e).to(device)

        gt2 = max_pooling_3d(gt).detach()
        gt3 = max_pooling_3d(gt2).detach()
        gt4 = max_pooling_3d(gt3).detach()

        out4, out3, out2, out, pred = model(DCE, sub, ADC, T2W, Infor_DCE, Infor_ADC, Infor_T2)

        Dice_loss = dice_loss(pred, gt)
        Bce_loss = bce_loss(pred, gt)
        Supervised_loss = dice_loss(pred, gt) + dice_loss(out4, gt4) + dice_loss(out3, gt3) + dice_loss(out2, gt2) + dice_loss(out, gt)

        loss = Bce_loss + 5 * Dice_loss

        Loss.update(loss.item(), DCE.size(0))
        DICE_Loss.update(Dice_loss.item(), DCE.size(0))
        BCE_Loss.update(Bce_loss.item(), DCE.size(0))
        Supervised_Loss.update(Supervised_loss.item(), DCE.size(0))
    time_elapsed = time.time() - since
    print("=======Val Epoch:{}======Learning_rate:{}======Validate complete in {:.0f}m {:.0f}s=======".format(epoch, optimizer.param_groups[0]['lr'], time_elapsed // 60, time_elapsed % 60))
    return OrderedDict({'Loss': Loss.avg, 'DICE_Loss': DICE_Loss.avg, 'BCE_Loss': BCE_Loss.avg, 'Supervised_Loss': Supervised_Loss.avg})


def train(train_dataloader, epoch):
    since = time.time()

    Loss = LossAverage()
    DICE_Loss = LossAverage()
    BCE_Loss = LossAverage()
    Supervised_Loss = LossAverage()
    max_pooling_3d = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

    model.train()

    for i, (DCE, sub, ADC, T2W, Infor_DCE, Infor_ADC, Infor_T2, gt) in enumerate(train_dataloader):  # inner loop within one epoch
        b, c, l, w, e = DCE.shape[0], DCE.shape[1], DCE.shape[2], DCE.shape[3], DCE.shape[4]

        DCE = DCE.view(-1, 1, l, w, e).to(device)
        sub = sub.view(-1, 1, l, w, e).to(device)
        ADC = ADC.view(-1, 1, l, w, e).to(device)
        T2W = T2W.view(-1, 1, l, w, e).to(device)
        Infor_DCE = Infor_DCE.view(-1, 1, l, w, e).to(device)
        Infor_ADC = Infor_ADC.view(-1, 1, l, w, e).to(device)
        Infor_T2 = Infor_T2.view(-1, 1, l, w, e).to(device)
        gt = gt.view(-1, 1, l, w, e).to(device)

        gt2 = max_pooling_3d(gt).detach()
        gt3 = max_pooling_3d(gt2).detach()
        gt4 = max_pooling_3d(gt3).detach()

        out4, out3, out2, out, pred = model(DCE, sub, ADC, T2W, Infor_DCE, Infor_ADC, Infor_T2)

        Dice_loss = dice_loss(pred, gt)
        Bce_loss = bce_loss(pred, gt)
        Supervised_loss = dice_loss(pred, gt) + dice_loss(out4, gt4) + dice_loss(out3, gt3) + dice_loss(out2, gt2) + dice_loss(out, gt)

        loss = Bce_loss + 5 * Dice_loss + Supervised_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer, epoch, opt)

        Loss.update(loss.item(), DCE.size(0))
        DICE_Loss.update(Dice_loss.item(), DCE.size(0))
        BCE_Loss.update(Bce_loss.item(), DCE.size(0))
        Supervised_Loss.update(Supervised_loss.item(), DCE.size(0))
    time_elapsed = time.time() - since
    print("=======Train Epoch:{}======Learning_rate:{}======Train complete in {:.0f}m {:.0f}s=======".format(epoch, optimizer.param_groups[0]['lr'], time_elapsed // 60, time_elapsed % 60))
    return OrderedDict({'Loss': Loss.avg, 'DICE_Loss': DICE_Loss.avg, 'BCE_Loss': BCE_Loss.avg, 'Supervised_Loss': Supervised_Loss.avg})


if __name__ == '__main__':
    opt = Options_x().parse()  # get training options
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model = MoSID().to(device)

    save_path = opt.checkpoints_dir
    dice_loss = DiceLoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    save_result_path = os.path.join(save_path, opt.task_name)
    util.mkdir(save_result_path)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)
    model_save_path = os.path.join(save_result_path, 'model')
    util.mkdir(model_save_path)
    logger_save_path = os.path.join(save_result_path, 'logger')
    util.mkdir(logger_save_path)

    log_train = logger.Train_Logger(logger_save_path, "train_log")
    log_val = logger.Val_Logger(logger_save_path, "val_log")

    train_dataset = Lits_DataSet(opt.datapath, opt.patch_size)
    val_dataset = Val_DataSet(opt.datapath, opt.patch_size)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, num_workers=opt.num_threads, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, num_workers=opt.num_threads, shuffle=True)

    types = ['train', 'val']
    val_dice_loss = 99
    best_epoch = 0

    for epoch in range(opt.epoch):
        epoch = epoch + 1
        for type in types:
            if type == 'train':
                train_log = train(train_dataloader, epoch)
                log_train.update(epoch, train_log)
            elif type == 'val':
                val_log = val(val_dataloader, epoch)
                log_val.update(epoch, val_log)
                if val_log['DICE_Loss'] < val_dice_loss:
                    best_epoch = epoch
                    val_dice_loss = val_log['DICE_Loss']
                    state = {'model': model.state_dict(), 'epoch': best_epoch}
                    torch.save(state, os.path.join(model_save_path, 'best_model.pth'))

        state = {'model': model.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(model_save_path, 'latest_model.pth'))

        if epoch % opt.model_save_fre == 0:
            torch.save(state, os.path.join(model_save_path, 'model_' + np.str(epoch) + '.pth'))

        torch.cuda.empty_cache()

    test_all('best_model.pth')
