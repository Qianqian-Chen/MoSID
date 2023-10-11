import torch
import numpy as np
import torch.optim as optim
from options.Options import Options_x
from dataset.dataset_lits_train import Lits_DataSet
from dataset.dataset_lits_val import Val_DataSet

from Model.runet import RUnet
from torch.utils.data import DataLoader
from setting.common import adjust_learning_rate
from setting import logger,util
import torch.nn as nn
from setting.metrics import LossAverage
import os
import time
from test import test_all
from collections import OrderedDict


def train(train_dataloader, epoch):
    since = time.time()
    Loss = LossAverage()
    model.train()

    for i, (DCE,ADC, gt) in enumerate(train_dataloader):  # inner loop within one epoch

        b, c, l, w, e = ADC.shape[0], ADC.shape[1], ADC.shape[2], ADC.shape[3], ADC.shape[4]

        ADC = ADC.view(-1, 1, l, w, e).to(device)
        DCE = DCE.view(-1, 1, l, w, e).to(device)

        pred = model(DCE)
        loss = nn.L1Loss()(pred, ADC)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer, epoch, opt)

        Loss.update(loss.item(), ADC.size(0))

    time_elapsed = time.time() - since
    print("=======Train Epoch:{}======Learning_rate:{}======Train complete in {:.0f}m {:.0f}s=======".format(epoch, optimizer.param_groups[0]['lr'], time_elapsed // 60, time_elapsed % 60))

    return OrderedDict({'Loss': Loss.avg})


def val(val_dataloader, epoch):
    since = time.time()

    Loss = LossAverage()
    model.eval()

    for i, (DCE,ADC, gt) in enumerate(val_dataloader):  # inner loop within one epoch

        b, c, l, w, e = ADC.shape[0], ADC.shape[1], ADC.shape[2], ADC.shape[3], ADC.shape[4]

        ADC = ADC.view(-1, 1, l, w, e).to(device)
        DCE = DCE.view(-1, 1, l, w, e).to(device)

        pred = model(DCE)
        loss = nn.L1Loss()(pred, ADC)

        Loss.update(loss.item(), ADC.size(0))

    time_elapsed = time.time() - since
    print("=======Val Epoch:{}======Learning_rate:{}======Val complete in {:.0f}m {:.0f}s=======".format(epoch, optimizer.param_groups[0]['lr'], time_elapsed // 60, time_elapsed % 60))

    return OrderedDict({'Loss': Loss.avg})


if __name__ == '__main__':
    opt = Options_x().parse()   # get training options
    device = torch.device('cuda:'+opt.gpu_ids if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model = RUnet(num_cls=1).to(device)

    save_path = opt.checkpoints_dir

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

    types = ['train','val']
    val_loss = 99
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
                if val_log['DICE_Loss'] < val_loss:
                    best_epoch = epoch
                    val_loss = val_log['DICE_Loss']
                    state = {'model': model.state_dict(), 'epoch': best_epoch}
                    torch.save(state, os.path.join(model_save_path, 'best_model.pth'))

        state = {'model': model.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(model_save_path, 'latest_model.pth'))

        if epoch % opt.model_save_fre == 0:
            torch.save(state, os.path.join(model_save_path, 'model_'+np.str(epoch)+'.pth'))

        torch.cuda.empty_cache()

    test_all('best_model.pth')













