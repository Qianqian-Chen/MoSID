import torch
import gc
import numpy as np
import SimpleITK as sitk
from options.Options import Options_x
from tqdm import tqdm
from Model.networks import MoSID
from torch.utils.data import DataLoader
from utils import logger, util
from utils.metrics import seg_metric
import os
from dataset.dataset_lits_test import Test_all_Datasets, Recompone_tool
from collections import OrderedDict


def load(file):
    itkimage = sitk.ReadImage(file)
    image = sitk.GetArrayFromImage(itkimage)
    return image


def test_all(model_name='model_200.pth'):
    opt = Options_x().parse()  # get training options
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model = MoSID().to(device)
    ckpt = torch.load(opt.checkpoints_dir + '/' + opt.task_name + '/model/' + model_name, map_location=device)
    model.load_state_dict(ckpt['model'])

    save_result_path = os.path.join(opt.checkpoints_dir, opt.task_name, 'test_all_result')
    util.mkdir(save_result_path)
    model.eval()
    log_test = logger.Test_Logger(save_result_path, "results")
    cut_param = {'patch_s': opt.patch_size[0], 'patch_h': opt.patch_size[1], 'patch_w': opt.patch_size[2],
                 'stride_s': opt.patch_stride[0], 'stride_h': opt.patch_stride[1], 'stride_w': opt.patch_stride[2]}
    datasets = Test_all_Datasets(opt.datapath, cut_param)

    for img_dataset, original_shape, new_shape, mask, file_idx, crop_box in datasets:
        save_tool = Recompone_tool(original_shape, new_shape, cut_param)
        dataloader = DataLoader(img_dataset, batch_size=opt.test_batch, num_workers=opt.num_threads, shuffle=False)
        with torch.no_grad():
            for pos, sub, adc, t2w, p_all_fake, p_fake_adc, p_fake_t2, gt in tqdm(dataloader):
                pos, sub, adc, t2w, p_all_fake, p_fake_adc, p_fake_t2, gt = pos.to(device), sub.to(device), adc.to(
                    device), t2w.to(device), p_all_fake.to(device), p_fake_adc.to(device), p_fake_t2.to(device), gt.to(device)

                pos = pos.unsqueeze(1).type(torch.float32)
                sub = sub.unsqueeze(1).type(torch.float32)
                adc = adc.unsqueeze(1).type(torch.float32)
                t2w = t2w.unsqueeze(1).type(torch.float32)
                p_all_fake = p_all_fake.unsqueeze(1).type(torch.float32)
                p_fake_adc = p_fake_adc.unsqueeze(1).type(torch.float32)
                p_fake_t2 = p_fake_t2.unsqueeze(1).type(torch.float32)

                _, _, _, _, output = model(pos, sub, adc, t2w, p_all_fake, p_fake_adc, p_fake_t2)

                output = (output >= 0.5).type(torch.float32)
                save_tool.add_result(output.detach().cpu())

        pred = save_tool.recompone_overlap()
        recon = (pred.numpy() > 0.5).astype(np.uint16)

        pred_coarse = load(os.path.join(opt.datapath, file_idx, 'pred_coarse.nii.gz'))
        pred_coarse[crop_box[0, 0]:crop_box[0, 1], crop_box[1, 0]:crop_box[1, 1], crop_box[2, 0]:crop_box[2, 1]] = recon

        if np.sum(pred_coarse) < 15:
            pred_coarse = load(os.path.join(opt.datapath, file_idx, 'pred_coarse.nii.gz'))

        pred_coarse = pred_coarse * mask

        gt = load(os.path.join(opt.datapath, file_idx, 'GT.nii.gz'))
        DCE = sitk.ReadImage(os.path.join(opt.datapath, file_idx, 'DCE.nii.gz'))
        DSC, IoU, SEN, ASD, HD = seg_metric(pred_coarse, gt, DCE)
        index_results = OrderedDict({'DSC': DSC, 'SEN': SEN, 'ASD': ASD, 'IoU': IoU, 'HD': HD})
        log_test.update(file_idx, index_results)
        Pred = sitk.GetImageFromArray(np.array(pred_coarse))
        result_save_path = os.path.join(save_result_path, file_idx)
        util.mkdir(result_save_path)
        sitk.WriteImage(Pred, os.path.join(result_save_path, 'pred.nii.gz'))
        del pred, recon, Pred, save_tool, pred_coarse, gt
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    test_all('best_model.pth')

