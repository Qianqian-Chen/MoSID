import torch
import gc
import numpy as np
import SimpleITK as sitk
from options.Options import Options_x
from tqdm import tqdm
from Model.networks import RUnet
from torch.utils.data import DataLoader
from setting import logger, util
from setting.metrics import seg_metric
import os
from dataset.dataset_lits_test import Test_all_Datasets, Recompone_tool
from collections import OrderedDict


def load(file):
    itkimage = sitk.ReadImage(file)
    image = sitk.GetArrayFromImage(itkimage)
    return image


def test_all(model_name, flag):
    opt = Options_x().parse()  # get training options
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model = RUnet().to(device)

    ckpt = torch.load(opt.checkpoints_dir + '/' + opt.task_name + '/model/' + model_name, map_location=device)
    model.load_state_dict(ckpt['model'])

    save_result_path = os.path.join(opt.checkpoints_dir, opt.task_name, 'test_all_result')
    util.mkdir(save_result_path)
    model.eval()

    save_excel_path = os.path.join(save_result_path, flag)
    util.mkdir(save_excel_path)
    log_test = logger.Test_Logger(save_excel_path, "results_train")

    cut_param = {'patch_s': opt.patch_size[0], 'patch_h': opt.patch_size[1], 'patch_w': opt.patch_size[2],
                 'stride_s': opt.patch_stride[0], 'stride_h': opt.patch_stride[1], 'stride_w': opt.patch_stride[2]}
    datasets = Test_all_Datasets(opt.datapath, cut_param, flag)

    for img_dataset, original_shape, new_shape, mask, file_idx in datasets:
        save_tool = Recompone_tool(original_shape, new_shape, cut_param)
        save_prob_tool = Recompone_tool(original_shape, new_shape, cut_param)
        dataloader = DataLoader(img_dataset, batch_size=opt.test_batch, num_workers=opt.num_threads, shuffle=False)
        with torch.no_grad():
            for pre, pos, sub, adc, t2w, gt in tqdm(dataloader):
                pos, sub, adc, t2w, gt = pos.to(device), sub.to(device), adc.to(device), t2w.to(device), gt.to(device)

                pos = pos.unsqueeze(1).type(torch.float32)
                sub = sub.unsqueeze(1).type(torch.float32)
                adc = adc.unsqueeze(1).type(torch.float32)
                t2w = t2w.unsqueeze(1).type(torch.float32)

                output = model(pos, sub, adc, t2w)

                probility = output.type(torch.float32)
                seg = (output >= 0.5).type(torch.float32)

                save_prob_tool.add_result(probility.detach().cpu())
                save_tool.add_result(seg.detach().cpu())

        probility = save_prob_tool.recompone_overlap()
        pred = save_tool.recompone_overlap()

        recon = (pred.numpy() > 0.5).astype(np.uint16) * mask.astype(np.uint16)
        gt = load(os.path.join(opt.datapath, file_idx, 'GT.nii.gz'))
        DCE = sitk.ReadImage(os.path.join(opt.datapath, file_idx, 'DCE.nii.gz'))
        DSC, IoU, SEN, ASD, HD = seg_metric(recon, gt, DCE)
        index_results = OrderedDict({'DSC': DSC, 'SEN': SEN, 'ASD': ASD, 'IoU': IoU, 'HD': HD})
        log_test.update(file_idx, index_results)

        Pred = sitk.GetImageFromArray(np.array(recon))
        prob_img = sitk.GetImageFromArray(np.array(probility))

        result_save_path = os.path.join(save_result_path, flag, file_idx)
        util.mkdir(result_save_path)

        sitk.WriteImage(Pred, os.path.join(result_save_path, 'pred.nii.gz'))
        sitk.WriteImage(prob_img, os.path.join(result_save_path, 'probility.nii.gz'))

        del pred, recon, Pred, save_tool, save_prob_tool, prob_img, probility, gt
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    for flag_in in ['all_true', 'fake_adc', 'fake_t2', 'all_fake']:
        test_all('best_model.pth', flag_in)

