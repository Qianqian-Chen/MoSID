import torch
import gc
import numpy as np
import SimpleITK as sitk
from options.Options import Options_x
from tqdm import tqdm
from Model.runet import RUnet
from torch.utils.data import DataLoader
from setting import logger, util
import os
from dataset.dataset_lits_test import Test_all_Datasets, Recompone_tool


def load(file):
    itkimage = sitk.ReadImage(file)
    image = sitk.GetArrayFromImage(itkimage)
    return image


def test_all(model_name='model_200.pth'):
    opt = Options_x().parse()
    device = torch.device('cuda:' + opt.gpu_ids if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model = RUnet(num_cls=1).to(device)
    ckpt = torch.load(opt.checkpoints_dir + '/' + opt.task_name + '/model/' + model_name, map_location=device)
    model.load_state_dict(ckpt['model'])

    save_result_path = os.path.join(opt.checkpoints_dir, opt.task_name, 'test_all_result')
    util.mkdir(save_result_path)
    model.eval()
    log_test = logger.Test_Logger(save_result_path, "results")
    cut_param = {'patch_s': 192, 'patch_h': 192, 'patch_w': 320,
                 'stride_s': 192, 'stride_h': 192, 'stride_w': 192}
    datasets = Test_all_Datasets(opt.datapath, cut_param)

    for img_dataset, original_shape, new_shape, file_idx in datasets:
        save_tool = Recompone_tool(original_shape, new_shape, cut_param)
        dataloader = DataLoader(img_dataset, batch_size=opt.test_batch, num_workers=opt.num_threads, shuffle=False)
        with torch.no_grad():
            for DCE in tqdm(dataloader):
                DCE = DCE.to(device)
                DCE = DCE.unsqueeze(1).type(torch.float32)
                pred = model(DCE)

                output = pred.type(torch.float32)
                save_tool.add_result(output.detach().cpu())

        recon = save_tool.recompone_overlap()
        Pred = sitk.GetImageFromArray(np.array(recon))
        aaaaaa = sitk.ReadImage(os.path.join(opt.datapath, file_idx, 'ADC.nii.gz'))
        Pred.SetSpacing(aaaaaa.GetSpacing())
        Pred.SetDirection(aaaaaa.GetDirection())
        Pred.SetOrigin(aaaaaa.GetOrigin())
        sitk.WriteImage(Pred, os.path.join(save_result_path, file_idx + '.nii.gz'))
        del pred, recon, Pred, save_tool
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    test_all('best_model.pth')
