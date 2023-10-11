import numpy as np
import torch, os
from torch.utils.data import Dataset, DataLoader
from glob import glob
import random
import SimpleITK as sitk


def normalization(img, lmin=1, rmax=None, dividend=None, quantile=1):
    newimg = img.copy()
    newimg = newimg.astype(np.float32)
    if quantile is not None:
        maxval = round(np.percentile(newimg, 100 - quantile))
        minval = round(np.percentile(newimg, quantile))
        newimg[newimg >= maxval] = maxval
        newimg[newimg <= minval] = minval

    if lmin is not None:
        newimg[newimg < lmin] = lmin
    if rmax is not None:
        newimg[newimg > rmax] = rmax

    minval = np.min(newimg)
    if dividend is None:
        maxval = np.max(newimg)
        newimg = (np.asarray(newimg).astype(np.float32) - minval) / (maxval - minval)
    else:
        newimg = (np.asarray(newimg).astype(np.float32) - minval) / dividend
    return newimg, minval, maxval


def normalization_fix(img, minval, maxval, lmin=1):
    newimg = img.copy()
    newimg = newimg.astype(np.float32)
    if lmin is not None:
        newimg[newimg < lmin] = lmin

    newimg = (np.asarray(newimg).astype(np.float32) - minval) / (maxval - minval)
    return newimg


def normalization_org(img, lmin=1, rmax=None, dividend=None, quantile=None):
    newimg = img.copy()
    newimg = newimg.astype(np.float32)
    if quantile is not None:
        maxval = round(np.percentile(newimg, 100 - quantile))
        minval = round(np.percentile(newimg, quantile))
        newimg[newimg >= maxval] = maxval
        newimg[newimg <= minval] = minval

    if lmin is not None:
        newimg[newimg < lmin] = lmin
    if rmax is not None:
        newimg[newimg > rmax] = rmax

    minval = np.min(newimg)
    if dividend is None:
        maxval = np.max(newimg)
        newimg = (np.asarray(newimg).astype(np.float32) - minval) / (maxval - minval)
    else:
        newimg = (np.asarray(newimg).astype(np.float32) - minval) / dividend
    return newimg


def load(file):
    itkimage = sitk.ReadImage(file)
    image = sitk.GetArrayFromImage(itkimage)
    return image


def maskcor_extract_3d(mask, padding=(5, 5, 5)):
    # mask_s = mask.shape
    p = np.where(mask > 0)
    a = np.zeros([3, 2], dtype=np.int)
    if len(p[0]) == 0:
        a[0, 0] = 0
        a[0, 1] = 32
        a[1, 0] = 0
        a[1, 1] = 128
        a[2, 0] = 0
        a[2, 1] = 128
        return a

    for i in range(3):
        s = p[i].min()
        e = p[i].max() + 1

        ss = s - padding[i]
        ee = e + padding[i]
        if ss < 0:
            ss = 0
        if ee > mask.shape[i]:
            ee = mask.shape[i]

        a[i, 0] = ss
        a[i, 1] = ee
    return a


class Img_DataSet(Dataset):
    def __init__(self, pos, sub, adc, t2w, Infor_DCE, Infor_ADC, Infor_T2, gt, cut_param):
        self.pos = pos
        self.sub = sub
        self.adc = adc
        self.t2w = t2w
        self.Infor_DCE = Infor_DCE
        self.Infor_ADC = Infor_ADC
        self.Infor_T2 = Infor_T2
        self.gt = gt

        self.ori_shape = self.pos.shape
        self.cut_param = cut_param

        self.pos = self.padding_img(self.pos, self.cut_param)
        self.pos = self.extract_ordered_overlap(self.pos, self.cut_param)
        self.sub = self.padding_img(self.sub, self.cut_param)
        self.sub = self.extract_ordered_overlap(self.sub, self.cut_param)
        self.adc = self.padding_img(self.adc, self.cut_param)
        self.adc = self.extract_ordered_overlap(self.adc, self.cut_param)
        self.t2w = self.padding_img(self.t2w, self.cut_param)
        self.t2w = self.extract_ordered_overlap(self.t2w, self.cut_param)

        self.Infor_DCE = self.padding_img(self.Infor_DCE, self.cut_param)
        self.Infor_DCE = self.extract_ordered_overlap(self.Infor_DCE, self.cut_param)
        self.Infor_ADC = self.padding_img(self.Infor_ADC, self.cut_param)
        self.Infor_ADC = self.extract_ordered_overlap(self.Infor_ADC, self.cut_param)
        self.Infor_T2 = self.padding_img(self.Infor_T2, self.cut_param)
        self.Infor_T2 = self.extract_ordered_overlap(self.Infor_T2, self.cut_param)

        self.gt = self.padding_img(self.gt, self.cut_param)
        self.gt = self.extract_ordered_overlap(self.gt, self.cut_param)

        self.new_shape = self.pos.shape

    def __getitem__(self, index):
        pos = self.pos[index]
        sub = self.sub[index]
        adc = self.adc[index]
        t2w = self.t2w[index]
        Infor_DCE = self.Infor_DCE[index]
        Infor_ADC = self.Infor_ADC[index]
        Infor_T2 = self.Infor_T2[index]
        gt = self.gt[index]

        return torch.from_numpy(pos), torch.from_numpy(sub), torch.from_numpy(
            adc), torch.from_numpy(t2w), torch.from_numpy(Infor_DCE), torch.from_numpy(Infor_ADC), torch.from_numpy(
            Infor_T2), torch.from_numpy(gt)

    def __len__(self):
        return len(self.pos)

    def padding_img(self, img, C):
        assert (len(img.shape) == 3)  # 3D array
        img_s, img_h, img_w = img.shape
        leftover_s = (img_s - C['patch_s']) % C['stride_s']
        leftover_h = (img_h - C['patch_h']) % C['stride_h']
        leftover_w = (img_w - C['patch_w']) % C['stride_w']
        if (leftover_s != 0):
            s = img_s + (C['stride_s'] - leftover_s)
        else:
            s = img_s

        if (leftover_h != 0):
            h = img_h + (C['stride_h'] - leftover_h)
        else:
            h = img_h

        if (leftover_w != 0):
            w = img_w + (C['stride_w'] - leftover_w)
        else:
            w = img_w

        tmp_full_imgs = np.zeros((s, h, w))
        tmp_full_imgs[:img_s, :img_h, 0:img_w] = img
        return tmp_full_imgs

    # Divide all the full_imgs in pacthes
    def extract_ordered_overlap(self, img, C):
        assert (len(img.shape) == 3)  # 3D arrays
        img_s, img_h, img_w = img.shape
        assert ((img_h - C['patch_h']) % C['stride_h'] == 0
                and (img_w - C['patch_w']) % C['stride_w'] == 0
                and (img_s - C['patch_s']) % C['stride_s'] == 0)
        N_patches_s = (img_s - C['patch_s']) // C['stride_s'] + 1
        N_patches_h = (img_h - C['patch_h']) // C['stride_h'] + 1
        N_patches_w = (img_w - C['patch_w']) // C['stride_w'] + 1
        N_patches_img = N_patches_s * N_patches_h * N_patches_w
        patches = np.empty((N_patches_img, C['patch_s'], C['patch_h'], C['patch_w']))
        iter_tot = 0  # iter over the total number of patches (N_patches)
        for s in range(N_patches_s):  # loop over the full images
            for h in range(N_patches_h):
                for w in range(N_patches_w):
                    patch = img[s * C['stride_s']: s * C['stride_s'] + C['patch_s'],
                            h * C['stride_h']: h * C['stride_h'] + C['patch_h'],
                            w * C['stride_w']: w * C['stride_w'] + C['patch_w']]

                    patches[iter_tot] = patch
                    iter_tot += 1  # total
        assert (iter_tot == N_patches_img)
        return patches  # array with all the full_imgs divided in patches


class Recompone_tool():
    def __init__(self, img_ori_shape, img_new_shape, Cut_para):
        self.result = None
        self.ori_shape = img_ori_shape
        self.new_shape = img_new_shape
        self.C = Cut_para

    def add_result(self, tensor):
        if self.result is not None:
            self.result = torch.cat((self.result, tensor), dim=0)
        else:
            self.result = tensor

    def recompone_overlap(self):
        """
        :param preds: output of model  shape：[N_patchs_img,3,patch_s,patch_h,patch_w]
        :return: result of recompone output shape: [3,img_s,img_h,img_w]
        """
        patch_s = self.result.shape[2]
        patch_h = self.result.shape[3]
        patch_w = self.result.shape[4]
        N_patches_s = (self.new_shape[0] - patch_s) // self.C['stride_s'] + 1
        N_patches_h = (self.new_shape[1] - patch_h) // self.C['stride_h'] + 1
        N_patches_w = (self.new_shape[2] - patch_w) // self.C['stride_w'] + 1
        N_patches_img = N_patches_s * N_patches_h * N_patches_w
        assert (self.result.shape[0] == N_patches_img)

        full_prob = torch.zeros((self.new_shape[0], self.new_shape[1],
                                 self.new_shape[2]))  # itialize to zero mega array with sum of Probabilities
        full_sum = torch.zeros((self.new_shape[0], self.new_shape[1], self.new_shape[2]))
        k = 0  # iterator over all the patches
        for s in range(N_patches_s):
            for h in range(N_patches_h):
                for w in range(N_patches_w):
                    # print(k,self.result[k].squeeze().sum())
                    full_prob[s * self.C['stride_s']:s * self.C['stride_s'] + patch_s,
                    h * self.C['stride_h']:h * self.C['stride_h'] + patch_h,
                    w * self.C['stride_w']:w * self.C['stride_w'] + patch_w] += self.result[k].squeeze()
                    full_sum[s * self.C['stride_s']:s * self.C['stride_s'] + patch_s,
                    h * self.C['stride_h']:h * self.C['stride_h'] + patch_h,
                    w * self.C['stride_w']:w * self.C['stride_w'] + patch_w] += 1
                    k += 1
        assert (k == self.result.size(0))
        assert (torch.min(full_sum) >= 1.0)  # at least one
        final_avg = full_prob / full_sum
        img = final_avg[:self.ori_shape[0], :self.ori_shape[1], :self.ori_shape[2]]
        return img


def cal_newshape(img, C):
    assert (len(img.shape) == 3)  # 3D array
    img_s, img_h, img_w = img.shape
    leftover_s = (img_s - C['patch_s']) % C['stride_s']
    leftover_h = (img_h - C['patch_h']) % C['stride_h']
    leftover_w = (img_w - C['patch_w']) % C['stride_w']
    if (leftover_s != 0):
        s = img_s + (C['stride_s'] - leftover_s)
    else:
        s = img_s

    if (leftover_h != 0):
        h = img_h + (C['stride_h'] - leftover_h)
    else:
        h = img_h

    if (leftover_w != 0):
        w = img_w + (C['stride_w'] - leftover_w)
    else:
        w = img_w

    return np.zeros((s, h, w)).shape


def package_torch(pre_patch, pos_patch, sub_patch, gt_patch):
    pre_patch = torch.from_numpy(pre_patch[np.newaxis, np.newaxis, :])
    pos_patch = torch.from_numpy(pos_patch[np.newaxis, np.newaxis, :])
    sub_patch = torch.from_numpy(sub_patch[np.newaxis, np.newaxis, :])
    gt_patch = torch.from_numpy(gt_patch[np.newaxis, np.newaxis, :])
    return pre_patch, pos_patch, sub_patch, gt_patch


def crop_patch(img, crop_box):
    img_patch = img[crop_box[0, 0]:crop_box[0, 1], crop_box[1, 0]:crop_box[1, 1], crop_box[2, 0]:crop_box[2, 1]]
    return img_patch


def fine_crop(pred, crop_size):
    cor_box = maskcor_extract_3d(pred, (10, 10, 10))
    box_shape = pred.shape
    for i in range(3):
        len_cor = cor_box[i, 1] - cor_box[i, 0]

        if (len_cor <= crop_size[i]):
            cor_box[i, 0] = np.int((cor_box[i, 1] + cor_box[i, 0]) // 2 - crop_size[i] // 2)
            cor_box[i, 1] = cor_box[i, 0] + crop_size[i]
            if cor_box[i, 0] < 0:
                cor_box[i, 0] = 0
                cor_box[i, 1] = cor_box[i, 0] + crop_size[i]
            if cor_box[i, 1] > box_shape[i]:
                cor_box[i, 1] = box_shape[i]
                cor_box[i, 0] = cor_box[i, 1] - crop_size[i]
        elif len_cor <= crop_size[i] * 2:
            cor_box[i, 0] = np.int((cor_box[i, 1] + cor_box[i, 0]) // 2 - crop_size[i])
            cor_box[i, 1] = cor_box[i, 0] + crop_size[i] * 2
            if cor_box[i, 0] < 0:
                cor_box[i, 0] = 0
                cor_box[i, 1] = cor_box[i, 0] + crop_size[i] * 2
            if cor_box[i, 1] > box_shape[i]:
                cor_box[i, 1] = box_shape[i]
                cor_box[i, 0] = cor_box[i, 1] - crop_size[i] * 2
        elif len_cor <= crop_size[i] * 3:
            cor_box[i, 0] = np.int((cor_box[i, 1] + cor_box[i, 0]) // 2 - crop_size[i] * 1.5)
            cor_box[i, 1] = cor_box[i, 0] + crop_size[i] * 3
            if cor_box[i, 0] < 0:
                cor_box[i, 0] = 0
                cor_box[i, 1] = cor_box[i, 0] + crop_size[i] * 3
            if cor_box[i, 1] > box_shape[i]:
                cor_box[i, 1] = box_shape[i]
                cor_box[i, 0] = cor_box[i, 1] - crop_size[i] * 3
        elif len_cor <= crop_size[i] * 4:
            cor_box[i, 0] = np.int((cor_box[i, 1] + cor_box[i, 0]) // 2 - crop_size[i] * 2)
            cor_box[i, 1] = cor_box[i, 0] + crop_size[i] * 4
            if cor_box[i, 0] < 0:
                cor_box[i, 0] = 0
                cor_box[i, 1] = cor_box[i, 0] + crop_size[i] * 4
            if cor_box[i, 1] > box_shape[i]:
                cor_box[i, 1] = box_shape[i]
                cor_box[i, 0] = cor_box[i, 1] - crop_size[i] * 4
        elif len_cor <= crop_size[i] * 5:
            cor_box[i, 0] = np.int((cor_box[i, 1] + cor_box[i, 0]) // 2 - crop_size[i] * 2.5)
            cor_box[i, 1] = cor_box[i, 0] + crop_size[i] * 5
            if cor_box[i, 0] < 0:
                cor_box[i, 0] = 0
                cor_box[i, 1] = cor_box[i, 0] + crop_size[i] * 5
            if cor_box[i, 1] > box_shape[i]:
                cor_box[i, 1] = box_shape[i]
                cor_box[i, 0] = cor_box[i, 1] - crop_size[i] * 5
        elif len_cor <= crop_size[i] * 6:
            cor_box[i, 0] = np.int((cor_box[i, 1] + cor_box[i, 0]) // 2 - crop_size[i] * 3)
            cor_box[i, 1] = cor_box[i, 0] + crop_size[i] * 6
            if cor_box[i, 0] < 0:
                cor_box[i, 0] = 0
                cor_box[i, 1] = cor_box[i, 0] + crop_size[i] * 6
            if cor_box[i, 1] > box_shape[i]:
                cor_box[i, 1] = box_shape[i]
                cor_box[i, 0] = cor_box[i, 1] - crop_size[i] * 6
        elif len_cor <= crop_size[i] * 7:
            cor_box[i, 0] = np.int((cor_box[i, 1] + cor_box[i, 0]) // 2 - crop_size[i] * 3.5)
            cor_box[i, 1] = cor_box[i, 0] + crop_size[i] * 7
            if cor_box[i, 0] < 0:
                cor_box[i, 0] = 0
                cor_box[i, 1] = cor_box[i, 0] + crop_size[i] * 7
            if cor_box[i, 1] > box_shape[i]:
                cor_box[i, 1] = box_shape[i]
                cor_box[i, 0] = cor_box[i, 1] - crop_size[i] * 7
        elif len_cor <= crop_size[i] * 8:

            cor_box[i, 0] = np.int((cor_box[i, 1] + cor_box[i, 0]) // 2 - crop_size[i] * 4)
            cor_box[i, 1] = cor_box[i, 0] + crop_size[i] * 8
            if cor_box[i, 0] < 0:
                cor_box[i, 0] = 0
                cor_box[i, 1] = cor_box[i, 0] + crop_size[i] * 8
            if cor_box[i, 1] > box_shape[i]:
                cor_box[i, 1] = box_shape[i]
                cor_box[i, 0] = cor_box[i, 1] - crop_size[i] * 8
        else:
            print('too large tumor')
            cor_box[i, 0] = 0
            cor_box[i, 1] = box_shape[i]

        if (cor_box[i, 1] - cor_box[i, 0]) >= box_shape[i]:
            cor_box[i, 1] = box_shape[i]
            cor_box[i, 0] = 0

        if (cor_box[i, 1] - cor_box[i, 0]) != box_shape[i]:
            if (cor_box[i, 1] - cor_box[i, 0]) % crop_size[i] != 0:
                print('Something goes wrong!')
        if (cor_box[i, 1] - cor_box[i, 0]) == box_shape[i]:
            cor_box[i, 1] = (len_cor // crop_size[i]) * crop_size[i] + cor_box[i, 0]
    return cor_box


def Test_all_Datasets(dataset_path, size):
    f = open(os.path.join(dataset_path, 'test.txt'))
    data_list = f.read().splitlines()
    print("The number of test samples is: ", len(data_list))
    for file in data_list:
        print("\nStart Evaluate: ", file)
        pre, pre_min, pre_max = normalization(load(os.path.join(dataset_path, file, 'DCE0.nii.gz')))
        pos = normalization_fix(load(os.path.join(dataset_path, file, 'DCE.nii.gz')), pre_min, pre_max).astype(
            np.float32)
        sub = pos - pre
        adc = normalization_org(load(os.path.join(dataset_path, file, 'ADC.nii.gz'))).astype(np.float32)
        t2w = normalization_org(load(os.path.join(dataset_path, file, 'T2.nii.gz'))).astype(np.float32)

        Infor_DCE = load(os.path.join(dataset_path, file, 'Infor_DCE.nii.gz')).astype(np.float32)
        Infor_ADC = load(os.path.join(dataset_path, file, 'Infor_ADC.nii.gz')).astype(np.float32)
        Infor_T2 = load(os.path.join(dataset_path, file, 'Infor_T2.nii.gz')).astype(np.float32)
        gt = load(os.path.join(dataset_path, file, 'GT.nii.gz')).astype(np.int16)

        breast_mask = load(os.path.join(dataset_path, file, 'Breast_mask.nii.gz')).astype(np.int16)
        pred = load(os.path.join(dataset_path, file, 'pred_coarse.nii.gz')).astype(np.float32)

        cor_box = fine_crop(pred, (32, 128, 128))
        pos = crop_patch(pos, cor_box)
        sub = crop_patch(sub, cor_box)
        t2w = crop_patch(t2w, cor_box)
        adc = crop_patch(adc, cor_box)
        Infor_DCE = crop_patch(Infor_DCE, cor_box)
        Infor_ADC = crop_patch(Infor_ADC, cor_box)
        Infor_T2 = crop_patch(Infor_T2, cor_box)
        gt = crop_patch(gt, cor_box)

        original_shape = gt.shape
        new_shape = cal_newshape(gt, size)

        yield Img_DataSet(pos, sub, adc, t2w, Infor_DCE, Infor_ADC, Infor_T2, gt, size), original_shape, new_shape, breast_mask, file, cor_box
