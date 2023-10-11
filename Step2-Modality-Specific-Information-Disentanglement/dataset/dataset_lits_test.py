import numpy as np
import torch, os
from torch.utils.data import Dataset
import random
import SimpleITK as sitk


def min_max_normalization(img):
    out = (img - np.min(img)) / (np.max(img) - np.min(img) + 0.000001)
    return out


def normalization(img, lmin=1, rmax=None, dividend=None, quantile=None):
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
    p = np.where(mask > 0)
    a = np.zeros([3, 2], dtype=np.int)
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
    def __init__(self, pre, pos, sub, adc, t2w, gt, cut_param):
        self.pre = pre
        self.pos = pos
        self.sub = sub
        self.adc = adc
        self.t2w = t2w
        self.gt = gt

        self.ori_shape = self.pre.shape
        self.cut_param = cut_param

        self.pre = self.padding_img(self.pre, self.cut_param)
        self.pre = self.extract_ordered_overlap(self.pre, self.cut_param)
        self.pos = self.padding_img(self.pos, self.cut_param)
        self.pos = self.extract_ordered_overlap(self.pos, self.cut_param)
        self.sub = self.padding_img(self.sub, self.cut_param)
        self.sub = self.extract_ordered_overlap(self.sub, self.cut_param)
        self.adc = self.padding_img(self.adc, self.cut_param)
        self.adc = self.extract_ordered_overlap(self.adc, self.cut_param)
        self.t2w = self.padding_img(self.t2w, self.cut_param)
        self.t2w = self.extract_ordered_overlap(self.t2w, self.cut_param)
        self.gt = self.padding_img(self.gt, self.cut_param)
        self.gt = self.extract_ordered_overlap(self.gt, self.cut_param)

        self.new_shape = self.pre.shape

    def __getitem__(self, index):
        pre = self.pre[index]
        pos = self.pos[index]
        sub = self.sub[index]
        adc = self.adc[index]
        t2w = self.t2w[index]
        gt = self.gt[index]

        return torch.from_numpy(pre), torch.from_numpy(pos), torch.from_numpy(sub), torch.from_numpy(
            adc), torch.from_numpy(t2w), torch.from_numpy(gt)

    def __len__(self):
        return len(self.pre)

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
        iter_tot = 0
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
                                 self.new_shape[2]))
        full_sum = torch.zeros((self.new_shape[0], self.new_shape[1], self.new_shape[2]))
        k = 0
        for s in range(N_patches_s):
            for h in range(N_patches_h):
                for w in range(N_patches_w):
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


def Test_all_Datasets(dataset_path, size, flag):
    f = open(os.path.join(dataset_path, 'train.txt'))
    data_list = f.read().splitlines()
    print("The number of test samples is: ", len(data_list))
    for file in data_list:
        print("\nStart Evaluate: ", file)
        pre = normalization(load(os.path.join(dataset_path, file, 'DCE0.nii.gz'))).astype(np.float32)
        pos = normalization(load(os.path.join(dataset_path, file, 'DCE.nii.gz'))).astype(np.float32)
        sub = pos - pre
        if flag == 'all_true':
            adc = normalization(load(os.path.join(dataset_path, file, 'ADC.nii.gz'))).astype(np.float32)
            t2w = normalization(load(os.path.join(dataset_path, file, 'T2.nii.gz'))).astype(np.float32)
        elif flag == 'fake_adc':
            adc = load(os.path.join(dataset_path, file, 'ADC_syn.nii.gz')).astype(np.float32)
            t2w = normalization(load(os.path.join(dataset_path, file, 'T2.nii.gz'))).astype(np.float32)
        elif flag == 'fake_t2':
            adc = normalization(load(os.path.join(dataset_path, file, 'ADC.nii.gz'))).astype(np.float32)
            t2w = load(os.path.join(dataset_path, file, 'T2_syn.nii.gz')).astype(np.float32)
        elif flag == 'all_fake':
            adc = load(os.path.join(dataset_path, file, 'ADC_syn.nii.gz')).astype(np.float32)
            t2w = load(os.path.join(dataset_path, file, 'T2_syn.nii.gz')).astype(np.float32)

        gt = load(os.path.join(dataset_path, file, 'GT.nii.gz')).astype(np.int16)
        breast_mask = load(os.path.join(dataset_path, file, 'Breast_mask.nii.gz')).astype(np.int16)
        original_shape = gt.shape
        new_shape = cal_newshape(gt, size)

        yield Img_DataSet(pre, pos, sub, adc, t2w, gt, size), original_shape, new_shape, breast_mask, file
