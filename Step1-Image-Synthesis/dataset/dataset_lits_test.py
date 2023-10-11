import numpy as np
import torch, os
from torch.utils.data import Dataset, DataLoader
from glob import glob
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


def random_crop_3d(adc, pos, sub, gt, crop_size):
    cor_box = maskcor_extract_3d(gt)
    random_x_min, random_x_max = max(cor_box[0, 1] - crop_size[0], 0), min(cor_box[0, 0], adc.shape[0] - crop_size[0])
    random_y_min, random_y_max = max(cor_box[1, 1] - crop_size[1], 0), min(cor_box[1, 0], adc.shape[1] - crop_size[1])
    random_z_min, random_z_max = max(cor_box[2, 1] - crop_size[2], 0), min(cor_box[2, 0], adc.shape[2] - crop_size[2])
    if random_x_min > random_x_max:
        random_x_min, random_x_max = cor_box[0, 0], cor_box[0, 1] - crop_size[0]
    if random_y_min > random_y_max:
        random_y_min, random_y_max = cor_box[1, 0], cor_box[1, 1] - crop_size[1]
    if random_z_min > random_z_max:
        random_z_min, random_z_max = cor_box[2, 0], cor_box[2, 1] - crop_size[2]

    x_random = random.randint(random_x_min, random_x_max)
    y_random = random.randint(random_y_min, random_y_max)
    z_random = random.randint(random_z_min, random_z_max)

    adc_patch = adc[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                z_random:z_random + crop_size[2]]
    pos_patch = pos[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                z_random:z_random + crop_size[2]]
    sub_patch = sub[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                z_random:z_random + crop_size[2]]
    gt_patch = gt[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
               z_random:z_random + crop_size[2]]

    return adc_patch, pos_patch, sub_patch, gt_patch


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
    def __init__(self, adc, cut_param):
        self.adc = adc

        self.ori_shape = self.adc.shape
        self.cut_param = cut_param

        self.adc = self.padding_img(self.adc, self.cut_param)
        self.adc = self.extract_ordered_overlap(self.adc, self.cut_param)

        self.new_shape = self.adc.shape

    def __getitem__(self, index):
        adc = self.adc[index]

        return torch.from_numpy(adc)

    def __len__(self):
        return len(self.adc)

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
        assert (len(img.shape) == 3)
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
        for s in range(N_patches_s):
            for h in range(N_patches_h):
                for w in range(N_patches_w):
                    patch = img[s * C['stride_s']: s * C['stride_s'] + C['patch_s'],
                            h * C['stride_h']: h * C['stride_h'] + C['patch_h'],
                            w * C['stride_w']: w * C['stride_w'] + C['patch_w']]

                    patches[iter_tot] = patch
                    iter_tot += 1
        assert (iter_tot == N_patches_img)
        return patches


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
        :param adcds: output of model  shape：[N_patchs_img,3,patch_s,patch_h,patch_w]
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
        assert (torch.min(full_sum) >= 1.0)
        final_avg = full_prob / full_sum
        img = final_avg[:self.ori_shape[0], :self.ori_shape[1], :self.ori_shape[2]]
        return img


def cal_newshape(img, C):
    assert (len(img.shape) == 3)
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


def package_torch(adc_patch, pos_patch, sub_patch, gt_patch):
    adc_patch = torch.from_numpy(adc_patch[np.newaxis, np.newaxis, :])
    pos_patch = torch.from_numpy(pos_patch[np.newaxis, np.newaxis, :])
    sub_patch = torch.from_numpy(sub_patch[np.newaxis, np.newaxis, :])
    gt_patch = torch.from_numpy(gt_patch[np.newaxis, np.newaxis, :])
    return adc_patch, pos_patch, sub_patch, gt_patch


def Test_Datasets(dataset_path, size, test_folder=1):
    f = open(os.path.join(dataset_path, 'data_folder', 'test' + str(test_folder) + '.txt'))
    data_list = f.read().splitlines()
    print("The number of test samples is: ", len(data_list))
    for file in data_list:
        adc = load(os.path.join(dataset_path, file, 'adc_contrast.nii.gz')).astype(np.float32)
        pos = load(os.path.join(dataset_path, file, 'Pos_contrast.nii.gz')).astype(np.float32)
        sub = normalization(pos - adc)
        print(sub.shape)
        gt = load(os.path.join(dataset_path, file, 'GT.nii.gz')).astype(np.int16)
        adc_patch, pos_patch, sub_patch, gt_patch = random_crop_3d(adc, pos, sub, gt, size)

        yield package_torch(adc_patch, pos_patch, sub_patch, gt_patch), file


def Test_all_Datasets(dataset_path, size):
    f = open(os.path.join(dataset_path, 'data.txt'))
    data_list = f.read().splitlines()
    print("The number of test samples is: ", len(data_list))
    for file in data_list:
        print("\nStart Evaluate: ", file)
        DCE = normalization(load(os.path.join(dataset_path, file, 'DCE.nii.gz'))).astype(np.float32)
        original_shape = DCE.shape
        new_shape = cal_newshape(DCE, size)

        yield Img_DataSet(DCE, size), original_shape, new_shape, file
