import random
import numpy as np
import SimpleITK as sitk
import os
from torch.utils.data import Dataset


class Lits_DataSet(Dataset):
    def __init__(self, root, sample_index='partial', size=(32, 128, 128)):
        self.root = root
        self.size = size
        self.sample_index = sample_index
        f = open(os.path.join(self.root, 'train.txt'))
        self.filename = f.read().splitlines()

    def __getitem__(self, index):

        file = self.filename[index]
        DCE0, DCE0_Min, DCE0_Max = self.normalization(self.load(os.path.join(self.root, file, 'DCE0.nii.gz')))
        DCE = self.normalization_fix(self.load(os.path.join(self.root, file, 'DCE.nii.gz')), DCE0_Min, DCE0_Max).astype(
            np.float32)
        sub = DCE - DCE0
        Coarse = self.load(os.path.join(self.root, file, 'pred_coarse.nii.gz')).astype(np.float32)
        ADC = self.normalization_org(self.load(os.path.join(self.root, file, 'ADC.nii.gz'))).astype(np.float32)
        T2W = self.normalization_org(self.load(os.path.join(self.root, file, 'T2.nii.gz'))).astype(np.float32)
        Infor_DCE = self.load(os.path.join(self.root, file, 'Infor_DCE.nii.gz')).astype(np.float32)
        Infor_ADC = self.load(os.path.join(self.root, file, 'Infor_ADC.nii.gz')).astype(np.float32)
        Infor_T2 = self.load(os.path.join(self.root, file, 'Infor_T2.nii.gz')).astype(np.float32)
        gt = self.load(os.path.join(self.root, file, 'GT.nii.gz')).astype(np.float32)

        DCE_patch, sub_patch, ADC_patch, T2W_patch, Infor_DCE_patch, Infor_ADC_patch, Infor_T2_patch, gt_patch = [], [], [], [], [], [], [], []
        for i in range(3):
            if i == 1:
                DCE_patch1, sub_patch1, ADC_patch1, T2W_patch1, Infor_DCE_patch1, Infor_ADC_patch1, Infor_T2_patch1, gt_patch1 = self.random_crop_3d_partial(
                    DCE, sub, ADC, T2W, Infor_DCE, Infor_ADC, Infor_T2, gt, Coarse, self.size, 'add')
            else:
                DCE_patch1, sub_patch1, ADC_patch1, T2W_patch1, Infor_DCE_patch1, Infor_ADC_patch1, Infor_T2_patch1, gt_patch1 = self.random_crop_3d_contain(
                    DCE, sub, ADC, T2W, Infor_DCE, Infor_ADC, Infor_T2, gt, Coarse, self.size, 'sub')

            DCE_patch.append(DCE_patch1), sub_patch.append(sub_patch1), ADC_patch.append(ADC_patch1), T2W_patch.append(
                T2W_patch1), Infor_DCE_patch.append(
                Infor_DCE_patch1), Infor_ADC_patch.append(Infor_ADC_patch1), Infor_T2_patch.append(
                Infor_T2_patch1), gt_patch.append(gt_patch1)

        return np.array(DCE_patch), np.array(sub_patch), np.array(ADC_patch), np.array(T2W_patch), np.array(
            Infor_DCE_patch), np.array(Infor_ADC_patch), np.array(Infor_T2_patch), np.array(gt_patch)

    def __len__(self):
        return len(self.filename)

    def random_crop_3d_contain(self, a, b, c, d, e, f, g, gt, coarse, crop_size, pattern='sub'):
        if pattern == 'sub':
            cor_box = self.maskcor_extract_3d(np.abs(gt - coarse))
        else:
            cor_box = self.maskcor_extract_3d(np.abs(gt + coarse))

        random_x_min, random_x_max = max(cor_box[0, 1] - crop_size[0], 0), min(cor_box[0, 0], a.shape[0] - crop_size[0])
        random_y_min, random_y_max = max(cor_box[1, 1] - crop_size[1], 0), min(cor_box[1, 0], a.shape[1] - crop_size[1])
        random_z_min, random_z_max = max(cor_box[2, 1] - crop_size[2], 0), min(cor_box[2, 0], a.shape[2] - crop_size[2])
        if random_x_min > random_x_max:
            random_x_min, random_x_max = cor_box[0, 0], cor_box[0, 1] - crop_size[0]
        if random_y_min > random_y_max:
            random_y_min, random_y_max = cor_box[1, 0], cor_box[1, 1] - crop_size[1]
        if random_z_min > random_z_max:
            random_z_min, random_z_max = cor_box[2, 0], cor_box[2, 1] - crop_size[2]

        x_random = random.randint(random_x_min, random_x_max)
        y_random = random.randint(random_y_min, random_y_max)
        z_random = random.randint(random_z_min, random_z_max)

        a_patch = a[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                  z_random:z_random + crop_size[2]]
        b_patch = b[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                  z_random:z_random + crop_size[2]]
        c_patch = c[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                  z_random:z_random + crop_size[2]]
        d_patch = d[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                  z_random:z_random + crop_size[2]]
        e_patch = e[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                  z_random:z_random + crop_size[2]]
        f_patch = f[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                  z_random:z_random + crop_size[2]]
        g_patch = g[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                  z_random:z_random + crop_size[2]]
        gt_patch = gt[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                   z_random:z_random + crop_size[2]]

        return a_patch, b_patch, c_patch, d_patch, e_patch, f_patch, g_patch, gt_patch

    def random_crop_3d_partial(self, a, b, c, d, e, f, g, gt, coarse, crop_size, pattern='sub'):
        if pattern == 'sub':
            cor_box = self.maskcor_extract_3d(np.abs(gt - coarse))
        else:
            cor_box = self.maskcor_extract_3d(np.abs(gt + coarse))
        random_x_min, random_x_max = max(cor_box[0, 0] - crop_size[0], 0), min(cor_box[0, 1], a.shape[0] - crop_size[0])
        random_y_min, random_y_max = max(cor_box[1, 0] - crop_size[1], 0), min(cor_box[1, 1], a.shape[1] - crop_size[1])
        random_z_min, random_z_max = max(cor_box[2, 0] - crop_size[2], 0), min(cor_box[2, 1], a.shape[2] - crop_size[2])
        x_random = random.randint(random_x_min, random_x_max)
        y_random = random.randint(random_y_min, random_y_max)
        z_random = random.randint(random_z_min, random_z_max)

        a_patch = a[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                  z_random:z_random + crop_size[2]]
        b_patch = b[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                  z_random:z_random + crop_size[2]]
        c_patch = c[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                  z_random:z_random + crop_size[2]]
        d_patch = d[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                  z_random:z_random + crop_size[2]]
        e_patch = e[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                  z_random:z_random + crop_size[2]]
        f_patch = f[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                  z_random:z_random + crop_size[2]]
        g_patch = g[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                  z_random:z_random + crop_size[2]]
        gt_patch = gt[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                   z_random:z_random + crop_size[2]]

        return a_patch, b_patch, c_patch, d_patch, e_patch, f_patch, g_patch, gt_patch

    def min_max_normalization(self, img):
        out = (img - np.min(img)) / (np.max(img) - np.min(img) + 0.000001)
        return out

    def normalization_org(self, img, lmin=1, rmax=None, dividend=None, quantile=None):
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

    def normalization(self, img, lmin=1, rmax=None, dividend=None, quantile=1):
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

    def normalization_fix(self, img, minval, maxval, lmin=1):
        newimg = img.copy()
        newimg = newimg.astype(np.float32)
        if lmin is not None:
            newimg[newimg < lmin] = lmin

        newimg = (np.asarray(newimg).astype(np.float32) - minval) / (maxval - minval)
        return newimg

    def load(self, file):
        itkimage = sitk.ReadImage(file)
        image = sitk.GetArrayFromImage(itkimage)
        return image

    def maskcor_extract_3d(self, mask, padding=(0, 0, 0)):
        # mask_s = mask.shape
        if np.sum(mask) == 0:
            mask[10:12, 100:102, 100:102] = 1

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
