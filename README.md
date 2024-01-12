# Modality-Specific Information Disentanglement from Multi-parametric MRI for Breast Tumor Segmentation and Computer-aided Diagnosis

## Paper:
Please see:   
  
 
* Journal paper: Modality-Specific Information Disentanglement from Multi-parametric MRI for Breast Tumor Segmentation and Computer-aided Diagnosis (https://ieeexplore.ieee.org/document/10388458)
* Confernce paper: MoSID: Modality-Specific Information Disentanglement from Multi-parametric MRI for Breast Tumor Segmentation (https://link.springer.com/chapter/10.1007/978-3-031-45350-2_8)  


## Introduction:
This project includes both train/test code for training the MoSID framwork.

![image](https://github.com/Qianqian-Chen/MoSID/blob/main/framework.png)

## Requirements:
* python 3.10
* pytorch 1.12.1
* numpy 1.23.3
* tensorboard 2.10.1
* simpleitk 2.1.1.1
* scipy 1.9.1

## Setup

### Dataset
* For training the segmentation models, you need to put the data in this format：  
```
./data
├─train.txt
├─test.txt
├─valid.txt
├─MRI1
      ├─ADC.nii.gz
      ├─T2w.nii.gz
      ├─P0.nii.gz
      ├─P2.nii.gz   
      └─GT.nii.gz
      ...
├─MRI99        
└─MRI100
... 
```

* The format of the train.txt / test.txt / valid.txt is as follow：    
```
./data/train.txt
├─'MRI1'
├─'MRI2'
├─'MRI3'
...
├─'MRI100'
...
```

### Whole Breast Segmentation Model
* The whole breast segmentation process can be used to remove the oversegmentation on non-breast regions.
* Partial images and whole breast annotations are available at: https://github.com/ZhangJD-ong/AI-assistant-for-breast-tumor-segmentation

## Citation
If you find the code useful, please consider citing the following papers:

* Chen et al., Modality-Specific Information Disentanglement from Multi-parametric MRI for Breast Tumor Segmentation and Computer-aided Diagnosis, IEEE Transactions on Medical Imaging (2023), https://doi.org/10.1109/TMI.2024.3352648
* Zhang et al., MoSID: Modality-Specific Information Disentanglement from Multi-parametric MRI for Breast Tumor Segmentation, MICCAI Workshop on Cancer Prevention through Early Detection (2023), https://doi.org/10.1007/978-3-031-45350-2_8
* Zhang et al., Recent advancements in artificial intelligence for breast cancer: Image augmentation, segmentation, diagnosis, and prognosis approaches, Seminars in Cancer Biology (2023), https://doi.org/10.1016/j.semcancer.2023.09.001
