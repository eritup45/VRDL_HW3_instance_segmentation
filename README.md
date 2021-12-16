# VRDL_HW3_instance_segmentation

### Objective
* Instance segmentation task to detect and segment all the nuclei.

![](https://i.imgur.com/tEKtEuV.jpg)



* Data: 
    * Nuclear segmentation dataset contains 24 training images with 14,598 nuclear and 6 test images with 2,360 nuclear.
        * train: 20 images, 12188 nuclear
        * val: 3 images, 2410 nuclear
        * test: 6 images, 2360 nuclear

 


### Environment
- Python 3.7
- Pytorch 1.6.0
- CUDA 10.2

## Reproducing Submission
To reproduce, do the following steps:
1. [Installation](#install-packages)
2. [Data Preparation](#data-preparation)
3. [Select Config file](#select-config-file)
4. [Download Pretrained Model](#download-pretrained-model)
5. [Training](#training)
6. [Testing](#testing)
7. [Results](#results)
8. [Report](#report)
9. [Reference](#reference)



### Install Packages
- install pytorch from https://pytorch.org/get-started/previous-versions/

- install mmdetection
    * Please refer to [mmdetection](https://github.com/open-mmlab/mmdetection) for installation.
    ```
    # quick install:
    # install pytorch correspond to your cuda version
    conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
    pip install openmim
    mim install mmdet
    ```

- install dependencies
    * scikit-image
    * tqdm
    * apex (optional for swin transformer)

### Data Preparation
* Recommend Way
  1. Download dataset which has been preprocessed. [Google Drive](https://drive.google.com/drive/folders/1M0n4axh-rFseZ86aeSgUrCPZ-87mcZ_R?usp=sharing). The data should be as follow.
  ```
  data
  └── coco
      ├── annotations
      │   ├── train_coco.json
      │   ├── val_coco.json
      │   └── test_coco.json
      ├── train2017/train_pic
      │   ├── 1.png
      │   ├── 2.png
      │   └── ...
      ├── val2017/val_pic
      │   ├── 1.png
      │   ├── 2.png
      │   └── ...
      └── test2017/test
          ├── 1.png
          ├── 2.png
          └── ...
  ```


* Or, Prepare Data From Scratch
  1. Download the given dataset from [Google Drive](https://drive.google.com/drive/folders/19r3ic3Z39vS6MHx4115926kGEbK3wqPr?usp=sharing).
 
  2. Split train data into "train" and "val" folder. (`python my_data_preprocess/data_process.py`)
  3. Run command `python my_data_preprocess/data_process.py` to create coco format input data.
  
  * The result will be like:
  ```
  - dataset/
    ├── train_coco.json
    ├── val_coco.json
    ├── test_coco.json
    ├── train_pic 
    ├── val_pic
    │   ├── 1.png
    │   └── ...    
    └── test
        ├── 1.png
        └── ...
  ```
  3. Reorganize data as [Data Preparation](#data-preparation)

### Select Config file
* cascade mask rcnn: `config_cascade_mask_rcnn_50.py`
* swin mask rcnn: `config_swinT.py`
* detectoRS: `config_detectoRS.py`

### Download Pretrained Model
- [Google Drive](https://drive.google.com/drive/folders/18rKU28lb-DmLwHfTR-io2vkAd1wBpGgn?usp=sharing)

### Training
- train model with pretrained model
```
# cd mmdetection
python tools/train.py config_cascade_mask_rcnn_50.py
```
> Edit "load_from" "data"
> in config_cascade_mask_rcnn_50.py 

### Testing
- Inference test data
```
python my_inference.py
```

- The result will be stored in "./answer.json". 
- Format of "./answer.json" is same as COCO results.


### Results
* Results: 
    * val result:
        * bbox_mAP: 0.061
        * segm_mAP: 0.060
    * Test mAP: 0.233557 (41)

### Interesting finding
* Using mmdetection tips:
	1. Pick a config file in ./configs/... 
	2. Run it anyway. 
	3. Then you can get the complete config file.

### Report
https://hackmd.io/@Bmj6Z_QbTMy769jUvLGShA/VRDL_HW3

### Reference

[mmdetection github](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn)
[建置到訓練](http://lanck.lzu.edu.cn/?p=490)
https://cxyzjd.com/article/Skies_/108142131
https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn
https://arxiv.org/pdf/1906.09756.pdf
https://www.itread01.com/content/1547120287.html








