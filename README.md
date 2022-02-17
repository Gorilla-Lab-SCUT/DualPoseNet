# DualPoseNet
Code for "DualPoseNet: Category-level 6D Object Pose and Size Estimation Using Dual Pose Network with Refined Learning of Pose Consistency", ICCV2021. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Lin_DualPoseNet_Category-Level_6D_Object_Pose_and_Size_Estimation_Using_Dual_ICCV_2021_paper.pdf)][[Supp](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Lin_DualPoseNet_Category-Level_6D_ICCV_2021_supplemental.pdf)][[Arxiv](https://arxiv.org/abs/2103.06526)]

Created by Jiehong Lin, Zewei Wei, Zhihao Li, Songcen Xu, [Kui Jia](http://kuijia.site/), and Yuanqing Li.

![image](https://github.com/Gorilla-Lab-SCUT/DualPoseNet/blob/main/doc/FigNetwork2.png)


## Citation
If you find our work useful in your research, please consider citing:

     @InProceedings{Lin_2021_ICCV,
         author    = {Lin, Jiehong and Wei, Zewei and Li, Zhihao and Xu, Songcen and Jia, Kui and Li, Yuanqing},
         title     = {DualPoseNet: Category-Level 6D Object Pose and Size Estimation Using Dual Pose Network With Refined Learning of Pose Consistency},
         booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
         month     = {October},
         year      = {2021},
         pages     = {3560-3569}
     }

## Requirements
This code has been tested with
- python 3.6.5
- tensorflow-gpu 1.11.0
- CUDA 11.2

## Downloads
- Pre-trained models [[link](https://drive.google.com/file/d/16DVaudTE_K_dqXbKoW7SASgxRQnfHApI/view?usp=sharing)]
- Segmentation predictions on CAMERA25 and REAL275 [[link](https://drive.google.com/file/d/1RwAbFWw2ITX9mXzLUEBjPy_g-MNdyHET/view?usp=sharing)]
- Pose Predicitons on CAMERA25 and REAL275 [[link](https://drive.google.com/file/d/10TBFY73BMmTxfErlbMqZKfClZgxMzMd9/view?usp=sharing)]

## Evaluation
Evaluate the results of DualPoseNet reported in the paper:

```
python eval.py
```
## Data Preparation

Download the data provided by [NOCS](https://github.com/hughw19/NOCS_CVPR2019) ([real_train](http://download.cs.stanford.edu/orion/nocs/real_train.zip), [real_test](http://download.cs.stanford.edu/orion/nocs/real_test.zip),
[ground truths](http://download.cs.stanford.edu/orion/nocs/gts.zip),
and [mesh models](http://download.cs.stanford.edu/orion/nocs/obj_models.zip)), and unzip them in the file ```data/``` as follows:

```
data
├── CAMERA
│   ├── train
│   └── val
├── Real
│   ├── train
│   └── test
├── gts
│   ├── val
│   └── real_test
└── obj_models
    ├── train
    ├── val
    ├── real_train
    └── real_test
```

Run the following scripts to prepare training instances:

```
cd provider
python training_data_prepare.py
```

## Training

Command for training DualPoseNet:
```
python main.py --phase train --dataset REAL275
```
The configurations can be modified in ```utils/config.py```.

## Testing
Command for testing DualPoseNet without refined learning:
```
python main.py --phase test --dataset REAL275
```

Command for testing DualPoseNet with refined learning:
```
python main.py --phase test_refine_encoder --dataset REAL275
```

We also provider another faster way of refinement by directly finetuning the pose-sensitive features:
```
python main.py --phase test_refine_feature --dataset REAL275
```

The configurations can also be modified in ```utils/config.py```.

## Acknowledgements

Our implementation leverages the code from [SCNN](https://github.com/daniilidis-group/spherical-cnn), [NOCS](https://github.com/hughw19/NOCS_CVPR2019) and [SPD](https://github.com/mentian/object-deformnet).

## License
Our code is released under MIT License (see LICENSE file for details).

## Contact
`lin.jiehong@mail.scut.edu.cn`

`kuijia@scut.edu.cn`


