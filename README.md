# DualPoseNet
Code for "DualPoseNet: Category-level 6D Object Pose and Size Estimation Using Dual Pose Network with Refined Learning of Pose Consistency", ICCV2021. [Paper][[Arxiv](https://arxiv.org/abs/2103.06526)]

Created by Jiehong Lin, Zewei Wei, Zhihao Li, Songcen Xu, [Kui Jia](http://kuijia.site/), and Yuanqing Li.

![image](https://github.com/Gorilla-Lab-SCUT/DualPoseNet/blob/main/doc/FigNetwork2.png)


## Citation
If you find our work useful in your research, please consider citing:

     @article{lin2021dualposenet,
        title={DualPoseNet: Category-level 6D Object Pose and Size Estimation Using Dual Pose Network with Refined Learning of Pose Consistency},
        author={Lin, Jiehong and Wei, Zewei and Li, Zhihao and Xu, Songcen and Jia, Kui and Li, Yuanqing},
        journal={arXiv preprint arXiv:2103.06526},
        year={2021}
      }

## Requirements
This code has been tested with
- python 3.6.5
- tensorflow-gpu 1.11.0
- CUDA 11.2

## Downloads
- Pre-trained models [link]
- Segmantation predictions on CAMERA25 and REAL275 [link]
- Pose Predicitons on CAMERA25 and REAL275 [link]

## Evaluation
Evaluate the results of DualPoseNet reported in the paper:

```
python eval.py
```

## Training
```
python main.py --phase train
```

## Test
```
python main.py --phase test
```

## Acknowledgements

Our implementation leverages the code from [SCNN](https://github.com/daniilidis-group/spherical-cnn) and [NOCS](https://github.com/hughw19/NOCS_CVPR2019).

## License
Our code is released under MIT License (see LICENSE file for details).

## Contact
`lin.jiehong@mail.scut.edu.cn`

`kuijia@scut.edu.cn`


