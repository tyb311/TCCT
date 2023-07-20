

#   Solutions for MICCAI2022-GOALS@Baidu
-   MICCAI2022 Challenge: Glaucoma Oct Analysis and Layer Segmentation (GOALS)
-   GOALS Challenge:https://aistudio.baidu.com/aistudio/competition/detail/230/0/introduction
-   Implementation by PaddlePaddle/PyTorch
-   Pytorch implementation of the paper "Retinal Layer Segmentation in OCT images with
Boundary Regression and Feature Polarization".

-   The project's code is constantly being updated, but sorting out the original code may take a few days, please be patient!
<!-- 
![tease](https://github.com/FuyaLuo/PearlGAN/blob/main/docs/Model.PNG)

### [Paper](https://ieeexplore.ieee.org/abstract/document/9703249) -->

![TCCT-ViT&CNN combined Net](docs/net.png)
![TCCT-Feature Polarization](docs/fpl.png)
![TCCT-Segmentation Results](docs/seg.png)

## Prerequisites
* Python 3.8 
* Paddle 2.3.2
* Pytorch 1.13.0

##  Task1:Layer Segmentation
    Model:TCCT/PyTorch



```
Project for task1:Segmentation
    ├── data (code for datasets)
        ├── tran.py (some python imports)  
        ├── octnpy.py (parent class for OCT datasets)  
        ├── octgen.py (child class for OCT datasets)  
        └── ...  
    ├── kite (package for segmentation with torch)  
        ├── loop_seg.py (child class for training)  
        ├── loopback.py (parent class for training)  
        ├── main.py   
        └── ...  
    ├── nets (related models)  
        ├── fcp.py (Feature Polarization Loss - file1)  
        ├── fcs.py (Feature Polarization Loss - file2)  
        ├── reg.py (loss functions [feature polarization & boundary regression])  
        ├── tcct.py (Tightly combined Cross-Convolution and Transformer)  
        └── ...   
    ├── onnx (trained weights)  
        ├── onnx.py (code to inference OCT images with *.onnx files)  
        └── ...   
```

And for the training on GOALS dataset, run the command
```bash
CUDA_VISIBLE_DEVICES=0 python kite/main.py --bs=8 --net=stc_tt --los=di --epochs=100 --db=goals
```
And for the training on HCMS dataset, run the command
```bash
CUDA_VISIBLE_DEVICES=1 python kite/main.py --bs=8 --net=stc_tt --los=di --epochs=100 --db=hcms
```

<!-- ```
@article{tan2023tcct,
  title={Retinal Layer Segmentation in OCT images with
Boundary Regression and Feature Polarization},
  author={Luo, Fuya and Li, Yunhan and Zeng, Guang and Peng, Peng and Wang, Gang and Li, Yongjie},
  journal={submitted to IEEE Transactions on Medical Imaging},
  year={2023},
  publisher={IEEE}
}
``` -->

##  Task2:Glaucoma Classification
    Model:ResNet/Paddle
-   Training:       
    "python t2_train.py --gpu=0"
-   Ensemble:            
    "python t2_ensemble.py --root=xxx --gpu=0"



##  Contact
-   tyb311@qq.com
-   ybt@std.uestc.edu.cn